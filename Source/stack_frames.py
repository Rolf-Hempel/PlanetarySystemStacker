# -*- coding: utf-8; -*-
"""
Copyright (c) 2018 Rolf Hempel, rolf6419@gmx.de

This file is part of the PlanetarySystemStacker tool (PSS).
https://github.com/Rolf-Hempel/PlanetarySystemStacker

PSS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PSS.  If not, see <http://www.gnu.org/licenses/>.

"""

import glob
import warnings

import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_uint, img_as_ubyte

from align_frames import AlignFrames
from alignment_points import AlignmentPoints
from configuration import Configuration
from exceptions import InternalError, NotSupportedError
from frames import Frames
from rank_frames import RankFrames
from timer import timer


class StackFrames(object):
    """
        For every frame de-warp the quality areas selected for stacking. Then stack all the
        de-warped frame sections into a single image.

    """

    def __init__(self, configuration, frames, align_frames, alignment_points, my_timer):
        """
        Initialze the StackFrames object. In particular, allocate empty numpy arrays used in the
        stacking process for buffering and the final stacked image. The size of all those objects
         in y and x directions is equal to the intersection of all frames.

        :param configuration: Configuration object with parameters
        :param frames: Frames object with all video frames
        :param align_frames: AlignFrames object with global shift information for all frames
        :param alignment_points: AlignmentPoints object with information of all alignment points
        :param my_timer: Timer object for accumulating times spent in specific code sections
        """

        # Suppress warnings about precision loss in skimage file format conversions.
        warnings.filterwarnings("ignore", category=UserWarning)

        self.configuration = configuration
        self.frames = frames
        self.align_frames = align_frames
        self.alignment_points = alignment_points
        self.my_timer = my_timer
        self.my_timer.create('Stacking: AP initialization')
        self.my_timer.create('Stacking: compute AP shifts')
        self.my_timer.create('Stacking: remapping and adding')
        self.my_timer.create('Stacking: merging AP buffers')

        # Allocate work space for image buffer and the image converted for output.
        # [dim_y, dim_x] is the size of the intersection of all frames.
        dim_y = self.align_frames.intersection_shape[0][1] - \
                self.align_frames.intersection_shape[0][0]
        dim_x = self.align_frames.intersection_shape[1][1] - \
                self.align_frames.intersection_shape[1][0]

        # The arrays for the stacked image and the summation buffer need to accommodate three
        # color channels in the case of color images.
        if self.frames.color:
            self.stacked_image_buffer = np.empty([dim_y, dim_x, 3], dtype=np.float32)
            self.stacked_image = np.empty([dim_y, dim_x, 3], dtype=np.int16)
        else:
            self.stacked_image_buffer = np.zeros([dim_y, dim_x], dtype=np.float32)
            self.stacked_image = np.zeros([dim_y, dim_x], dtype=np.int16)
        # Allocate a buffer which for each pixel of the image buffer counts the number of
        # contributing alignment patch images. This buffer is used to normalize the buffer.
        # Initialize the buffer to a small value to avoid divide by zero.
        self.single_frame_contributions = np.full([dim_y, dim_x], 0.0001, dtype=np.float32)

        self.my_timer.stop('Stacking: AP initialization')

    def stack_frames(self):
        """
        Compute the shifted contributions of all frames to all alignment points and add them to the
        appropriate alignment point stacking buffers.

        :return: -
        """

        # Go through the list of all frames.
        for frame_index, frame in enumerate(self.frames.frames):

            # Look up the constant shifts of the given frame with respect to the mean frame.
            dy = self.align_frames.dy[frame_index]
            dx = self.align_frames.dx[frame_index]

            # Go through all alignment points for which this frame was found to be among the best.
            for alignment_point_index in self.frames.used_alignment_points[frame_index]:
                alignment_point = self.alignment_points.alignment_points[alignment_point_index]

                # Compute the local warp shift for this frame.
                self.my_timer.start('Stacking: compute AP shifts')
                [shift_y, shift_x] = self.alignment_points.compute_shift_alignment_point(
                    frame_index, alignment_point_index)

                # The total shift consists of three components: different coordinate origins for
                # current frame and mean frame, global shift of current frame, and the local warp
                # shift at this alignment point. The first two components are accounted for by dy,
                # dx.
                total_shift_y = int(round(dy - shift_y))
                total_shift_x = int(round(dx - shift_x))
                self.my_timer.stop('Stacking: compute AP shifts')

                # Add the shifted alignment point patch to the AP's stacking buffer.
                self.my_timer.start('Stacking: remapping and adding')
                self.remap_rigid(frame, alignment_point['stacking_buffer'],
                                 total_shift_y, total_shift_x,
                                 alignment_point['patch_y_low'], alignment_point['patch_y_high'],
                                 alignment_point['patch_x_low'], alignment_point['patch_x_high'])
                self.my_timer.stop('Stacking: remapping and adding')

    def remap_rigid(self, frame, buffer, shift_y, shift_x, y_low, y_high, x_low, x_high):
        """
        The alignment point patch is taken from the given frame with a constant shift in x and y
        directions. The shifted patch is then added to the given alignment point buffer.

        :param frame: frame to be stacked
        :param buffer: Stacking buffer of the corresponding alignment point
        :param shift_y: Constant shift in y direction between frame stack and current frame
        :param shift_x: Constant shift in x direction between frame stack and current frame
        :param y_low: Lower y index of the quality window on which this method operates
        :param y_high: Upper y index of the quality window on which this method operates
        :param x_low: Lower x index of the quality window on which this method operates
        :param x_high: Upper x index of the quality window on which this method operates
        :return: -
        """

        # Compute index bounds for "source" patch in current frame, and for summation buffer
        # ("target"). Because of local warp effects, the indexing may reach beyond frame borders.
        # In this case reduce the copy area.
        frame_size_y = frame.shape[0]
        y_low_source = y_low + shift_y
        y_high_source = y_high + shift_y
        y_low_target = 0
        # If the shift reaches beyond the frame, reduce the copy area.
        if y_low_source < 0:
            y_low_target = -y_low_source
            y_low_source = 0
        if y_high_source > frame_size_y:
            y_high_source = frame_size_y
        y_high_target = y_low_target + y_high_source - y_low_source

        frame_size_x = frame.shape[1]
        x_low_source = x_low + shift_x
        x_high_source = x_high + shift_x
        x_low_target = 0
        # If the shift reaches beyond the frame, reduce the copy area.
        if x_low_source < 0:
            x_low_target = -x_low_source
            x_low_source = 0
        if x_high_source > frame_size_x:
            x_high_source = frame_size_x
        x_high_target = x_low_target + x_high_source - x_low_source

        # If frames are in color, stack all three color channels using the same mapping. Add the
        # frame contribution to the stacking buffer.
        if self.frames.color:
            # Add the contribution from the shifted window in this frame to the stacking buffer.
            buffer[y_low_target:y_high_target, x_low_target:x_high_target, :] += \
                frame[y_low_source:y_high_source, x_low_source:x_high_source, :]

        # The same for monochrome mode.
        else:
            if x_high_source - x_low_source != x_high_target - x_low_target or \
                    y_high_source - y_low_source != y_high_target - y_low_target:
                print("")
            buffer[y_low_target:y_high_target, x_low_target:x_high_target] += \
                frame[y_low_source:y_high_source, x_low_source:x_high_source]

    def merge_alignment_point_buffers(self):
        """
        Merge the summation buffers for all alignment points into the global stacking buffer. For
        every pixel location divide the global buffer by the number of contributing image patches.
        This results in a uniform brightness level across the whole image, even if alignment point
        patches overlap.

        :return: The final stacked image
        """

        self.my_timer.start('Stacking: merging AP buffers')
        # For each image buffer pixel count the number of image contributions.
        single_stack_size = float(self.alignment_points.stack_size)
        for alignment_point in self.alignment_points.alignment_points:
            # Add the stacking buffer of the alignment point to the appropriate location of the
            # global stacking buffer.
            if self.frames.color:
                self.stacked_image_buffer[
                alignment_point['patch_y_low']:alignment_point['patch_y_high'],
                alignment_point['patch_x_low']: alignment_point['patch_x_high'], :] += \
                    alignment_point['stacking_buffer']
            else:
                self.stacked_image_buffer[
                alignment_point['patch_y_low']:alignment_point['patch_y_high'],
                alignment_point['patch_x_low']: alignment_point['patch_x_high']] += \
                    alignment_point['stacking_buffer']

            # For each image buffer pixel count the number of image contributions.
            self.single_frame_contributions[
            alignment_point['patch_y_low']:alignment_point['patch_y_high'],
            alignment_point['patch_x_low']: alignment_point['patch_x_high']] += single_stack_size

        # Divide the global stacking buffer pixel-wise by the number of image contributions.
        if self.frames.color:
            self.stacked_image_buffer /= self.single_frame_contributions[:, :, np.newaxis]
        else:
            self.stacked_image_buffer /= self.single_frame_contributions

        # Scale the image buffer such that entries are in the interval [0., 1.]. Then convert the
        # float image buffer to 16bit int (or 48bit in color mode).
        self.stacked_image = img_as_uint(self.stacked_image_buffer / float(255))

        self.my_timer.stop('Stacking: merging AP buffers')
        return self.stacked_image


if __name__ == "__main__":
    # Initalize the timer object used to measure execution times of program sections.
    my_timer = timer()

    # Images can either be extracted from a video file or a batch of single photographs. Select
    # the example for the test run.
    type = 'video'
    if type == 'image':
        names = glob.glob('Images/2012*.tif')
        # names = glob.glob('Images/Moon_Tile-031*ap85_8b.tif')
        # names = glob.glob('Images/Example-3*.jpg')
    else:
        names = 'Videos/short_video.avi'
    print(names)

    my_timer.create('Execution over all')

    # Get configuration parameters.
    configuration = Configuration()

    my_timer.create('Read all frames')
    try:
        frames = Frames(configuration, names, type=type, convert_to_grayscale=False)
        print("Number of images read: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Exception as e:
        print("Error: " + e.message)
        exit()
    my_timer.stop('Read all frames')

    # Rank the frames by their overall local contrast.
    my_timer.create('Ranking images')
    rank_frames = RankFrames(frames, configuration)
    rank_frames.frame_score()
    my_timer.stop('Ranking images')

    print("Index of maximum: " + str(rank_frames.frame_ranks_max_index))
    print("Frame scores: " + str(rank_frames.frame_ranks))
    print("Frame scores (sorted): " + str(
        [rank_frames.frame_ranks[i] for i in rank_frames.quality_sorted_indices]))
    print("Sorted index list: " + str(rank_frames.quality_sorted_indices))

    # Initialize the frame alignment object.
    my_timer.create('Select optimal alignment patch')
    align_frames = AlignFrames(frames, rank_frames, configuration)
    # Select the local rectangular patch in the image where the L gradient is highest in both x
    # and y direction. The scale factor specifies how much smaller the patch is compared to the
    # whole image frame.
    (x_low_opt, x_high_opt, y_low_opt, y_high_opt) = align_frames.select_alignment_rect(
        configuration.align_frames_rectangle_scale_factor)
    my_timer.stop('Select optimal alignment patch')
    print("optimal alignment rectangle, x_low: " + str(x_low_opt) + ", x_high: " + str(
        x_high_opt) + ", y_low: " + str(y_low_opt) + ", y_high: " + str(y_high_opt))
    reference_frame_with_alignment_points = align_frames.frames_mono[
        align_frames.frame_ranks_max_index].copy()
    reference_frame_with_alignment_points[y_low_opt,
    x_low_opt:x_high_opt] = reference_frame_with_alignment_points[y_high_opt - 1,
                            x_low_opt:x_high_opt] = 255
    reference_frame_with_alignment_points[y_low_opt:y_high_opt,
    x_low_opt] = reference_frame_with_alignment_points[y_low_opt:y_high_opt, x_high_opt - 1] = 255
    # plt.imshow(reference_frame_with_alignment_points, cmap='Greys_r')
    # plt.show()

    # Align all frames globally relative to the frame with the highest score.
    my_timer.create('Global frame alignment')
    try:
        align_frames.align_frames()
    except NotSupportedError as e:
        print("Error: " + e.message)
        exit()
    except InternalError as e:
        print("Warning: " + e.message)
    my_timer.stop('Global frame alignment')

    # print("Frame shifts: " + str(align_frames.frame_shifts))
    print("Intersection: " + str(align_frames.intersection_shape))

    # Compute the reference frame by averaging the best frames.
    my_timer.create('Compute reference frame')
    average = align_frames.average_frame()
    my_timer.stop('Compute reference frame')
    print("Average frame computed from the best " + str(
        align_frames.average_frame_number) + " frames.")
    # plt.imshow(align_frames.mean_frame, cmap='Greys_r')
    # plt.show()

    # Initialize the AlignmentPoints object. This includes the computation of the average frame
    # against which the alignment point shifts are measured.
    my_timer.create('Initialize alignment point object')
    alignment_points = AlignmentPoints(configuration, frames, rank_frames, align_frames)
    my_timer.stop('Initialize alignment point object')

    # Create alignment points, and show alignment point boxes and patches.
    my_timer.create('Create alignment points')
    alignment_points.create_ap_grid(average)
    my_timer.stop('Create alignment points')
    print("Number of alignment points created: " + str(len(alignment_points.alignment_points)) +
          ", number of dropped aps: " + str(len(alignment_points.alignment_points_dropped)))
    color_image = alignment_points.show_alignment_points(average)

    plt.imshow(color_image)
    plt.show()

    # For each alignment point rank frames by their quality.
    my_timer.create('Rank frames at alignment points')
    alignment_points.compute_frame_qualities()
    my_timer.stop('Rank frames at alignment points')

    # Allocate StackFrames object.
    stack_frames = StackFrames(configuration, frames, align_frames, alignment_points, my_timer)

    # Stack all frames.
    stack_frames.stack_frames()

    # Merge the stacked alignment point buffers into a single image.
    stacked_image = stack_frames.merge_alignment_point_buffers()

    # Save the stacked image as 16bit int (color or mono).
    frames.save_image('Images/example_stacked.tiff', stacked_image, color=frames.color)

    # Convert to 8bit and show in Window.
    plt.imshow(img_as_ubyte(stacked_image))
    plt.show()

    # Print out timer results.
    my_timer.stop('Execution over all')
    my_timer.print()
