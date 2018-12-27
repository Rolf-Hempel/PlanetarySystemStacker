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
import os
from cv2 import remap, calcOpticalFlowFarneback, INTER_LINEAR, BORDER_TRANSPARENT, \
    OPTFLOW_FARNEBACK_GAUSSIAN
from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage import img_as_uint, img_as_ubyte

from align_frames import AlignFrames
from alignment_points import AlignmentPoints
from configuration import Configuration
from exceptions import InternalError
from frames import Frames
from quality_areas import QualityAreas
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

        self.configuration = configuration
        self.frames = frames
        self.align_frames = align_frames
        self.alignment_points = alignment_points
        self.my_timer = my_timer
        self.my_timer.create('Stacking: AP initialization')
        self.my_timer.create('Stacking: compute AP shifts')
        self.my_timer.create('Stacking: remapping and adding')

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

        # Divide the buffers by the number of frame contributions.
        for alignment_point in self.alignment_points.alignment_points:
            alignment_point['stacking_buffer'] /= float(self.alignment_points.stack_size)


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
        y_high_target = y_high_source - y_low_source

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
        x_high_target = x_high_source - x_low_source

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

        # Finally, convert the float image buffer to 16bit int (or 48bit in color mode).
        self.stacked_image = img_as_uint(self.stacked_image_buffer / float(255))

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
        file = 'short_video'
        # file = 'Moon_Tile-024_043939'
        names = 'Videos/' + file + '.avi'
    print(names)

    my_timer.create('Execution over all')
    start_over_all = time()
    # Get configuration parameters.
    configuration = Configuration()
    try:
        frames = Frames(configuration, names, type=type)
        print("Number of images read: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Exception as e:
        print("Error: " + e.message)
        exit()

    # Rank the frames by their overall local contrast.
    rank_frames = RankFrames(frames, configuration)
    start = time()
    rank_frames.frame_score()
    end = time()
    print('Elapsed time in ranking images: {}'.format(end - start))
    print("Index of maximum: " + str(rank_frames.frame_ranks_max_index))

    # Initialize the frame alignment object.
    align_frames = AlignFrames(frames, rank_frames, configuration)
    start = time()
    # Select the local rectangular patch in the image where the L gradient is highest in both x
    # and y direction. The scale factor specifies how much smaller the patch is compared to the
    # whole image frame.
    (x_low_opt, x_high_opt, y_low_opt, y_high_opt) = align_frames.select_alignment_rect(
        configuration.align_frames_rectangle_scale_factor)
    end = time()
    print('Elapsed time in computing optimal alignment rectangle: {}'.format(end - start))
    print("optimal alignment rectangle, x_low: " + str(x_low_opt) + ", x_high: " + str(
        x_high_opt) + ", y_low: " + str(y_low_opt) + ", y_high: " + str(y_high_opt))

    # Align all frames globally relative to the frame with the highest score.
    start = time()
    align_frames.align_frames()
    end = time()
    print('Elapsed time in aligning all frames: {}'.format(end - start))
    print("Intersection: " + str(align_frames.intersection_shape))

    start = time()
    # Compute the reference frame by averaging the best frames.
    average = align_frames.average_frame()

    # Initialize the AlignmentPoints object. This includes the computation of the average frame
    # against which the alignment point shifts are measured.

    alignment_points = AlignmentPoints(configuration, frames, rank_frames, align_frames)
    end = time()
    print('Elapsed time in computing average frame: {}'.format(end - start))
    print("Average frame computed from the best " + str(
        align_frames.average_frame_number) + " frames.")
    # plt.imshow(align_frames.mean_frame, cmap='Greys_r')
    # plt.show()

    # Create a regular grid with small boxes. A subset of those boxes will be selected as
    # alignment points.
    step_size = configuration.alignment_box_step_size
    box_size = configuration.alignment_box_size
    start = time()
    alignment_points.create_alignment_boxes(step_size, box_size)
    end = time()
    print('Elapsed time in alignment box creation: {}'.format(end - start))
    print("Number of alignment boxes created: " + str(
        len(alignment_points.alignment_boxes) * len(alignment_points.alignment_boxes[0])))

    # An alignment box is selected as an alignment point if it satisfies certain conditions
    # regarding local contrast etc.
    structure_threshold = configuration.alignment_point_structure_threshold
    brightness_threshold = configuration.alignment_point_brightness_threshold
    contrast_threshold = configuration.alignment_point_contrast_threshold
    print("Selection of alignment points, structure threshold: " + str(
        structure_threshold) + ", brightness threshold: " + str(
        brightness_threshold) + ", contrast threshold: " + str(contrast_threshold))
    start = time()
    alignment_points.select_alignment_points(structure_threshold, brightness_threshold,
                                             contrast_threshold)
    end = time()
    print('Elapsed time in alignment point selection: {}'.format(end - start))
    print("Number of alignment points selected: " + str(len(alignment_points.alignment_points)))

    # Create a regular grid of quality areas. The fractional sizes of the areas in x and y,
    # as compared to the full frame, are specified in the configuration object.
    start = time()
    quality_areas = QualityAreas(configuration, frames, align_frames, alignment_points)

    print("")
    if not configuration.stacking_rigid_ap_shift:
        print("Distribution of alignment point indices among quality areas in y direction:")
        for index_y, y_low in enumerate(quality_areas.y_lows):
            y_high = quality_areas.y_highs[index_y]
            print("QA y index: " + str(index_y) + ", Lower y pixel: " + str(
                y_low) + ", upper y pixel index: " + str(y_high) + ", lower ap coordinate: " + str(
                alignment_points.y_locations[
                    quality_areas.qa_ap_index_y_lows[index_y]]) + ", upper ap coordinate: " + str(
                alignment_points.y_locations[quality_areas.qa_ap_index_y_highs[index_y] - 1]))
        print("")
        print("Distribution of alignment point indices among quality areas in x direction:")
        for index_x, x_low in enumerate(quality_areas.x_lows):
            x_high = quality_areas.x_highs[index_x]
            print("QA x index: " + str(index_x) + ", Lower x pixel: " + str(
                x_low) + ", upper x pixel index: " + str(x_high) + ", lower ap coordinate: " + str(
                alignment_points.x_locations[
                    quality_areas.qa_ap_index_x_lows[index_x]]) + ", upper ap coordinate: " + str(
                alignment_points.x_locations[quality_areas.qa_ap_index_x_highs[index_x] - 1]))
        print("")

    # For each quality area rank the frames according to the local contrast.
    quality_areas.select_best_frames()

    # Truncate the list of frames to be stacked to the same number for each quality area.
    quality_areas.truncate_best_frames()
    end = time()
    print('Elapsed time in quality area creation and frame ranking: {}'.format(end - start))
    print("Number of frames to be stacked for each quality area: " + str(quality_areas.stack_size))

    # Allocate StackFrames object.
    stack_frames = StackFrames(configuration, frames, align_frames, alignment_points, quality_areas,
                               my_timer)

    # Stack all frames.
    start = time()
    output_stacking_buffer = True
    # qa_list = None
    qa_list = [(12, 11), (10, 14), (16, 16)]
    if output_stacking_buffer:
        for file in os.listdir('QA_videos'):
            os.unlink('QA_videos/' + file)
    result = stack_frames.stack_frames(output_stacking_buffer=output_stacking_buffer,
                                       qa_list=qa_list)
    end = time()
    print('Elapsed time in frame stacking: {}'.format(end - start))
    print('Elapsed time total: {}'.format(end - start_over_all))

    # Save the stacked image as 16bit int (color or mono).
    frames.save_image('Images/' + file + '_stacked.tiff', result)

    # Convert to 8bit and show in Window.
    plt.imshow(img_as_ubyte(result))
    plt.show()

    # Print out timer results.
    my_timer.stop('Execution over all')
    my_timer.print()
