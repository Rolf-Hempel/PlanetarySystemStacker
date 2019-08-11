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

from glob import glob
from warnings import filterwarnings

from cv2 import GaussianBlur
import matplotlib.pyplot as plt
from numpy import int as np_int
from numpy import zeros, full, empty, float32, int32, newaxis, arange, count_nonzero, \
    where, sqrt, logical_or
from skimage import img_as_uint, img_as_ubyte

from align_frames import AlignFrames
from alignment_points import AlignmentPoints
from configuration import Configuration
from exceptions import InternalError, NotSupportedError, Error
from frames import Frames
from rank_frames import RankFrames
from timer import timer


class StackFrames(object):
    """
        For every frame de-warp the quality areas selected for stacking. Then stack all the
        de-warped frame sections into a single image.

    """

    def __init__(self, configuration, frames, align_frames, alignment_points, my_timer,
                 progress_signal=None):
        """
        Initialze the StackFrames object. In particular, allocate empty numpy arrays used in the
        stacking process for buffering and the final stacked image. The size of all those objects
         in y and x directions is equal to the intersection of all frames.

        :param configuration: Configuration object with parameters
        :param frames: Frames object with all video frames
        :param align_frames: AlignFrames object with global shift information for all frames
        :param alignment_points: AlignmentPoints object with information of all alignment points
        :param my_timer: Timer object for accumulating times spent in specific code sections
        :param progress_signal: Either None (no progress signalling), or a signal with the signature
                                (str, int) with the current activity (str) and the progress in
                                percent (int).
        """

        # Suppress warnings about precision loss in skimage file format conversions.
        filterwarnings("ignore", category=UserWarning)

        self.configuration = configuration
        self.frames = frames
        self.align_frames = align_frames
        self.alignment_points = alignment_points
        self.my_timer = my_timer
        self.progress_signal = progress_signal
        self.signal_step_size = max(int(self.frames.number / 10), 1)
        self.shift_distribution = None
        for name in ['Stacking: AP initialization', 'Stacking: Initialize background blending',
                     'Stacking: compute AP shifts', 'Stacking: remapping and adding',
                     'Stacking: computing background', 'Stacking: merging AP buffers']:
            self.my_timer.create_no_check(name)

        self.my_timer.start('Stacking: AP initialization')
        # Allocate work space for image buffer and the image converted for output.
        # [dim_y, dim_x] is the size of the intersection of all frames.
        self.dim_y = self.align_frames.intersection_shape[0][1] - \
                self.align_frames.intersection_shape[0][0]
        self.dim_x = self.align_frames.intersection_shape[1][1] - \
                self.align_frames.intersection_shape[1][0]
        self.number_pixels = self.dim_y * self.dim_x

        # If the AP stacking buffers have been used already, reset them.
        for ap in self.alignment_points.alignment_points:
            AlignmentPoints.initialize_ap_stacking_buffer(ap, self.frames.color)

        # The summation buffer needs to accommodate three color channels in the case of color
        # images.
        if self.frames.color:
            self.stacked_image_buffer = zeros([self.dim_y, self.dim_x, 3], dtype=float32)
        else:
            self.stacked_image_buffer = zeros([self.dim_y, self.dim_x], dtype=float32)

        # If the alignment point patches do not cover the entire frame, a background image must
        # be computed and blended in. At this point it is not yet clear if this is necessary.
        self.background_patches = None
        self.stacked_background_buffer = None

        # Allocate a buffer which for each pixel of the image buffer counts the number of
        # contributing alignment patch images. Also, allocate a second buffer of same type and size
        # to accumulate the weights at each pixel. This buffer is used to normalize the image
        # buffer. The second buffer is initialized with a small value to avoid divide by zero.
        self.number_single_frame_contributions = full([self.dim_y, self.dim_x], 0, dtype=int32)
        self.sum_single_frame_weights = full([self.dim_y, self.dim_x], 0.0001, dtype=float32)

        self.my_timer.stop('Stacking: AP initialization')

    def prepare_for_stack_blending(self):
        """
        Find image locations where the background image is needed for stacking. If the fraction of
        such pixels is above a given parameter, a full background image is constructed from the best
        frames during the stacking process. If the fraction is low, the image is subdivided into
        quadratic patches. To save computing time, the background image is constructed only in those
        patches which contain at least one pixel where the background is needed.

        :return:
        """

        self.my_timer.start('Stacking: Initialize background blending')
        # Loop over all APs and add the number of image contributions to all pixels covered by them.
        for alignment_point in self.alignment_points.alignment_points:
            patch_y_low = alignment_point['patch_y_low']
            patch_y_high = alignment_point['patch_y_high']
            patch_x_low = alignment_point['patch_x_low']
            patch_x_high = alignment_point['patch_x_high']

            # For each image buffer pixel count the number of image contributions.
            self.number_single_frame_contributions[patch_y_low:patch_y_high,
            patch_x_low: patch_x_high] += self.alignment_points.stack_size

            # For AP patches on the frame border, avoid blending with a non-existing background
            # image patch. This is done by adding "virtual" frame contributions between the frame
            # border and the AP box. Since in those areas it looks like two APs were contributing,
            # the background is assigned a zero weight. Therefore, the background does not have to
            # be computed in those locations. Quite tricky, isn't it?
            y_low = alignment_point['box_y_low']
            y_high = alignment_point['box_y_high']
            x_low = alignment_point['box_x_low']
            x_high = alignment_point['box_x_high']
            if patch_y_low == 0:
                y_low = 0
            if patch_y_high == self.dim_y:
                y_high = self.dim_y
            if patch_x_low == 0:
                x_low = 0
            if patch_x_high == self.dim_x:
                x_high = self.dim_x
            self.number_single_frame_contributions[y_low:y_high, x_low:x_high] += \
                self.alignment_points.stack_size

        # The stack size is the number of frames which contribute to each AP stack.
        single_stack_size_float = float(self.alignment_points.stack_size)

        # Add the contributions of all alignment points into a single buffer.
        for alignment_point in self.alignment_points.alignment_points:
            patch_y_low = alignment_point['patch_y_low']
            patch_y_high = alignment_point['patch_y_high']
            patch_x_low = alignment_point['patch_x_low']
            patch_x_high = alignment_point['patch_x_high']

            # Compute the weights used in AP blending and store them with the AP.
            alignment_point['weights_yx'] = self.one_dim_weight(patch_y_low, patch_y_high,
                    alignment_point['box_y_low'], alignment_point['box_y_high'])[:, newaxis] * \
                    self.one_dim_weight(patch_x_low, patch_x_high, alignment_point['box_x_low'],
                    alignment_point['box_x_high'])

            # For each image buffer pixel add the weights. This is used for normalization later.
            self.sum_single_frame_weights[patch_y_low:patch_y_high,
            patch_x_low: patch_x_high] += single_stack_size_float * alignment_point['weights_yx']

        # Compute the fraction of pixels where no AP patch contributes.
        self.number_stacking_holes = count_nonzero(
            self.number_single_frame_contributions == 0)
        self.fraction_stacking_holes = self.number_stacking_holes / self.number_pixels

        # If all pixels are covered by AP patches, no background image is required.
        if self.number_stacking_holes == 0:
            self.my_timer.stop('Stacking: Initialize background blending')
            return

        # If the alignment points do not cover the full frame, blend the AP contributions with
        # a background computed as the average of globally shifted best frames. The background
        # should only shine through outside AP patches.
        #
        # The "real" alignment point contributions are already in the
        # "stacked_image_buffer".
        # Only the "holes" in the buffer have to be filled with the "averaged_background".
        # Initialize a mask for blending both buffers with each other. Set values to:
        #   0.  at points not covered by a single AP patch.
        #   1.  at points covered by more than one AP, and at points within the interior
        #       (AP box) of a single AP.
        #   0.5 between those areas. After blurring the mask, values in these transition
        #       regions will show a smooth transition between APs and background.
        self.mask = where((self.number_single_frame_contributions == 0), float32(0.), float32(0.5))
        self.mask = where((
        logical_or(self.number_single_frame_contributions > self.alignment_points.stack_size,
                      self.sum_single_frame_weights > single_stack_size_float)), 1.,
                      self.mask)

        # Apply a Gaussian blur to the mask to make transitions smoother.
        blur_width = self.configuration.stack_frames_gauss_width
        self.mask = GaussianBlur(self.mask, (blur_width, blur_width), 0)

        # The Gaussian blur might have created nonzero mask entries outside AP patches. Reset
        # them to zero to avoid artifacts.
        self.mask = where((self.number_single_frame_contributions == 0), 0., self.mask)

        # Re-use the array "number_single_frame_contributions". It is set to 0 at all pixels
        # where a background is required. At all other locations it is set to 1.
        self.number_single_frame_contributions = where((self.mask >= 1.), 1, 0)

        # Allocate a buffer for the background image.
        if self.frames.color:
            self.averaged_background = zeros([self.dim_y, self.dim_x, 3], dtype=float32)
        else:
            self.averaged_background = zeros([self.dim_y, self.dim_x], dtype=float32)

        # If the fraction is below a certain limit, it is worthwhile to compute the background
        # image only where it is needed. Construct a list with patches where the background is
        # needed. The mask blurring has slightly changed the number of pixels where the background
        # is needed, so the value of "fraction_stacking_holes" is not exact. The difference will be
        # very small, though. Since the fraction is only used to decide if a complete background
        # image should be computed, the approximate value is more than sufficient.
        if self.fraction_stacking_holes < self.configuration.stack_frames_background_fraction:

            # Initialize a list of background patches.
            self.background_patches = []

            # Subdivide the image area in quadratic patches. Cycle through all patch locations.
            for patch_y_low in range(0, self.dim_y,
                                     self.configuration.stack_frames_background_patch_size):
                patch_y_high = min(
                    patch_y_low + self.configuration.stack_frames_background_patch_size,
                    self.dim_y - 1)

                # Handle the special case where the patch has zero size in one direction.
                if patch_y_low == patch_y_high:
                    continue
                for patch_x_low in range(0, self.dim_x,
                                         self.configuration.stack_frames_background_patch_size):
                    patch_x_high = min(
                        patch_x_low + self.configuration.stack_frames_background_patch_size,
                        self.dim_x - 1)
                    if patch_x_low == patch_x_high:
                        continue

                    # If the patch contains pixels where the background is used, add it to the list.
                    if count_nonzero(
                            self.number_single_frame_contributions[patch_y_low:patch_y_high,
                            patch_x_low:patch_x_high] == 0) > 0:
                        background_patch = {}
                        background_patch['patch_y_low'] = patch_y_low
                        background_patch['patch_y_high'] = patch_y_high
                        background_patch['patch_x_low'] = patch_x_low
                        background_patch['patch_x_high'] = patch_x_high
                        self.background_patches.append(background_patch)

        self.my_timer.stop('Stacking: Initialize background blending')

    def stack_frames(self):
        """
        Compute the shifted contributions of all frames to all alignment points and add them to the
        appropriate alignment point stacking buffers.

        :return: -
        """

        # First find out if there are holes between AP patches.
        self.prepare_for_stack_blending()

        # Initialize the array for shift distribution statistics.
        self.shift_distribution = full((self.configuration.alignment_points_search_width*2,), 0,
                                          dtype=np_int)

        # If multi-level AP matching is selected, prepare the required level information for each
        # alignment point.
        if self.configuration.alignment_points_method == 'MultiLevel':
            self.alignment_points.set_reference_boxes()

        # Go through the list of all frames.
        for frame_index in range(self.frames.number):
            frame = self.frames.frames(frame_index)
            frame_mono_blurred = self.frames.frames_mono_blurred(frame_index)

            # After every "signal_step_size"th frame, send a progress signal to the main GUI.
            if self.progress_signal is not None and frame_index % self.signal_step_size == 1:
                self.progress_signal.emit("Stack frames", int((frame_index / self.frames.number)
                                                                  * 100.))

            # Look up the constant shifts of the given frame with respect to the mean frame.
            dy = self.align_frames.dy[frame_index]
            dx = self.align_frames.dx[frame_index]

            # Go through all alignment points for which this frame was found to be among the best.
            for alignment_point_index in self.frames.used_alignment_points[frame_index]:
                alignment_point = self.alignment_points.alignment_points[alignment_point_index]

                # Compute the local warp shift for this frame.
                self.my_timer.start('Stacking: compute AP shifts')
                [shift_y, shift_x] = self.alignment_points.compute_shift_alignment_point(
                    frame_mono_blurred, frame_index, alignment_point_index,
                    de_warp=self.configuration.alignment_points_de_warp)

                # Increment the counter corresponding to the 2D warp shift.
                try:
                    self.shift_distribution[int(round(sqrt(shift_y**2 + shift_x**2)))] += 1
                except:
                    print ("Error: shift dy: " + str(shift_y) + ", dx: " + str(shift_x) +
                           " too large for statistics vector.")

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

            # If there are holes between AP patches, add this frame's contribution (if any) to the
            # averaged background image.
            if self.number_stacking_holes > 0 and \
                    frame_index in self.align_frames.quality_sorted_indices[
                        :self.alignment_points.stack_size]:
                self.my_timer.start('Stacking: computing background')

                # Treat the case that the background is computed for specific patches only.
                if self.background_patches:
                    if self.frames.color:
                        for patch in self.background_patches:
                            self.averaged_background[patch['patch_y_low']:patch['patch_y_high'],
                                      patch['patch_x_low']:patch['patch_x_high'], :] += \
                                frame[patch['patch_y_low'] + self.align_frames.dy[frame_index] :
                                      patch['patch_y_high'] + self.align_frames.dy[frame_index],
                                      patch['patch_x_low'] + self.align_frames.dx[frame_index] :
                                      patch['patch_x_high'] + self.align_frames.dx[frame_index], :]
                    else:
                        for patch in self.background_patches:
                            self.averaged_background[patch['patch_y_low']:patch['patch_y_high'],
                                      patch['patch_x_low']:patch['patch_x_high']] += \
                                frame[patch['patch_y_low'] + self.align_frames.dy[frame_index] :
                                      patch['patch_y_high'] + self.align_frames.dy[frame_index],
                                      patch['patch_x_low'] + self.align_frames.dx[frame_index] :
                                      patch['patch_x_high'] + self.align_frames.dx[frame_index]]

                # The complete background image is computed.
                else:
                    if self.frames.color:
                        self.averaged_background += frame[self.align_frames.dy[frame_index]:
                                                    self.dim_y + self.align_frames.dy[frame_index],
                                                    self.align_frames.dx[frame_index]:
                                                    self.dim_x + self.align_frames.dx[frame_index],
                                                    :]
                    else:
                        self.averaged_background += frame[self.align_frames.dy[frame_index]:
                                                    self.dim_y + self.align_frames.dy[frame_index],
                                                    self.align_frames.dx[frame_index]:
                                                    self.dim_x + self.align_frames.dx[frame_index]]
                self.my_timer.stop('Stacking: computing background')

        if self.progress_signal is not None:
            self.progress_signal.emit("Stack frames", 100)

        # If a background image is being computed, divide the buffer by the number of contributions.
        if self.number_stacking_holes > 0:
            self.my_timer.start('Stacking: computing background')
            self.averaged_background /= self.alignment_points.stack_size
            self.my_timer.stop('Stacking: computing background')

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
        single_stack_size_int = self.alignment_points.stack_size

        # Add the contributions of all alignment points into a single buffer.
        for alignment_point in self.alignment_points.alignment_points:
            patch_y_low = alignment_point['patch_y_low']
            patch_y_high = alignment_point['patch_y_high']
            patch_x_low = alignment_point['patch_x_low']
            patch_x_high = alignment_point['patch_x_high']

            # Add the stacking buffer of the alignment point to the appropriate location of the
            # global stacking buffer.
            if self.frames.color:
                self.stacked_image_buffer[patch_y_low:patch_y_high,
                patch_x_low: patch_x_high, :] += alignment_point['stacking_buffer'] * \
                                                 alignment_point['weights_yx'][:, :, newaxis]
            else:
                self.stacked_image_buffer[
                patch_y_low:patch_y_high,
                patch_x_low: patch_x_high] += alignment_point['stacking_buffer'] * \
                                              alignment_point['weights_yx']

        # Divide the global stacking buffer pixel-wise by the number of image contributions.
        if self.frames.color:
            self.stacked_image_buffer /= self.sum_single_frame_weights[:, :, newaxis]
        else:
            self.stacked_image_buffer /= self.sum_single_frame_weights

        self.my_timer.stop('Stacking: merging AP buffers')

        # If the alignment points do not cover the full frame, blend the AP contributions with
        # a background computed as the average of globally shifted best frames. The background
        # should only shine through outside AP patches.
        if self.fraction_stacking_holes > 0:
            self.my_timer.create_no_check('Stacking: blending APs with background')

            # blend the AP buffer with the background.
            if self.frames.color:
                self.stacked_image_buffer = (self.stacked_image_buffer-self.averaged_background) * \
                                            self.mask[:, :, newaxis] + self.averaged_background
            else:
                self.stacked_image_buffer = (self.stacked_image_buffer-self.averaged_background) * \
                                            self.mask + self.averaged_background

            self.my_timer.stop('Stacking: blending APs with background')

        # Scale the image buffer such that entries are in the interval [0., 1.]. Then convert the
        # float image buffer to 16bit int (or 48bit in color mode).
        if self.frames.depth == 8:
            self.stacked_image = img_as_uint(self.stacked_image_buffer / 255)
        else:
            self.stacked_image = img_as_uint(self.stacked_image_buffer / 65535)

        return self.stacked_image

    @staticmethod
    def one_dim_weight(patch_low, patch_high, box_low, box_high):
        """
        Compute one-dimensional weighting ramps between box and patch borders. This function is
        called for y and x dimensions separately.

        :param patch_low: Lower index of AP patch in the given coordinate direction
        :param patch_high: Upper index of AP patch in the given coordinate direction
        :param box_low: Lower index of AP box in the given coordinate direction
        :param box_high: Upper index of AP box in the given coordinate direction
        :return: Vector with weights, starting with 1./(border_width+1) at patch_low, ramping up to
                 1. at box_low, staying at 1. up to box_high-1, and than ramping down to
                 1./(border_width+1) at patch_high-1.
        """

        # Compute offsets relative to patch_low.
        patch_high_offset = patch_high - patch_low
        box_low_offset = box_low - patch_low
        box_high_offset = box_high - patch_low

        # Allocate weights array, length given by patch size.
        weights = empty((patch_high_offset,), dtype=float32)

        # Ramping up between lower patch and box borders.
        if box_low_offset > 0:
            weights[0:box_low_offset] = arange(1 ,box_low_offset+1 , 1) / float32(box_low_offset+1)
        # Box interior
        weights[box_low_offset:box_high_offset] = 1.
        # Ramping down between upper box and patch borders.
        if patch_high_offset > box_high_offset:
            weights[box_high_offset:patch_high_offset] = arange(patch_high - box_high, 0, -1) /\
                                                         float32(patch_high - box_high + 1)
        return weights

    def print_shift_table(self):
        """
        Print a table giving for each shift (in pixels) the number of occurrences. The table ends at
        the last non-zero entry.

        :return: String with three lines to be printed to the protocol file(s)
        """

        # Find the last non-zero entry in the array.
        if max(self.shift_distribution) > 0:
            max_index = [index for index, item in enumerate(self.shift_distribution) if item != 0][-1] \
                        + 1

            # Initialize the three table lines.
            s =    "           Shift (pixels):"
            line = "           ---------------"
            t =    "           Count:         "

            # Extend the three table lines up to the max index.
            for index in range(max_index):
                s += "|{:6d} ".format(index)
                line += "--------"
                t += "|{:6d} ".format(self.shift_distribution[index])

            # Finish the three table lines.
            s += "|"
            line += "-"
            t += "|"

            # Return the three lines to be printed to the protocol.
            return s + "\n" + line + "\n" + t
        else:
            return ""


if __name__ == "__main__":
    # Initalize the timer object used to measure execution times of program sections.
    my_timer = timer()

    # Images can either be extracted from a video file or a batch of single photographs. Select
    # the example for the test run.
    type = 'video'
    if type == 'image':
        names = glob('Images/2012*.tif')
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
        frames = Frames(configuration, names, type=type, convert_to_grayscale=True)
        print("Number of images read: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Error as e:
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

    if configuration.align_frames_mode == "Surface":
        # Select the local rectangular patch in the image where the L gradient is highest in both x
        # and y direction. The scale factor specifies how much smaller the patch is compared to the
        # whole image frame.
        (y_low_opt, y_high_opt, x_low_opt, x_high_opt) = align_frames.compute_alignment_rect(
            configuration.align_frames_rectangle_scale_factor)
        my_timer.stop('Select optimal alignment patch')
        print("optimal alignment rectangle, x_low: " + str(x_low_opt) + ", x_high: " + str(
            x_high_opt) + ", y_low: " + str(y_low_opt) + ", y_high: " + str(y_high_opt))
        reference_frame_with_alignment_points = frames.frames_mono(
            align_frames.frame_ranks_max_index).copy()
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
    alignment_points.create_ap_grid()
    my_timer.stop('Create alignment points')
    print("Number of alignment points created: " + str(len(alignment_points.alignment_points)) +
          ", number of dropped aps (dim): " + str(alignment_points.alignment_points_dropped_dim) +
          ", number of dropped aps (structure): " + str(
          alignment_points.alignment_points_dropped_structure))
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
