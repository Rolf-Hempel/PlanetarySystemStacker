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

    def __init__(self, configuration, frames, align_frames, alignment_points, quality_areas,
                 my_timer):
        """
        Initialze the StackFrames object. In particular, allocate empty numpy arrays used in the
        stacking process for buffering, pixel shifts in y and x, and the final stacked image. The
        size of all those objects in y and x directions is equal to the intersection of all frames.

        :param configuration: Configuration object with parameters
        :param frames: Frames object with all video frames
        :param align_frames: AlignFrames object with global shift information for all frames
        :param alignment_points: AlignmentPoints object with information of all alignment points
        :param quality_areas: QualityAreas object with information on all quality areas
        :param my_timer: Timer object for accumulating times spent in specific code sections
        """

        self.configuration = configuration
        self.frames = frames
        self.align_frames = align_frames
        self.alignment_points = alignment_points
        self.quality_areas = quality_areas
        self.stack_size = quality_areas.stack_size
        self.my_timer = my_timer
        self.my_timer.create('Stacking: AP initialization')
        self.my_timer.create('Stacking: compute AP shifts')
        self.my_timer.create('Stacking: AP interpolation')
        self.my_timer.create('Stacking: compute optical flow')
        self.my_timer.create('Stacking: remapping and adding')

        # Create a mask array which specifies the alignment box locations where shifts are to be
        # computed.
        self.my_timer.create('StackFrames initialization')
        self.alignment_points.ap_mask_initialize()

        # Allocate work space for image buffer and the image converted for output.
        # [dim_y, dim_x] is the size of the intersection of all frames.
        dim_y = self.align_frames.intersection_shape[0][1] - \
                self.align_frames.intersection_shape[0][0]
        dim_x = self.align_frames.intersection_shape[1][1] - \
                self.align_frames.intersection_shape[1][0]

        # The arrays for the stacked image and the intermediate buffer need to accommodate three
        # color channels in the case of color images.
        if self.frames.color:
            self.stacked_image_buffer = np.empty([self.stack_size, dim_y, dim_x, 3],
                                                 dtype=np.float32)
            self.stacked_image = np.empty([dim_y, dim_x, 3], dtype=np.int16)
        else:
            self.stacked_image_buffer = np.zeros([self.stack_size, dim_y, dim_x], dtype=np.float32)
            self.stacked_image = np.zeros([dim_y, dim_x], dtype=np.int16)

        # Initialize arrays used to store y and x shift values for each frame pixel. Not used for
        # rigid stacking.
        if not self.configuration.stacking_rigid_ap_shift:
            self.pixel_map_y = np.empty([
                self.align_frames.intersection_shape[0][1] -
                self.align_frames.intersection_shape[0][0],
                self.align_frames.intersection_shape[1][1] -
                self.align_frames.intersection_shape[1][
                    0]], dtype=np.float32)
            self.pixel_map_x = self.pixel_map_y.copy()
        self.my_timer.stop('StackFrames initialization')

    def add_frame_contribution_rigid(self, frame_index):
        """
        For a given frame compute the contributions of all quality areas which have been marked for
        stacking. For each quality area, compute the warp shift at the quality area center (i.e.
        the central alignment point). The contribution of the quality area to the frame stack is
        computed as a rigid translation of the frame window by the constant shift.

        If the quality area does not contain an alignment point, set the warp shift to zero.

        :param frame_index: Index of the current frame
        :return: -
        """

        # If this frame is used in at least one quality area, prepare mask for shift computation.
        if self.frames.used_quality_areas[frame_index]:
            # Compute global offsets of current frame relative to intersection frame.
            dy = self.align_frames.intersection_shape[0][0] - \
                 self.align_frames.frame_shifts[frame_index][0]
            dx = self.align_frames.intersection_shape[1][0] - \
                 self.align_frames.frame_shifts[frame_index][1]

            # For each quality area used for this frame, compute its contribution to the stacked
            # image. First, interpolate y and x shifts between alignment boxes.
            for [index_y, index_x] in self.frames.used_quality_areas[frame_index]:
                self.my_timer.start('Stacking: compute AP shifts')
                quality_area = self.quality_areas.quality_areas[index_y][index_x]

                # Extract index bounds in y and x for this quality area. Interpolations and stacking
                # are restricted to this index range.
                y_low, y_high, x_low, x_high = quality_area['coordinates'][0:4]

                # Compute the shift for alignment point in quality area (if there is one).
                self.alignment_point_index_list = quality_area['alignment_point_indices']

                if self.alignment_point_index_list is None:
                    # There is no alignment point in this quality area. Set warp shift to zero, i.e.
                    # apply the global frame shift only.
                    shift_y = dy
                    shift_x = dx

                elif len(self.alignment_point_index_list) == 1:

                    alignment_point_list = [
                        self.alignment_points.alignment_points[self.alignment_point_index_list[0]]]

                    # For debugging only: Initialize shift fields to None. Remove after testing!!
                    self.alignment_points.y_shifts[:][:] = None
                    self.alignment_points.x_shifts[:][:] = None

                    # There is one internal alignment point. Compute its shift.
                    self.alignment_points.compute_alignment_point_shifts(frame_index,
                                                        alignment_point_list=alignment_point_list)
                    # Shifts are stored at alignment box locations. Get them from there. Note that
                    # alignment box indices include boundary points.
                    shift_y = int(
                        round(dy - self.alignment_points.y_shifts[index_y + 1][index_x + 1]))
                    shift_x = int(
                        round(dx - self.alignment_points.x_shifts[index_y + 1][index_x + 1]))
                else:
                    raise InternalError("More than one alignment point in rigid stacking")
                self.my_timer.stop('Stacking: compute AP shifts')

                # Look up and increment the stacking buffer counter for this quality area.
                self.my_timer.start('Stacking: remapping and adding')
                buffer_counter = quality_area['stacking_buffer_counter']
                quality_area['stacking_buffer_counter'] += 1

                # Copy the shifted QA window to stacking buffer.
                self.remap_rigid(self.frames.frames[frame_index], shift_y, shift_x,
                                 y_low, y_high, x_low, x_high, buffer_counter)
                self.my_timer.stop('Stacking: remapping and adding')

    def remap_rigid(self, frame, shift_y, shift_x, y_low, y_high, x_low, x_high, buffer_counter):
        """
        This is a simple variant where the quality area window is copied from the given frame with
        a constant shift in x and y directions.

        :param frame: frame to be stacked
        :param shift_y: Constant shift in y direction between frame stack and current frame
        :param shift_x: Constant shift in x direction between frame stack and current frame
        :param y_low: Lower y index of the quality window on which this method operates
        :param y_high: Upper y index of the quality window on which this method operates
        :param x_low: Lower x index of the quality window on which this method operates
        :param x_high: Upper x index of the quality window on which this method operates
        :param buffer_counter: Index of the next free stacking buffer for this quality area
        :return: -
        """

        frame_size_y = frame.shape[0]
        y_low_source = y_low + shift_y
        y_high_source = y_high + shift_y
        y_low_target = y_low
        y_high_target = y_high
        # If the shift reaches beyond the frame, reduce the copy area.
        if y_low_source < 0:
            y_low_target = -y_low_source
            y_low_source = 0
        if y_high_source > frame_size_y:
            y_high_source = frame_size_y
            y_high_target = y_low_target + (y_high_source - y_low_source)

        frame_size_x = frame.shape[1]
        x_low_source = x_low + shift_x
        x_high_source = x_high + shift_x
        x_low_target = x_low
        x_high_target = x_high
        # If the shift reaches beyond the frame, reduce the copy area.
        if x_low_source < 0:
            x_low_target = -x_low_source
            x_low_source = 0
        if x_high_source > frame_size_x:
            x_high_source = frame_size_x
            x_high_target = x_low_target + (x_high_source - x_low_source)

        # If frames are in color, stack all three color channels using the same mapping.
        if self.frames.color:
            # Copy the contribution from the shifted window in this frame to the stacking buffer.
            self.stacked_image_buffer[buffer_counter, y_low_target:y_high_target,
            x_low_target:x_high_target, :] = \
                frame[y_low_source:y_high_source, x_low_source:x_high_source, :]

        # The same for monochrome mode.
        else:
            if x_high_source - x_low_source != x_high_target - x_low_target or \
                    y_high_source - y_low_source != y_high_target - y_low_target:
                print("")
            self.stacked_image_buffer[buffer_counter, y_low_target:y_high_target,
            x_low_target:x_high_target] = \
                frame[y_low_source:y_high_source, x_low_source:x_high_source]

    def add_frame_contribution_interpolate(self, frame_index):
        """
        For a given frame de-warp those quality areas which have been marked for stacking.
        To this end, first interpolate the shift vectors between the alignment box positions, then
        use the remap function of OpenCV to de-warp the frame. Finally, add the processed
        parts of this frame to the contributions by the other frames to produce the stacked image.

        :param frame_index: Index of the current frame
        :return: -
        """

        # Because the areas selected for stacking are different for every frame, first reset the
        # alignment point mask.
        self.alignment_points.ap_mask_reset()

        # If this frame is used in at least one quality area, prepare mask for shift computation.
        if self.frames.used_quality_areas[frame_index]:
            # Compute global offsets of current frame relative to intersection frame.
            dy = self.align_frames.intersection_shape[0][0] - \
                 self.align_frames.frame_shifts[frame_index][0]
            dx = self.align_frames.intersection_shape[1][0] - \
                 self.align_frames.frame_shifts[frame_index][1]

            # If "optical flow" is used to compute the de-warping, no alignment points are needed.
            # In this case skip the computation of alignment point shifts.
            if not self.configuration.stacking_use_optical_flow:
                self.my_timer.start('Stacking: AP initialization')
                for [index_y, index_x] in self.frames.used_quality_areas[frame_index]:
                    self.alignment_points.ap_mask_set(
                        self.quality_areas.qa_ap_index_y_lows[index_y],
                        self.quality_areas.qa_ap_index_y_highs[index_y],
                        self.quality_areas.qa_ap_index_x_lows[index_x],
                        self.quality_areas.qa_ap_index_x_highs[index_x])
                self.my_timer.stop('Stacking: AP initialization')

                # Compute the shifts in y and x for all mask locations.
                self.my_timer.start('Stacking: compute AP shifts')
                self.alignment_points.compute_alignment_point_shifts(frame_index, use_ap_mask=True)
                self.my_timer.stop('Stacking: compute AP shifts')

            # For each quality area used for this frame, compute its contribution to the stacked
            # image. First, interpolate y and x shifts between alignment boxes.
            for [index_y, index_x] in self.frames.used_quality_areas[frame_index]:
                quality_area = self.quality_areas.quality_areas[index_y][index_x]

                # Extract index bounds in y and x for this quality area. Interpolations and stacking
                # are restricted to this index range.
                y_low, y_high, x_low, x_high = quality_area['coordinates'][0:4]

                # Use "optical flow" to compute the de-warping. The flow field "flow" contains
                # pixel-wise displacements.
                if self.configuration.stacking_use_optical_flow:
                    self.my_timer.start('Stacking: compute optical flow')

                    # Compute the flow field with some overlap.
                    y_low_overlap = max(y_low - self.configuration.stacking_optical_flow_overlap, 0)
                    y_high_overlap = min(y_high + self.configuration.stacking_optical_flow_overlap,
                                         self.align_frames.mean_frame.shape[0])
                    x_low_overlap = max(x_low - self.configuration.stacking_optical_flow_overlap, 0)
                    x_high_overlap = min(x_high + self.configuration.stacking_optical_flow_overlap,
                                         self.align_frames.mean_frame.shape[1])
                    if self.configuration.use_gaussian_filter:
                        gaussian_filter_flag = OPTFLOW_FARNEBACK_GAUSSIAN
                    else:
                        gaussian_filter_flag = 0
                    flow = calcOpticalFlowFarneback(
                        self.align_frames.mean_frame[y_low_overlap:y_high_overlap,
                        x_low_overlap:x_high_overlap],
                        self.frames.frames_mono[frame_index]
                        [y_low_overlap + dy:y_high_overlap + dy,
                        x_low_overlap + dx:x_high_overlap + dx],
                        flow=None,
                        pyr_scale=self.configuration.pyramid_scale,
                        levels=self.configuration.levels,
                        winsize=self.configuration.winsize,
                        iterations=self.configuration.iterations,
                        poly_n=self.configuration.poly_n,
                        poly_sigma=self.configuration.poly_sigma,
                        flags=gaussian_filter_flag)

                    # the "flow" field only covers the quality area (with overlap). Copy the
                    # displacements to the pixel_map window of this quality area. Discard the
                    # shifts in the overlap areas.
                    self.pixel_map_x[y_low:y_high, x_low:x_high] = flow[
                                                       y_low - y_low_overlap:y_high - y_low_overlap,
                                                       x_low - x_low_overlap:x_high - x_low_overlap,
                                                       0]
                    self.pixel_map_y[y_low:y_high, x_low:x_high] = flow[
                                                       y_low - y_low_overlap:y_high - y_low_overlap,
                                                       x_low - x_low_overlap:x_high - x_low_overlap,
                                                       1]
                    # The flow field so far contains the displacements only. The remap function
                    # requires absolute pixel coordinates. Therefore, add the (y, x) pixel
                    # coordinates to each point of the flow field. Also, add the global offsets
                    # (dy, dx)of this frame relative to the intersection frame.
                    self.pixel_map_x[y_low:y_high, x_low:x_high] += np.arange(x_low, x_high) + dx
                    self.pixel_map_y[y_low:y_high, x_low:x_high] += np.arange(y_low, y_high)[:,
                                                                    np.newaxis] + dy
                    self.my_timer.stop('Stacking: compute optical flow')

                # Compute the de-warping by interpolating shifts between alignment points.
                else:
                    # Cut out the 2D window with y shift values for all alignment boxes used by this
                    # quality area. Combine global offsets with local shifts, and add the y
                    # coordinates at the alignment box positions. As the result, "data_y" (and
                    # "data_x", below, respectively) contain the mapping coordinates as required by
                    # the OpenCV function "remap".

                    # Set the y and x coordinates at the alignment box locations within the window.
                    self.my_timer.start('Stacking: AP interpolation')
                    x_coords, y_coords = np.meshgrid(self.alignment_points.x_locations[
                                                     self.quality_areas.qa_ap_index_x_lows[index_x]:
                                                     self.quality_areas.qa_ap_index_x_highs[
                                                         index_x]],
                                                     self.alignment_points.y_locations[
                                                     self.quality_areas.qa_ap_index_y_lows[index_y]:
                                                     self.quality_areas.qa_ap_index_y_highs[
                                                         index_y]],
                                                     sparse=True)

                    # Compute the y mapping coordinates at alignment box positions.
                    data_y = y_coords + dy - self.alignment_points.y_shifts[
                                             self.quality_areas.qa_ap_index_y_lows[index_y]:
                                             self.quality_areas.qa_ap_index_y_highs[index_y],
                                             self.quality_areas.qa_ap_index_x_lows[index_x]:
                                             self.quality_areas.qa_ap_index_x_highs[index_x]]

                    # Build the linear interpolation operator.
                    interpolator_y = RegularGridInterpolator(
                        (quality_area['interpolation_coords_y'],
                         quality_area['interpolation_coords_x']),
                        data_y, bounds_error=False,
                        method='linear',
                        fill_value=None)

                    # Interpolate y shifts for all points within the quality area.
                    self.pixel_map_y[y_low:y_high, x_low:x_high] = interpolator_y(
                        quality_area['interpolation_points']).reshape(y_high - y_low,
                                                                      x_high - x_low)

                    # Do the same for x shifts.
                    data_x = x_coords + dx - self.alignment_points.x_shifts[
                                             self.quality_areas.qa_ap_index_y_lows[index_y]:
                                             self.quality_areas.qa_ap_index_y_highs[index_y],
                                             self.quality_areas.qa_ap_index_x_lows[index_x]:
                                             self.quality_areas.qa_ap_index_x_highs[index_x]]

                    # Build the linear interpolation operator.
                    interpolator_x = RegularGridInterpolator(
                        (quality_area['interpolation_coords_y'],
                         quality_area['interpolation_coords_x']),
                        data_x, bounds_error=False,
                        method='linear',
                        fill_value=None)

                    # Interpolate x shifts for all points within the quality area.
                    self.pixel_map_x[y_low:y_high, x_low:x_high] = interpolator_x(
                        quality_area['interpolation_points']).reshape(y_high - y_low,
                                                                      x_high - x_low)
                    self.my_timer.stop('Stacking: AP interpolation')

                # Look up and increment the stacking buffer counter for this quality area.
                buffer_counter = quality_area['stacking_buffer_counter']
                quality_area['stacking_buffer_counter'] += 1

                # There are two variants for remapping: An own (slow) implementation with
                # sub-pixel accuracy, and the less accurate (but fast) OpenCV method.
                self.my_timer.start('Stacking: remapping and adding')
                if self.configuration.stacking_own_remap_method:
                    # De-warp the quality area window and store the result in the stacking buffer.
                    self.remap(self.frames.frames[frame_index], self.pixel_map_y, self.pixel_map_x,
                               y_low, y_high, x_low, x_high, buffer_counter)
                else:
                    if self.frames.color:
                        self.stacked_image_buffer[buffer_counter, y_low:y_high, x_low:x_high, :] = \
                            remap(self.frames.frames[frame_index],
                                  self.pixel_map_x[y_low:y_high, x_low:x_high],
                                  self.pixel_map_y[y_low:y_high, x_low:x_high], INTER_LINEAR, None,
                                  BORDER_TRANSPARENT)
                    else:
                        self.stacked_image_buffer[buffer_counter, y_low:y_high, x_low:x_high] = \
                            remap(self.frames.frames[frame_index],
                                  self.pixel_map_x[y_low:y_high, x_low:x_high],
                                  self.pixel_map_y[y_low:y_high, x_low:x_high], INTER_LINEAR, None,
                                  BORDER_TRANSPARENT)
                self.my_timer.stop('Stacking: remapping and adding')

    def remap(self, frame, pixel_map_y, pixel_map_x, y_low, y_high, x_low, x_high, buffer_counter):
        """
        This is an alternative routine to OpenCV.remap. It works directly on the image buffer.
        Interpolation is linear with sub-pixel accuracy. Stacking is restricted to a (quality)
        window.

        :param frame: frame to be stacked
        :param pixel_map_y: For every pixel this array points to the y location from where the
                            pixel is to be interpolated.
        :param pixel_map_x: For every pixel this array points to the x location from where the
                            pixel is to be interpolated.
        :param y_low: Lower y index of the quality window on which this method operates
        :param y_high: Upper y index of the quality window on which this method operates
        :param x_low: Lower x index of the quality window on which this method operates
        :param x_high: Upper x index of the quality window on which this method operates
        :param buffer_counter: Index of the next free stacking buffer for this quality area
        :return: -
        """

        # Restrict pixel map coordinates to within the frame intersection area.
        clip_y_low = 0.
        clip_y_high = self.frames.shape[0] - 1.01
        pixel_map_y[y_low:y_high, x_low:x_high] = np.clip(pixel_map_y[y_low:y_high, x_low:x_high],
                                                          clip_y_low, clip_y_high)
        clip_x_low = 0.
        clip_x_high = self.frames.shape[1] - 1.01
        pixel_map_x[y_low:y_high, x_low:x_high] = np.clip(pixel_map_x[y_low:y_high, x_low:x_high],
                                                          clip_x_low, clip_x_high)

        # Prepare for sub-pixel interpolation. The integer coordinates point to the
        # upper left corner of the interpolation square. The fractions point to the
        # location within the square from where the pixel value is to be interpolated.
        self.pixel_j = pixel_map_y[y_low:y_high, x_low:x_high].astype(np.int32)
        self.fraction_j = pixel_map_y[y_low:y_high, x_low:x_high] - self.pixel_j
        self.one_minus_fraction_j = 1. - self.fraction_j
        self.pixel_i = pixel_map_x[y_low:y_high, x_low:x_high].astype(np.int32)
        self.fraction_i = pixel_map_x[y_low:y_high, x_low:x_high] - self.pixel_i
        self.one_minus_fraction_i = 1. - self.fraction_i

        # If frames are in color, stack all three color channels using the same mapping.
        if self.frames.color:
            # Do the interpolation. The weights are computed as the surface areas covered by
            # the four quadrants in the interpolation square.
            self.stacked_image_buffer[buffer_counter, y_low:y_high, x_low:x_high, 0] = \
                frame[self.pixel_j, self.pixel_i, 0] * \
                self.one_minus_fraction_j * self.one_minus_fraction_i + \
                frame[self.pixel_j, self.pixel_i + 1, 0] * \
                self.one_minus_fraction_j * self.fraction_i + \
                frame[self.pixel_j + 1, self.pixel_i, 0] * \
                self.fraction_j * self.one_minus_fraction_i + \
                frame[self.pixel_j + 1, self.pixel_i + 1, 0] * self.fraction_j * self.fraction_i
            self.stacked_image_buffer[buffer_counter, y_low:y_high, x_low:x_high, 1] = \
                frame[self.pixel_j, self.pixel_i, 1] * \
                self.one_minus_fraction_j * self.one_minus_fraction_i + \
                frame[self.pixel_j, self.pixel_i + 1, 1] * \
                self.one_minus_fraction_j * self.fraction_i + \
                frame[self.pixel_j + 1, self.pixel_i, 1] * \
                self.fraction_j * self.one_minus_fraction_i + \
                frame[self.pixel_j + 1, self.pixel_i + 1, 1] * self.fraction_j * self.fraction_i
            self.stacked_image_buffer[buffer_counter, y_low:y_high, x_low:x_high, 2] = \
                frame[self.pixel_j, self.pixel_i, 2] * \
                self.one_minus_fraction_j * self.one_minus_fraction_i + \
                frame[self.pixel_j, self.pixel_i + 1, 2] * \
                self.one_minus_fraction_j * self.fraction_i + \
                frame[self.pixel_j + 1, self.pixel_i, 2] * \
                self.fraction_j * self.one_minus_fraction_i + \
                frame[self.pixel_j + 1, self.pixel_i + 1, 2] * self.fraction_j * self.fraction_i

        # The same for monochrome mode.
        else:
            self.stacked_image_buffer[buffer_counter, y_low:y_high, x_low:x_high] = \
                frame[self.pixel_j, self.pixel_i] * \
                self.one_minus_fraction_j * self.one_minus_fraction_i + \
                frame[self.pixel_j, self.pixel_i + 1] * \
                self.one_minus_fraction_j * self.fraction_i + \
                frame[self.pixel_j + 1, self.pixel_i] * \
                self.fraction_j * self.one_minus_fraction_i + \
                frame[self.pixel_j + 1, self.pixel_i + 1] * self.fraction_j * self.fraction_i

    def stack_frames(self, output_stacking_buffer=False, qa_list=None):
        """
        Combine the sharp areas of all frames into a single stacked image.

        :param output_stacking_buffer: If this optional argument is set to "True", video files
                                       which show all frame contributions of a quality area are
                                       produced for selected quality areas. Colored crosses are
                                       inserted at AP locations.
        :param qa_list: List of tuples (index_y, index_x). Each tuple specifies a quality area
                        location for which a video is to be produced. If "None", videos are
                        produced for all qa locations.
        :return: -
        """

        # Compute the contributions of all frames and store them in the summation buffer.
        for idx, frame in enumerate(self.frames.frames):
            if self.configuration.stacking_rigid_ap_shift:
                # Rigid stacking: use simplified stacking routine.
                self.add_frame_contribution_rigid(idx)
            else:
                # Use the general stacking routine based on bilinear interpolation.
                self.add_frame_contribution_interpolate(idx)

        # Accumulate the single frame contributions, divide the summation buffer by the number of
        # contributing frames per quality area, and scale entries to the interval [0., 1.].
        # Finally, convert the float image buffer to 16bit int (or 48bit in color mode).
        self.stacked_image = img_as_uint(np.sum(self.stacked_image_buffer, axis=0) / (
                self.stack_size * 255.))

        # For each quality area create a video with all single frame contributions.
        if output_stacking_buffer:
            self.my_timer.create('Stacking: QA video output')
            # Create a colored uint8 version of stacked image buffers and insert colored crosses at
            # alignment box locations.
            annotated_buffers = []
            for buffer in self.stacked_image_buffer:
                annotated_buffers.append(self.alignment_points.show_alignment_box_types(buffer))

            fps = 2
            for index_y, quality_area_row in enumerate(self.quality_areas.quality_areas):
                for index_x, quality_area in enumerate(quality_area_row):
                    if not qa_list or (index_y, index_x) in qa_list:
                        # Extract index bounds in y and x for this quality area. Interpolations and
                        # stacking are restricted to this index range.
                        y_low, y_high, x_low, x_high = quality_area['coordinates'][0:4]
                        self.frames.write_video(
                            'QA_videos/quality_area_video_(' + str(index_y) + ',' + str(
                                index_x) + ').avi',
                            annotated_buffers, y_low, y_high, x_low, x_high, fps)
            self.my_timer.stop('Stacking: QA video output')

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
        configuration.alignment_rectangle_scale_factor)
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

    # Initialize the AlignmentPoints object. This includes the computation of the average frame
    # against which the alignment point shifts are measured.
    start = time()
    alignment_points = AlignmentPoints(configuration, frames, rank_frames, align_frames)
    end = time()
    print('Elapsed time in computing average frame: {}'.format(end - start))
    print("Average frame computed from the best " + str(
        alignment_points.average_frame_number) + " frames.")
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
