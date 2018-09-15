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
from time import time

from numpy import arange, ceil, float32, empty
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from align_frames import AlignFrames
from alignment_points import AlignmentPoints
from configuration import Configuration
from frames import Frames
from miscellaneous import Miscellaneous
from rank_frames import RankFrames
from exceptions import InternalError
from quality_areas import QualityAreas


class StackFrames(object):

    def __init__(self, configuration, frames, align_frames, alignment_points, quality_areas):
        self.configuration = configuration
        self.frames = frames
        self.align_frames = align_frames
        self.alignment_points = alignment_points
        self.quality_areas = quality_areas
        self.stack_size = quality_areas.stack_size
        self.alignment_points.ap_mask_initialize()
        if self.frames.color:
            self.stacked_image = empty(
                [self.align_frames.intersection_shape[0][1] -
                 self.align_frames.intersection_shape[0][0],
                 self.align_frames.intersection_shape[1][1] -
                 self.align_frames.intersection_shape[1][0], 3], dtype=float32)
            self.buffer = self.stacked_image.copy()
        else:
            self.stacked_image = empty(
                [self.align_frames.intersection_shape[0][1] -
                 self.align_frames.intersection_shape[0][0],
                 self.align_frames.intersection_shape[1][1] -
                 self.align_frames.intersection_shape[1][0]], dtype=float32)
            self.buffer = self.stacked_image.copy()
        self.pixel_shift_y = self.stacked_image = empty(
                [self.align_frames.intersection_shape[0][1] -
                 self.align_frames.intersection_shape[0][0],
                 self.align_frames.intersection_shape[1][1] -
                 self.align_frames.intersection_shape[1][0]], dtype=float32)
        self.pixel_shift_x = self.pixel_shift_y.copy()

    def stack_frame(self, frame_index):
        self.alignment_points.ap_mask_reset()
        if self.frames.used_quality_areas[frame_index]:
            for [index_y, index_x] in self.frames.used_quality_areas[frame_index]:
                self.alignment_points.ap_mask_set(self.quality_areas.qa_ap_index_y_lows[index_y],
                                                  self.quality_areas.qa_ap_index_y_highs[index_y],
                                                  self.quality_areas.qa_ap_index_x_lows[index_x],
                                                  self.quality_areas.qa_ap_index_x_highs[index_x])
            self.alignment_points.compute_alignment_point_shifts(frame_index, use_ap_mask=True)


if __name__ == "__main__":

    # Images can either be extracted from a video file or a batch of single photographs. Select
    # the example for the test run.
    type = 'video'
    if type == 'image':
        names = glob.glob(
            'Images/2012*.tif')
        # names = glob.glob('Images/Moon_Tile-031*ap85_8b.tif')
        # names = glob.glob('Images/Example-3*.jpg')
    else:
        names = 'Videos/short_video.avi'
    print(names)

    # Get configuration parameters.
    configuration = Configuration()
    try:
        frames = Frames(names, type=type)
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
    print("Number of alignment boxes created: " + str(len(alignment_points.alignment_boxes)))

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

    print ("")
    print ("Distribution of alignment point indices among quality areas in y direction:")
    for index_y, y_low in enumerate(quality_areas.y_lows):
        y_high = quality_areas.y_highs[index_y]
        print ("Lower y pixel: " + str(y_low) + ", upper y pixel index: " + str(y_high) +
               ", lower ap coordinate: " +
               str(alignment_points.y_locations[quality_areas.qa_ap_index_y_lows[index_y]]) +
               ", upper ap coordinate: " +
               str(alignment_points.y_locations[quality_areas.qa_ap_index_y_highs[index_y]]))
    for index_x, x_low in enumerate(quality_areas.x_lows):
        x_high = quality_areas.x_highs[index_x]
        print("Lower x pixel: " + str(x_low) + ", upper x pixel index: " + str(x_high) +
              ", lower ap coordinate: " +
              str(alignment_points.x_locations[quality_areas.qa_ap_index_x_lows[index_x]]) +
              ", upper ap coordinate: " +
              str(alignment_points.x_locations[quality_areas.qa_ap_index_x_highs[index_x]]))
    print("")

    # For each quality area rank the frames according to the local contrast.
    quality_areas.select_best_frames()

    # Truncate the list of frames to be stacked to the same number for each quality area.
    quality_areas.truncate_best_frames()
    end = time()
    print('Elapsed time in quality area creation and frame ranking: {}'.format(end - start))
    print("Number of frames to be stacked for each quality area: " + str(quality_areas.stack_size))

    stack_frames = StackFrames(configuration, frames, align_frames, alignment_points, quality_areas)
