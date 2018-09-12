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

from numpy import arange, ceil

from align_frames import AlignFrames
from alignment_points import AlignmentPoints
from configuration import Configuration
from frames import Frames
from miscellaneous import Miscellaneous
from rank_frames import RankFrames


class QualityAreas(object):
    """
        Create a rectangular grid of quality areas which cover the entire frame. For each patch
        the image quality of all frames is ranked, and a certain percentage of the best frames is
        stacked to produce the final image. Since the seeing deteriorates the image quality in a
        non-uniform way across the image, the list of best frames is different for different quality
        areas.

    """

    def __init__(self, configuration, frames, align_frames, alignment_points):
        """

        :param configuration: Configuration object with parameters
        :param frames: Frames object with all video frames
        :param align_frames: AlignFrames object with global shift information for all frames
        :param alignment_points: AlignmentPoints object with information of all alignment points
        """

        self.configuration = configuration
        self.frames = frames
        self.align_frames = align_frames
        self.alignment_points = alignment_points

        # Initialize the number of frames to be used for the image stack.
        self.stack_size = None

        mean_frame = self.align_frames.mean_frame
        mean_frame_shape = mean_frame.shape

        # Compute the size of the quality areas in y and x directions.
        self.quality_area_size_y = int(
            mean_frame_shape[0] / self.configuration.quality_area_number_y)
        self.quality_area_size_x = int(
            mean_frame_shape[1] / self.configuration.quality_area_number_x)

        # Compute the pixel coordinate bounds of the quality areas, first in y direction.
        self.y_lows = arange(0, mean_frame_shape[0] - self.quality_area_size_y + 1,
                             self.quality_area_size_y)
        self.y_highs = self.y_lows + self.quality_area_size_y

        # Because in the general case the frame cannot be divided evenly, extend the last patch
        # to extend up do the frame border.
        self.y_highs[-1] = mean_frame_shape[0]
        self.y_dim = len(self.y_lows)

        # Repeat the same for the x pixel coordinate.
        self.x_lows = arange(0, mean_frame_shape[1] - self.quality_area_size_x + 1,
                             self.quality_area_size_x)
        self.x_highs = self.x_lows + self.quality_area_size_x
        self.x_highs[-1] = mean_frame_shape[1]
        self.x_dim = len(self.x_lows)

        # Initialize the list of quality areas.
        self.quality_areas = []

        # Cycle through all quality areas, row by row.
        for index_y, y_low in enumerate(self.y_lows):
            y_high = self.y_highs[index_y]
            quality_area_row = []
            for index_x, x_low in enumerate(self.x_lows):
                # For each quality area, store coordinate bounds and empty lists for alignment
                # points and frame qualities in a dictionary. The dictionaries are stored in a a 2D
                # list structure (row-wise).
                x_high = self.x_highs[index_x]
                quality_area = {}
                quality_area['coordinates'] = [y_low, y_high, x_low, x_high]
                quality_area['alignment_point_indices'] = []
                quality_area['frame_qualities'] = []
                quality_area_row.append(quality_area)
            self.quality_areas.append(quality_area_row)

        # Cycle through all alignment points. For each point append its index to the alignment
        # point list of the quality area which contains the point.
        for point_index, [box_index,
                          [j, i, y_center, x_center, y_low, y_high, x_low, x_high]] in enumerate(
                self.alignment_points.alignment_points):
            y_index = min(int(y_center / self.quality_area_size_y), self.y_dim - 1)
            x_index = min(int(x_center / self.quality_area_size_x), self.x_dim - 1)
            self.quality_areas[y_index][x_index]['alignment_point_indices'].append(point_index)

    def select_best_frames(self):
        """
        For each quality area, rank all frames by their local image quality. For quality areas
        containing at least one alignment point, use their "local contrast" as the ranking
        criterion. For areas which do not contain any alignment point, copy the frame ranks from the
        nearest quality are with alignment points.

        The rationale for this differentiation is that areas without alignment points are either
        empty space or very smooth surface sections where the contrast measurement does not make
        much sense.

        :return: -
        """

        # Cycle through all frames. Use the monochrome image for frame ranking.
        for frame in self.frames.frames_mono:
            # Cycle through all quality areas:
            for index_y, quality_area_row in enumerate(self.quality_areas):
                for index_x, quality_area in enumerate(quality_area_row):
                    # If the alignment point list of the quality area is non-empty, compute the
                    # local contrast.
                    if quality_area['alignment_point_indices']:
                        [y_low, y_high, x_low, x_high] = quality_area['coordinates']
                        quality_area['frame_qualities'].append(
                            Miscellaneous.local_contrast(frame[y_low:y_high, x_low:x_high],
                                                         self.configuration.quality_area_pixel_stride))

        # For quality areas with alignment points, sort the computed quality ranks in descending
        # order.
        for index_y, quality_area_row in enumerate(self.quality_areas):
            for index_x, quality_area in enumerate(quality_area_row):
                if quality_area['alignment_point_indices']:
                    quality_area['best_frame_indices'] = [b[0] for b in sorted(
                        enumerate(quality_area['frame_qualities']), key=lambda i: i[1],
                        reverse=True)]

        # For quality areas without alignment points, use method "best_frame_indices_in_empty_areas"
        # to copy ranks from the nearest quality are with alignment points.
        for index_y, quality_area_row in enumerate(self.quality_areas):
            for index_x, quality_area in enumerate(quality_area_row):
                if not quality_area['alignment_point_indices']:
                    quality_area['best_frame_indices'] = self.best_frame_indices_in_empty_areas(
                        index_y, index_x)

    def best_frame_indices_in_empty_areas(self, index_y, index_x):
        """
        For a quality area without any alignment point, find the closest quality area with
        alignment points and return its list of frame indices ranked by the local frame quality in
        decreasing order.

        :param index_y: y coordinate of the quality area in the rectangular grid of quality areas
        :param index_x: x coordinate of the quality area in the rectangular grid of quality areas
        :return: frame index list, ranked by the image quality at the closest quality area with
                 alignment points
        """

        # Go though circles with increasing radius "distance" around the current quality area.
        for distance in arange(1, max(self.y_dim, self.x_dim)):
            circle = Miscellaneous.circle_around(index_x, index_y, distance)
            for (compare_x, compare_y) in circle:
                # If the coordinates are within the quality area grid, and if the area at this
                # location has a non-empty list of alignment points, return its list.
                if 0 <= compare_x < self.x_dim and 0 <= compare_y < self.y_dim and \
                        self.quality_areas[compare_y][compare_x]['alignment_point_indices']:
                    return self.quality_areas[compare_y][compare_x]['best_frame_indices']
        # This should never happen, because it means that there is not any quality area with an
        # alignment point.
        return []

    def truncate_best_frames(self):
        """
        Compute the number of frames to be stacked and for all quality areas truncate the ranking
        lists with the best frame indices.

        :return: -
        """

        # Compute the stack size from the given percentage. Take at least one frame.
        max_frames = max(
            int(ceil(self.frames.number * self.configuration.quality_area_frame_percent / 100.)), 1)

        # This is a precaution measure: Do not take more frames than elements in the shortest
        # ranking list.
        self.stack_size = min(max_frames, min(
            [min([len(self.quality_areas[j][i]['best_frame_indices']) for i in arange(self.x_dim)])
             for j in arange(self.y_dim)]))

        # For all quality areas: Truncate the "best frame indices" list to the uniform selected
        # stack size.
        for index_y, quality_area_row in enumerate(self.quality_areas):
            for index_x, quality_area in enumerate(quality_area_row):
                quality_area['best_frame_indices'] = quality_area['best_frame_indices'][
                                                     :self.stack_size]


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
    print("Frame scores: " + str(rank_frames.frame_ranks))
    print("Frame scores (sorted): " + str(
        [rank_frames.frame_ranks[i] for i in rank_frames.quality_sorted_indices]))
    print("Sorted index list: " + str(rank_frames.quality_sorted_indices))

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
    start = time()
    align_frames.align_frames()
    end = time()
    print('Elapsed time in aligning all frames: {}'.format(end - start))
    print("Frame shifts: " + str(align_frames.frame_shifts))
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

    # For all frames: Compute the local shifts for all alignment points (to be used for de-warping).
    for frame_index in range(frames.number):
        start = time()
        point_shifts, errors, diffphases = alignment_points.compute_alignment_point_shifts(
            frame_index)
        end = time()
        print("Elapsed time in computing point shifts for frame number " + str(
            frame_index) + ": " + str(end - start))

    # Create a regular grid of quality areas. The fractional sizes of the areas in x and y,
    # as compared to the full frame, are specified in the configuration object.
    start = time()
    quality_areas = QualityAreas(configuration, frames, align_frames, alignment_points)

    # For each quality area rank the frames according to the local contrast.
    quality_areas.select_best_frames()

    # Truncate the list of frames to be stacked to the same number for each quality area.
    quality_areas.truncate_best_frames()
    end = time()
    print('Elapsed time in quality area creation and frame ranking: {}'.format(end - start))
    print("Number of frames to be stacked for each quality area: " + str(quality_areas.stack_size))
