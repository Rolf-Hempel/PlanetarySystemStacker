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
from math import ceil
from time import time

import matplotlib.pyplot as plt
from numpy import arange, amax, stack, amin, hypot, zeros, full, float64
from skimage.feature import register_translation

from align_frames import AlignFrames
from configuration import Configuration
from exceptions import WrongOrderingError, NotSupportedError
from frames import Frames
from miscellaneous import Miscellaneous
from rank_frames import RankFrames


class AlignmentPoints(object):
    """
        Create a rectangular grid of potential places for alignment points. For each location
        create a small "alignment box" around it. For each alignment box test if there is enough
        structure and brightness in the picture to use it as an alignment point. For all
        alignment points in all frames, compute the local shifts relative to the mean frame.

    """

    def __init__(self, configuration, frames, rank_frames, align_frames):
        """
        Initialize the AlignmentPoints object and compute the mean frame.

        :param configuration: Configuration object with parameters
        :param frames: Frames object with all video frames
        :param rank_frames: RankFrames object with global quality ranks (between 0. and 1.,
                            1. being optimal) for all frames
        :param align_frames: AlignFrames object with global shift information for all frames
        """

        self.configuration = configuration
        self.frames = frames
        self.rank_frames = rank_frames
        self.align_frames = align_frames
        self.y_locations = None
        self.x_locations = None
        self.y_locations_number = None
        self.x_locations_number = None
        self.alignment_boxes = None
        self.alignment_points = None
        self.alignment_point_neighbors = None
        self.y_shifts = None
        self.x_shifts = None
        self.ap_mask = None

        # Compute the number of frames to be used for creating the mean frame.
        self.average_frame_number = max(
            ceil(frames.number * configuration.average_frame_percent / 100.), 1)

        # Call method "average_frame" of the AlignFrames object to compute the mean frame. Use
        # the best frames according to their ranks.
        self.align_frames.average_frame([self.frames.frames_mono[i] for i in
                                         self.rank_frames.quality_sorted_indices[
                                         :self.average_frame_number]],
                                        [self.align_frames.frame_shifts[i] for i in
                                         self.rank_frames.quality_sorted_indices[
                                         :self.average_frame_number]])

    def create_alignment_boxes(self, step_size, box_size):
        """
        Create a rectangular grid of potential places for alignment points. For each location
        create a small "alignment box" around it.

        :param step_size: Number of pixels in y and x directions between alignment boxes
        :param box_size: Extension of alignment boxes in both y and x directions
        :return: -
        """

        mean_frame = self.align_frames.mean_frame
        mean_frame_shape = mean_frame.shape
        box_size_half = int(box_size / 2)
        self.alignment_boxes = []

        # Create the y and x coordinates for the rectangular grid of alignment boxes. Keep the
        # boxes away from the frame borders enough to avoid that boxes reach beyond the frame border
        # in the search for the local displacement.
        self.y_locations = arange(box_size_half + self.configuration.alignment_point_search_width,
                                  mean_frame_shape[0] - box_size_half -
                                  self.configuration.alignment_point_search_width,
                                  step_size, dtype=int)
        self.y_locations_number = len(self.y_locations)
        self.x_locations = arange(box_size_half + self.configuration.alignment_point_search_width,
                                  mean_frame_shape[1] - box_size_half -
                                  self.configuration.alignment_point_search_width,
                                  step_size, dtype=int)
        self.x_locations_number = len(self.x_locations)

        # Initialize the pixel shift arrays in y and x for all alignment box locations.
        self.y_shifts = zeros((self.y_locations_number, self.x_locations_number), dtype=float64)
        self.x_shifts = zeros((self.y_locations_number, self.x_locations_number), dtype=float64)

        for j, y in enumerate(self.y_locations):
            # Compute the y index bounds of the row of alignment boxes.
            y_low = y - box_size_half
            y_high = y + box_size_half
            alignment_box_row = []
            for i, x in enumerate(self.x_locations):
                # Compute the x index bounds of the alignment box.
                x_low = x - box_size_half
                x_high = x + box_size_half

                # Cut the reference box from the mean frame.
                box = mean_frame[y_low:y_high, x_low:x_high]

                # Initialize a dictionary for the alignment box which contains all related info.
                alignment_box = {}

                # Initialize the type of the alignment box to None. Later it will be changed to
                # 'alignment point' or 'alignment point neighbor' if there is enough structure at
                # this point or in the neighborhood.
                alignment_box['type'] = None
                alignment_box['box'] = box
                alignment_box['coordinates'] = (j, i, y, x, y_low, y_high, x_low, x_high)

                # Compute structure and brightness information for the alignment box, and add the
                # box to the list.
                alignment_box['structure'] = Miscellaneous.quality_measure(box)
                alignment_box['max_brightness'] = amax(box)
                alignment_box['min_brightness'] = amin(box)

                # Initialize the index of the corresponding element in the list of alignment
                # points to "None".
                alignment_box['alignment_point_index'] = None
                alignment_box_row.append(alignment_box)

            # Append the row of alignment boxes to the 2D grid structure of alignment boxes.
            self.alignment_boxes.append(alignment_box_row)

        # Normalize the structure information for all boxes by dividing by the maximum value.
        structure_max = max(
            alignment_box_row[i]['structure'] for alignment_box_row in self.alignment_boxes for
            i in range(self.x_locations_number))
        for alignment_box_row in self.alignment_boxes:
            for alignment_box in alignment_box_row:
                alignment_box['structure'] /= structure_max

    def select_alignment_points(self, structure_threshold, brightness_threshold,
                                contrast_threshold):
        """
        For each alignment box decide if there is enough structure and brightness to make it an
        alignment point. If so, add it to the list of alignment points.

        :param structure_threshold: Minimum structure value for an alignment point (between 0.
                                    and 1.)
        :param brightness_threshold: The brightest pixel must be brighter than this value
                                     (0 < value <256)
        :param contrast_threshold: The difference between the brightest and darkest pixel values
                                   must be larger than this value (0 < value < 256)
        :return: -
        """

        if self.alignment_boxes == None:
            raise WrongOrderingError(
                "Attempt to select alignment points before alignment boxes are created")

        # Check for each alignment box if it qualifies for being an alignment point.
        self.alignment_points = []
        alignment_point_index = 0
        for alignment_box_row in self.alignment_boxes:
            for alignment_box in alignment_box_row:
                if alignment_box['structure'] > structure_threshold and alignment_box[
                    'max_brightness'] > brightness_threshold and alignment_box['max_brightness'] - \
                        alignment_box['min_brightness'] > contrast_threshold:
                    # This point satisfies the conditions for an alignment point. Append its
                    # coordinates to the list.
                    self.alignment_points.append(alignment_box['coordinates'])
                    alignment_box['type'] = 'alignment point'
                    alignment_box['alignment_point_index'] = alignment_point_index
                    alignment_point_index += 1

        # Check all alignment boxes which did not qualify for being an alignment point, if there
        # are alignment points in the neighborhood. In this case compute the weights with which the
        # shifts at those points are interpolated to this position.
        #
        # First, initialize the list of the points in the neighborhood of alignment points.
        self.alignment_point_neighbors = []
        alignment_point_neighbor_index = 0
        for alignment_box_row in self.alignment_boxes:
            for alignment_box in alignment_box_row:
                # The box did not qualify for being an alignment point.
                if alignment_box['alignment_point_index'] is None:
                    j_center = alignment_box['coordinates'][0]
                    i_center = alignment_box['coordinates'][1]

                    # Circle around the point and look for "contributing" alignment points in the
                    #  neighborhood.
                    contributing_alignment_points = []
                    weight_sum = 0.

                    # Limit the radius of the search circle according to a configuration parameter.
                    for r in arange(1, configuration.alignment_box_max_neighbor_distance):
                        circle = Miscellaneous.circle_around(i_center, j_center, r)
                        for (i, j) in circle:
                            if 0 <= i < self.x_locations_number and 0 <= j < \
                                    self.y_locations_number and \
                                    self.alignment_boxes[j][i]['type'] == 'alignment point':
                                # Alignment point found. Compute its distance and weight,
                                # and add it to the list.
                                contributing_alignment_point = {}
                                contributing_alignment_point['alignment_point_index'] = \
                                self.alignment_boxes[j][i]['alignment_point_index']
                                contributing_alignment_point['weight'] = 1. / hypot(i - i_center,
                                                                                    j - j_center)
                                weight_sum += contributing_alignment_point['weight']
                                contributing_alignment_points.append(contributing_alignment_point)

                    # Normalize the weights such that their sum is 1.
                    for contributing_alignment_point in contributing_alignment_points:
                        contributing_alignment_point['weight'] /= weight_sum

                    # If the list of alignment points in the neighborhood is not empty. Add the
                    # point to the list of "alignment point neighbors".
                    if contributing_alignment_points:
                        self.alignment_point_neighbors.append(
                            [j_center, i_center, contributing_alignment_points])
                        alignment_box['type'] = 'alignment point neighbor'
                        alignment_box[
                            'alignment_point_neighbor_index'] = alignment_point_neighbor_index
                        alignment_point_neighbor_index += 1

    def compute_alignment_point_shifts(self, frame_index, alignment_point_list=None,
                                       alignment_point_neighbor_list=None,
                                       use_ap_mask=False):
        """
        For each alignment point compute the shifts in y and x relative to the mean frame. Three
        different methods can be used to compute the shift values:
        - a subpixel algorithm from "skimage.feature"
        - a phase correlation algorithm (miscellaneous.translation)
        - a local search algorithm (spiralling outwards), see method "search_local_match"
        After computing the shift vectors for the alignment points, shift vectors are interpolated
        for neighbors of those alignment points. Make sure that all contributing alignment points
        for those neighbors are included in the list of alignment points.

        This method can be called in three different ways, depending on optional parameters:
        - Shifts are computed for all alignment boxes if no optional parameter is set.
        - If "alignment_point_list" and "alignment_point_neighbor_list" are specified, shifts are
          computed for points on those lists only.
        - If "use_ap_mask" is set to "True", shifts are computed for alignment box positions where
          the corresponding mask entry is "True". This mode requires that the mask has been
          set before by calling method "ap_mask_set".

        :param frame_index: Index of the selected frame in the list of frames.
        :param alignment_point_list: List of alignment point indices. If not specified, shifts are
               computed for all alignment points in the frame.
        :param alignment_point_neighbor_list: List of alignment point neighbor indices. If not
               specified, shifts are computed for all alignment point neighbors in the frame.
        :param alignment_box_mask: 2D boolean array. Shifts are computed at locations where the
                                   mask entry is "True".
        :return: -
        """

        if self.alignment_points is None:
            raise WrongOrderingError(
                "Attempt to compute alignment point shifts before selecting alingment points")

        # Reset the pixel shift array values in y and x.
        self.y_shifts[:, :] = 0.
        self.x_shifts[:, :] = 0.

        # A mask defines the locations where shift vectors are to be computed.
        if use_ap_mask:

            # First make sure that all "contributing alignment points" of alignment point neighbors
            # are included in the mask.
            for alignment_box_row in self.alignment_boxes:
                for alignment_box in alignment_box_row:
                    # If not an alignment point neighbor, go to the next box.
                    if alignment_box['type'] != 'alignment point neighbor':
                        continue
                    for ap in self.alignment_point_neighbors[alignment_box[
                                                            'alignment_point_neighbor_index']][2]:
                        self.ap_mask[self.alignment_points[ap['alignment_point_index']][0],
                                     self.alignment_points[ap['alignment_point_index']][1]] = True

            # Compute shifts at alignment points.
            for alignment_box_row in self.alignment_boxes:
                for alignment_box in alignment_box_row:
                    # If not an alignment point, go to the next box.
                    if alignment_box['type'] != 'alignment point':
                        continue
                    j = alignment_box['coordinates'][0]
                    i = alignment_box['coordinates'][1]
                    # The mask is "True": Compute the shift vector.
                    if self.ap_mask[j][i]:
                        self.compute_shift_alignment_point(j, i,
                            alignment_box['coordinates'][4], alignment_box['coordinates'][5],
                            alignment_box['coordinates'][6], alignment_box['coordinates'][7])

            # Now compute shifts at alignment point neighbors.
            for alignment_box_row in self.alignment_boxes:
                for alignment_box in alignment_box_row:
                    # If not an alignment point neighbor, go to the next box.
                    if alignment_box['type'] != 'alignment point neighbor':
                        continue
                    j = alignment_box['coordinates'][0]
                    i = alignment_box['coordinates'][1]
                    # The mask is "True": Compute the shift vector.
                    if self.ap_mask[j][i]:
                        self.compute_shift_neighbor_point(j, i, self.alignment_point_neighbors[
                            alignment_box['alignment_point_neighbor_index']][2])

        # Alignment box positions are specified via lists.
        else:
            # If no list is specified explicitly, compute shifts for all alignment points.
            if alignment_point_list is not None:
                ap_list = alignment_point_list
            else:
                ap_list = self.alignment_points

            # If no list is specified explicitly, compute shifts for all alignment point neighbors.
            if alignment_point_neighbor_list is not None:
                ap_neighbor_list = alignment_point_neighbor_list
            else:
                ap_neighbor_list = self.alignment_point_neighbors

            # For each alignment point, compute the shift for the given frame index.
            for [j, i, y_center, x_center, y_low, y_high, x_low, x_high] in ap_list:
                self.compute_shift_alignment_point(j, i, y_low, y_high, x_low, x_high)

            # For each alignment point neighbor, compute the shifts for the given frame index.
            for [j, i, contributing_alignment_points] in ap_neighbor_list:
                self.compute_shift_neighbor_point(j, i, contributing_alignment_points)

    def compute_shift_alignment_point(self, j, i, y_low, y_high, x_low, x_high):
        """
        Compute the pixel shift vector at a given alignment point. The resulting shifts in y and x
        direction are assigned to the corresponding entries in arrays self.y_shifts and
        self.x_shifts.

        :param j: Row index (y) of alignment box
        :param i: Column index (x) of alignment box
        :param y_low: Lower y pixel index bound of alignment box
        :param y_high: Upper y pixel index bound of alignment box
        :param x_low: Lower x pixel index bound of alignment box
        :param x_high: Upper x pixel index bound of alignment box
        :return: -
        """

        # The offsets dy and dx are caused by two effects: First, the mean frame is smaller
        # than the original frames. It only contains their intersection. And second, because the
        # given frame is globally shifted as compared to the mean frame.
        dy = self.align_frames.intersection_shape[0][0] - \
             self.align_frames.frame_shifts[frame_index][0]
        dx = self.align_frames.intersection_shape[1][0] - \
             self.align_frames.frame_shifts[frame_index][1]

        # Cut out the alignment box from the given frame. Take into account the offsets
        # explained above.
        box_in_frame = self.frames.frames_mono[frame_index][y_low + dy:y_high + dy,
                       x_low + dx:x_high + dx]

        # Use subpixel registration from skimage.feature, with accuracy 1/10 pixels.
        if self.configuration.alignment_point_method == 'Subpixel':
            shift_pixel, error, diffphase = register_translation(
                self.alignment_boxes[j][i]['box'], box_in_frame, 10, space='real')

        # Use a simple phase shift computation (contained in module "miscellaneous").
        elif self.configuration.alignment_point_method == 'CrossCorrelation':
            shift_pixel = Miscellaneous.translation(self.alignment_boxes[j][i]['box'],
                                                    box_in_frame, box_in_frame.shape)

        # Use a local search (see method "search_local_match" below.
        elif self.configuration.alignment_point_method == 'LocalSearch':
            shift_pixel = self.search_local_match(self.alignment_boxes[j][i]['box'],
                                                  self.frames.frames_mono[frame_index],
                                                  y_low + dy, y_high + dy, x_low + dx,
                                                  x_high + dx,
                                                  self.configuration.alignment_point_search_width)
        else:
            raise NotSupportedError("The point shift computation method " +
                                    self.configuration.alignment_point_method + " is not implemented")

        # Copy pixel shift values into the shift arrays, and append them to the point_shifts
        # list.
        self.y_shifts[j][i] = shift_pixel[0]
        self.x_shifts[j][i] = shift_pixel[1]

    def compute_shift_neighbor_point(self, j, i, contributing_alignment_points):
        """
        Compute the pixel shift vector at a given alignment point neighbor, by interpolating from
        the alignment points in the neighborhood. Make sure that the shifts for those alignment
        points are computed before calling this method.. The resulting shifts in y and x
        direction are assigned to the corresponding entries in arrays self.y_shifts and
        self.x_shifts.

        :param j: Row index (y) of alignment box
        :param i: Column index (x) of alignment box
        :param contributing_alignment_points: locations and interpolation weights for
                                              alignment points in the neighborhood
        :return: -
        """

        self.y_shifts[j][i] = 0.
        self.x_shifts[j][i] = 0.
        for ap in contributing_alignment_points:
            j_ap = self.alignment_points[ap['alignment_point_index']][0]
            i_ap = self.alignment_points[ap['alignment_point_index']][1]
            self.y_shifts[j][i] += ap['weight'] * self.y_shifts[j_ap][i_ap]
            self.x_shifts[j][i] += ap['weight'] * self.x_shifts[j_ap][i_ap]

    def search_local_match(self, reference_box, frame, y_low, y_high, x_low, x_high, search_width):
        """
        Try shifts in y, x between the box around the alignment point in the mean frame and the
        corresponding box in the given frame. Start with shifts [0, 0] and move out in a circular
        fashion, until the radius "search_width" is reached. The global frame shift is accounted for
        beforehand already.

        :param reference_box: Image box around alignment point in mean frame.
        :param frame: Given frame for which the local shift at the alignment point is to be
                      computed.
        :param y_low: Lower y coordinate limit of box in given frame, taking into account the
                      global shift and the different sizes of the mean frame and the original
                      frames.
        :param y_high: Upper y coordinate limit
        :param x_low: Lower x coordinate limit
        :param x_high: Upper x coordinate limit
        :param search_width: maximum radius of the search spiral
        :return: Local shift in the form [shift_y, shift_x], or [0, 0] if no optimum could be found.
        """

        # Initialize the global optimum with an impossibly large value.
        deviation_min = 1000000
        dy_min = None
        dx_min = None
        dev = []

        # Start with shift [0, 0] and proceed in a circular pattern.
        for r in arange(search_width + 1):

            # Create an enumerator which produces shift values [dy, dx] in a circular pattern
            # with radius "r".
            circle_r = Miscellaneous.circle_around(0, 0, r)

            # Initialize the optimum for radius "r" to an impossibly large value,
            # and the corresponding shifts to None.
            deviation_min_r, dy_min_r, dx_min_r = 1000000, None, None

            # Go through the circle with radius "r" and compute the difference (deviation)
            # between the shifted frame and the corresponding box in the mean frame. Find the
            # minimum "deviation_min_r" for radius "r".
            for (dx, dy) in circle_r:
                deviation = abs(
                    reference_box - frame[y_low - dy:y_high - dy, x_low - dx:x_high - dx]).sum()
                if deviation < deviation_min_r:
                    deviation_min_r, dy_min_r, dx_min_r = deviation, dy, dx

            # Store the optimal value for radius "r".
            dev.append(deviation_min_r)

            # If for the current radius there is no improvement compared to the previous radius,
            # the optimum if reached.
            if deviation_min_r >= deviation_min:
                return [dy_min, dx_min]

            # Otherwise, update the current optimum and continue.
            else:
                deviation_min, dy_min, dx_min = deviation_min_r, dy_min_r, dx_min_r
        # print("search local match unsuccessful: y_low: " + str(y_low) + ", x_low: " + str(x_low))
        # print("search local match unsuccessful: y: " + str((y_high + y_low) / 2.) + ",
        #       x: " + str((x_high + x_low) / 2.))

        # If within the maximum search radius no optimum could be found, return [0, 0].
        return [0, 0]

    def ap_mask_initialize(self):
        """
        Initialize the alignment point mask used to specify at which alignment box locations shift
        vectors are to be computed. This method (and at least one call to "ap_mask_set" must be
        called before calling "compute_alignment_point_shifts".

        :return: -
        """

        self.ap_mask = full((self.y_locations_number, self.x_locations_number), False, dtype=bool)

    def ap_mask_set(self, j_ap_low, j_ap_high, i_ap_low, i_ap_high):
        """
        Register a rectangular patch of alignment box locations for shift vector computation.

        :param j_ap_low:  lower bound of alignment box indices in y direction
        :param j_ap_high: upper bound of alignment box indices in y direction
        :param i_ap_low:  lower bound of alignment box indices in x direction
        :param i_ap_high: upper bound of alignment box indices in x direction
        :return: -
        """

        self.ap_mask[j_ap_low:j_ap_high, i_ap_low:i_ap_high] = True

    def ap_mask_reset(self):
        """
        Reset the alignment point mask everywhere to False. After this call, the mask can be re-used
        for the definition of another alignment box selection.

        :return: -
        """

        self.ap_mask[:, :] = False


if __name__ == "__main__":
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
    print("Number of alignment boxes created: " + str(
        len(alignment_points.alignment_boxes)) + " rows, each with " + str(
        len(alignment_points.alignment_boxes[0])) + " boxes.")

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
    print("Number of alignment point neighbors selected: " + str(
        len(alignment_points.alignment_point_neighbors)))

    # Create a version of the reference frame with small white crosses at alignment points and
    # alignment point neighbors.
    start = time()
    reference_frame_with_alignment_points = stack(
        (align_frames.frames_mono[align_frames.frame_ranks_max_index],) * 3, -1)
    cross_half_len = 5
    for j, y_center in enumerate(alignment_points.y_locations):
        for i, x_center in enumerate(alignment_points.x_locations):
            if alignment_points.alignment_boxes[j][i]['type'] == 'alignment point' or \
                    alignment_points.alignment_boxes[j][i]['type'] == 'alignment point neighbor':
                Miscellaneous.insert_cross(reference_frame_with_alignment_points, y_center,
                                           x_center, cross_half_len, 'white')

    for [j, i, y_center, x_center, y_low, y_high, x_low,
         x_high] in alignment_points.alignment_points:
        Miscellaneous.insert_cross(reference_frame_with_alignment_points, y_center, x_center,
                                   cross_half_len, 'white')
    for [j, i, aps] in alignment_points.alignment_point_neighbors:
        y_center = alignment_points.alignment_boxes[j][i]['coordinates'][2]
        x_center = alignment_points.alignment_boxes[j][i]['coordinates'][3]
        Miscellaneous.insert_cross(reference_frame_with_alignment_points, y_center, x_center,
                                   cross_half_len, 'white')
    end = time()
    print('Elapsed time in drawing alignment points: {}'.format(end - start))
    # plt.imshow(reference_frame_with_alignment_points)
    # plt.show()

    # Select the frame index and an area on this frame where details on local shifts are to be
    # visualized.
    frame_index_details = 0
    y_center_low_details = 0
    y_center_high_details = 4000
    x_center_low_details = 0
    x_center_high_details = 6000
    warp_threshold = 0.1
    box_size_half = int(configuration.alignment_box_size / 2)

    for frame_index in range(frames.number):

        # Get a copy of the reference frame with white crosses at alignment points.
        frame_with_shifts = reference_frame_with_alignment_points.copy()

        # For all frames: Compute the local shifts for all alignment points (to be used for
        # de-warping).
        start = time()
        alignment_points.compute_alignment_point_shifts(frame_index)
        end = time()
        print("Elapsed time in computing point shifts for frame number " + str(
            frame_index) + ": " + str(end - start))

        # Insert small crosses into the reference frame showing local shifts. For alignment points
        # insert a red cross next to the unshifted white one. For alignment point neighbors insert
        # a green cross to show the interpolated shift. Mark empty boxes with a blue cross.
        for j, y_center in enumerate(alignment_points.y_locations):
            for i, x_center in enumerate(alignment_points.x_locations):
                if alignment_points.alignment_boxes[j][i]['type'] == 'alignment point':
                    color_cross = 'red'
                elif alignment_points.alignment_boxes[j][i]['type'] == 'alignment point neighbor':
                    color_cross = 'green'
                else:
                    color_cross = 'blue'
                Miscellaneous.insert_cross(frame_with_shifts,
                                           y_center + int(round(alignment_points.y_shifts[j][i])),
                                           x_center + int(round(alignment_points.x_shifts[j][i])),
                                           cross_half_len, color_cross)
        plt.imshow(frame_with_shifts)
        plt.show()

        # The following is executed only for one selected frame ...
        if frame_index == frame_index_details:
            reference_frame = reference_frame_with_alignment_points.copy()
            for j, y_center in enumerate(alignment_points.y_locations):
                for i, x_center in enumerate(alignment_points.x_locations):
                    # ... and only if the alignment point lies in the specified region.
                    if y_center_low_details <= y_center <= y_center_high_details and \
                            x_center_low_details <= x_center <= x_center_high_details:

                        # Display on the left the reference frame at the box position with a
                        # centered white cross.
                        reference_frame_box = reference_frame[
                                              y_center - box_size_half:y_center + box_size_half,
                                              x_center - box_size_half:x_center + box_size_half]
                        dy = align_frames.intersection_shape[0][0] - \
                             align_frames.frame_shifts[frame_index][0]
                        dx = align_frames.intersection_shape[1][0] - \
                             align_frames.frame_shifts[frame_index][1]

                        # Display in the middle the globally shifted selected frame with a centered
                        # red cross. First, produce a three channel RGB object from the mono
                        # channel.
                        box_in_frame = stack((frames.frames_mono[frame_index],) * 3, -1)[
                                       y_center - box_size_half + dy:y_center + box_size_half + dy,
                                       x_center - box_size_half + dx:x_center + box_size_half + dx]
                        Miscellaneous.insert_cross(box_in_frame, box_size_half, box_size_half,
                                                   cross_half_len, 'red')

                        # On the right, the de-warped box is displayed with a red cross for an
                        # alignment point, or a green cross for an alignment point neighbor. For
                        # all other points no shift is applied and the cross is colored blue.
                        point_dy = alignment_points.y_shifts[j][i]
                        point_dx = alignment_points.x_shifts[j][i]
                        point_dy_int = int(round(point_dy))
                        point_dx_int = int(round(point_dx))
                        if alignment_points.alignment_boxes[j][i]['type'] == 'alignment point':
                            color_cross = 'red'
                        elif alignment_points.alignment_boxes[j][i][
                            'type'] == 'alignment point neighbor':
                            color_cross = 'green'
                        else:
                            color_cross = 'blue'

                        # The three views are displayed only if the shift is larger than a
                        # given threshold
                        if max(abs(point_dy), abs(point_dx)) < warp_threshold:
                            continue

                        print("frame shifts: " + str(dy) + ", " + str(dx))
                        print("Point shifts: " + str(point_dy) + ", " + str(point_dx))

                        # Again, create an RGB image from the mono channel, and insert either a
                        # green or red cross.
                        box_in_frame_shifted = stack((frames.frames_mono[frame_index],) * 3, -1)[
                                               y_center - box_size_half + dy -
                                               point_dy_int:y_center + box_size_half + dy -
                                                            point_dy_int,
                                               x_center - box_size_half + dx -
                                               point_dx_int:x_center + box_size_half + dx -
                                                            point_dx_int]
                        Miscellaneous.insert_cross(box_in_frame_shifted, box_size_half,
                                                   box_size_half, cross_half_len, color_cross)

                        # Display the three views around the alignment point.
                        fig = plt.figure(figsize=(12, 6))
                        ax1 = plt.subplot(1, 3, 1)
                        ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
                        ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)
                        ax1.imshow(reference_frame_box)
                        ax1.set_axis_off()
                        ax1.set_title(
                            'Reference frame, y :' + str(y_center) + ", x:" + str(x_center))
                        ax2.imshow(box_in_frame)
                        ax2.set_axis_off()
                        ax2.set_title('Frame, dy: ' + str(dy) + ", dx: " + str(dx))
                        ax3.imshow(box_in_frame_shifted)
                        ax3.set_axis_off()
                        ax3.set_title('De-warped, dy: ' + str(point_dy) + ", dx: " + str(point_dx))
                        plt.show()
