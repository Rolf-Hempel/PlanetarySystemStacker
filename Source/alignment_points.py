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
from numpy import arange, amax, stack, amin
from skimage.feature import register_translation

from align_frames import AlignFrames
from configuration import Configuration
from exceptions import WrongOrderingError, NotSupportedError
from frames import Frames
from miscellaneous import quality_measure, insert_cross, circle_around, translation
from rank_frames import RankFrames


class AlignmentPoints(object):
    """
        Create a rectangular grid of potential places for alignment points. For each location create a small "alignment
        box" around it. For each alignment box test if there is enough structure and brightness in the picture to use it
        as an alignment point. For all alignment points in all frames, compute the local shift relative to the mean
        frame.

    """

    def __init__(self, configuration, frames, rank_frames, align_frames):
        """
        Initialize the AlignmentPoints object and compute the mean frame.

        :param configuration: Configuration object with parameters
        :param frames: Frames object with all video frames
        :param rank_frames: RankFrames object with global quality ranks (between 0. and 1., 1. being optimal) for all
                            frames
        :param align_frames: AlignFrames object with global shift information for all frames
        """

        self.configuration = configuration
        self.frames = frames
        self.rank_frames = rank_frames
        self.align_frames = align_frames
        self.y_locations = None
        self.x_locations = None
        self.alignment_boxes = None
        self.alignment_points = None

        # Compute the number of frames to be used for creating the mean frame.
        self.average_frame_number = max(ceil(frames.number * configuration.average_frame_percent / 100.), 1)

        # Call method "average_frame" of the AlignFrames object to compute the mean frame. Use the best frames
        # according to their ranks.
        self.align_frames.average_frame(
            [self.frames.frames_mono[i] for i in self.rank_frames.quality_sorted_indices[:self.average_frame_number]],
            [self.align_frames.frame_shifts[i] for i in
             self.rank_frames.quality_sorted_indices[:self.average_frame_number]])

    def create_alignment_boxes(self, step_size, box_size):
        """
        Create a rectangular grid of potential places for alignment points. For each location create a small "alignment
        box" around it.

        :param step_size: Number of pixels in y and x directions between alignment boxes
        :param box_size: Extension of alignment boxes in both y and x directions
        :return: -
        """

        mean_frame = self.align_frames.mean_frame
        mean_frame_shape = mean_frame.shape
        box_size_half = int(box_size / 2)
        self.alignment_boxes = []

        # Create the y and x coordinates for the rectangular grid of alignment boxes. Keep the boxes away from the
        # frame borders enough to avoid that boxes reach beyond the frame border in the search for the local
        # displacement.
        self.y_locations = arange(box_size_half + self.configuration.alignment_point_search_width,
                                  mean_frame_shape[0] - box_size_half - self.configuration.alignment_point_search_width,
                                  step_size, dtype=int)
        self.x_locations = arange(box_size_half + self.configuration.alignment_point_search_width,
                                  mean_frame_shape[1] - box_size_half - self.configuration.alignment_point_search_width,
                                  step_size, dtype=int)
        for j, y in enumerate(self.y_locations):
            for i, x in enumerate(self.x_locations):

                # Compute the index bounds of the alignment box.
                y_low = y - box_size_half
                y_high = y + box_size_half
                x_low = x - box_size_half
                x_high = x + box_size_half

                # Cut the reference box from the mean frame.
                box = mean_frame[y_low:y_high, x_low:x_high]

                # Initialize a dictionary for the alignment box which contains all related info.
                alignment_box = {}
                alignment_box['box'] = box
                alignment_box['coordinates'] = (j, i, y, x, y_low, y_high, x_low, x_high)

                # Compute structure and brightness information for the alignment box, and add the box to the list.
                alignment_box['structure'] = quality_measure(box)
                alignment_box['max_brightness'] = amax(box)
                alignment_box['min_brightness'] = amin(box)
                self.alignment_boxes.append(alignment_box)

        # Normalize the structure information for all boxes by dividing by the maximum value.
        structure_max = max(alignment_box['structure'] for alignment_box in self.alignment_boxes)
        for alignment_box in self.alignment_boxes:
            alignment_box['structure'] /= structure_max

    def select_alignment_points(self, structure_threshold, brightness_threshold, contrast_threshold):
        """
        For each alignment box decide if there is enough structure and brightness to make it an alignment point. If so,
        add it to the list of alignment points.

        :param structure_threshold: Minimum structure value for an alignment point (between 0. and 1.)
        :param brightness_threshold: The brightest pixel must be brighter than this value (0 < value <256)
        :param contrast_threshold: The difference between the brightest and darkest pixel values must be larger than
                                   this value (0 < value < 256)
        :return: -
        """

        if self.alignment_boxes == None:
            raise WrongOrderingError("Attempt to select alignment points before alignment boxes are created")
        self.alignment_points = [[box_index, coordinates] for [box_index, coordinates] in
                                 enumerate(box['coordinates'] for box in self.alignment_boxes)
                                 if
                                 self.alignment_boxes[box_index]['structure'] > structure_threshold and
                                 self.alignment_boxes[box_index]['max_brightness'] > brightness_threshold and
                                 self.alignment_boxes[box_index]['max_brightness'] -
                                 self.alignment_boxes[box_index]['min_brightness'] > contrast_threshold]

    def compute_alignment_point_shifts(self, frame_index):
        """
        For each alignment point compute the shifts in y and x relative to the mean frame. Three different methods can
        be used to compute the shift values:
        - a subpixel algorithm from "skimage.feature"
        - a phase correlation algorithm (miscellaneous.translation)
        - a local search algorithm (spiralling from center outwards), see method "search_local_match"

        :param frame_index: Index of the selected frame in the list of frames.
        :return: For the first method, shifts, errors, diffphrases. For an explanation, refer to skimage documentation.
                 For the other two methods: a list of shifts [shift_y, shift_x] for all alignment points.
        """

        if self.alignment_points == None:
            raise WrongOrderingError("Attempt to compute alignment point shifts before selecting alingment points")

        # Initialize lists to be returned.
        point_shifts = []
        diffphases = []
        errors = []

        # For each alignment point, compute the shift for the given frame index.
        for point_index, [box_index, [j, i, y_center, x_center, y_low, y_high, x_low, x_high]] in enumerate(
                self.alignment_points):

            # The offsets dy and dx are caused by two effects: First, the mean frame is smaller than the original
            # frames. It only contains their intersection. And second, because the given frame is globally shifted
            # as compared to the mean frame.
            dy = self.align_frames.intersection_shape[0][0] - self.align_frames.frame_shifts[frame_index][0]
            dx = self.align_frames.intersection_shape[1][0] - self.align_frames.frame_shifts[frame_index][1]

            # Cut out the alignment box from the given frame. Take into account the offsets explained above.
            box_in_frame = self.frames.frames_mono[frame_index][y_low + dy:y_high + dy, x_low + dx:x_high + dx]

            # Use subpixel registration from skimage.feature, with accuracy 1/10 pixels.
            if self.configuration.alignment_point_method == 'Subpixel':
                shift_pixel, error, diffphase = register_translation(self.alignment_boxes[box_index]['box'],
                                                                     box_in_frame,
                                                                     10, space='real')
                diffphases.append(diffphase)
                errors.append(error)

            # Use a simple phase shift computation (contained in module "miscellaneous").
            elif self.configuration.alignment_point_method == 'CrossCorrelation':
                shift_pixel = translation(self.alignment_boxes[box_index]['box'], box_in_frame,
                                                            box_in_frame.shape)

            # Use a local search (see method "search_local_match" below.
            elif self.configuration.alignment_point_method == 'LocalSearch':
                shift_pixel = self.search_local_match(self.alignment_boxes[box_index]['box'],
                                                      self.frames.frames_mono[frame_index],
                                                      y_low + dy, y_high + dy, x_low + dx, x_high + dx,
                                                      self.configuration.alignment_point_search_width)
            else:
                raise NotSupportedError(
                    "The point shift computation method " + self.configuration.alignment_point_method + " is not implemented")
            point_shifts.append(shift_pixel)
        return point_shifts, errors, diffphases

    def search_local_match(self, reference_box, frame, y_low, y_high, x_low, x_high, search_width):
        """
        Try shifts in y, x between the box around the alignment point in the mean frame and the corresponding box in
        the given frame. Start with shifts [0, 0] and move out in a circular fashion, until the radius "search_width"
        is reached. The global frame shift is accounted for beforehand already.

        :param reference_box: Image box around alignment point in mean frame.
        :param frame: Given frame for which the local shift at the alignment point is to be computed.
        :param y_low: Lower y coordinate limit of box in given frame, taking into account the global shift and the
                      different sizes of the mean frame and the original frames.
        :param y_high: Upper y coordinate limit
        :param x_low: Lower x coordinate limit
        :param x_high: Upper x coordinate limit
        :param search_width: maximum radius of the search sprial
        :return: Local shift in the form [shift_y, shift_x], or [0, 0] if no optimum could be found.
        """

        # Initialize the global optimum with an impossibly large value.
        deviation_min = 1000000
        dy_min = None
        dx_min = None
        dev = []

        # Start with shift [0, 0] and proceed in a circular pattern.
        for r in arange(search_width + 1):

            # Create an enumerator which produces shift values [dy, dx] in a circular pattern with radius "r".
            circle_r = circle_around(0, 0, r)

            # Initialize the optimum for radius "r" to an impossibly large value, and the corresponding shifts to None.
            deviation_min_r, dy_min_r, dx_min_r = 1000000, None, None

            # Go through the circle with radius "r" and compute the difference (deviation) between the shifted frame
            # and the corresponding box in the mean frame. Find the minimum "deviation_min_r" for radius "r".
            for (dx, dy) in circle_r:
                deviation = abs(reference_box - frame[y_low - dy:y_high - dy, x_low - dx:x_high - dx]).sum()
                if deviation < deviation_min_r:
                    deviation_min_r, dy_min_r, dx_min_r = deviation, dy, dx

            # Store the optimal value for radius "r".
            dev.append(deviation_min_r)

            # If for the current radius there is no improvement compared to the previous radius, the optimum if reached.
            if deviation_min_r >= deviation_min:
                return [dy_min, dx_min]

            # Otherwise, update the current optimum and continue.
            else:
                deviation_min, dy_min, dx_min = deviation_min_r, dy_min_r, dx_min_r
        # print("search local match unsuccessful: y_low: " + str(y_low) + ", x_low: " + str(x_low))
        # print(
        #     "search local match unsuccessful: y: " + str((y_high + y_low) / 2.) + ", x: " + str((x_high + x_low) / 2.))

        # If within the maximum search radius no optimum could be found, return [0, 0].
        return [0, 0]


if __name__ == "__main__":
    # Images can either be extracted from a video file or a batch of single photographs. Select the example for
    # the test run.
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
    print("Frame scores (sorted): " + str([rank_frames.frame_ranks[i] for i in rank_frames.quality_sorted_indices]))
    print("Sorted index list: " + str(rank_frames.quality_sorted_indices))

    # Initialize the frame alignment object.
    align_frames = AlignFrames(frames, rank_frames, configuration)
    start = time()
    # Select the local rectangular patch in the image where the L gradient is highest in both x and y direction. The
    # scale factor specifies how much smaller the patch is compared to the whole image frame.
    (x_low_opt, x_high_opt, y_low_opt, y_high_opt) = align_frames.select_alignment_rect(
        configuration.alignment_rectangle_scale_factor)
    end = time()
    print('Elapsed time in computing optimal alignment rectangle: {}'.format(end - start))
    print("optimal alignment rectangle, x_low: " + str(x_low_opt) + ", x_high: " + str(x_high_opt) + ", y_low: " + str(
        y_low_opt) + ", y_high: " + str(y_high_opt))
    reference_frame_with_alignment_points = align_frames.frames_mono[align_frames.frame_ranks_max_index].copy()
    reference_frame_with_alignment_points[y_low_opt, x_low_opt:x_high_opt] = reference_frame_with_alignment_points[
                                                                             y_high_opt - 1, x_low_opt:x_high_opt] = 255
    reference_frame_with_alignment_points[y_low_opt:y_high_opt, x_low_opt] = reference_frame_with_alignment_points[
                                                                             y_low_opt:y_high_opt, x_high_opt - 1] = 255
    # plt.imshow(reference_frame_with_alignment_points, cmap='Greys_r')
    # plt.show()

    # Align all frames globally relative to the frame with the highest score.
    start = time()
    align_frames.align_frames()
    end = time()
    print('Elapsed time in aligning all frames: {}'.format(end - start))
    print("Frame shifts: " + str(align_frames.frame_shifts))
    print("Intersection: " + str(align_frames.intersection_shape))

    # Initialize the AlignmentPoints object. This includes the computation of the average frame against which the
    # alignment point shifts are measured.
    start = time()
    alignment_points = AlignmentPoints(configuration, frames, rank_frames, align_frames)
    end = time()
    print('Elapsed time in computing average frame: {}'.format(end - start))
    print("Average frame computed from the best " + str(alignment_points.average_frame_number) + " frames.")
    # plt.imshow(align_frames.mean_frame, cmap='Greys_r')
    # plt.show()

    # Create a regular grid with small boxes. A subset of those boxes will be selected as alignment points.
    step_size = configuration.alignment_box_step_size
    box_size = configuration.alignment_box_size
    start = time()
    alignment_points.create_alignment_boxes(step_size, box_size)
    end = time()
    print('Elapsed time in alignment box creation: {}'.format(end - start))
    print("Number of alignment boxes created: " + str(len(alignment_points.alignment_boxes)))

    # An alignment box is selected as an alignment point if it satisfies certain conditions regarding local contrast
    # etc.
    structure_threshold = configuration.alignment_point_structure_threshold
    brightness_threshold = configuration.alignment_point_brightness_threshold
    contrast_threshold = configuration.alignment_point_contrast_threshold
    print("Selection of alignment points, structure threshold: " + str(
        structure_threshold) + ", brightness threshold: " + str(brightness_threshold) + ", contrast threshold: " + str(
        contrast_threshold))
    start = time()
    alignment_points.select_alignment_points(structure_threshold, brightness_threshold, contrast_threshold)
    end = time()
    print('Elapsed time in alignment point selection: {}'.format(end - start))
    print("Number of alignment points selected: " + str(len(alignment_points.alignment_points)))

    # Create a version of the reference frame with small white crosses at alignment points.
    start = time()
    reference_frame_with_alignment_points = stack((align_frames.frames_mono[align_frames.frame_ranks_max_index],) * 3,
                                                  -1)
    cross_half_len = 5
    for [index, [j, i, y_center, x_center, y_low, y_high, x_low, x_high]] in alignment_points.alignment_points:
        insert_cross(reference_frame_with_alignment_points, y_center, x_center, cross_half_len, 'white')
    end = time()
    print('Elapsed time in drawing alignment points: {}'.format(end - start))
    # plt.imshow(reference_frame_with_alignment_points)
    # plt.show()

    # Select the frame index an area on this frame where details on local shifts are to be visualized.
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

        # For all frames: Compute the local shifts for all alignment points (to be used for de-warping).
        start = time()
        point_shifts, errors, diffphases = alignment_points.compute_alignment_point_shifts(frame_index)
        end = time()
        print("Elapsed time in computing point shifts for frame number " + str(frame_index) + ": " + str(end - start))

        # Insert small crosses into the reference frame showing local shifts. If a shift has been computed, insert
        # a red cross next to the unshifted white one. Otherwise replace the white cross with a green one.
        for point_index, [index, [j, i, y_center, x_center, y_low, y_high, x_low, x_high]] in enumerate(
                alignment_points.alignment_points):
            if point_shifts[point_index][0] == None:
                insert_cross(frame_with_shifts, y_center, x_center, cross_half_len, 'green')
            else:
                insert_cross(frame_with_shifts, y_center + int(round(point_shifts[point_index][0])),
                             x_center + int(round(point_shifts[point_index][1])),
                             cross_half_len, 'red')
        plt.imshow(frame_with_shifts)
        plt.show()

        # The following is executed only for one selected frame ...
        if frame_index == frame_index_details:
            reference_frame = reference_frame_with_alignment_points.copy()
            for point_index, [index, [j, i, y_center, x_center, y_low, y_high, x_low, x_high]] in enumerate(
                    alignment_points.alignment_points):

                # ... and only if the alignment point lies in the specified region.
                if y_center_low_details <= y_center <= y_center_high_details and x_center_low_details <= x_center <= x_center_high_details:

                    # Display on the left the reference frame at the box position with a centered white cross.
                    reference_frame_box = reference_frame[y_center - box_size_half:y_center + box_size_half,
                                          x_center - box_size_half:x_center + box_size_half]
                    dy = align_frames.intersection_shape[0][0] - align_frames.frame_shifts[frame_index][0]
                    dx = align_frames.intersection_shape[1][0] - align_frames.frame_shifts[frame_index][1]

                    # Display in the middle the globally shifted selected frame with a centered red cross. First,
                    # produce a three channel RGB object from the mono channel.
                    box_in_frame = stack((frames.frames_mono[frame_index],) * 3, -1)[
                                   y_center - box_size_half + dy:y_center + box_size_half + dy,
                                   x_center - box_size_half + dx:x_center + box_size_half + dx]
                    insert_cross(box_in_frame, box_size_half, box_size_half, cross_half_len, 'red')

                    # On the right, the de-warped box is displayed with a red cross, if a shift has been computed for
                    # this alignment point. Otherwise no shift is applied and the cross is colored green.
                    if point_shifts[point_index][0] == None:
                        point_dy = point_dx = point_dy_int = point_dx_int = 0
                        color_cross = 'green'
                    else:
                        point_dy = point_shifts[point_index][0]
                        point_dx = point_shifts[point_index][1]
                        point_dy_int = int(round(point_dy))
                        point_dx_int = int(round(point_dx))
                        color_cross = 'red'

                    # The three views are displayed only if the shift is larger than a given threshold and, and
                    # a shift has been computed at all.
                    if max(abs(point_dy), abs(point_dx)) < warp_threshold and not point_shifts[point_index][0] == None:
                        continue
                    print("frame shifts: " + str(dy) + ", " + str(dx))

                    # For different alignment methods, different information is available for display.
                    if configuration.alignment_point_method == 'Subpixel':
                        print("Point shifts: " + str(point_dy) + ", " + str(point_dx) + ", Error: "
                              + str(errors[point_index]) + ", diffphase: " + str(diffphases[point_index]))
                    else:
                        print("Point shifts: " + str(point_dy) + ", " + str(point_dx))

                    # Again, create an RGB image from the mono channel, and insert either a green or red cross.
                    box_in_frame_shifted = stack((frames.frames_mono[frame_index],) * 3, -1)[
                                           y_center - box_size_half + dy - point_dy_int:y_center + box_size_half + dy - point_dy_int,
                                           x_center - box_size_half + dx - point_dx_int:x_center + box_size_half + dx - point_dx_int]
                    insert_cross(box_in_frame_shifted, box_size_half, box_size_half, cross_half_len, color_cross)

                    # Display the three views around the alignment point.
                    fig = plt.figure(figsize=(12, 6))
                    ax1 = plt.subplot(1, 3, 1)
                    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
                    ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)
                    ax1.imshow(reference_frame_box)
                    ax1.set_axis_off()
                    ax1.set_title('Reference frame, y :' + str(y_center) + ", x:" + str(x_center))
                    ax2.imshow(box_in_frame)
                    ax2.set_axis_off()
                    ax2.set_title('Frame, dy: ' + str(dy) + ", dx: " + str(dx))
                    ax3.imshow(box_in_frame_shifted)
                    ax3.set_axis_off()
                    ax3.set_title('De-warped, dy: ' + str(point_dy) + ", dx: " + str(point_dx))
                    plt.show()
