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
from time import time, sleep
from decimal import Decimal

import matplotlib.pyplot as plt
from math import ceil
from numpy import arange, amax, stack, amin, float32, uint8, zeros, sqrt, empty, int32, uint16
from scipy import ndimage
from skimage.feature import register_translation
from cv2 import meanStdDev, GaussianBlur, destroyAllWindows, waitKey, imshow, FONT_HERSHEY_SIMPLEX,\
    putText, resize

from align_frames import AlignFrames
from configuration import Configuration
from exceptions import NotSupportedError, Error
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

    def __init__(self, configuration, frames, rank_frames, align_frames, progress_signal=None):
        """
        Initialize the AlignmentPoints object and compute the mean frame.

        :param configuration: Configuration object with parameters
        :param frames: Frames object with all video frames
        :param rank_frames: RankFrames object with global quality ranks (between 0. and 1.,
                            1. being optimal) for all frames
        :param align_frames: AlignFrames object with global shift information for all frames
        :param progress_signal: Either None (no progress signalling), or a signal with the signature
                                (str, int) with the current activity (str) and the progress in
                                percent (int).
        """

        self.configuration = configuration
        self.frames = frames
        self.rank_frames = rank_frames
        self.align_frames = align_frames
        self.progress_signal = progress_signal

        # Apply a low-pass filter on the mean frame as a preparation for shift detection.
        self.mean_frame = GaussianBlur(align_frames.mean_frame.astype(uint16),
                                        (self.configuration.frames_gauss_width,
                                        self.configuration.frames_gauss_width), 0).astype(int32)

        if self.configuration.alignment_points_method == 'MultiLevel':
            self.mean_frames = []

            for level in range(0, self.configuration.alignment_points_number_levels):
                stride = 2**level

                self.mean_frames.append(GaussianBlur(align_frames.mean_frame[::stride, ::stride].astype(uint16),
                                           (self.configuration.alignment_points_mean_frame_noise[level],
                                            self.configuration.alignment_points_mean_frame_noise[level]), 0).astype(int32))

            self.mean_frame = self.mean_frames[0]

        else:
            # Apply a low-pass filter on the mean frame as a preparation for shift detection.
            self.mean_frame = GaussianBlur(align_frames.mean_frame.astype(uint16),
                                           (self.configuration.frames_gauss_width,
                                            self.configuration.frames_gauss_width), 0).astype(int32)

        self.num_pixels_y = self.mean_frame.shape[0]
        self.num_pixels_x = self.mean_frame.shape[1]
        self.alignment_points_dropped_dim = None
        self.alignment_points_dropped_structure = None

        # Initialize counters for APs which are dropped because
        # they do not satisfy the brightness or structure condition.
        self.alignment_points = []
        self.alignment_points_dropped_dim = 0
        self.alignment_points_dropped_structure = 0

        # Initialize the number of frames to be stacked at each AP.
        self.stack_size = None

        self.dev_table = empty((2 * self.configuration.alignment_points_search_width + 1,
                               2 * self.configuration.alignment_points_search_width + 1), dtype=float32)

        self.debug_AP = False
        self.update_image_window_signal = None
        self.scale_factor = None
        self.border = None
        self.image_delay = None

    @staticmethod
    def ap_locations(num_pixels, min_boundary_distance, step_size, even):
        """
        Compute optimal alignment patch coordinates in one coordinate direction. Place boundary
        neighbors as close as possible to the boundary.

        :param num_pixels: Number of pixels in the given coordinate direction
        :param min_boundary_distance: Minimum distance of an AP from the boundary
        :param step_size: Distance of alignment boxes
        :param even: If True, compute locations for even row indices. Otherwise, for odd ones.
        :return: List of alignment point coordinates in the given direction
        """

        # The number of interior alignment boxes in general is not an integer. Round to the next
        # higher number.
        num_interior_odd = int(ceil((num_pixels - 2 * min_boundary_distance) / step_size))
        # Because alignment points are arranged in a staggered grid, in even rows there is one point
        # more.
        num_interior_even = num_interior_odd + 1

        # The precise distance between alignment points will differ slightly from the specified
        # step_size. Compute the exact distance. Integer locations will be rounded later.
        distance_corrected = (num_pixels - 2 * min_boundary_distance) / num_interior_odd

        # Compute the AP locations, separately for even and odd rows.
        if even:
            locations = [int(min_boundary_distance + i * distance_corrected) for i in range(num_interior_even)]
        else:
            locations = [int(min_boundary_distance + 0.5 * distance_corrected +
                             i * distance_corrected) for i in range(num_interior_odd)]
        return locations

    def create_ap_grid(self):
        """
        Create a 2D staggered grid of alignment points. For each AP compute its center coordinates,
        and the coordinate limits of its alignment box and alignment patch. Only alignment points
        which satisfy the conditions on brightness, contrast and structure are eventually added to
        the list.

        :return: List of alignment points.
        """

        # The alignment patch is the area which is stacked after a rigid displacement.
        half_patch_width = self.configuration.alignment_points_half_patch_width
        # The alignment box is the object on which the displacement computation is performed.
        half_box_width = self.configuration.alignment_points_half_box_width
        # Number of pixels in one coordinate direction between alignment points
        step_size = self.configuration.alignment_points_step_size
        # Maximum displacement searched for in the alignment process.
        search_width = self.configuration.alignment_points_search_width
        # Minimum structure value for an alignment point (between 0. and 1.)
        structure_threshold = self.configuration.alignment_points_structure_threshold
        # The brightest pixel must be brighter than this value (0 < value <256). Please note that
        # brightness and contrast values are converted to 16bit resolution.
        brightness_threshold = self.configuration.alignment_points_brightness_threshold * 256
        # The difference between the brightest and darkest pixel values must be larger than this
        # value (0 < value < 256)
        contrast_threshold = self.configuration.alignment_points_contrast_threshold * 256

        # Compute the minimum distance of an AP from the boundary.
        min_boundary_distance = max(half_box_width + search_width, half_patch_width)

        # Compute y and x coordinate locations of alignemnt points. Note that the grid is staggered.
        ap_locations_y = self.ap_locations(self.num_pixels_y, min_boundary_distance,
                                           step_size, True)
        ap_locations_x_even = self.ap_locations(self.num_pixels_x, min_boundary_distance,
                                                step_size, True)
        ap_locations_x_odd = self.ap_locations(self.num_pixels_x, min_boundary_distance,
                                               step_size, False)

        # Reset the alignment point list, and initialize counters for APs which are dropped because
        # they do not satisfy the brightness or structure condition.
        self.alignment_points = []
        self.alignment_points_dropped_dim = 0
        self.alignment_points_dropped_structure = 0

        # Compute the minimum distance of an AP center from the frame boundary.
        min_boundary_distance = max(
            self.configuration.alignment_points_half_box_width + \
            self.configuration.alignment_points_search_width,
            self.configuration.alignment_points_half_patch_width)

        # Create alignment point rows, start with an even one.
        even = True
        for index_y, y in enumerate(ap_locations_y):
            # For the first row extend the patch to the upper frame border, and for the last row
            # to the lower frame border.
            extend_y_low  = (index_y == 0)
            extend_y_high = (index_y == len(ap_locations_y)-1)

            # Create x coordinate, depending on the y row being even or odd (staggered grid).
            if even:
                ap_locations_x = ap_locations_x_even
            else:
                ap_locations_x = ap_locations_x_odd

            # For each location create an alignment point.
            for index_x, x in enumerate(ap_locations_x):
                # For the first point in a row, extend the patch to the left frame border, and for
                # the last point in a row to the right frame border.
                extend_x_low  = (index_x == 0)
                extend_x_high = (index_x == len(ap_locations_x) - 1)

                alignment_point = self.new_alignment_point(y, x, extend_x_low, extend_x_high,
                                                           extend_y_low, extend_y_high)

                # Compute structure and brightness information for the alignment box.
                max_brightness = amax(alignment_point['reference_box'])
                min_brightness = amin(alignment_point['reference_box'])
                # If the alignment box satisfies the brightness conditions, add the AP to the list.
                if max_brightness > brightness_threshold and max_brightness - \
                        min_brightness > contrast_threshold:

                    # Check if the fraction of dark pixels exceeds a threshold.
                    box = alignment_point['reference_box']
                    fraction = (box < brightness_threshold).sum() / (box.shape[0] * box.shape[1])
                    if fraction > self.configuration.alignment_points_dim_fraction_threshold:

                        # Compute the center of mass of the brightness distribution within the box,
                        # and shift the box center to this location.
                        com = ndimage.measurements.center_of_mass(box)
                        y_adapted = y + int(com[0]) - half_box_width
                        x_adapted = x + int(com[1]) - half_box_width

                        y_adapted = max(y_adapted, min_boundary_distance)
                        y_adapted = min(y_adapted, self.num_pixels_y - min_boundary_distance)
                        x_adapted = max(x_adapted, min_boundary_distance)
                        x_adapted = min(x_adapted, self.num_pixels_x - min_boundary_distance)

                        # Replace the alignment point with a new one, using the updated
                        # coordinates.
                        alignment_point = self.new_alignment_point(y_adapted, x_adapted,
                                                                   extend_x_low, extend_x_high,
                                                                   extend_y_low, extend_y_high)

                    alignment_point['structure'] = Miscellaneous.quality_measure(
                        alignment_point['reference_box'])
                    self.alignment_points.append(alignment_point)
                else:
                    # If a point does not satisfy the conditions, increase the counter.
                    self.alignment_points_dropped_dim += 1

            # Switch between even and odd rows.
            even = not even

        # Normalize the structure information for all alignment point boxes by dividing by the
        # maximum value.
        structure_max = max(
            alignment_point['structure'] for alignment_point in self.alignment_points)
        alignment_points_dropped_structure_indices = []
        for alignment_point_index, alignment_point in enumerate(self.alignment_points):
            alignment_point['structure'] /= structure_max
            # Remove alignment points with too little structure and increment the counter.
            if alignment_point['structure'] < structure_threshold:
                alignment_points_dropped_structure_indices.append(alignment_point_index)
                self.alignment_points_dropped_structure += 1

        # Remove alignment points which do not satisfy the structure condition, if there is any.
        if alignment_points_dropped_structure_indices:
            alignment_points_new = []
            dropped_index = 0
            for alignment_point_index, alignment_point in enumerate(self.alignment_points):
                if alignment_point_index != alignment_points_dropped_structure_indices[
                    dropped_index]:
                    alignment_points_new.append(alignment_point)
                elif dropped_index < len(alignment_points_dropped_structure_indices) - 1:
                    dropped_index += 1
            self.alignment_points = alignment_points_new

    def new_alignment_point(self, y, x, extend_x_low, extend_x_high, extend_y_low, extend_y_high):
        """
        Create a new alignment point. This method is called in creating the initial alignment point
        grid. Later it can be invoked by the user to add single alignment points.

        :param y: y coordinate of alignment point center
        :param x: x coordinate of alignment point center
        :param extend_x_low: True, if patch is to be extended to the left frame boundary.
                             False otherwise.
        :param extend_x_high: True, if patch is to be extended to the right frame boundary.
                              False otherwise.
        :param extend_y_low: True, if patch is to be extended to the upper frame boundary.
                             False otherwise.
        :param extend_y_high: True, if patch is to be extended to the lower frame boundary.
                              False otherwise.
        :return: If successful, the alignment point object is returned; otherwise None.
        """

        # If the patch does not fit into the frame, or the AP box is too close to the border,
        # the AP cannot be created at this place.
        min_boundary_distance = max(self.configuration.alignment_points_half_box_width + \
                                self.configuration.alignment_points_search_width,
                                self.configuration.alignment_points_half_patch_width)
        if y<min_boundary_distance or y>self.num_pixels_y-min_boundary_distance or \
            x<min_boundary_distance or x>self.num_pixels_x-min_boundary_distance:
            return None

        alignment_point = {}
        alignment_point['y'] = y
        alignment_point['x'] = x
        alignment_point['half_box_width'] = self.configuration.alignment_points_half_box_width
        alignment_point['box_y_low'] = y - self.configuration.alignment_points_half_box_width
        alignment_point['box_y_high'] = y + self.configuration.alignment_points_half_box_width
        alignment_point['box_x_low'] = x - self.configuration.alignment_points_half_box_width
        alignment_point['box_x_high'] = x + self.configuration.alignment_points_half_box_width

        alignment_point['patch_y_low'] = y - self.configuration.alignment_points_half_patch_width
        alignment_point['patch_y_high'] = y + self.configuration.alignment_points_half_patch_width
        if extend_y_low:
            alignment_point['patch_y_low'] = 0
        elif extend_y_high:
            alignment_point['patch_y_high'] = self.num_pixels_y

        alignment_point['patch_x_low'] = x - self.configuration.alignment_points_half_patch_width
        alignment_point['patch_x_high'] = x + self.configuration.alignment_points_half_patch_width
        if extend_x_low:
            alignment_point['patch_x_low'] = 0
        elif extend_x_high:
            alignment_point['patch_x_high'] = self.num_pixels_x

        # Initialize the reference to the corresponding widget in the AP viewer scene.
        alignment_point['graphics_item'] = None

        # Allocate buffers and fill alignment point box with mean frame.
        AlignmentPoints.set_reference_box(alignment_point, self.mean_frame)

        return alignment_point

    def add_alignment_point(self, ap):
        """
        Add an alignment point to the list.

        :param ap: AP to be added
        :return: -
        """

        self.alignment_points.append(ap)

    def remove_alignment_points(self, ap_list):
        """
        Remove a list of alignment points from the current list.

        :param ap_list: List of point objects to be removed
        :return: -
        """

        # Create list with unique identifiers of all items on the list.
        ap_list_ids = [id(ap_list_item) for ap_list_item in ap_list]

        # Replace the original AP list with the reduced one.
        # If the identifier of an alignment point does not match any list item, keep it.
        self.alignment_points = [ap for ap in self.alignment_points if id(ap) not in ap_list_ids]

    def replace_alignment_point(self, ap_old, ap_new):
        """
        Replace an alignment point with a new one.

        :param ap_old: Existing alignment point to be replaced
        :param ap_new: New alignment point to replace the old one
        :return: True if successful, False otherwise
        """

        ap_old_id = id(ap_old)
        for index, ap in enumerate(self.alignment_points):
            if id(ap) == ap_old_id:
                self.alignment_points[index] = ap_new
                return True
        return False

    def move_alignment_point(self, ap, y_new, x_new):
        """
        Move an existing AP to a different position.

        :param ap: Esisting alignment point to be moved
        :param y_new: New y coordinate of AP center
        :param x_new: New x coordinate of AP center
        :return: The updated AP; or None, if unsuccessful
        """

        # Try to do the changes. If unsuccessful, return None.
        return self.change_alignment_point(ap, y_new, x_new, ap['half_box_width'])

    def resize_alignment_point(self, ap, factor):
        """
        Change the size of an existing alignment point.

        :param ap: Esisting alignment point to be resized
        :param factor: Factor by which the size is to be changed
        :return: The resized alignment point; or None, if unsuccessful
        """

        half_box_width_new = ap['half_box_width'] * factor

        # Try to do the changes. If unsuccessful, return None.
        return self.change_alignment_point(ap, ap["y"], ap["x"], half_box_width_new)

    def change_alignment_point(self, ap, y, x, half_box_width):
        """
        Try to fit a resized or moved AP into the frame.

        :param ap: AP to be resized or moved
        :param y: New y coordinate of center
        :param x: New x coordinate of center
        :param half_box_width: New half width of AP box
        :return: The resized / moved AP; or None, if unsuccessful
        """

        # Test if the AP box is too small.
        if half_box_width < self.configuration.alignment_points_min_half_box_width:
            return None

        # Compute new values for patch and box sizes.
        half_patch_width_new_int = round(half_box_width *
                                             self.configuration.alignment_points_half_patch_width /
                                             self.configuration.alignment_points_half_box_width)
        half_box_width_new_int = round(half_box_width)

        # Compute resized patch bounds. If resizing hits the image boundary on at least one side,
        # the operation is aborted.
        patch_y_low = y - half_patch_width_new_int
        if patch_y_low < 0:
            return None
        patch_y_high = y + half_patch_width_new_int
        if patch_y_high > self.num_pixels_y:
            return None
        patch_x_low = x - half_patch_width_new_int
        if patch_x_low < 0:
            return None
        patch_x_high = x + half_patch_width_new_int
        if patch_x_high > self.num_pixels_x:
            return None

        # Compute resized box bounds. If resizing moves the box closer to the frame border than
        # the search width on at least one side, the operation is aborted.
        box_y_low = y - half_box_width_new_int
        if box_y_low < self.configuration.alignment_points_search_width:
            return None
        box_y_high = y + half_box_width_new_int
        if box_y_high > self.num_pixels_y - self.configuration.alignment_points_search_width:
            return None
        box_x_low = x - half_box_width_new_int
        if box_x_low < self.configuration.alignment_points_search_width:
            return None
        box_x_high = x + half_box_width_new_int
        if box_x_high > self.num_pixels_x - self.configuration.alignment_points_search_width:
            return None

        # Perform the changes.
        ap['y'] = y
        ap['x'] = x
        ap['patch_y_low'] = patch_y_low
        ap['patch_y_high'] = patch_y_high
        ap['patch_x_low'] = patch_x_low
        ap['patch_x_high'] = patch_x_high
        ap['box_y_low'] = box_y_low
        ap['box_y_high'] = box_y_high
        ap['box_x_low'] = box_x_low
        ap['box_x_high'] = box_x_high
        ap['half_box_width'] = half_box_width

        # Invalidate buffers. To save computing time, they are not re-computed at every wheel
        # event.
        ap['graphics_item'] = None
        ap['reference_box'] = None
        ap['stacking_buffer'] = None

        return ap

    @staticmethod
    def set_reference_box(alignment_point, mean_frame):
        """
        For APs which have been changed in the AP editor, buffers have been invalidated. They have
        to be re-computed after AP editing is done.

        :param alignment_point: Alignment_point object
        :param mean_frame: Average frame
        :return: -
        """

        # Cut out the reference box from the mean frame, used in alignment.
        box = mean_frame[alignment_point['box_y_low']:alignment_point['box_y_high'],
              alignment_point['box_x_low']:alignment_point['box_x_high']]
        alignment_point['reference_box'] = box

    def set_reference_boxes(self):
        """
        This method is used if multi-level AP matching is selected. In this case a hierarchy of
        reference boxes with different resolution is constructed. Level 0 is the original frame,
        and each higher level has half the pixel count in x and y as compared to the previous level.

        This method is invoked when all APs are set, just before stacking.

        :return: -
        """

        for ap_index, alignment_point in enumerate(self.alignment_points):
            half_box_width_finest = alignment_point['x'] - alignment_point['box_x_low']
            y_finest = alignment_point['y']
            x_finest = alignment_point['x']

            alignment_point['y_levels'] = []
            alignment_point['x_levels'] = []
            alignment_point['half_box_widths'] = []
            alignment_point['reference_boxes'] = []

            for level in range(self.configuration.alignment_points_number_levels):
                half_box_width = int(
                    round(half_box_width_finest * self.configuration.alignment_points_box_factors[
                        level] / 2 ** level))

                y_level = int(round(y_finest / 2 ** level))
                x_level = int(round(x_finest / 2 ** level))

                alignment_point['y_levels'].append(y_level * 2 ** level)
                alignment_point['x_levels'].append(x_level * 2 ** level)
                alignment_point['half_box_widths'].append(half_box_width * 2 ** level)
                alignment_point['reference_boxes'].append(
                    self.mean_frames[level][y_level - half_box_width:y_level + half_box_width,
                    x_level - half_box_width:x_level + half_box_width])

            # if self.debug_AP and not ap_index:
            #     print('half_box_width_finest: ' + str(half_box_width_finest))
            #     for level in range(self.configuration.alignment_points_number_levels):
            #         print(str(alignment_point['y_levels'][level]) + ", " + str(
            #             alignment_point['x_levels'][level]) + ", " + str(
            #             alignment_point['half_box_widths'][level]))
            #         imshow(str(level), alignment_point['reference_boxes'][level])
            #         waitKey(0)
            #         destroyAllWindows()

    @staticmethod
    def initialize_ap_stacking_buffer(alignment_point, color):
        """
        In the stacking initialization, for each AP a stacking buffer has to be allocated.

        :param alignment_point: Alignment_point object
        :param color: True, if stacking is to be done for color frames. False for
        monochrome case.
        :return: -
        """

        # Allocate space for the stacking buffer.
        if color:
            alignment_point['stacking_buffer'] = zeros(
                [alignment_point['patch_y_high'] - alignment_point['patch_y_low'],
                 alignment_point['patch_x_high'] - alignment_point['patch_x_low'], 3],
                dtype=float32)
        else:
            alignment_point['stacking_buffer'] = zeros(
                [alignment_point['patch_y_high'] - alignment_point['patch_y_low'],
                 alignment_point['patch_x_high'] - alignment_point['patch_x_low']], dtype=float32)

    def find_alignment_points(self, y_low, y_high, x_low, x_high):
        """
        Find all alignment points the centers of which are within given (y, x) bounds.

        :param y_low: Lower y pixel coordinate bound
        :param y_high: Upper y pixel coordinate bound
        :param x_low: Lower x pixel coordinate bound
        :param x_high: Upper x pixel coordinate bound
        :return: List of all alignment points with centers within the given coordinate bounds.
                 If no AP satisfies the condition, return an empty list.
        """

        return [ap for ap in self.alignment_points if y_low <= ap['y'] <= y_high and
                x_low <= ap['x'] <= x_high]

    @staticmethod
    def find_neighbor(ap_y, ap_x, alignment_points):
        """
        For a given (y, x) position find the closest "real" alignment point.

        :param ap_y: y cocrdinate of location of interest
        :param ap_x: x cocrdinate of location of interest
        :param alignment_points: List of alignment points to be searched

        :return: Alignment point object of closest AP, and distance in pixels. If the list of
                 alignment points is empty, (None, None) is returned.
        """

        # If no APs have been created yet, return None for both results.
        if not alignment_points:
            return None, None

        # Start with an impossibly large distance, and find the closest AP on the list.
        min_distance_squared = 1.e30
        ap_neighbor = None
        for ap in alignment_points:
            distance_squared = (ap['y'] - ap_y) ** 2 + (ap['x'] - ap_x) ** 2
            if distance_squared < min_distance_squared:
                ap_neighbor = ap
                min_distance_squared = distance_squared
        return ap_neighbor, sqrt(min_distance_squared)

    def compute_frame_qualities(self):
        """
        For each alignment point compute a ranking of best frames. Store the list in the
        alignment point dictionary with the key 'best_frame_indices'.

        Consider the special case that sampled-down Laplacians have been stored for frame ranking.
        In this case they can be re-used for ranking the boxes around alignment points (but only
        if "Laplace" has been selected for alignment point ranking).

        :return: -
        """

        # If the user has entered a value for the number of frames, use it.
        if self.configuration.alignment_points_frame_number is not None:
            self.stack_size = self.configuration.alignment_points_frame_number
        # Otherwise compute the stack size from the given percentage. Take at least one frame.
        else:
            self.stack_size = max(int(ceil(
                self.frames.number * self.configuration.alignment_points_frame_percent / 100.)), 1)
        # Select the ranking method.
        if self.configuration.alignment_points_rank_method == "xy gradient":
            method = Miscellaneous.local_contrast
        elif self.configuration.alignment_points_rank_method == "Laplace":
            method = Miscellaneous.local_contrast_laplace
        elif self.configuration.alignment_points_rank_method == "Sobel":
            method = Miscellaneous.local_contrast_sobel
        else:
            raise NotSupportedError(
                "Ranking method " + self.configuration.alignment_points_rank_method +
                " not supported")

        # Compute the frequency of progress signals in the computational loop.
        if self.progress_signal is not None:
            self.signal_loop_length = max(self.frames.number, 1)
            self.signal_step_size = max(round(self.frames.number / 10), 1)

        # Initialize a list which for each AP contains the qualities of all frames at this point.
        for alignment_point in self.alignment_points:
            alignment_point['frame_qualities'] = []

        if self.configuration.rank_frames_method != "Laplace" or \
                self.configuration.alignment_points_rank_method != "Laplace":
            # There are no stored Laplacians, or they cannot be used for the specified method.
            # Cycle through all frames and alignment points:
            for frame_index in range(self.frames.number):
                frame = self.frames.frames_mono_blurred(frame_index)

                # After every "signal_step_size"th frame, send a progress signal to the main GUI.
                if self.progress_signal is not None and frame_index % self.signal_step_size == 1:
                    self.progress_signal.emit("Rank frames at APs",
                                              int((frame_index / self.signal_loop_length) * 100.))

                for ap_index, alignment_point in enumerate(self.alignment_points):
                    # Compute patch bounds within the current frame.
                    y_low = max(0,
                                alignment_point['patch_y_low'] + self.align_frames.dy[frame_index])
                    y_high = min(self.frames.shape[0],
                                 alignment_point['patch_y_high'] + self.align_frames.dy[
                                     frame_index])
                    x_low = max(0,
                                alignment_point['patch_x_low'] + self.align_frames.dx[frame_index])
                    x_high = min(self.frames.shape[1],
                                 alignment_point['patch_x_high'] + self.align_frames.dx[
                                     frame_index])
                    # Compute the frame quality and append it to the list for this alignment point.
                    alignment_point['frame_qualities'].append(
                        method(frame[y_low:y_high, x_low:x_high],
                               self.configuration.alignment_points_rank_pixel_stride))
        else:
            # Sampled-down Laplacians of all blurred frames have been computed in
            # "frames.frames_mono_blurred_laplacian". Cut out boxes around alignment points from
            # those objects, rather than computing new Laplacians. Cycle through all frames and
            # alignment points. Use the blurred monochrome image for ranking.
            for frame_index in range(self.frames.number):
                frame = self.frames.frames_mono_blurred_laplacian(frame_index)

                # After every "signal_step_size"th frame, send a progress signal to the main GUI.
                if self.progress_signal is not None and frame_index % self.signal_step_size == 1:
                    self.progress_signal.emit("Rank frames at APs",
                                              int((frame_index / self.signal_loop_length) * 100.))

                for ap_index, alignment_point in enumerate(self.alignment_points):
                    # Compute patch bounds within the current frame.
                    y_low = int(max(0, alignment_point['patch_y_low'] + self.align_frames.dy[
                        frame_index]) / self.configuration.align_frames_sampling_stride)
                    y_high = int(min(self.frames.shape[0],
                                     alignment_point['patch_y_high'] + self.align_frames.dy[
                                         frame_index]) / self.configuration.align_frames_sampling_stride)
                    x_low = int(max(0, alignment_point['patch_x_low'] + self.align_frames.dx[
                        frame_index]) / self.configuration.align_frames_sampling_stride)
                    x_high = int(min(self.frames.shape[1],
                                     alignment_point['patch_x_high'] + self.align_frames.dx[
                                         frame_index]) / self.configuration.align_frames_sampling_stride)
                    # Compute the frame quality and append it to the list for this alignment point.
                    alignment_point['frame_qualities'].append(meanStdDev(frame[y_low:y_high, x_low:x_high])[1][0][0])

        if self.progress_signal is not None:
            self.progress_signal.emit("Rank frames at APs", 100)

        # Initialize the alignment point lists for all frames.
        self.frames.reset_alignment_point_lists()
        # For each alignment point sort the computed quality ranks in descending order.
        for alignment_point_index, alignment_point in enumerate(self.alignment_points):
            # Truncate the list to the number of frames to be stacked for each alignmeent point.
            alignment_point['best_frame_indices'] = sorted(range(len(alignment_point['frame_qualities'])),
                                                    key=alignment_point['frame_qualities'].__getitem__, reverse=True)[:self.stack_size]
            # Add this alignment point to the AP lists of those frames where the AP is to be used.
            for frame_index in alignment_point['best_frame_indices']:
                self.frames.used_alignment_points[frame_index].append(alignment_point_index)

    def prepare_for_debugging(self, update_image_window_signal):
        """
        The effect of de-warping is to be visualized in the GUI. To this end, method
         "compute_shift_alignment_point" for each frame sends an image to the GUI. The image
         visualizes the effect of global frame stabilization and local de-warping at the given AP.

        :param update_image_window_signal:
        :return:
        """

        self.debug_AP = True
        self.update_image_window_signal = update_image_window_signal
        self.scale_factor = 3
        self.border = 2
        self.image_delay = 3.

    def compute_shift_alignment_point(self, frame_mono_blurred, frame_index, alignment_point_index,
                                      de_warp=True):
        """
        Compute the [y, x] pixel shift vector at a given alignment point relative to the mean frame.
        Four different methods can be used to compute the shift values:
        - a subpixel algorithm from "skimage.feature"
        - a phase correlation algorithm (miscellaneous.translation)
        - a local search algorithm (spiralling outwards), see method "search_local_match",
          optionally with subpixel accuracy.
        - a local search algorithm, based on steepest descent, see method
          "search_local_match_gradient". This method is faster than the previous one, but it has no
          subpixel option.

        Be careful with the sign of the local shift values. For the first two methods, a positive
        value means that the current frame has to be shifted in the positive coordinate direction
        in order to make objects in the frame match with their counterparts in the reference frame.
        In other words: If the shift is positive, an object in the current frame is at lower pixel
        coordinates as compared to the reference. This is very counter-intuitive, but to make the
        three methods consistent, the same approach was followed in the implementation of the third
        method "search_local_match", contained in this module. There, a pixel box around an
        alignment point in the current frame is moved until the content of the box matches with the
        corresponding box in the reference frame. If at this point the box is shifted towards a
        higher coordinate value, this value is returned with a negative sign as the local shift.

        :param frame_mono_blurred: Gaussian-blurred version of the frame with index "frame_index"
        :param frame_index: Index of the selected frame in the list of frames
        :param alignment_point_index: Index of the selected alignment point
        :param de_warp: If True, include local warp shift computation. If False, only apply
                        global frame shift.
        :return: ([dy, dx], deviation) with:
                 [dy, dx]: Local shift vector.
                 deviation: remaining difference of patch relative to reference frame.
        """

        alignment_point = self.alignment_points[alignment_point_index]
        y_low = alignment_point['box_y_low']
        y_high = alignment_point['box_y_high']
        x_low = alignment_point['box_x_low']
        x_high = alignment_point['box_x_high']
        reference_box = alignment_point['reference_box']

        # Initialize the remaining deviation to an impossibly large value.
        deviation = 1.e30

        # The offsets dy and dx are caused by two effects: First, the mean frame is smaller
        # than the original frames. It only contains their intersection. And second, because the
        # given frame is globally shifted as compared to the mean frame.
        dy_global = self.align_frames.dy[frame_index]
        dx_global = self.align_frames.dx[frame_index]

        if de_warp:
            # Use subpixel registration from skimage.feature, with accuracy 1/10 pixels.
            if self.configuration.alignment_points_method == 'Subpixel':
                # Cut out the alignment box from the given frame. Take into account the offsets
                # explained above.
                box_in_frame = frame_mono_blurred[y_low + dy_global:y_high + dy_global,
                               x_low + dx_global:x_high + dx_global]
                shift_pixel, error, diffphase = register_translation(
                    reference_box, box_in_frame, 10, space='real')

            # Use a simple phase shift computation (contained in module "miscellaneous").
            elif self.configuration.alignment_points_method == 'CrossCorrelation':
                # Cut out the alignment box from the given frame. Take into account the offsets
                # explained above.
                box_in_frame = frame_mono_blurred[y_low + dy_global:y_high + dy_global,
                                                  x_low + dx_global:x_high + dx_global]
                shift_pixel = Miscellaneous.translation(reference_box,
                                                        box_in_frame, box_in_frame.shape)

            # Use a local search (see method "search_local_match" below.
            elif self.configuration.alignment_points_method == 'RadialSearch':
                shift_pixel, dev_r = Miscellaneous.search_local_match(reference_box,
                    frame_mono_blurred, y_low + dy_global, y_high + dy_global, x_low + dx_global, x_high + dx_global,
                    self.configuration.alignment_points_search_width,
                    self.configuration.alignment_points_sampling_stride,
                    sub_pixel=self.configuration.alignment_points_local_search_subpixel)

            # Use the steepest descent search method.
            elif self.configuration.alignment_points_method == 'SteepestDescent':
                shift_pixel, dev_r = Miscellaneous.search_local_match_gradient(reference_box,
                    frame_mono_blurred, y_low + dy_global, y_high + dy_global, x_low + dx_global, x_high + dx_global,
                    self.configuration.alignment_points_search_width,
                    self.configuration.alignment_points_sampling_stride, self.dev_table)
                deviation = dev_r[-1]

            # Use the multi-level steepest descent search method.
            elif self.configuration.alignment_points_method == 'MultiLevel':
                try:
                    shift_pixel_levels, deviation = Miscellaneous.search_local_match_multilevel(alignment_point,
                        frame_mono_blurred, dy_global, dx_global,
                        self.configuration.alignment_points_number_levels,
                        self.configuration.alignment_points_noise_levels,
                        self.configuration.alignment_points_iterations,
                        self.configuration.alignment_points_sampling_stride)
                    # The full shift is contained in the level 0 entry.
                    shift_pixel = shift_pixel_levels[0]
                except:
                    # Close to the boundary the multi-level method can fail. In this case switch
                    # over to steepest descent.
                    shift_pixel, dev_r = Miscellaneous.search_local_match_gradient(reference_box,
                        frame_mono_blurred, y_low + dy_global, y_high + dy_global, x_low + dx_global,
                        x_high + dx_global, self.configuration.alignment_points_search_width,
                        self.configuration.alignment_points_sampling_stride, self.dev_table)
                    deviation = dev_r[-1]

            else:
                raise NotSupportedError("The point shift computation method " +
                                        self.configuration.alignment_points_method +
                                        " is not implemented")

            # In debug mode: visualize shifted patch of the first AP and compare it with the
            # corresponding patch of the reference frame.
            if self.debug_AP and not alignment_point_index:

                font = FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                fontColor = (0, 255, 0)
                lineType = 1

                if self.configuration.alignment_points_method == 'MultiLevel':
                    # Special case 'MultiLevel': The de-warp effect is visualized at all scales.
                    try:
                        partial_images = []
                        dy_warp = 0
                        dx_warp = 0
                        for level in reversed(
                                range(self.configuration.alignment_points_number_levels)):
                            shift_pixel_level = shift_pixel_levels[level]
                            stride = 2 ** level
                            reference_patch = alignment_point['reference_boxes'][level].astype(uint16)
                            reference_patch = resize(reference_patch, None,
                                                     fx=float(self.scale_factor),
                                                     fy=float(self.scale_factor))


                            y_level = alignment_point['y_levels'][level] + dy_global - dy_warp
                            x_level = alignment_point['x_levels'][level] + dx_global - dx_warp
                            half_box_width = alignment_point['half_box_widths'][level]
                            frame_stabilized = GaussianBlur(frame_mono_blurred[
                                    y_level - half_box_width:y_level + half_box_width:stride,
                                    x_level - half_box_width:x_level + half_box_width:stride],
                                        (self.configuration.alignment_points_noise_levels[level],
                                        self.configuration.alignment_points_noise_levels[level]), 0)
                            frame_stabilized = resize(frame_stabilized, None,
                                                      fx=float(self.scale_factor),
                                                      fy=float(self.scale_factor))
                            putText(frame_stabilized,
                                    str(dy_global - dy_warp) + ', ' + str(dx_global - dx_warp),
                                    (5, 25), font, fontScale, fontColor, lineType)

                            dy_warp = shift_pixel_level[0]
                            dx_warp = shift_pixel_level[1]
                            y_level = alignment_point['y_levels'][level] + dy_global - dy_warp
                            x_level = alignment_point['x_levels'][level] + dx_global - dx_warp
                            frame_dewarped = GaussianBlur(frame_mono_blurred[
                                    y_level - half_box_width:y_level + half_box_width:stride,
                                    x_level - half_box_width:x_level + half_box_width:stride],
                                        (self.configuration.alignment_points_noise_levels[level],
                                        self.configuration.alignment_points_noise_levels[level]), 0)
                            frame_dewarped = resize(frame_dewarped, None,
                                                    fx=float(self.scale_factor),
                                                    fy=float(self.scale_factor))
                            putText(frame_dewarped,
                                    str(dy_warp) + ', ' + str(dx_warp),
                                    (5, 25), font, fontScale, fontColor, lineType)
                            putText(frame_dewarped,
                                    "{:.2E}".format(Decimal(str(deviation))),
                                    (5, 45), font, fontScale, fontColor, lineType)

                            partial_images.append(frame_stabilized)
                            partial_images.append(reference_patch)
                            partial_images.append(frame_dewarped)

                        # Compose the three patches into a single image and send it to the
                        # visualization window.
                        composed_image = Miscellaneous.compose_image(partial_images,
                                                                     border=self.border)
                        self.update_image_window_signal.emit(composed_image)
                    except Exception as e:
                        print(str(e))

                else:
                    # For alignment methods other than 'multilevel' the de-warp visualization is
                    # only done at the original image scale.
                    total_shift_y = int(round(dy_global - shift_pixel[0]))
                    total_shift_x = int(round(dx_global - shift_pixel[1]))

                    y_low = alignment_point['patch_y_low']
                    y_high = alignment_point['patch_y_high']
                    x_low = alignment_point['patch_x_low']
                    x_high = alignment_point['patch_x_high']
                    reference_patch = (self.mean_frame[y_low:y_high, x_low:x_high]).astype(uint16)
                    reference_patch = resize(reference_patch, None,
                                             fx=float(self.scale_factor),
                                             fy=float(self.scale_factor))

                    try:
                        # Cut out the globally stabilized and the de-warped patches
                        frame_stabilized = frame_mono_blurred[y_low + dy_global:y_high + dy_global,
                                           x_low + dx_global:x_high + dx_global]
                        frame_stabilized = resize(frame_stabilized, None,
                                                  fx=float(self.scale_factor),
                                                  fy=float(self.scale_factor))
                        putText(frame_stabilized,
                                'stabilized: ' + str(dy_global) + ', ' + str(dx_global),
                                (5, 25), font, fontScale, fontColor, lineType)

                        frame_dewarped = frame_mono_blurred[
                                         y_low + total_shift_y:y_high + total_shift_y,
                                         x_low + total_shift_x:x_high + total_shift_x]
                        frame_dewarped = resize(frame_dewarped, None,
                                                fx=float(self.scale_factor),
                                                fy=float(self.scale_factor))
                        putText(frame_dewarped,
                                'de-warped: ' + str(shift_pixel[0]) + ', ' + str(shift_pixel[1]),
                                (5, 25), font, fontScale, fontColor, lineType)
                        # Compose the three patches into a single image and send it to the
                        # visualization window.
                        composed_image = Miscellaneous.compose_image([frame_stabilized,
                                                                      reference_patch,
                                                                      frame_dewarped],
                                                                     border=self.border)
                        self.update_image_window_signal.emit(composed_image)
                    except Exception as e:
                        print(str(e))

                # Insert a delay to keep the current frame long enough in the visualization
                # window.
                sleep(self.image_delay)

            # Return the computed shift vector and the remaining deviation.
            return shift_pixel, float(deviation)
        else:
            # If no de-warping is computed, just return the zero vector.
            return [0, 0], float(deviation)

    def show_alignment_points(self, image):
        """
        Create an RGB version of a monochrome image and insert red crosses at all alignment
        point locations. Draw green alignment point boxes and white alignment point patches.

        :return: 8-bit RGB image with annotations.
        """

        if len(image.shape) == 3:
            color_image = image.astype(uint8)
        else:
            # Expand the monochrome reference frame to RGB
            color_image = stack((image.astype(uint8),) * 3, -1)

        # For all alignment boxes insert a color-coded cross.
        cross_half_len = 5

        for alignment_point in (self.alignment_points):
            y_center = alignment_point['y']
            x_center = alignment_point['x']
            Miscellaneous.insert_cross(color_image, y_center,
                                       x_center, cross_half_len, 'red')
            box_y_low = max(alignment_point['box_y_low'], 0)
            box_y_high = min(alignment_point['box_y_high'], image.shape[0]) - 1
            box_x_low = max(alignment_point['box_x_low'], 0)
            box_x_high = min(alignment_point['box_x_high'], image.shape[1]) - 1
            for y in arange(box_y_low, box_y_high):
                color_image[y, box_x_low] = [255, 255, 255]
                color_image[y, box_x_high] = [255, 255, 255]
            for x in arange(box_x_low, box_x_high):
                color_image[box_y_low, x] = [255, 255, 255]
                color_image[box_y_high, x] = [255, 255, 255]

            patch_y_low = max(alignment_point['patch_y_low'], 0)
            patch_y_high = min(alignment_point['patch_y_high'], image.shape[0]) - 1
            patch_x_low = max(alignment_point['patch_x_low'], 0)
            patch_x_high = min(alignment_point['patch_x_high'], image.shape[1]) - 1
            for y in arange(patch_y_low, patch_y_high):
                color_image[y, patch_x_low] = [0, int(
                    (255 + color_image[y, patch_x_low][1]) / 2.), 0]
                color_image[y, patch_x_high] = [0, int(
                    (255 + color_image[y, patch_x_high][1]) / 2.), 0]
            for x in arange(patch_x_low, patch_x_high):
                color_image[patch_y_low, x] = [0, int(
                    (255 + color_image[patch_y_low, x][1]) / 2.), 0]
                color_image[patch_y_high, x] = [0, int(
                    (255 + color_image[patch_y_high, x][1]) / 2.), 0]

        return color_image


if __name__ == "__main__":
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

    # Get configuration parameters.
    configuration = Configuration()
    try:
        frames = Frames(configuration, names, type=type)
        print("Number of images read: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Error as e:
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

    if configuration.align_frames_mode == "Surface":
        start = time()
        # Select the local rectangular patch in the image where the L gradient is highest in both x
        # and y direction. The scale factor specifies how much smaller the patch is compared to the
        # whole image frame.
        (y_low_opt, y_high_opt, x_low_opt, x_high_opt) = align_frames.compute_alignment_rect(
            configuration.align_frames_rectangle_scale_factor)
        end = time()
        print('Elapsed time in computing optimal alignment rectangle: {}'.format(end - start))
        print("optimal alignment rectangle, x_low: " + str(x_low_opt) + ", x_high: " + str(
            x_high_opt) + ", y_low: " + str(y_low_opt) + ", y_high: " + str(y_high_opt))
        reference_frame_with_alignment_points = frames.frames_mono(
            align_frames.frame_ranks_max_index).copy()
        reference_frame_with_alignment_points[y_low_opt,
        x_low_opt:x_high_opt] = reference_frame_with_alignment_points[y_high_opt - 1,
                                x_low_opt:x_high_opt] = 255
        reference_frame_with_alignment_points[y_low_opt:y_high_opt,
        x_low_opt] = reference_frame_with_alignment_points[y_low_opt:y_high_opt,
                     x_high_opt - 1] = 255
        # plt.imshow(reference_frame_with_alignment_points, cmap='Greys_r')
        # plt.show()

    # Align all frames globally relative to the frame with the highest score.
    start = time()
    align_frames.align_frames()
    end = time()
    print('Elapsed time in aligning all frames: {}'.format(end - start))
    print("Frame shifts: " + str(align_frames.frame_shifts))
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

    # Create alignment points, and show alignment point boxes and patches.
    alignment_points.create_ap_grid()
    print("Number of alignment points created: " + str(len(alignment_points.alignment_points)) +
          ", number of dropped aps (dim): " + str(
        alignment_points.alignment_points_dropped_dim) +
          ", number of dropped aps (structure): " + str(
        alignment_points.alignment_points_dropped_structure))
    color_image = alignment_points.show_alignment_points(average)

    plt.imshow(color_image)
    plt.show()

    # For each alignment point rank frames by their quality.
    start = time()
    alignment_points.compute_frame_qualities()
    end = time()
    print('Elapsed time in ranking frames for every alignment point: {}'.format(end - start))

    y_low = 490
    y_high = 570
    x_low = 880
    x_high = 960
    found_ap_list = alignment_points.find_alignment_points(y_low, y_high, x_low, x_high)
    print("Removing alignment points between bounds " + str(y_low) + " <= y <= " + str(y_high) +
          ", " + str(x_low) + " <= x <= " + str(x_high) + ":")
    for ap in found_ap_list:
        print("y: " + str(ap['y']) + ", x: " + str(ap['x']))
    alignment_points.remove_alignment_points(found_ap_list)

    y_new = 530
    x_new = 920
    half_box_width_new = 40
    half_patch_width_new = 50
    num_pixels_y = average.shape[0]
    num_pixels_x = average.shape[1]
    alignment_points.alignment_points.append(
        alignment_points.new_alignment_point(y_new, x_new, False, False, False, False))
    print("Added alignment point at y: " + str(y_new) + ", x: " + str(x_new) + ", box size: "
          + str(2 * half_box_width_new) + ", patch size: " + str(2 * half_patch_width_new))

    # Show updated alignment point boxes and patches.
    print("Number of alignment points created: " + str(len(alignment_points.alignment_points)) +
          ", number of dropped aps (dim): " + str(alignment_points.alignment_points_dropped_dim) +
          ", number of dropped aps (structure): " + str(
        alignment_points.alignment_points_dropped_structure))
    color_image = alignment_points.show_alignment_points(average)

    plt.imshow(color_image)
    plt.show()
