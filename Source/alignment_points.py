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
from numpy import arange, amax, stack, amin, hypot, zeros, full, float32, uint8, uint16, array, \
    matmul
from numpy.linalg import solve
from skimage.feature import register_translation

from align_frames import AlignFrames
from configuration import Configuration
from exceptions import WrongOrderingError, NotSupportedError, DivideByZeroError
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
        self.alignment_points = None
        self.alignment_points_dropped = None

    def ap_locations(self, num_pixels, half_box_width, search_width, step_size, even):
        """
        Compute optimal alignment patch coordinates in one coordinate direction. Place boundary
        neighbors as close as possible to the boundary.

        :param num_pixels: Number of pixels in the given coordinate direction
        :param half_box_width: Half-width of the alignment boxes
        :param search_width: Maximum search width for alignment point matching
        :param step_size: Distance of alignment boxes
        :param even: If True, compute locations for even row indices. Otherwise, for odd ones.
        :return: List of alignment point coordinates in the given direction
        """

        # The number of interior alignment boxes in general is not an integer. Round to the next
        # higher number.
        num_interior_odd = int(
            ceil(float(num_pixels - 2* (half_box_width + search_width)) / float(step_size)))
        # Because alignment points are arranged in a staggered grid, in even rows there is one point
        # more.
        num_interior_even = num_interior_odd + 1

        # The precise distance between alignment points will differ slightly from the specified
        # step_size. Compute the exact distance. Integer locations will be rounded later.
        distance_corrected = float(num_pixels - 2* half_box_width - 2 * search_width) / float(num_interior_odd)

        # Compute the AP locations, separately for even and odd rows.
        if even:
            locations = []
            for i in range(num_interior_even):
                locations.append(int(search_width+half_box_width+i*distance_corrected))
        else:
            locations = []
            for i in range(num_interior_odd):
                locations.append(int(search_width + half_box_width + 0.5*distance_corrected+ i * distance_corrected))
        return locations

    def create_ap_grid(self, mean_frame):
        """
        Create a 2D staggered grid of alignment points. For each AP compute its center coordinates,
        and the coordinate limits of its alignment box and alignment patch. Only alignment points
        which satisfy the conditions on brightness, contrast and structure are eventually added to
        the list.

        :param mean_frame: Mean frame as computed by "align_frames.average_frame".

        :return: List of alignment points.
        """

        # Size of the mean frame in y and x directions
        num_pixels_y = mean_frame.shape[0]
        num_pixels_x = mean_frame.shape[1]
        # The alignment box is the object on which the displacement computation is performed.
        half_box_width = self.configuration.alignment_points_half_box_width
        # The alignment patch is the area which is stacked after a rigid displacement.
        half_patch_width = self.configuration.alignment_points_half_patch_width
        # Maximum displacement searched for in the alignment process.
        search_width = self.configuration.alignment_points_search_width
        # Number of pixels in one coordinate direction between alignment points
        step_size = self.configuration.alignment_points_step_size
        # Minimum structure value for an alignment point (between 0. and 1.)
        structure_threshold = self.configuration.alignment_points_structure_threshold
        # The brightest pixel must be brighter than this value (0 < value <256)
        brightness_threshold = self.configuration.alignment_points_brightness_threshold
        # The difference between the brightest and darkest pixel values must be larger than this
        # value (0 < value < 256)
        contrast_threshold = self.configuration.alignment_points_contrast_threshold

        # Compute y and x coordinate locations of alignemnt points. Note that the grid is staggered.
        ap_locations_y = self.ap_locations(num_pixels_y, half_box_width, search_width, step_size,
                                           True)
        ap_locations_x_even = self.ap_locations(num_pixels_x, half_box_width, search_width,
                                                step_size,
                                                True)
        ap_locations_x_odd = self.ap_locations(num_pixels_x, half_box_width, search_width,
                                               step_size,
                                               False)

        self.alignment_points = []
        self.alignment_points_dropped = []

        # Create alignment point rows, start with an even one.
        even = True
        for y in ap_locations_y:
            # Create x coordinate, depending on the y row being even or odd (staggered grid).
            if even:
                ap_locations_x = ap_locations_x_even
            else:
                ap_locations_x = ap_locations_x_odd

            # For each location create an alignment point.
            for x in ap_locations_x:
                alignment_point = {}
                alignment_point['y'] = y
                alignment_point['x'] = x
                alignment_point['box_y_low'] = y - half_box_width
                alignment_point['box_y_high'] = y + half_box_width
                alignment_point['box_x_low'] = x - half_box_width
                alignment_point['box_x_high'] = x + half_box_width
                alignment_point['patch_y_low'] = y - half_patch_width
                alignment_point['patch_y_high'] = y + half_patch_width
                alignment_point['patch_x_low'] = x - half_patch_width
                alignment_point['patch_x_high'] = x + half_patch_width

                # Compute structure and brightness information for the alignment box.
                box = mean_frame[alignment_point['box_y_low']:alignment_point['box_y_high'],
                      alignment_point['box_x_low']:alignment_point['box_x_high']]
                alignment_point['reference_box'] = box
                max_brightness = amax(box)
                min_brightness = amin(box)
                # If the alignment box satisfies the brightness conditions, add the AP to the list.
                if max_brightness > brightness_threshold and max_brightness - \
                        min_brightness > contrast_threshold:
                    alignment_point['structure'] = Miscellaneous.quality_measure(box)
                    self.alignment_points.append(alignment_point)
                else:
                    # If a point does not satisfy the conditions, add it to the dropped list.
                    self.alignment_points_dropped.append(alignment_point)

            # Switch between even and odd rows.
            even = not even

        # Normalize the structure information for all alignment point boxes by dividing by the
        # maximum value.
        structure_max = max(
            alignment_point['structure'] for alignment_point in self.alignment_points)
        for alignment_point in self.alignment_points:
            alignment_point['structure'] /= structure_max
            # Remove alignment points with too little structure.
            if alignment_point['structure'] < structure_threshold:
                self.alignment_points.remove(alignment_point)
                self.alignment_points_dropped.append(alignment_point)

    def compute_shift_alignment_point(self, frame_index, alignment_point_index):
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

        :param frame_index: Index of the selected frame in the list of frames
        :param alignment_point_index: Index of the selected alignment point
        :return: Local shift vector [dy, dx]
        """

        alignment_point = self.alignment_points[alignment_point_index]
        y_low = alignment_point['box_y_low']
        y_high = alignment_point['box_y_high']
        x_low = alignment_point['box_x_low']
        x_high = alignment_point['box_x_high']
        reference_box = alignment_point['reference_box']

        # The offsets dy and dx are caused by two effects: First, the mean frame is smaller
        # than the original frames. It only contains their intersection. And second, because the
        # given frame is globally shifted as compared to the mean frame.
        dy = self.align_frames.intersection_shape[0][0] - \
             self.align_frames.frame_shifts[frame_index][0]
        dx = self.align_frames.intersection_shape[1][0] - \
             self.align_frames.frame_shifts[frame_index][1]

        if self.configuration.alignment_points_dewarp:
            # Use subpixel registration from skimage.feature, with accuracy 1/10 pixels.
            if self.configuration.alignment_point_method == 'Subpixel':
                # Cut out the alignment box from the given frame. Take into account the offsets
                # explained above.
                box_in_frame = self.frames.frames_mono_blurred[frame_index][y_low + dy:y_high + dy,
                               x_low + dx:x_high + dx]
                shift_pixel, error, diffphase = register_translation(
                    reference_box, box_in_frame, 10, space='real')

            # Use a simple phase shift computation (contained in module "miscellaneous").
            elif self.configuration.alignment_point_method == 'CrossCorrelation':
                # Cut out the alignment box from the given frame. Take into account the offsets
                # explained above.
                box_in_frame = self.frames.frames_mono_blurred[frame_index][y_low + dy:y_high + dy,
                               x_low + dx:x_high + dx]
                shift_pixel = Miscellaneous.translation(reference_box,
                                                        box_in_frame, box_in_frame.shape)

            # Use a local search (see method "search_local_match" below.
            elif self.configuration.alignment_point_method == 'RadialSearch':
                shift_pixel, dev_r = Miscellaneous.search_local_match(
                    reference_box, self.frames.frames_mono_blurred[frame_index],
                    y_low + dy, y_high + dy, x_low + dx, x_high + dx,
                    self.configuration.alignment_points_search_width,
                    self.configuration.alignment_points_sampling_stride,
                    sub_pixel=self.configuration.alignment_points_local_search_subpixel)

            # Use the steepest descent search method.
            elif self.configuration.alignment_point_method == 'SteepestDescent':
                shift_pixel, dev_r = Miscellaneous.search_local_match_gradient(
                    reference_box, self.frames.frames_mono_blurred[frame_index],
                    y_low + dy, y_high + dy, x_low + dx, x_high + dx,
                    self.configuration.alignment_points_search_width,
                    self.configuration.alignment_points_sampling_stride)
            else:
                raise NotSupportedError("The point shift computation method " +
                                        self.configuration.alignment_point_method +
                                        " is not implemented")

            # Return the computed shift vector.
            return shift_pixel
        else:
            # If no de-warping is computed, just return the zero vector.
            return [0, 0]

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

        # For all alignment box insert a color-coded cross.
        cross_half_len = 5

        for alignment_point in (self.alignment_points):
            y_center = alignment_point['y']
            x_center = alignment_point['x']
            Miscellaneous.insert_cross(color_image, y_center,
                                       x_center, cross_half_len, 'red')
            for y in arange(alignment_point['box_y_low'], alignment_point['box_y_high']):
                color_image[y, alignment_point['box_x_low']] = [255, 255, 255]
                color_image[y, alignment_point['box_x_high']] = [255, 255, 255]
            for x in arange(alignment_point['box_x_low'], alignment_point['box_x_high']):
                color_image[alignment_point['box_y_low'], x] = [255, 255, 255]
                color_image[alignment_point['box_y_high'], x] = [255, 255, 255]

            patch_y_low = max(alignment_point['patch_y_low'], 0)
            patch_y_high = min(alignment_point['patch_y_high'], image.shape[0]) -1
            patch_x_low = max(alignment_point['patch_x_low'], 0)
            patch_x_high = min(alignment_point['patch_x_high'], image.shape[1]) - 1
            for y in arange(patch_y_low, patch_y_high):
                color_image[y, patch_x_low] = [0, int(
                    (255+color_image[y, patch_x_low][1])/2.), 0]
                color_image[y, patch_x_high] = [0, int(
                    (255 + color_image[y, patch_x_high][1]) / 2.), 0]
            for x in arange(patch_x_low, patch_x_high):
                color_image[patch_y_low, x] = [0, int(
                    (255+color_image[patch_y_low, x][1])/2.), 0]
                color_image[patch_y_high, x] = [0, int(
                    (255 + color_image[patch_y_high, x][1]) / 2.), 0]

        return color_image

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
        configuration.align_frames_rectangle_scale_factor)
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
    alignment_points.create_ap_grid(average)
    print("Number of alignment points created: " + str(len(alignment_points.alignment_points)) +
          ", number of dropped aps: " + str(len(alignment_points.alignment_points_dropped)))
    color_image = alignment_points.show_alignment_points(average)

    plt.imshow(color_image)
    plt.show()