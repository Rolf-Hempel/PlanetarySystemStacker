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

import matplotlib.pyplot as plt
from scipy import ndimage
from numpy import empty, mean, arange, float32

from configuration import Configuration
from exceptions import WrongOrderingError, NotSupportedError, InternalError, ArgumentError
from frames import Frames
from miscellaneous import Miscellaneous
from rank_frames import RankFrames


class AlignFrames(object):
    """
        Based on a list of frames and a (parallel) list of frame quality values, an averaged
        reference frame is created and all frames are aligned with this frame. The alignment is
        performed on a small rectangular area where structure is optimal in both x and y directions.

    """

    def __init__(self, frames, rank_frames, configuration):
        """
        Initialize the AlignFrames object with info from the objects "frames" and "rank_frames".

        :param frames: Frames object with all video frames
        :param rank_frames: RankFrames object with global quality ranks (between 0. and 1.,
                            1. being optimal) for all frames
        :param configuration: Configuration object with parameters
        """

        self.frames = frames.frames
        self.frames_mono = frames.frames_mono
        self.frames_mono_blurred = frames.frames_mono_blurred
        self.number = frames.number
        self.shape = frames.shape
        self.frame_shifts = None
        self.intersection_shape = None
        self.intersection_shape_original = None
        self.mean_frame = None
        self.mean_frame_original = None
        self.configuration = configuration
        self.quality_sorted_indices = rank_frames.quality_sorted_indices
        self.frame_ranks_max_index = rank_frames.frame_ranks_max_index
        self.x_low_opt = self.x_high_opt = self.y_low_opt = self.y_high_opt = None
        self.dev_r_list = None
        self.failed_index_list = None
        self.dy = self.dx = None

    def select_alignment_rect(self, scale_factor):
        """
        Using the frame with the highest rank (sharpest image), select the rectangular patch
        where structure is best in both x and y directions. The size of the patch is the size of the
        frame divided by "scale_factor" in both coordinate directions.

        :param scale_factor: Ratio of the size of the frame and the alignment patch in both
                             coordinate directions
        :return: A four tuple (y_low, y_high, x_low, x_high) with pixel coordinates of the
                 optimal patch
        """

        dim_y, dim_x = self.shape[0:2]

        # Compute the extensions of the alignment rectangle in y and x directions. Take into account
        # a border and the search radius for frame shifts.
        border_width = self.configuration.align_frames_border_width + \
                       self.configuration.align_frames_search_width
        rect_y = int((self.shape[0] - 2 * border_width) / scale_factor)
        rect_x = int((self.shape[1] - 2 * border_width) / scale_factor)

        # Initialize the quality measure of the optimal location to an impossible value (<0).
        quality = -1.

        # Compute for all locations in the frame the "quality measure" and find the place with
        # the maximum value.
        for x_low in arange(border_width, dim_x - rect_x - border_width + 1, rect_x):
            x_high = x_low + rect_x
            for y_low in arange(border_width, dim_y - rect_y - border_width + 1, rect_y):
                y_high = y_low + rect_y
                # new_quality = Miscellaneous.local_contrast(
                #     self.frames_mono_blurred[self.frame_ranks_max_index][y_low:y_high,
                #     x_low:x_high], self.configuration.quality_area_pixel_stride)
                new_quality = Miscellaneous.quality_measure_alternative(
                    self.frames_mono_blurred[self.frame_ranks_max_index][y_low:y_high,
                    x_low:x_high],
                    black_threshold=self.configuration.align_frames_rectangle_black_threshold)
                if new_quality > quality:
                    (self.x_low_opt, self.x_high_opt, self.y_low_opt, self.y_high_opt) = (
                        x_low, x_high, y_low, y_high)
                    quality = new_quality
        return (self.y_low_opt, self.y_high_opt, self.x_low_opt, self.x_high_opt)

    def set_alignment_rect(self, y_low_opt, y_high_opt, x_low_opt, x_high_opt):
        """
        As an alternative to "select_alignment_rect", the rectangular patch can be set explicitly

        :param y_low_opt: Lower y pixel coordinate of patch
        :param y_high_opt: Upper y pixel coordinate of patch
        :param x_low_opt: Lower x pixel coordinate of patch
        :param x_high_opt: Upper x pixel coordinate of patch
        :return: -
        """

        self.y_low_opt = y_low_opt
        self.y_high_opt = y_high_opt
        self.x_low_opt = x_low_opt
        self.x_high_opt = x_high_opt

    def align_frames(self):
        """
        Compute the displacement of all frames relative to the sharpest frame using the alignment
        rectangle.

        :return: -
        """

        if self.configuration.align_frames_mode == "Surface":
            # For "Surface" mode the alignment rectangle has to be selected first.
            if self.x_low_opt is None:
                raise WrongOrderingError(
                    "Method 'align_frames' is called before 'select_alignment_rect'")

            # From the sharpest frame cut out the alignment rectangle. The shifts of all other frames
            #  will be computed relativ to this patch.
            self.reference_window = self.frames_mono_blurred[self.frame_ranks_max_index][
                                    self.y_low_opt:self.y_high_opt,
                                    self.x_low_opt:self.x_high_opt]
            self.reference_window_shape = self.reference_window.shape

        elif self.configuration.align_frames_mode == "Planet":
            # For "Planetary" mode compute the center of gravity for the reference image.
            cog_real = ndimage.measurements.center_of_mass(
                self.frames_mono_blurred[self.frame_ranks_max_index])
            cog_reference_y = int(round(cog_real[0]))
            cog_reference_x = int(round(cog_real[1]))

        else:
            raise NotSupportedError(
                "Frame alignment mode " + self.configuration.align_frames_mode +
                " not supported")

        # Initialize a list which for each frame contains the shifts in y and x directions.
        self.frame_shifts = []

        # Initialize lists with info on failed frames.
        self.dev_r_list = []
        self.failed_index_list = []

        # Initialize two variables which keep the shift values of the previous step as the starting
        # point for the next step. This reduces the search radius if frames are drifting.
        dy_min_cum = dx_min_cum = 0

        for idx, frame in enumerate(self.frames_mono_blurred):

            # For the sharpest frame the displacement is 0 because it is used as the reference.
            if idx == self.frame_ranks_max_index:
                self.frame_shifts.append([0, 0])

            # For all other frames: Compute the global shift, using the "blurred" monochrome image.
            else:
                frame = self.frames_mono_blurred[idx]

                if self.configuration.align_frames_mode == "Planet":
                    # In Planetary mode the shift of the "center of gravity" of the image is
                    # computed. This algorithm cannot fail.
                    cog_frame_real = ndimage.measurements.center_of_mass(frame)
                    self.frame_shifts.append([cog_reference_y - int(round(cog_frame_real[0])),
                                              cog_reference_x - int(round(cog_frame_real[1]))])
                    continue

                # In "Surface" mode three alignment algorithms can be chosen from. In each case
                # the result is the shift vector [dy_min, dx_min]. The second and third algorithm
                # do a local search. It can fail if within the search radius no minimum is found.
                # The first algorithm (cross-correlation) can fail as well, but in this case there
                # is no indication that this happened.
                elif self.configuration.align_frames_method == "Translation":
                    # The shift is computed with cross-correlation. Cut out the alignment patch and
                    # compute its translation relative to the reference.
                    frame_window = self.frames_mono_blurred[idx][self.y_low_opt:self.y_high_opt,
                                   self.x_low_opt:self.x_high_opt]
                    self.frame_shifts.append(
                        Miscellaneous.translation(self.reference_window, frame_window,
                                                  self.reference_window_shape))
                    continue

                elif self.configuration.align_frames_method == "RadialSearch":
                    # Spiral out from the shift position of the previous frame and search for the
                    # local optimum.
                    [dy_min, dx_min], dev_r = Miscellaneous.search_local_match(
                        self.reference_window, frame, self.y_low_opt - dy_min_cum,
                                                      self.y_high_opt - dy_min_cum,
                                                      self.x_low_opt - dx_min_cum,
                                                      self.x_high_opt - dx_min_cum,
                        self.configuration.align_frames_search_width,
                        self.configuration.align_frames_sampling_stride, sub_pixel=False)
                elif self.configuration.align_frames_method == "SteepestDescent":
                    # Spiral out from the shift position of the previous frame and search for the
                    # local optimum.
                    [dy_min, dx_min], dev_r = Miscellaneous.search_local_match_gradient(
                        self.reference_window, frame, self.y_low_opt - dy_min_cum,
                                                      self.y_high_opt - dy_min_cum,
                                                      self.x_low_opt - dx_min_cum,
                                                      self.x_high_opt - dx_min_cum,
                        self.configuration.align_frames_search_width,
                        self.configuration.align_frames_sampling_stride)
                else:
                    raise NotSupportedError(
                        "Frame alignment method " + configuration.align_frames_method +
                        " not supported")

                # Update the cumulative shift values to be used as starting point for the
                # next frame.
                dy_min_cum += dy_min
                dx_min_cum += dx_min
                self.frame_shifts.append([dy_min_cum, dx_min_cum])

                # In "Surface" mode shift computation can fail if no minimum is found within
                # the pre-defined search radius.
                if len(dev_r) > 2 and dy_min == 0 and dx_min == 0:
                    self.failed_index_list.append(idx)
                    self.dev_r_list.append(dev_r)
                    continue

                # If the alignment window gets too close to a frame edge, move it away from
                # that edge by half the border width.
                new_reference_window = False
                # Start with the lower y edge.
                if self.y_low_opt - dy_min_cum < \
                        self.configuration.align_frames_search_width + \
                        self.configuration.align_frames_border_width / 2:
                    self.y_low_opt += ceil(self.configuration.align_frames_border_width / 2.)
                    self.y_high_opt += ceil(self.configuration.align_frames_border_width / 2.)
                    new_reference_window = True
                # Now the upper y edge.
                elif self.y_high_opt - dy_min_cum > self.shape[
                    0] - self.configuration.align_frames_search_width - \
                        self.configuration.align_frames_border_width / 2:
                    self.y_low_opt -= ceil(self.configuration.align_frames_border_width / 2.)
                    self.y_high_opt -= ceil(self.configuration.align_frames_border_width / 2.)
                    new_reference_window = True
                # Now the lower x edge.
                if self.x_low_opt - dx_min_cum < \
                        self.configuration.align_frames_search_width + \
                        self.configuration.align_frames_border_width / 2:
                    self.x_low_opt += ceil(self.configuration.align_frames_border_width / 2.)
                    self.x_high_opt += ceil(self.configuration.align_frames_border_width / 2.)
                    new_reference_window = True
                # Now the upper x edge.
                elif self.x_high_opt - dx_min_cum > self.shape[
                    1] - self.configuration.align_frames_search_width - \
                        self.configuration.align_frames_border_width / 2:
                    self.x_low_opt -= ceil(self.configuration.align_frames_border_width / 2.)
                    self.x_high_opt -= ceil(self.configuration.align_frames_border_width / 2.)
                    new_reference_window = True
                # If the window was moved, update the "reference_window".
                if new_reference_window:
                    self.reference_window = self.frames_mono_blurred[
                                                self.frame_ranks_max_index][
                                                self.y_low_opt:self.y_high_opt,
                                                self.x_low_opt:self.x_high_opt]

        # Compute the shape of the area contained in all frames in the form [[y_low, y_high],
        # [x_low, x_high]]
        self.intersection_shape = [[max(b[0] for b in self.frame_shifts),
                                    min(b[0] for b in self.frame_shifts) + self.shape[0]],
                                   [max(b[1] for b in self.frame_shifts),
                                    min(b[1] for b in self.frame_shifts) + self.shape[1]]]


        if len(self.failed_index_list) > 0:
            raise InternalError("No valid shift computed for " + str(len(self.failed_index_list)) +
                                " frames: " + str(self.failed_index_list))

    def average_frame(self, average_frame_number=None, color=False):
        """
        Compute an averaged frame from the best (monochrome) frames.

        :param average_frame_number: Number of best frames to be averaged. If None, the number is
                                     computed from the configuration parameter
                                      "align_frames_average_frame_percent"
        :param color: If True, compute an average of the original (color) images. Otherwise use the
                      monochrome frame versions.
        :return: The averaged frame
        """

        if self.intersection_shape is None:
            raise WrongOrderingError("Method 'average_frames' is called before 'align_frames'")

        # Compute global offsets of current frame relative to intersection frame. Start with
        # Initializing lists which for each frame give the dy and dx displacements between the
        # reference frame and current frame.
        self.dy = []
        self.dx = []
        for idx in range(len(self.frames_mono_blurred)):
            self.dy.append(self.intersection_shape[0][0] - self.frame_shifts[idx][0])
            self.dx.append(self.intersection_shape[1][0] - self.frame_shifts[idx][1])

        # If the number of frames is not specified explicitly, compute it from configuration.
        if average_frame_number is not None:
            self.average_frame_number = average_frame_number
        else:
            self.average_frame_number = max(
                ceil(self.number * self.configuration.align_frames_average_frame_percent / 100.), 1)

        shifts = [self.frame_shifts[i] for i in self.quality_sorted_indices[:self.average_frame_number]]

        # Create an empty numpy buffer. The first dimension is the frame index, the second and
        # third dimenstions are the y and x coordinates. For color frames add a fourth dimension.
        if color:
            frames = [self.frames[i] for i in
                      self.quality_sorted_indices[:self.average_frame_number]]
            buffer = empty([self.average_frame_number,
                 self.intersection_shape[0][1] - self.intersection_shape[0][0],
                 self.intersection_shape[1][1] - self.intersection_shape[1][0], 3], dtype=float32)
            for idx, frame in enumerate(frames):
                buffer[idx, :, :, :] = frame[self.intersection_shape[0][0] - shifts[idx][0]:
                                          self.intersection_shape[0][1] - shifts[idx][0],
                                    self.intersection_shape[1][0] - shifts[idx][1]:
                                    self.intersection_shape[1][1] - shifts[idx][1], :]
        else:
            frames = [self.frames_mono[i] for i in
                      self.quality_sorted_indices[:self.average_frame_number]]
            buffer = empty([self.average_frame_number,
                 self.intersection_shape[0][1] - self.intersection_shape[0][0],
                 self.intersection_shape[1][1] - self.intersection_shape[1][0]], dtype=float32)
            # For each frame, cut out the intersection area and copy it to the buffer.
            for idx, frame in enumerate(frames):
                buffer[idx, :, :] = frame[self.intersection_shape[0][0] - shifts[idx][0]:
                                          self.intersection_shape[0][1] - shifts[idx][0],
                                    self.intersection_shape[1][0] - shifts[idx][1]:
                                    self.intersection_shape[1][1] - shifts[idx][1]]

        # Compute the mean frame by averaging over the first index.
        self.mean_frame = mean(buffer, axis=0)

        return self.mean_frame

    def set_roi(self, y_min, y_max, x_min, x_max):
        """
        Make the stacking region snmaller than the intersection size. Be careful: The pixel
        indices in this method refer to the shape of the intersection of all frames, i.e., the
        shape of the full mean frame. In general, the original frames are somewhat larger.

        :param y_min: Lower y pixel bound
        :param y_max: Upper y pixel bound
        :param x_min: Lower x pixel bound
        :param x_max: Upper x pixel bound
        :return: The new averaged frame, restricted to the ROI
        """

        if self.intersection_shape is None:
            raise WrongOrderingError("Method 'set_roi' is called before 'align_frames'")

        # On the first call, keep a copy of the full mean frame and original intersection shape.
        if not self.mean_frame_original:
            self.mean_frame_original = self.mean_frame.copy()
            self.intersection_shape_original = self.intersection_shape.copy()

        if y_min < 0 or y_max > self.intersection_shape_original[0][1] - \
                self.intersection_shape_original[0][0] or \
                x_min < 0 or x_max > self.intersection_shape_original[1][1] - \
                self.intersection_shape_original[1][0] or \
                y_min >= y_max or x_min >= x_max:
            raise ArgumentError("Invalid ROI index bounds specified")

        # Reduce the intersection shape and mean frame to the ROI.
        self.intersection_shape = [[y_min+self.intersection_shape_original[0][0],
                                    y_max+self.intersection_shape_original[0][0]],
                                   [x_min+self.intersection_shape_original[1][0],
                                    x_max+self.intersection_shape_original[1][0]]]

        # Re-compute global offsets of current frame relative to reference frame.
        self.dy = []
        self.dx = []
        for idx in range(len(self.frames_mono_blurred)):
            self.dy.append(self.intersection_shape[0][0] - self.frame_shifts[idx][0])
            self.dx.append(self.intersection_shape[1][0] - self.frame_shifts[idx][1])

        self.mean_frame = self.mean_frame_original[y_min:y_max, x_min:x_max]

        return self.mean_frame


    def write_stabilized_video(self, name, fps, stabilized=True):
        """
        Write out a stabilized videos. For all frames the part common to all frames is extracted
        and written into a video file.

        :param name: File name of the video output
        :param fps: Frames per second of video
        :param stabilized: if False, switch off image stabilization. Write original frames.
        :return: -
        """

        # Initialize lists of stabilized frames and index strings.
        frames_mono_stabilized = []
        frame_indices = []

        if stabilized:
            # For each frame: cut out the shifted window with the intersection of all frames.
            # Append it to the list, and add its index to the list of index strings.
            for idx, frame_mono in enumerate(self.frames_mono):
                frames_mono_stabilized.append(frame_mono[
                                              self.intersection_shape[0][0] -
                                              self.frame_shifts[idx][0]:
                                              self.intersection_shape[0][1] -
                                              self.frame_shifts[idx][0],
                                              self.intersection_shape[1][0] -
                                              self.frame_shifts[idx][1]:
                                              self.intersection_shape[1][1] -
                                              self.frame_shifts[idx][1]])
                frame_indices.append(str(idx))
            Miscellaneous.write_video(name, frames_mono_stabilized, frame_indices, 5)
        else:
            # Write the original frames (not stabilized) with index number insertions.
            for idx in range(len(self.frames_mono)):
                frame_indices.append(str(idx))
            Miscellaneous.write_video(name, self.frames_mono, frame_indices, 5)


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
        # file = 'short_video'
        file = 'another_short_video'
        # file = 'Moon_Tile-024_043939'
        names = 'Videos/' + file + '.avi'
    print(names)

    # Get configuration parameters.
    configuration = Configuration()
    try:
        # In creating the Frames object the images are read from the specified file(s).
        frames = Frames(configuration, names, type=type)
        print("Number of images read: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Exception as e:
        print("Error: " + str(e))
        exit()

    # Create monochrome versions of all frames. If the original frames are monochrome,
    # just point the monochrome frame list to the original images (no deep copy!).
    try:
        frames.add_monochrome(configuration.frames_mono_channel)
    except ArgumentError as e:
        print("Error: " + e.message)
        exit()

    # Rank the frames by their overall local contrast.
    rank_frames = RankFrames(frames, configuration)
    rank_frames.frame_score()

    # Initialize the frame alignment object.
    align_frames = AlignFrames(frames, rank_frames, configuration)

    if configuration.align_frames_mode == "Surface":
        # Select the local rectangular patch in the image where the L gradient is highest in both x
        # and y direction. The scale factor specifies how much smaller the patch is compared to the
        # whole image frame.
        (y_low_opt, y_high_opt, x_low_opt, x_high_opt) = align_frames.select_alignment_rect(
            configuration.align_frames_rectangle_scale_factor)

        # Alternative: Set the alignment rectangle by hand.
        # (align_frames.x_low_opt, align_frames.x_high_opt, align_frames.y_low_opt,
        #  align_frames.y_high_opt) = (x_low_opt, x_high_opt, y_low_opt, y_high_opt) = (
        # 650, 950, 550, 750)

        print("optimal alignment rectangle, x_low: " + str(x_low_opt) + ", x_high: " + str(
            x_high_opt) + ", y_low: " + str(y_low_opt) + ", y_high: " + str(y_high_opt))
        frame = align_frames.frames_mono_blurred[align_frames.frame_ranks_max_index].copy()
        frame[y_low_opt, x_low_opt:x_high_opt] = frame[y_high_opt - 1, x_low_opt:x_high_opt] = 255
        frame[y_low_opt:y_high_opt, x_low_opt] = frame[y_low_opt:y_high_opt, x_high_opt - 1] = 255
        plt.imshow(frame, cmap='Greys_r')
        plt.show()

    # Align all frames globally relative to the frame with the highest score.
    try:
        align_frames.align_frames()
    except NotSupportedError as e:
        print("Error: " + e.message)
        exit()
    except InternalError as e:
        print("Warning: " + e.message)
        for index, frame_number in enumerate(align_frames.failed_index_list):
            print("Shift computation failed for frame " +
                  str(align_frames.failed_index_list[index]) + ", minima list: " +
                  str(align_frames.dev_r_list[index]))

    print("Frame shifts: " + str(align_frames.frame_shifts))
    print("Intersection: " + str(align_frames.intersection_shape))

    # Compute the reference frame by averaging the best frames.
    average = align_frames.average_frame()
    plt.imshow(average, cmap='Greys_r')
    plt.show()

    # Write video with stabilized frames, annotated with their frame indices.
    stabilized = False
    if stabilized:
        name = 'Videos/stabilized_video_with_frame_numbers.avi'
    else:
        name = 'Videos/not_stabilized_video_with_frame_numbers.avi'
    align_frames.write_stabilized_video(name, 5, stabilized=stabilized)

    print ("Setting ROI and computing new average frame")
    average = align_frames.set_roi(300, 600, 500, 1000)
    plt.imshow(average, cmap='Greys_r')
    plt.show()

