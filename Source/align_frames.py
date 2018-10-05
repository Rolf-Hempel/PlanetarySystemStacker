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

import matplotlib.pyplot as plt
from numpy import empty, mean, arange

from configuration import Configuration
from exceptions import WrongOrderingError
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

        self.frames_mono = frames.frames_mono
        self.number = frames.number
        self.shape = frames.shape
        self.frame_shifts = None
        self.intersection_shape = None
        self.mean_frame = None
        self.configuration = configuration
        self.quality_sorted_indices = rank_frames.quality_sorted_indices
        self.frame_ranks_max_index = rank_frames.frame_ranks_max_index
        self.x_low_opt = self.x_high_opt = self.y_low_opt = self.y_high_opt = None

    def select_alignment_rect(self, scale_factor):
        """
        Using the frame with the highest rank (sharpest image), select the rectangular patch
        where structure is best in both x and y directions. The size of the patch is the size of the
        frame divided by "scale_factor" in both coordinate directions.

        :param scale_factor: Ratio of the size of the frame and the alignment patch in both
                             coordinate directions
        :return: A four tuple (x_low, x_high, y_low, y_high) with pixel coordinates of the
                 optimal patch
        """

        dim_y, dim_x = self.shape[0:2]

        # Compute the extensions of the alignment rectangle in y and x directions.
        rect_y = int(self.shape[0] / scale_factor)
        rect_x = int(self.shape[1] / scale_factor)

        # Initialize the quality measure of the optimal location to an impossible value (<0).
        quality = -1.

        # Compute for all locations in the frame the "quality measure" and find the place with
        # the maximum value.
        for x_low in arange(0, dim_x - rect_x + 1, rect_x):
            x_high = x_low + rect_x
            for y_low in arange(0, dim_y - rect_y + 1, rect_y):
                y_high = y_low + rect_y
                new_quality = Miscellaneous.quality_measure(
                    self.frames_mono[self.frame_ranks_max_index][y_low:y_high, x_low:x_high])
                if new_quality > quality:
                    (self.x_low_opt, self.x_high_opt, self.y_low_opt, self.y_high_opt) = (
                    x_low, x_high, y_low, y_high)
                    quality = new_quality
        return (self.x_low_opt, self.x_high_opt, self.y_low_opt, self.y_high_opt)

    def align_frames(self):
        """
        Compute the displacement of all frames relative to the sharpest frame using the alignment
        rectangle.

        :return: -
        """

        if self.x_low_opt is None:
            raise WrongOrderingError(
                "Method 'align_frames' is called before 'select_alignment_rect'")

        # Initialize a list which for each frame contains the shifts in y and x directions.
        self.frame_shifts = []

        # From the sharpest frame cut out the alignment rectangle. The shifts of all other frames
        #  will be computed relativ to this patch.
        self.reference_window = self.frames_mono[self.frame_ranks_max_index][
                                self.y_low_opt:self.y_high_opt, self.x_low_opt:self.x_high_opt]
        self.reference_window_shape = self.reference_window.shape
        for idx, frame in enumerate(self.frames_mono):

            # For the sharpest frame the displacement is 0 because it is used as the reference.
            if idx == self.frame_ranks_max_index:
                self.frame_shifts.append([0, 0])

            # For all other frames: Cut out the alignment patch and compute its translation
            # relative to the reference.
            else:
                frame_window = self.frames_mono[idx][self.y_low_opt:self.y_high_opt,
                               self.x_low_opt:self.x_high_opt]
                self.frame_shifts.append(
                    Miscellaneous.translation(self.reference_window, frame_window,
                                              self.reference_window_shape))

        # Compute the shape of the area contained in all frames in the form [[y_low, y_high],
        # [x_low, x_high]]
        self.intersection_shape = [[max(b[0] for b in self.frame_shifts),
                                    min(b[0] for b in self.frame_shifts) + self.shape[0]],
                                    [max(b[1] for b in self.frame_shifts),
                                    min(b[1] for b in self.frame_shifts) + self.shape[1]]]
        self.intersection_number_pixels = (self.intersection_shape[0][1] -
                                           self.intersection_shape[0][0]) * \
                                          (self.intersection_shape[1][1] -
                                           self.intersection_shape[1][0])

    def average_frame(self, frames, shifts):
        """
        Compute an averaged frame from a list of (monochrome) frames along with their
        corresponding shift values.

        :param frames: List of frames to be averaged
        :param shifts: List of shifts for all frames. Each shift is given as [shift_y, shift_x].
        :return: The averaged frame
        """

        if self.intersection_shape is None:
            raise WrongOrderingError("Method 'average_frames' is called before 'align_frames'")

        # "number_frames" are to be averaged.
        number_frames = len(frames)

        # Create an empty numpy buffer. The first dimension is the frame index, the second and
        # third dimenstions are the y and x coordinates.
        buffer = empty(
            [number_frames, self.intersection_shape[0][1] - self.intersection_shape[0][0],
             self.intersection_shape[1][1] - self.intersection_shape[1][0]])

        # For each frame, cut out the intersection area and copy it to the buffer.
        for idx, frame in enumerate(frames):
            buffer[idx, :, :] = frame[self.intersection_shape[0][0] - shifts[idx][0]:
                                        self.intersection_shape[0][1] - shifts[idx][0],
                                  self.intersection_shape[1][0] - shifts[idx][1]:
                                  self.intersection_shape[1][1] - shifts[idx][1]]
        # Compute the mean frame by averaging over the first index.
        self.mean_frame = mean(buffer, axis=0)
        return self.mean_frame


if __name__ == "__main__":
    # Images can either be extracted from a video file or a batch of single photographs. Select
    # the example for the test run.
    type = 'video'
    if type == 'image':
        names = glob.glob(
            'Images/2012*.tif')  # names = glob.glob('Images/Moon_Tile-031*ap85_8b.tif')  # names
        #  = glob.glob('Images/Example-3*.jpg')
    else:
        names = 'Videos/short_video.avi'
    print(names)

    # Get configuration parameters.
    configuration = Configuration()
    try:
        # In creating the Frames object the images are read from the specified file(s).
        frames = Frames(names, type=type)
        print("Number of images read: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Exception as e:
        print("Error: " + e.message)
        exit()

    # Rank the frames by their overall local contrast.
    rank_frames = RankFrames(frames, configuration)
    rank_frames.frame_score()

    # Initialize the frame alignment object.
    align_frames = AlignFrames(frames, rank_frames, configuration)

    # Select the local rectangular patch in the image where the L gradient is highest in both x
    # and y direction. The scale factor specifies how much smaller the patch is compared to the
    # whole image frame.
    (x_low_opt, x_high_opt, y_low_opt, y_high_opt) = align_frames.select_alignment_rect(
        configuration.alignment_rectangle_scale_factor)

    print("optimal alignment rectangle, x_low: " + str(x_low_opt) + ", x_high: " + str(
        x_high_opt) + ", y_low: " + str(y_low_opt) + ", y_high: " + str(y_high_opt))
    frame = align_frames.frames_mono[align_frames.frame_ranks_max_index].copy()
    frame[y_low_opt, x_low_opt:x_high_opt] = frame[y_high_opt - 1, x_low_opt:x_high_opt] = 255
    frame[y_low_opt:y_high_opt, x_low_opt] = frame[y_low_opt:y_high_opt, x_high_opt - 1] = 255
    plt.imshow(frame, cmap='Greys_r')
    plt.show()

    # Align all frames globally relative to the frame with the highest score.
    align_frames.align_frames()
    print("Frame shifts: " + str(align_frames.frame_shifts))
    print("Intersection: " + str(align_frames.intersection_shape))

    # Compute the reference frame by averaging the best frames.
    average = align_frames.average_frame(align_frames.frames_mono, align_frames.frame_shifts)
    plt.imshow(average, cmap='Greys_r')
    plt.show()
