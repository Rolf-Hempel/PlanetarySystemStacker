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
from time import time
from numpy import array, full
import matplotlib.pyplot as plt
from cv2 import meanStdDev

from configuration import Configuration
from frames import Frames
from miscellaneous import Miscellaneous
from exceptions import NotSupportedError, Error


class RankFrames(object):
    """
        Rank frames according to their overall sharpness. Experiments with different algorithms
        have been made. The classical "Sobel" algorithm is good but slow. An alternative is
        implemented in method "local_contrast" in module "miscellaneous".

    """

    def __init__(self, frames, configuration, progress_signal=None):
        """
        Initialize the object and instance variables.

        :param frames: Frames object with all video frames
        :param configuration: Configuration object with parameters
        :param progress_signal: Either None (no progress signalling), or a signal with the signature
                                (str, int) with the current activity (str) and the progress in
                                percent (int).
        """

        self.number = frames.number
        self.shape = frames.shape
        self.configuration = configuration
        self.frames = frames
        self.quality_sorted_indices = None
        self.frame_ranks = []
        self.frame_ranks_max_index = None
        self.frame_ranks_max_value = None
        self.progress_signal = progress_signal
        self.signal_step_size = max(int(self.number / 10), 1)

    def frame_score(self):
        """
        Compute the frame quality values and normalize them such that the best value is 1.

        :return: -
        """

        if self.configuration.rank_frames_method == "xy gradient":
            method = Miscellaneous.local_contrast
        elif self.configuration.rank_frames_method == "Laplace":
            method = Miscellaneous.local_contrast_laplace
        elif self.configuration.rank_frames_method == "Sobel":
            method = Miscellaneous.local_contrast_sobel
        else:
            raise NotSupportedError("Ranking method " + self.configuration.rank_frames_method +
                                    " not supported")

        # For all frames compute the quality with the selected method.
        if method != Miscellaneous.local_contrast_laplace:
            for frame_index in range(self.frames.number):
                frame = self.frames.frames_mono_blurred(frame_index)
                if self.progress_signal is not None and frame_index % self.signal_step_size == 1:
                    self.progress_signal.emit("Rank all frames",
                                              int((frame_index / self.number) * 100.))
                self.frame_ranks.append(method(frame, self.configuration.rank_frames_pixel_stride))
        else:
            for frame_index in range(self.frames.number):
                frame = self.frames.frames_mono_blurred_laplacian(frame_index)
                # self.frame_ranks.append(mean((frame - frame.mean())**2))
                if self.progress_signal is not None and frame_index % self.signal_step_size == 1:
                    self.progress_signal.emit("Rank all frames",
                                              int((frame_index / self.number) * 100.))
                self.frame_ranks.append(meanStdDev(frame)[1][0][0])

        if self.progress_signal is not None:
            self.progress_signal.emit("Rank all frames", 100)
        # Sort the frame indices in descending order of quality.
        self.quality_sorted_indices = sorted(range(len(self.frame_ranks)), key=self.frame_ranks.__getitem__, reverse=True)

        # Set the index of the best frame, and normalize all quality values.
        self.frame_ranks_max_index = self.quality_sorted_indices[0]
        self.frame_ranks_max_value = self.frame_ranks[self.frame_ranks_max_index]
        self.frame_ranks /= self.frame_ranks_max_value


if __name__ == "__main__":

    # Images can either be extracted from a video file or a batch of single photographs. Select
    # the example for the test run.
    type = 'video'
    if type == 'image':
        # names = glob.glob('Images/2012*.tif')
        # names = glob.glob('Images/Moon_Tile-031*ap85_8b.tif')
        # names = glob.glob('Images/Example-3*.jpg')
        names = glob('Images/Mond_*.jpg')
    else:
        names = 'Videos/short_video.avi'
        # names = 'Videos/Moon_Tile-024_043939.avi'
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
    start = time()
    rank_frames = RankFrames(frames, configuration)
    rank_frames.frame_score()
    end = time()
    print('Elapsed time in ranking all frames: {}'.format(end - start))

    # for rank, index in enumerate(rank_frames.quality_sorted_indices):
    #     frame_quality = rank_frames.frame_ranks[index]
    #     print("Rank: " + str(rank) + ", Frame no. " + str(index) + ", quality: " + str(frame_quality))
    for index, frame_quality in enumerate(rank_frames.frame_ranks):
        rank = rank_frames.quality_sorted_indices.index(index)
        print("Frame no. " + str(index) + ", Rank: " + str(rank) + ", quality: " +
              str(frame_quality))
    print('Elapsed time in ranking frames: {}'.format(end - start))

    print("")
    num_frames = len(rank_frames.frame_ranks)
    frame_percent = 10
    num_frames_stacked = max(1, round(num_frames*frame_percent/100.))
    print("Percent of frames to be stacked: ", str(frame_percent), ", numnber: "
           + str(num_frames_stacked))
    quality_cutoff = rank_frames.frame_ranks[rank_frames.quality_sorted_indices[num_frames_stacked]]
    print("Quality cutoff: ", str(quality_cutoff))

    # Plot the frame qualities in chronological order.
    ax1 = plt.subplot(211)

    x = array(rank_frames.frame_ranks)
    plt.ylabel('Frame number')
    plt.gca().invert_yaxis()
    y = array(range(num_frames))
    x_cutoff = full((num_frames,), quality_cutoff)
    plt.xlabel('Quality')
    line1, = plt.plot(x, y, lw=1)
    line2, = plt.plot(x_cutoff, y, lw=1)
    index = 37
    plt.scatter(x[index], y[index], s=20)
    plt.grid(True)

    # Plot the frame qualities ordered by value.
    ax2 = plt.subplot(212)

    x = array([rank_frames.frame_ranks[i] for i in rank_frames.quality_sorted_indices])
    plt.ylabel('Frame rank')
    plt.gca().invert_yaxis()
    y = array(range(num_frames))
    y_cutoff = full((num_frames,), num_frames_stacked)
    plt.xlabel('Quality')
    line3, = plt.plot(x, y, lw=1)
    line4, = plt.plot(x, y_cutoff, lw=1)
    index = 37
    plt.scatter(x[index], y[index], s=20)
    plt.grid(True)

    plt.show()
