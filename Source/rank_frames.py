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

from configuration import Configuration
from frames import Frames
from miscellaneous import Miscellaneous
from exceptions import NotSupportedError


class RankFrames(object):
    """
        Rank frames according to their overall sharpness. Experiments with different algorithms
        have been made. The classical "Sobel" algorithm is good but slow. An alternative is
        implemented in method "local_contrast" in module "miscellaneous".

    """

    def __init__(self, frames, configuration):
        """
        Initialize the object and instance variables.

        :param frames: Frames object with all video frames
        :param configuration: Configuration object with parameters
        """

        self.number = frames.number
        self.shape = frames.shape
        self.configuration = configuration

        # The whole quality analysis and shift determination process is performed on a monochrome
        # version of the frames. If the original frames are in RGB, the monochrome channel can be
        # selected via a configuration parameter. Add a list of monochrome images for all frames to
        # the "Frames" object.
        frames.add_monochrome(self.configuration.mono_channel)
        self.frames_mono = frames.frames_mono
        self.frames_mono_blurred = frames.frames_mono_blurred
        self.quality_sorted_indices = None
        self.frame_ranks = []
        self.frame_ranks_max_index = None
        self.frame_ranks_max_value = None

    def frame_score(self):
        """
        Compute the frame quality values and normalize them such that the best value is 1.

        :return: -
        """

        if self.configuration.frame_score_method == "xy gradient":
            method = Miscellaneous.local_contrast
        elif self.configuration.frame_score_method == "Laplace":
            method = Miscellaneous.local_contrast_laplace
        elif self.configuration.frame_score_method == "Sobel":
            method = Miscellaneous.local_contrast_sobel
        else:
            raise NotSupportedError("Ranking method " + self.configuration.frame_score_method +
                                    " not supported")

        # For all frames compute the quality with the selected method.
        for frame in self.frames_mono_blurred:
            self.frame_ranks.append(method(frame, self.configuration.frame_score_pixel_stride))

        # Sort the frame indices in descending order of quality.
        self.quality_sorted_indices = [b[0] for b in sorted(enumerate(self.frame_ranks),
                                                            key=lambda i: i[1], reverse=True)]

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
        names = glob.glob('Images/Mond_*.jpg')
    else:
        # names = 'Videos/short_video.avi'
        names = 'Videos/Moon_Tile-024_043939.avi'
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
    # for rank, index in enumerate(rank_frames.quality_sorted_indices):
    #     frame_quality = rank_frames.frame_ranks[index]
    #     print("Rank: " + str(rank) + ", Frame no. " + str(index) + ", quality: " + str(frame_quality))
    for index, frame_quality in enumerate(rank_frames.frame_ranks):
        rank = rank_frames.quality_sorted_indices.index(index)
        print("Frame no. " + str(index) + ", Rank: " + str(rank) + ", quality: " +
              str(frame_quality))
    print('Elapsed time in ranking frames: {}'.format(end - start))
