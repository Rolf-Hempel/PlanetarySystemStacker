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
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from PIL import ImageChops, Image
from numpy import array, uint8
from scipy import misc

from configuration import Configuration
from exceptions import TypeError, ShapeError, ArgumentError


class Frames(object):
    """
        This object stores the image data of all frames.

    """

    def __init__(self, configuration, names, type='video', convert_to_grayscale=False):
        """
        Initialize the Frame object, and read all images. Images can be stored in a video file or
        as single images in a directory.

        :param configuration: Configuration object with parameters
        :param names: In case "video": name of the video file. In case "image": list of names for
                      all images.
        :param type: Either "video" or "image".
        :param convert_to_grayscale: If "True", convert frames to grayscale if they are RGB.
        """

        self.configuration = configuration

        if type == 'image':
            # Use scipy.misc to read in image files. If "convert_to_grayscale" is True, convert
            # pixel values to 32bit floats.
            if convert_to_grayscale:
                self.frames = [misc.imread(path, mode='F') for path in names]
            else:
                self.frames = [misc.imread(path) for path in names]
            self.number = len(names)
            self.shape = self.frames[0].shape

            # Test if all images have the same shape. If not, raise an exception.
            for image in self.frames:
                if image.shape != self.shape:
                    raise ShapeError("Images have different size")
                elif len(self.shape) != len(image.shape):
                    raise ShapeError("Mixing grayscale and color images not supported")

        elif type == 'video':
            # In case "video", use OpenCV to capture frames from video file. Revert the implicit
            # conversion from RGB to BGR in OpenCV input.
            cap = cv2.VideoCapture(names)
            self.number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frames = []
            for frame_index in range(self.number):
                ret, frame = cap.read()
                if ret:
                    if convert_to_grayscale:
                        self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    else:
                        self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    raise IOError("Error in reading video frame")
            cap.release()
            self.shape = self.frames[0].shape
        else:
            raise TypeError("Image type not supported")

        # Monochrome images are stored as 2D arrays, color images as 3D.
        if len(self.shape) == 2:
            self.color = False
        elif len(self.shape) == 3:
            self.color = True
        else:
            raise ShapeError("Image shape not supported")

        # Initialize lists of monochrome frames (with and without Gaussian blur).
        self.frames_mono = None
        self.frames_mono_blurred = None

        # For every frame initialize the list with used alignment points.
        self.used_alignment_points = []
        for frame_index in range(self.number):
            self.used_alignment_points.append([])

    def add_monochrome(self, color):
        """
        Create monochrome versions of all frames. Add a list of monochrome frames
        "self.frames_mono". If the original frames are monochrome, just point the monochrome frame
        list to the original images (no deep copy!). Also, add a blurred version of the frame list
        (using a Gaussian filter) "self.frames_mono_blurred".

        :param color: Either "red" or "green" or "blue"
        :return: -
        """

        if self.color:
            colors = ['red', 'green', 'blue', 'panchromatic']
            if not color in colors:
                raise ArgumentError("Invalid color selected for channel extraction")
            elif color == 'panchromatic':
                self.frames_mono = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in self.frames]
            else:
                self.frames_mono = [frame[:, :, colors.index(color)] for frame in self.frames]
        else:
            self.frames_mono = self.frames

        self.frames_mono_blurred = [cv2.GaussianBlur(frame, (
            self.configuration.frames_gauss_width, self.configuration.frames_gauss_width),
                                                     0) for frame in self.frames_mono]

    def save_image(self, filename, image, color=False):
        """
        Save an image to a file.

        :param filename: Name of the file where the image is to be written
        :param image: ndarray object containing the image data
        :param color: If True, a three channel RGB image is to be saved. Otherwise, monochrome.
        :return: -
        """

        # If a file or directory with the given name already exists, append the word "_file".
        if Path(filename).is_dir():
            while True:
                filename += '_file'
                if not Path(filename).exists():
                    break
            filename += '.jpg'
        # If it is a file, try to append "_copy.tiff" to its basename. If it still exists, repeat.
        elif Path(filename).is_file():
            suffix = Path(filename).suffix
            while True:
                p = Path(filename)
                filename = Path.joinpath(p.parents[0], p.stem + '_copy' + suffix)
                if not Path(filename).exists():
                    break
        else:
            # If the file name is new and has no suffix, add ".tiff".
            suffix = Path(filename).suffix
            if not suffix:
                filename += '.tiff'

        # Write the image to the file. Before writing, convert the internal RGB representation into
        # the BGR representation assumed by OpenCV.
        if color:
            cv2.imwrite(str(filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(filename), image)


if __name__ == "__main__":

    # Images can either be extracted from a video file or a batch of single photographs. Select
    # the example for the test run.
    type = 'video'
    if type == 'image':
        names = glob.glob('Images/2012_*.tif')
    else:
        names = 'Videos/short_video.avi'

    # Get configuration parameters.
    configuration = Configuration()

    try:
        frames = Frames(configuration, names, type=type, convert_to_grayscale=True)
        print("Number of images read: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Exception as e:
        print("Error: " + e.message)
        exit()

    # Create monochrome versions of all frames. If the original frames are monochrome, just point
    # the monochrome frame list to the original images (no deep copy!).
    try:
        frames.add_monochrome('red')
    except ArgumentError as e:
        print("Error: " + e.message)
        exit()

    plt.imshow(frames.frames_mono[0], cmap='Greys_r')
    plt.show()
