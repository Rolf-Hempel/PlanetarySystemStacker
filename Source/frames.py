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

import cv2
import matplotlib.pyplot as plt
from PIL import ImageChops, Image
from numpy import array
from scipy import misc

from exceptions import TypeError, ShapeError, ArgumentError


class Frames(object):
    """
        This object stores the image data of all frames.

    """

    def __init__(self, names, type='video'):
        """
        Initialize the Frame object, and read all images. Images can be stored in a video file or as single images in
        a directory.

        :param names: In case "video": name of the video file. In case "image": list of names for all images.
        :param type: Either "video" or "image"
        """

        if type == 'image':
            # Use scipy.misc to read in image files.
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
            # In case "video", use OpenCV to capture frames from video file.
            cap = cv2.VideoCapture(names)
            self.number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frames = []
            for frame_index in range(self.number):
                ret, frame = cap.read()
                if ret:
                    self.frames.append(frame)
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

        # Initialize list of monochrome frames.
        self.frames_mono = None

    def extract_channel(self, index, color):
        """
        Extract a color channel from an RGB frame.

        :param index: Frame index
        :param color: Either "red" or "green" or "blue"
        :return: 2D array with the selected color channel of the frame with index "index"
        """

        if not self.color:
            raise ShapeError("Cannot extract green channel from monochrome image")
        colors = ['red', 'green', 'blue']
        if not color in colors:
            raise ArgumentError("Invalid color selected for channel extraction")

        # Collapse the 3D array by the color dimension.
        return self.frames[index][:, :, colors.index(color)]

    def add_monochrome(self, color):
        """
        Same as method "extract_channel", but for all frames. Add a list of monochrome frames "self.frames_mono". If
        the original frames are monochrome, just point the monochrome frame list to the original images (no deep copy!).

        :param color: Either "red" or "green" or "blue"
        :return: -
        """

        if self.color:
            colors = ['red', 'green', 'blue']
            if not color in colors:
                raise ArgumentError("Invalid color selected for channel extraction")
            self.frames_mono = [frame[:, :, colors.index(color)] for frame in self.frames]
        else:
            self.frames_mono = self.frames

    def shift_frame_with_wraparound(self, index, shift_x, shift_y):
        """
        This is an experimental method, not used in the current Code. It shifts a frame in y and x directions with
        wrap-around in both directions. The result is stored at the original location!

        :param index: Frame index
        :param shift_x: Shift in x (pixels)
        :param shift_y: Shift in y (pixels)
        :return: -
        """

        pil_image = Image.fromarray(self.frames[index])
        im2_offset = ImageChops.offset(pil_image, xoffset=shift_x, yoffset=shift_y)
        self.frames[index] = array(im2_offset)


if __name__ == "__main__":

    # Images can either be extracted from a video file or a batch of single photographs. Select the example for
    # the test run.
    type = 'video'
    if type == 'image':
        names = glob.glob('Images/2012_*.tif')
    else:
        names = 'Videos/short_video.avi'
    try:
        frames = Frames(names, type=type)
        print("Number of images read: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Exception as e:
        print("Error: " + e.message)
        exit()

    # Test the "shift with wraparound" method (not used in actual application).
    frames.shift_frame_with_wraparound(0, 100, -200)

    # Extract the green channel of the RGB image.
    try:
        image_green = frames.extract_channel(0, 'green')
    except ArgumentError as e:
        print("Error: " + e.message)
        exit()
    plt.imshow(image_green, cmap='Greys_r')
    plt.show()

    # Create monochrome versions of all frames. If the original frames are monochrome, just point the monochrome frame
    # list to the original images (no deep copy!).
    try:
        frames.add_monochrome('red')
    except ArgumentError as e:
        print("Error: " + e.message)
        exit()

    plt.imshow(frames.frames_mono[1], cmap='Greys_r')
    plt.show()
