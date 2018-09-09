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
    def __init__(self, names, type='video'):
        if type == 'image':
            self.frames = [misc.imread(path) for path in names]
            self.number = len(names)
            self.shape = self.frames[0].shape
            for image in self.frames:
                if image.shape != self.shape:
                    raise ShapeError("Images have different size")
                elif len(self.shape) != len(image.shape):
                    raise ShapeError("Mixing grayscale and color images not supported")
        elif type == 'video':
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

        if len(self.shape) == 2:
            self.color = False
        elif len(self.shape) == 3:
            self.color = True
        else:
            raise ShapeError("Image shape not supported")

        self.frames_mono = None

    def extract_channel(self, index, color):
        if not self.color:
            raise ShapeError("Cannot extract green channel from monochrome image")
        colors = ['red', 'green', 'blue']
        if not color in colors:
            raise ArgumentError("Invalid color selected for channel extraction")
        return self.frames[index][:, :, colors.index(color)]

    def add_monochrome(self, color):
        if self.color:
            colors = ['red', 'green', 'blue']
            if not color in colors:
                raise ArgumentError("Invalid color selected for channel extraction")
            self.frames_mono = [frame[:, :, colors.index(color)] for frame in self.frames]
        else:
            self.frames_mono = self.frames

    def shift_frame_with_wraparound(self, index, shift_x, shift_y):
        pil_image = Image.fromarray(self.frames[index])
        im2_offset = ImageChops.offset(pil_image, xoffset=shift_x, yoffset=shift_y)
        self.frames[index] = array(im2_offset)


if __name__ == "__main__":
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

    frames.shift_frame_with_wraparound(0, 100, -200)

    try:
        image_green = frames.extract_channel(0, 'green')
    except ArgumentError as e:
        print("Error: " + e.message)
        exit()
    plt.imshow(image_green, cmap='Greys_r')
    plt.show()

    try:
        frames.add_monochrome('red')
    except ArgumentError as e:
        print("Error: " + e.message)
        exit()

    plt.imshow(frames.frames_mono[1], cmap='Greys_r')
    plt.show()
