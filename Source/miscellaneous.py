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

from numpy import sqrt, average, diff, sum, hypot
from numpy import unravel_index, argmax
from numpy.fft import fft2, ifft2
from scipy.ndimage import sobel


class Miscellaneous(object):
    """
    This class provides static methods for various auxiliary purposes.

    """

    @staticmethod
    def quality_measure(frame):
        """
        Measure the amount of structure in a rectangular frame in both coordinate directions and
        return the minimum value.

        :param frame: 2D image
        :return: Measure for amount of local structure in the image (scalar)
        """

        # Compute for each point the local gradient in both coordinate directions.
        dx = diff(frame)[:, :]
        dy = diff(frame, axis=0)[:, :]

        # Compute the sharpness per coordinate direction as the 2-norm of point values.
        sharpness_x = average(sqrt(dx ** 2))
        sharpness_y = average(sqrt(dy ** 2))

        # Return the sharpness in the direction where it is minimal.
        sharpness = min(sharpness_x, sharpness_y)
        return sharpness


    @staticmethod
    def quality_measure_alternative(frame, black_threshold=40.):
        """
        This is an alternative method for computing the amount of local structure. Here the
        summation only takes into accound points where the luminosity exceeds a certain
        threshold.

        :param frame: 2D image
        :param black_threshold: Threshold for points to be considered
        :return:
        """

        sum_horizontal = sum(sum(abs(frame[:, 2:] - frame[:, :-2]) / (frame[:, 1:-1] + 0.0001) * (
                frame[:, 1:-1] > black_threshold)))
        sum_vertical = sum(sum(abs(frame[2:, :] - frame[:-2, :]) / (frame[1:-1, :] + 0.0001) * (
                frame[1:-1, :] > black_threshold)))
        return min(sum_horizontal, sum_vertical)


    @staticmethod
    def local_contrast_sobel(frame):
        """
        Compute a measure for local contrast in an image using the Sobel method.

        :param frame: 2D image
        :return: Overall sharpness measure (scalar)
        """

        frame_int32 = frame.astype('int32')
        dx = sobel(frame_int32, 0)  # vertical derivative
        dy = sobel(frame_int32, 1)  # horizontal derivative
        mag = hypot(dx, dy)  # magnitude
        sharpness = sum(mag)
        return sharpness


    @staticmethod
    def local_contrast(frame, stride):
        """
        Compute a measure for local contrast in an image based on local gradients. Down-sample the
        image by factor "stride" before measuring the contrast.

        :param frame: 2D image
        :param stride: Factor for down-sampling
        :return: Overall measure for local contrast (scalar)
        """
        frame_strided = frame[::stride, ::stride]

        # Remove a row or column, respectively, to make the dx and dy arrays of the same shape.
        dx = diff(frame_strided)[1:, :]  # remove the first row
        dy = diff(frame_strided, axis=0)[:, 1:]  # remove the first column
        dnorm = sqrt(dx ** 2 + dy ** 2)
        sharpness = average(dnorm)
        return sharpness


    @staticmethod
    def translation(frame_0, frame_1, shape):
        """
        Compute the translation vector of frame_1 relative to frame_0 using phase correlation.

        :param frame_0: Reference frame
        :param frame_1: Frame shifted slightly relative to frame_0
        :param shape: Shape of frames
        :return: [shift in y, shift in x] of frame_1 relative to frame_0. More precisely:
                 frame_1 must be shifted by this amount to register with frame_0.
        """

        # Compute the fast Fourier transforms of both frames.
        f0 = fft2(frame_0)
        f1 = fft2(frame_1)

        # Compute the phase correlation. The resulting image has a bright peak at the pixel location
        # which corresponds to the shift vector.
        ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))

        # Compute the pixel coordinates of the image maximum.
        ty, tx = unravel_index(argmax(ir), shape)

        # Bring the shift values as close as possible to the coordinate origin.
        if ty > shape[0] // 2:
            ty -= shape[0]
        if tx > shape[1] // 2:
            tx -= shape[1]

        return [ty, tx]


    @staticmethod
    def insert_cross(frame, y_center, x_center, cross_half_len, color):
        """
        Insert a colored cross into an image at a given location.

        :param frame: Image into which the cross is to be inserted
        :param y_center: y pixel coordinate of the center of the cross
        :param x_center: x pixel coordinate of the center of the cross
        :param cross_half_len: Extension of the cross from center in all four directions
        :param color: Color, one out of "white", "red", "green" and "blue"
        :return: -
        """

        if color == 'white':
            rgb = [255, 255, 255]
        elif color == 'red':
            rgb = [255, 0, 0]
        elif color == 'green':
            rgb = [0, 255, 0]
        elif color == 'blue':
            rgb = [0, 0, 255]
        else:
            rgb = [255, 255, 255]
        for y in range(y_center - cross_half_len, y_center + cross_half_len + 1):
            frame[y, x_center] = rgb
        for x in range(x_center - cross_half_len, x_center + cross_half_len + 1):
            frame[y_center, x] = rgb


    @staticmethod
    def circle_around(x, y, r):
        """
        Create an iterator which returns y, x pixel coordinates around a given position in an
        image. Successive elements describe a circle around y, x with distance r (in the 1-norm).
        The iterator ends when the full circle is traversed.

        :param x: x coordinate of the center location
        :param y: y coordinate of the center location
        :param r: distance of the iterator items from position (y, x) (1-norm, in pixels)
        :return: -
        """

        # Special case 0: The only iterator element is the center point itself.
        if r == 0:
            yield (x, y)

        # For the general case, set the first element and circle around (y, x) counter-clockwise.
        i, j = x - r, y - r
        while i < x + r:
            i += 1
            yield (i, j)
        while j < y + r:
            j += 1
            yield (i, j)
        while i > x - r:
            i -= 1
            yield (i, j)
        while j > y - r:
            j -= 1
            yield (i, j)
