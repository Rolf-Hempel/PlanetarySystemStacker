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


def quality_measure(frame):
    dx = diff(frame)[1:, :]  # remove the first row
    dy = diff(frame, axis=0)[:, 1:]  # remove the first column
    sharpness_x = average(sqrt(dx ** 2))
    sharpness_y = average(sqrt(dy ** 2))
    sharpness = min(sharpness_x, sharpness_y)
    return sharpness


def quality_measure_alternative(frame, black_threshold=40.):
    sum_horizontal = sum(sum(abs(frame[:, 2:] - frame[:, :-2]) / (frame[:, 1:-1] + 0.0001) * (
            frame[:, 1:-1] > black_threshold)))
    sum_vertical = sum(sum(abs(frame[2:, :] - frame[:-2, :]) / (frame[1:-1, :] + 0.0001) * (
            frame[1:-1, :] > black_threshold)))
    return min(sum_horizontal, sum_vertical)


def quality_measure_sobel(frame):
    frame_int32 = frame.astype('int32')
    dx = sobel(frame_int32, 0)  # vertical derivative
    dy = sobel(frame_int32, 1)  # horizontal derivative
    mag = hypot(dx, dy)  # magnitude
    sharpness = sum(mag)
    return sharpness


def local_contrast(frame, stride):
    frame_strided = frame[::stride, ::stride]
    dx = diff(frame_strided)[1:, :]  # remove the first row
    dy = diff(frame_strided, axis=0)[:, 1:]  # remove the first column
    dnorm = sqrt(dx ** 2 + dy ** 2)
    sharpness = average(dnorm)
    return sharpness


def translation(frame_0, frame_1, shape):
    """Return translation vector to register images."""

    f0 = fft2(frame_0)
    f1 = fft2(frame_1)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    ty, tx = unravel_index(argmax(ir), shape)

    if ty > shape[0] // 2:
        ty -= shape[0]
    if tx > shape[1] // 2:
        tx -= shape[1]
    # The shift value means that frame_1 must be shifted by this amount to register with frame_0.
    return [ty, tx]


def insert_cross(frame, y_center, x_center, cross_half_len, color):
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


def circle_around(x, y, r):
    if r == 0:
        yield (x, y)
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