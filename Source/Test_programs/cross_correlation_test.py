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

import ctypes
import platform
from ctypes import CDLL
from time import sleep

import matplotlib.pyplot as plt
import pylab as pl
from cv2 import GaussianBlur, matchTemplate, TM_CCORR, TM_SQDIFF_NORMED, CV_32F, TM_SQDIFF, minMaxLoc
from numpy import float32, unravel_index, argmin

from align_frames import AlignFrames
from configuration import Configuration
from miscellaneous import Miscellaneous
from exceptions import NotSupportedError, InternalError, Error
from frames import Frames
from rank_frames import RankFrames
from timer import timer

img = None


def display_image(image, delay=0.1):
    """
    Display a succession of images in the same Matplotlib window. The window is opened in the
    first call. To close the window, call the function with image = None.
    """
    global img

    if image is not None:
        if img is None:
            img = pl.imshow(image, cmap='Greys_r')
        else:
            img.set_data(image)
        pl.draw()
        pl.pause(delay)
    else:
        img = None
        pl.close()


def blurr_image(image, strength):
    return GaussianBlur(image, (strength, strength), 0)


if __name__ == "__main__":
    """
    This File contains a test program for the measurement of local warp shifts. The idea is to use
    cross correlation on a coarser pixel grid, followed by a local steepest descent on the finest
    grid.
    """

    if platform.system() == 'Windows':
        mkl_rt = CDLL('mkl_rt.dll')
    else:
        mkl_rt = CDLL('libmkl_rt.so')

    mkl_get_max_threads = mkl_rt.mkl_get_max_threads


    def mkl_set_num_threads(cores):
        mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))


    mkl_set_num_threads(2)
    print("Number of threads used by mkl: " + str(mkl_get_max_threads()))

    input_name = 'E:/SW-Development/Python/PlanetarySystemStacker/Examples/Jupiter/2019-05-26' \
                 '-0115_4-L-Jupiter_ZWO ASI290MM Mini_short.avi'

    # Initalize the timer object used to measure execution times of program sections.
    my_timer = timer()

    print(
        "\n" +
        "*************************************************************************************\n"
        + "Start processing " + str(
            input_name) +
        "\n*************************************************************************************")
    my_timer.create('Execution over all')

    # Get configuration parameters.
    configuration = Configuration()

    # Create the frames object.
    try:
        frames = Frames(configuration, input_name, type='video', convert_to_grayscale=False,
                        buffer_original=True, buffer_monochrome=True, buffer_gaussian=True,
                        buffer_laplacian=True)
        print("Number of images read: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Error as e:
        print("Error: " + str(e))
        exit()

    # Rank the frames by their overall local contrast.
    print("+++ Start ranking images")
    my_timer.create('Ranking images')
    rank_frames = RankFrames(frames, configuration)
    rank_frames.frame_score()
    my_timer.stop('Ranking images')
    print("Index of best frame: " + str(rank_frames.frame_ranks_max_index))

    # Initialize the frame alignment object.
    align_frames = AlignFrames(frames, rank_frames, configuration)

    if configuration.align_frames_mode == "Surface":
        my_timer.create('Select optimal alignment patch')
        # Select the local rectangular patch in the image where the L gradient is highest in both x
        # and y direction. The scale factor specifies how much smaller the patch is compared to the
        # whole image frame.
        (y_low_opt, y_high_opt, x_low_opt, x_high_opt) = align_frames.compute_alignment_rect(
            configuration.align_frames_rectangle_scale_factor)
        my_timer.stop('Select optimal alignment patch')

        print("optimal alignment rectangle, y_low: " + str(y_low_opt) + ", y_high: " + str(
            y_high_opt) + ", x_low: " + str(x_low_opt) + ", x_high: " + str(x_high_opt))

    # Align all frames globally relative to the frame with the highest score.
    print("+++ Start aligning all frames")
    my_timer.create('Global frame alignment')
    try:
        align_frames.align_frames()
    except NotSupportedError as e:
        print("Error: " + e.message)
        exit()
    except InternalError as e:
        print("Warning: " + e.message)
    my_timer.stop('Global frame alignment')

    print("Intersection, y_low: " + str(align_frames.intersection_shape[0][0]) + ", y_high: " + str(
        align_frames.intersection_shape[0][1]) + ", x_low: " + str(
        align_frames.intersection_shape[1][0]) + ", x_high: " + str(
        align_frames.intersection_shape[1][1]))

    # Compute the average frame.
    print("+++ Start computing average frame")
    my_timer.create('Compute reference frame')
    average = align_frames.average_frame()
    my_timer.stop('Compute reference frame')
    print("Average frame computed from the best " + str(
        align_frames.average_frame_number) + " frames.")

    # Show the full average frame.
    # plt.imshow(average, cmap='Greys_r')
    # plt.show()

    # Show the sharpest single frame.
    # plt.imshow(frames.frames_mono_blurred(rank_frames.frame_ranks_max_index), cmap='Greys_r')
    # plt.show()

    ap_position_y = 350
    ap_position_x = 450

    ap_size = 40
    blurr_strength = 5
    stride = 1

    y_low = ap_position_y - int(ap_size / 2)
    y_high = y_low + ap_size
    x_low = ap_position_x - int(ap_size / 2)
    x_high = x_low + ap_size

    reference_index = rank_frames.frame_ranks_max_index
    reference_window = blurr_image(
        frames.frames_mono_blurred(reference_index)[y_low:y_high:stride, x_low:x_high:stride],
        blurr_strength)

    search_width = 8
    index_extension = search_width * stride

    for idx in range(frames.number):
        shift_y_global = align_frames.frame_shifts[idx][0]
        shift_x_global = align_frames.frame_shifts[idx][1]
        frame_window = blurr_image(
            frames.frames_mono_blurred(idx)[y_low - shift_y_global - index_extension:y_high - shift_y_global + index_extension:stride,
            x_low - shift_x_global - index_extension:x_high - shift_x_global + index_extension:stride], blurr_strength)

        result = matchTemplate(frame_window.astype(float32), reference_window.astype(float32), TM_SQDIFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = minMaxLoc(result)
        shift_y_local = (minLoc[1] - search_width) * stride
        shift_x_local = (minLoc[0] - search_width) * stride

        frame_window_shifted = blurr_image(
            frames.frames_mono_blurred(idx)[y_low - shift_y_global + shift_y_local:y_high - shift_y_global + shift_y_local:stride,
            x_low - shift_x_global + shift_x_local:x_high - shift_x_global + shift_x_local:stride], blurr_strength)

        composite_image = Miscellaneous.compose_image([reference_window, frame_window, frame_window_shifted],
                                                      scale_factor=1)
        display_image(composite_image, delay=0.1)

    display_image(None)

    # Print out timer results.
    my_timer.stop('Execution over all')
    my_timer.print()
