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
from cv2 import GaussianBlur, matchTemplate, TM_CCORR, TM_SQDIFF_NORMED, CV_32F, TM_SQDIFF, \
    minMaxLoc, meanStdDev
from numpy import float32, unravel_index, argmin, zeros, uint8, uint16

from align_frames import AlignFrames
from alignment_points import AlignmentPoints
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

    input_name = 'D:/SW-Development/Python/PlanetarySystemStacker/Examples/Jupiter/2019-05-26' \
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

    # Create a single alignment point.
    ap_position_y = 330
    ap_position_x = 430
    ap_size = 40
    configuration.alignment_points_half_box_width = int(round(ap_size / 2))
    alignment_points = AlignmentPoints(configuration, frames, rank_frames, align_frames)
    alignment_points.add_alignment_point(
        alignment_points.new_alignment_point(ap_position_y, ap_position_x, False, False, False,
                                             False))

    # Set a reference to the single alignment point.
    alignment_point = alignment_points.alignment_points[0]

    # Rank the frames at the alignment point.
    alignment_points.compute_frame_qualities()

    # Set the combined search width.
    search_width = 16

    # Set parameters for noise reduction and search width for both correlation phases.
    blurr_strength_first_phase = 11
    configuration.alignment_points_blurr_strength_first_phase = blurr_strength_first_phase

    blurr_strength_second_phase = 7
    configuration.alignment_points_blurr_strength_second_phase = blurr_strength_second_phase

    # Create reference boxes for both phases using the locally sharpest frame.
    alignment_points.set_reference_boxes_correlation()

    best_index = alignment_point['best_frame_indices'][0]
    print("Index of best frame (local): " + str(best_index))

    # Initialize variables for the computation of the mean shifts.
    shift_y_local_sum = 0
    shift_x_local_sum = 0

    # Intialize lists to store the warp shifts. They will be corrected later by the bias caused by
    # the shift of the reference patch.
    shift_y_local_corrected = []
    shift_x_local_corrected = []

    for idx in range(frames.number):
        y_low = alignment_point['box_y_low'] + align_frames.dy[idx]
        y_high = alignment_point['box_y_high'] + align_frames.dy[idx]
        x_low = alignment_point['box_x_low'] + align_frames.dx[idx]
        x_high = alignment_point['box_x_high'] + align_frames.dx[idx]

        # Apply a Gaussian filter to the original frame for the second correlation phase.
        frame_blurred_second_phase = blurr_image(frames.frames_mono_blurred(idx),
                                                 blurr_strength_second_phase)

        # Compute the warp shift using multi-level normalized cross correlation.
        shift_y_local_first_phase, shift_x_local_first_phase, success_first_phase, \
        shift_y_local_second_phase, shift_x_local_second_phase, success_second_phase = \
            Miscellaneous.multilevel_correlation(
                alignment_point['reference_box_first_phase'], frames.frames_mono_blurred(idx),
                blurr_strength_first_phase,
                alignment_point['reference_box_second_phase'], frame_blurred_second_phase, y_low,
                y_high, x_low, x_high, search_width)

        # Combine the shifts of both phases and update the summation variables (to determine the
        # mean warp shift).
        # shift_y_local_second_phase = 0
        # shift_x_local_second_phase = 0
        shift_y_local = shift_y_local_first_phase + shift_y_local_second_phase
        shift_x_local = shift_x_local_first_phase + shift_x_local_second_phase
        shift_y_local_sum += shift_y_local
        shift_x_local_sum += shift_x_local

        # Store the warps measured for this frame.
        shift_y_local_corrected.append(shift_y_local)
        shift_x_local_corrected.append(shift_x_local)

        # The following is code for visualizing the effect of warp correction (of both phases).
        search_width_second_phase = 4
        search_width_first_phase = int((search_width - search_width_second_phase) / 2)
        index_extension = search_width_first_phase * 2

        frame_window_first_phase = blurr_image(frames.frames_mono_blurred(idx)[
           y_low - index_extension:y_high + index_extension:2,
           x_low - index_extension:x_high + index_extension:2],
           blurr_strength_first_phase)

        frame_window_shifted_first_phase = blurr_image(
            frames.frames_mono_blurred(idx)[
            y_low - shift_y_local_first_phase:y_high - shift_y_local_first_phase:2,
            x_low - shift_x_local_first_phase:x_high - shift_x_local_first_phase:2],
            blurr_strength_first_phase)

        frame_window_shifted_second_phase = blurr_image(
            frames.frames_mono_blurred(idx)[
            y_low - shift_y_local:y_high - shift_y_local,
            x_low - shift_x_local:x_high - shift_x_local],
            blurr_strength_second_phase)

        composite_image = Miscellaneous.compose_image(
            [frame_window_first_phase,
             blurr_image(alignment_point['reference_box_first_phase'].astype(uint16), blurr_strength_first_phase),
             frame_window_shifted_first_phase,
             frame_window_shifted_second_phase], scale_factor=1)
        display_image(composite_image, delay=0.1)

        print("frame index: " + str(idx) + ", shift first phase: [" + str(
            shift_y_local_first_phase) + ", " + str(shift_x_local_first_phase) +
              "], success first phase: " + str(
            success_first_phase) + ", shift second phase: [" + str(
            shift_y_local_second_phase) + ", " + str(
            shift_x_local_second_phase) +
              "], success second phase: " + str(success_second_phase) + ", total shift: [" + str(
            shift_y_local) + ", " + str(
            shift_x_local) + "]")

    # The warp of the reference patch is computed as the mean value of all shifts.
    shift_y_reference = shift_y_local_sum / frames.number
    shift_x_reference = shift_x_local_sum / frames.number

    # For all frames correct the measured shift values for the bias caused by the shift of the
    # reference patch.
    for idx in range(frames.number):
        shift_y_local_corrected[idx] -= shift_y_reference
        shift_x_local_corrected[idx] -= shift_x_reference

    print("")
    print("reference patch shift, y: " + str(shift_y_reference) + ", x: " + str(shift_x_reference))

    print("")
    for idx in range(frames.number):
        print("frame index: " + str(idx) + ", local AP shift: [" + str(
            shift_y_local_corrected[idx]) + ", " + str(shift_x_local_corrected[idx]) + "]")

    display_image(None)

    # Print out timer results.
    my_timer.stop('Execution over all')
    my_timer.print()
