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


def search_local_match_correlation(reference_box, frame, y_low, y_high, x_low, x_high,
                                   search_width, sampling_stride, cor_table):
    """
    Try shifts in y, x between the box around the alignment point in the mean frame and the
    corresponding box in the given frame. Start with shifts [0, 0] and move out in steps until
    a local optimum is reached. In each step try all positions with distance 1 in y and/or x
    from the optimum found in the previous step (steepest ascent). The global frame shift is
    accounted for beforehand already.

    :param reference_box: Image box around alignment point in mean frame.
    :param frame: Given frame for which the local shift at the alignment point is to be
                  computed.
    :param y_low: Lower y coordinate limit of box in given frame, taking into account the
                  global shift and the different sizes of the mean frame and the original
                  frames.
    :param y_high: Upper y coordinate limit
    :param x_low: Lower x coordinate limit
    :param x_high: Upper x coordinate limit
    :param search_width: Maximum distance in y and x from origin of the search area
    :param sampling_stride: Stride in both coordinate directions used in computing deviations
    :param cor_table: Scratch table to be used internally for storing intermediate results,
                      size: [2*search_width, 2*search_width], dtype=float32.
    :return: ([shift_y, shift_x], [dev_r]) with:
               shift_y, shift_x: shift values of optimum or [0, 0] if no optimum could be found.
               [dev_r]: list of optimum deviations for all steps until a local optimum is found.
    """

    # Set up a table which keeps correlation values from earlier iteration steps. This way,
    # correlation evaluations can be avoided at coordinates which have been visited before.
    # Initialize correlation with an impossibly low value.
    cor_table[:, :] = -1.

    # Initialize the global optimum with the value at dy=dx=0.
    if sampling_stride != 1:
        correlation_max = (reference_box[::sampling_stride, ::sampling_stride] * frame[
                                                                                 y_low:y_high:sampling_stride,
                                                                                 x_low:x_high:sampling_stride]).sum()
    else:
        correlation_max = (reference_box * frame[y_low:y_high, x_low:x_high]).sum()
    cor_table[0, 0] = correlation_max
    dy_max = 0
    dx_max = 0

    counter_new = 0
    counter_reused = 0

    # Initialize list of maximum correlations for each search radius.
    cor_r = [correlation_max]

    # Start with shift [0, 0]. Stop when a circle with radius 1 around the current optimum
    # reaches beyond the search area.
    while max(abs(dy_max), abs(dx_max)) <= search_width - 1:

        # Create an enumerator which produces shift values [dy, dx] in a circular pattern
        # with radius 1 around the current optimum [dy_min, dx_min].
        circle_1 = Miscellaneous.circle_around(dy_max, dx_max, 1)

        # Initialize the optimum for the new circle to an impossibly large value,
        # and the corresponding shifts to None.
        correlation_max_1, dy_max_1, dx_max_1 = -1., None, None

        # Go through the circle with radius 1 and compute the correlation
        # between the shifted frame and the corresponding box in the mean frame. Find the
        # maximum "correlation_max_1".
        if sampling_stride != 1:
            for (dy, dx) in circle_1:
                correlation = cor_table[dy, dx]
                if correlation < -0.5:
                    counter_new += 1
                    correlation = (reference_box[::sampling_stride, ::sampling_stride] * frame[
                                                                                         y_low - dy:y_high - dy:sampling_stride,
                                                                                         x_low - dx:x_high - dx:sampling_stride]).sum()
                    cor_table[dy, dx] = correlation
                else:
                    counter_reused += 1
                if correlation > correlation_max_1:
                    correlation_max_1, dy_max_1, dx_max_1 = correlation, dy, dx

        else:
            for (dy, dx) in circle_1:
                correlation = cor_table[dy, dx]
                if correlation < -0.5:
                    correlation = (reference_box * frame[y_low - dy:y_high - dy,
                                                   x_low - dx:x_high - dx]).sum()
                    cor_table[dy, dx] = correlation
                if correlation > correlation_max_1:
                    correlation_max_1, dy_max_1, dx_max_1 = correlation, dy, dx

        # Append the minimal deviation found in this step to list of minima.
        cor_r.append(correlation_max_1)

        # If for the current center the match is better than for all neighboring points, a
        # local optimum is found.
        if correlation_max_1 <= correlation_max:
            # print ("new: " + str(counter_new) + ", reused: " + str(counter_reused))
            return [dy_max, dx_max], cor_r

        # Otherwise, update the current optimum and continue.
        else:
            correlation_max, dy_max, dx_max = correlation_max_1, dy_max_1, dx_max_1

    # If within the maximum search radius no optimum could be found, return [0, 0].
    return [0, 0], cor_r


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
    ap_position_y = 350
    ap_position_x = 450
    ap_size = 40
    configuration.alignment_points_half_box_width = int(round(ap_size / 2))
    alignment_points = AlignmentPoints(configuration, frames, rank_frames, align_frames)
    alignment_points.add_alignment_point(alignment_points.new_alignment_point(ap_position_y, ap_position_x, False, False, False, False))

    # Set a reference to the single alignment point.
    alignment_point = alignment_points.alignment_points[0]

    # Rank the frames at the alignment point.
    alignment_points.compute_frame_qualities()

    # Set parameters for noise reduction and search width for both correlation phases.
    blurr_strength_first_phase = 5
    configuration.alignment_points_blurr_strength_first_phase = blurr_strength_first_phase
    stride_first_phase = 2
    search_width_first_phase = 6

    blurr_strength_second_phase = 13
    configuration.alignment_points_blurr_strength_second_phase = blurr_strength_second_phase
    sampling_stride_second_phase = 2
    search_width_second_phase = stride_first_phase + 2

    # Create reference boxes for both phases using the locally sharpest frame.
    alignment_points.set_reference_boxes_correlation()

    best_index = alignment_point['best_frame_indices'][0]
    print("Index of best frame (local): " + str(best_index))

    y_low = alignment_point['box_y_low']
    y_high = alignment_point['box_y_high']
    x_low = alignment_point['box_x_low']
    x_high = alignment_point['box_x_high']

    index_extension = search_width_first_phase * stride_first_phase

    cor_table = zeros((2 * search_width_second_phase, 2 * search_width_second_phase), dtype=float32)

    shift_y_local_total_sum = 0
    shift_x_local_total_sum = 0

    shift_y_corrected = []
    shift_x_corrected = []

    for idx in range(frames.number):
        shift_y_global = align_frames.dy[idx]
        shift_x_global = align_frames.dx[idx]
        frame_window = blurr_image(
            frames.frames_mono_blurred(idx)[
            y_low + shift_y_global - index_extension:y_high + shift_y_global + index_extension:stride_first_phase,
            x_low + shift_x_global - index_extension:x_high + shift_x_global + index_extension:stride_first_phase],
            blurr_strength_first_phase)

        result = matchTemplate((frame_window / 256).astype(uint8),
                               alignment_point['reference_box_first_phase'], TM_SQDIFF_NORMED)
        # result = matchTemplate(frame_window.astype(float32),
        #                        reference_window_first_phase.astype(float32), TM_SQDIFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = minMaxLoc(result)
        shift_y_local_first_phase = (minLoc[1] - search_width_first_phase) * stride_first_phase
        shift_x_local_first_phase = (minLoc[0] - search_width_first_phase) * stride_first_phase

        y_lo = y_low + shift_y_global + shift_y_local_first_phase
        y_hi = y_high + shift_y_global + shift_y_local_first_phase
        x_lo = x_low + shift_x_global + shift_x_local_first_phase
        x_hi = x_high + shift_x_global + shift_x_local_first_phase


        frame_blurred_second_phase = blurr_image(frames.frames_mono_blurred(idx),
                                                 blurr_strength_second_phase)
        [shift_y_local_second_phase, shift_x_local_second_phase], dev_r \
            = search_local_match_correlation(alignment_point['reference_box_second_phase'],
                                             frame_blurred_second_phase, y_lo, y_hi,
                                             x_lo, x_hi, search_width_second_phase,
                                             sampling_stride_second_phase, cor_table)

        shift_y_local_total = shift_y_local_first_phase + shift_y_local_second_phase
        shift_x_local_total = shift_x_local_first_phase + shift_x_local_second_phase
        shift_y_local_total_sum += shift_y_local_total
        shift_x_local_total_sum += shift_x_local_total

        shift_y_corrected.append(shift_y_local_total)
        shift_x_corrected.append(shift_x_local_total)

        frame_window_shifted_first_phase = blurr_image(
            frames.frames_mono_blurred(idx)[y_lo:y_hi:stride_first_phase,
            x_lo:x_hi:stride_first_phase],
            blurr_strength_first_phase)

        frame_window_shifted_second_phase = blurr_image(
            frames.frames_mono_blurred(idx)[
            y_low + shift_y_global + shift_y_local_total:y_high + shift_y_global + shift_y_local_total,
            x_low + shift_x_global + shift_x_local_total:x_high + shift_x_global + shift_x_local_total],
            blurr_strength_first_phase)

        composite_image = Miscellaneous.compose_image(
            [(alignment_point['reference_box_first_phase']*256).astype(uint16), frame_window, frame_window_shifted_first_phase,
             frame_window_shifted_second_phase],
            scale_factor=1)
        display_image(composite_image, delay=0.1)

        print("frame index: " + str(idx) + ", shift first phase: [" + str(
            shift_y_local_first_phase) + ", " + str(shift_x_local_first_phase) +
              "], shift second phase: [" + str(shift_y_local_second_phase) + ", " + str(
            shift_x_local_second_phase) +
              "], total shift: [" + str(shift_y_local_total) + ", " + str(
            shift_x_local_total) + "]")

    shift_y_reference = shift_y_local_total_sum / frames.number
    shift_x_reference = shift_x_local_total_sum / frames.number

    for idx in range(frames.number):
        shift_y_corrected[idx] -= shift_y_reference
        shift_x_corrected[idx] -= shift_x_reference

    print("")
    print("reference patch shift, y: " + str(shift_y_reference) + ", x: " + str(shift_x_reference))

    print("")
    for idx in range(frames.number):
        print("frame index: " + str(idx) + ", local AP shift: [" + str(
            shift_y_corrected[idx]) + ", " + str(shift_x_corrected[idx]) + "]")

    display_image(None)

    # Print out timer results.
    my_timer.stop('Execution over all')
    my_timer.print()
