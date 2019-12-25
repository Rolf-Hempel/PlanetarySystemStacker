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

from datetime import datetime
from os import unlink
from sys import stdout
from time import time

from cv2 import CV_32F, Laplacian, VideoWriter_fourcc, VideoWriter, FONT_HERSHEY_SIMPLEX, LINE_AA, \
    putText, GaussianBlur, cvtColor, COLOR_BGR2HSV, COLOR_HSV2BGR, BORDER_DEFAULT, meanStdDev,\
    resize, matchTemplate, TM_SQDIFF_NORMED, minMaxLoc, TM_CCORR_NORMED
from numpy import abs as np_abs
from numpy import diff, average, hypot, sqrt, unravel_index, argmax, zeros, arange, array, matmul, \
    empty, argmin, stack, sin, uint8, float32, uint16, full, mean
from math import exp
from numpy import min as np_min
from numpy.fft import fft2, ifft2
from numpy.linalg import solve
from scipy.ndimage import sobel

from exceptions import DivideByZeroError, ArgumentError


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

        # Compute the sharpness per coordinate direction as the 1-norm of point values.
        sharpness_x = average(np_abs(dx))
        sharpness_y = average(np_abs(dy))

        # Return the sharpness in the direction where it is minimal.
        return min(sharpness_x, sharpness_y)

    @staticmethod
    def quality_measure_threshold(frame, black_threshold=40.):
        """
        This is an alternative method for computing the amount of local structure. Here the
        summation only takes into account points where the luminosity exceeds a certain
        threshold.

        :param frame: 2D image
        :param black_threshold: Threshold for points to be considered
        :return:
        """

        sum_horizontal = abs((frame[:, 2:] - frame[:, :-2])[frame[:, 1:-1] > black_threshold]).sum()
        sum_vertical = abs((frame[2:, :] - frame[:-2, :])[frame[1:-1, :] > black_threshold]).sum()

        return min(sum_horizontal, sum_vertical)

    @staticmethod
    def quality_measure_threshold_weighted(frame, stride=2, black_threshold=40., min_fraction=0.7):
        """
        This is an alternative method for computing the amount of local structure. Here the
        summation only takes into account points where the luminosity exceeds a certain
        threshold. Additionally, if not too many points are discarded, the reduced sampling size is
        compensated.

        :param frame: 2D image
        :param stride: Stride for gradient computation. For blurry images increase this value.
        :param black_threshold: Threshold for points to be considered.
        :param min_fraction: Minimum fraction of points to pass the threshold.
        :return:
        """

        frame_size = frame.shape[0] * frame.shape[1]
        stride_2 = 2 * stride

        # Compute a mask for all pixels which are bright enough (to avoid background noise).
        mask = frame[:, :] > black_threshold
        mask_fraction = mask.sum() / frame_size

        # If most pixels are bright enough, compensate for different pixel counts.
        if mask_fraction > min_fraction:
            sum_horizontal = abs((frame[:, stride_2:] - frame[:, :-stride_2])[mask[:, stride:-stride]]).sum() / mask_fraction
            sum_vertical = abs((frame[stride_2:, :] - frame[:-stride_2, :])[mask[stride:-stride, :]]).sum() / mask_fraction
        # If many pixels are too dim, penalize this patch by not compensating for pixel count.
        else:
            sum_horizontal = abs((frame[:, stride_2:] - frame[:, :-stride_2])[mask[:, stride:-stride]]).sum()
            sum_vertical = abs((frame[stride_2:, :] - frame[:-stride_2, :])[mask[stride:-stride, :]]).sum()

        return min(sum_horizontal, sum_vertical)

    @staticmethod
    def local_contrast_laplace(frame, stride):
        """
        Measure the amount of structure in a rectangular frame in both coordinate directions and
        return the minimum value. The discrete variance of the Laplacian is computed.

        :param frame: 2D image
        :param stride: Factor for down-sampling
        :return: Measure for amount of local structure in the image (scalar)
        """

        # sharpness = sum(laplace(frame[::stride, ::stride])**2)
        return meanStdDev(Laplacian(frame[::stride, ::stride], CV_32F))[1][0][0]

    @staticmethod
    def local_contrast_sobel(frame, stride):
        """
        Compute a measure for local contrast in an image using the Sobel method.

        :param frame: 2D image
        :param stride: Factor for down-sampling
        :return: Overall sharpness measure (scalar)
        """

        frame_int32 = frame[::stride, ::stride].astype('int32')
        dx = sobel(frame_int32, 0)  # vertical derivative
        dy = sobel(frame_int32, 1)  # horizontal derivative
        mag = hypot(dx, dy)  # magnitude

        return mag.sum(axis=0)

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
        dnorm = hypot(dx, dy)

        return average(dnorm)

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
    def search_local_match(reference_box, frame, y_low, y_high, x_low, x_high, search_width,
                           sampling_stride, sub_pixel=True):
        """
        Try shifts in y, x between the box around the alignment point in the mean frame and the
        corresponding box in the given frame. Start with shifts [0, 0] and move out in a circular
        fashion, until the radius "search_width" is reached. The global frame shift is accounted for
        beforehand already.

        :param reference_box: Image box around alignment point in mean frame.
        :param frame: Given frame for which the local shift at the alignment point is to be
                      computed.
        :param y_low: Lower y coordinate limit of box in given frame, taking into account the
                      global shift and the different sizes of the mean frame and the original
                      frames.
        :param y_high: Upper y coordinate limit
        :param x_low: Lower x coordinate limit
        :param x_high: Upper x coordinate limit
        :param search_width: Maximum radius of the search spiral
        :param sampling_stride: Stride in both coordinate directions used in computing deviations
        :param sub_pixel: If True, compute local shifts with sub-pixel accuracy
        :return: ([shift_y, shift_x], [min_r]) with:
                   shift_y, shift_x: shift values of minimum or [0, 0] if no optimum could be found.
                   [dev_r]: list of minimum deviations for radius r=0 ... r_max, where r_max is the
                            widest search radius tested.
        """

        # Initialize the global optimum with an impossibly large value.
        deviation_min = 1.e30
        dy_min = None
        dx_min = None

        # Initialize list of minimum deviations for each search radius and field of deviations.
        dev_r = []
        deviations = zeros((2 * search_width + 1, 2 * search_width + 1))

        # Start with shift [0, 0] and proceed in a circular pattern.
        for r in arange(search_width + 1):

            # Create an enumerator which produces shift values [dy, dx] in a circular pattern
            # with radius "r".
            circle_r = Miscellaneous.circle_around(0, 0, r)

            # Initialize the optimum for radius "r" to an impossibly large value,
            # and the corresponding shifts to None.
            deviation_min_r, dy_min_r, dx_min_r = 1.e30, None, None

            # Go through the circle with radius "r" and compute the difference (deviation)
            # between the shifted frame and the corresponding box in the mean frame. Find the
            # minimum "deviation_min_r" for radius "r".
            if sampling_stride != 1:
                for (dy, dx) in circle_r:
                    deviation = abs(
                        reference_box[::sampling_stride, ::sampling_stride] - frame[
                                      y_low - dy:y_high - dy:sampling_stride,
                                      x_low - dx:x_high - dx:sampling_stride]).sum()
                    # deviation = sqrt(square(
                    #     reference_box - frame[y_low - dy:y_high - dy, x_low - dx:x_high - dx]).sum())
                    if deviation < deviation_min_r:
                        deviation_min_r, dy_min_r, dx_min_r = deviation, dy, dx
                    deviations[dy + search_width, dx + search_width] = deviation
            else:
                for (dy, dx) in circle_r:
                    deviation = abs(
                        reference_box - frame[y_low - dy:y_high - dy, x_low - dx:x_high - dx]).sum()
                    if deviation < deviation_min_r:
                        deviation_min_r, dy_min_r, dx_min_r = deviation, dy, dx
                    deviations[dy + search_width, dx + search_width] = deviation

            # Append the minimal deviation for radius r to list of minima.
            dev_r.append(deviation_min_r)

            # If for the current radius there is no improvement compared to the previous radius,
            # the optimum is reached.
            if deviation_min_r >= deviation_min:

                # For sub-pixel accuracy, find local minimum of fitting paraboloid.
                if sub_pixel:
                    try:
                        y_correction, x_correction = Miscellaneous.sub_pixel_solve(deviations[
                               dy_min + search_width - 1: dy_min + search_width + 2,
                               dx_min + search_width - 1: dx_min + search_width + 2])
                    except DivideByZeroError as ex:
                        print(ex.message)
                        x_correction = y_correction = 0.

                    # Add the sub-pixel correction to the local shift. If the correction is above
                    # one pixel, something went wrong.
                    if abs(y_correction) < 1. and abs(x_correction) < 1.:
                        dy_min += y_correction
                        dx_min += x_correction
                    # else:
                    #     print ("y_correction: " + str(y_correction) + ", x_correction: " +
                    #           str(x_correction))

                return [dy_min, dx_min], dev_r

            # Otherwise, update the current optimum and continue.
            else:
                deviation_min, dy_min, dx_min = deviation_min_r, dy_min_r, dx_min_r

        # If within the maximum search radius no optimum could be found, return [0, 0].
        return [0, 0], dev_r

    @staticmethod
    def sub_pixel_solve(function_values):
        """
        Compute the sub-pixel correction for method "search_local_match".

        :param function_values: Matching differences at (3 x 3) pixels around the minimum found
        :return: Corrections in y and x to the center position for local minimum
        """

        # If the functions are not yet reduced to 1D, do it now.
        function_values_1d = function_values.reshape((9,))

        # There are nine equations for six unknowns. Use normal equations to solve for optimum.
        a_transpose = array(
            [[1., 0., 1., 1., 0., 1., 1., 0., 1.], [1., 1., 1., 0., 0., 0., 1., 1., 1.],
             [1., -0., -1., -0., 0., 0., -1., 0., 1.], [-1., 0., 1., -1., 0., 1., -1., 0., 1.],
             [-1., -1., -1., 0., 0., 0., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1.]])
        a_transpose_a = array(
            [[6., 4., 0., 0., 0., 6.], [4., 6., 0., 0., 0., 6.], [0., 0., 4., 0., 0., 0.],
             [0., 0., 0., 6., 0., 0.], [0., 0., 0., 0., 6., 0.], [6., 6., 0., 0., 0., 9.]])

        # Right hand side is "a transposed times input vector".
        rhs = matmul(a_transpose, function_values_1d)

        # Solve for parameters of the fitting function
        # f = a_f * x ** 2 + b_f * y ** 2 + c_f * x * y + d_f * x + e_f * y + g_f
        a_f, b_f, c_f, d_f, e_f, g_f = solve(a_transpose_a, rhs)

        # The corrected pixel values of the minimum result from setting the first derivatives of
        # the fitting funtion in y and x direction to zero, and solving for y and x.
        denominator_y = c_f ** 2 - 4. * a_f * b_f
        if abs(denominator_y) > 1.e-10 and abs(a_f) > 1.e-10:
            y_correction = (2. * a_f * e_f - c_f * d_f) / denominator_y
            x_correction = (- c_f * y_correction - d_f) / (2. * a_f)
        elif abs(denominator_y) > 1.e-10 and abs(c_f) > 1.e-10:
            y_correction = (2. * a_f * e_f - c_f * d_f) / denominator_y
            x_correction = (-2. * b_f * y_correction - e_f) / c_f
        else:
            raise DivideByZeroError("Sub-pixel shift cannot be computed, set to zero")

        return y_correction, x_correction

    @staticmethod
    def search_local_match_init(reference_frame, y_low, y_high, x_low, x_high, search_width):
        """
        As an alternative to the method "search_local_match" above there is a split version
        consisting of this initialization method and the execution method. For a given alignment
        point the initialization has to be called once only. The data structures allocated during
        initialization are re-used with each frame for which the shift against the reference frame
        is to be computed. The individual frame shift is computed with the execution method below.

        The split version can be better suited for execution on a graphic accelerator. The biggest
        part of the data does not change between frame executions, so it has to be copied to the
        graphics memory only once. The executions for individual frames only need little additional
        data, and they can be performed in parallel.

        The split version is not available with subpixel accuracy.

        :param reference_frame: Reference frame, against which the current frame shift is to be
                                computed (note: The full frame, not the alignment box)
        :param y_low: Lower y coordinate limit of alignment box in reference frame
        :param y_high: Upper y coordinate limit of alignment box in reference frame
        :param x_low: Lower x coordinate limit of alignment box in reference frame
        :param x_high: Upper x coordinate limit of alignment box in reference frame
        :param search_width: Maximum radius of the search spiral
        :return: (reference_stack, displacements,  radius_start) witn
                  reference_stack: Stack of shifted reference frame alignment boxes
                  displacements: List of [dy, dx] displacements for all stack elements
                  radius_start: Start indices into "reference_stack" and "displacements" for each
                                radius (from 0 to search_width)
        """
        # Compute the maximal number of shifts to be tested.
        search_dim = (2*search_width+1)**2

        # Allocate permanent data structures.
        window_height = y_high - y_low
        window_width = x_high - x_low
        reference_stack = empty([search_dim, window_height, window_width],
                                dtype=reference_frame.dtype)
        displacements = []
        radius_start = [0]

        # Create the reference_stack containing shifted windows into the reference frame.
        index = 0
        for r in range(search_width + 1):
            # Create an enumerator which produces shift values [dy, dx] in a circular pattern
            # with radius "r".
            for (dy, dx) in Miscellaneous.circle_around(0, 0, r):
                reference_stack[index, :, :] = reference_frame[y_low + dy:y_high + dy,
                                               x_low + dx:x_high + dx]
                displacements.append([dy, dx])
                index += 1
            radius_start.append(index)
        return (reference_stack, displacements, radius_start)

    @staticmethod
    def search_local_match_execute(frame_window, reference_stack, displacements, radius_start):
        """
        The "execute" method of the split version of "search_local_match". This method is executed
        for each frame, potentially in parallel for all frames.

        :param frame_window: Window into the current frame of the same shape as specified for the
                             reference window in the init method. Be careful: the current frame and
                             the reference frame have different shapes.
        :param reference_stack: Permanent object allocated by init routing (see above)
        :param displacements: Permanent object allocated by init routing (see above)
        :param radius_start: Permanent object allocated by init routing (see above)
        :return: ([shift_y, shift_x], [min_r]) with:
                   shift_y, shift_x: shift values of minimum or [0, 0] if no optimum could be found.
                   [dev_r]: list of minimum deviations for radius r=0 ... r_max, where r_max is the
                            widest search radius tested.
        """

        search_width_plus_1 = len(radius_start)
        # Initialize the global optimum with an impossibly large value.
        deviation_min = 1.e30
        index_min = None

        # Initialize list of minimum deviations for each search radius and field of deviations.
        dev_r = []

        # Compare frame_window with a stack of shifted reference frame windows stored for radius r.
        for r in arange(search_width_plus_1):
            temp_vec = abs(
                reference_stack[radius_start[r]:radius_start[r + 1], :, :] - frame_window).sum(
                axis=(1, 2))
            deviation_min_r = np_min(temp_vec)
            index_min_r = argmin(temp_vec) + radius_start[r]

            # The same in loop notation:
            #
            # deviation_min_r, index_min_r = 1.e30, None
            # for index in arange(radius_start[r], radius_start[r+1]):
            #     deviation = abs(reference_stack[index, :,:] - frame_window).sum()
            #     if deviation < deviation_min_r:
            #         deviation_min_r = deviation
            #         index_min_r = index

            # Append the minimal deviation for radius r to list of minima.
            dev_r.append(deviation_min_r)

            if deviation_min_r >= deviation_min:
                return displacements[index_min], dev_r

            # Otherwise, update the current optimum and continue.
            else:
                deviation_min = deviation_min_r
                index_min = index_min_r

        # If within the maximum search radius no optimum could be found, return [0, 0].
        return [0, 0], dev_r

    @staticmethod
    def search_local_match_gradient(reference_box, frame, y_low, y_high, x_low, x_high, search_width,
                           sampling_stride, dev_table):
        """
        Try shifts in y, x between the box around the alignment point in the mean frame and the
        corresponding box in the given frame. Start with shifts [0, 0] and move out in steps until
        a local optimum is reached. In each step try all positions with distance 1 in y and/or x
        from the optimum found in the previous step (steepest descent). The global frame shift is
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
        :param dev_table: Scratch table to be used internally for storing intermediate results,
                          size: [2*search_width, 2*search_width], dtype=float32.
        :return: ([shift_y, shift_x], [dev_r]) with:
                   shift_y, shift_x: shift values of minimum or [0, 0] if no optimum could be found.
                   [dev_r]: list of minimum deviations for all steps until a local minimum is found.
        """

        # Set up a table which keeps deviation values from earlier iteration steps. This way,
        # deviation evaluations can be avoided at coordinates which have been visited before.
        # Initialize deviations with an impossibly high value.
        dev_table[:,:] = 1.e30

        # Initialize the global optimum with the value at dy=dx=0.
        if sampling_stride != 1:
            deviation_min = abs(reference_box[::sampling_stride, ::sampling_stride] - frame[
                                          y_low:y_high:sampling_stride,
                                          x_low:x_high:sampling_stride]).sum()
        else:
            deviation_min = abs(reference_box - frame[y_low:y_high, x_low:x_high]).sum()
        dev_table[0, 0] = deviation_min
        dy_min = 0
        dx_min = 0

        counter_new = 0
        counter_reused = 0

        # Initialize list of minimum deviations for each search radius.
        dev_r = [deviation_min]

        # Start with shift [0, 0]. Stop when a circle with radius 1 around the current optimum
        # reaches beyond the search area.
        while max(abs(dy_min), abs(dx_min)) <= search_width-1:

            # Create an enumerator which produces shift values [dy, dx] in a circular pattern
            # with radius 1 around the current optimum [dy_min, dx_min].
            circle_1 = Miscellaneous.circle_around(dy_min, dx_min, 1)

            # Initialize the optimum for the new circle to an impossibly large value,
            # and the corresponding shifts to None.
            deviation_min_1, dy_min_1, dx_min_1 = 1.e30, None, None

            # Go through the circle with radius 1 and compute the difference (deviation)
            # between the shifted frame and the corresponding box in the mean frame. Find the
            # minimum "deviation_min_1".
            if sampling_stride != 1:
                for (dy, dx) in circle_1:
                    deviation = dev_table[dy, dx]
                    if deviation > 1.e29:
                        counter_new += 1
                        deviation = abs(reference_box[::sampling_stride, ::sampling_stride] - frame[
                                          y_low - dy:y_high - dy:sampling_stride,
                                          x_low - dx:x_high - dx:sampling_stride]).sum()
                        dev_table[dy, dx] = deviation
                    else:
                        counter_reused += 1
                    if deviation < deviation_min_1:
                        deviation_min_1, dy_min_1, dx_min_1 = deviation, dy, dx

                    # deviation = abs(reference_box[::sampling_stride, ::sampling_stride] - frame[
                    #                   y_low - dy:y_high - dy:sampling_stride,
                    #                   x_low - dx:x_high - dx:sampling_stride]).sum()
                    # if deviation < deviation_min_1:
                    #     deviation_min_1, dy_min_1, dx_min_1 = deviation, dy, dx
            else:
                for (dy, dx) in circle_1:
                    deviation = dev_table[dy, dx]
                    if deviation > 1.e29:
                        deviation = abs(
                            reference_box - frame[y_low - dy:y_high - dy,
                                            x_low - dx:x_high - dx]).sum()
                        dev_table[dy, dx] = deviation
                    if deviation < deviation_min_1:
                        deviation_min_1, dy_min_1, dx_min_1 = deviation, dy, dx

            # Append the minimal deviation found in this step to list of minima.
            dev_r.append(deviation_min_1)

            # If for the current center the match is better than for all neighboring points, a
            # local optimum is found.
            if deviation_min_1 >= deviation_min:
                # print ("new: " + str(counter_new) + ", reused: " + str(counter_reused))
                return [dy_min, dx_min], dev_r

            # Otherwise, update the current optimum and continue.
            else:
                deviation_min, dy_min, dx_min = deviation_min_1, dy_min_1, dx_min_1

        # If within the maximum search radius no optimum could be found, return [0, 0].
        return [0, 0], dev_r

    @staticmethod
    def search_local_match_multilevel(alignment_point,
                    frame, dy_global, dx_global, number_levels, alignment_points_noise_levels,
                    alignment_points_iterations, alignment_points_sampling_stride):
        """
        Measure the warp shift at various levels, starting with the coarsest one. At each level
        determine the best location in a circle of radius 1 around the optimum found at the
        previous level. Since the pixel count at each level is doubled as compared to the previous
        one, the total shift value gets more and more refined.

        :param alignment_point: AP at which the warp shift is to be measured.
        :param frame: Given frame (with low-pass filter applied) for which the local shift at
                      the alignment point is to be measured.
        :param dy_global: Global y shift measured in frame stabilization.
        :param dx_global: Global x shift measured in frame stabilization.
        :param number_levels: Number of levels, with level 0 being the original frame.
        :param alignment_points_noise_levels: List, for each level: Width of Gaussian filter
                                              to be applied to frame befor matching.
        :param alignment_points_iterations: List, for each level: Number of iterations for AP
                                            matching.
        :param alignment_points_sampling_stride: Stride in both coordinate directions used in
                                                 computing deviations (so far not used)
        :return: ([shift_y, shift_x], dev) with:
                   shift_y, shift_x: Total warp shift including contributions from all levels.
                   dev: Remaining deviation of shifted patch relative to reference frame.
        """

        # Initialize the total warp shift (including contributions from all levels).
        dy_warp = 0
        dx_warp = 0
        de_warp_shifts = [None] * number_levels

        # Measure and accumulate shifts at all levels starting with the coarsest one.
        for level in reversed(range(number_levels)):

            # The "reference_box" already contains strided data. The current frame is only given in
            # full resolution. At each level a strided window centered around the shifted AP position
            # (including shifts from all previous levels) is compared with the reference box.
            stride = 2 ** level
            reference_box = alignment_point['reference_boxes'][level]

            # Compute the AP position including shifts from all previous levels.
            y_level = alignment_point['y_levels'][level] + dy_global - dy_warp
            x_level = alignment_point['x_levels'][level] + dx_global - dx_warp
            half_box_width = alignment_point['half_box_widths'][level]

            # Initialize the global optimum with the value at dy=dx=0.
            y_low = y_level - half_box_width
            y_high = y_level + half_box_width
            x_low = x_level - half_box_width
            x_high = x_level + half_box_width
            if level == 0:
                deviation_min = abs(
                    reference_box - frame[y_low:y_high:stride, x_low:x_high:stride]).sum()
            else:
                deviation_min = abs(
                    reference_box - GaussianBlur(frame[y_low:y_high:stride, x_low:x_high:stride],
                                                 (alignment_points_noise_levels[level],
                                                  alignment_points_noise_levels[level]), 0)).sum()
            dy_min = 0
            dx_min = 0

            for iteration in range(alignment_points_iterations[level]):
                # Go through a circle with radius 1 and compare the matches with the center match.
                circle_1 = Miscellaneous.circle_around(0, 0, 1)
                deviation_min_1, dy_min_1, dx_min_1  = 1.e30, None, None
                for (dy, dx) in circle_1:
                    dy_strided = dy_min + dy * stride
                    dx_strided = dx_min + dx * stride
                    if level == 0:
                        deviation = abs(
                            reference_box - frame[y_low - dy_strided:y_high - dy_strided:stride,
                                            x_low - dx_strided:x_high - dx_strided:stride]).sum()
                    else:
                        deviation = abs(
                            reference_box - GaussianBlur(
                                frame[y_low - dy_strided:y_high - dy_strided:stride,
                                x_low - dx_strided:x_high - dx_strided:stride],
                                (alignment_points_noise_levels[level],
                                 alignment_points_noise_levels[level]), 0)).sum()
                    # If the match is better, update the optimal point.
                    if deviation < deviation_min_1:
                        deviation_min_1, dy_min_1, dx_min_1 = deviation, dy_strided, dx_strided

                if deviation_min_1 < deviation_min:
                    deviation_min, dy_min, dx_min = deviation_min_1, dy_min_1, dx_min_1
                else:
                    break

            # Include this level's shift in the total warp shift.
            dy_warp += dy_min
            dx_warp += dx_min
            de_warp_shifts[level] = [dy_warp, dx_warp]

        # Return the warp shifts for all levels and remaining deviation.
        return de_warp_shifts, deviation_min

    @staticmethod
    def search_local_match_full(reference_box, frame, y_low, y_high, x_low, x_high,
                                    search_width,
                                    sampling_stride, dev_table):
        """
        For all shifts with -search_width < shift < search_width in y, x between the box around the
        alignment point in the mean frame and the corresponding box in the given frame compute the
        deviation. Return the y and x offsets where the deviation is smallest. The global frame
        shift is accounted for beforehand already.

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
        :param dev_table: Scratch table to be used internally for storing intermediate results,
                          size: [2*search_width+1, 2*search_width+1], dtype=float32.
        :return: ([shift_y, shift_x], [min_r]) with:
                   shift_y, shift_x: shift values of minimum or [0, 0] if no optimum could be found.
                   [dev_r]: list of minimum deviations for all steps until a local minimum is found.
        """

        # Set up a table which keeps deviation values from earlier iteration steps. This way,
        # deviation evaluations can be avoided at coordinates which have been visited before.
        # Initialize deviations with an impossibly high value.
        dev_table[:, :] = 1.e30

        if sampling_stride != 1:
            for dy in range(-search_width, search_width + 1):
                for dx in range(-search_width, search_width + 1):
                    dev_table[search_width + dy, search_width + dx] = abs(
                        reference_box[::sampling_stride, ::sampling_stride] - frame[
                                              y_low - dy:y_high - dy:sampling_stride,
                                              x_low - dx:x_high - dx:sampling_stride]).sum()
        else:
            for dy in range(-search_width, search_width + 1):
                for dx in range(-search_width, search_width + 1):
                    dev_table[search_width + dy, search_width + dx] = abs(
                        reference_box - frame[y_low - dy:y_high - dy, x_low - dx:x_high - dx]).sum()

        dy, dx = unravel_index(argmin(dev_table, axis=None), dev_table.shape)

        # Return the coordinate shifts for the minimum position and the value at that position.
        return [dy-search_width, dx-search_width], dev_table[dy, dx]

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

        shape_y, shape_x = frame.shape[0:2]

        if color == 'white':
            rgb = [255, 255, 255]
        elif color == 'red':
            rgb = [255, 0, 0]
        elif color == 'green':
            rgb = [0, 255, 0]
        elif color == 'blue':
            rgb = [0, 0, 255]
        elif color == 'cyan':
            rgb = [0, 255, 255]
        else:
            rgb = [255, 255, 255]

        # Be sure not to draw beyond frame boundaries.
        if 0 <= x_center < shape_x:
            for y in range(y_center - cross_half_len, y_center + cross_half_len + 1):
                if 0 <= y < shape_y:
                    frame[y, x_center] = rgb
        if 0 <= y_center < shape_y:
            for x in range(x_center - cross_half_len, x_center + cross_half_len + 1):
                if 0 <= x < shape_x:
                    frame[y_center, x] = rgb

    @staticmethod
    def multilevel_correlation(reference_box_first_phase, frame_first_phase, blurr_strength_first_phase,
                               reference_box_second_phase, frame_blurred_second_phase, y_low,
                               y_high, x_low, x_high, shift_y_global, shift_x_global,
                               search_width):
        """

        :param reference_box_first_phase:
        :param frame_first_phase:
        :param blurr_strength_first_phase:
        :param reference_box_second_phase:
        :param frame_blurred_second_phase:
        :param y_low:
        :param y_high:
        :param x_low:
        :param x_high:
        :param shift_y_global:
        :param shift_x_global:
        :param search_width:
        :return: (shift_y_local_first_phase, shift_x_local_first_phase, success_first_phase,
                  shift_y_local_second_phase, shift_x_local_second_phase, success_second_phase)
                  with:
                  shift_y_local_first_phase:
                  shift_x_local_first_phase:
                  success_first_phase:
                  shift_y_local_second_phase:
                  shift_x_local_second_phase:
                  success_second_phase:
        """

        search_width_second_phase = 4
        search_width_first_phase = int((search_width - search_width_second_phase) / 2)
        index_extension = search_width_first_phase * 2

        frame_window_first_phase = GaussianBlur(frame_first_phase[
            y_low + shift_y_global - index_extension:y_high + shift_y_global + index_extension:2,
            x_low + shift_x_global - index_extension:x_high + shift_x_global + index_extension:2],
            (blurr_strength_first_phase, blurr_strength_first_phase), 0)

        result = matchTemplate((frame_window_first_phase / 256).astype(uint8),
                               reference_box_first_phase, TM_CCORR_NORMED)

        minVal, maxVal, minLoc, maxLoc = minMaxLoc(result)
        shift_y_local_first_phase = (maxLoc[1] - search_width_first_phase) * 2
        shift_x_local_first_phase = (maxLoc[0] - search_width_first_phase) * 2

        success_first_phase = abs(shift_y_local_first_phase) != index_extension and abs(
            shift_x_local_first_phase) != index_extension
        if not success_first_phase:
            shift_y_local_first_phase = shift_x_local_first_phase = 0

        y_lo = y_low + shift_y_global + shift_y_local_first_phase - search_width_second_phase
        y_hi = y_high + shift_y_global + shift_y_local_first_phase + search_width_second_phase
        x_lo = x_low + shift_x_global + shift_x_local_first_phase - search_width_second_phase
        x_hi = x_high + shift_x_global + shift_x_local_first_phase + search_width_second_phase

        frame_window_second_phase = frame_blurred_second_phase[y_lo:y_hi, x_lo:x_hi]

        result = matchTemplate((frame_window_second_phase / 256).astype(uint8),
                               reference_box_second_phase, TM_CCORR_NORMED)

        minVal, maxVal, minLoc, maxLoc = minMaxLoc(result)
        shift_y_local_second_phase = maxLoc[1] - search_width_second_phase
        shift_x_local_second_phase = maxLoc[0] - search_width_second_phase

        success_second_phase = abs(shift_y_local_second_phase) != search_width_second_phase and abs(
                shift_x_local_second_phase) != search_width_second_phase
        if not success_second_phase:
            shift_y_local_second_phase = shift_x_local_second_phase = 0

        return shift_y_local_first_phase, shift_x_local_first_phase, success_first_phase, \
               shift_y_local_second_phase, shift_x_local_second_phase, success_second_phase

    @staticmethod
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
                   [dev_r]: list of optimum correlations for all steps until a local optimum is found.
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

    @staticmethod
    def compose_image(image_list, scale_factor=1, border=5):
        """
        Arrange a list of monochrome images horizontally in a single image, with constant gaps in
        between. If images are of different size, center-align them vertically. Optionally, the
        resulting image can be re-scaled with the same factor in x and y directions.

        :param image_list: List containing images (all of the same dtype)
        :param scale_factor: Scaling factor for the resulting composite image
        :param border: Width of black border and gaps between images
        :return: Composite image of the same dtype as the image_list items
        """

        shapes_y = [image.shape[0] for image in image_list]
        shapes_x = [image.shape[1] for image in image_list]
        max_shape_y = max(shapes_y)
        sum_shape_x = sum(shapes_x)

        # Check if all images of the list are of the same type.
        type = image_list[0].dtype
        for image in image_list[1:]:
            if image.dtype != type:
                raise ArgumentError("Trying to compose images of different types")

        # Compute the dimensions of the composite image. The border and the gaps between images are
        # 5 pixels wide.
        composite_dim_y = max_shape_y + 2 * border
        composite_dim_x = sum_shape_x + border * (len(image_list) + 1)

        # Allocate the composite image.
        composite = full((composite_dim_y, composite_dim_x), int(mean(image_list[0])), dtype=type)

        # Copy the images into the composite image.
        x_pos = border
        for index, image in enumerate(image_list):
            y_pos = int((composite_dim_y - shapes_y[index]) / 2)
            composite[y_pos: y_pos + shapes_y[index], x_pos: x_pos + shapes_x[index]] = image
            x_pos += shapes_x[index] + border

        # Return the resized composite image.
        return resize(composite, None, fx=float(scale_factor), fy=float(scale_factor))

    @staticmethod
    def circle_around(y, x, r):
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
            yield (y, x)

        # For the general case, set the first element and circle around (y, x) counter-clockwise.
        j, i = y - r, x - r
        while i < x + r:
            i += 1
            yield (j, i)
        while j < y + r:
            j += 1
            yield (j, i)
        while i > x - r:
            i -= 1
            yield (j, i)
        while j > y - r:
            j -= 1
            yield (j, i)

    @staticmethod
    def write_video(name, frame_list, annotations, fps, depth=8):
        """
        Create a video file from a list of frames.

        :param name: File name of the video output
        :param frame_list: List of frames to be written
        :param annotations: List of text strings to be written into the corresponding frame
        :param fps: Frames per second of video
        :param depth: Bit depth of the image "frame", either 8 or 16.
        :return: -
        """

        # Delete the output file if it exists.
        try:
            unlink(name)
        except:
            pass

        # Compute video frame size.
        frame_height = frame_list[0].shape[0]
        frame_width = frame_list[0].shape[1]

        # Define the codec and create VideoWriter object
        fourcc = VideoWriter_fourcc('D', 'I', 'V', 'X')
        out = VideoWriter(name, fourcc, fps, (frame_width, frame_height))

        # Define font for annotations.
        font = FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (20, 50)
        fontScale = 1
        fontColor = (255, 255, 255)
        fontThickness = 1
        lineType = LINE_AA

        # For each frame: If monochrome, convert it to three-channel color mode. insert annotation
        # and write the frame.
        for index, frame in enumerate(frame_list):
            # If the frames are 16bit, convert them to 8bit.
            if depth == 8:
                frame_8b = frame
            else:
                frame_8b = (frame / 255.).astype(uint8)

            if len(frame.shape) == 2:
                rgb_frame = stack((frame_8b,) * 3, -1)
            else:
                rgb_frame = frame_8b
            putText(rgb_frame, annotations[index],
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        fontThickness,
                        lineType)
            out.write(rgb_frame)
        out.release()

    @staticmethod
    def gaussian_sharpen(input_image, amount, radius, luminance_only=False):
        """
        Sharpen an image with a Gaussian kernel. The input image can be B/W or color.

        :param input_image: Input image, type uint16
        :param amount: Amount of sharpening
        :param radius: Radius of Gaussian kernel (in pixels)
        :param luminance_only: True, if only the luminance channel of a color image is to be
                               sharpened. Default is False.
        :return: The sharpened image (B/W or color, as input), type uint16
        """

        color = len(input_image.shape) == 3

        # Translate the kernel radius into standard deviation.
        sigma = radius / 3

        # Convert the image to floating point format.
        image = input_image.astype(float32)

        # Special case: Only sharpen the luminance channel of a color image.
        if color and luminance_only:
            hsv = cvtColor(image, COLOR_BGR2HSV)
            luminance = hsv[:, :, 2]

            # Apply a Gaussian blur filter, subtract it from the original image, and add a multiple
            # of this correction to the original image. Clip values out of range.
            luminance_blurred = GaussianBlur(luminance, (0, 0), sigma, borderType=BORDER_DEFAULT)
            hsv[:, :, 2] = (luminance + amount * (luminance - luminance_blurred)).clip(min=0.,
                                                                                       max=65535.)
            # Convert the image back to uint16.
            return cvtColor(hsv, COLOR_HSV2BGR).astype(uint16)
        # General case: Treat the entire image (B/W or color 16bit mode).
        else:
            image_blurred = GaussianBlur(image, (0, 0), sigma, borderType=BORDER_DEFAULT)
            return (image + amount * (image - image_blurred)).clip(min=0., max=65535.).astype(
                uint16)

    @staticmethod
    def gaussian_blur(input_image, amount, radius, luminance_only=False):
        """
        Soften an image with a Gaussian kernel. The input image can be B/W or color.

        :param input_image: Input image, type uint16
        :param amount: Amount of blurring, between 0. and 1.
        :param radius: Radius of Gaussian kernel (in pixels)
        :param luminance_only: True, if only the luminance channel of a color image is to be
                               blurred. Default is False.
        :return: The blurred image (B/W or color, as input), type uint16
        """

        color = len(input_image.shape) == 3

        # Translate the kernel radius into standard deviation.
        sigma = radius / 3

        # Convert the image to floating point format.
        image = input_image.astype(float32)

        # Special case: Only blur the luminance channel of a color image.
        if color and luminance_only:
            hsv = cvtColor(image, COLOR_BGR2HSV)
            luminance = hsv[:, :, 2]

            # Apply a Gaussian blur filter, subtract it from the original image, and add a multiple
            # of this correction to the original image. Clip values out of range.
            luminance_blurred = GaussianBlur(luminance, (0, 0), sigma, borderType=BORDER_DEFAULT)
            hsv[:, :, 2] = (luminance_blurred*amount + luminance*(1.-amount)).clip(min=0.,
                                                                                       max=65535.)
            # Convert the image back to uint16.
            return cvtColor(hsv, COLOR_HSV2BGR).astype(uint16)
        # General case: Treat the entire image (B/W or color 16bit mode).
        else:
            image_blurred = GaussianBlur(image, (0, 0), sigma, borderType=BORDER_DEFAULT)
            return (image_blurred*amount + image*(1.-amount)).clip(min=0., max=65535.).astype(
                uint16)

    @staticmethod
    def wavelet_sharpen(input_image, amount, radius):
        """
        Sharpen a B/W or color image with wavelets. The underlying algorithm was taken from the
        Gimp wavelet plugin, originally written in C and published under the GPLv2+ license at:
        https://github.com/mrossini-ethz/gimp-wavelet-sharpen/blob/master/src/wavelet.c

        :param input_image: Input image (B/W or color), type uint16
        :param amount: Amount of sharpening
        :param radius: Radius in pixels
        :return: Sharpened image, same format as input image
        """

        height, width = input_image.shape[:2]
        color = len(input_image.shape) == 3

        # Allocate workspace: Three complete images, plus 1D object with length max(row, column).
        if color:
            fimg = empty((3, height, width, 3), dtype=float32)
            temp = zeros((max(width, height), 3), dtype=float32)
        else:
            fimg = empty((3, height, width), dtype=float32)
            temp = zeros(max(width, height), dtype=float32)

        # Convert input image to floats.
        fimg[0] = input_image / 65535

        # Start with level 0. Store its Laplacian on level 1. The operator is separated in a
        # column and a row operator.
        hpass = 0
        for lev in range(5):
            # Highpass and lowpass levels use image indices 1 and 2 in alternating mode to save
            # space.
            lpass = ((lev & 1) + 1)

            if color:
                for row in range(height):
                    Miscellaneous.mexican_hat_color(temp, fimg[hpass][row, :, :], width, 1 << lev)
                    fimg[lpass][row, :, :] = temp[:width, :] * 0.25
                for col in range(width):
                    Miscellaneous.mexican_hat_color(temp, fimg[lpass][:, col, :], height, 1 << lev)
                    fimg[lpass][:, col, :] = temp[:height, :] * 0.25
            else:
                for row in range(height):
                    Miscellaneous.mexican_hat(temp, fimg[hpass][row, :], width, 1 << lev)
                    fimg[lpass][row, :] = temp[:width] * 0.25
                for col in range(width):
                    Miscellaneous.mexican_hat(temp, fimg[lpass][:, col], height, 1 << lev)
                    fimg[lpass][:, col] = temp[:height] * 0.25

            # Compute the amount of the correction at the current level.
            amt = amount * exp(-(lev - radius) * (lev - radius) / 1.5) + 1.

            fimg[hpass] -= fimg[lpass]
            fimg[hpass] *= amt

            # Accumulate all corrections in the first workspace image.
            if hpass:
                fimg[0] += fimg[hpass]

            hpass = lpass

        # At the end add the coarsest level and convert back to 16bit integer format.
        fimg[0] = ((fimg[0] + fimg[lpass]) * 65535.).clip(min=0., max=65535.)
        return fimg[0].astype(uint16)

    @staticmethod
    def mexican_hat(temp, base, size, sc):
        """
        Apply a 1D strided second derivative to a row or column of a B/W image. Store the result
        in the temporary workspace "temp".

        :param temp: Workspace (type float32), length at least "size" elements
        :param base: Input image (B/W), Type float32
        :param size: Length of image row / column
        :param sc: Stride (power of 2) of operator
        :return: -
        """

        # Special case at begin of row/column. Full operator not applicable.
        temp[:sc] = 2 * base[:sc] + base[sc:0:-1] + base[sc:2 * sc]
        # Apply the full operator.
        temp[sc:size - sc] = 2 * base[sc:size - sc] + base[:size - 2 * sc] + base[2 * sc:size]
        # Special case at end of row/column. The full operator is not applicable.
        temp[size - sc:size] = 2 * base[size - sc:size] + base[size - 2 * sc:size - sc] + \
                               base[size - 2:size - 2 - sc:-1]

    @staticmethod
    def mexican_hat_color(temp, base, size, sc):
        """
        Apply a 1D strided second derivative to a row or column of a color image. Store the result
        in the temporary workspace "temp".

        :param temp: Workspace (type float32), length at least "size" elements (first dimension)
                     times 3 colors (second dimension).
        :param base: Input image (color), Type float32
        :param size: Length of image row / column
        :param sc: Stride (power of 2) of operator
        :return: -
        """

        # Special case at begin of row/column. Full operator not applicable.
        temp[:sc, :] = 2 * base[:sc, :] + base[sc:0:-1, :] + base[sc:2 * sc, :]
        # Apply the full operator.
        temp[sc:size - sc, :] = 2 * base[sc:size - sc, :] + base[:size - 2 * sc, :] + base[
                                2 * sc:size, :]
        # Special case at end of row/column. The full operator is not applicable.
        temp[size - sc:size, :] = 2 * base[size - sc:size, :] + base[size - 2 * sc:size - sc, :] + \
                                  base[size - 2:size - 2 - sc:-1, :]

    @staticmethod
    def protocol(string, logfile, precede_with_timestamp=True):
        """
        Print a message (optionally plus time stamp) to standard output. If it is requested to store
        a logfile with the stacked image, write the string to that file as well.

        :param string: Message to be printed after the time stamp
        :param logfile: logfile or None (no logging)
        :return: -
        """

        # Precede the text with a time stamp and print it to stdout. Note that stdout may be
        # redirected to a file.
        if precede_with_timestamp:
            output_string = '{0} {1}'.format(datetime.now().strftime("%H-%M-%S.%f")[:-5], string)
        else:
            output_string = string
        print (output_string)
        stdout.flush()

        # If a logfile per stacked image was requested, write the string to that file as well.
        if logfile:
            logfile.write(output_string + "\n")

    @staticmethod
    def print_postproc_parameters(layers, logfile):
        """
        Print a table with postprocessing layer info for the selected postprocessing version.

        :return: -
        """

        output_string = "\n           Postprocessing method: " + layers[0].postproc_method + "\n\n" + \
                        "           Layer    |    Radius    |   Amount   |   Luminance only   |\n" \
                        "           -----------------------------------------------------------" \
                        "\n           "

        # Extend the three table lines up to the max index.
        for index, layer in enumerate(layers):
            output_string += " {0:3d}     |     {1:5.2f}    |    {2:5.2f}   |      {3:8s}      |" \
                 "\n           ".format(index + 1, layer.radius, layer.amount, str(layer.luminance_only))

        Miscellaneous.protocol(output_string, logfile, precede_with_timestamp=False)


if __name__ == "__main__":

    # Test program for routine search_local_match

    # Set the size of the image frame.
    frame_height = 960
    frame_width = 1280

    # Initialize the frame with a wave-like pattern in x and y directions.
    x_max = 30.
    x_vec = arange(0., x_max, x_max / frame_width)
    y_max = 25.
    y_vec = arange(0., y_max, y_max / frame_height)
    frame = empty((frame_height, frame_width))
    for y_j, y in enumerate(y_vec):
        for x_i, x in enumerate(x_vec):
            frame[y_j, x_i] = sin(y) * sin(x)

    # Set the size and location of the reference frame window and cut it out from the frame.
    window_height = 40
    window_width = 50
    reference_y_low = 400
    reference_x_low = 500
    reference_box = frame[reference_y_low:reference_y_low + window_height,
                    reference_x_low:reference_x_low + window_width]

    # Set the true displacement vector to be checked against the result of the search function.
    displacement_y = 1
    displacement_x = -2

    # The start point for the local search is offset from the true matching point.
    y_low = reference_y_low + displacement_y
    y_high = y_low + window_height
    x_low = reference_x_low + displacement_x
    x_high = x_low + window_width

    # Set the radius of the search area.
    search_width = 20
    sampling_stride = 1

    dev_table = empty((2 * search_width + 1, 2 * search_width + 1), dtype=float32)

    # First try the standard method "search_local_match". Compute the displacement vector, and
    # print a comparison of the true and computed values.
    start = time()
    rep_count = 1
    for iter in range(rep_count):
        [dy, dx], dev_r = Miscellaneous.search_local_match(reference_box, frame, y_low, y_high,
                                                           x_low, x_high, search_width,
                                                           sampling_stride, sub_pixel=False)
    end = time()
    print("Standard method, true displacements: " + str([displacement_y, displacement_x]) +
          ", computed: " + str([dy, dx]) + ", execution time (s): " + str((end - start) / rep_count))

    # As a comparison, do the same using the steepest descent method.
    start = time()
    for iter in range(rep_count):
        [dy, dx], dev_r = Miscellaneous.search_local_match_gradient(reference_box, frame,
                                                                    y_low, y_high,
                                                                    x_low, x_high, search_width,
                                                                    sampling_stride, dev_table)
    end = time()
    print("steepest descent, true displacements: " + str([displacement_y, displacement_x]) +
          ", computed: " + str([dy, dx]) + ", execution time (s): " + str((end - start) / rep_count))

    # Now test the alternative method in comparison to the straight-forward one. First,
    # compute the reference frame box stack. The initialization has to be performed only once for
    # each alignment point. It can be reused for all frames containing the AP.
    start = time()
    for iter in range(rep_count):
        reference_stack, displacements, radius_start = \
            Miscellaneous.search_local_match_init(frame, reference_y_low,
                                                  reference_y_low + window_height, reference_x_low,
                                                  reference_x_low + window_width, search_width)
    end = time()
    print("Match initialization time (s): " + str((end - start) / rep_count))

    # Compute the displacement. The result should be the same as above.
    start = time()
    for iter in range(rep_count):
        [dy, dx], dev_r = Miscellaneous.search_local_match_execute(
            frame[y_low:y_high, x_low:x_high], reference_stack, displacements,
            radius_start)
    end = time()
    print("Match execution, true displacements: " + str(
        [displacement_y, displacement_x]) + ", computed: " + str(
        [dy, dx]) + ", execution time (s): " + str((end - start) / rep_count))
