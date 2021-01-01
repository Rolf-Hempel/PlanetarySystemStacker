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
from time import time, sleep

from cv2 import CV_32F, Laplacian, VideoWriter_fourcc, VideoWriter, FONT_HERSHEY_SIMPLEX, LINE_AA, \
    putText, GaussianBlur, cvtColor, COLOR_BGR2HSV, COLOR_HSV2BGR, BORDER_DEFAULT, meanStdDev,\
    resize, matchTemplate, minMaxLoc, TM_CCORR_NORMED, bilateralFilter, INTER_CUBIC
from numpy import abs as np_abs
from numpy import diff, average, hypot, sqrt, unravel_index, argmax, zeros, arange, array, matmul, \
    empty, argmin, stack, sin, uint8, float32, uint16, full
from math import exp
from numpy import min as np_min
from numpy.fft import fft2, ifft2
from numpy.linalg import solve
from scipy.ndimage import sobel

from exceptions import DivideByZeroError, ArgumentError, Error


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
        dx = diff(frame)
        dy = diff(frame, axis=0)

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
        mask = frame > black_threshold
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
    def multilevel_correlation(reference_box_first_phase, frame_mono_blurred,
                               blurr_strength_first_phase, reference_box_second_phase,
                               y_low, y_high, x_low, x_high, search_width,
                               weight_matrix_first_phase=None, subpixel_solve=False):
        """
        Determine the local warp shift at an alignment point using a multi-level approach based on
        normalized cross correlation. The first level uses a pixel grid which is coarser by a factor
        of two in both coordinate directions. The second level uses the original pixel grid.
        An additional noise reduction is applied on the first level as chosen by the corresponding
        blurring parameter.

        The global search width is split between the two phases: A fixed search width of 4 is used
        in the second phase (only for local corrections). Therefore, a width of (search_width-4)
        in each coordinate direction remains for the first phase.

        In both phases it is determined if the optimum is attained on the border of the search
        area. If so, the correlation is regarded as unsuccessful (because the real optimum may be
        outside of the search area).

        :param reference_box_first_phase: Image box with stride 2 around alignment point in the
                                          locally sharpest frame. A Gaussian filter with strength
                                          "blurr_strength_first_phase" has been applied.
        :param frame_mono_blurred: Given frame (stride 1) for which the local shift at the alignment
                                   point is to be computed. This is the Gaussian blurred version of
                                   the monochrome frame as computed in class "frames".
        :param blurr_strength_first_phase: Additional Gaussian blur strength to be applied to
                                           images in the first phase.
        :param reference_box_second_phase: Image box with stride 1 around alignment point in the
                                           locally sharpest frame.

        :param y_low: Lower y coordinate limit of box in given frame, taking into account the
                      global shift and the different sizes of the mean frame and the original
                      frames.
        :param y_high: Upper y coordinate limit.
        :param x_low: Lower x coordinate limit.
        :param x_high: Upper x coordinate limit.
        :param search_width: Maximum distance in y and x from origin of the search area. (See the
                             comment in the explanatory text above.)
        :param weight_matrix_first_phase: This parameter (if not None) may give a weighting array
                                          by which the cross correlation results are multiplied
                                          before the maximum value is determined. The size of this
                                          2D array in each coordinate direction is that of the
                                          reference_box_first_phase plus two times the first phase
                                          search width (see below).
        :param subpixel_solve: If True, in the second phase the optimum is computed with
                               sub-pixel accuracy (i.e. the returned shifts are not integer).
                               If False, shifts are computed as integer values.

        :return: (shift_y_local_first_phase, shift_x_local_first_phase, success_first_phase,
                  shift_y_local_second_phase, shift_x_local_second_phase, success_second_phase)
                  with:
                  shift_y_local_first_phase: Local y warp shift determined in first phase
                                             (expressed in terms of the original pixel grid).
                  shift_x_local_first_phase: Local x warp shift determined in first phase.
                  success_first_phase: "True" if the optimum was attained in the interior of the
                                       search domain. "False" otherwise.
                  shift_y_local_second_phase: Local y warp shift determined in second phase.
                  shift_x_local_second_phase: Local x warp shift determined in second phase.
                  success_second_phase: "True" if the optimum was attained in the interior of the
                                       search domain. "False" otherwise.
        """

        # The optimization in the second phase is only meant for local fine grid adjustments.
        search_width_second_phase = 4

        # In the first phase the largest part of the local warp is detected. Divide the search
        # width by two because the first phase uses a coarser pixel grid.
        search_width_first_phase = int((search_width - search_width_second_phase) / 2)

        # Define a window around the alignment point box. Extend the window in each coordinate
        # direction. This defines the search space for the template matching. Coarsen the grid
        # by a factor of two and apply an additional Gaussian blur.
        index_extension = search_width_first_phase * 2
        frame_window_first_phase = GaussianBlur(frame_mono_blurred[
                                                y_low - index_extension:y_high + index_extension:2,
                                                x_low - index_extension:x_high + index_extension:2],
                                                (blurr_strength_first_phase,
                                                 blurr_strength_first_phase), 0)

        # Compute the normalized cross correlation.
        result = matchTemplate((frame_window_first_phase).astype(float32),
                               reference_box_first_phase.astype(float32), TM_CCORR_NORMED)

        # Determine the position of the maximum correlation and compute the corresponding warp
        # shift. The factor of 2 transforms the shift to the fine pixel grid. If a non-trivial
        # weight matrix is specified, multiply the results before looking for the maximum position.
        if weight_matrix_first_phase is not None:
            minVal, maxVal, minLoc, maxLoc = minMaxLoc(result * weight_matrix_first_phase)
        else:
            minVal, maxVal, minLoc, maxLoc = minMaxLoc(result)
        shift_y_local_first_phase = (search_width_first_phase - maxLoc[1]) * 2
        shift_x_local_first_phase = (search_width_first_phase - maxLoc[0]) * 2

        # The first phase is regarded as successful if the maximum correlation was attained in the
        # interior of the search space.
        success_first_phase = abs(shift_y_local_first_phase) != index_extension and abs(
            shift_x_local_first_phase) != index_extension

        # If the first phase was successful, add a second phase with a local search on the finest
        # (original) pixel grid.
        if success_first_phase:

            # Define the search window for the second phase. Apply the shift found in the
            # first phase.
            y_lo = y_low - shift_y_local_first_phase - search_width_second_phase
            y_hi = y_high - shift_y_local_first_phase + search_width_second_phase
            x_lo = x_low - shift_x_local_first_phase - search_width_second_phase
            x_hi = x_high - shift_x_local_first_phase + search_width_second_phase

            # Cut out the frame window for the second phase (fine grid) correlation.
            frame_window_second_phase = frame_mono_blurred[y_lo:y_hi, x_lo:x_hi]

            # Perform the template matching on the fine grid, again using normalized cross
            # correlation.
            result = matchTemplate(frame_window_second_phase.astype(float32),
                                   reference_box_second_phase, TM_CCORR_NORMED)

            # Find the position of the local optimum and compute the corresponding shift values.
            minVal, maxVal, minLoc, maxLoc = minMaxLoc(result)
            shift_y_local_second_phase = search_width_second_phase - maxLoc[1]
            shift_x_local_second_phase = search_width_second_phase - maxLoc[0]

            # Again, the second phase is deemed successful only if the optimum was attained in the
            # interior of the search space.
            success_second_phase = abs(
                shift_y_local_second_phase) != search_width_second_phase and abs(
                shift_x_local_second_phase) != search_width_second_phase

            # If the second phase was not successful, set the corresponding shifts to zero.
            if not success_second_phase:
                shift_y_local_second_phase = shift_x_local_second_phase = 0
            # The following code computes the sub-pixel shift correction to be used in drizzling.
            elif subpixel_solve:
                # Cut a 3x3 window around the optimum from the matching results.
                surroundings = result[maxLoc[1]-1:maxLoc[1]+2, maxLoc[0]-1:maxLoc[0]+2]
                try:
                    # Compute the correction to the center position. Only if the found sub-pixel
                    # correction is within the 3x3 box, it is trusted (and used).
                    y_corr, x_corr = Miscellaneous.sub_pixel_solve(surroundings)
                    if abs(y_corr) <= 1. and abs(x_corr) <= 1.:
                        shift_y_local_second_phase -= y_corr
                        shift_x_local_second_phase -= x_corr
                except:
                    # print ("Subpixel solve not successful")
                    pass

        # If the first phase was unsuccessful, drop the second phase and set all warp shifts to 0.
        else:
            success_second_phase = False
            shift_y_local_first_phase = shift_x_local_first_phase = shift_y_local_second_phase = \
                shift_x_local_second_phase = 0

        return shift_y_local_first_phase, shift_x_local_first_phase, success_first_phase, \
               shift_y_local_second_phase, shift_x_local_second_phase, success_second_phase

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

    # Define the matrix used in method "sub_pixel_solve" to solve the normal equations. It is
    # computed once and for all as "inv(a_transpose * a) * a_transpose". Using this matrix, the
    # solution of the normal equations is reduced to a simple matrix multiplication.
    sub_pixel_solve_matrix = [
        [0.16666667, -0.33333333, 0.16666667, 0.16666667, -0.33333333, 0.16666667, 0.16666667,
         -0.33333333, 0.16666667],
        [0.16666667, 0.16666667, 0.16666667, -0.33333333, -0.33333333, -0.33333333, 0.16666667,
         0.16666667, 0.16666667], [0.25, 0., -0.25, 0., 0., 0., -0.25, 0., 0.25],
        [-0.16666667, 0., 0.16666667, -0.16666667, 0., 0.16666667, -0.16666667, 0., 0.16666667],
        [-0.16666667, -0.16666667, -0.16666667, 0., 0., 0., 0.16666667, 0.16666667, 0.16666667],
        [-0.11111111, 0.22222222, -0.11111111, 0.22222222, 0.55555556, 0.22222222, -0.11111111,
         0.22222222, -0.11111111]]

    @staticmethod
    def sub_pixel_solve(function_values):
        """
        Compute the sub-pixel correction for method "search_local_match".

        :param function_values: Matching differences at (3 x 3) pixels around the minimum / maximum
                                found
        :return: Corrections in y and x to the center position for local minimum / maximum
        """

        # If the functions are not yet reduced to 1D, do it now.
        function_values_1d = function_values.reshape((9,))

        # Solve for parameters of the fitting function:
        # f = a_f * x ** 2 + b_f * y ** 2 + c_f * x * y + d_f * x + e_f * y + g_f
        # using normal equations. The problem is reduced to a matrix multiplication with the fixed
        # system matrix defined above.
        a_f, b_f, c_f, d_f, e_f, g_f = matmul(Miscellaneous.sub_pixel_solve_matrix, function_values_1d)
        # print("\nSolve, coeffs: " + str((a_f, b_f, c_f, d_f, e_f, g_f)))

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
    def sub_pixel_solve_old(function_values):
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
        :return: ([shift_y, shift_x], [min_r]) with:
                   shift_y, shift_x: shift values of minimum or [0, 0] if no optimum could be found.
                   [dev_r]: list of minimum deviations for all steps until a local minimum is found.
        """

        # Set up a table which keeps deviation values from earlier iteration steps. This way,
        # deviation evaluations can be avoided at coordinates which have been visited before.
        # Initialize deviations with an impossibly high value.
        dev_table[:, :] = 1.e30

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
        while max(abs(dy_min), abs(dx_min)) < search_width-1:

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
    def auto_rgb_align(input_image, max_shift, interpolation_factor=1, reduce_output=True,
                       blur_strength=None):
        """
        Align the three color channels of an RGB image automatically. For sub-pixel resolution the
        image can be interpolated before the shift is measured. Optionally, a Gaussian blur can be
        added to suppress noise.

        If no matching shift can be found within the range given by "max_shift", an Exception of
        type Error is thrown.

        :param input_image: Three-channel RGB image.
        :param max_shift: Maximal displacement between channels.
        :param interpolation_factor: Scaling factor (integer) for subpixel measurements.
        :param reduce_output: If True, the corrected image is reduced to the input resolution.
                              If False, it stays at the interpolated resolution. In this case the
                              output values for correction_red and correction_blue are in terms of
                              the interpolated resolution as well.
        :param blur_strength: Optional blur strength, must be an uneven integer > 0.
        :return: (corrected_image, correction_red, correction_blue) with:
                 corrected_image: The corrected image with the same datatype as the input image.
                                  Please note that the size may be reduced because of channel
                                  mismatch at the borders.
                 correction_red:  Tuple (shift_y, shift_x) with coordinate shifts in y and x
                                  applied to the red channel of input_image to produce the
                                  corrected_image.
                 correction_blue: Tuple (shift_y, shift_x) with coordinate shifts in y and x
                                  applied to the blue channel of input_image to produce the
                                  corrected_image.
        """

        # sleep(2.)
        # Immediately return for monochrome input.
        if len(input_image.shape) != 3:
            return input_image

        # If subpixel resolution is asked for, interpolate the input image first.
        if interpolation_factor != 1:
            input_interpolated = Miscellaneous.shift_colors(input_image, (0, 0), (0, 0),
                                                            interpolate_input=interpolation_factor)
        else:
            input_interpolated = input_image

        # Measure the shifts of the red and blue channels, respectively, with respect to the green
        # channel.
        channel_green = 1
        channel_red = 0
        shift_red = Miscellaneous.measure_rgb_shift(input_interpolated, channel_red,
                                                    channel_green, max_shift * interpolation_factor,
                                                    blur_strength=blur_strength)
        channel_blue = 2
        shift_blue = Miscellaneous.measure_rgb_shift(input_interpolated, channel_blue,
                                                     channel_green, max_shift*interpolation_factor,
                                                     blur_strength=blur_strength)

        # Reverse the shift measured in the input image.
        if reduce_output:
            factor = interpolation_factor
        else:
            factor = 1
        return Miscellaneous.shift_colors(input_interpolated, (-shift_red[0], -shift_red[1]),
               (-shift_blue[0], -shift_blue[1]), reduce_output=factor), \
               (-shift_red[0]/factor, -shift_red[1]/factor), \
               (-shift_blue[0]/factor, -shift_blue[1]/factor)


    @staticmethod
    def shift_colors(input_image, shift_red, shift_blue, interpolate_input=1, reduce_output=1):
        """
        Shift the red and blue channel of a color image in y and x direction, and leave the green
        channel unchanged. The shift is sub-pixel accurate if the input is interpolated. Optionally
        the resulting image is reduced in size by a given factor.

        :param input_image: Three-channel RGB image.
        :param shift_red: Tuple (shift_y, shift_x) with shifts in y and x direction for the red
                          channel. After multiplication with the interpolate_input factor (if given)
                          the shifts are rounded to the next integer value. This enables sub-pixel
                          shifts at the original image scale.
        :param shift_blue: Tuple (shift_y, shift_x) with shifts in y and x direction for the blue
                           channel.
        :param interpolate_input: If set, it must be an integer >= 1. The input image is
                                  interpolated in both directions by this factor before the shift is
                                  applied.
        :param reduce_output: If set, it must be an integer >= 1. The intermediate image is reduced
                              in size by this factor. Please note that interpolate_input =
                              reduce_output assures that the input and output images are the same
                              size.
        :return: Three-channel RGB image with the color shifts applied.
        """

        # sleep(1.)
        # Immediately return for monochrome input.
        if len(input_image.shape) != 3:
            return input_image

        # If all shifts are zero, nothing is to be done except interpolation / reduction.
        if not (shift_red[0] or shift_red[1] or shift_blue[0] or shift_blue[1]):
            # If the factors for interpolation and reduction are the same, nothing is to be done.
            if not (interpolate_input - reduce_output) or not reduce_output:
                return input_image
            # Simple resizing. reduce_output is not zero (see above)!
            else:
                scale_factor = float(interpolate_input) / float(reduce_output)
                return resize(input_image, (round(input_image.shape[1] * scale_factor),
                                round(input_image.shape[0] * scale_factor)),
                                interpolation=INTER_CUBIC)

        # If interpolation is requested, resize the input image and multiply the shift values.
        if interpolate_input != 1:
            dim_y = input_image.shape[0] * interpolate_input
            dim_x = input_image.shape[1] * interpolate_input
            interp_image = resize(input_image, (dim_x, dim_y), interpolation=INTER_CUBIC)
            s_red_y = round(interpolate_input * shift_red[0])
            s_red_x = round(interpolate_input * shift_red[1])
            s_blue_y = round(interpolate_input * shift_blue[0])
            s_blue_x = round(interpolate_input * shift_blue[1])
        else:
            dim_y = input_image.shape[0]
            dim_x = input_image.shape[1]
            interp_image = input_image
            s_red_y = round(shift_red[0])
            s_red_x = round(shift_red[1])
            s_blue_y = round(shift_blue[0])
            s_blue_x = round(shift_blue[1])

        # Remember the data type of the input image. The output image will be of the same type.
        type = input_image.dtype

        # In the following, for both the red and blue channels the index areas are computed for
        # which no shift is beyond the image dimensions. First treat the y coordinate.
        y_low_source_r = -s_red_y
        y_high_source_r = dim_y - s_red_y
        y_low_target_r = 0
        # If the shift reaches beyond the frame, reduce the copy area.
        if y_low_source_r < 0:
            y_low_target_r = -y_low_source_r
            y_low_source_r = 0
        if y_high_source_r > dim_y:
            y_high_source_r = dim_y
        y_high_target_r = y_low_target_r + y_high_source_r - y_low_source_r

        y_low_source_b = -s_blue_y
        y_high_source_b = dim_y - s_blue_y
        y_low_target_b = 0
        # If the shift reaches beyond the frame, reduce the copy area.
        if y_low_source_b < 0:
            y_low_target_b = -y_low_source_b
            y_low_source_b = 0
        if y_high_source_b > dim_y:
            y_high_source_b = dim_y
        y_high_target_b = y_low_target_b + y_high_source_b - y_low_source_b

        # The same for the x coordinate.
        x_low_source_r = -s_red_x
        x_high_source_r = dim_x - s_red_x
        x_low_target_r = 0
        # If the shift reaches beyond the frame, reduce the copy area.
        if x_low_source_r < 0:
            x_low_target_r = -x_low_source_r
            x_low_source_r = 0
        if x_high_source_r > dim_x:
            x_high_source_r = dim_x
        x_high_target_r = x_low_target_r + x_high_source_r - x_low_source_r

        x_low_source_b = -s_blue_x
        x_high_source_b = dim_x - s_blue_x
        x_low_target_b = 0
        # If the shift reaches beyond the frame, reduce the copy area.
        if x_low_source_b < 0:
            x_low_target_b = -x_low_source_b
            x_low_source_b = 0
        if x_high_source_b > dim_x:
            x_high_source_b = dim_x
        x_high_target_b = x_low_target_b + x_high_source_b - x_low_source_b

        # Now the coordinate bounds can be computed for which the output image has entries in all
        # three channels. The green channel is not shifted and can just be copied in place.
        min_y_g = max(y_low_target_r, y_low_target_b)
        max_y_g = min(y_high_target_r, y_high_target_b)
        min_x_g = max(x_low_target_r, x_low_target_b)
        max_x_g = min(x_high_target_r, x_high_target_b)

        # Allocate space for the output image (still not reduced in size), and copy the three color
        # channels with the appropriate shifts applied.
        output_image_interp = empty((max_y_g - min_y_g, max_x_g - min_x_g, 3), dtype=type)
        output_image_interp[:, :, 0] = interp_image[min_y_g - s_red_y:max_y_g - s_red_y,
                                       min_x_g - s_red_x:max_x_g - s_red_x, 0]
        output_image_interp[:, :, 1] = interp_image[min_y_g:max_y_g, min_x_g:max_x_g, 1]
        output_image_interp[:, :, 2] = interp_image[min_y_g - s_blue_y:max_y_g - s_blue_y,
                                       min_x_g - s_blue_x:max_x_g - s_blue_x, 2]

        # If output size reduction is specified, resize the intermediate image.
        if reduce_output != 1:
            output_image = resize(output_image_interp,
                                  (round(output_image_interp.shape[1] / reduce_output),
                                   round(output_image_interp.shape[0] / reduce_output)),
                                  interpolation=INTER_CUBIC)
        else:
            output_image = output_image_interp

        return output_image

    @staticmethod
    def measure_rgb_shift(image, channel_id, reference_id, max_shift, blur_strength=None):
        """
        Measure the shift between two color channels of a three-color RGB image. Before measuring
        the shift with cross correlation, a Gaussian blur can be applied to both channels to reduce
        noise.

        :param image: Input image (3 channel RGB).
        :param channel_id: Index of the channel, the shift of which is to be measured against the
                           reference channel.
        :param reference_id: Index of the reference channel, usually 1 for "green".
        :param max_shift: Maximal search space radius in pixels.
        :param blur_strength: Strength of the Gaussian blur to be applied first.
        :return: Tuple (shift_y, shift_x) with coordinate shifts in y and x.
        """

        # Test for invalid channel ids.
        if not (0 <= channel_id <= 2):
            raise ArgumentError("Invalid color channel id: " + str(channel_id))
        if not (0 <= reference_id <= 2):
            raise ArgumentError("Invalid reference channel id: " + str(reference_id))

        # Reduce the size of the window for which the correlation will be computed to make space for
        # the search space around it.
        channel_window = image[max_shift:-max_shift, max_shift:-max_shift, channel_id]
        channel_reference = image[:, :, reference_id]

        # Apply Gaussian blur if requested.
        if blur_strength:
            channel_blurred = GaussianBlur(channel_window, (blur_strength, blur_strength), 0)
            channel_reference_blurred = GaussianBlur(channel_reference,
                                                     (blur_strength, blur_strength), 0)
        else:
            channel_blurred = channel_window
            channel_reference_blurred = channel_reference

        # Compute the normalized cross correlation.
        result = matchTemplate((channel_reference_blurred).astype(float32),
                               channel_blurred.astype(float32), TM_CCORR_NORMED)

        # Determine the position of the maximum correlation, and compute the corresponding shifts.
        minVal, maxVal, minLoc, maxLoc = minMaxLoc(result)
        shift_y = max_shift - maxLoc[1]
        shift_x = max_shift - maxLoc[0]
        if abs(shift_y) != max_shift and abs(shift_x) != max_shift:
            return (shift_y, shift_x)
        else:
            raise Error ("No matching shift found within given range")

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
        composite = full((composite_dim_y, composite_dim_x), 0, dtype=type)

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
    def post_process(image, layers):
        """
        Apply all postprocessing layers to the input image "image". If the image is in color mode,
        the postprocessing is computed either in BGR mode or on the luminance channel only. All
        computations are performed in 32bit floating point mode.

        :param image: Input image, either BGR or Grayscale (16bit uint).
        :param layers: postprocessing layers with all parameters.
        :return: Processed image in the same 16bit uint format as the input image.
        """

        # Check if the original image is selected (version 0). In this case nothing is to be done.
        if not layers:
            return image.astype(uint16)

        # Convert the image to 32bit floating point format.
        input_image = image.astype(float32)

        color = len(image.shape) == 3

        # If the luminance channel is to be processed only, extract the luminance channel.
        if color and layers[0].luminance_only:
            input_image_hsv = cvtColor(input_image, COLOR_BGR2HSV)
            layer_input = input_image_hsv[:, :, 2]
        else:
            layer_input = input_image

        # Go through all layers and apply the sharpening filters.
        for layer_index, layer in enumerate(layers):

            # Bilateral filter is needed:
            if abs(layer.bi_fraction) > 1.e-5:
                layer_bilateral = bilateralFilter(layer_input, 0, layer.bi_range * 256.,
                                                  layer.radius / 3., borderType=BORDER_DEFAULT)
            # Gaussian filter is needed:
            if abs(layer.bi_fraction - 1.) > 1.e-5:
                layer_gauss = GaussianBlur(layer_input, (0, 0), layer.radius / 3.,
                                           borderType=BORDER_DEFAULT)

            # Compute the input for the next layer. First case: bilateral only.
            if abs(layer.bi_fraction - 1.) <= 1.e-5:
                next_layer_input = layer_bilateral
            # Case Gaussian only.
            elif abs(layer.bi_fraction) <= 1.e-5:
                next_layer_input = layer_gauss
            # Mixed case.
            else:
                next_layer_input = layer_bilateral * layer.bi_fraction + \
                                   layer_gauss * (1. - layer.bi_fraction)

            layer_component_before_denoise = layer_input - next_layer_input

            # If denoising is chosen for this layer, apply a Gaussian filter.
            if layer.denoise > 1.e-5:
                layer_component = GaussianBlur(layer_component_before_denoise, (0, 0),
                    layer.radius / 3., borderType=BORDER_DEFAULT) * layer.denoise + \
                    layer_component_before_denoise * (1. - layer.denoise)
            else:
                layer_component = layer_component_before_denoise

            # Accumulate the contributions from all layers. On the first layer initialize the
            # summation buffer.
            if layer_index:
                components_accumulated += layer_component * layer.amount
            else:
                components_accumulated = layer_component * layer.amount

            layer_input = next_layer_input

        # After all layers are accumulated, finally add the maximally blurred image.
        components_accumulated += next_layer_input

        # Reduce the value range so that they fit into 16bit uint, and convert to uint16. In case
        # the luminance channel was processed only, insert the processed luminance channel into the
        # HSV representation of the original image, and convert back to BGR.
        if color and layers[0].luminance_only:
            input_image_hsv[:, :, 2] = components_accumulated
            return cvtColor(input_image_hsv.clip(min=0., max=65535.), COLOR_HSV2BGR).astype(uint16)
        else:
            return components_accumulated.clip(min=0., max=65535.).astype(uint16)

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
        print(output_string)
        stdout.flush()

        # If a logfile per stacked image was requested, write the string to that file as well.
        if logfile:
            logfile.write(output_string + "\n")

    @staticmethod
    def print_stacking_parameters(configuration, logfile):
        """
        Print a table with all stacking parameters.

        :param configuration: Object holding all configuration parameters.
        :param logfile: logfile or None (no logging)
        :return: -
        """

        # Compile all parameters and their values to be printed.
        parameters = [["Debayering default", configuration.frames_debayering_default],
                      ["Debayering method", configuration.frames_debayering_method],
                      ["Noise level (add Gaussian blur)", str(configuration.frames_gauss_width)],
                      ["Frame stabilization mode", configuration.align_frames_mode]]

        # The following parameters are only active in "Surface" mode.
        if configuration.align_frames_mode == "Surface":
            parameters = parameters + [
                ["Automatic frame stabilization", str(configuration.align_frames_automation)],
                ["Stabilization patch size (% of frame size)",
                 str(int(round(100. / configuration.align_frames_rectangle_scale_factor)))],
                ["Stabilization search width (pixels)",
                 str(configuration.align_frames_search_width)]]

        # Continue with general parameters.
        parameters = parameters + [["Percentage of best frames for reference frame computation",
                                    str(configuration.align_frames_average_frame_percent)],
                                   ["Object is changing fast (e.g. Jupiter, Sun)",
                                    str(configuration.align_frames_fast_changing_object)],
                                   ["Alignment box width (pixels)",
                                    str(2 * configuration.alignment_points_half_box_width)],
                                   ["Max. alignment search width (pixels)",
                                    str(configuration.alignment_points_search_width)],
                                   ["Minimum structure",
                                    str(configuration.alignment_points_structure_threshold)],
                                   ["Minimum brightness",
                                    str(configuration.alignment_points_brightness_threshold)],
                                   ["Percentage of best frames to be stacked",
                                    str(configuration.alignment_points_frame_percent)],
                                   ["Normalize frame brightness", str(
                                           configuration.frames_normalization)]
                                   ]

        # If brightness normalization is checked, add the black cut-off value.
        if configuration.frames_normalization:
            parameters = parameters + [
                ["Normalization black cut-off", str(configuration.frames_normalization_threshold)]]

        # If drizzling is active, add the factor.
        if configuration.stack_frames_drizzle_factor_string != "Off":
            parameters = parameters + [
                ["Drizzle factor in stacking", str(configuration.stack_frames_drizzle_factor_string)]]

        output_string = "\n           Stacking parameters:                                         | Value                        |\n" \
                        "           ---------------------------------------------------------------------------------------------" \
                        "\n          "

        # Extend the output string with a line for every parameter to be printed.
        for line in parameters:
            output_string += " {0:60s} | {1:29s}|\n          ".format(line[0], line[1])

        # Write the complete table.
        Miscellaneous.protocol(output_string, logfile, precede_with_timestamp=False)

    @staticmethod
    def print_postproc_parameters(postproc_version, logfile):
        """
        Print a table with postprocessing layer info for the selected postprocessing version.

        :param postproc_version: Object holding postprocessing parameters for the selected version.
        :param logfile: logfile or None (no logging)
        :return: -
        """

        # Test if an RGB correction has been applied.
        if postproc_version.shift_red != (0., 0.) or postproc_version.shift_blue != (0., 0.):
            # Find out if the RGB correction was done automatically.
            if postproc_version.rgb_automatic:
                intro = "           Automatic RGB correction, "
            else:
                intro = "           Manual RGB correction, "
            (shift_red_y, shift_red_x) = postproc_version.shift_red
            (shift_blue_y, shift_blue_x) = postproc_version.shift_blue

            n_digits = [0, 1, 2][postproc_version.rgb_resolution_index]
            if shift_red_y >= 0.:
                dir_red_y = " pixels down"
            else:
                dir_red_y = " pixels up"
            if shift_red_x >= 0.:
                dir_red_x = " pixels right"
            else:
                dir_red_x = " pixels left"
            if shift_blue_y >= 0.:
                dir_blue_y = " pixels down"
            else:
                dir_blue_y = " pixels up"
            if shift_blue_x >= 0.:
                dir_blue_x = " pixels right"
            else:
                dir_blue_x = " pixels left"

            # Special case 0 digits: In this case the number of digits must be omitted.
            # If the round function is called with "n_digits=0", the result still has one digit
            # after the decimal point.
            if n_digits:
                Miscellaneous.protocol(
                    intro + "red channel shifted " +
                    str(round(abs(shift_red_y), n_digits)) + dir_red_y + ", " +
                    str(round(abs(shift_red_x), n_digits)) + dir_red_x + ", blue channel shifted " +
                    str(round(abs(shift_blue_y), n_digits)) + dir_blue_y + ", " +
                    str(round(abs(shift_blue_x), n_digits)) + dir_blue_x + ".",
                    logfile, precede_with_timestamp=False)
            else:
                Miscellaneous.protocol(
                    intro + "red channel shifted " +
                    str(round(abs(shift_red_y))) + dir_red_y + ", " +
                    str(round(abs(shift_red_x))) + dir_red_x + ", blue channel shifted " +
                    str(round(abs(shift_blue_y))) + dir_blue_y + ", " +
                    str(round(abs(shift_blue_x))) + dir_blue_x + ".",
                    logfile, precede_with_timestamp=False)

        if postproc_version.layers:
            output_string = "           Postprocessing method: " + postproc_version.layers[0].postproc_method + "\n\n" + \
                            "           Layer    |    Radius    |   Amount   |   Bi fraction (%)   |   Bi range   |   Denoise (%)   |   Luminance only   |\n" \
                            "           ------------------------------------------------------------------------------------------------------------------" \
                            "\n           "

            # Extend the three table lines up to the max index.
            for index, layer in enumerate(postproc_version.layers):
                output_string += " {0:3d}     |     {1:5.2f}    |   {2:6.2f}   |         {3:4.0f}        |    {4:5.1f}     |       {5:4.0f}      |       {6:8s}     |" \
                     "\n           ".format(index + 1, layer.radius, layer.amount, layer.bi_fraction*100., layer.bi_range, layer.denoise*100., str(layer.luminance_only))

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
