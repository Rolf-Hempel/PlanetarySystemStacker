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

import cv2
import os
from numpy import sqrt, average, diff, sum, hypot, arange, zeros, unravel_index, argmax, array, \
    matmul, stack, empty, sin
from numpy.fft import fft2, ifft2
from numpy.linalg import solve
from scipy.ndimage import sobel
from time import time

from exceptions import DivideByZeroError


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
        summation only takes into account points where the luminosity exceeds a certain
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
    def local_contrast_laplace(frame, stride):
        """
        Measure the amount of structure in a rectangular frame in both coordinate directions and
        return the minimum value. The discrete variance of the Laplacian is computed.

        :param frame: 2D image
        :param stride: Factor for down-sampling
        :return: Measure for amount of local structure in the image (scalar)
        """

        sharpness = cv2.Laplacian(frame[::stride, ::stride], cv2.CV_32F).var()
        # sharpness = sum(laplace(frame[::stride, ::stride])**2)
        return sharpness

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
    def search_local_match(reference_box, frame, y_low, y_high, x_low, x_high, search_width,
                           sub_pixel=True):
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
        :param search_width: maximum radius of the search spiral
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
            for (dx, dy) in circle_r:
                deviation = abs(
                    reference_box - frame[y_low - dy:y_high - dy, x_low - dx:x_high - dx]).sum()
                # deviation = sqrt(square(
                #     reference_box - frame[y_low - dy:y_high - dy, x_low - dx:x_high - dx]).sum())
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

    @staticmethod
    def write_video(name, frame_list, annotations, fps):
        """
        Create a video file from a list of frames.

        :param name: File name of the video output
        :param frame_list: List of frames to be written
        :param annotations: List of text strings to be written into the corresponding frame
        :param fps: Frames per second of video
        :return: -
        """

        # Delete the output file if it exists.
        try:
            os.unlink(name)
        except:
            pass

        # Compute video frame size.
        frame_height = frame_list[0].shape[0]
        frame_width = frame_list[0].shape[1]

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        out = cv2.VideoWriter(name, fourcc, fps, (frame_width, frame_height))

        # Define font for annotations.
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (20, 50)
        fontScale = 1
        fontColor = (255, 255, 255)
        fontThickness = 1
        lineType = cv2.LINE_AA

        # For each frame: If monochrome, convert it to three-channel color mode. insert annotation
        # and write the frame.
        for index, frame in enumerate(frame_list):
            # If
            if len(frame.shape) == 2:
                rgb_frame = stack((frame,) * 3, -1)
            else:
                rgb_frame = frame
            cv2.putText(rgb_frame, annotations[index],
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        fontThickness,
                        lineType)
            out.write(rgb_frame)
        out.release()


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
    displacement_y = 6
    displacement_x = -5

    # The start point for the local search is offset from the true matching point.
    y_low = reference_y_low + displacement_y
    y_high = y_low + window_height
    x_low = reference_x_low + displacement_x
    x_high = x_low + window_width

    # Set the radius of the search area.
    search_width = 20

    # compute the displacement vector, and print a comparison of the true and computed values.
    start = time()
    rep_count = 100
    for iter in range(rep_count):
        [dy, dx], dev_r = Miscellaneous.search_local_match(reference_box, frame, y_low, y_high,
                                                           x_low, x_high, search_width,
                                                           sub_pixel=False)
    end = time()

    print("True displacements: " + str([displacement_y, displacement_x]) + ", computed: " + str(
        [dy, dx]) + ", execution time (s): " + str((end-start)/rep_count))
