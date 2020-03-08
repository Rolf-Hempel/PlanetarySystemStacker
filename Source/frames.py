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

from glob import glob
from math import ceil
from os import path, remove, listdir, stat
from os.path import splitext
from pathlib import Path
from time import time

from PyQt5 import QtCore
from astropy.io import fits
from cv2 import imread, VideoCapture, CAP_PROP_FRAME_COUNT, cvtColor, COLOR_RGB2GRAY, \
    COLOR_BGR2RGB, COLOR_BayerGB2BGR, COLOR_BayerBG2BGR, THRESH_TOZERO, threshold, \
    GaussianBlur, Laplacian, CV_32F, COLOR_RGB2BGR, imwrite, convertScaleAbs, CAP_PROP_POS_FRAMES, \
    IMREAD_UNCHANGED, flip, COLOR_GRAY2RGB, COLOR_BayerRG2BGR, COLOR_BayerGR2BGR
from cv2 import mean as cv_mean
from numpy import max as np_max
from numpy import min as np_min
from numpy import sum as np_sum
from numpy import uint8, uint16, int32, float32, clip, zeros, float64, where, average, moveaxis, \
    unravel_index, ndarray

import ser_parser
from configuration import Configuration
from exceptions import TypeError, ShapeError, ArgumentError, WrongOrderingError, Error, \
    InternalError
from frames_old import FramesOld


def debayer_frame(frame_in, debayer_pattern='No change', BGR_input=False):
    """
    Process a given input frame "frame_in", either containing one layer (B/W) or three layers
    (color) into an output frame "frame_out" as specified by the parameter "debayer_pattern".

    The rules for this transformation are:
    - If the "debayer_pattern" is "No change", the input frame is not changed, i.e. the
      output frame is identical to the input frame. The same applies if the input frame is of type
       "B/W" and the "debayer_pattern" is "Grayscale".
    - If the input frame is of type "color" and "debayer_pattern" is "Grayscale", the RGB / BGR
      image is converted into a B/W one.
    - If the input frame is of type "color", the "debayer_pattern" is "BGR" and "BGR_input" is
     'False', the "B" and "R" channels are exchanged. The same happens if "debayer_pattern" is "RGB"
      and "BGR_input" is 'True'. For the other two combinations the channels are not exchanged.
    - If the input frame is of type "Grayscale" and "debayer_pattern" is "RGB" or "BGR", the result
      is a three-channel RGB / BGR image where all channels are the same.
    - If a non-standard "debayer_pattern" (i.e. "RGGB", "GRBG", "GBRG", "BGGR") is specified and the
      input is a B/W image, decode the image using the given Bayer pattern. If the input image is
      of type "color" (three-channel RGB or BGR), first convert it into grayscale and then decode
      the image as in the B/W case. In both cases the result is a three-channel RGB color image.

    :param frame_in: Input image, either 2D (grayscale) or 3D (color). The type is either 8 or 16
                     bit unsigned int.
    :param debayer_pattern: Pattern used to convert the input image into the output image. One out
                            of 'Grayscale', 'RGB', 'Force Bayer RGGB', 'Force Bayer GRBG',
                            'Force Bayer GBRG', 'Force Bayer BGGR'
    :param BGR_input: If 'True', a color input frame is interpreted as 'BGR'; Otherwise as 'RGB'.
                      OpenCV reads color images in 'BGR' format.
    :return: frame_out: output image (see above)
    """

    debayer_codes = {
        'Force Bayer RGGB': COLOR_BayerRG2BGR,
        'Force Bayer GRBG': COLOR_BayerGR2BGR,
        'Force Bayer GBRG': COLOR_BayerGB2BGR,
        'Force Bayer BGGR': COLOR_BayerBG2BGR
    }

    type_in = frame_in.dtype

    if type_in != uint8 and type_in != uint16:
        raise Exception("Image type " + str(type_in) + " not supported")

    # If the input frame is 3D, it represents a color image.
    color_in = len(frame_in.shape) == 3

    # Case color input image.
    if color_in:
        # Three-channel input, interpret as RGB color and leave it unchanged.
        if debayer_pattern == 'No change' or debayer_pattern == 'RGB' and not BGR_input or \
                debayer_pattern == 'BGR' and BGR_input:
            frame_out = frame_in

        # If the Bayer pattern and the BGR_input flag don't match, flip channels.
        elif debayer_pattern == 'RGB' and BGR_input or debayer_pattern == 'BGR' and not BGR_input:
            frame_out = cvtColor(frame_in, COLOR_BGR2RGB)

        # Three-channel (color) input, reduce to two-channel (B/W) image.
        elif debayer_pattern in ['Grayscale', 'Force Bayer RGGB', 'Force Bayer GRBG',
                                 'Force Bayer GBRG', 'Force Bayer BGGR']:

            frame_2D = cvtColor(frame_in, COLOR_RGB2GRAY)

            # Output is B/W image.
            if debayer_pattern == 'Grayscale':
                frame_out = frame_2D

            # Decode the B/W image into a color image using a Bayer pattern.
            else:
                frame_out = cvtColor(frame_2D, debayer_codes[debayer_pattern])

        # Invalid debayer pattern specified.
        else:
            raise Exception("Debayer pattern " + debayer_pattern + " not supported")

    # Case B/W input image.
    else:
        # Two-channel input, interpret as B/W image and leave it unchanged.
        if debayer_pattern in ['No change', 'Grayscale']:
            frame_out = frame_in

        # Transform the one-channel B/W image in an RGB one where all three channels are the same.
        elif debayer_pattern == 'RGB' or debayer_pattern == 'BGR':
            frame_out = cvtColor(frame_in, COLOR_GRAY2RGB)

        # Non-standard Bayer pattern, decode into color image.
        elif debayer_pattern in ['Force Bayer RGGB', 'Force Bayer GRBG',
                                 'Force Bayer GBRG', 'Force Bayer BGGR']:
            frame_out = cvtColor(frame_in, debayer_codes[debayer_pattern])

        # Invalid Bayer pattern specified.
        else:
            raise Exception("Debayer pattern " + debayer_pattern + " not supported")

    # Return the decoded image.
    return frame_out


def detect_bayer(frame, frames_bayer_max_noise_diff_green, frames_bayer_min_distance_from_blue,
                 frames_color_difference_threshold):
    """
    Detect a Bayer pattern in a grayscale image. The assumption is that statistically the
    brightness differences at adjacent pixels are greater than those at neighboring pixels of the
    same color. It is also assumed that the noise in the blue channel is greater than in the red
    and green channels.

    Acknowledgements: This method uses an algorithm developed by Chris Garry for his 'PIPP'
                      software package.

    :param frame: Numpy array (2D or 3D) of type uint8 or uint16 containing the image data.
    :param frames_bayer_max_noise_diff_green: Maximum allowed difference in noise levels at the two
                                              green pixels of the bayer matrix in percent of value
                                              at the blue pixel.
    :param frames_bayer_min_distance_from_blue: Maximum allowed noise level at Bayer matrix pixels
                                                other than blue, in percent of value at the blue
                                                pixel.
    :param frames_color_difference_threshold: If the brightness values of all three color channels
                                              of a three-channel frame at all pixels do not differ
                                              by more than this value, the frame is regarded as
                                              monochrome.

    :return: If the input frame is a (3D) color image, 'Color' is returned.
             If the input frame is a (2D or 3D) grayscale image without any detectable Bayer
             pattern, 'Grayscale' is returned.
             If a Bayer pattern is detected in the grayscale image, its type (one out of
             'Force Bayer BGGR', 'Force Bayer GBRG', 'Force Bayer GRBG', 'Force Bayer RGGB') is
             returned.
             If none of the above is true, 'None' is returned.
    """

    # Frames are stored as 3D arrays. Test if all three color levels are the same.
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        first_minus_second = np_max(
            abs(frame[:, :, 0].astype(int32) - frame[:, :, 1].astype(int32)))
        first_minus_third = np_max(abs(frame[:, :, 0].astype(int32) - frame[:, :, 2].astype(int32)))
        # print("first_minus_second: " + str(first_minus_second) + ", first_minus_third: " + str(
        #     first_minus_third))

        # The color levels differ more than some threshold. Probably color image.
        if first_minus_second > frames_color_difference_threshold \
                or first_minus_third > frames_color_difference_threshold:
            return 'Color'

        # All three levels are the same, convert to 2D grayscale image and proceed.
        else:
            frame_grayscale = cvtColor(frame, COLOR_RGB2GRAY).astype(int32)

    # Input frame is already grayscale.
    elif len(frame.shape) == 2:
        frame_grayscale = frame.astype(int32)

    # Neither a color image nor grayscale.
    else:
        return 'None'

    try:
        height, width = frame_grayscale.shape
        analysis_height = height - height % 2
        analysis_width = width - width % 2

        # Look for signs of a bayer pattern.
        adjacent_pixel_diffs = np_sum(
            abs(frame_grayscale[:, 0:analysis_width - 2] - frame_grayscale[:,
                                                           1:analysis_width - 1]))
        apart_pixel_diffs = np_sum(
            abs(frame_grayscale[:, 0:analysis_width - 2] - frame_grayscale[:, 2:analysis_width]))

        # Pixels are more like the pixels next to them than they are like the pixels 2 pixel away.
        # This indicates that there is no bayer pattern present
        if apart_pixel_diffs > adjacent_pixel_diffs:
            return 'Grayscale'

        # Analyse noise characteristics of image to guess at positions of R, G and B in bayer pattern.
        noise_level = ndarray((2, 2), dtype=float)
        for y in range(2):
            for x in range(2):
                # Apply a five point (Poisson) stencil and sum over all points.
                neighbors = (frame_grayscale[y:analysis_height - 6 + y:2,
                             2 + x:analysis_width - 4 + x:2] +
                             frame_grayscale[4 + y:analysis_height - 2 + y:2,
                             2 + x:analysis_width - 4 + x:2] +
                             frame_grayscale[2 + y:analysis_height - 4 + y:2,
                             x:analysis_width - 6 + x:2] +
                             frame_grayscale[2 + y:analysis_height - 4 + y:2,
                             4 + x:analysis_width - 2 + x:2]) / 4.
                noise_level[y, x] = np_sum(
                    abs(frame_grayscale[2 + y:analysis_height - 4 + y:2,
                        2 + x:analysis_width - 4 + x:2] - neighbors))

        # Normalize noise levels.
        max_y, max_x = unravel_index(noise_level.argmax(), noise_level.shape)
        max_noise_level = noise_level[max_y, max_x]
        if max_noise_level > 0.:
            noise_level = noise_level * (100. / max_noise_level)
        # Zero noise - cannot detect bayer pattern.
        else:
            return 'Grayscale'

        # The location of the maximum noise level is interpreted as the blue channel.
        # It is in position (0, 0).
        if (max_y, max_x) == (0, 0):
            # The noise levels of the green pixels are too different for this to be a bayer pattern.
            if abs(noise_level[0, 1] - noise_level[1, 0]) > frames_bayer_max_noise_diff_green:
                return 'Grayscale'
            # Noise levels of the other pixels are too close to the blue values for this to definitely
            # be a bayer pattern.
            if noise_level[0, 1] > frames_bayer_min_distance_from_blue or noise_level[
                1, 0] > frames_bayer_min_distance_from_blue or noise_level[
                1, 1] > frames_bayer_min_distance_from_blue:
                return 'Grayscale'
            # Bayer pattern "BGGR" found.
            return 'Force Bayer BGGR'

        # Case "GBRG":
        elif (max_y, max_x) == (0, 1):
            if abs(noise_level[0, 0] - noise_level[1, 1]) > frames_bayer_max_noise_diff_green:
                return 'Grayscale'
            if noise_level[0, 0] > frames_bayer_min_distance_from_blue or noise_level[
                1, 0] > frames_bayer_min_distance_from_blue or noise_level[
                1, 1] > frames_bayer_min_distance_from_blue:
                return 'Grayscale'
            # Bayer pattern "GBRG" found.
            return 'Force Bayer GBRG'

        # Case "GRBG":
        elif (max_y, max_x) == (1, 0):
            if abs(noise_level[0, 0] - noise_level[1, 1]) > frames_bayer_max_noise_diff_green:
                return 'Grayscale'
            if noise_level[0, 0] > frames_bayer_min_distance_from_blue or noise_level[
                0, 1] > frames_bayer_min_distance_from_blue or noise_level[
                1, 1] > frames_bayer_min_distance_from_blue:
                return 'Grayscale'
            # Bayer pattern "GRBG" found.
            return 'Force Bayer GRBG'

        # Case "RGGB"
        elif (max_y, max_x) == (1, 1):
            if abs(noise_level[0, 1] - noise_level[1, 0]) > frames_bayer_max_noise_diff_green:
                return 'Grayscale'
            if noise_level[0, 0] > frames_bayer_min_distance_from_blue or noise_level[
                0, 1] > frames_bayer_min_distance_from_blue or noise_level[
                1, 0] > frames_bayer_min_distance_from_blue:
                return 'Grayscale'
            # Bayer pattern "RGGB" found.
            return 'Force Bayer RGGB'

        # Theoretically this cannot be reached.
        return 'None'

    # If something bad has happened, return 'None'.
    except Exception as e:
        # print("An Exception occurred: " + str(e))
        return 'None'

def detect_rgb_bgr(frame):
    """
    Given a color (3D) frame, find out if the channels are arranged as 'RGB' or 'BGR'. To this end,
    the noise level of the first and third channels are compared. The channel with the highest noise
    level is interpreted as the blue channel.

    :param frame: Numpy array (3D) of type uint8 or uint16 containing the image data.
    :return: Either 'RGB' or 'BGR', depending on whether the noise in the third or first channel is
             highest. If an error occurs, 'None' is returned.
    """

    # Not a color (3D) frame:
    if len(frame.shape) != 3:
        return 'None'

    try:
        height, width = frame.shape[0:2]
        analysis_height = height - height % 2
        analysis_width = width - width % 2

        # Analyse noise characteristics of image to guess at positions of R, G and B in bayer pattern.
        frame_int32 = frame.astype(int32)
        noise_level = [0., 0., 0.]
        for channel in [0, 2]:
            # Apply a five point (Poisson) stencil and sum over all points.
            neighbors = (frame_int32[0:analysis_height - 2, 1:analysis_width - 1, channel] +
                         frame_int32[2:analysis_height, 1:analysis_width - 1, channel] +
                         frame_int32[1:analysis_height - 1, 0:analysis_width - 2, channel] +
                         frame_int32[1:analysis_height - 1, 2:analysis_width, channel]) / 4.
            noise_level[channel] = np_sum(
                abs(frame_int32[1:analysis_height - 1, 1:analysis_width - 1, channel] - neighbors))

        # print("noise level 0:" + str(noise_level[0]) + ", noise level 2:" + str(noise_level[2]))
        if noise_level[0] > noise_level[2]:
            return 'BGR'
        else:
            return 'RGB'
    except Exception as e:
        # print("An Exception occurred: " + str(e))
        return 'None'

class VideoReader(object):
    """
    The VideoReader deals with the import of frames from a video file. Frames can be read either
    consecutively, or at an arbitrary frame index. Eventually, all common video types (such as .avi,
    .mov, .mp4, .ser) should be supported.
    """

    def __init__(self, configuration):
        """
        Create the VideoReader object and initialize instance variables.

        :param configuration: Configuration object with parameters
        """

        self.configuration = configuration
        self.last_read = None
        self.last_frame_read = None
        self.frame_count = None
        self.shape = None
        self.color = None
        self.convert_to_grayscale = False
        self.dtype = None
        self.SERFile = False
        self.shift_pixels = 0
        self.bayer_option_selected = None
        self.bayer_pattern = None
        self.BGR_input = None

    def sanity_check(self, file_path):
        """
        Performs a sanity check of input file.

        :return: -
        """

        if not path.isfile(file_path):
            raise IOError("File does not exist")
        elif stat(file_path).st_size == 0:
            raise IOError("File is empty")

    def open(self, file_path, bayer_option_selected='Auto detect color',
             SER_16bit_shift_correction=True):
        """
        Initialize the VideoReader object and return parameters with video metadata.
        Throws an IOError if the video file format is not supported.

        :param file_path: Full name of the video file.
        :param bayer_option_selected: Bayer pattern, one out of: "Auto detect color", "Grayscale",
                              "RGB", "BGR", "Force Bayer RGGB", "Force Bayer GRBG",
                               "Force Bayer GBRG", "Force Bayer BGGR".
        :param SER_16bit_shift_correction: If True and the frame type is 16bit, the video frames
                                           are analyzed to find the number of unused high bits in
                                           pixel data. In read operations data are shifted up by
                                           this number of bits.
        :return: (frame_count, color, dtype, shape) with
                 frame_count: Total number of frames in video.
                 color: True, if frames are in color; False otherwise.
                 dtype: Numpy type, either uint8 or uint16
                 shape: Tuple with the shape of a single frame; (num_px_y, num_px_x, 3) for color,
                        (num_px_y, num_px_x) for B/W.
                 shift_pixels: Number of unused (high) bits in pixel values. Frame data can be
                               left-shifted by this number without going into saturation.
        """

        # Do sanity check
        self.sanity_check(file_path)

        # Check, if file has SER extension
        self.SERFile = path.splitext(file_path)[1].lower() == '.ser'

        # Check if input file is SER file
        if self.SERFile:
            try:
                # Create the VideoCapture object.
                self.cap = ser_parser.SERParser(file_path, SER_16bit_shift_correction)
                self.shift_pixels = self.cap.shift_pixels

                # Read the first frame.
                self.last_frame_read = self.cap.read_frame_raw(0)

                # Look up video metadata.
                self.frame_count = self.cap.frame_count
                self.color_in = self.cap.color
                self.BGR_input = False
                self.dtype = self.cap.PixelDepthPerPlane
                # Set the bayer pattern. In automatic mode, use the pattern encoded in the SER
                # file header.
                if bayer_option_selected == 'Auto detect color':
                    self.bayer_pattern = self.cap.header['ColorIDDecoded']
                else:
                    self.bayer_pattern = bayer_option_selected
            except:
                raise IOError("Error in reading first video frame")
        else:
            try:
                # Create the VideoCapture object.
                self.cap = VideoCapture(file_path)

                # Read the first frame.
                ret, self.last_frame_read = self.cap.read()
                if not ret:
                    raise IOError("Error in reading first video frame")

                # Look up video metadata. Set "BGR_input" to 'True' because OpenCV reads color
                # images in BGR channel ordering. R and B channels have to be swapped because
                # PSS works with RGB color internally.
                self.frame_count = int(self.cap.get(CAP_PROP_FRAME_COUNT))
                self.color_in = (len(self.last_frame_read.shape) == 3)
                self.BGR_input = True
                self.dtype = self.last_frame_read.dtype

                # Set the bayer pattern.
                if bayer_option_selected == 'Auto detect color':
                    # Look for a Bayer pattern in the 2D or 3D data.
                    bayer_pattern_computed = detect_bayer(self.last_frame_read,
                            self.configuration.frames_bayer_max_noise_diff_green,
                            self.configuration.frames_bayer_min_distance_from_blue,
                            self.configuration.frames_color_difference_threshold)
                    # If the image was classified as 'Color', test the ordering of color channels.
                    if bayer_pattern_computed == 'Color':
                        # Analyze first frame to detect ordering of color channels. Note that the
                        # frame was read by OpenCV in BGR mode. Channels will be swapt later.
                        rgb_order = detect_rgb_bgr(self.last_frame_read)
                        if rgb_order == 'BGR':
                            self.bayer_pattern = 'RGB'
                            # print("Color channel ordering 'RGB' detected")
                        elif rgb_order == 'RGB':
                            self.bayer_pattern = 'BGR'
                            # print("Color channel ordering  'BGR' detected")
                        else:
                            self.bayer_pattern = 'RGB'
                            # print("No color channel ordering  detected, apply 'RGB'")
                    elif bayer_pattern_computed == 'Grayscale':
                        self.bayer_pattern = 'Grayscale'
                        # print("Image has been found to be grayscale")
                    elif bayer_pattern_computed == 'None':
                        if self.color_in:
                            self.bayer_pattern = 'RGB'
                            # print("No Bayer pattern detected, apply 'RGB' because 3D")
                        else:
                            self.bayer_pattern = 'Grayscale'
                            # print("No Bayer pattern detected, apply 'Grayscale' because 2D")
                    else:
                        self.bayer_pattern = bayer_pattern_computed
                        # print("Bayer pattern " + bayer_pattern_computed + " detected")

                # The user has selected a debayering mode explicitly.
                else:
                    # Leave the pattern unchanged.
                    self.bayer_pattern = bayer_option_selected
            except:
                raise IOError("Error in reading first video frame. Try to convert the video with "
                              "PIPP into some standard format")

        # Assign "last_read"
        self.last_read = 0

        # Convert the first frame read into the desired output format and set the metadata.
        self.last_frame_read = debayer_frame(self.last_frame_read,
                                             debayer_pattern=self.bayer_pattern,
                                             BGR_input=self.BGR_input)
        self.shape = self.last_frame_read.shape
        self.color = (len(self.shape) == 3)

        # Return the metadata.
        return self.frame_count, self.color, self.dtype, self.shape, self.shift_pixels

    def read_frame(self, index=None):
        """
        Read a single frame from the video.

        :param index: Frame index (optional). If no index is specified, the next frame is read.
        :return: Numpy array containing the frame. For B/W, the shape is (num_px_y, num_px_x).
                 For a color video, it is (num_px_y, num_px_x, 3). The type is uint8 or uint16 for
                 8 or 16 bit resolution.
        """

        # Check the index.
        if index is None:
            # Consecutive reading. Just increment the frame pointer.
            self.last_read += 1
        elif index == self.last_read:
            # An index is the same as at last call, just return the last frame.
            return self.last_frame_read
        else:
            # If it is the next frame after the one read last time, for AVI videos the frame pointer
            # does not have to be set. The read_frame method of the "ser_parser" module always does
            # a seek operation.
            if not self.SERFile and index != self.last_read + 1:
                self.cap.set(CAP_PROP_POS_FRAMES, index)
            self.last_read = index

        # A new frame has to be read. First check if the index is not out of bounds.
        if 0 <= self.last_read < self.frame_count:
            try:
                # Read the next frame.
                if self.SERFile:
                    self.last_frame_read = self.cap.read_frame_raw(self.last_read)
                else:
                    ret, self.last_frame_read = self.cap.read()
                    if not ret:
                        raise IOError("Error in reading video frame, index: {0}. Try to convert the video with "
                              "PIPP into some standard format".format(index))
            except:
                raise IOError("Error in reading video frame, index: {0}. Try to convert the video with "
                              "PIPP into some standard format".format(index))
        else:
            raise ArgumentError("Error in reading video frame, index {0} is out of bounds".format(index))

        # Convert the frame read into the desired output format.
        self.last_frame_read = debayer_frame(self.last_frame_read,
                                             debayer_pattern=self.bayer_pattern,
                                             BGR_input=self.BGR_input)

        return self.last_frame_read

    def close(self):
        """
        Close the VideoReader object.

        :return:
        """

        self.cap.release()


class ImageReader(object):
    """
    The ImageReader deals with the import of frames from a list of single images. Frames can
    be read either consecutively, or at an arbitrary frame index. It is assumed that the
    lexicographic order of file names corresponds to their chronological order.
    """

    def __init__(self, configuration):
        """
        Create the ImageReader object and initialize instance variables.

        :param configuration: Configuration object with parameters.
        """

        self.configuration = configuration
        self.opened = False
        self.just_opened = False
        self.last_read = None
        self.last_frame_read = None
        self.frame_count = None
        self.shape = None
        self.color = None
        self.convert_to_grayscale = False
        self.dtype = None
        self.bayer_pattern = None
        self.shift_pixels = 0

    def open(self, file_path_list, bayer_option_selected='Auto detect color'):
        """
        Initialize the ImageReader object and return parameters with image metadata.

        :param file_path_list: List with path names to the image files.
        :param bayer_option_selected: Bayer pattern, one out of: "Auto detect color", "Grayscale",
                              "RGB", "BGR", "Force Bayer RGGB", "Force Bayer GRBG",
                               "Force Bayer GBRG", "Force Bayer BGGR".
                                This parameter is ignored for now.
        :return: (frame_count, color, dtype, shape) with
                 frame_count: Total number of frames.
                 color: True, if frames are in color; False otherwise.
                 dtype: Numpy type, either uint8 or uint16
                 shape: Tuple with the shape of a single frame; (num_px_y, num_px_x, 3) for color,
                        (num_px_y, num_px_x) for B/W.
                 shift_pixels: Number of unused (high) bits in pixel values. Frame data can be
                               left-shifted by this number without going into saturation.
        """

        self.file_path_list = file_path_list
        self.bayer_option_selected = bayer_option_selected
        self.bayer_pattern = bayer_option_selected

        try:
            self.frame_count = len(self.file_path_list)

            self.last_frame_read = Frames.read_image(self.file_path_list[0])

            if self.convert_to_grayscale:
                self.last_frame_read = cvtColor(self.last_frame_read, COLOR_RGB2GRAY)

            # Look up metadata.
            self.last_read = 0
            self.shape = self.last_frame_read.shape
            self.color = (len(self.shape) == 3)
            self.dtype = self.last_frame_read.dtype
        except Exception as ex:
            raise IOError("Reading first frame: " + str(ex))

        self.opened = True
        self.just_opened = True

        # Return the metadata.
        return self.frame_count, self.color, self.dtype, self.shape, self.shift_pixels

    def read_frame(self, index=None):
        """
        Read a single frame.

        :param index: Frame index (optional). If no index is specified, the next frame is read.
                      At the first invocation, this is frame number 0.
        :return: Numpy array containing the frame. For B/W, the shape is (num_px_y, num_px_x).
                 For a color video, it is (num_px_y, num_px_x, 3). The type is uint8 or uint16 for
                 8 or 16 bit resolution.
        """

        if not self.opened:
            raise WrongOrderingError(
                "Error: Attempt to read image file frame before opening ImageReader")

        # Special case: first call after initialization.
        if self.just_opened:
            self.just_opened = False

            # Frame 0 has been read during initialization. Not necessary to read it again.
            if index is None or index == 0:
                return self.last_frame_read

        # General case: not the first call.
        else:

            # Consecutive reading. Just increment the frame index.
            if index is None:
                self.last_read += 1

            # An index is specified explicitly. If it is the same as at last call, just return the
            # last frame.
            elif index == self.last_read:
                return self.last_frame_read

            # Some other frame was specified explicitly.
            else:
                self.last_read = index

        # A new frame has to be read. First check if the index is not out of bounds.
        if 0 <= self.last_read < self.frame_count:
            try:
                self.last_frame_read = Frames.read_image(self.file_path_list[self.last_read])
                if self.convert_to_grayscale:
                    self.last_frame_read = cvtColor(self.last_frame_read, COLOR_RGB2GRAY)
            except Exception as ex:
                raise IOError("Reading image with index: " + str(index) + ", " + str(ex))
        else:
            raise ArgumentError("Reading image with index: " + str(index) +
                                ", index is out of bounds")

        # Check if the metadata match.
        shape = self.last_frame_read.shape
        color = (len(shape) == 3)

        # Check if all images have matching metadata.
        if color != self.color:
            raise ShapeError(
                "Mixing grayscale and color images not supported, index: " + str(index))
        elif shape != self.shape:
            raise ShapeError("Images have different size, index: " + str(index))
        elif self.last_frame_read.dtype != self.dtype:
            raise TypeError("Images have different type, index: " + str(index))

        return self.last_frame_read

    def close(self):
        """
        Close the ImageReader object.

        :return:
        """

        self.opened = False


class Calibration(QtCore.QObject):
    """
    This class performs the dark / flat calibration of frames. Master frames are created from
    video files or image directories. Flats, darks and the stacking input must match in terms of
    types, shapes and color modes.

    """

    report_calibration_error_signal = QtCore.pyqtSignal(str)

    def __init__(self, configuration):
        """
        Initialize the  object for dark / flat calibration.

        :param configuration: Configuration object with parameters
        """

        super(Calibration, self).__init__()
        self.configuration = configuration
        self.reset_masters()

    def reset_masters(self):
        """
        De-activate master dark and flat frames.

        :return: -
        """

        self.reset_master_dark()
        self.reset_master_flat()

        self.color = None
        self.shape = None
        self.dtype = None

    def reset_master_dark(self):
        """
        De-activate a master dark frame.

        :return: -
        """

        self.master_dark_frame = None
        self.master_dark_frame_adapted = None
        self.high_value = None
        self.dark_color = None
        self.dark_dtype = None
        self.dark_shape = None

    def reset_master_flat(self):
        """
        De-activate a master flat frame.

        :return: -
        """

        self.master_flat_frame = None
        self.inverse_master_flat_frame = None
        self.flat_color = None
        self.flat_dtype = None
        self.flat_shape = None

    def create_master(self, master_name, output_dtype=uint16):
        """
        Create a master frame by averaging a number of video frames or still images.

        :param master_name: Path name of video file or image directory.
        :param output_dtype: Data type of resulting master frame, one of:
                             - uint8 (high value = 255)
                             - uint16 (high value = 65535)
                             default: uint16
        :return: Master frame
        """

        # Case video file:
        if Path(master_name).is_file():
            extension = Path(master_name).suffix
            if extension in ('.avi', '.mov', '.mp4', '.ser'):
                reader = VideoReader(self.configuration)
                # Switch off dynamic range correction for 16bit SER files.
                frame_count, input_color, input_dtype, input_shape, shift_pixels = reader.open(master_name,
                     bayer_option_selected=self.configuration.frames_debayering_default,
                     SER_16bit_shift_correction=False)
                self.configuration.hidden_parameters_current_dir = str(Path(master_name).parent)
            else:
                raise InternalError(
                    "Unsupported file type '" + extension + "' specified for master frame "
                                                            "construction")
        # Case image directory:
        elif Path(master_name).is_dir():
            names = [path.join(master_name, name) for name in listdir(master_name)]
            reader = ImageReader(self.configuration)
            frame_count, input_color, input_dtype, input_shape, shift_pixels = reader.open(names,
                                        bayer_option_selected=self.configuration.frames_debayering_default)
            self.configuration.hidden_parameters_current_dir = str(master_name)
        else:
            raise InternalError("Cannot decide if input file is video or image directory")

        # Sum all frames in a 64bit buffer.
        master_frame_64 = zeros(input_shape, float64)
        for index in range(frame_count):
            master_frame_64 += reader.read_frame(index)

        # Return the average frame in the format specified.
        if output_dtype == input_dtype:
            return (master_frame_64 / frame_count).astype(output_dtype)
        elif output_dtype == uint8 and input_dtype == uint16:
            factor = 1. / (frame_count * 256)
            return (master_frame_64 * factor).astype(output_dtype)
        elif output_dtype == uint16 and input_dtype == uint8:
            factor = 256. / frame_count
            return (master_frame_64 * factor).astype(output_dtype)
        else:
            raise ArgumentError("Cannot convert dtype from " + str(input_dtype) + " to " +
                                str(output_dtype))

    def create_master_dark(self, dark_name, load_from_file=False):
        """
        Create a master dark image, or read it from a file.

        :param dark_name: If a new master frame is to be created, path name of video file or image
                          directory. Otherwise the file name (Tiff or Fits) of the master frame.
        :param load_from_file: True, if to be loaded from file. False, if to be created anew.
        :return: -
        """

        # Reset a master dark frame if previously allocated.
        self.reset_master_dark()

        # Create the master frame or read it from a file.
        if load_from_file:
            try:
                self.master_dark_frame = Frames.read_image(dark_name)
            except Exception as e:
                self.report_calibration_error_signal.emit("Error: " + str(e))
                return

            if self.master_dark_frame.dtype == uint8:
                self.master_dark_frame = (self.master_dark_frame * 256).astype(uint16)
        else:
            self.master_dark_frame = self.create_master(dark_name, output_dtype=uint16)

        self.shape = self.dark_shape = self.master_dark_frame.shape
        self.color = self.dark_color = (len(self.dark_shape) == 3)
        self.dark_dtype = self.master_dark_frame.dtype

        # If a flat frame has been processed already, check for consistency. If master frames do not
        # match, remove the master flat.
        if self.inverse_master_flat_frame is not None:
            if self.dark_color != self.flat_color or self.dark_shape != self.flat_shape:
                self.reset_master_flat()
                # Send a message to the main GUI indicating that a non-matching master flat is
                # removed.
                self.report_calibration_error_signal.emit(
                    "A non-matching master flat was de-activated")

    def load_master_dark(self, dark_name):
        """
        Read a master dark frame from disk and initialize its metadata.

        :param dark_name: Path name of master dark frame (type TIFF)
        :return: -
        """

        self.create_master_dark(dark_name, load_from_file=True)

    def create_master_flat(self, flat_name, load_from_file=False):
        """
        Create a master flat image, or read it from a file.

        :param flat_name: If a new master frame is to be created, path name of video file or image
                          directory. Otherwise the file name (Tiff or Fits) of the master frame.
        :param load_from_file: True, if to be loaded from file. False, if to be created anew.
        :return: -
        """

        # Reset a master flat frame if previously allocated.
        self.reset_master_flat()

        # Create the master frame or read it from a file.
        if load_from_file:
            try:
                self.master_flat_frame = Frames.read_image(flat_name)
            except Exception as e:
                self.report_calibration_error_signal.emit("Error: " + str(e))
                return

            if self.master_flat_frame.dtype == uint8:
                self.master_flat_frame = (self.master_flat_frame * 256).astype(uint16)
        else:
            self.master_flat_frame = self.create_master(flat_name, output_dtype=uint16)

        self.shape = self.flat_shape = self.master_flat_frame.shape
        self.color = self.flat_color = (len(self.flat_shape) == 3)
        self.flat_dtype = self.master_flat_frame.dtype

        # If a dark frame has been processed already, check for consistency. If master frames do not
        # match, remove the master dark.
        if self.master_dark_frame is not None:
            if self.dark_color != self.flat_color or self.dark_shape != self.flat_shape:
                self.reset_master_dark()
                # Send a message to the main GUI indicating that a non-matching master dark is
                # removed.
                self.report_calibration_error_signal.emit(
                    "A non-matching master dark was de-activated")

        average_flat_frame = average(self.master_flat_frame).astype(uint16)

        # If a new flat frame is to be constructed, apply a dark frame (if available).
        if not load_from_file:
            if self.master_dark_frame is not None:
                # If there is a matching dark frame, use it to correct the flat frame. Avoid zeros
                # in places where darks and flats are the same (hot pixels??).
                self.master_flat_frame = where(self.master_flat_frame > self.master_dark_frame,
                                               self.master_flat_frame - self.master_dark_frame,
                                               average_flat_frame)

        # Compute the inverse master flat (float32) so that its entries are close to one.
        if average_flat_frame > 0:
            self.inverse_master_flat_frame = (average_flat_frame / self.master_flat_frame).astype(
                float32)
        else:
            self.reset_master_flat()
            raise InternalError("Invalid input for flat frame computation")

    def load_master_flat(self, flat_name):
        """
        Read a master flat frame from disk and initialize its metadata.

        :param flat_name: Path name of master flat frame (type TIFF)
        :return: -
        """

        try:
            self.create_master_flat(flat_name, load_from_file=True)

        # Send a signal to the main GUI and trigger error message printing there.
        except Error as e:
            self.report_calibration_error_signal.emit(
                "Error in loading master flat: " + str(e) + ", flat correction de-activated")

    def flats_darks_match(self, color, shape):
        """
        Check if the master flat / master dark match frame attributes.

        :param color: True, if frames are in color; False otherwise.
        :param shape: Tuple with the shape of a single frame; (num_px_y, num_px_x, 3) for color,
                      (num_px_y, num_px_x) for B/W.
        :return: True, if attributes match; False otherwise.
        """

        return color == self.color and shape == self.shape

    def adapt_dark_frame(self, frame_dtype, shift_pixels):
        """
        Adapt the type of the master dark frame to the type of frames to be corrected.

        :param frame_dtype: Dtype of frames to be corrected. Either uint8 or uint16.
        :param shift_pixels: Number of unused (high) bits in pixel values. Frame data are
                             left-shifted by this number so that they fill the full dynamic range.
                             Since the dark frame is subtracted from all frames, it must be shifted
                             by the same number of bits before being applied.

                             This correction is only relevant for 16bit SER videos.
        :return: -
        """

        self.dtype = frame_dtype

        if self.master_dark_frame is None:
            self.high_value = None
            self.master_dark_frame_adapted = None
        elif frame_dtype == uint8:
            self.high_value = 255
            self.master_dark_frame_adapted = (self.master_dark_frame / 256.).astype(uint8)
        elif frame_dtype == uint16:
            self.high_value = 65535
            if shift_pixels:
                self.master_dark_frame_adapted = self.master_dark_frame << shift_pixels
            else:
                self.master_dark_frame_adapted = self.master_dark_frame

    def correct(self, frame):
        """
        Correct a stacking frame using a master dark and / or a master flat.

        :param frame: Frame to be stacked.
        :return: Frame corrected for dark/flat, same type as input frame.
        """

        # Case neither darks nor flats are available:
        if self.master_dark_frame_adapted is None and self.inverse_master_flat_frame is None:
            return frame

        # Case only flats are available:
        elif self.master_dark_frame_adapted is None:
            return (frame * self.inverse_master_flat_frame).astype(self.dtype)

        # Case only darks are available:
        elif self.inverse_master_flat_frame is None:
            return where(frame > self.master_dark_frame_adapted,
                         frame - self.master_dark_frame_adapted, 0)

        # Case both darks and flats are available:
        else:
            return clip(
                (frame - self.master_dark_frame_adapted) * self.inverse_master_flat_frame,
                0., self.high_value).astype(self.dtype)


class Frames(object):
    """
        This object stores the image data of all frames. Four versions of the original frames are
        used throughout the data processing workflow. They are (re-)used in the folliwing phases:
        1. Original (color) frames, type: uint8 / uint16
            - Frame stacking ("stack_frames.stack_frames")
        2. Monochrome version of 1., type: uint8 / uint16
            - Computing the average frame (only average frame subset, "align_frames.average_frame")
        3. Gaussian blur added to 2., type: type: uint16
            - Aligning all frames ("align_frames.align_frames")
            - Frame stacking ("stack_frames.stack_frames")
        4. Down-sampled Laplacian of 3., type: uint8
            - Overall image ranking ("rank_frames.frame_score")
            - Ranking frames at alignment points("alignment_points.compute_frame_qualities")

        Buffering at various levels is available. It is controlled with four flags set at object
        initialization time.

        A complete PSS execution processes all "n" frames in four complete passes. Additionally,
        in module "align_frames" there are some extra accesses:

        1. In "rank_frames.frame_score": Access to all "Laplacians of Gaussian" (frame 0 to n-1)
           In "align_frames.select_alignment_rect and .align_frames": Access to the Gaussian of the
           best frame.
        2. In "align_frames.align_frames": Access to all Gaussians (frame 0 to n-1)
           In "align_frames.average_frame": Access to the monochrome frames of the best images for
           averaging
        3. In "alignment_points.compute_frame_qualities": Access to all "Laplacians of Gaussian"
           (frame 0 to n-1)
        4. In "stack_frames.stack_frames": Access to all frames + Gaussians (frame 0 to n-1)

    """

    @staticmethod
    def set_buffering(buffering_level):
        """
        Decide on the objects to be buffered, depending on "buffering_level" configuration
        parameter.

        :param buffering_level: Buffering level parameter as set in configuration.
        :return: Tuple of four booleans:
                 buffer_original: Keep all original frames in buffer.
                 buffer_monochrome: Keep the monochrome version of  all original frames in buffer.
                 buffer_gaussian: Keep the monochrome version with Gaussian blur added of  all
                                  frames in buffer.
                 buffer_laplacian: Keep the Laplacian of Gaussian (LoG) of the monochrome version of
                                   all frames in buffer.
        """

        buffer_original = False
        buffer_monochrome = False
        buffer_gaussian = False
        buffer_laplacian = False

        if buffering_level > 0:
            buffer_laplacian = True
        if buffering_level > 1:
            buffer_gaussian = True
        if buffering_level > 2:
            buffer_original = True
        if buffering_level > 3:
            buffer_monochrome = True

        return buffer_original, buffer_monochrome, buffer_gaussian, buffer_laplacian

    def __init__(self, configuration, names, type='video', bayer_option_selected="Auto detect color",
                 calibration=None, progress_signal=None,
                 buffer_original=True, buffer_monochrome=False, buffer_gaussian=True,
                 buffer_laplacian=True):
        """
        Initialize the Frame object, and read all images. Images can be stored in a video file or
        as single images in a directory.

        :param configuration: Configuration object with parameters
        :param names: In case "video": name of the video file. In case "image": list of names for
                      all images.
        :param type: Either "video" or "image".
        :param bayer_option_selected: Bayer pattern, one out of: "Auto detect color", "Grayscale",
                              "RGB", "BGR", "Force Bayer RGGB", "Force Bayer GRBG",
                               "Force Bayer GBRG", "Force Bayer BGGR".
        :param calibration: (Optional) calibration object for darks/flats correction.
        :param progress_signal: Either None (no progress signalling), or a signal with the signature
                                (str, int) with the current activity (str) and the progress in
                                percent (int).
        :param buffer_original: If "True", read the original frame data only once, otherwise
                                read them again if required.
        :param buffer_monochrome: If "True", compute the monochrome image only once, otherwise
                                  compute it again if required. This may include re-reading the
                                  original image data.
        :param buffer_gaussian: If "True", compute the gaussian-blurred image only once, otherwise
                                compute it again if required. This may include re-reading the
                                original image data.
        :param buffer_laplacian: If "True", compute the "Laplacian of Gaussian" only once, otherwise
                                 compute it again if required. This may include re-reading the
                                 original image data.
        """

        self.configuration = configuration
        self.names = names
        self.calibration = calibration
        self.progress_signal = progress_signal
        self.type = type
        self.bayer_pattern = None
        self.bayer_option_selected = bayer_option_selected
        self.shift_pixels = None

        self.buffer_original = buffer_original
        self.buffer_monochrome = buffer_monochrome
        self.buffer_gaussian = buffer_gaussian
        self.buffer_laplacian = buffer_laplacian

        # In non-buffered mode, the index of the image just read/computed is stored for re-use.
        self.original_available = None
        self.original_available_index = -1
        self.monochrome_available = None
        self.monochrome_available_index = -1
        self.gaussian_available = None
        self.gaussian_available_index = -1
        self.laplacian_available = None
        self.laplacian_available_index = None

        # Set a flag that no monochrome image has been computed before.
        self.first_monochrome = True

        # Initialize the list of original frames.
        self.frames_original = None

        # Compute the scaling value for Laplacian computation.
        self.alpha = 1. / 256.

        # Initialize and open the reader object.
        if self.type == 'image':
            self.reader = ImageReader(self.configuration)
        elif self.type == 'video':
            self.reader = VideoReader(self.configuration)
        else:
            raise TypeError("Image type " + self.type + " not supported")

        self.number, self.color, self.dt0, self.shape, self.shift_pixels = self.reader.open(self.names,
            bayer_option_selected=self.bayer_option_selected)

        # Look up the Bayer pattern the reader has identified.
        self.bayer_pattern = self.reader.bayer_pattern

        # Set the depth value of all images to either 16 or 8 bits.
        if self.dt0 == 'uint16':
            self.depth = 16
        elif self.dt0 == 'uint8':
            self.depth = 8
        else:
            raise TypeError("Frame type " + str(self.dt0) + " not supported")

        # Check if the darks / flats of the calibration object match the current reader.
        if self.calibration:
            self.calibration_matches = self.calibration.flats_darks_match(self.color, self.shape)
            # If there are matching darks or flats, adapt their type to the current frame type.
            if self.calibration_matches:
                self.calibration.adapt_dark_frame(self.dt0, self.shift_pixels)
        else:
            self.calibration_matches = False

        # Initialize lists of monochrome frames (with and without Gaussian blur) and their
        # Laplacians.
        colors = ['red', 'green', 'blue', 'panchromatic']
        if self.configuration.frames_mono_channel in colors:
            self.color_index = colors.index(self.configuration.frames_mono_channel)
        else:
            raise ArgumentError("Invalid color selected for channel extraction")
        self.frames_monochrome = [None] * self.number
        self.frames_monochrome_blurred = [None] *self.number
        self.frames_monochrome_blurred_laplacian = [None] *self.number
        if self.configuration.frames_normalization:
            self.frames_average_brightness = [None] *self.number
        else:
            self.frames_average_brightness = None
        self.first_monochrome_index = None
        self.used_alignment_points = None

    def compute_required_buffer_size(self, buffering_level):
        """
        Compute the RAM required to store original images and their derivatives, and other objects
        which scale with the image size.

        Additional to the original images and their derivatives, the following large objects are
        allocated during the workflow:                                                   mono color
            calibration.master_dark_frame: pixels * colors (float32)                       4    12
            calibration.master_dark_frame_uint8: pixels * colors (uint8)                   1     3
            calibration.master_dark_frame_uint16: pixels * colors (uint16)                 2     6
            calibration.master_flat_frame: pixels * colors (float32)                       4    12
            align_frames.mean_frame: image pixels (int32)                                  4     4
            align_frames.mean_frame_original: image pixels (int32)                         4     4
            alignment_points, reference boxes: < 2 * image pixels (int32)                  8     8
            alignment_points, stacking buffers: < 2 * image pixels * colors (float32)      8    24
            stack_frames.stacked_image_buffer: image pixels * colors (float32)             4    12
            stack_frames.number_single_frame_contributions: image pixels (int32)           4     4
            stack_frames.sum_single_frame_weights: image pixels (float32)                  4     4
            stack_frames.averaged_background: image pixels * colors (float32)              4    12
            stack_frames.stacked_image: image pixels * colors (uint16)                     2     6
                                                                                          ---------
                                                                  Total (bytes / pixel):  53   111
        :param buffering_level: Buffering level parameter.
        :return: Number of required buffer space in bytes.
        """

        # Compute the number of image pixels.
        number_pixel = self.shape[0] * self.shape[1]

        # Compute the size of a monochrome image in bytes.
        image_size_monochrome_bytes = number_pixel * self.depth / 8

        # Compute the size of an original image in bytes.
        if self.color:
            image_size_bytes = 3 * image_size_monochrome_bytes
        else:
            image_size_bytes = image_size_monochrome_bytes

        # Compute the size of the monochrome images with Gaussian blur added in bytes.
        image_size_gaussian_bytes = number_pixel * 2

        # Compute the size of a "Laplacian of Gaussian" in bytes. Remember that it is down-sampled.
        image_size_laplacian_bytes = number_pixel / \
                                     self.configuration.align_frames_sampling_stride ** 2

        # Compute the buffer space per image, based on the buffering level.
        buffer_original, buffer_monochrome, buffer_gaussian, buffer_laplacian = \
            Frames.set_buffering(buffering_level)
        buffer_per_image = 0
        if buffer_original:
            buffer_per_image += image_size_bytes
        if buffer_monochrome:
            buffer_per_image += image_size_monochrome_bytes
        if buffer_gaussian:
            buffer_per_image += image_size_gaussian_bytes
        if buffer_laplacian:
            buffer_per_image += image_size_laplacian_bytes

        # Multiply with the total number of frames.
        buffer_for_all_images = buffer_per_image * self.number

        # Compute the size of additional workspace objects allocated during the workflow. For the
        # details see the comment block at the beginning of this method.
        if self.color:
            buffer_additional_workspace = number_pixel * 111
        else:
            buffer_additional_workspace = number_pixel * 53

        # Return the total buffer space required.
        return (buffer_for_all_images + buffer_additional_workspace) / 1e9

    def frames(self, index):
        """
        Read or look up the original frame object with a given index.

        :param index: Frame index
        :return: Frame with index "index".
        """

        if not 0 <= index < self.number:
            raise ArgumentError("Frame index " + str(index) + " is out of bounds")
        # print ("Accessing frame " + str(index))

        # If the original frames are to be buffered, read them in one go at the first call to this
        # method. In this case, a progress bar is displayed in the main GUI.
        if self.frames_original is None:
            if self.buffer_original:
                self.frames_original = []
                self.signal_step_size = max(int(self.number / 10), 1)
                for frame_index in range(self.number):
                    # After every "signal_step_size"th frame, send a progress signal to the main GUI.
                    if self.progress_signal is not None and frame_index % self.signal_step_size == 1:
                        self.progress_signal.emit("Read all frames",
                                                  int(round(10 * frame_index / self.number) * 10))
                    # Read the next frame. If dark/flat correction is active, do the corrections.
                    if self.calibration_matches:
                        self.frames_original.append(self.calibration.correct(
                            self.reader.read_frame(frame_index)))
                    else:
                        self.frames_original.append(self.reader.read_frame(frame_index))

                self.reader.close()

            # If original frames are not buffered, initialize an empty frame list, so frames can be
            # read later in non-consecutive order.
            else:
                self.frames_original = [None] *self.number

        # The original frames are buffered. Just return the frame.
        if self.buffer_original:
            return self.frames_original[index]

        # This frame has been cached. Just return it.
        if self.original_available_index == index:
            return self.original_available

        # The frame has not been stored for re-use, read it. If dark/flat correction is active, do
        # the corrections.
        else:
            if self.calibration_matches:
                frame = self.calibration.correct(self.reader.read_frame(index))
            else:
                frame = self.reader.read_frame(index)

            # Cache the frame just read.
            self.original_available = frame
            self.original_available_index = index

            return frame

    def frames_mono(self, index):
        """
        Look up or compute the monochrome version of the frame object with a given index.

        :param index: Frame index
        :return: Monochrome frame with index "index".
        """

        if not 0 <= index < self.number:
            raise ArgumentError("Frame index " + str(index) + " is out of bounds")

        # print("Accessing frame monochrome " + str(index))
        # The monochrome frames are buffered, and this frame has been stored before. Just return
        # the frame.
        if self.frames_monochrome[index] is not None:
            return self.frames_monochrome[index]

        # If the monochrome frame is cached, just return it.
        if self.monochrome_available_index == index:
            return self.monochrome_available

        # The frame has not been stored for re-use, compute it.
        else:

            # Get the original frame. If it is not cached, this involves I/O.
            frame_original = self.frames(index)

            # If frames are in color mode produce a B/W version.
            if self.color:
                if self.color_index == 3:
                    frame_mono = cvtColor(frame_original, COLOR_RGB2GRAY)
                else:
                    frame_mono = frame_original[:, :, self.color_index]
            # Frames are in B/W mode already
            else:
                frame_mono = frame_original

            # Normalize the overall frame brightness. The first monochrome frame for which this
            # method is invoked is taken as the reference. The average brightness of all other
            # monochrome frames is adjusted to match the brightness of the referenence.
            if self.configuration.frames_normalization:
                frame_type = frame_mono.dtype
                if self.first_monochrome:
                    if frame_type == uint8:
                        self.normalization_lower_threshold = \
                            self.configuration.frames_normalization_threshold
                        self.normalization_upper_threshold = 255
                    else:
                        self.normalization_lower_threshold = \
                            self.configuration.frames_normalization_threshold * 256
                        self.normalization_upper_threshold = 255

                    self.frames_average_brightness[index] = cv_mean(
                        threshold(frame_mono, self.normalization_lower_threshold,
                                  self.normalization_upper_threshold,
                                  THRESH_TOZERO)[1])[0] + 1.e-10
                    # Keep the index of the first monochrome frame as the reference index.
                    self.first_monochrome_index = index
                    self.first_monochrome = False
                # Not the first monochrome frame. Adjust brightness to match the reference.
                else:
                    self.frames_average_brightness[index] = cv_mean(
                        threshold(frame_mono, self.normalization_lower_threshold,
                                  self.normalization_upper_threshold,
                                  THRESH_TOZERO)[1])[0] + 1.e-10

            # If the monochrome frames are buffered, store it at the current index.
            if self.buffer_monochrome:
                self.frames_monochrome[index] = frame_mono

            # If frames are not buffered, cache the current frame.
            else:
                self.monochrome_available_index = index
                self.monochrome_available = frame_mono

            return frame_mono

    def frames_mono_blurred(self, index):
        """
        Look up a Gaussian-blurred frame object with a given index.

        :param index: Frame index
        :return: Gaussian-blurred frame with index "index".
        """

        if not 0 <= index < self.number:
            raise ArgumentError("Frame index " + str(index) + " is out of bounds")

        # print("Accessing frame with Gaussian blur " + str(index))
        # The blurred frames are buffered, and this frame has been stored before. Just return
        # the frame.
        if self.frames_monochrome_blurred[index] is not None:
            return self.frames_monochrome_blurred[index]

        # If the blurred frame is cached, just return it.
        if self.gaussian_available_index == index:
            return self.gaussian_available

        # The frame has not been stored for re-use, compute it.
        else:

            # Get the monochrome frame. If it is not cached, this involves I/O.
            frame_mono = self.frames_mono(index)

            # If the mono image is 8bit, interpolate it to 16bit.
            if frame_mono.dtype == uint8:
                frame_mono = frame_mono.astype(uint16) * 256

            # Compute a version of the frame with Gaussian blur added.
            frame_monochrome_blurred = GaussianBlur(frame_mono,
                                                    (self.configuration.frames_gauss_width,
                                                     self.configuration.frames_gauss_width), 0)

            # If the blurred frames are buffered, store the current frame at the current index.
            if self.buffer_gaussian:
                self.frames_monochrome_blurred[index] = frame_monochrome_blurred

            # If frames are not buffered, cache the current frame.
            else:
                self.gaussian_available_index = index
                self.gaussian_available = frame_monochrome_blurred

            return frame_monochrome_blurred

    def frames_mono_blurred_laplacian(self, index):
        """
        Look up a Laplacian-of-Gaussian of a frame object with a given index.

        :param index: Frame index
        :return: LoG of a frame with index "index".
        """

        if not 0 <= index < self.number:
            raise ArgumentError("Frame index " + str(index) + " is out of bounds")

        # print("Accessing LoG number " + str(index))
        # The LoG frames are buffered, and this frame has been stored before. Just return the frame.
        if self.frames_monochrome_blurred_laplacian[index] is not None:
            return self.frames_monochrome_blurred_laplacian[index]

        # If the blurred frame is cached, just return it.
        if self.laplacian_available_index == index:
            return self.laplacian_available

        # The frame has not been stored for re-use, compute it.
        else:

            # Get the monochrome frame. If it is not cached, this involves I/O.
            frame_monochrome_blurred = self.frames_mono_blurred(index)

            # Compute a version of the frame with Gaussian blur added.
            frame_monochrome_laplacian = convertScaleAbs(Laplacian(
                frame_monochrome_blurred[::self.configuration.align_frames_sampling_stride,
                ::self.configuration.align_frames_sampling_stride], CV_32F),
                alpha=self.alpha)

            # If the blurred frames are buffered, store the current frame at the current index.
            if self.buffer_laplacian:
                self.frames_monochrome_blurred_laplacian[index] = frame_monochrome_laplacian

            # If frames are not buffered, cache the current frame.
            else:
                self.laplacian_available_index = index
                self.laplacian_available = frame_monochrome_laplacian

            return frame_monochrome_laplacian

    def reset_alignment_point_lists(self):
        """
        Every frame keeps a list with the alignment points for which this frame is among the
        sharpest ones (so it is used in stacking). Reset this list for all frames.

        :return: -
        """

        # For every frame initialize the list with used alignment points.
        self.used_alignment_points = [[] for index in range(self.number)]

    @staticmethod
    def save_image(filename, image, color=False, avoid_overwriting=True,
                   header="PlanetarySystemStacker"):
        """
        Save an image to a file. If "avoid_overwriting" is set to False, images can have either
        ".png", ".tiff" or ".fits" format.

        :param filename: Name of the file where the image is to be written
        :param image: ndarray object containing the image data
        :param color: If True, a three channel RGB image is to be saved. Otherwise, it is assumed
                      that the image is monochrome.
        :param avoid_overwriting: If True, append a string to the input name if necessary so that
                                  it does not match any existing file. If False, overwrite
                                  an existing file.
        :param header: String with information on the PSS version being used (optional).
        :return: -
        """

        if avoid_overwriting:
            # If a file or directory with the given name already exists, append the word "_file".
            if Path(filename).is_dir():
                while True:
                    filename += '_file'
                    if not Path(filename).exists():
                        break
                filename += '.jpg'
            # If it is a file, try to append "_copy.tiff" to its basename. If it still exists, repeat.
            elif Path(filename).is_file():
                suffix = Path(filename).suffix
                while True:
                    p = Path(filename)
                    filename = Path.joinpath(p.parents[0], p.stem + '_copy' + suffix)
                    if not Path(filename).exists():
                        break
            else:
                # If the file name is new and has no suffix, add ".tiff".
                suffix = Path(filename).suffix
                if not suffix:
                    filename += '.tiff'

        elif Path(filename).suffix == '.png' or Path(filename).suffix == '.tiff':
            # Don't care if a file with the given name exists. Overwrite it if necessary.
            if path.exists(filename):
                remove(filename)
            # Write the image to the file. Before writing, convert the internal RGB representation into
            # the BGR representation assumed by OpenCV.
            if color:
                imwrite(str(filename), cvtColor(image, COLOR_RGB2BGR))
            else:
                imwrite(str(filename), image)

        elif Path(filename).suffix == '.fits':
            # Flip image horizontally to preserve orientation
            image = flip(image, 0)
            if color:
                image = moveaxis(image, -1, 0)
            hdu = fits.PrimaryHDU(image)
            hdu.header['CREATOR'] = header
            hdu.writeto(filename, overwrite=True)

        else:
            raise TypeError("Attempt to write image format other than 'tiff' or 'fits'")

    @staticmethod
    def read_image(filename):
        """
        Read an image (in tiff, fits, png or jpg format) from a file.

        :param filename: Path name of the input image.
        :return: RGB or monochrome image.
        """

        name, suffix = splitext(filename)

        # Make sure files with extensions written in large print can be read as well.
        suffix = suffix.lower()

        # Case FITS format:
        if suffix in ('.fit', '.fits'):
            image = fits.getdata(filename)

            # FITS output file from AS3 is 16bit depth file, even though BITPIX
            # has been set to "-32", which would suggest "numpy.float32"
            # https://docs.astropy.org/en/stable/io/fits/usage/image.html
            # To process this data in PSS, do "round()" and convert numpy array to "np.uint16"
            if image.dtype == '>f4':
                image = image.round().astype(uint16)

            # If color image, move axis to be able to process the content
            if len(image.shape) == 3:
                image = moveaxis(image, 0, -1).copy()

            # Flip image horizontally to recover original orientation
            image = flip(image, 0)

        # Case other supported image formats:
        elif suffix in ('.tiff', '.tif', '.png', '.jpg'):
            input_image = imread(filename, IMREAD_UNCHANGED)
            if input_image is None:
                raise IOError("Cannot read image file. Possible cause: Path contains non-ascii characters")

            # If color image, convert to RGB mode.
            if len(input_image.shape) == 3:
                image = cvtColor(input_image, COLOR_BGR2RGB)
            else:
                image = input_image

        else:
            raise TypeError("Attempt to read image format other than 'tiff', 'tif',"
                            " '.png', '.jpg' or 'fit', 'fits'")

        return image

def access_pattern(frames_object, average_frame_percent):
    """
    Simulate the access pattern of PSS to frame data, without any other activity in between. Return
    the overall time.

    :param frames_object: Frames object to access frames.
    :param average_frame_percent: Percentage of frames for average image computation.
    :return: Total time in seconds.
    """

    number = frames_object.number
    average_frame_number = max(
        ceil(number * average_frame_percent / 100.), 1)
    start = time()

    for index in range(number):
        frames_object.frames_mono_blurred_laplacian(index)

    frames_object.frames_mono_blurred(number - 1)
    frames_object.frames_mono_blurred(number - 1)

    for index in range(number):
        frames_object.frames_mono_blurred(index)

    for index in range(average_frame_number):
        frames_object.frames_mono(index)

    for index in range(number):
        frames_object.frames_mono_blurred_laplacian(index)

    for index in range(number):
        frames_object.frames(index)
        frames_object.frames_mono_blurred(index)

    return time() - start


def access_pattern_simple(frames_object, average_frame_percent):
    """
    Simulate the access pattern of PSS to frame data, without any other activity in between. Return
    the overall time.

    :param frames_object: Frames object to access frames.
    :param average_frame_percent: Percentage of frames for average image computation.
    :return: Total time in seconds.
    """

    number = frames.number
    start = time()

    for rep_cnt in range(5):
        for index in range(number):
            frames_object.frames_mono_blurred(index)

    return time() - start


if __name__ == "__main__":

    # Images can either be extracted from a video file or a batch of single photographs. Select
    # the example for the test run.
    type = 'video'
    version = 'frames'
    buffering_level = 4

    name_flats = None
    name_darks = None
    if type == 'image':
        # names = glob('Images/2012_*.tif')
        # names = glob('D:\SW-Development\Python\PlanetarySystemStacker\Examples\Moon_2011-04-10\South\*.TIF')
        names = glob(
            'D:\SW-Development\Python\PlanetarySystemStacker\Examples\Moon_2019-01-20\Images\*.TIF')
    else:
        # names = 'Videos/another_short_video.avi'
        # names = 'Videos/Moon_Tile-024_043939.avi'
        names = r'E:\SW-Development\Python\PlanetarySystemStacker\Examples\SER_Chris-Garry' \
                r'\SER_RGGB_16bit_LittleEndian_397_397.ser'
        # names = r'E:\SW-Development\Python\PlanetarySystemStacker\Examples\SER_Chris-Garry' \
        #         r'\SER_GRAYSCALED_16bit_LittleEndian_397_397.ser'
        # names = r'E:\SW-Development\Python\PlanetarySystemStacker\Examples\SER_Chris-Garry' \
        #         r'\SER_GRAYSCALED_12bit_BigEndian_352_400.ser'
        # name_flats = 'D:\SW-Development\Python\PlanetarySystemStacker\Examples\Darks_and_Flats\ASI120MM-S_Flat.avi'
        # name_darks = 'D:\SW-Development\Python\PlanetarySystemStacker\Examples\Darks_and_Flats\ASI120MM-S_Dark.avi'

    # Get configuration parameters.
    configuration = Configuration()
    configuration.initialize_configuration()

    # Initialize the Dark / Flat correction.
    if name_darks or name_flats:
        calibration = Calibration(configuration)
    else:
        calibration = None

    # Create the master dark if requested.
    if name_darks:
        calibration.create_master_dark(name_darks)
        print("Master dark created, shape: " + str(calibration.master_dark_frame.shape))
        dark_min = np_min(calibration.master_dark_frame)
        dark_max = np_max(calibration.master_dark_frame)
        print("Dark min: " + str(dark_min) + ", Dark max: " + str(dark_max))

    # Create the master flat if requested.
    if name_flats:
        calibration.create_master_flat(name_flats)
        print("Master flat created, shape: " + str(calibration.inverse_master_flat_frame.shape))
        flat_min = np_min(calibration.inverse_master_flat_frame)
        flat_max = np_max(calibration.inverse_master_flat_frame)
        print("Flat min: " + str(flat_min) + ", Flat max: " + str(flat_max))

    # Decide on the objects to be buffered, depending on configuration parameter.
    buffer_original = False
    buffer_monochrome = False
    buffer_gaussian = False
    buffer_laplacian = False

    if buffering_level > 0:
        buffer_laplacian = True
    if buffering_level > 1:
        buffer_gaussian = True
    if buffering_level > 2:
        buffer_original = True
    if buffering_level > 3:
        buffer_monochrome = True

    start = time()
    if version == 'frames':
        try:
            frames = Frames(configuration, names, type=type, calibration=calibration,
                            buffer_original=buffer_original, buffer_monochrome=buffer_monochrome,
                            buffer_gaussian=buffer_gaussian, buffer_laplacian=buffer_laplacian)
        except Error as e:
            print("Error: " + e.message)
            exit()
    else:
        try:
            frames = FramesOld(configuration, names, type=type)
            frames.add_monochrome(configuration.frames_mono_channel)
        except Error as e:
            print("Error: " + e.message)
            exit()
    initialization_time = time() - start

    print("Number of images read: " + str(frames.number))
    print("Image shape: " + str(frames.shape))

    # total_access_time = access_pattern_simple(frames,
    #                                       configuration.align_frames_average_frame_percent)
    total_access_time = access_pattern(frames, configuration.align_frames_average_frame_percent)

    print("\nInitialization time: {0:7.3f}, frame accesses and variant computations: {1:7.3f},"
          " total: {2:7.3f} (seconds)".format(initialization_time, total_access_time,
                                              initialization_time + total_access_time))

    frame = frames.frames(3)
    print("Image type: " + str(frame.dtype))

    # Check the OpenCV BGR and Matplotlibs RGB color orders
    import matplotlib.pyplot as plt
    if frames.color:
        plt.imshow(frame)
    else:
        plt.imshow(frame, cmap='gray')
    plt.show()
