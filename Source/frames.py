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

import datetime
import struct
from glob import glob
from os import path, remove, listdir, stat
from pathlib import Path
from time import time

from PyQt5 import QtCore
from cv2 import imread, VideoCapture, CAP_PROP_FRAME_COUNT, cvtColor, COLOR_BGR2GRAY, \
    COLOR_RGB2GRAY, COLOR_BGR2RGB, GaussianBlur, Laplacian, CV_32F, COLOR_RGB2BGR, imwrite, \
    convertScaleAbs, CAP_PROP_POS_FRAMES, IMREAD_GRAYSCALE, IMREAD_UNCHANGED, \
    COLOR_BayerRG2RGB, COLOR_BayerGR2RGB, COLOR_BayerGB2RGB, COLOR_BayerBG2RGB
from math import ceil
from numpy import max as np_max
from numpy import min as np_min
from numpy import uint8, uint16, float32, clip, zeros, float64, where, average, \
    frombuffer, dtype

from configuration import Configuration
from exceptions import TypeError, ShapeError, ArgumentError, WrongOrderingError, Error, \
    InternalError
from frames_old import FramesOld


class VideoReader(object):
    """
    The VideoReader deals with the import of frames from a video file. Frames can be read either
    consecutively, or at an arbitrary frame index. Eventually, all common video types (such as .avi,
    .ser, .mov) should be supported.
    """

    def __init__(self):
        """
        Create the VideoReader object and initialize instance variables.
        """

        self.opened = False
        self.just_opened = False
        self.last_read = None
        self.last_frame_read = None
        self.frame_count = None
        self.shape = None
        self.color = None
        self.convert_to_grayscale = False
        self.dtype = None
        self.SERFile = False

    def sanity_check(self, file_path):
        """
        Performs a sanity check of input file.

        :return: -
        """

        if not path.isfile(file_path):
            raise IOError("File does not exist")
        elif stat(file_path).st_size == 0:
            raise IOError("File is empty")
        else:
            if path.splitext(file_path)[1].lower() == '.ser':
                with open(file_path, 'rb') as fid:
                    HEADER = fid.read(14).decode()
                if HEADER == 'LUCAM-RECORDER':
                    self.SERFile = True
                else:
                    raise IOError("File has structure not conform with SER file format")

    def open(self, file_path, convert_to_grayscale=False):
        """
        Initialize the VideoReader object and return parameters with video metadata.
         Throws an IOError if the video file format is not supported.

        :param file_path: Full name of the video file.
        :param convert_to_grayscale: If True, convert color frames to grayscale;
                                     otherwise return RGB color frames.
        :return: (frame_count, color, dtype, shape) with
                 frame_count: Total number of frames in video.
                 color: True, if frames are in color; False otherwise.
                 dtype: Numpy type, either uint8 or uint16
                 shape: Tuple with the shape of a single frame; (num_px_y, num_px_x, 3) for color,
                        (num_px_y, num_px_x) for B/W.
        """

        # Do sanity check
        self.sanity_check(file_path)

        # Check if input file is SER file
        if self.SERFile:
            SER_header = self.read_ser_header(file_path)

            self.frame_count = SER_header['FrameCount']
            self.color = 0 < SER_header['ColorID'] < 100 and \
                         SER_header['DebayerPattern'] is not None

            if SER_header['PixelDepthPerPlane'] <= 8:
                self.dtype = dtype(uint8)
            else:
                self.dtype = dtype(uint16)

            # Debayer raw RGB data
            if self.color:
                self.SER_data = [cvtColor(frame, SER_header['DebayerPattern']) for frame in self.read_ser_image_data(file_path, SER_header)]
            else:
                self.SER_data = self.read_ser_image_data(file_path, SER_header)

            self.last_read = 0
            self.last_frame_read = self.SER_data[self.last_read]
            self.shape = self.last_frame_read.shape

        else:
            try:
                # Create the VideoCapture object.
                self.cap = VideoCapture(file_path)

                # Read the first frame.
                ret, self.last_frame_read = self.cap.read()
                if not ret:
                    raise IOError("Error in reading first video frame")

                # Look up video metadata.
                self.last_read = 0
                self.frame_count = int(self.cap.get(CAP_PROP_FRAME_COUNT))
                self.shape = self.last_frame_read.shape
                self.color = (len(self.shape) == 3)
                self.dtype = self.last_frame_read.dtype
            except:
                raise IOError("Error in reading first video frame")

        # If file is in color mode and grayscale output is requested, do the conversion and change
        # metadata.
        if self.color:
            if convert_to_grayscale:
                # Remember to do the conversion when reading frames later on.
                self.convert_to_grayscale = True
                self.last_frame_read = cvtColor(self.last_frame_read, COLOR_BGR2GRAY)
                self.color = False
                self.shape = self.last_frame_read.shape
            else:
                # If color mode should stay, change image read by OpenCV into RGB.
                self.last_frame_read = cvtColor(self.last_frame_read, COLOR_BGR2RGB)

        self.opened = True
        self.just_opened = True

        # Return the metadata.
        return self.frame_count, self.color, self.dtype, self.shape

    def read_frame(self, index=None):
        """
        Read a single frame from the video.

        :param index: Frame index (optional). If no index is specified, the next frame is read.
                      At the first invocation, this is frame number 0.
        :return: Numpy array containing the frame. For B/W, the shape is (num_px_y, num_px_x).
                 For a color video, it is (num_px_y, num_px_x, 3). The type is uint8 or uint16 for
                 8 or 16 bit resolution.
        """

        if not self.opened:
            raise WrongOrderingError(
                "Error: Attempt to read video frame before opening VideoReader")

        # Special case: first call after initialization.
        if self.just_opened:
            self.just_opened = False

            # Frame 0 has been read during initialization. Not necessary to read it again.
            if index is None or index == 0:
                return self.last_frame_read
            # Otherwise set the frame pointer to the specified position.
            else:
                if not self.SERFile:
                    self.cap.set(CAP_PROP_POS_FRAMES, index)
                self.last_read = index

        # General case: not the first call.
        else:

            # Consecutive reading. Just increment the frame pointer.
            if index is None:
                self.last_read += 1

            # An index is specified explicitly. If it is the same as at last call, just return the
            # last frame.
            elif index == self.last_read:
                return self.last_frame_read

            # Some other frame was specified explicitly. If it is the next frame after the one read
            # last time, the frame pointer does not have to be set.
            else:
                if index != self.last_read + 1 and not self.SERFile:
                    self.cap.set(CAP_PROP_POS_FRAMES, index)
                self.last_read = index

        # A new frame has to be read. First check if the index is not out of bounds.
        if 0 <= self.last_read < self.frame_count:
            try:
                # Read the next frame.
                if self.SERFile:
                    ret = True
                    self.last_frame_read = self.SER_data[self.last_read]
                else:
                    ret, self.last_frame_read = self.cap.read()
                if not ret:
                    raise IOError("Error in reading video frame, index: " + str(index))
            except:
                raise IOError("Error in reading video frame, index: " + str(index))

            # Do the conversion to grayscale or into RGB color if necessary.
            if self.convert_to_grayscale:
                self.last_frame_read = cvtColor(self.last_frame_read, COLOR_BGR2GRAY)
            elif self.color:
                self.last_frame_read = cvtColor(self.last_frame_read, COLOR_BGR2RGB)
        else:
            raise ArgumentError("Error in reading video frame, index " + str(index) +
                                " is out of bounds")

        return self.last_frame_read

    def close(self):
        """
        Close the VideoReader object.

        :return:
        """

        if not self.SERFile:
            self.cap.release()
        self.opened = False

    def read_ser_header(self, file_path):
        """
        Read the "Header" of SER file with fixed size of 178 Byte.

        :param      file_path           Absolute file path of the SER file
        :return:    header              Dictionary containing the both "raw"
                                        and "decoded" values of "header"
        """

        KEYS = ('FileId', 'LuID', 'ColorID', 'LittleEndian', 'ImageWidth',
                'ImageHeight', 'PixelDepthPerPlane', 'FrameCount', 'Observer',
                'Instrument', 'Telescope', 'DateTime', 'DateTime_UTC')

        ColorID = {0:   'MONO',
                   8:   'BAYER_RGGB',
                   9:   'BAYER_GRBG',
                   10:  'BAYER_GBRG',
                   11:  'BAYER_BGGR',
                   16:  'BAYER_CYYM',
                   17:  'BAYER_YCMY',
                   18:  'BAYER_YMCY',
                   19:  'BAYER_MYYC',
                   100: 'RGB',
                   101: 'BGR'}

        with open(file_path, 'rb') as fid:
            content = fid.read(178)

        header = {key: value.decode('latin1') if isinstance(value, bytes) else
                  value for (key, value) in zip(KEYS, struct.unpack(
                          '<14s 7i 40s 40s 40s 2q', content))}

        if header['ColorID'] == 8:
            header['DebayerPattern'] = COLOR_BayerRG2RGB
        elif header['ColorID'] == 9:
            header['DebayerPattern'] = COLOR_BayerGR2RGB
        elif header['ColorID'] == 10:
            header['DebayerPattern'] = COLOR_BayerGB2RGB
        elif header['ColorID'] == 11:
            header['DebayerPattern'] = COLOR_BayerBG2RGB
        else:
            header['DebayerPattern'] = None

        header['ColorIDDecoded'] = ColorID[header['ColorID']]

        if header['ColorID'] < 100:
            header['NumberOfPlanes'] = 1
        else:
            header['NumberOfPlanes'] = 3

        if header['PixelDepthPerPlane'] <= 8:
            header['BytesPerPixel'] = header['NumberOfPlanes']
            if header['ColorID'] < 100:
                header['PixelDataOrganization'] = 'M'
            elif header['ColorID'] == 100:
                header['PixelDataOrganization'] = 'R G B'
            else:
                header['PixelDataOrganization'] = 'B G R'
        else:
            header['BytesPerPixel'] = 2 * header['NumberOfPlanes']
            if header['ColorID'] < 100:
                header['PixelDataOrganization'] = 'MM'
            elif header['ColorID'] == 100:
                header['PixelDataOrganization'] = 'RR GG BB'
            else:
                header['PixelDataOrganization'] = 'BB GG RR'

        header['DateTime_Decoded'] = datetime.datetime(1, 1, 1) + \
            datetime.timedelta(microseconds=header['DateTime'] // 10)

        header['DateTime_UTC_Decoded'] = datetime.datetime(1, 1, 1) + \
            datetime.timedelta(microseconds=header['DateTime_UTC'] // 10)

        # Check, if FireCature metadata is available
        if 'fps=' in header['Telescope']:
            header['FPS'] = float(header['Telescope'].split('fps=')[1].split('gain')[0])
            header['Gain'] = int(header['Telescope'].split('gain=')[1].split('exp')[0])
            header['Exposure [ms]'] = float(header['Telescope'].split('exp=')[1].split('\x00')[0])

        return header

    def read_ser_image_data(self, file_path, header=None):
        """
        Read the "Image Data" of SER file.

        :param      file_path           Absolute file path of the SER file
        :param      header (optional)   Ser file header, if already available
        :return:    image_data          Multi dimmensional Numpy array
                                        containig image frame data
        """

        if header is None:
            header = self.read_ser_header(file_path)

        if header['PixelDepthPerPlane'] <= 8:
            PixelDepthPerPlane = uint8
        else:
            # FireCapture uses "LittleEndian".
            # Until FireCatpure 2.7 this flag was not set properly.
            PixelDepthPerPlane = dtype(uint16).newbyteorder('<')

        AMOUNT = header['FrameCount'] * header['ImageWidth'] * \
            header['ImageHeight'] * header['BytesPerPixel']

        with open(file_path, 'rb') as fid:
            fid.seek(178)
            content = fid.read(AMOUNT)

        return frombuffer(content, dtype=PixelDepthPerPlane).reshape(
                header['FrameCount'], header['ImageHeight'],
                header['ImageWidth'])

    def read_ser_trailer(self, file_path, header=None):
        """
        Read the "Trailer" of SER file with time stamps in UTC for every image
        frame. Those values are "optional".

        :param      file_path           Absolute file path of the SER file
        :param      header (optional)   Ser file header, if already available
        :return:    trailer             List containing "datetime" objects for
                                        time stamps in UTC. Otherwise "None"
        """

        if header is None:
            header = self.read_ser_header(file_path)

        OFFSET = 178 + header['FrameCount'] * header['ImageWidth'] * \
            header['ImageHeight'] * header['BytesPerPixel']

        with open(file_path, 'rb') as fid:
            fid.seek(OFFSET)
            content = fid.read()

        if content:
            return [datetime.datetime(1, 1, 1) + datetime.timedelta(
                microseconds=value // 10) for value in struct.unpack(
                        '<{0}Q'.format(header['FrameCount']), content)]
        else:
            return None


class ImageReader(object):
    """
    The ImageReader deals with the import of frames from a list of single images. Frames can
    be read either consecutively, or at an arbitrary frame index. It is assumed that the
    lexicographic order of file names corresponds to their chronological order. Eventually, all
    common image types (such as .tiff, .png, .jpg) should be supported.
    """

    def __init__(self):
        """
        Create the ImageReader object and initialize instance variables.
        """

        self.opened = False
        self.just_opened = False
        self.last_read = None
        self.last_frame_read = None
        self.frame_count = None
        self.shape = None
        self.color = None
        self.convert_to_grayscale = False
        self.dtype = None

    def open(self, file_path_list, convert_to_grayscale=False):
        """
        Initialize the ImageReader object and return parameters with image metadata.

        :param file_path_list: List with path names to the image files.
        :param convert_to_grayscale: If True, convert color frames to grayscale;
                                     otherwise return RGB color frames.
        :return: (frame_count, color, dtype, shape) with
                 frame_count: Total number of frames.
                 color: True, if frames are in color; False otherwise.
                 dtype: Numpy type, either uint8 or uint16
                 shape: Tuple with the shape of a single frame; (num_px_y, num_px_x, 3) for color,
                        (num_px_y, num_px_x) for B/W.
        """

        self.file_path_list = file_path_list

        try:
            self.frame_count = len(self.file_path_list)

            if convert_to_grayscale:
                self.last_frame_read = imread(self.file_path_list[0], IMREAD_GRAYSCALE)
                # Remember to do the conversion when reading frames later on.
                self.convert_to_grayscale = True
            else:
                self.last_frame_read = imread(self.file_path_list[0], IMREAD_UNCHANGED)

            # Look up metadata.
            self.last_read = 0
            self.shape = self.last_frame_read.shape
            self.color = (len(self.shape) == 3)
            self.dtype = self.last_frame_read.dtype
        except:
            raise IOError("Error in reading first frame")

        # If in color mode, swap B and R channels to convert from cv2 to standard RGB.
        if self.color:
            self.last_frame_read = cvtColor(self.last_frame_read, COLOR_BGR2RGB)

        self.opened = True
        self.just_opened = True

        # Return the metadata.
        return self.frame_count, self.color, self.dtype, self.shape

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
                if self.convert_to_grayscale:
                    self.last_frame_read = imread(self.file_path_list[self.last_read],
                                                  IMREAD_GRAYSCALE)
                else:
                    self.last_frame_read = imread(self.file_path_list[self.last_read],
                                                  IMREAD_UNCHANGED)
            except:
                raise IOError("Error in reading image frame, index: " + str(index))
        else:
            raise ArgumentError("Error in reading image frame, index: " + str(index) +
                                " is out of bounds")

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
            if extension in ('.avi', '.ser'):
                reader = VideoReader()
                frame_count, input_color, input_dtype, input_shape = reader.open(master_name)
                self.configuration.hidden_parameters_current_dir = str(Path(master_name).parent)
            else:
                raise InternalError(
                    "Unsupported file type '" + extension + "' specified for master frame "
                                                            "construction")
        # Case image directory:
        elif Path(master_name).is_dir():
            names = [path.join(master_name, name) for name in listdir(master_name)]
            reader = ImageReader()
            frame_count, input_color, input_dtype, input_shape = reader.open(names)
            self.configuration.hidden_parameters_current_dir = str(master_name)
        else:
            raise InternalError("Cannot decide if input file is video or image directory")

        # Sum all frames in a 64bit buffer.
        master_frame_64 = zeros(input_shape, float64)
        for index in range(frame_count):
            master_frame_64 += reader.read_frame()

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
                          directory. Otherwise the file name (TIFF) of the master frame.
        :param load_from_file: True, if to be loaded from file. False, if to be created anew.
        :return: -
        """

        # Reset a master dark frame if previously allocated.
        self.reset_master_dark()

        # Create the master frame or read it from a file.
        if load_from_file:
            self.master_dark_frame = imread(dark_name, IMREAD_UNCHANGED)
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
                          directory. Otherwise the file name (TIFF) of the master frame.
        :param load_from_file: True, if to be loaded from file. False, if to be created anew.
        :return: -
        """

        # Reset a master flat frame if previously allocated.
        self.reset_master_flat()

        # Create the master frame or read it from a file.
        if load_from_file:
            self.master_flat_frame = imread(flat_name, IMREAD_UNCHANGED)
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

    def adapt_frame_type(self, frame_dtype):
        """
        Adapt the type of the master frames to the type of frames to be corrected.

        :param frame_dtype: Dtype of frames to be corrected. Either uint8 or uint16.
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

    def __init__(self, configuration, names, type='video', calibration=None,
                 convert_to_grayscale=False, progress_signal=None, buffer_original=True,
                 buffer_monochrome=False, buffer_gaussian=True, buffer_laplacian=True):
        """
        Initialize the Frame object, and read all images. Images can be stored in a video file or
        as single images in a directory.

        :param configuration: Configuration object with parameters
        :param names: In case "video": name of the video file. In case "image": list of names for
                      all images.
        :param type: Either "video" or "image".
        :param calibration: (Optional) calibration object for darks/flats correction.
        :param convert_to_grayscale: If "True", convert frames to grayscale if they are RGB.
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
        self.convert_to_grayscale = convert_to_grayscale

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

        # Initialize the list of original frames.
        self.frames_original = None

        # Compute the scaling value for Laplacian computation.
        self.alpha = 1. / 256.

        # Initialize and open the reader object.
        if self.type == 'image':
            self.reader = ImageReader()
        elif self.type == 'video':
            self.reader = VideoReader()
        else:
            raise TypeError("Image type " + self.type + " not supported")

        self.number, self.color, self.dt0, self.shape = self.reader.open(self.names,
            convert_to_grayscale=self.convert_to_grayscale)

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
                self.calibration.adapt_frame_type(self.dt0)
        else:
            self.calibration_matches = False

        # Initialize lists of monochrome frames (with and without Gaussian blur) and their
        # Laplacians.
        colors = ['red', 'green', 'blue', 'panchromatic']
        if self.configuration.frames_mono_channel in colors:
            self.color_index = colors.index(self.configuration.frames_mono_channel)
        else:
            raise ArgumentError("Invalid color selected for channel extraction")
        self.frames_monochrome = [None for index in range(self.number)]
        self.frames_monochrome_blurred = [None for index in range(self.number)]
        self.frames_monochrome_blurred_laplacian = [None for index in range(self.number)]
        self.used_alignment_points = None

    def compute_required_buffer_size(self, buffering_level):
        """
        Compute the RAM required to store original images and their derivatives, and other objects
        which scale with the image size.

        Additional to the original images and their derivatives, the following large objects are
        allocated during the workflow:
            calibration.master_dark_frame: pixels * colors (float32)
            calibration.master_dark_frame_uint8: pixels * colors (uint8)
            calibration.master_dark_frame_uint16: pixels * colors (uint16)
            calibration.master_flat_frame: pixels * colors (float32)
            align_frames.mean_frame: image pixels (int32)
            align_frames.mean_frame_original: image pixels (int32)
            alignment_points, reference boxes: < 2 * image pixels (int32)
            alignment_points, stacking buffers: < 2 * image pixels * colors (float32)
            stack_frames.stacked_image_buffer: image pixels * colors (float32)
            stack_frames.number_single_frame_contributions: image pixels (int32)
            stack_frames.sum_single_frame_weights: image pixels (float32)
            stack_frames.mask: image pixels (float32)
            stack_frames.averaged_background: image pixels * colors (float32)
            stack_frames.stacked_image: image pixels * colors (uint16)

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

        # Compute the size of additional workspace objects allocated during the workflow.
        buffer_additional_workspace = number_pixel * 57
        if self.color:
            buffer_additional_workspace += number_pixel * 58

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
                                                  int((frame_index / self.number) * 100.))
                    # Read the next frame. If dark/flat correction is active, do the corrections.
                    if self.calibration_matches:
                        self.frames_original.append(self.calibration.correct(
                            self.reader.read_frame()))
                    else:
                        self.frames_original.append(self.reader.read_frame())

                self.reader.close()

            # If original frames are not buffered, initialize an empty frame list, so frames can be
            # read later in non-consecutive order.
            else:
                self.frames_original = [None for index in range(self.number)]

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
    def save_image(filename, image, color=False, avoid_overwriting=True):
        """
        Save an image to a file.

        :param filename: Name of the file where the image is to be written
        :param image: ndarray object containing the image data
        :param color: If True, a three channel RGB image is to be saved. Otherwise, monochrome.
        :param avoid_overwriting: If True, append a string to the input name if necessary so that
                                  it does not match any existing file. If False, overwrite
                                  an existing file.
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

        # Don't care if a file with the given name exists. Overwrite it if necessary.
        elif path.exists(filename):
            remove(filename)

        # Write the image to the file. Before writing, convert the internal RGB representation into
        # the BGR representation assumed by OpenCV.
        if color:
            imwrite(str(filename), cvtColor(image, COLOR_RGB2BGR))
        else:
            imwrite(str(filename), image)


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
        names = 'Videos/another_short_video.avi'
        # names = 'Videos/Moon_Tile-024_043939.avi'
        name_flats = 'D:\SW-Development\Python\PlanetarySystemStacker\Examples\Darks_and_Flats\ASI120MM-S_Flat.avi'
        name_darks = 'D:\SW-Development\Python\PlanetarySystemStacker\Examples\Darks_and_Flats\ASI120MM-S_Dark.avi'

    # Get configuration parameters.
    configuration = Configuration()

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
                            convert_to_grayscale=False,
                            buffer_original=buffer_original, buffer_monochrome=buffer_monochrome,
                            buffer_gaussian=buffer_gaussian, buffer_laplacian=buffer_laplacian)
        except Error as e:
            print("Error: " + e.message)
            exit()
    else:
        try:
            frames = FramesOld(configuration, names, type=type, convert_to_grayscale=False)
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
