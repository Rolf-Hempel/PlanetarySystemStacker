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

import os
import cv2
import struct
import datetime
import numpy as np


class SERParser(object):

    __author__ = 'Michal Powalko'
    __author_email__ = 'm.powalko@gmail.com'
    __version__ = '1.1'
    __name__ = 'SER parser for PlanetarySystemStacker tool (PSS)'

    def __init__(self, ser_file):
        super().__init__()

        self.sanity_check(ser_file)

        self.fid = self.open_file(ser_file)

        self.header = self.read_header()

        self.frame_count = self.header['FrameCount']

        self.frame_size = self.header['ImageWidth'] * \
                          self.header['ImageHeight'] * \
                          self.header['BytesPerPixel']

        if self.header['PixelDepthPerPlane'] <= 8:
            self.PixelDepthPerPlane = np.dtype(np.uint8)
        else:
            # FireCapture uses "LittleEndian".
            # Until FireCatpure 2.7 this flag was not set properly.
            self.PixelDepthPerPlane = np.dtype(np.uint16).newbyteorder('<')

        self.color = 8 <= self.header['ColorID'] <= 19 and self.header['DebayerPattern'] is not None \
                     or 100 <= self.header['ColorID'] <= 101

    def sanity_check(self, ser_file):
        if not os.path.isfile(ser_file):
            raise IOError("File does not exist")
        elif os.stat(ser_file).st_size == 0:
            raise IOError("File is empty")
        else:
            with open(ser_file, 'rb') as fid:
                HEADER = fid.read(14).decode()
            if HEADER != 'LUCAM-RECORDER':
                raise IOError("File has structure not conform with SER file format")

    def open_file(self, ser_file):
        return open(ser_file, 'rb')

    def read_header(self):
        """
        Read the "Header" of SER file with fixed size of 178 Byte.

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

        header = {key: value.decode('latin1') if isinstance(value, bytes) else
                  value for (key, value) in zip(KEYS, struct.unpack(
                          '<14s 7i 40s 40s 40s 2q', self.fid.read(178)))}

        if header['ColorID'] == 8:
            header['DebayerPattern'] = cv2.COLOR_BayerRG2BGR
        elif header['ColorID'] == 9:
            header['DebayerPattern'] = cv2.COLOR_BayerGR2BGR
        elif header['ColorID'] == 10:
            header['DebayerPattern'] = cv2.COLOR_BayerGB2BGR
        elif header['ColorID'] == 11:
            header['DebayerPattern'] = cv2.COLOR_BayerBG2BGR
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

        try:
            header['DateTime_Decoded'] = datetime.datetime(1, 1, 1) + \
                datetime.timedelta(microseconds=header['DateTime'] // 10)
        except Exception:
            header['DateTime_Decoded'] = None

        try:
            header['DateTime_UTC_Decoded'] = datetime.datetime(1, 1, 1) + \
                datetime.timedelta(microseconds=header['DateTime_UTC'] // 10)
        except Exception:
            header['DateTime_UTC_Decoded'] = None

        # Check, if FireCature metadata is available
        if 'fps=' in header['Telescope']:
            header['FPS'] = float(header['Telescope'].split('fps=')[1].split('gain')[0])
            header['Gain'] = int(header['Telescope'].split('gain=')[1].split('exp')[0])
            header['Exposure [ms]'] = float(header['Telescope'].split('exp=')[1].split('\x00')[0])

        self.frame_number = -1

        return header

    def read_frame(self, frame_number=None):
        """
        Read the "Image Data" of SER file.

        :return:    image_data          Multi dimmensional Numpy array
                                        containig image frame data
        """

        if frame_number is None:
            frame_number = self.frame_number + 1

        if 0 <= frame_number < self.frame_count:
            if frame_number != self.frame_number + 1:
                if frame_number == 0:
                    self.fid.seek(178)
                else:
                    self.fid.seek(178 + frame_number * self.frame_size)
        else:
            raise IOError('Error in reading video frame, index: {0} is out of bounds'.format(frame_number))

        self.frame_number = frame_number

        if self.header['NumberOfPlanes'] == 1:
            if self.color:
                return cv2.cvtColor(
                        np.frombuffer(self.fid.read(self.frame_size),
                        dtype=self.PixelDepthPerPlane).reshape(
                        self.header['ImageHeight'],
                        self.header['ImageWidth']),
                        self.header['DebayerPattern'])
            else:
                return np.frombuffer(self.fid.read(self.frame_size),
                        dtype=self.PixelDepthPerPlane).reshape(
                        self.header['ImageHeight'],
                        self.header['ImageWidth'])
        else:
            if self.header['ColorID'] == 101:
                return cv2.cvtColor(
                        np.frombuffer(self.fid.read(self.frame_size),
                        dtype=self.PixelDepthPerPlane).reshape(
                        self.header['ImageHeight'],
                        self.header['ImageWidth'],
                        self.header['NumberOfPlanes']),
                        cv2.COLOR_BGR2RGB)
            else:
                return np.frombuffer(self.fid.read(self.frame_size),
                        dtype=self.PixelDepthPerPlane).reshape(
                        self.header['ImageHeight'],
                        self.header['ImageWidth'],
                        self.header['NumberOfPlanes'])

    def read_all_frames(self):
        return [self.read_frame(idx) for idx in range(self.frame_count)]

    def read_trailer(self):
        """
        Read the "Trailer" of SER file with time stamps in UTC for every image
        frame. Those values are "optional".

        :return:    trailer             List containing "datetime" objects for
                                        time stamps in UTC. Otherwise "None"
        """

        self.fid.seek(178 + self.frame_count * self.frame_size)

        content = self.fid.read()

        self.frame_number = self.frame_count

        if content:
            return [datetime.datetime(1, 1, 1) + datetime.timedelta(
                microseconds=value // 10) for value in struct.unpack(
                        '<{0}Q'.format(self.frame_count), content)]
        else:
            return None

    def release(self):
        self.fid.close()


if __name__ == "__main__":

    import ser_parser
    import matplotlib.pyplot as plt

    file_path = r'C:\Temp\SER\Mars_GRBG.ser'

    cap = ser_parser.SERParser(file_path)
    last_frame_read = cap.read_frame(25)
    frame_count = cap.frame_count
    shape = last_frame_read.shape
    color = cap.color
    dtype = cap.PixelDepthPerPlane
    all_frames = cap.read_all_frames()

    # Check the OpenCV BGR and Matplotlibs RGB color orders
    if color:
        plt.imshow(last_frame_read)
    else:
        plt.imshow(last_frame_read, cmap='gray')
    plt.show()

    cv2.imshow('image', last_frame_read)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
