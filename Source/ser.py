# -*- coding: utf-8; -*-
"""
Copyright (c) 2018 Michal Powalko, m.powalko@gmx.de

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

__author__ = 'Michal Powalko'
__author_email__ = 'm.powalko@gmail.com'
__version__ = '1.0'

import os
import struct
import datetime
import numpy as np


# TODO Support of 16bit MONO
# TODO Support of ColorID different than MONO
# TODO Support of debayer via OpenCV

def load(ser_file):
    '''
    Main function to be used to import the data from Ser file.
    It performs sanity check and import of header, image data and trailer.

        [int iErr, dict header, numpy.ndarray image_data, list trailer ] = ser.load(ser_file)

    ARGUMENTS:
        str ser_file    Absolute file path of the SER file

    RETURNS:
        int iErr                    Intiger code for error message
        dict header                 Dictionary containing the both "raw" and
                                    "decoded" values of  "header"
        numpy.ndarray image_data    Multi dimmensional Numpy array containig
                                    image frame data.
        list trailer                List containing "datetime" objects for time
                                    stamps in UTC. Otherwise returns "None"
    '''

    MESSAGE = {0: 'Ser file is OK',
               1: 'Ser file does not exist',
               2: 'Ser file is empty',
               3: 'Ser file has structure not conform with SER file format'}

    iErr = sanity_check(ser_file)

    if iErr == 0:
        header = read_header(ser_file)
        image_data = read_image_data(ser_file, header)
        trailer = read_trailer(ser_file, header)
    else:
        header = image_data = trailer = None
        print(MESSAGE[iErr])

    return iErr, header, image_data, trailer


def sanity_check(ser_file):
    '''
    Performs a sanity check of SER file according to SER file format.

        int iErr = ser.sanity_check(str ser_file)

    ARGUMENTS:
        str ser_file    Absolute file path of the SER file

    RETURNS:
        int iErr        Intiger code for error message:
                        0   Ser file is OK
                        1   Ser file does not exist
                        2   Ser file is empty
                        3   Ser file has structure not conform with SER file format
    '''

    iErr = 0

    if not os.path.isfile(ser_file):
        iErr = 1
    elif os.stat(ser_file).st_size == 0:
        iErr = 2
    else:
        with open(ser_file, 'rb') as fid:
            FileId = fid.read(14).decode()
        if FileId != 'LUCAM-RECORDER':
            iErr = 3

    return iErr


def read_header(ser_file):
    '''
    Read the "Header" of SER file with fixed size of 178 Byte.

        dict header = ser.read_header(str ser_file)

    ARGUMENTS:
        str ser_file    Absolute file path of the SER file

    RETURNS:
        dict header     Dictionary containing the both "raw" and "decoded"
                        values of  "header"
    '''

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

    with open(ser_file, 'rb') as fid:
        content = fid.read(178)

    header = dict(zip(KEYS, struct.unpack('<14s 7i 40s 40s 40s 2q', content)))

    for key, value in header.items():
        if isinstance(value, bytes):
            header[key] = value.decode('latin-1')

    header['ColorID_Decoded'] = ColorID[header['ColorID']]

    if header['ColorID'] < 100:
        header['NumberOfPlanes'] = 1
    else:
        header['NumberOfPlanes'] = 3

    if header['PixelDepthPerPlane'] < 9:
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

    return header


def read_image_data(ser_file, header=None):
    '''
    Read the "Image Data" of SER file.

        numpy.ndarray image_data = ser.read_image_data(str ser_file, dict header)

    ARGUMENTS:
        str ser_file                Absolute file path of the SER file
        dict header (optional)      Ser file header, if already available

    RETURNS:
        numpy.ndarray image_data    Multi dimmensional Numpy array containig
                                    image frame data.
    '''

    if header is None:
        header = read_header(ser_file)

    if header['PixelDepthPerPlane'] < 9:
        PixelDepthPerPlane = np.uint8
    else:
        PixelDepthPerPlane = np.uint16

    AMOUNT = header['FrameCount'] * header['ImageWidth'] * \
        header['ImageHeight'] * header['BytesPerPixel']

    with open(ser_file, 'rb') as fid:
        fid.seek(178)
        content = fid.read(AMOUNT)

    return np.frombuffer(content, dtype=PixelDepthPerPlane).reshape(
            header['FrameCount'], header['ImageHeight'], header['ImageWidth'])


def read_trailer(ser_file, header=None):
    '''
    Read the "Trailer" of SER file with time stamps in UTC for every image
    frame. Those value are "optional".

        list trailer = ser.read_trailer(str ser_file, dict header)

    ARGUMENTS:
        str ser_file                Absolute file path of the SER file
        dict header (optional)      Ser file header, if already available

    RETURNS:
        list trailer                List containing "datetime" objects for time
                                    stamps in UTC. Otherwise returns "None"
    '''

    if header is None:
        header = read_header(ser_file)

    OFFSET = 178 + header['FrameCount'] * header['ImageWidth'] * \
        header['ImageHeight'] * header['BytesPerPixel']

    with open(ser_file, 'rb') as fid:
        fid.seek(OFFSET)
        content = fid.read()

    if content:
        trailer = [datetime.datetime(1, 1, 1) + datetime.timedelta(
            microseconds=value // 10) for value in
            struct.unpack('<{0}Q'.format(header['FrameCount']), content)]
    else:
        trailer = None

    return trailer


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import ser

    FILE = r'Videos\Ser_8bit_mono.ser'

    iErr, header, image_data, trailer = ser.load(FILE)

    if iErr == 0:
        plt.imshow(image_data[1,:,:], cmap='gray')
