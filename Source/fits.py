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
import numpy as np
from astropy.io import fits


def write(fits_file, image, header=None, split_rgb2files=False):

    '''
    Wrapper of "astropy.io.fits" module.

        fits_writer.write(fits_file, image, header, split_rgb2files)

    ARGUMENTS:
        str fits_file               Absolute file path for FITS file
        numpy.ndarray image_data    Multi dimmensional Numpy array containig
                                    image data (MONO \ Bayer \ BGR)
        dict header                 Dictionary to be saved as header
        bool split_rgb2files        Save RGB as seperate FITS files

    RETURNS:
        int iErr                    Intiger code for error message
    '''

    if image.ndim == 3:
        if split_rgb2files:
            blue = fits.PrimaryHDU(image[:, :, 0])
            blue.header['CHANNEL'] = 'Blue'
            red = fits.PrimaryHDU(image[:, :, 1])
            red.header['CHANNEL'] = 'Red'
            green = fits.PrimaryHDU(image[:, :, 2])
            green.header['CHANNEL'] = 'Green'
            hdul = [red, green, blue]
        else:
            hdul = [fits.PrimaryHDU(np.moveaxis(image, -1, 0))]
    else:
        hdul = [fits.PrimaryHDU(image)]

    for hdu in hdul:
        hdu.header['CREATOR'] = 'PlanetarySystemStacker'
        if header is not None:
            for key in header:
                hdu.header[key] = header[key]
        if 'CHANNEL' in hdu.header:
            file = '{0}_{1}.fits'.format(os.path.splitext(fits_file)[0],
                    hdu.header['CHANNEL'][0])
        else:
            file = fits_file
        hdu.writeto(file, overwrite=True)
