from frames import Frames
import pyfits
from numpy import moveaxis

import cv2
from cv2 import cvtColor, COLOR_RGB2BGR, COLOR_BGR2RGB

def astropy_fits(filename_in, filename_out):
    frame = cvtColor(Frames.read_image(filename_in), COLOR_RGB2BGR)

    cv2.imshow('Example - Show image in window', frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    Frames.save_image(filename_out, cvtColor(frame, COLOR_BGR2RGB), color=True)

def pyfits_fits(filename_in, filename_out):
    frame = cvtColor(pyfits.getdata(filename_in), COLOR_RGB2BGR)

    cv2.imshow('Example - Show image in window', frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(frame.shape) == 3:
        frame = moveaxis(frame, -1, 0)
    hdu = pyfits.PrimaryHDU(frame)
    header = "PlanetarySystemStacker"
    hdu.header['CREATOR'] = header
    hdu.writeto(filename_out, overwrite=True)

# filename = "D:\SW-Development\Python\PlanetarySystemStacker\Examples\Jupiter_Richard\\2020-07-29-2145_3-L-Jupiter_ALTAIRGP224C_pss_p70_b48_gpp.fits"
filename = 'frame.fits'
# astropy_fits(filename, "frame_out.fits")
pyfits_fits(filename, "frame_out.fits")