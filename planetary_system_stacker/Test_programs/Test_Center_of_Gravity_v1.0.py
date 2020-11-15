# -*- coding: utf-8 -*-
"""
@author: Michal Powalko m.powalko@gmail.com

Version 1.0 [2019-08-01]
------------------------
    INF: Inititial release

"""

import cv2
import time
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
# https://www.learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
# https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
# https://en.wikipedia.org/wiki/Image_moment#Examples

LOOPS = int(1)

file_path = r'D:\SW-Development\Python\PlanetarySystemStacker\Examples\Jupiter\2019-05-26-0115_4-L-Jupiter_ZWO ASI290MM Mini_pipp.avi'
cap = cv2.VideoCapture(file_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

tic = time.time()
frames = [cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY) for idx in range(frame_count)]
print('Importing the video file took:\t\t{0:.2f} [s]'.format(time.time() - tic))
cap.release()

tic = time.time()
frames = [cv2.GaussianBlur(frame, (3, 3), 0) for frame in frames]
print('GaussianBlur took:\t\t\t{0:.2f} [s]'.format(time.time() - tic))

tic = time.time()
for idx in range(LOOPS):
    cog_rolf = [ndimage.measurements.center_of_mass(np.uint8(1)*(frame >= (np.max(frame)/2).astype(frame.dtype)))
                for frame in frames]
    cog_rolf = [(int(round(aaa[0])), int(round(aaa[1]))) for aaa in cog_rolf]
print('Rolf/Jens CoG estimation took:\t\t{0:.2f} [s]'.format(time.time() - tic))

tic = time.time()
for idx in range(LOOPS):
    MM = [cv2.moments(cv2.threshold(frame, frame.max()/2, 1, cv2.THRESH_BINARY)[1])
         for frame in frames]
    cog_michal = [(round(M["m01"] / M["m00"]), round(M['m10'] / M["m00"])) for M in MM]
print('Michals CoG estimation took:\t\t{0:.2f} [s]'.format(time.time() - tic))

dev_x = [a[1] - b[1] for a,b in zip(cog_rolf, cog_michal)]
dev_y = [a[0] - b[0] for a,b in zip(cog_rolf, cog_michal)]

plt.subplot(2, 1, 1)
plt.title('Center of gravity estimation')
plt.plot(dev_x, '-b')
plt.xlabel('Frame chronological number')
plt.ylabel('Error in x axis [pixel]')
plt.grid(True)
plt.subplot(2, 1, 2)
plt.title('Center of gravity estimation')
plt.plot(dev_y, '-r')
plt.xlabel('Frame chronological number')
plt.ylabel('Error in y axis [pixel]')
plt.grid(True)
pass

