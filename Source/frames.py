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

import glob
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc

from configuration import Configuration
from exceptions import TypeError, ShapeError, ArgumentError


class Frames(object):
    """
        This object stores the image data of all frames.

    """

    def __init__(self, configuration, names, type='video', convert_to_grayscale=False,
                 progress_signal=None):
        """
        Initialize the Frame object, and read all images. Images can be stored in a video file or
        as single images in a directory.

        :param configuration: Configuration object with parameters
        :param names: In case "video": name of the video file. In case "image": list of names for
                      all images.
        :param type: Either "video" or "image".
        :param convert_to_grayscale: If "True", convert frames to grayscale if they are RGB.
        :param progress_signal: Either None (no progress signalling), or a signal with the signature
                                (str, int) with the current activity (str) and the progress in
                                percent (int).
        """

        self.configuration = configuration

        if type == 'image':
            # Use scipy.misc to read in image files. If "convert_to_grayscale" is True, convert
            # pixel values to 32bit floats.
            self.number = len(names)
            signal_step_size = max(int(self.number / 10), 1)
            if convert_to_grayscale:
                self.frames = [misc.imread(path, mode='F') for path in names]
            else:
                self.frames = []
                conversion_factor = np.float32(1. / 255.)
                for frame_index, path in enumerate(names):
                    # After every "signal_step_size"th frame, send a progress signal to the main GUI.
                    if progress_signal is not None and frame_index % signal_step_size == 0:
                        progress_signal.emit("Read all frames",
                                             int((frame_index / self.number) * 100.))
                    # Read the next frame, and scale it to the range [0., 1.] if necessary.
                    frame = cv2.imread(path, -1)
                    if frame.dtype == 'uint16':
                        frame = frame * conversion_factor
                    self.frames.append(frame)

                if progress_signal is not None:
                    progress_signal.emit("Read all frames", 100)
            self.shape = self.frames[0].shape

            # Test if all images have the same shape. If not, raise an exception.
            for image in self.frames:
                if image.shape != self.shape:
                    raise ShapeError("Images have different size")
                elif len(self.shape) != len(image.shape):
                    raise ShapeError("Mixing grayscale and color images not supported")

        elif type == 'video':
            # In case "video", use OpenCV to capture frames from video file. Revert the implicit
            # conversion from RGB to BGR in OpenCV input.
            cap = cv2.VideoCapture(names)
            self.number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frames = []
            signal_step_size = max(int(self.number / 10), 1)
            for frame_index in range(self.number):
                # After every "signal_step_size"th frame, send a progress signal to the main GUI.
                if progress_signal is not None and frame_index%signal_step_size == 0:
                    progress_signal.emit("Read all frames", int((frame_index/self.number)*100.))
                # Read the next frame.
                ret, frame = cap.read()
                if ret:
                    if convert_to_grayscale:
                        self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    else:
                        self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    raise IOError("Error in reading video frame")
            cap.release()
            if progress_signal is not None:
                progress_signal.emit("Read all frames", 100)
            self.shape = self.frames[0].shape
        else:
            raise TypeError("Image type not supported")

        # Monochrome images are stored as 2D arrays, color images as 3D.
        if len(self.shape) == 2:
            self.color = False
        elif len(self.shape) == 3:
            self.color = True
        else:
            raise ShapeError("Image shape not supported")

        # Initialize lists of monochrome frames (with and without Gaussian blur) and their
        # Laplacians.
        self.frames_mono = None
        self.frames_mono_blurred = None
        self.frames_mono_blurred_laplacian = None
        self.used_alignment_points = None

        # Initialize the alignment point lists for all frames.
        self.reset_alignment_point_lists()

    def reset_alignment_point_lists(self):
        """
        Every frame keeps a list with the alignment points for which this frame is among the
        sharpest ones (so it is used in stacking). Reset this list for all frames.

        :return: -
        """

        # For every frame initialize the list with used alignment points.
        self.used_alignment_points = []
        for frame_index in range(self.number):
            self.used_alignment_points.append([])

    def add_monochrome(self, color):
        """
        Create monochrome versions of all frames. Add a list of monochrome frames
        "self.frames_mono". If the original frames are monochrome, just point the monochrome frame
        list to the original images (no deep copy!). Also, add a blurred version of the frame list
        (using a Gaussian filter) "self.frames_mono_blurred", and the Laplacian of that image.

        :param color: Either "red" or "green", "blue", or "panchromatic"
        :return: -
        """

        if self.color:
            colors = ['red', 'green', 'blue', 'panchromatic']
            if not color in colors:
                raise ArgumentError("Invalid color selected for channel extraction")
            elif color == 'panchromatic':
                self.frames_mono = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in self.frames]
            else:
                self.frames_mono = [frame[:, :, colors.index(color)] for frame in self.frames]
        else:
            self.frames_mono = self.frames

        # Add versions of all frames with Gaussian blur added.
        self.frames_mono_blurred = [cv2.GaussianBlur(frame, (
            self.configuration.frames_gauss_width, self.configuration.frames_gauss_width),
                                                     0) for frame in self.frames_mono]

        if self.configuration.rank_frames_method == "Laplace":
            # Add the Laplacians of down-sampled blurred images.
            self.frames_mono_blurred_laplacian = [cv2.Laplacian(
                frame[::self.configuration.align_frames_sampling_stride,
                ::self.configuration.align_frames_sampling_stride], cv2.CV_32F) for frame in
                                                  self.frames_mono_blurred]

    def save_image(self, filename, image, color=False, avoid_overwriting=True):
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
        elif os.path.exists(filename):
            os.remove(filename)

        # Write the image to the file. Before writing, convert the internal RGB representation into
        # the BGR representation assumed by OpenCV.
        if color:
            cv2.imwrite(str(filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(filename), image)


if __name__ == "__main__":

    # Images can either be extracted from a video file or a batch of single photographs. Select
    # the example for the test run.
    type = 'image'
    if type == 'image':
        # names = glob.glob('Images/2012_*.tif')
        names = glob.glob('D:\SW-Development\Python\PlanetarySystemStacker\Examples\Moon_2011-04-10\South\*.TIF')
    else:
        names = 'Videos/short_video.avi'

    # Get configuration parameters.
    configuration = Configuration()

    try:
        frames = Frames(configuration, names, type=type, convert_to_grayscale=False)
        print("Number of images read: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Exception as e:
        print("Error: " + e.message)
        exit()

    # Create monochrome versions of all frames. If the original frames are monochrome, just point
    # the monochrome frame list to the original images (no deep copy!).
    try:
        frames.add_monochrome(configuration.frames_mono_channel)
    except ArgumentError as e:
        print("Error: " + e.message)
        exit()

    plt.imshow(frames.frames_mono[0], cmap='Greys_r')
    plt.show()

    if type == 'video':
        stacked_image_name = os.path.splitext(names)[0] + '_pss.tiff'
    # For single image input, the Frames constructor expects a list of image file names for
    # "names".
    else:
        image_dir = 'Images'
        stacked_image_name = image_dir + '_pss.tiff'

    frames.save_image(stacked_image_name, frames.frames_mono[0], avoid_overwriting=False)
