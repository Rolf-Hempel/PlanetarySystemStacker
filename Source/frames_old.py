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
from os import path, remove
from pathlib import Path
from time import time

import numpy as np
from cv2 import imread, VideoCapture, CAP_PROP_FRAME_COUNT, cvtColor, COLOR_BGR2GRAY, \
    COLOR_BGR2RGB, GaussianBlur, Laplacian, CV_32F, COLOR_RGB2BGR, imwrite, convertScaleAbs
from scipy import misc

from configuration import Configuration
from exceptions import TypeError, ShapeError, ArgumentError, WrongOrderingError, Error
# from frames import Frames


class FramesOld(object):
    """
        This object stores the image data of all frames. Four versions of the original frames are
        used throughout the data processing workflow. They are (re-)used in the folliwing phases:
        1. Original (color) frames, type: uint8 / uint16
            - Frame stacking ("stack_frames.stack_frames")
        2. Monochrome version of 1., type: uint8
            - Computing the average frame (only average frame subset, "align_frames.average_frame")
        3. Gaussian blur added to 2., type: type: uint8
            - Aligning all frames ("align_frames.align_frames")
            - Frame stacking ("stack_frames.stack_frames")
        4. Down-sampled Laplacian of 3., type: uint8
            - Overall image ranking ("rank_frames.frame_score")
            - Ranking frames at alignment points("alignment_points.compute_frame_qualities")

        Currently all versions of all frames are stored during the entire workflow. As an
        alternative, a non-buffered version should be added.

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
        self.progress_signal = progress_signal

        if type == 'image':
            # Use scipy.misc to read in image files. If "convert_to_grayscale" is True, convert
            # pixel values to 32bit floats.
            self.number = len(names)
            self.signal_step_size = max(int(self.number / 10), 1)
            if convert_to_grayscale:
                self.frames_original = [misc.imread(path, mode='F') for path in names]
            else:
                self.frames_original = []
                for frame_index, path in enumerate(names):
                    # After every "signal_step_size"th frame, send a progress signal to the main GUI.
                    if self.progress_signal is not None and frame_index%self.signal_step_size == 0:
                        self.progress_signal.emit("Read all frames",
                                             int((frame_index / self.number) * 100.))
                    # Read the next frame.
                    frame = imread(path, -1)
                    self.frames_original.append(frame)

                if self.progress_signal is not None:
                    self.progress_signal.emit("Read all frames", 100)
            self.shape = self.frames_original[0].shape
            dt0 = self.frames_original[0].dtype

            # Test if all images have the same shape, color type and depth.
            # If not, raise an exception.
            for image in self.frames_original:
                if image.shape != self.shape:
                    raise ShapeError("Images have different size")
                elif len(self.shape) != len(image.shape):
                    raise ShapeError("Mixing grayscale and color images not supported")
                if image.dtype != dt0:
                    raise TypeError("Images have different type")

        elif type == 'video':
            # In case "video", use OpenCV to capture frames from video file. Revert the implicit
            # conversion from RGB to BGR in OpenCV input.
            cap = VideoCapture(names)
            self.number = int(cap.get(CAP_PROP_FRAME_COUNT))
            self.frames_original = []
            self.signal_step_size = max(int(self.number / 10), 1)
            for frame_index in range(self.number):
                # After every "signal_step_size"th frame, send a progress signal to the main GUI.
                if self.progress_signal is not None and frame_index%self.signal_step_size == 0:
                    self.progress_signal.emit("Read all frames", int((frame_index/self.number)*100.))
                # Read the next frame.
                ret, frame = cap.read()
                if ret:
                    if convert_to_grayscale:
                        self.frames_original.append(cvtColor(frame, COLOR_BGR2GRAY))
                    else:
                        self.frames_original.append(cvtColor(frame, COLOR_BGR2RGB))
                else:
                    raise IOError("Error in reading video frame")
            cap.release()
            if self.progress_signal is not None:
                self.progress_signal.emit("Read all frames", 100)
            self.shape = self.frames_original[0].shape
            dt0 = self.frames_original[0].dtype
        else:
            raise TypeError("Image type " + type + " not supported")

        # Monochrome images are stored as 2D arrays, color images as 3D.
        if len(self.shape) == 2:
            self.color = False
        elif len(self.shape) == 3:
            self.color = True
        else:
            raise ShapeError("Image shape not supported")

        # Set the depth value of all images to either 16 or 8 bits.
        if dt0 == 'uint16':
            self.depth = 16
        elif dt0 == 'uint8':
            self.depth = 8
        else:
            raise TypeError("Frame type " + str(dt0) + " not supported")

        # Initialize lists of monochrome frames (with and without Gaussian blur) and their
        # Laplacians.
        self.frames_monochrome = None
        self.frames_monochrome_blurred = None
        self.frames_monochrome_blurred_laplacian = None
        self.used_alignment_points = None

    def frames(self, index):
        """
        Look up the original frame object with a given index.

        :param index: Frame index
        :return: Frame with index "index".
        """

        # # Code for positioning at arbitrary index.
        # # Check for valid frame number.
        # if 0 <= index < self.number:
        #     # Set frame position
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        if not 0<=index<self.number:
            raise ArgumentError("Frame index " + str(index) + " is out of bounds")
        # print ("Accessing frame " + str(index))
        return self.frames_original[index]

    def frames_mono(self, index):
        """
        Look up the monochrome version of the frame object with a given index.

        :param index: Frame index
        :return: Monochrome frame with index "index".
        """

        if not 0 <= index < self.number:
            raise ArgumentError("Frame index " + str(index) + " is out of bounds")
        if self.frames_monochrome is not None:
            # print("Accessing frame monochrome " + str(index))
            return self.frames_monochrome[index]
        else:
            raise WrongOrderingError("Attempt to look up a monochrome frame before computing it")

    def frames_mono_blurred(self, index):
        """
        Look up a Gaussian-blurred frame object with a given index.

        :param index: Frame index
        :return: Gaussian-blurred frame with index "index".
        """

        if not 0 <= index < self.number:
            raise ArgumentError("Frame index " + str(index) + " is out of bounds")
        if self.frames_monochrome_blurred is not None:
            # print("Accessing frame with Gaussian blur " + str(index))
            return self.frames_monochrome_blurred[index]
        else:
            raise WrongOrderingError("Attempt to look up a Gaussian-blurred frame version before"
                                     " computing it")

    def frames_mono_blurred_laplacian(self, index):
        """
        Look up a Laplacian-of-Gaussian of a frame object with a given index.

        :param index: Frame index
        :return: LoG of a frame with index "index".
        """

        if not 0 <= index < self.number:
            raise ArgumentError("Frame index " + str(index) + " is out of bounds")
        if self.frames_monochrome_blurred_laplacian is not None:
            # print("Accessing LoG number " + str(index))
            return self.frames_monochrome_blurred_laplacian[index]
        else:
            raise WrongOrderingError("Attempt to look up a LoG frame version before computing it")

    def reset_alignment_point_lists(self):
        """
        Every frame keeps a list with the alignment points for which this frame is among the
        sharpest ones (so it is used in stacking). Reset this list for all frames.

        :return: -
        """

        # For every frame initialize the list with used alignment points.
        self.used_alignment_points = [[] for frame_index in range(self.number)]

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
            self.frames_monochrome = []
        elif self.depth == 8:
            self.frames_monochrome = self.frames_original

        self.frames_monochrome_blurred = []
        self.frames_monochrome_blurred_laplacian = []

        for frame_index, frame in enumerate(self.frames_original):
            # After every "signal_step_size"th frame, send a progress signal to the main GUI.
            if self.progress_signal is not None and frame_index % self.signal_step_size == 0:
                self.progress_signal.emit("Gaussians / Laplacians",
                                          int((frame_index / self.number) * 100.))

            # If frames are in color mode, or the depth is 16bit, produce a 8bit B/W version.
            if self.color or self.depth != 8:
                # If frames are in color mode, create a monochrome version with same depth.
                if self.color:
                    if color == 'panchromatic':
                        frame_mono = cvtColor(frame, COLOR_BGR2GRAY)
                    else:
                        frame_mono = frame[:, :, colors.index(color)]
                else:
                    frame_mono = frame

                # If depth is larger than 8bit, reduce the depth to 8bit.
                if self.depth != 8:
                    frame_mono = ((frame_mono) / 255.).astype(np.uint8)

                self.frames_monochrome.append(frame_mono)

            # Add a version of the frame with Gaussian blur added.
            frame_monochrome_blurred = GaussianBlur(self.frames_monochrome[frame_index], (
                self.configuration.frames_gauss_width, self.configuration.frames_gauss_width), 0)
            self.frames_monochrome_blurred.append(frame_monochrome_blurred)

            # Compute the scaling factor for "convertScaleAbs" depending on the data type of the
            # monochrome image. The Laplacian is 8bit. If the monochrome image is 16bit, values
            # have to be scaled down.

            # Add the Laplacian of the down-sampled blurred image.
            if self.configuration.rank_frames_method == "Laplace":
                self.frames_monochrome_blurred_laplacian.append(convertScaleAbs(Laplacian(
                    frame_monochrome_blurred[::self.configuration.align_frames_sampling_stride,
                    ::self.configuration.align_frames_sampling_stride], CV_32F),
                    alpha=1))

        if self.progress_signal is not None:
            self.progress_signal.emit("Gaussians / Laplacians", 100)

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
        elif path.exists(filename):
            remove(filename)

        # Write the image to the file. Before writing, convert the internal RGB representation into
        # the BGR representation assumed by OpenCV.
        if color:
            imwrite(str(filename), cvtColor(image, COLOR_RGB2BGR))
        else:
            imwrite(str(filename), image)


if __name__ == "__main__":

    # Images can either be extracted from a video file or a batch of single photographs. Select
    # the example for the test run.
    type = 'image'
    if type == 'image':
        # names = glob('Images/2012_*.tif')
        # names = glob('D:\SW-Development\Python\PlanetarySystemStacker\Examples\Moon_2011-04-10\South\*.TIF')
        names = glob('D:\SW-Development\Python\PlanetarySystemStacker\Examples\Moon_2019-01-20\Images\*.TIF')
    else:
        names = 'Videos/short_video.avi'

    # Get configuration parameters.
    configuration = Configuration()
    convert_to_grayscale = False

    start = time()
    try:
        frames_old = FramesOld(configuration, names, type=type,
                               convert_to_grayscale=convert_to_grayscale)
        print("Frames old: Number of images read: " + str(frames_old.number))
        print("Image shape: " + str(frames_old.shape))
    except Error as e:
        print("Error: " + e.message)
        exit()
    end = time()
    print("Read " + str(frames_old.number) + " frames in " + str(end-start) + " seconds.")

    # Create monochrome versions of all frames. If the original frames are monochrome, just point
    # the monochrome frame list to the original images (no deep copy!).
    try:
        frames_old.add_monochrome(configuration.frames_mono_channel)
    except ArgumentError as e:
        print("Error: " + e.message)
        exit()

    # try:
    #     frames = Frames(configuration, names, type=type, convert_to_grayscale=convert_to_grayscale,
    #                     buffer_original=False, buffer_monochrome=False, buffer_gaussian=False,
    #                     buffer_laplacian=False)
    #     print("\nFrames: Number of images: " + str(frames.number))
    #     print("Image shape: " + str(frames.shape))
    # except Error as e:
    #     print("Error: " + e.message)
    #     exit()
    #
    # indices = [1, 0, 2, 2, 8]
    #
    # for index in indices:
    #     frame_old = frames_old.frames_mono_blurred_laplacian(index)
    #     frame = frames.frames_mono_blurred_laplacian(index)
    #     difference = np.max(np.abs(frame - frame_old))
    #     print ("Error for frame " + str(index) + ": " + str(difference))
