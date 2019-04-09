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
    COLOR_BGR2RGB, GaussianBlur, Laplacian, CV_32F, COLOR_RGB2BGR, imwrite, convertScaleAbs, \
    CAP_PROP_POS_FRAMES
from math import ceil
from scipy import misc

from configuration import Configuration
from exceptions import TypeError, ShapeError, ArgumentError
from frames_old import FramesOld


class Frames(object):
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

    def __init__(self, configuration, names, type='video', convert_to_grayscale=False,
                 progress_signal=None, buffer_original=True, buffer_monochrome=False,
                 buffer_gaussian=True, buffer_laplacian=True):
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

        # If the original frames are to be buffered, read them in one go. In this case, a progress
        # bar is displayed in the main GUI.
        if self.buffer_original:
            if self.type == 'image':
                # Use scipy.misc to read in image files. If "convert_to_grayscale" is True, convert
                # pixel values to 32bit floats.
                self.number = len(self.names)
                self.signal_step_size = max(int(self.number / 10), 1)
                if self.convert_to_grayscale:
                    self.frames_original = [misc.imread(path, mode='F') for path in self.names]
                else:
                    self.frames_original = []
                    for frame_index, path in enumerate(self.names):
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
                self.dt0 = self.frames_original[0].dtype

                # Test if all images have the same shape, color type and depth.
                # If not, raise an exception.
                for image in self.frames_original:
                    if image.shape != self.shape:
                        raise ShapeError("Images have different size")
                    elif len(self.shape) != len(image.shape):
                        raise ShapeError("Mixing grayscale and color images not supported")
                    if image.dtype != self.dt0:
                        raise TypeError("Images have different type")

            elif self.type == 'video':
                # In case "video", use OpenCV to capture frames from video file. Revert the implicit
                # conversion from RGB to BGR in OpenCV input.
                self.cap = VideoCapture(self.names)
                self.number = int(self.cap.get(CAP_PROP_FRAME_COUNT))
                self.frames_original = []
                self.signal_step_size = max(int(self.number / 10), 1)
                for frame_index in range(self.number):
                    # After every "signal_step_size"th frame, send a progress signal to the main GUI.
                    if self.progress_signal is not None and frame_index%self.signal_step_size == 0:
                        self.progress_signal.emit("Read all frames", int((frame_index/self.number)*100.))
                    # Read the next frame.
                    ret, frame = self.cap.read()
                    if ret:
                        if self.convert_to_grayscale:
                            self.frames_original.append(cvtColor(frame, COLOR_BGR2GRAY))
                        else:
                            self.frames_original.append(cvtColor(frame, COLOR_BGR2RGB))
                    else:
                        raise IOError("Error in reading video frame")
                self.cap.release()
                if self.progress_signal is not None:
                    self.progress_signal.emit("Read all frames", 100)
                self.shape = self.frames_original[0].shape
                self.dt0 = self.frames_original[0].dtype
            else:
                raise TypeError("Image type " + self.type + " not supported")

            # Monochrome images are stored as 2D arrays, color images as 3D.
            if len(self.shape) == 2:
                self.color = False
            elif len(self.shape) == 3:
                self.color = True
            else:
                raise ShapeError("Image shape not supported")

            # Set the depth value of all images to either 16 or 8 bits.
            if self.dt0 == 'uint16':
                self.depth = 16
            elif self.dt0 == 'uint8':
                self.depth = 8
            else:
                raise TypeError("Frame type " + str(self.dt0) + " not supported")

        else:
            if self.type == 'image':
                self.number = len(names)
            elif self.type == 'video':
                self.cap = VideoCapture(names)
                self.number = int(self.cap.get(CAP_PROP_FRAME_COUNT))

            # Initialize metadata
            self.shape = None
            self.depth = None
            self.dt0 = None
            self.color = None
            self.frames_original = [None for index in range(self.number)]

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

    def frames(self, index):
        """
        Read or look up the original frame object with a given index.

        :param index: Frame index
        :return: Frame with index "index".
        """

        if not 0<=index<self.number:
            raise ArgumentError("Frame index " + str(index) + " is out of bounds")
        # print ("Accessing frame " + str(index))

        # The original frames are buffered. Just return the frame.
        if self.buffer_original:
            return self.frames_original[index]

        # This frame has been cached. Just return it.
        if self.original_available_index == index:
            return self.original_available

        # The frame has not been stored for re-use, read it.
        else:
            if self.type == 'image':
                if self.convert_to_grayscale:
                    frame = misc.imread(self.names[index], mode='F')
                else:
                    frame = imread(self.names[index], -1)
            else:
                # Set the read position in the file to frame "index", and read the frame.
                if index != self.original_available_index + 1:
                    self.cap.set(CAP_PROP_POS_FRAMES, index)
                ret, frame = self.cap.read()
                if ret:
                    if self.convert_to_grayscale:
                        frame = cvtColor(frame, COLOR_BGR2GRAY)
                    else:
                        frame = cvtColor(frame, COLOR_BGR2RGB)
                else:
                    raise IOError("Error in reading video frame")

            # Cache the frame just read.
            self.original_available = frame
            self.original_available_index = index

            # For the first frame read, set image metadata.
            if self.shape is None:
                self.shape = frame.shape
                # Monochrome images are stored as 2D arrays, color images as 3D.
                if len(self.shape) == 2:
                    self.color = False
                elif len(self.shape) == 3:
                    self.color = True
                else:
                    raise ShapeError("Image shape not supported")

                self.dt0 = frame.dtype
                # Set the depth value of all images to either 16 or 8 bits.
                if self.dt0 == 'uint16':
                    self.depth = 16
                elif self.dt0 == 'uint8':
                    self.depth = 8
                else:
                    raise TypeError("Frame type " + str(self.dt0) + " not supported")

            # For every other frame, check for consistency.
            else:
                if len(frame.shape) != len(self.shape):
                    raise ShapeError("Mixing grayscale and color images not supported")
                elif frame.shape != self.shape:
                    raise ShapeError("Images have different size")
                if frame.dtype != self.dt0:
                    raise TypeError("Images have different type")

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

            # If frames are in color mode, or the depth is 16bit, produce a 8bit B/W version.
            if self.color or self.depth != 8:
                # If frames are in color mode, create a monochrome version with same depth.
                if self.color:
                    if self.color_index == 3:
                        frame_mono = cvtColor(frame_original, COLOR_BGR2GRAY)
                    else:
                        frame_mono = frame_original[:, :, self.color_index]
                else:
                    frame_mono = frame_original

                # If depth is larger than 8bit, reduce the depth to 8bit.
                if self.depth != 8:
                    frame_mono = ((frame_mono) / 255.).astype(np.uint8)

            # Frames are in 8bit B/W mode already
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

            # Compute a version of the frame with Gaussian blur added.
            frame_monochrome_blurred = GaussianBlur(frame_mono,
                (self.configuration.frames_gauss_width, self.configuration.frames_gauss_width), 0)

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
                    alpha=1)

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


def access_pattern(frames_object, average_frame_percent):
    """
    Simulate the access pattern of PSS to frame data, without any other activity in between. Return
    the overall time.

    :param frames_object: Frames object to access frames.
    :param average_frame_percent: Percentage of frames for average image computation.
    :return: Total time in seconds.
    """


    number = frames.number
    average_frame_number = max(
        ceil(number * average_frame_percent / 100.), 1)
    start = time()

    for index in range(number):
        frames_object.frames_mono_blurred_laplacian(index)

    frames_object.frames_mono_blurred(number-1)
    frames_object.frames_mono_blurred(number-1)

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

if __name__ == "__main__":

    # Images can either be extracted from a video file or a batch of single photographs. Select
    # the example for the test run.
    type = 'video'
    version = 'frames_old'
    buffering_level = 4

    if type == 'image':
        # names = glob('Images/2012_*.tif')
        # names = glob('D:\SW-Development\Python\PlanetarySystemStacker\Examples\Moon_2011-04-10\South\*.TIF')
        names = glob('D:\SW-Development\Python\PlanetarySystemStacker\Examples\Moon_2019-01-20\Images\*.TIF')
    else:
        # names = 'Videos/another_short_video.avi'
        names = 'Videos/Moon_Tile-024_043939.avi'

    # Get configuration parameters.
    configuration = Configuration()

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
            frames = Frames(configuration, names, type=type, convert_to_grayscale=False,
                   buffer_original=buffer_original, buffer_monochrome=buffer_monochrome,
                   buffer_gaussian=buffer_gaussian, buffer_laplacian=buffer_laplacian)
        except Exception as e:
            print("Error: " + e.message)
            exit()
    else:
        try:
            frames = FramesOld(configuration, names, type=type, convert_to_grayscale=False)
            frames.add_monochrome(configuration.frames_mono_channel)
        except Exception as e:
            print("Error: " + e.message)
            exit()
    initialization_time = time() - start

    print("Number of images read: " + str(frames.number))
    print("Image shape: " + str(frames.shape))

    total_access_time = access_pattern(frames, configuration.align_frames_average_frame_percent)

    print("\nInitialization time: {0:7.3f}, frame accesses and variant computations: {1:7.3f},"
          " total: {2:7.3f} (seconds)".format(initialization_time, total_access_time,
                                              initialization_time+total_access_time))
