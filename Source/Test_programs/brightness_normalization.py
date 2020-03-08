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

from os import path, remove
from os.path import splitext
from pathlib import Path

from astropy.io import fits
from cv2 import imread, cvtColor, COLOR_RGB2GRAY, COLOR_BGR2RGB, THRESH_TOZERO, threshold, \
    GaussianBlur, Laplacian, CV_32F, COLOR_RGB2BGR, imwrite, convertScaleAbs, IMREAD_UNCHANGED, flip
from cv2 import mean as cv_mean
from numpy import uint8, uint16, clip, moveaxis

from configuration import Configuration
from exceptions import TypeError, ArgumentError, Error
from frames import ImageReader, VideoReader
from rank_frames import RankFrames
from timer import timer


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
                                  THRESH_TOZERO)[1])[0]
                    # self.frames_average_brightness[index] = cv_mean(frame_mono)[0]
                    # Keep the index of the first monochrome frame as the reference index.
                    self.first_monochrome_index = index
                    self.first_monochrome = False
                # Not the first monochrome frame. Adjust brightness to match the reference.
                else:
                    self.frames_average_brightness[index] = cv_mean(
                        threshold(frame_mono, self.normalization_lower_threshold,
                                  self.normalization_upper_threshold,
                                  THRESH_TOZERO)[1])[0]
                    # self.frames_average_brightness[index] = cv_mean(frame_mono)[0]
                    frame_mono = frame_mono * (self.frames_average_brightness[self.first_monochrome_index] /
                                               self.frames_average_brightness[index])
                    # Clip the pixel values to the range allowed.
                    if frame_type == uint8:
                        clip(frame_mono, 0, 255, out=frame_mono)
                    else:
                        clip(frame_mono, 0, 65535, out=frame_mono)
                    frame_mono = frame_mono.astype(frame_type)

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


if __name__ == "__main__":
    """
    This File contains a test main program. It goes through the whole process without using a
    graphical unser interface. It is not intended to be used in production runs.
    """

    ####################################### Specify test case ######################################

    input_file = 'E:/SW-Development/Python/PlanetarySystemStacker/Examples/Jupiter/2019-05-26-0115_4-L-Jupiter_ZWO ASI290MM Mini_short.avi'

    ####################################### Specify test case end ##################################

    # Initalize the timer object used to measure execution times of program sections.
    my_timer = timer()

    print(
        "\n" +
        "*************************************************************************************\n"
        + "Start processing " + str(
            input_file) +
        "\n*************************************************************************************")
    my_timer.create('Execution over all')

    # Get configuration parameters.
    configuration = Configuration()
    configuration.initialize_configuration()

    # Read the frames.
    print("+++ Start reading frames")
    my_timer.create('Read all frames')
    try:
        frames = Frames(configuration, input_file, type='video', buffer_original=True,
                        buffer_monochrome=False, buffer_gaussian=False, buffer_laplacian=False)
        print("Number of images read: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Error as e:
        print("Error: " + str(e))
        exit()
    my_timer.stop('Read all frames')

    # Rank the frames by their overall local contrast.
    print("+++ Start ranking images")
    my_timer.create('Ranking images')
    rank_frames = RankFrames(frames, configuration)

    configuration.frames_normalization = True
    rank_frames.frame_score()
    print ("Ranks with normalization: " + str(rank_frames.quality_sorted_indices))

    # Print out timer results.
    my_timer.stop('Execution over all')
    my_timer.print()

    # Initalize the timer object used to measure execution times of program sections.
    my_timer = timer()

    average_brightness_list = frames.frames_average_brightness

    # Now do the same without normalization.

    print(
        "\n" +
        "*************************************************************************************\n"
        + "Start processing " + str(
            input_file) +
        "\n*************************************************************************************")
    my_timer.create('Execution over all')

    # Get configuration parameters.
    configuration = Configuration()
    configuration.initialize_configuration()

    # Read the frames.
    print("+++ Start reading frames")
    my_timer.create('Read all frames')
    try:
        frames = Frames(configuration, input_file, type='video', buffer_original=True,
                        buffer_monochrome=False, buffer_gaussian=False, buffer_laplacian=False)
        print("Number of images read: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Error as e:
        print("Error: " + str(e))
        exit()
    my_timer.stop('Read all frames')

    # Rank the frames by their overall local contrast.
    print("+++ Start ranking images")
    my_timer.create('Ranking images')
    rank_frames = RankFrames(frames, configuration)

    configuration.frames_normalization = False
    rank_frames.frame_score_no_normalization(average_brightness_list)
    print("Ranks without normalization: " + str(rank_frames.quality_sorted_indices))

    # Print out timer results.
    my_timer.stop('Execution over all')
    my_timer.print()

