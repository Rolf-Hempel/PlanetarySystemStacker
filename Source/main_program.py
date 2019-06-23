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

import ctypes
import glob
import sys
import os
import traceback

import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from PyQt5 import QtWidgets

from align_frames import AlignFrames
from alignment_points import AlignmentPoints
from alignment_point_editor import AlignmentPointEditorWidget
from configuration import Configuration
from exceptions import NotSupportedError, InternalError
from frames import Frames
from rank_frames import RankFrames
from stack_frames import StackFrames
from timer import timer

def workflow(input_name, input_type='video', roi=None, convert_to_grayscale=False,
             automatic_ap_creation=True):
    """
    Execute the whole stacking workflow for a test case. This can either use a video file (.avi)
    or still images stored in a single directory.

    :param input_name: Video file (.avi) or name of a directory containing still images
    :param input_type: Either "video" or "image" (see "input_name")
    :param roi: If specified, tuple (y_low, y_high, x_low, x_high) with pixel bounds for "region
                of interest"
    :param convert_to_grayscale: If True, input frames are converted to grayscale mode before
                                 processing. In this case, the stacked image is grayscale as well.
    :return: average, [average_roi,] color_image_with_aps, stacked_image
             with: - average: global mean frame
                   - average_roi: mean frame restricted to ROI (only if roi is specified)
                   - color_image_with_aps: mean frame overlaid with alignment points and their
                                           boxes (white) and patches (green)
    """

    # Initalize the timer object used to measure execution times of program sections.
    my_timer = timer()
    # Images can either be extracted from a video file or a batch of single photographs. Select
    # the example for the test run.

    # For video file input, the Frames constructor expects the video file name for "names".
    if input_type == 'video':
        names = input_name
    # For single image input, the Frames constructor expects a list of image file names for "names".
    else:
        names = os.listdir(input_name)
        names = [os.path.join(input_name, name) for name in names]
    stacked_image_name = input_name + '.stacked.tiff'

    # The name of the alignment point visualization file is derived from the input video name or
    # the input directory name.
    ap_image_name = input_name + ".aps.jpg"

    print("\n" +
          "*************************************************************************************\n"
          + "Start processing " + str(input_name) +
          "\n*************************************************************************************")
    my_timer.create('Execution over all')

    # Get configuration parameters.
    configuration = Configuration()
    configuration.align_frames_method = 'Planet'

    # Read the frames.
    print("+++ Start reading frames")
    my_timer.create('Read all frames')
    try:
        frames = Frames(configuration, names, type=input_type,
                        convert_to_grayscale=convert_to_grayscale, buffer_gaussian=False, buffer_laplacian=False)
        print("Number of images read: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Exception as e:
        print("Error: " + str(e))
        exit()
    my_timer.stop('Read all frames')

    # Rank the frames by their overall local contrast.
    print("+++ Start ranking images")
    my_timer.create('Ranking images')
    rank_frames = RankFrames(frames, configuration)
    rank_frames.frame_score()
    my_timer.stop('Ranking images')
    print("Index of best frame: " + str(rank_frames.frame_ranks_max_index))

    # Initialize the frame alignment object.
    align_frames = AlignFrames(frames, rank_frames, configuration)

    if configuration.align_frames_mode == "Surface":
        my_timer.create('Select optimal alignment patch')
        # Select the local rectangular patch in the image where the L gradient is highest in both x
        # and y direction. The scale factor specifies how much smaller the patch is compared to the
        # whole image frame.
        (y_low_opt, y_high_opt, x_low_opt, x_high_opt) = align_frames.compute_alignment_rect(
            configuration.align_frames_rectangle_scale_factor)
        my_timer.stop('Select optimal alignment patch')

        print("optimal alignment rectangle, y_low: " + str(y_low_opt) + ", y_high: " +
              str(y_high_opt) + ", x_low: " + str(x_low_opt) + ", x_high: " + str(x_high_opt))

    # Align all frames globally relative to the frame with the highest score.
    print("+++ Start aligning all frames")
    my_timer.create('Global frame alignment')
    try:
        align_frames.align_frames()
    except NotSupportedError as e:
        print("Error: " + e.message)
        exit()
    except InternalError as e:
        print("Warning: " + e.message)
    my_timer.stop('Global frame alignment')

    print("Intersection, y_low: " + str(align_frames.intersection_shape[0][0]) + ", y_high: "
          + str(align_frames.intersection_shape[0][1]) + ", x_low: " \
          + str(align_frames.intersection_shape[1][0]) + ", x_high: " \
          + str(align_frames.intersection_shape[1][1]))

    # Compute the average frame.
    print("+++ Start computing average frame")
    my_timer.create('Compute reference frame')
    average = align_frames.average_frame()
    my_timer.stop('Compute reference frame')
    print("Average frame computed from the best " + str(
        align_frames.average_frame_number) + " frames.")

    # If the ROI is to be set to a smaller size than the whole intersection, do so.
    if roi:
        print("+++ Start setting ROI and computing new average frame")
        my_timer.create('Setting ROI and new reference')
        average_roi = align_frames.set_roi(roi[0], roi[1], roi[2], roi[3])
        my_timer.stop('Setting ROI and new reference')

    # Initialize the AlignmentPoints object.
    my_timer.create('Initialize alignment point object')
    alignment_points = AlignmentPoints(configuration, frames, rank_frames, align_frames)
    my_timer.stop('Initialize alignment point object')

    if automatic_ap_creation:
        # Create alignment points, and create an image with wll alignment point boxes and patches.
        print("+++ Start creating alignment points")
        my_timer.create('Create alignment points')

        # If a ROI is selected, alignment points are created in the ROI window only.
        alignment_points.create_ap_grid()

        my_timer.stop('Create alignment points')
        print("Number of alignment points selected: " + str(len(alignment_points.alignment_points)) +
              ", aps dropped because too dim: " + str(alignment_points.alignment_points_dropped_dim) +
              ", aps dropped because too little structure: " + str(
              alignment_points.alignment_points_dropped_structure))
    else:
        # Open the alignment point editor.
        app = QtWidgets.QApplication(sys.argv)
        alignment_point_editor = AlignmentPointEditorWidget(None, configuration, align_frames,
                                                            alignment_points, None)
        alignment_point_editor.setMinimumSize(800, 600)
        alignment_point_editor.showMaximized()
        app.exec_()

        print("After AP editing, number of APs: " + str(len(alignment_points.alignment_points)))
        count_updates = 0
        for ap in alignment_points.alignment_points:
            if ap['reference_box'] is not None:
                continue
            count_updates += 1
            AlignmentPoints.set_reference_box(ap, align_frames.mean_frame)
        print("Buffers allocated for " + str(count_updates) + " alignment points.")

    # Produce an overview image showing all alignment points.
    if roi:
        color_image_with_aps = alignment_points.show_alignment_points(average_roi)
    else:
        color_image_with_aps = alignment_points.show_alignment_points(average)

    # For each alignment point rank frames by their quality.
    my_timer.create('Rank frames at alignment points')
    print("+++ Start ranking frames at alignment points")
    alignment_points.compute_frame_qualities()
    my_timer.stop('Rank frames at alignment points')

    # Allocate StackFrames object.
    stack_frames = StackFrames(configuration, frames, align_frames, alignment_points, my_timer)

    # Stack all frames.
    print("+++ Start stacking frames")
    stack_frames.stack_frames()

    # Merge the stacked alignment point buffers into a single image.
    print("+++ Start merging alignment patches")
    stacked_image = stack_frames.merge_alignment_point_buffers()

    # Save the stacked image as 16bit int (color or mono).
    my_timer.create('Saving the final image')
    frames.save_image(stacked_image_name, stacked_image, color=frames.color)
    my_timer.stop('Saving the final image')

    # Print out timer results.
    my_timer.stop('Execution over all')
    my_timer.print()

    # Write the image with alignment points.
    frames.save_image(ap_image_name, color_image_with_aps, color=True)

    # If a ROI is selected, return both the original and the reduced-size average frame.
    if roi:
        return average, average_roi, color_image_with_aps, stacked_image
    else:
        return average, color_image_with_aps, stacked_image


if __name__ == "__main__":
    """
    This File contains a test main program. It goes through the whole process without using a 
    graphical unser interface. It is not intended to be used in production runs.
    """

    ####################################### Specify test case ######################################
    redirect_stdout = True
    show_results = True
    # input_type = 'video'
    # input_directory = 'D:/SW-Development/Python/PlanetarySystemStacker/Examples/Moon_2018-03-24'
    input_type = 'image'
    input_directory = 'D:/Bilder/2019/06/2019-06-17_MondJupiter'
    convert_to_grayscale = False
    automatic_ap_creation = True
    roi = None
    # roi = (400, 700, 300, 800)
    ####################################### Specify test case end ##################################

    # Redirect standard output to a file if requested.
    if redirect_stdout:
        stdout_saved = sys.stdout
        protocol_file = open(os.path.join(input_directory, 'Protocol.txt'), 'a')
        sys.stdout = protocol_file

    mkl_rt = ctypes.CDLL('mkl_rt.dll')
    mkl_get_max_threads = mkl_rt.mkl_get_max_threads

    def mkl_set_num_threads(cores):
        mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))

    mkl_set_num_threads(2)
    print("Number of threads used by mkl: " + str(mkl_get_max_threads()))


    # For videos, batch processing is done for all videos in the input directory.
    if input_type == 'video':
        input_names = glob.glob(os.path.join(input_directory, '*.avi'))
    # For images, it is assumed that the input directory contains one or several directories with
    # image files.
    else:
        if input_type != 'image':
            print("WARNING: Wrong input spec, assuming 'image'")
        input_directory_content = os.listdir(input_directory)
        # input_names = [dir for dir in input_directory_content if os.path.isdir(dir)]
        input_names = []
        for dir in input_directory_content:
            dir_abs = os.path.join(input_directory, dir)
            if os.path.isdir(dir_abs):
                input_names.append(dir_abs)

    print("Inputs: ", input_names)
    
    # Start the processing workflow in batch mode for all AVIs / file directories.
    try:
        for input_name in input_names:
            if roi:
                average, average_roi, color_image_with_aps, stacked_image = workflow(input_name,
                    input_type=input_type, roi=roi, convert_to_grayscale=convert_to_grayscale,
                    automatic_ap_creation=automatic_ap_creation)
            else:
                average, color_image_with_aps, stacked_image = workflow(input_name,
                    input_type=input_type, convert_to_grayscale=convert_to_grayscale,
                    automatic_ap_creation=automatic_ap_creation)
    
            # Interrupt the workflow to display resulting images only if requested.
            if show_results:
                # Show the full average frame.
                plt.imshow(average, cmap='Greys_r')
                plt.show()
    
                if roi:
                    # Show the ROI average frame.
                    plt.imshow(average_roi, cmap='Greys_r')
                    plt.show()
    
                # Show alignment points and patches
                plt.imshow(color_image_with_aps)
                plt.show()
    
                # Convert the stacked image to 8bit and show in Window.
                plt.imshow(img_as_ubyte(stacked_image))
                plt.show()
    except:
        exec_info = sys.exc_info()
        print(exec_info[1])
        traceback.print_tb(exec_info[2])
    else:
        # Redirect stdout back to normal.
        if redirect_stdout:
            sys.stdout = stdout_saved
