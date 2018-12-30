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

import matplotlib.pyplot as plt
from skimage import img_as_ubyte

from align_frames import AlignFrames
from alignment_points import AlignmentPoints
from configuration import Configuration
from exceptions import NotSupportedError, InternalError
from frames import Frames
from rank_frames import RankFrames
from stack_frames import StackFrames
from timer import timer

if __name__ == "__main__":
    """
    This File contains a test main program. It goes through the whole process without using a 
    graphical unser interface. It is not used in production runs.
    
    """

    mkl_rt = ctypes.CDLL('mkl_rt.dll')
    mkl_get_max_threads = mkl_rt.mkl_get_max_threads


    def mkl_set_num_threads(cores):
        mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))


    mkl_set_num_threads(2)
    print("Number of threads used by mkl: " + str(mkl_get_max_threads()))

    # Initalize the timer object used to measure execution times of program sections.
    my_timer = timer()
    # Images can either be extracted from a video file or a batch of single photographs. Select
    # the example for the test run.
    type = 'video'
    if type == 'image':
        input_file = '2012'
        names = glob.glob('Images/2012*.tif')
        # names = glob.glob('Images/Moon_Tile-031*ap85_8b.tif')
        # names = glob.glob('Images/Example-3*.jpg')
    else:
        # input_file = 'short_video'
        # input_file = 'another_short_video'
        input_file = 'Moon_Tile-024_043939'
        names = 'Videos/' + input_file + '.avi'
    print(names)

    my_timer.create('Execution over all')

    # Get configuration parameters.
    configuration = Configuration()

    # Read the frames.
    my_timer.create('Read all frames')
    try:
        frames = Frames(configuration, names, type=type, convert_to_grayscale=False)
        print("Number of images read: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Exception as e:
        print("Error: " + e.message)
        exit()
    my_timer.stop('Read all frames')

    # Rank the frames by their overall local contrast.
    my_timer.create('Ranking images')
    rank_frames = RankFrames(frames, configuration)
    rank_frames.frame_score()
    my_timer.stop('Ranking images')
    print("Index of best frame: " + str(rank_frames.frame_ranks_max_index))

    # Initialize the frame alignment object.
    align_frames = AlignFrames(frames, rank_frames, configuration)
    my_timer.create('Select optimal alignment patch')
    # Select the local rectangular patch in the image where the L gradient is highest in both x
    # and y direction. The scale factor specifies how much smaller the patch is compared to the
    # whole image frame.
    (x_low_opt, x_high_opt, y_low_opt, y_high_opt) = align_frames.select_alignment_rect(
        configuration.align_frames_rectangle_scale_factor)
    my_timer.stop('Select optimal alignment patch')

    print("optimal alignment rectangle, y_low: " + str(y_low_opt) + ", y_high: " +
          str(y_high_opt) + ", x_low: " + str(x_low_opt) + ", x_high: " + str(x_high_opt))

    # Align all frames globally relative to the frame with the highest score.
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
    my_timer.create('Compute reference frame')
    average = align_frames.average_frame()
    my_timer.stop('Compute reference frame')
    print("Average frame computed from the best " + str(
        align_frames.average_frame_number) + " frames.")

    # Initialize the AlignmentPoints object.
    my_timer.create('Initialize alignment point object')
    alignment_points = AlignmentPoints(configuration, frames, rank_frames, align_frames)
    my_timer.stop('Initialize alignment point object')

    # Create alignment points, and create an image with wll alignment point boxes and patches.
    my_timer.create('Create alignment points')
    alignment_points.create_ap_grid(average)
    my_timer.stop('Create alignment points')
    print("Number of alignment points created: " + str(len(alignment_points.alignment_points)) +
          ", aps dropped because too dim: " + str(
        len(alignment_points.alignment_points_dropped_dim)) +
          ", aps dropped because too little structure: " + str(
        len(alignment_points.alignment_points_dropped_structure)))
    color_image_with_aps = alignment_points.show_alignment_points(average)

    # For each alignment point rank frames by their quality.
    my_timer.create('Rank frames at alignment points')
    alignment_points.compute_frame_qualities()
    my_timer.stop('Rank frames at alignment points')

    # Allocate StackFrames object.
    stack_frames = StackFrames(configuration, frames, align_frames, alignment_points, my_timer)

    # Stack all frames.
    stack_frames.stack_frames()

    # Merge the stacked alignment point buffers into a single image.
    stacked_image = stack_frames.merge_alignment_point_buffers()

    # Save the stacked image as 16bit int (color or mono).
    my_timer.create('Saving the final image')
    frames.save_image('Images/' + input_file + '_stacked.tiff', stacked_image, color=frames.color)
    my_timer.stop('Saving the final image')

    # Print out timer results.
    my_timer.stop('Execution over all')
    my_timer.print()

    # Write the image with alignment points.
    frames.save_image('Images/' + input_file + '_alignment_points.tiff', color_image_with_aps,
                      color=True)

    # Show alignment points and patches
    plt.imshow(color_image_with_aps)
    plt.show()

    # Convert the stacked image to 8bit and show in Window.
    plt.imshow(img_as_ubyte(stacked_image))
    plt.show()
