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
from time import time

import matplotlib.pyplot as plt
from skimage import img_as_ubyte

from align_frames import AlignFrames
from alignment_points import AlignmentPoints
from configuration import Configuration
from frames import Frames
from quality_areas import QualityAreas
from rank_frames import RankFrames
from stack_frames import StackFrames
from timer import timer

if __name__ == "__main__":
    """
    This File contains a test main program. It goes through the whole process without using a 
    graphical unser interface. It is not used in production runs.
    
    """

    # Initalize the timer object used to measure execution times of program sections.
    my_timer = timer()
    # Images can either be extracted from a video file or a batch of single photographs. Select
    # the example for the test run.
    type = 'video'
    if type == 'image':
        names = glob.glob('Images/2012*.tif')
        # names = glob.glob('Images/Moon_Tile-031*ap85_8b.tif')
        # names = glob.glob('Images/Example-3*.jpg')
    else:
        # file = 'short_video'
        file = 'Moon_Tile-024_043939'
        names = 'Videos/' + file + '.avi'
    print(names)

    my_timer.create('Execution over all')

    # Get configuration parameters.
    configuration = Configuration()

    # Read the frames.
    my_timer.create('Read all frames')
    try:
        frames = Frames(names, type=type, convert_to_grayscale=True)
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
        configuration.alignment_rectangle_scale_factor)
    my_timer.stop('Select optimal alignment patch')

    print("optimal alignment rectangle, y_low: " + str(y_low_opt) + ", y_high: " +
          str(y_high_opt) + ", x_low: " + str(x_low_opt) + ", x_high: " + str(x_high_opt))

    # Align all frames globally relative to the frame with the highest score.
    my_timer.create('Global frame alignment')
    align_frames.align_frames()
    my_timer.stop('Global frame alignment')

    print("Intersection, y_low: " + str(align_frames.intersection_shape[0][0]) + ", y_high: "
            + str(align_frames.intersection_shape[0][1]) + ", x_low: "\
            + str(align_frames.intersection_shape[1][0]) + ", x_high: "\
            + str(align_frames.intersection_shape[1][1]))

    # Initialize the AlignmentPoints object. This includes the computation of the average frame
    # against which the alignment point shifts are measured.
    my_timer.create('Compute reference frame')
    alignment_points = AlignmentPoints(configuration, frames, rank_frames, align_frames)
    my_timer.stop('Compute reference frame')
    print("Average frame computed from the best " + str(
        alignment_points.average_frame_number) + " frames.")

    # Create a regular grid with small boxes. A subset of those boxes will be selected as
    # alignment points.
    step_size = configuration.alignment_box_step_size
    box_size = configuration.alignment_box_size
    my_timer.create('Create alignment boxes')
    alignment_points.create_alignment_boxes(step_size, box_size)
    my_timer.stop('Create alignment boxes')
    print("Number of alignment boxes created: " + str(
        len(alignment_points.alignment_boxes) * len(alignment_points.alignment_boxes[0])))
    print("Alignment point y locations: " + str(alignment_points.y_locations))
    print("Alignment point x locations: " + str(alignment_points.x_locations))

    # An alignment box is selected as an alignment point if it satisfies certain conditions
    # regarding local contrast etc.
    structure_threshold = configuration.alignment_point_structure_threshold
    brightness_threshold = configuration.alignment_point_brightness_threshold
    contrast_threshold = configuration.alignment_point_contrast_threshold
    print("Selection of alignment points, structure threshold: " + str(
        structure_threshold) + ", brightness threshold: " + str(
        brightness_threshold) + ", contrast threshold: " + str(contrast_threshold))
    my_timer.create('Select alignment points')
    alignment_points.select_alignment_points(structure_threshold, brightness_threshold,
                                             contrast_threshold)
    my_timer.stop('Select alignment points')
    print("Number of alignment points selected: " + str(len(alignment_points.alignment_points)))

    # Insert color-coded crosses at alignment box locations in the reference frame and write
    # out the image.
    reference_frame_with_alignment_points = alignment_points.show_alignment_box_types()
    frames.save_image('Images/reference_frame_with_alignment_points.jpg',
                      reference_frame_with_alignment_points, color=True)

    # Create a regular grid of quality areas. The fractional sizes of the areas in x and y,
    # as compared to the full frame, are specified in the configuration object.
    my_timer.create('Create quality areas and rank frames')
    quality_areas = QualityAreas(configuration, frames, align_frames, alignment_points)

    # For each quality area rank the frames according to the local contrast.
    quality_areas.select_best_frames()

    # Truncate the list of frames to be stacked to the same number for each quality area.
    quality_areas.truncate_best_frames()
    my_timer.stop('Create quality areas and rank frames')
    print("Number of frames to be stacked for each quality area: " + str(quality_areas.stack_size))

    # Allocate StackFrames object.
    stack_frames = StackFrames(configuration, frames, align_frames, alignment_points, quality_areas,
                               my_timer)

    # Stack all frames.
    result = stack_frames.stack_frames()

    # Save the stacked image as 16bit int (color or mono).
    my_timer.create('Save Image')
    frames.save_image('Images/' + file + '_stacked.tiff', result, color=frames.color)
    my_timer.stop('Save Image')
    my_timer.stop('Execution over all')

    # Convert to 8bit and show in Window.
    plt.imshow(img_as_ubyte(result))
    plt.show()

    # Print out timer results.
    my_timer.print()