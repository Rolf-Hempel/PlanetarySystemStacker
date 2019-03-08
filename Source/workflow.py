# -*- coding: utf-8; -*-
"""
Copyright (c) 2019 Rolf Hempel, rolf6419@gmx.de

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

import os
import ctypes

from PyQt5 import QtCore

from align_frames import AlignFrames
from exceptions import NotSupportedError, InternalError
from frames import Frames
from rank_frames import RankFrames
from alignment_points import AlignmentPoints
from stack_frames import StackFrames
from timer import timer


class Workflow(QtCore.QObject):

    work_next_task_signal = QtCore.pyqtSignal(str)

    def __init__(self, main_gui):
        super(Workflow, self).__init__()
        self.main_gui = main_gui
        self.configuration = main_gui.configuration

        # Initalize the timer object used to measure execution times of program sections.
        self.my_timer = timer()

        self.frames = None
        self.rank_frames = None
        self.align_frames = None
        self.alignment_points = None
        self.stack_frames = None
        self.stacked_image_name = None

        mkl_rt = ctypes.CDLL('mkl_rt.dll')
        mkl_get_max_threads = mkl_rt.mkl_get_max_threads

        def mkl_set_num_threads(cores):
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))

        mkl_set_num_threads(2)
        print("Number of threads used by mkl: " + str(mkl_get_max_threads()))

    @QtCore.pyqtSlot()
    def execute_frames(self, input_name, input_type, convert_to_grayscale):

        # Images can either be extracted from a video file or a batch of single photographs. Select
        # the example for the test run.

        # For video file input, the Frames constructor expects the video file name for "names".
        if input_type == 'video':
            names = input_name
        # For single image input, the Frames constructor expects a list of image file names for
        # "names".
        else:
            names = os.listdir(input_name)
            names = [os.path.join(input_name, name) for name in names]
        self.stacked_image_name = input_name + '.stacked.tiff'

        print(
            "\n" +
            "***********************************************************************************\n"
            + "Start processing " + str(input_name) +
            "\n***********************************************************************************")
        self.my_timer.create('Execution over all')

        # Read the frames.
        print("+++ Start reading frames")
        self.my_timer.create('Read all frames')
        try:
            self.frames = Frames(self.configuration, names, type=input_type,
                            convert_to_grayscale=convert_to_grayscale)
            print("Number of images read: " + str(self.frames.number))
            print("Image shape: " + str(self.frames.shape))
        except Exception as e:
            print("Error: " + str(e))
            exit()
        self.my_timer.stop('Read all frames')

        # The whole quality analysis and shift determination process is performed on a monochrome
        # version of the frames. If the original frames are in RGB, the monochrome channel can be
        # selected via a configuration parameter. Add a list of monochrome images for all frames to
        # the "Frames" object.
        print("+++ Start creating blurred monochrome images and Laplacians")
        self.my_timer.create('Blurred monochrome images and Laplacians')
        self.frames.add_monochrome(self.configuration.frames_mono_channel)
        self.my_timer.stop('Blurred monochrome images and Laplacians')

        self.work_next_task_signal.emit("Rank frames")

    @QtCore.pyqtSlot()
    def execute_rank_frames(self):

        # Rank the frames by their overall local contrast.
        print("+++ Start ranking images")
        self.my_timer.create('Ranking images')
        self.rank_frames = RankFrames(self.frames, self.configuration)
        self.rank_frames.frame_score()
        self.my_timer.stop('Ranking images')
        print("Index of best frame: " + str(self.rank_frames.frame_ranks_max_index))

        self.work_next_task_signal.emit("Align frames")

    @QtCore.pyqtSlot()
    def execute_align_frames(self, auto_execution, x_low_opt, x_high_opt, y_low_opt, y_high_opt):

        # Initialize the frame alignment object.
        self.align_frames = AlignFrames(self.frames, self.rank_frames, self.configuration)

        if self.configuration.align_frames_mode == "Surface":

            # Compute the local rectangular patch in the image where the L gradient is highest
            # in both x and y direction. The scale factor specifies how much smaller the patch
            # is compared to the whole image frame. In batch mode, variable "auto_execution" is
            # set to "True", and the automatic patch computation is the only option.
            if auto_execution or self.configuration.align_frames_automation:
                self.my_timer.create('Select optimal alignment patch')
                (y_low_opt, y_high_opt, x_low_opt, x_high_opt) = \
                    self.align_frames.select_alignment_rect(
                        self.configuration.align_frames_rectangle_scale_factor)
                self.my_timer.stop('Select optimal alignment patch')

            # As an alternative, set the coordinates of the rectangular patch explicitly.
            else:
                self.align_frames.set_alignment_rect(y_low_opt, y_high_opt, x_low_opt, x_high_opt)

            print("Alignment rectangle, y_low: " + str(y_low_opt) + ", y_high: " + str(
                y_high_opt) + ", x_low: " + str(x_low_opt) + ", x_high: " + str(x_high_opt))

        # Align all frames globally relative to the frame with the highest score.
        print("+++ Start aligning all frames")
        self.my_timer.create('Global frame alignment')
        try:
            self.align_frames.align_frames()
        except NotSupportedError as e:
            print("Error: " + e.message)
            exit()
        except InternalError as e:
            print("Warning: " + e.message)
        self.my_timer.stop('Global frame alignment')

        print("Intersection, y_low: " + str(
            self.align_frames.intersection_shape[0][0]) + ", y_high: " + str(
            self.align_frames.intersection_shape[0][1]) + ", x_low: " + str(
            self.align_frames.intersection_shape[1][0]) + ", x_high: " + str(
            self.align_frames.intersection_shape[1][1]))

        # Compute the average frame.
        print("+++ Start computing average frame")
        self.my_timer.create('Compute reference frame')
        self.align_frames.average_frame()
        self.my_timer.stop('Compute reference frame')
        print("Average frame computed from the best " + str(
            self.align_frames.average_frame_number) + " frames.")

        self.work_next_task_signal.emit("Set ROI")

    @QtCore.pyqtSlot()
    def execute_set_roi(self, y_min, y_max, x_min, x_max):

        print("+++ Start setting ROI and computing new average frame")
        self.my_timer.create('Setting ROI and new reference')
        self.align_frames.set_roi(y_min, y_max, x_min, x_max)
        self.my_timer.stop('Setting ROI and new reference')

        self.work_next_task_signal.emit("Set alignment points")

    @QtCore.pyqtSlot()
    def execute_set_alignment_points(self):

        # Initialize the AlignmentPoints object.
        self.my_timer.create('Initialize alignment point object')
        self.alignment_points = AlignmentPoints(self.configuration, self.frames, self.rank_frames,
                                           self.align_frames)
        self.my_timer.stop('Initialize alignment point object')

        # Create alignment points, and create an image with wll alignment point boxes and patches.
        print("+++ Start creating alignment points")
        self.my_timer.create('Create alignment points')

        # If a ROI is selected, alignment points are created in the ROI window only.
        self.alignment_points.create_ap_grid()

        self.my_timer.stop('Create alignment points')
        print("Number of alignment points selected: " + str(
            len(self.alignment_points.alignment_points)) +
              ", aps dropped because too dim: " + str(
            self.alignment_points.alignment_points_dropped_dim) +
              ", aps dropped because too little structure: " + str(
            self.alignment_points.alignment_points_dropped_structure))

        self.work_next_task_signal.emit("Compute frames qualities")

    @QtCore.pyqtSlot()
    def execute_compute_frame_qualities(self):

        # For each alignment point rank frames by their quality.
        self.my_timer.create('Rank frames at alignment points')
        print("+++ Start ranking frames at alignment points")
        self.alignment_points.compute_frame_qualities()
        self.my_timer.stop('Rank frames at alignment points')

        self.work_next_task_signal.emit("Stack frames")

    @QtCore.pyqtSlot()
    def execute_stack_frames(self):

        # Allocate StackFrames object.
        self.stack_frames = StackFrames(self.configuration, self.frames, self.align_frames,
                                   self.alignment_points, self.my_timer)

        # Stack all frames.
        print("+++ Start stacking frames")
        self.stack_frames.stack_frames()

        # Merge the stacked alignment point buffers into a single image.
        print("+++ Start merging alignment patches")
        self.stack_frames.merge_alignment_point_buffers()

        self.work_next_task_signal.emit("Save stacked image")

    @QtCore.pyqtSlot()
    def execute_save_stacked_image(self):

        # Save the stacked image as 16bit int (color or mono).
        self.my_timer.create('Saving the final image')
        self.frames.save_image(self.stacked_image_name, self.stack_frames.stacked_image,
                               color=self.frames.color)
        self.my_timer.stop('Saving the final image')

        self.work_next_task_signal.emit("Next job")
