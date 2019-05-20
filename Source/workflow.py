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

import sys
from ctypes import CDLL, byref, c_int
from os import listdir
from os.path import splitext, join

from PyQt5 import QtCore
from cv2 import imread, cvtColor, COLOR_BGR2RGB
from numpy import uint16, uint8

from align_frames import AlignFrames
from alignment_points import AlignmentPoints
from configuration import PostprocDataObject
from exceptions import NotSupportedError, InternalError, ArgumentError
from frames import Frames
from miscellaneous import Miscellaneous
from rank_frames import RankFrames
from stack_frames import StackFrames
from timer import timer


class Workflow(QtCore.QObject):

    work_next_task_signal = QtCore.pyqtSignal(str)
    work_current_progress_signal = QtCore.pyqtSignal(str, int)
    set_status_bar_signal = QtCore.pyqtSignal(str, str)

    def __init__(self, main_gui):
        super(Workflow, self).__init__()
        self.main_gui = main_gui
        self.configuration = main_gui.configuration

        self.my_timer = None

        self.frames = None
        self.rank_frames = None
        self.align_frames = None
        self.alignment_points = None
        self.stack_frames = None
        self.stacked_image_name = None
        self.postprocessed_image_name = None
        self.postprocessed_image = None
        self.postproc_input_image = None
        self.postproc_input_name = None
        self.job_type = None
        self.attached_log_name = None
        self.attached_log_file = None
        self.stdout_saved = None
        self.output_redirected = False
        self.protocol_file = None

        # The following code works on Windows systems only. It is not necessary, though.
        try:
            mkl_rt = CDLL('mkl_rt.dll')
            mkl_get_max_threads = mkl_rt.mkl_get_max_threads

            def mkl_set_num_threads(cores):
                mkl_rt.mkl_set_num_threads(byref(c_int(cores)))

            mkl_set_num_threads(mkl_get_max_threads())
            if self.configuration.global_parameters_protocol_level > 1:
                Miscellaneous.protocol("Number of threads used by mkl: " + str(mkl_get_max_threads()),
                                       self.attached_log_file, precede_with_timestamp=True)
        except Exception as e:
            Miscellaneous.protocol("mkl_rt.dll does not work (not a Windows system?): " + str(e),
                                   self.attached_log_file, precede_with_timestamp = True)

    @QtCore.pyqtSlot(str, str, bool)
    def execute_frames(self, input_name, input_type, convert_to_grayscale):

        # If objects are left over from previous run, delete them.
        for obj in [self.frames, self.rank_frames, self.align_frames, self.alignment_points,
                    self.stack_frames]:
            if obj is not None:
                del obj

        # Update the status bar in the main GUI.
        self.input_name = input_name
        self.set_status_bar_processing_phase("reading frames")

        # A jobs can either "stack" images or "postprocess" a single image. In the latter case,
        # input is a single image file.
        #
        # Images for stacking can either be extracted from a video file or a batch of single
        # photographs. In the first case, input_type is set to 'video', in the second case to
        # 'image'.

        if input_type == 'postproc':
            self.job_type = 'postproc'
            self.postproc_input_name = input_name

            # Reset the postprocessed image to None. This way, in saving the postprocessing result,
            # it can be checked if an image was computed in the workflow thread.
            self.postprocessed_image = None
            self.postprocessed_image_name = PostprocDataObject.set_file_name_processed(
                input_name, self.configuration.postproc_suffix)
            self.attached_log_name = splitext(input_name)[0] + '_postproc-log.txt'

        # For video file input, the Frames constructor expects the video file name for "names".
        elif input_type == 'video':
            self.job_type = 'stacking'
            names = input_name
            self.stacked_image_name = splitext(input_name)[0] + \
                                      self.configuration.stack_frames_suffix + '.tiff'
            self.attached_log_name = splitext(input_name)[0] + '_stacking-log.txt'

        # For single image input, the Frames constructor expects a list of image file names for
        # "names".
        else: # input_type = 'image'
            self.job_type = 'stacking'
            names = listdir(input_name)
            names = [join(input_name, name) for name in names]
            self.stacked_image_name = input_name + self.configuration.stack_frames_suffix + '.tiff'
            self.attached_log_name = input_name + '_stacking-log.txt'

        # Redirect stdout to a file if requested.
        if self.configuration.global_parameters_write_protocol_to_file != self.output_redirected:
            # Output currently redirected. Reset to stdout.
            if self.output_redirected:
                sys.stdout = self.stdout_saved
                self.output_redirected = False
            # Currently set to stdout, redirect to file now.
            else:
                try:
                    self.stdout_saved = sys.stdout
                    sys.stdout = open(self.configuration.protocol_filename, 'a+')
                    self.output_redirected = True
                except IOError:
                    pass

        # Create logfile if requested to store the log with the stacked file.
        if self.attached_log_file:
            self.attached_log_file.close()
        if self.configuration.global_parameters_store_protocol_with_result:
            self.attached_log_file = open(self.attached_log_name, "w+")
        else:
            self.attached_log_file = None

        # Write a header to stdout and optionally to the logfile.
        if self.configuration.global_parameters_protocol_level > 0:
            decorator_line = (len(input_name)+28)*"*"
            Miscellaneous.protocol(decorator_line, self.attached_log_file,
                                   precede_with_timestamp=False)
            Miscellaneous.protocol("Start processing " + input_name, self.attached_log_file)
            Miscellaneous.protocol(decorator_line, self.attached_log_file,
                                   precede_with_timestamp=False)


        # Initalize the timer object used to measure execution times of program sections.
        self.my_timer = timer()
        self.my_timer.create('Execution over all')

        if self.job_type == 'stacking':
            # Decide on the objects to be buffered, depending on configuration parameter.
            buffer_original = False
            buffer_monochrome = False
            buffer_gaussian = False
            buffer_laplacian = False

            if self.configuration.global_parameters_buffering_level > 0:
                buffer_laplacian = True
            if self.configuration.global_parameters_buffering_level > 1:
                buffer_gaussian = True
            if self.configuration.global_parameters_buffering_level > 2:
                buffer_original = True
            if self.configuration.global_parameters_buffering_level > 3:
                buffer_monochrome = True

            if self.configuration.global_parameters_protocol_level > 1:
                Miscellaneous.protocol("+++ Buffering level is " +
                                       str(self.configuration.global_parameters_buffering_level) + " +++",
                                       self.attached_log_file)
            if buffer_original:
                if self.configuration.global_parameters_protocol_level > 0:
                    Miscellaneous.protocol("+++ Start reading frames +++", self.attached_log_file)
                self.my_timer.create('Read all frames')
            try:
                self.frames = Frames(self.configuration, names, type=input_type,
                                convert_to_grayscale=convert_to_grayscale,
                                progress_signal=self.work_current_progress_signal,
                                buffer_original=buffer_original, buffer_monochrome=buffer_monochrome,
                                buffer_gaussian=buffer_gaussian, buffer_laplacian=buffer_laplacian)
                if buffer_original and self.configuration.global_parameters_protocol_level > 1:
                    Miscellaneous.protocol(
                                "           Number of images read: " + str(self.frames.number) +
                                ", image shape: " + str(self.frames.shape), self.attached_log_file,
                                precede_with_timestamp=False)
                elif self.configuration.global_parameters_protocol_level > 1:
                    Miscellaneous.protocol(
                        "           Total number of frames: " + str(self.frames.number),
                        self.attached_log_file, precede_with_timestamp=False)
            except Exception as e:
                if self.configuration.global_parameters_protocol_level > 0:
                    Miscellaneous.protocol("Error: " + e.message + ", continue with next job\n",
                                           self.attached_log_file)
                if buffer_original:
                    self.my_timer.stop('Read all frames')
                self.work_next_task_signal.emit("Next job")
                return
            if buffer_original:
                self.my_timer.stop('Read all frames')

            self.work_next_task_signal.emit("Rank frames")

        # Job type is 'postproc'.
        else:
            input_image = imread(self.postproc_input_name, -1)

            # If color image, convert to RGB mode.
            if len(input_image.shape) == 3:
                self.postproc_input_image = cvtColor(input_image, COLOR_BGR2RGB)
            else:
                self.postproc_input_image = input_image

            # Convert 8 bit to 16 bit.
            if self.postproc_input_image.dtype == uint8:
                self.postproc_input_image = self.postproc_input_image.astype(uint16)*256
            self.work_next_task_signal.emit("Postprocessing")

    @QtCore.pyqtSlot()
    def execute_rank_frames(self):

        self.set_status_bar_processing_phase("ranking frames")
        # Rank the frames by their overall local contrast.
        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ Start ranking images +++", self.attached_log_file)
        self.my_timer.create_no_check('Ranking images')
        self.rank_frames = RankFrames(self.frames, self.configuration,
                                      self.work_current_progress_signal)
        self.rank_frames.frame_score()
        self.my_timer.stop('Ranking images')
        if self.configuration.global_parameters_protocol_level > 1:
            Miscellaneous.protocol(
                "           Index of best frame: " + str(self.rank_frames.frame_ranks_max_index),
                self.attached_log_file, precede_with_timestamp=False)

        self.work_next_task_signal.emit("Align frames")

    @QtCore.pyqtSlot(int, int, int, int)
    def execute_align_frames(self, y_low_opt, y_high_opt, x_low_opt, x_high_opt):

        self.set_status_bar_processing_phase("aligning frames")
        # Initialize the frame alignment object.
        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ Initializing frame alignment +++",
                                   self.attached_log_file)
        self.align_frames = AlignFrames(self.frames, self.rank_frames, self.configuration,
                                        progress_signal=self.work_current_progress_signal)

        if self.configuration.align_frames_mode == "Surface":

            auto_execution = False
            if y_low_opt == 0 and y_high_opt == 0 and x_low_opt==0 and x_high_opt == 0:
                auto_execution = True
            elif (y_high_opt - y_low_opt) / self.frames.shape[0] > \
                self.configuration.align_frames_max_stabilization_patch_fraction or \
                (x_high_opt - x_low_opt) / self.frames.shape[1] > \
                self.configuration.align_frames_max_stabilization_patch_fraction and \
                self.configuration.global_parameters_protocol_level > 0:
                Miscellaneous.protocol("           Stabilization patch selected manually is "
                                       "too large, switch to automatic mode",
                                       self.attached_log_file, precede_with_timestamp=False)
                auto_execution = True
            elif (y_high_opt - y_low_opt) / self.frames.shape[0] < \
                self.configuration.align_frames_min_stabilization_patch_fraction or \
                (x_high_opt - x_low_opt) / self.frames.shape[1] < \
                self.configuration.align_frames_min_stabilization_patch_fraction and \
                self.configuration.global_parameters_protocol_level > 0:
                Miscellaneous.protocol("           Stabilization patch selected manually is "
                                       "too small, switch to automatic mode",
                                       self.attached_log_file, precede_with_timestamp=False)
                auto_execution = True


            # Compute the local rectangular patch in the image where the L gradient is highest
            # in both x and y direction. The scale factor specifies how much smaller the patch
            # is compared to the whole image frame. In batch mode, variable "auto_execution" is
            # set to "True", and the automatic patch computation is the only option.
            if auto_execution or self.configuration.align_frames_automation:

                self.my_timer.create_no_check('Select optimal alignment patch')
                (y_low_opt, y_high_opt, x_low_opt, x_high_opt) = \
                    self.align_frames.compute_alignment_rect(
                        self.configuration.align_frames_rectangle_scale_factor)
                self.my_timer.stop('Select optimal alignment patch')
                if self.configuration.global_parameters_protocol_level > 1:
                    Miscellaneous.protocol(
                                       "           Alignment rectangle, computed automatically: " +
                                       str(y_low_opt) + "<y<" + str(y_high_opt) +
                                       ", " + str(x_low_opt) + "<x<" +
                                       str(x_high_opt), self.attached_log_file,
                                       precede_with_timestamp=False)

            # As an alternative, set the coordinates of the rectangular patch explicitly.
            else:
                # The image displayed in the stabilization patch editor was shrunk on all four
                # sides by a number of pixels given by the alignment search width parameter.
                # Therefore, the resulting coordinates of the stabilization patch have to be
                # corrected by this offset now.
                y_low_opt += self.configuration.align_frames_search_width
                y_high_opt += self.configuration.align_frames_search_width
                x_low_opt += self.configuration.align_frames_search_width
                x_high_opt += self.configuration.align_frames_search_width

                self.align_frames.set_alignment_rect(y_low_opt, y_high_opt, x_low_opt, x_high_opt)
                if self.configuration.global_parameters_protocol_level > 1:
                    Miscellaneous.protocol("           Alignment rectangle, set by the user: " +
                                           str(y_low_opt) + "<y<" + str(y_high_opt) +
                                       ", " + str(x_low_opt) + "<x<" +
                                           str(x_high_opt), self.attached_log_file,
                                           precede_with_timestamp=False)

        # Align all frames globally relative to the frame with the highest score.
        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ Start aligning all frames +++", self.attached_log_file)

        self.my_timer.create_no_check('Global frame alignment')

        # Align all frames in "Surface" mode.
        if self.configuration.align_frames_mode == "Surface":
            # Try the frame alignment using the alignment patch with the highest quality first. If
            # for at least one frame no valid shift can be found, try the next alignment patch. If
            # valid shifts cannot be computed with any patch, abort processing of this job and go to
            # the next one.
            number_patches = len(self.align_frames.alignment_rect_qualities)
            for patch_index in range(number_patches):
                self.align_frames.select_alignment_rect(patch_index)
                try:
                    self.align_frames.align_frames()
                    # Everything is fine, no need to try another stabilization patch.
                    break
                except (NotSupportedError, ArgumentError) as e:
                    if self.configuration.global_parameters_protocol_level > 0:
                        Miscellaneous.protocol("Error: " + e.message + ", continue with next job\n",
                                               self.attached_log_file)
                    self.my_timer.stop('Global frame alignment')
                    self.work_next_task_signal.emit("Next job")
                    return
                # For some frames no valid shift could be computed. This would create problems later
                # in the workflow. Therefore, try again with another stabilization patch.
                except InternalError as e:
                    if self.configuration.global_parameters_protocol_level > 0:
                        Miscellaneous.protocol("Warning: " + e.message + ", will try another"
                                                " stabilization patch", self.attached_log_file)
                    # If there is no more patch available, skip this job.
                    if patch_index == number_patches - 1:
                        if self.configuration.global_parameters_protocol_level > 0:
                            Miscellaneous.protocol("Error: No alternative stabilization patch"
                                                   " available, continue with next job\n",
                                                   self.attached_log_file)
                            self.my_timer.stop('Global frame alignment')
                            self.work_next_task_signal.emit("Next job")
                            return
                    # Continue with the next best stabilization patch.
                    else:
                        if self.configuration.global_parameters_protocol_level > 0:
                            y_low_opt, y_high_opt, x_low_opt, x_high_opt = \
                                self.align_frames.alignment_rect_bounds[patch_index+1]
                            Miscellaneous.protocol("           Next alignment rectangle tried: " +
                                                   str(y_low_opt) + "<y<" + str(y_high_opt) +
                                       ", " + str(x_low_opt) + "<x<" +
                                                   str(x_high_opt), self.attached_log_file,
                                                   precede_with_timestamp=False)

        # Align all frames in "Planet" mode.
        else:
            try:
                self.align_frames.align_frames()
            except NotSupportedError as e:
                if self.configuration.global_parameters_protocol_level > 0:
                    Miscellaneous.protocol("Error: " + e.message + ", continue with next job\n",
                                           self.attached_log_file)
                self.my_timer.stop('Global frame alignment')
                self.work_next_task_signal.emit("Next job")
                return

        self.my_timer.stop('Global frame alignment')

        if self.configuration.global_parameters_protocol_level > 1:
            Miscellaneous.protocol("           Pixel range common to all frames: " + str(
                self.align_frames.intersection_shape[0][0]) + "<y<" + str(
                self.align_frames.intersection_shape[0][1]) + ", " + str(
                self.align_frames.intersection_shape[1][0]) + "<x<" + str(
                self.align_frames.intersection_shape[1][1]), self.attached_log_file,
                                   precede_with_timestamp=False)

        # Compute the average frame.
        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ Start computing the average frame +++",
                                   self.attached_log_file)
        self.my_timer.create_no_check('Compute reference frame')
        self.align_frames.average_frame()
        self.my_timer.stop('Compute reference frame')
        if self.configuration.global_parameters_protocol_level > 1:
            Miscellaneous.protocol(
                "           The average frame was computed using the best " + str(
                self.align_frames.average_frame_number) + " frames.", self.attached_log_file,
                precede_with_timestamp=False)

        self.work_next_task_signal.emit("Select stack size")

    @QtCore.pyqtSlot(int, int, int, int)
    def execute_set_roi(self, y_min, y_max, x_min, x_max):

        self.set_status_bar_processing_phase("setting the ROI")
        if self.configuration.global_parameters_protocol_level > 0 and y_min==0 and y_max==0:
            Miscellaneous.protocol("+++ Start setting a ROI and computing a new average frame +++",
                                   self.attached_log_file)
        self.my_timer.create_no_check('Setting ROI and new reference')
        self.align_frames.set_roi(y_min, y_max, x_min, x_max)
        self.my_timer.stop('Setting ROI and new reference')

        if self.configuration.global_parameters_protocol_level > 1 and y_min!=0 or y_max!=0:
            Miscellaneous.protocol("           ROI, set by the user: " +
                                   str(y_min) + "<y<" + str(y_max) +
                                   ", " + str(x_min) + "<x<" +
                                   str(x_max), self.attached_log_file,
                                   precede_with_timestamp=False)

        self.work_next_task_signal.emit("Set alignment points")

    @QtCore.pyqtSlot()
    def execute_set_alignment_points(self):

        # If not executing in "automatic" mode, the APs are created on the main_gui thread.
        if self.main_gui.automatic:
            self.set_status_bar_processing_phase("creating alignment points")
            # Initialize the AlignmentPoints object.
            self.my_timer.create_no_check('Initialize alignment point object')
            self.alignment_points = AlignmentPoints(self.configuration, self.frames, self.rank_frames,
                self.align_frames, progress_signal=self.work_current_progress_signal)
            self.my_timer.stop('Initialize alignment point object')

            # Create alignment points, and create an image with wll alignment point boxes and patches.
            if self.configuration.global_parameters_protocol_level > 0:
                Miscellaneous.protocol("+++ Start creating alignment points +++",
                                       self.attached_log_file)
            self.my_timer.create_no_check('Create alignment points')

            # If a ROI is selected, alignment points are created in the ROI window only.
            self.alignment_points.create_ap_grid()

            self.my_timer.stop('Create alignment points')

        self.work_next_task_signal.emit("Compute frame qualities")

    @QtCore.pyqtSlot()
    def execute_compute_frame_qualities(self):

        if self.configuration.global_parameters_protocol_level > 1:
            Miscellaneous.protocol("           Number of alignment points selected: " + str(
                len(self.alignment_points.alignment_points)) +
                  ", aps dropped because too dim: " + str(
                self.alignment_points.alignment_points_dropped_dim) +
                  ", aps dropped because too little structure: " + str(
                self.alignment_points.alignment_points_dropped_structure),
                                   self.attached_log_file, precede_with_timestamp=False)

        self.set_status_bar_processing_phase("ranking all frames at all alignment points")
        # For each alignment point rank frames by their quality.
        self.my_timer.create_no_check('Rank frames at alignment points')
        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ Start ranking all frames at all alignment points +++",
                                   self.attached_log_file)
        self.alignment_points.compute_frame_qualities()
        self.my_timer.stop('Rank frames at alignment points')

        self.work_next_task_signal.emit("Stack frames")

    @QtCore.pyqtSlot()
    def execute_stack_frames(self):

        self.set_status_bar_processing_phase("stacking frames")
        # Allocate StackFrames object.
        self.stack_frames = StackFrames(self.configuration, self.frames, self.align_frames,
                                   self.alignment_points, self.my_timer,
                                   progress_signal=self.work_current_progress_signal)

        # Stack all frames.
        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ Start stacking " + str(self.alignment_points.stack_size) +
                                   " frames +++", self.attached_log_file)
        self.stack_frames.stack_frames()

        if self.configuration.global_parameters_protocol_level > 1:
            Miscellaneous.protocol("\n           Distribution of shifts at alignment points:",
                                   self.attached_log_file, precede_with_timestamp=False)
            Miscellaneous.protocol(self.stack_frames.print_shift_table() + "\n",
                                   self.attached_log_file, precede_with_timestamp=False)

        self.set_status_bar_processing_phase("merging AP patches")
        # Merge the stacked alignment point buffers into a single image.
        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ Start merging all alignment patches and the background +++",
                                   self.attached_log_file)
        self.stack_frames.merge_alignment_point_buffers()

        self.work_next_task_signal.emit("Save stacked image")

    @QtCore.pyqtSlot()
    def execute_save_stacked_image(self):

        self.set_status_bar_processing_phase("saving result")
        # Save the image as 16bit int (color or mono).
        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ Start saving the stacked image +++",
                                   self.attached_log_file)
        self.my_timer.create_no_check('Saving the stacked image')
        self.frames.save_image(self.stacked_image_name, self.stack_frames.stacked_image,
                               color=self.frames.color, avoid_overwriting=False)
        self.my_timer.stop('Saving the stacked image')
        if self.configuration.global_parameters_protocol_level > 1:
            Miscellaneous.protocol("           The stacked image was written to: " +
                                   self.stacked_image_name,
                                   self.attached_log_file, precede_with_timestamp=False)

        # If postprocessing is included after stacking, set the stacked image as input.
        if self.configuration.global_parameters_include_postprocessing:
            self.postproc_input_image = self.stack_frames.stacked_image
            self.postproc_input_name = self.stacked_image_name
            self.postprocessed_image_name = PostprocDataObject.set_file_name_processed(
                self.stacked_image_name, self.configuration.postproc_suffix)
            self.work_next_task_signal.emit("Postprocessing")
        else:
            self.work_next_task_signal.emit("Next job")

            # Print timing info for this job.
            self.my_timer.stop('Execution over all')
            if self.configuration.global_parameters_protocol_level > 0:
                self.my_timer.protocol(self.attached_log_file)

    @QtCore.pyqtSlot()
    def execute_postprocess_image(self):

        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ Start postprocessing +++",
                                   self.attached_log_file)
        self.my_timer.create_no_check('Conputing image postprocessing')

        # Initialize the new image with the original image.
        self.postprocessed_image = self.postproc_input_image

        # Apply all sharpening layers of the postprocessing version selected last time.
        version_index = self.configuration.postproc_data_object.version_selected
        postproc_layers = self.configuration.postproc_data_object.versions[version_index].layers
        for layer in postproc_layers:
            self.postprocessed_image = Miscellaneous.gaussian_sharpen(self.postprocessed_image,
                                                        layer.amount, layer.radius,
                                                        luminance_only=layer.luminance_only)
        self.my_timer.stop('Conputing image postprocessing')

        self.work_next_task_signal.emit("Save postprocessed image")

    @QtCore.pyqtSlot(object)
    def execute_save_postprocessed_image(self, postprocessed_image):

        # The signal payload is None only if the editor was left with "cancel" in interactive mode.
        # In this case, skip saving the result and proceed with the next job.
        if postprocessed_image is not None:
            self.set_status_bar_processing_phase("saving result")
            # Save the image as 16bit int (color or mono).
            if self.configuration.global_parameters_protocol_level > 0:
                Miscellaneous.protocol("+++ Start saving the postprocessed image +++",
                                       self.attached_log_file)
            self.my_timer.create_no_check('Saving the postprocessed image')
            Frames.save_image(self.postprocessed_image_name, postprocessed_image,
                                   color=(len(postprocessed_image.shape)==3),
                                   avoid_overwriting=False)
            self.my_timer.stop('Saving the postprocessed image')
            if self.configuration.global_parameters_protocol_level > 1:
                Miscellaneous.protocol(
                    "           The postprocessed image was written to: " +
                    self.postprocessed_image_name,
                    self.attached_log_file, precede_with_timestamp=False)

            if self.configuration.global_parameters_protocol_level > 1:
                Miscellaneous.print_postproc_parameters(
                        self.configuration.postproc_data_object.versions[
                        self.configuration.postproc_data_object.version_selected].layers,
                        self.attached_log_file)

        self.work_next_task_signal.emit("Next job")

        # Print timing info for this job.
        self.my_timer.stop('Execution over all')
        if self.configuration.global_parameters_protocol_level > 0:
            self.my_timer.protocol(self.attached_log_file)

    def set_status_bar_processing_phase(self, phase):
        """
        Put a text of the form
            "Processing < job name >, < processing step >."
        on the main window status bar.

        :param phase: Processing phase (string)
        :return: -
        """

        self.set_status_bar_signal.emit("Processing " + self.input_name + ", " + phase + ".",
                                        "black")
