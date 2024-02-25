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

import gc
import platform
import sys
from os import listdir, rename, remove
from os.path import splitext, join, dirname

import psutil
from PyQt6 import QtCore
from numpy import uint16, uint8

from align_frames import AlignFrames
from alignment_points import AlignmentPoints
from configuration import PostprocDataObject
from exceptions import NotSupportedError, InternalError, ArgumentError, Error
from frames import Frames, Calibration
from miscellaneous import Miscellaneous
from rank_frames import RankFrames
from stack_frames import StackFrames
from timer import timer

# The following lists define the allowed file extensions for still images.
image_extensions = ['.tif', '.tiff', '.fit', '.fits', '.jpg', '.png']

class Workflow(QtCore.QObject):
    master_dark_created_signal = QtCore.pyqtSignal(bool)
    master_flat_created_signal = QtCore.pyqtSignal(bool)
    work_next_task_signal = QtCore.pyqtSignal(str)
    report_error_signal = QtCore.pyqtSignal(str)
    abort_job_signal = QtCore.pyqtSignal(str)
    work_current_progress_signal = QtCore.pyqtSignal(str, int)
    set_main_gui_busy_signal = QtCore.pyqtSignal(bool)
    set_status_bar_signal = QtCore.pyqtSignal(str, str)
    create_image_window_signal = QtCore.pyqtSignal()
    update_image_window_signal = QtCore.pyqtSignal(object)
    terminate_image_window_signal = QtCore.pyqtSignal()

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
        self.activity = None
        self.attached_log_name = None
        self.attached_log_name_new = None
        self.attached_log_file = None
        self.stdout_saved = None
        self.output_redirected = False
        self.protocol_file = None

        # Switch alignment point debugging on / off.
        self.debug_AP = False

        # Print info on the platform used.
        platform_name = platform.system()
        processor_name = platform.processor()

        python_dir = dirname(sys.executable)
        if self.configuration.global_parameters_protocol_level > 1:
            Miscellaneous.protocol(
                "Operating system used: " + platform_name + ", Processor used: " + processor_name +
                "\n           Python interpreter location: " +
                python_dir, self.attached_log_file, precede_with_timestamp=True)

        # Check if the configuration was imported from an older version. If so, print a message.
        # Please note that the "version imported from" parameter is set only if a configuration
        # has been read.
        if self.configuration.global_parameters_protocol_level > 1 and \
                self.configuration.configuration_read and \
                self.configuration.global_parameters_version_imported_from != \
                self.configuration.global_parameters_version:
            Miscellaneous.protocol("           Configuration imported from older version: " +
                                   self.configuration.global_parameters_version_imported_from,
                                   self.attached_log_file, precede_with_timestamp=False)

        # Create the calibration object, used for potential flat / dark corrections.
        self.calibration = Calibration(self.configuration)

    @QtCore.pyqtSlot(list)
    def execute_create_master_dark(self, dark_names):
        # Create a new master dark.
        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ Creating a new master dark frame +++",
                                   self.attached_log_file, precede_with_timestamp=True)
        if self.configuration.global_parameters_protocol_level > 1:
            Miscellaneous.protocol("           Input frames: " + dark_names[0],
                                   self.attached_log_file, precede_with_timestamp=False)

        try:
            self.set_main_gui_busy_signal.emit(True)
            self.calibration.create_master_dark(dark_names[0])
            if self.calibration.warn_message is not None and \
                    self.configuration.global_parameters_protocol_level > 0:
                Miscellaneous.protocol("Warning in creating a new master dark frame: " +
                                       self.calibration.warn_message,
                                       self.attached_log_file, precede_with_timestamp=True)
            self.master_dark_created_signal.emit(True)
        except Exception as e:
            if self.configuration.global_parameters_protocol_level > 0:
                self.report_error_signal.emit("Error in creating master dark frame: " + str(e))
            self.master_dark_created_signal.emit(False)

    @QtCore.pyqtSlot(list)
    def execute_create_master_flat(self, flat_names):

        # Create a new master flat.
        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ Creating a new master flat frame +++",
                                   self.attached_log_file, precede_with_timestamp=True)
        if self.configuration.global_parameters_protocol_level > 1:
            Miscellaneous.protocol("           Input frames: " + flat_names[0],
                                   self.attached_log_file, precede_with_timestamp=False)

        try:
            self.set_main_gui_busy_signal.emit(True)
            self.calibration.create_master_flat(flat_names[0])
            if self.calibration.warn_message is not None and \
                    self.configuration.global_parameters_protocol_level > 0:
                Miscellaneous.protocol("Warning in creating a new master flat frame: " +
                                       self.calibration.warn_message,
                                       self.attached_log_file, precede_with_timestamp=True)
            self.master_flat_created_signal.emit(True)
        except Error as e:
            if self.configuration.global_parameters_protocol_level > 0:
                self.report_error_signal.emit("Error in creating master flat frame: " +
                                              str(e) + ", flat frame calibration de-activated")
            self.master_flat_created_signal.emit(False)

    @QtCore.pyqtSlot()
    def execute_reset_masters(self):

        # De-activate master frames.
        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ De-activating master frames +++", self.attached_log_file,
                                   precede_with_timestamp=True)
        self.calibration.reset_masters()

    @QtCore.pyqtSlot(object)
    def execute_frames(self, job):
        self.job = job

        # Reset the potential new log file name (required if parameters are to be encoded in file
        # name).
        self.attached_log_name_new = None

        # Remove objects from previous job to clean up RAM.
        for obj in [self.frames, self.rank_frames, self.align_frames, self.alignment_points,
                    self.stack_frames]:
            if obj is not None:
                del obj
        self.frames = None
        self.rank_frames = None
        self.align_frames = None
        self.alignment_points = None
        self.stack_frames = None

        # Force the garbage collector to release unreferenced objects.
        gc.collect()

        # Update the status bar in the main GUI.
        self.input_name = self.job.file_name
        self.set_status_bar_processing_phase("reading frames")

        # A job can either "stack" images or "postprocess" a single image. In the latter case,
        # input is a single image file.
        #
        # Images for stacking can either be extracted from a video file or a batch of single
        # photographs. In the first case, input_type is set to 'video', in the second case to
        # 'image'.

        if self.job.type == 'postproc':
            self.activity = 'postproc'
            self.postproc_input_name = self.job.name

            # Reset the postprocessed image to None. This way, in saving the postprocessing result,
            # it can be checked if an image was computed in the workflow thread.
            self.postprocessed_image = None
            self.postprocessed_image_name = PostprocDataObject.set_file_name_processed(
                self.job.name, self.configuration.postproc_suffix,
                self.configuration.global_parameters_image_format)
            self.attached_log_name = splitext(self.job.name)[0] + '_postproc-log.txt'

        # For video file input, the Frames constructor expects the video file name for "names".
        elif self.job.type == 'video':
            self.activity = 'stacking'
            names = self.job.name
            self.attached_log_name = splitext(self.job.name)[0] + '_stacking-log.txt'

        # For single image input, the Frames constructor expects a list of image file names for
        # "names".
        else:  # input_type = 'image'
            self.activity = 'stacking'
            # Include only names with image extensions.
            names = [join(self.job.name, name) for name in listdir(self.job.name) if
                     splitext(name)[-1].lower() in image_extensions]
            self.attached_log_name = self.job.name + '_stacking-log.txt'

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
            decorator_line = (len(self.job.name) + 28) * "*"
            Miscellaneous.protocol(decorator_line, self.attached_log_file,
                                   precede_with_timestamp=False)
            Miscellaneous.protocol("Start processing " + self.job.name, self.attached_log_file)
            Miscellaneous.protocol("           Software version used: " +
                                   self.configuration.global_parameters_version,
                                   self.attached_log_file, precede_with_timestamp=False)
            Miscellaneous.protocol(decorator_line, self.attached_log_file,
                                   precede_with_timestamp=False)

        # Initalize the timer object used to measure execution times of program sections.
        self.my_timer = timer()
        self.my_timer.create('Execution over all')

        if self.activity == 'stacking':

            # Write parameters to the protocol.
            if self.configuration.global_parameters_protocol_level > 1:
                Miscellaneous.print_stacking_parameters(self.configuration, self.attached_log_file)

            try:
                if self.configuration.global_parameters_maximum_memory_active:
                    available_ram = float(self.configuration.global_parameters_maximum_memory_amount)
                else:
                    # Look up the available RAM (without paging)
                    virtual_memory = dict(psutil.virtual_memory()._asdict())
                    available_ram = virtual_memory['available'] / 1e9

                self.frames = Frames(self.configuration, names, type=self.job.type,
                                     bayer_option_selected=self.job.bayer_option_selected,
                                     calibration=self.calibration,
                                     progress_signal=self.work_current_progress_signal)

                # If buffering is not automatic, set the buffering_level as requested by the user.
                if self.configuration.global_parameters_buffering_level != -1:
                    # Decide on the objects to be buffered, depending on configuration parameter.
                    buffering_level_set = self.configuration.global_parameters_buffering_level
                    # Compute the approximate RAM usage of this job at the selected buffering level.
                    needed_ram = self.frames.compute_required_buffer_size(buffering_level_set)
                else:
                    needed_ram = None

                # If buffering was set to "auto", compute the highest possible value. If the
                # buffering level requested explicitly was too high, test if lowering the
                # buffering level would help.
                if self.configuration.global_parameters_buffering_level == -1 or needed_ram > available_ram:
                    buffering_level_set = None
                    for level in range(4, -1, -1):
                        alternative_ram = self.frames.compute_required_buffer_size(level)
                        if alternative_ram < available_ram:
                            buffering_level_set = level
                            needed_ram = alternative_ram
                            break

                    # Check if the job can be processed with the requested buffering level.
                    message = None
                    if buffering_level_set is None:
                        message = "Error: Too little RAM for this job, continuing with the next one"

                    # If an appropriate level other then the one set by the user was found, write it
                    # as a recommendation to the protocol. The job is aborted anyway.
                    elif self.configuration.global_parameters_buffering_level != -1:
                        message = "Error: Too little RAM for chosen buffering level, recommended " \
                                  "level: " + str(
                            buffering_level_set) + ", continuing with next job"

                    # Buffering is set to "auto" and an appropriate buffering level was found. Set
                    # the buffering in the frames object.
                    else:
                        self.frames.set_buffering(buffering_level_set)

                    # The job does not fit in RAM, continue with the next job.
                    if message:
                        self.abort_job_signal.emit(message)
                        return

                # The user has set the buffering level explicitly, and the job fits in memory. Set
                # the buffering in the frames object.
                else:
                    self.frames.set_buffering(buffering_level_set)

                if self.configuration.global_parameters_protocol_level > 1:
                    Miscellaneous.protocol("+++ Buffering level is " +
                                           str(buffering_level_set) +
                                           " +++", self.attached_log_file)
                    Miscellaneous.protocol(
                        "           RAM required (Gbytes): " + str(round(needed_ram, 2)) +
                        ", available: " + str(round(available_ram, 2)), self.attached_log_file,
                        precede_with_timestamp=False)

                if self.configuration.global_parameters_protocol_level > 0:
                    Miscellaneous.protocol("+++ Start reading frames +++", self.attached_log_file)

                if self.frames.warn_message is not None and \
                    self.configuration.global_parameters_protocol_level > 0:
                    Miscellaneous.protocol(
                        "Warning in opening SER file: " + self.frames.warn_message,
                        self.attached_log_file, precede_with_timestamp=True)
                if self.configuration.global_parameters_protocol_level > 1:
                    Miscellaneous.protocol(
                        "           Number of frames: " + str(self.frames.number) +
                        ", image shape: " + str(self.frames.shape), self.attached_log_file,
                        precede_with_timestamp=False)
                    if self.job.bayer_option_selected == 'Auto detect color':
                        Miscellaneous.protocol(
                            "           Debayer pattern detected automatically: '" +
                            self.frames.bayer_pattern + "'",
                            self.attached_log_file, precede_with_timestamp=False)
                        self.job.bayer_pattern = self.frames.bayer_pattern
                    else:
                        Miscellaneous.protocol(
                            "           Debayer pattern selected manually: '" +
                            self.job.bayer_option_selected + "'",
                            self.attached_log_file, precede_with_timestamp=False)
                    if self.frames.dt0 == 'uint16':
                        dynamic_range = '16 bit'
                    elif self.frames.dt0 == 'uint8':
                        dynamic_range = '8 bit'
                    else:
                        dynamic_range = 'undefined'
                    Miscellaneous.protocol(
                        "           Dynamic range of input frames: " + dynamic_range + ".",
                        self.attached_log_file, precede_with_timestamp=False)
                    if self.frames.shift_pixels:
                        Miscellaneous.protocol(
                            "           Pixel values are multiplied by 2**" + str(
                                self.frames.shift_pixels) + " to use the full dynamic range of 16 "
                                                            "bits.",
                            self.attached_log_file, precede_with_timestamp=False)
                    if self.frames.calibration_matches:
                        if self.calibration.master_dark_frame_adapted is not None and \
                                self.calibration.inverse_master_flat_frame is not None:
                            Miscellaneous.protocol(
                                "           Dark / flat frame calibration is active",
                                self.attached_log_file, precede_with_timestamp=False)
                        elif self.calibration.master_dark_frame_adapted is not None:
                            Miscellaneous.protocol("           Dark frame calibration is active",
                                                   self.attached_log_file,
                                                   precede_with_timestamp=False)
                        elif self.calibration.inverse_master_flat_frame is not None:
                            Miscellaneous.protocol("           Flat frame calibration is active",
                                                   self.attached_log_file,
                                                   precede_with_timestamp=False)
                    else:
                        Miscellaneous.protocol(
                            "           No matching master dark / flat frames found, "
                            "calibration de-activated",
                            self.attached_log_file, precede_with_timestamp=False)
            except Error as e:
                self.abort_job_signal.emit("Error: " + e.message + ", continuing with next job")
                return
            except Exception as e:
                self.abort_job_signal.emit(
                    "Error in opening/reading frames: " + str(e) + ", continuing with next job")
                return



            # The RAM seems to be sufficient, continue with ranking frames.
            self.work_next_task_signal.emit("Rank frames")

        # Job type is 'postproc'.
        else:
            try:
                self.postproc_input_image = Frames.read_image(self.postproc_input_name)
            except Error as e:
                self.abort_job_signal.emit("Error: " + e.message + ", continuing with next job")
                return
            except Exception as e:
                self.abort_job_signal.emit(
                    "Error in reading image file: " + str(e) + ", continuing with next job")
                return

            # Convert 8 bit to 16 bit.
            if self.postproc_input_image.dtype == uint8:
                self.postproc_input_image = self.postproc_input_image.astype(uint16) * 256
            self.work_next_task_signal.emit("Postprocessing")

    @QtCore.pyqtSlot()
    def execute_rank_frames(self):

        # Reset the frame index translation.
        self.frames.reset_index_translation()

        self.set_status_bar_processing_phase("ranking frames")
        # Rank the frames by their overall local contrast.
        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ Start ranking frames +++", self.attached_log_file)
        self.my_timer.create_no_check('Ranking frames')

        try:
            self.rank_frames = RankFrames(self.frames, self.configuration,
                                          self.work_current_progress_signal)
            self.rank_frames.frame_score()
            self.my_timer.stop('Ranking frames')
        except Error as e:
            self.abort_job_signal.emit("Error: " + e.message + ", continuing with next job")
            self.my_timer.stop('Ranking frames')
            return
        except Exception as e:
            self.abort_job_signal.emit(
                "Error: " + str(e) + ", continuing with next job")
            self.my_timer.stop('Ranking frames')
            return

        if self.configuration.global_parameters_protocol_level > 1:
            Miscellaneous.protocol(
                "           Index of best frame: " + str(self.rank_frames.frame_ranks_max_index + 1),
                self.attached_log_file, precede_with_timestamp=False)

        self.work_next_task_signal.emit("Select frames")

    @QtCore.pyqtSlot()
    def execute_set_index_translation_table(self):
        # If in the frame selection dialog the status of at least one frame was changed, update
        # the index translation table.
        if not all(self.frames.index_included):
            self.frames.set_index_translation()
            self.rank_frames.set_index_translation(self.frames.index_translation)
        else:
            self.frames.reset_index_translation()
            self.rank_frames.reset_index_translation()

        self.work_next_task_signal.emit("Align frames")

    @QtCore.pyqtSlot(int, int, int, int)
    def execute_align_frames(self, y_low_opt, y_high_opt, x_low_opt, x_high_opt):

        self.set_status_bar_processing_phase("aligning frames")
        # Initialize the frame alignment object.
        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ Initializing frame alignment +++", self.attached_log_file)
        self.align_frames = AlignFrames(self.frames, self.rank_frames, self.configuration,
                                        progress_signal=self.work_current_progress_signal)

        if self.configuration.align_frames_mode == "Surface":

            auto_execution = False
            if y_low_opt == 0 and y_high_opt == 0 and x_low_opt == 0 and x_high_opt == 0:
                auto_execution = True
            elif (y_high_opt - y_low_opt) / self.frames.shape[
                0] > self.configuration.align_frames_max_stabilization_patch_fraction or (
                    x_high_opt - x_low_opt) / self.frames.shape[
                1] > self.configuration.align_frames_max_stabilization_patch_fraction and \
                    self.configuration.global_parameters_protocol_level > 0:
                Miscellaneous.protocol("           Stabilization patch selected manually is "
                                       "too large, switch to automatic mode",
                                       self.attached_log_file, precede_with_timestamp=False)
                auto_execution = True
            elif (y_high_opt - y_low_opt) / self.frames.shape[
                0] < self.configuration.align_frames_min_stabilization_patch_fraction or (
                    x_high_opt - x_low_opt) / self.frames.shape[
                1] < self.configuration.align_frames_min_stabilization_patch_fraction and \
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
                try:
                    (y_low_opt, y_high_opt, x_low_opt,
                     x_high_opt) = self.align_frames.compute_alignment_rect(
                        self.configuration.align_frames_rectangle_scale_factor)
                except (ArgumentError) as e:
                    self.abort_job_signal.emit("Error: " + e.message + ", continuing with next job")
                    self.my_timer.stop('Select optimal alignment patch')
                    return
                self.my_timer.stop('Select optimal alignment patch')
                if self.configuration.global_parameters_protocol_level > 1:
                    Miscellaneous.protocol(
                        "           Alignment rectangle, computed automatically: " + str(
                            y_low_opt) + "<y<" + str(y_high_opt) + ", " + str(
                            x_low_opt) + "<x<" + str(x_high_opt), self.attached_log_file,
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
                    Miscellaneous.protocol(
                        "           Alignment rectangle, set by the user: " + str(
                            y_low_opt) + "<y<" + str(y_high_opt) + ", " + str(
                            x_low_opt) + "<x<" + str(x_high_opt), self.attached_log_file,
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
                    self.abort_job_signal.emit("Error: " + e.message + ", continuing with next job")
                    self.my_timer.stop('Global frame alignment')
                    return
                # For some frames no valid shift could be computed. This would create problems later
                # in the workflow. Therefore, try again with another stabilization patch.
                except InternalError as e:
                    if self.configuration.global_parameters_protocol_level > 0:
                        Miscellaneous.protocol("Warning: No valid shift computed at " + e.message +
                                               ", will try another stabilization patch",
                                               self.attached_log_file)
                    # If there is no more patch available, skip this job.
                    if patch_index == number_patches - 1:
                        if self.configuration.align_frames_search_width < self.configuration.align_frames_max_search_width:
                            self.abort_job_signal.emit(
                                "Error: Frame stabilization failed at " + e.message +
                                ", continuing with next job. "
                                "Try a higher value for parameter 'stabilization search width'.")
                        else:
                            self.abort_job_signal.emit(
                                "Error: Frame stabilization failed at " + e.message +
                                ", continuing with next job. "
                                "Try stabilizing the frames with another program, e.g. PIPP.")
                        self.my_timer.stop('Global frame alignment')
                        return
                    # Continue with the next best stabilization patch.
                    else:
                        y_low_opt, y_high_opt, x_low_opt, x_high_opt = \
                            self.align_frames.alignment_rect_bounds[patch_index + 1]
                        if self.configuration.global_parameters_protocol_level > 0:
                            Miscellaneous.protocol(
                                "           Next alignment rectangle tried: " + str(
                                    y_low_opt) + "<y<" + str(y_high_opt) + ", " + str(
                                    x_low_opt) + "<x<" + str(x_high_opt), self.attached_log_file,
                                precede_with_timestamp=False)

                # Catch all other exceptions. In this case abort the job and try the next one.
                except Exception as e:
                    self.abort_job_signal.emit("Error: " + str(e) + ", continuing with next job")
                    self.my_timer.stop('Global frame alignment')
                    return

        # Align all frames in "Planet" mode.
        else:
            try:
                self.align_frames.align_frames()
            except Error as e:
                self.abort_job_signal.emit("Error: " + e.message + ", continuing with next job")
                self.my_timer.stop('Global frame alignment')
                return
            except Exception as e:
                self.abort_job_signal.emit(
                    "Error in aligning frames: " + str(e) + ", continuing with next job")
                self.my_timer.stop('Global frame alignment')
                return

        self.my_timer.stop('Global frame alignment')

        if self.configuration.global_parameters_protocol_level > 1:
            Miscellaneous.protocol("           Pixel range common to all frames: " + str(
                self.align_frames.intersection_shape[0][0]) + "<y<" + str(
                self.align_frames.intersection_shape[0][1]) + ", " + str(
                self.align_frames.intersection_shape[1][0]) + "<x<" + str(
                self.align_frames.intersection_shape[1][1]), self.attached_log_file,
                                   precede_with_timestamp=False)

            # Compute and print the maximum shift between two consecutive frames:
            max_shift_y = max([abs(
                self.align_frames.frame_shifts[idx][0] - self.align_frames.frame_shifts[idx - 1][0])
                               for idx in range(1, self.frames.number)])
            max_shift_x = max([abs(
                self.align_frames.frame_shifts[idx][1] - self.align_frames.frame_shifts[idx - 1][1])
                               for idx in range(1, self.frames.number)])
            Miscellaneous.protocol(
                "           Maximal pixel shift between consecutive frames, vertical: " + str(
                    max_shift_y) + ", horizontal: " + str(max_shift_x), self.attached_log_file,
                                   precede_with_timestamp=False)

        # Compute the average frame.
        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ Start computing the reference frame +++",
                                   self.attached_log_file)
        self.my_timer.create_no_check('Compute reference frame')
        self.align_frames.average_frame()
        self.my_timer.stop('Compute reference frame')
        if self.configuration.global_parameters_protocol_level > 1:
            if self.configuration.align_frames_fast_changing_object:
                Miscellaneous.protocol(
                    "           The reference frame was computed using the best " + str(
                        self.align_frames.average_frame_number) + " frames within a window of " +
                    str(self.align_frames.average_frame_number *
                        self.configuration.align_frames_best_frames_window_extension) +
                    " frames.\n           Quality loss of reference frame due to time restriction: " +
                    str(self.align_frames.quality_loss_percent) +
                    "%\n           Position of reference frame in video time line: " +
                    str(self.align_frames.cog_mean_frame) + "%",
                    self.attached_log_file, precede_with_timestamp=False)
            else:
                Miscellaneous.protocol(
                    "           The reference frame was computed using the best " + str(
                        self.align_frames.average_frame_number) + " frames.",
                    self.attached_log_file, precede_with_timestamp=False)

        self.work_next_task_signal.emit("Select stack size")

    @QtCore.pyqtSlot(int, int, int, int)
    def execute_set_roi(self, y_min, y_max, x_min, x_max):

        self.set_status_bar_processing_phase("setting the ROI")
        if self.configuration.global_parameters_protocol_level > 0 and (y_min != 0 or y_max != 0):
            Miscellaneous.protocol("+++ Start setting a ROI and computing a new reference frame +++",
                                   self.attached_log_file)
        self.my_timer.create_no_check('Setting ROI and new reference')
        self.align_frames.set_roi(y_min, y_max, x_min, x_max)
        self.my_timer.stop('Setting ROI and new reference')

        if self.configuration.global_parameters_protocol_level > 1 and (y_min != 0 or y_max != 0):
            Miscellaneous.protocol(
                "           ROI, set by the user: " + str(y_min) + "<y<" + str(y_max) + ", " + str(
                    x_min) + "<x<" + str(x_max), self.attached_log_file,
                precede_with_timestamp=False)

        self.work_next_task_signal.emit("Set alignment points")

    @QtCore.pyqtSlot()
    def execute_set_alignment_points(self):

        # If not executing in "automatic" mode, the APs are created on the main_gui thread.
        if self.main_gui.automatic:
            self.set_status_bar_processing_phase("creating alignment points")
            # Initialize the AlignmentPoints object.
            self.my_timer.create_no_check('Initialize alignment point object')
            self.alignment_points = AlignmentPoints(self.configuration, self.frames,
                                                    self.rank_frames, self.align_frames,
                                                    progress_signal=self.work_current_progress_signal)
            self.my_timer.stop('Initialize alignment point object')

            # Create alignment points, and create an image with wll alignment point boxes and
            # patches.
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
            Miscellaneous.protocol("           Number of alignment points selected: " + str(len(
                self.alignment_points.alignment_points)) + ", aps dropped because too dim: " + str(
                self.alignment_points.alignment_points_dropped_dim) + ", aps dropped because too "
                                                                      "little structure: " + str(
                self.alignment_points.alignment_points_dropped_structure), self.attached_log_file,
                                   precede_with_timestamp=False)

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
        self.stack_frames = StackFrames(self.configuration, self.frames, self.rank_frames,
                                        self.align_frames, self.alignment_points, self.my_timer,
                                        progress_signal=self.work_current_progress_signal,
                                        debug=self.debug_AP,
                                        create_image_window_signal=self.create_image_window_signal,
                                        update_image_window_signal=self.update_image_window_signal,
                                        terminate_image_window_signal=self.terminate_image_window_signal)

        # Stack all frames.
        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol(
                "+++ Start stacking " + str(self.alignment_points.stack_size) + " frames +++",
                self.attached_log_file)
        try:
            self.stack_frames.stack_frames()
        except Error as e:
            self.abort_job_signal.emit("Error: " + e.message + ", continuing with next job")
            return
        except Exception as e:
            self.abort_job_signal.emit(
                "Error in stacking frames: " + str(e) + ", continuing with next job")
            return

        if self.configuration.global_parameters_protocol_level > 1 and len(
            self.alignment_points.alignment_points) > 0:
            Miscellaneous.protocol("\n           Distribution of shifts at alignment points:",
                                   self.attached_log_file, precede_with_timestamp=False)
            Miscellaneous.protocol(self.stack_frames.print_shift_table() + "\n",
                                   self.attached_log_file, precede_with_timestamp=False)

        self.set_status_bar_processing_phase("merging AP patches")
        # Merge the stacked alignment point buffers into a single image.
        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ Start merging all alignment patches and the background +++",
                                   self.attached_log_file)

        try:
            self.stack_frames.merge_alignment_point_buffers()
        except Error as e:
            self.abort_job_signal.emit("Error: " + e.message + ", continuing with next job")
            return
        except Exception as e:
            self.abort_job_signal.emit(
                "Error in merging AP patches: " + str(e) + ", continuing with next job")
            return

        if self.stack_frames.border_y_low or self.stack_frames.border_y_high or self.stack_frames.border_x_low or self.stack_frames.border_x_high:
            if self.configuration.global_parameters_protocol_level > 1:
                Miscellaneous.protocol(
                    "           Stacked image borders cropped to avoid artifacts; top: " + str(
                        self.stack_frames.border_y_low) + ", bottom: " + str(
                        self.stack_frames.border_y_high) + ", left: " + str(
                        self.stack_frames.border_x_low) + ", right: " + str(
                        self.stack_frames.border_x_high) + "\n",
                        self.attached_log_file, precede_with_timestamp=False)

        # If the drizzle factor is 1.5, reduce the pixel resolution of the stacked image buffer
        # to half the size used in stacking.
        if self.configuration.drizzle_factor_is_1_5:
            self.stack_frames.half_stacked_image_buffer_resolution()

        self.work_next_task_signal.emit("Save stacked image")

    @QtCore.pyqtSlot()
    def execute_save_stacked_image(self):

        self.set_status_bar_processing_phase("saving result")

        # Create suffix containing parameter info.
        parameter_suffix = self.compose_suffix()
        if self.job.type == 'video':
            self.stacked_image_name = splitext(self.job.name)[0] + \
                                      self.configuration.stack_frames_suffix + \
                                      parameter_suffix + '.' + \
                                      self.configuration.global_parameters_image_format
        else:  # self.job.type == 'image'
            self.stacked_image_name = self.job.name + self.configuration.stack_frames_suffix + \
                                      parameter_suffix + '.' + \
                                      self.configuration.global_parameters_image_format

        # Save the image as 16bit int (color or mono).
        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ Start saving the stacked image +++", self.attached_log_file)
        self.my_timer.create_no_check('Saving the stacked image')
        Frames.save_image(self.stacked_image_name, self.stack_frames.stacked_image,
                          color=self.frames.color, avoid_overwriting=False,
                          header=self.configuration.global_parameters_version)
        self.my_timer.stop('Saving the stacked image')
        if self.configuration.global_parameters_protocol_level > 1:
            Miscellaneous.protocol(
                "           The stacked image was written to: " + self.stacked_image_name,
                self.attached_log_file, precede_with_timestamp=False)

        # If parameter info is to be included in output file names, compose the new name for
        # the attached log file. The existing file will be renamed after closing.
        if self.attached_log_file:
            self.attached_log_name_new = self.attached_log_name[:self.attached_log_name.index(
                '_stacking-log')] + '_stacking-log' + parameter_suffix + '.txt'

        # If postprocessing is included after stacking, set the stacked image as input.
        if self.configuration.global_parameters_include_postprocessing:
            self.postproc_input_image = self.stack_frames.stacked_image
            self.postproc_input_name = self.stacked_image_name
            self.postprocessed_image_name = PostprocDataObject.set_file_name_processed(
                self.stacked_image_name, self.configuration.postproc_suffix,
                self.configuration.global_parameters_image_format)
            self.work_next_task_signal.emit("Postprocessing")
        else:
            self.work_next_task_signal.emit("Next job")

            # Print timing info for this job.
            self.my_timer.stop('Execution over all')
            if self.configuration.global_parameters_protocol_level > 0:
                self.my_timer.protocol(self.attached_log_file)
            if self.attached_log_file:
                self.attached_log_file.close()
                if self.attached_log_name_new and self.attached_log_name_new != self.attached_log_name:
                    # If a logfile with the new name exists, remove it.
                    try:
                        remove(self.attached_log_name_new)
                    except:
                        pass
                    rename(self.attached_log_name, self.attached_log_name_new)
                    self.attached_log_name = self.attached_log_name_new

    def compose_suffix(self):
        """
        If process parameters are to be included in the stacked output file name, compose the
        suffix from the parameters computed during the workflow.

        :return: Additional suffix string
        """
        string = ''
        if self.configuration.global_parameters_parameters_in_filename:
            if self.configuration.global_parameters_stack_number_frames:
                string += '_f' + str(self.alignment_points.stack_size)
            if self.configuration.global_parameters_stack_percent_frames:
                string += '_p' + str(int(round(100*self.alignment_points.stack_size/self.frames.number)))
            if self.configuration.global_parameters_ap_box_size:
                string += '_b' + str(self.configuration.alignment_points_half_box_width*2)
            if self.configuration.global_parameters_ap_number:
                string += '_ap' + str(len(self.alignment_points.alignment_points))

        return string


    @QtCore.pyqtSlot()
    def execute_postprocess_image(self):

        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ Start postprocessing +++", self.attached_log_file)
        self.my_timer.create_no_check('Conputing image postprocessing')

        # Look up parameters of the last postproc version which was selected in interactive mode.
        version_index = self.configuration.postproc_data_object.version_selected
        postproc_layers = self.configuration.postproc_data_object.versions[version_index].layers
        rgb_automatic = self.configuration.postproc_data_object.versions[
            version_index].rgb_automatic
        rgb_gauss_width = self.configuration.postproc_data_object.versions[
            version_index].rgb_gauss_width
        rgb_resolution_index = self.configuration.postproc_data_object.versions[
            version_index].rgb_resolution_index
        shift_red = self.configuration.postproc_data_object.versions[version_index].shift_red
        shift_blue = self.configuration.postproc_data_object.versions[version_index].shift_blue

        try:
            # Auto-align RGB channels, if requested.
            if rgb_automatic:
                sharpening_input, self.configuration.postproc_data_object.versions[
                    version_index].shift_red, self.configuration.postproc_data_object.versions[
                    version_index].shift_blue = Miscellaneous.auto_rgb_align(
                    self.postproc_input_image, self.configuration.postproc_max_shift,
                    interpolation_factor=[1, 2, 4][rgb_resolution_index], reduce_output=True,
                    blur_strength=rgb_gauss_width)
            elif shift_red != (0., 0.) or shift_blue != (0., 0.):
                # Shift the image with the resolution given by the selected interpolation factor.
                interpolation_factor = [1, 2, 4][rgb_resolution_index]
                sharpening_input = Miscellaneous.shift_colors(self.postproc_input_image,
                                                           shift_red, shift_blue,
                                                           interpolate_input=interpolation_factor,
                                                           reduce_output=interpolation_factor)
            else:
                sharpening_input = self.postproc_input_image

            # Apply all sharpening layers of the postprocessing version selected last time.
            self.postprocessed_image = Miscellaneous.post_process(sharpening_input, postproc_layers)

        except Exception as e:
            self.abort_job_signal.emit(
                "Error in postprocessing: " + str(e) + ", continuing with next job")
            return

        self.my_timer.stop('Conputing image postprocessing')

        self.work_next_task_signal.emit("Save postprocessed image")

    @QtCore.pyqtSlot(object)
    def execute_save_postprocessed_image(self, postprocessed_image):

        # The signal payload is None only if the editor was left with "cancel" in interactive mode.
        # In this case, skip saving the result and proceed with the next job.
        if postprocessed_image is not None:

            # Print postprocessing info if sharpening layers have been applied or RGB alignment was
            # active.
            if self.configuration.global_parameters_protocol_level > 1:
                version_selected = self.configuration.postproc_data_object.version_selected
                postproc_version = self.configuration.postproc_data_object.versions[
                    self.configuration.postproc_data_object.version_selected]
                if version_selected or postproc_version.rgb_automatic:
                    Miscellaneous.print_postproc_parameters(postproc_version,
                                                            self.attached_log_file)
                else:
                    Miscellaneous.protocol(
                        "           The image was not modified in postprocessing.",
                        self.attached_log_file, precede_with_timestamp=False)

            self.set_status_bar_processing_phase("saving result")
            # Save the image as 16bit int (color or mono).
            if self.configuration.global_parameters_protocol_level > 0:
                Miscellaneous.protocol("+++ Start saving the postprocessed image +++",
                                       self.attached_log_file)
            self.my_timer.create_no_check('Saving the postprocessed image')
            Frames.save_image(self.postprocessed_image_name, postprocessed_image,
                              color=(len(postprocessed_image.shape) == 3), avoid_overwriting=False,
                              header=self.configuration.global_parameters_version)
            self.my_timer.stop('Saving the postprocessed image')
            if self.configuration.global_parameters_protocol_level > 1:
                Miscellaneous.protocol(
                    "           The postprocessed image was written to: " +
                    self.postprocessed_image_name,
                    self.attached_log_file, precede_with_timestamp=False)

        self.work_next_task_signal.emit("Next job")

        # Print timing info for this job.
        self.my_timer.stop('Execution over all')
        if self.configuration.global_parameters_protocol_level > 0:
            self.my_timer.protocol(self.attached_log_file)
        if self.attached_log_file:
            self.attached_log_file.close()
            # If the attached log name was defined for a stacking job, rename it to include
            # parameter information.
            if self.attached_log_name_new and self.attached_log_name_new != self.attached_log_name:
                try:
                    remove(self.attached_log_name_new)
                except:
                    pass
                rename(self.attached_log_name, self.attached_log_name_new)
                self.attached_log_name = self.attached_log_name_new

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
