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

Part of this module (in class "AlignmentPointEditor" was copied from
https://stackoverflow.com/questions/35508711/how-to-enable-pan-and-zoom-in-a-qgraphicsview

"""

from sys import exit, argv
from os import remove
from pathlib import Path

from PyQt5 import QtWidgets, QtCore

from main_gui import Ui_MainWindow

from configuration import Configuration
from configuration_editor import ConfigurationEditor
from job_editor import JobEditor
from rectangular_patch_editor import RectangularPatchEditorWidget
from frame_viewer import FrameViewerWidget
from alignment_points import AlignmentPoints
from alignment_point_editor import AlignmentPointEditorWidget
from shift_distribution_viewer import ShiftDistributionViewerWidget
from postproc_editor import PostprocEditorWidget
from miscellaneous import Miscellaneous
from workflow import Workflow


class PlanetarySystemStacker(QtWidgets.QMainWindow):
    """
    This class is the main class of the "Planetary System Stacker" software. It implements
    the main GUI for the communication with the user. It creates the workflow thread which controls
    all program activities asynchronously.

    """

    signal_frames = QtCore.pyqtSignal(str, str, bool)
    signal_rank_frames = QtCore.pyqtSignal()
    signal_align_frames = QtCore.pyqtSignal(int, int, int, int)
    signal_set_roi = QtCore.pyqtSignal(int, int, int, int)
    signal_set_alignment_points = QtCore.pyqtSignal()
    signal_compute_frame_qualities = QtCore.pyqtSignal()
    signal_stack_frames = QtCore.pyqtSignal()
    signal_save_stacked_image = QtCore.pyqtSignal()
    signal_postprocess_image = QtCore.pyqtSignal()
    signal_save_postprocessed_image = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        """
        Initialize the Planetary System Stacker environment.

        :param parent: None
        """

        # The (generated) QtGui class is contained in module main_gui.py.
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Make sure that the progress widgets retain their size when hidden.
        size_policy = self.ui.label_current_progress.sizePolicy()
        size_policy.setRetainSizeWhenHidden(True)
        self.ui.label_current_progress.setSizePolicy(size_policy)
        size_policy = self.ui.progressBar_current.sizePolicy()
        size_policy.setRetainSizeWhenHidden(True)
        self.ui.progressBar_current.setSizePolicy(size_policy)

        # Create configuration object and set configuration parameters to standard values.
        self.configuration = Configuration()

        # Look up the location and size of the main GUI. Replace the location parameters with those
        # stored in the configuration file when the GUI was closed last time. This way, the GUI
        # memorizes its location between MPM invocations.
        x0 = self.configuration.hidden_parameters_main_window_x0
        y0 = self.configuration.hidden_parameters_main_window_y0
        width = self.configuration.hidden_parameters_main_window_width
        height = self.configuration.hidden_parameters_main_window_height
        self.setGeometry(x0, y0, width, height)

        # Initialize variables.
        self.widget_saved = None

        # Write the program version into the window title.
        self.setWindowTitle(self.configuration.global_parameters_version)

        # Connect GUI events with methods of this class.
        self.ui.actionQuit.triggered.connect(self.closeEvent)
        self.ui.comboBox_back.currentTextChanged.connect(self.go_back)
        self.ui.pushButton_start.clicked.connect(self.play)
        self.ui.pushButton_pause.clicked.connect(self.pause)
        self.ui.pushButton_next_job.clicked.connect(self.go_next)
        self.ui.pushButton_quit.clicked.connect(self.close)
        self.ui.box_automatic.stateChanged.connect(self.automatic_changed)
        self.ui.actionLoad_video_directory.triggered.connect(self.load_video_directory)
        self.ui.actionSave.triggered.connect(self.save_result)
        self.ui.actionSave_as.triggered.connect(self.save_result_as)
        self.ui.actionEdit_configuration.triggered.connect(self.edit_configuration)
        self.ui.actionLoad_config.triggered.connect(self.load_config_file)
        self.ui.actionSave_config.triggered.connect(self.save_config_file)

        # Create the workflow thread and start it.
        self.thread = QtCore.QThread()
        self.workflow = Workflow(self)
        self.workflow.moveToThread(self.thread)
        self.workflow.work_next_task_signal.connect(self.work_next_task)
        self.workflow.work_current_progress_signal.connect(self.set_current_progress)
        self.workflow.set_status_bar_signal.connect(self.write_status_bar)
        # self.workflow.set_status_signal.connect(self.set_status)
        # self.workflow.set_error_signal.connect(self.show_error_message)
        self.thread.start()

        # Connect signals to start activities on the workflow thread (in method "work_next_task").
        self.signal_frames.connect(self.workflow.execute_frames)
        self.signal_rank_frames.connect(self.workflow.execute_rank_frames)
        self.signal_align_frames.connect(self.workflow.execute_align_frames)
        self.signal_set_roi.connect(self.workflow.execute_set_roi)
        self.signal_set_alignment_points.connect(self.workflow.execute_set_alignment_points)
        self.signal_compute_frame_qualities.connect(self.workflow.execute_compute_frame_qualities)
        self.signal_stack_frames.connect(self.workflow.execute_stack_frames)
        self.signal_save_stacked_image.connect(self.workflow.execute_save_stacked_image)
        self.signal_postprocess_image.connect(self.workflow.execute_postprocess_image)
        self.signal_save_postprocessed_image.connect(self.workflow.execute_save_postprocessed_image)

        # Initialize status variables
        self.automatic = self.ui.box_automatic.isChecked()
        self.busy = False
        self.pause = False
        self.job_number = 0
        self.job_index = 0
        self.job_names = []
        self.job_types = []
        self.activity = 'Read frames'

        # Initialize the "backwards" combobox: The user can only go back to those program steps
        # which have been executed already.
        self.set_previous_actions_button()

        # Deactivate GUI elements which do not make sense yet.
        self.activate_gui_elements(
            [self.ui.box_automatic, self.ui.comboBox_back, self.ui.pushButton_start,
             self.ui.pushButton_pause,
             self.ui.pushButton_next_job, self.ui.actionSave, self.ui.actionSave_as,
             self.ui.actionLoad_postproc_config, self.ui.actionSave_postproc_config,
             self.ui.actionEdit_postproc_config], False)
        self.show_current_progress_widgets(False)
        self.show_batch_progress_widgets(False)

        # If the configuration was not read in from a previous run (i.e. only default values have
        # been set so far), open the configuration editor GUI to let the user make adjustments if
        # necessary.
        if not self.configuration.config_file_exists:
            # Tell the user to begin with changing / confirming config parameters.
            self.write_status_bar(
                'Adapt configuration to your needs and / or confirm by pressing "OK".', 'red')
            self.edit_configuration()

        else:
            # Tell the user to begin with specifying jobs to be executed.
            self.write_status_bar(
                "Specify video(s) or dir(s) with image files to be stacked, or single image "
                "files for postprocessing (menu: File / Open).", 'red')

    def automatic_changed(self):
        """
        If the user checks / unchecks the "Automatic" checkbox, change the corresponding status
        variable.

        :return: -
        """

        self.automatic = not self.automatic

    def edit_configuration(self):
        """
        This method is invoked by selecting the "edit stacking config" menu entry.

        :return: -
        """

        # Display the configuration editor widget in the central QFrame.
        self.display_widget(ConfigurationEditor(self))

    def display_widget(self, widget, display=True):
        """
        Display a widget in the central main GUI location, or remove it from there.

        :param widget: Widget object to be displayed.
        :param display: If "True", display the widget. If "False", remove the current widget from
                        the GUI.
        :return: -
        """

        if display:
            if self.widget_saved:
                self.ui.verticalLayout_2.removeWidget(self.widget_saved)
                self.widget_saved.close()
            self.widget_saved = widget
            self.ui.verticalLayout_2.insertWidget(0, widget)
            self.ui.verticalLayout_2.setStretch(1, 0)
        else:
            if self.widget_saved:
                self.ui.verticalLayout_2.removeWidget(self.widget_saved)
                self.widget_saved.close()
                self.widget_saved = None

    def load_config_file(self):
        """
        Load a stacking configuration file (extension ".pss").
        :return: -
        """

        options = QtWidgets.QFileDialog.Options()
        filename = QtWidgets.QFileDialog.getOpenFileName(self, "Load configuration file",
                    self.configuration.hidden_parameters_current_dir, "*.pss", options=options)
        file_name = filename[0]

        # If a valid file was selected, read parameters from the file and open the configuration
        # editor.
        if file_name != '':
            # Read configuration parameters from the selected file.
            self.configuration.read_config(file_name=file_name)
            # Remember the current directory for next file dialog.
            self.configuration.current_dir = str(Path(file_name).parents[0])
            self.display_widget(ConfigurationEditor(self))

    def save_config_file(self):
        """
        Save all stacking parameters in a configuration file.

        :return: -
        """

        # Open the file chooser.
        options = QtWidgets.QFileDialog.Options()
        filename = QtWidgets.QFileDialog.getSaveFileName(self, "Save configuration file",
                    self.configuration.hidden_parameters_current_dir, "Config file (*.pss)",
                    options=options)

        # Store file only if the chooser did not return with a cancel.
        file_name = filename[0]
        if file_name != "":
            my_file = Path(file_name)

            # Remember the current directory for next file dialog.
            self.configuration.hidden_parameters_current_dir = str(my_file.parents[0])

            # If the config file exists, delete it first.
            if my_file.is_file():
                remove(str(my_file))
            self.configuration.write_config(file_name=str(my_file))

    def load_video_directory(self):
        """
        This method is invoked by selecting "Open" from the "file" menu.

        :return: -
        """

        # Open the job editor widget.
        self.display_widget(JobEditor(self))

    def go_back(self):
        """
        Repeat processing steps as specified via the choice of the "comboBox_back" button. The user
        can either repeat steps of the current job, or go back to the previous job.

        :return: -
        """

        # Get the choice of the combobox button.
        task = self.ui.comboBox_back.currentText()
        Miscellaneous.protocol("", self.workflow.attached_log_file,
                               precede_with_timestamp=False)
        Miscellaneous.protocol("+++ Repeating from task: " + task + " +++",
                               self.workflow.attached_log_file)

        # If the end of the job queue was reached, reverse the last job index increment.
        if self.job_index == self.job_number and self.job_index>0:
            self.job_index -= 1

        # Restart from the specified task within the current job.
        if task in ['Read frames', 'Rank frames', 'Align frames', 'Select stack size', 'Set ROI',
                    'Set alignment points', 'Compute frame qualities',
                    'Stack frames', 'Save stacked image', 'Postprocessing',
                    'Save postprocessed image']:
            # Make sure to remove any active interaction widget.
            self.display_widget(None, display=False)
            self.work_next_task(task)

        # Go back to the previous job and start with the first task.
        elif task == 'Previous job':
            self.job_index -= 1
            self.display_widget(None, display=False)
            self.work_next_task("Read frames")


    def play(self):
        """
        This method is invoked when the "Start / Cont." button is pressed.
        :return: -
        """

        # Start the next task.
        self.work_next_task(self.activity)

    def pause(self):
        """
        This method is invoked when the "Pause" button is pressed.
        :return: -
        """

        # If the pause flag is not yet set, append a message to the status bar.
        if not self.pause:
            self.append_status_bar(" Execution will be suspended after current phase.")
        # Set the pause flag.
        self.pause = True

    def go_next(self):
        """
        This method is invoked when the "Next Job" button is pressed.
        :return: -
        """

        self.work_next_task("Next job")

    def work_next_task(self, next_activity):
        """
        This is the central place where all activities are scheduled. Depending on the
         "next_activity" chosen, the appropriate activity is started on the workflow thread.

        :param next_activity: Activity to be performed next.
        :return: -
        """

        # Make sure not to process an empty job list, or a job index out of range.
        if not self.job_names or self.job_index >= self.job_number:
            return

        self.activity = next_activity

        # Deactivate the current progress widgets. They are reactivated when the workflow thread
        # sends a progress signal.
        self.show_current_progress_widgets(False)

        # If the "Pause" button was pressed during the last activity, stop the workflow.
        if self.pause:
            self.pause = False
            self.busy = False

        # Start workflow activities. When a workflow method terminates, it invokes this method on
        # the GUI thread, with "next_activity" denoting the next step in the processing chain.
        elif self.activity == "Read frames":
            # For the first activity (reading all frames from the file system) there is no
            # GUI interaction. Start the workflow action immediately.
            self.signal_frames.emit(self.job_names[self.job_index],
                                    self.job_types[self.job_index], False)
            self.busy = True

        elif self.activity == "Rank frames":

            # If batch mode is deselected, start GUI activity.
            # if not self.automatic:
            #     self.write_status_bar("Processing " + self.job_names[self.job_index] + ".", "black")
            #     self.place_holder_manual_activity('Rank frames')

            # Now start the corresponding action on the workflow thread.
            self.signal_rank_frames.emit()
            self.busy = True

        elif self.activity == "Align frames":

            # If manual stabilization patch selection was requested in 'Surface' mode, invoke the
            # patch editor.
            if not self.automatic and not self.configuration.align_frames_automation and \
                    self.configuration.align_frames_mode == 'Surface':
                border = self.configuration.align_frames_search_width

                # When the editor is finished, it sends a signal (last argument) to the workflow
                # thread with the four coordinate index bounds.
                rpew = RectangularPatchEditorWidget(self, self.workflow.frames.frames_mono(
                    self.workflow.rank_frames.frame_ranks_max_index)[border:-border,
                    border:-border], "With 'ctrl' and the left mouse button pressed, draw a "
                                     "rectangular patch to be used for frame alignment. Or just "
                                     "press 'OK / Cancel' (automatic selection).",
                                      self.signal_align_frames)

                self.display_widget(rpew)
                rpew.viewer.setFocus()

            else:
                # If all index bounds are set to zero, the stabilization patch is computed
                # automatically by the workflow thread.
                self.signal_align_frames.emit(0, 0, 0, 0)

            self.busy = True

        elif self.activity == "Select stack size":

            if not self.automatic:

                # Reset the ROI, if one was defined before.
                self.workflow.align_frames.reset_roi()

                # When the frame viewer is finished, it sends a signal which invokes this same
                # method on the main thread.
                fvw = FrameViewerWidget(self, self.workflow.configuration, self.workflow.frames,
                                        self.workflow.rank_frames, self.workflow.align_frames,
                                        self.workflow.attached_log_file,
                                        self.workflow.work_next_task_signal, "Set ROI")

                self.display_widget(fvw)
                fvw.frame_viewer.setFocus()

            else:
                # In automatic mode, nothing is to be done in the workflow thread. Start the next
                # activity on the main thread immediately.
                self.workflow.work_next_task_signal.emit("Set ROI")
            self.busy = True

        elif self.activity == "Set ROI":

            # Reset the ROI, if one was defined before.
            self.workflow.align_frames.reset_roi()

            if not self.automatic:

                # When the editor is finished, it sends a signal (last argument) to the workflow
                # thread with the four coordinate index bounds.
                rpew = RectangularPatchEditorWidget(self, self.workflow.align_frames.mean_frame,
                    "With 'crtl' and the left mouse button pressed, draw a rectangle to set the ROI,"
                    " or just press 'OK' (no ROI).", self.signal_set_roi)

                self.display_widget(rpew)
                rpew.viewer.setFocus()

            else:
                # If all index bounds are set to zero, no ROI is selected.
                self.signal_set_roi.emit(0, 0, 0, 0)
            self.busy = True

        elif self.activity == "Set alignment points":
            # In automatic mode, compute the AP grid automatically in the workflow thread. In this
            # case, the AlignmentPoints object is created there as well.
            if self.automatic:
                self.signal_set_alignment_points.emit()
                self.busy = True
            else:
                # If the APs are created interactively, create the AlignmentPoints object here, but
                # assign it to the workflow object.
                if self.configuration.global_parameters_protocol_level > 0:
                    Miscellaneous.protocol("+++ Start creating alignment points +++",
                                           self.workflow.attached_log_file)
                # Initialize the AlignmentPoints object.
                self.workflow.my_timer.create_no_check('Initialize alignment point object')
                self.workflow.alignment_points = AlignmentPoints(self.workflow.configuration,
                    self.workflow.frames, self.workflow.rank_frames, self.workflow.align_frames,
                    progress_signal=self.workflow.work_current_progress_signal)
                self.workflow.my_timer.stop('Initialize alignment point object')
                # Open the alignment point editor.
                apew = AlignmentPointEditorWidget(self, self.workflow.configuration,
                                                               self.workflow.align_frames,
                                                               self.workflow.alignment_points,
                                                               self.signal_set_alignment_points)

                self.display_widget(apew)
                apew.viewer.setFocus()

        elif self.activity == "Compute frame qualities":
            if not self.automatic:
                pass
            self.signal_compute_frame_qualities.emit()
            self.busy = True

        elif self.activity == "Stack frames":
            if not self.automatic:
                pass
            self.signal_stack_frames.emit()
            self.busy = True

        elif self.activity == "Save stacked image":
            if self.automatic:
                self.signal_save_stacked_image.emit()
            else:
                sdv = ShiftDistributionViewerWidget(self,
                                                    self.workflow.stack_frames.shift_distribution,
                                                    self.signal_save_stacked_image)
                self.display_widget(sdv)
            self.busy = True

        elif self.activity == "Postprocessing":
            if self.automatic:
                self.signal_postprocess_image.emit()
            else:
                if self.configuration.global_parameters_protocol_level > 0:
                    Miscellaneous.protocol("+++ Start postprocessing +++",
                                           self.workflow.attached_log_file)
                # In interactive mode the postprocessed image is computed in the GUI thread. The
                # resulting image is sent via the "signal_save_postprocessed_image" signal.
                pew = PostprocEditorWidget(self.workflow.configuration,
                                           self.workflow.postproc_input_image,
                                           self.workflow.postproc_input_name,
                                           self.write_status_bar,
                                           self.signal_save_postprocessed_image)
                self.display_widget(pew)
            self.busy = True

        elif self.activity == "Save postprocessed image":
            if not self.automatic:
                pass
            # This path is executed for "automatic" mode. In that case the postprocessed image
            # is available in the workflow object.
            if self.workflow.postprocessed_image is not None:
                self.signal_save_postprocessed_image.emit(self.workflow.postprocessed_image)
            # Otherwise it has been computed in the PostprocEditor on the GUI thread, and stored
            # in the postproc_data_object. If the editor was left with "cancel", the image is set
            # to None. In that case the workflow thread will not save the image.
            else:
                self.signal_save_postprocessed_image.emit(
                    self.workflow.configuration.postproc_data_object.versions[
                    self.workflow.configuration.postproc_data_object.version_selected].image)
            self.busy = True

        elif self.activity == "Next job":
            self.job_index += 1
            if self.job_index < self.job_number:
                # If the end of the queue is not reached yet, start with reading frames of next job.
                self.activity = "Read frames"
                if not self.automatic:
                    pass
                self.signal_frames.emit(self.job_names[self.job_index],
                                        self.job_types[self.job_index], False)
            else:
                # End of queue reached, give control back to the user.
                self.busy = False

        # Activate / Deactivate GUI elements depending on the current situation.
        self.update_status()

    def save_result(self):
        """
        save the result as 16bit Tiff at the standard location.

        :return: -
        """

        self.workflow.frames.save_image(self.workflow.stacked_image_name,
                                        self.workflow.stack_frames.stacked_image,
                                        color=self.workflow.frames.color, avoid_overwriting=False)

    def save_result_as(self):
        """
        save the result as 16bit Tiff at a location selected by the user.

        :return: -
        """

        options = QtWidgets.QFileDialog.Options()
        filename, extension = QtWidgets.QFileDialog.getSaveFileName(self,
            "Save result as 16bit Tiff image", self.workflow.stacked_image_name ,
            "Image Files (*.tiff)", options=options)

        if filename and extension:
            self.workflow.frames.save_image(filename, self.workflow.stack_frames.stacked_image,
                                            color=self.workflow.frames.color,
                                            avoid_overwriting=False)

    def place_holder_manual_activity(self, activity):
        # Ask the user for confirmation.
        quit_msg = "This is a placeholder for the manual activity " + activity + \
                   ". press OK when you want to continue with the workflow."
        QtWidgets.QMessageBox.question(self, 'Message', quit_msg,
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)

    def show_current_progress_widgets(self, show):
        """
        Show or hide the GUI widgets showing the progress of the current job.

        :param show: If "True", show the widgets, otherwise hide them.
        :return: -
        """

        if show:
            self.ui.progressBar_current.show()
            self.ui.label_current_progress.show()
        else:
            self.ui.progressBar_current.hide()
            self.ui.label_current_progress.hide()

    @QtCore.pyqtSlot(str, int)
    def set_current_progress(self, activity, percent):
        """
        Triggered by signal "work_current_progress_signal" on the workflow thread, show the progress
        of the current activity in a progress bar on the main GUI.

        :param activity: Textual description of the current activity.
        :param percent: Fraction of task finished (in %)
        :return: -
        """

        self.show_current_progress_widgets(True)
        self.ui.label_current_progress.setText(activity)
        self.ui.progressBar_current.setValue(percent)


    def show_batch_progress_widgets(self, show, value=0):
        """
        Show or hide the GUI widgets showing the progress of the batch.

        :param show: If "True", show the widgets, otherwise hide them.
        :return: -
        """

        if show:
            self.ui.progressBar_batch.show()
            self.ui.progressBar_batch.setValue(value)
            self.ui.label_batch_progress.show()
        else:
            self.ui.progressBar_batch.hide()
            self.ui.label_batch_progress.hide()

    def set_previous_actions_button(self):
        """
        Initialize the "backwards" combobox: The user can only go back to those program steps
        which have been executed already.

        :return: -
        """

        self.ui.comboBox_back.currentTextChanged.disconnect(self.go_back)
        self.ui.comboBox_back.clear()
        self.ui.comboBox_back.addItem('Go back to:')
        if self.job_index > 0 and self.job_number > 1:
            self.ui.comboBox_back.addItem('Previous job')
        if self.activity == "Read frames":
            self.ui.comboBox_back.addItems(['Read frames'])
        elif self.activity == "Rank frames":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames'])
        elif self.activity == "Align frames":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames'])
        elif self.activity == "Select stack size":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames',
                                            'Select stack size'])
        elif self.activity == "Set ROI":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames',
                                            'Select stack size', 'Set ROI'])
        elif self.activity == "Set alignment points":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames',
                                            'Select stack size', 'Set ROI', 'Set alignment points'])
        elif self.activity == "Compute frame qualities":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames',
                                            'Select stack size', 'Set ROI', 'Set alignment points',
                                            'Compute frame qualities'])
        elif self.activity == "Stack frames":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames',
                                            'Select stack size', 'Set ROI', 'Set alignment points',
                                            'Compute frame qualities', 'Stack frames'])
        elif self.activity == "Save stacked image":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames',
                                            'Select stack size', 'Set ROI', 'Set alignment points',
                                            'Compute frame qualities', 'Stack frames',
                                            'Save stacked image'])
        elif self.activity == "Postprocessing":
            if self.workflow.job_type == 'stacking':
                self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames',
                                                'Select stack size', 'Set ROI', 'Set alignment points',
                                                'Compute frame qualities', 'Stack frames',
                                                'Save stacked image', 'Postprocessing'])
            # This is to be added for both job types.
            self.ui.comboBox_back.addItems(['Postprocessing'])
        elif self.activity == "Save postprocessed image":
            if self.workflow.job_type == 'stacking':
                self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames',
                                                'Select stack size', 'Set ROI', 'Set alignment points',
                                                'Compute frame qualities', 'Stack frames',
                                                'Save stacked image', 'Postprocessing',
                                                'Save postprocessed image'])
            # This is to be added for both job types.
            self.ui.comboBox_back.addItems(['Postprocessing', 'Save postprocessed image'])
        elif self.activity == "Next job":
            if self.workflow.job_type == 'stacking':
                self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames',
                                                'Select stack size', 'Set ROI', 'Set alignment points',
                                                'Compute frame qualities', 'Stack frames',
                                                'Save stacked image'])
                if self.workflow.configuration.global_parameters_include_postprocessing == True:
                    self.ui.comboBox_back.addItems(['Postprocessing', 'Save postprocessed image'])
            else:
                self.ui.comboBox_back.addItems(['Postprocessing', 'Save postprocessed image'])
        self.ui.comboBox_back.setCurrentIndex(0)
        self.ui.comboBox_back.currentTextChanged.connect(self.go_back)


    def update_status(self):
        """
        Activate / Deactivate GUI elements depending on the current processing status.

        :return: -
        """

        # Depending on the current activity, the "previous action button" presents different
        # choices. One can only go back to activities which have been performed already.
        self.set_previous_actions_button()

        # While a computation is going on: Deactivate most buttons and menu entries.
        if self.busy:
            # For tasks with interactive GUI elements, activate the "Go back to:" button. During
            # the interaction, the status line shows the "busy" status, and the workflow is
            # suspended. By pressing "Go back to:", however, the user can restart from a previous
            # task.
            if self.activity not in ['Align frames', 'Select stack size', 'Set ROI',
                                     'Set alignment points']:
                self.activate_gui_elements([self.ui.comboBox_back], False)
            else:
                self.activate_gui_elements([self.ui.comboBox_back], True)
            self.activate_gui_elements([self.ui.pushButton_start,
                                        self.ui.pushButton_next_job, self.ui.menuFile,
                                        self.ui.menuEdit], False)
            self.activate_gui_elements([self.ui.pushButton_pause], True)
            self.write_status_bar("Busy processing " + self.job_names[self.job_index], "black")

        # In manual mode, activate buttons and menu entries. Update the status bar.
        else:
            self.activate_gui_elements([self.ui.menuFile, self.ui.menuEdit], True)
            self.activate_gui_elements([self.ui.pushButton_pause], False)
            if self.activity == "Next job":
                self.activate_gui_elements([self.ui.actionSave, self.ui.actionSave_as], True)
            else:
                self.activate_gui_elements([self.ui.actionSave, self.ui.actionSave_as], False)
            activated_buttons = []
            if self.job_index < self.job_number:
                activated_buttons.append(self.ui.pushButton_start)
            if self.job_index > 0 or self.activity != "Read frames":
                activated_buttons.append(self.ui.comboBox_back)
            if self.job_index < self.job_number - 1:
                activated_buttons.append(self.ui.pushButton_next_job)
            if activated_buttons:
                self.activate_gui_elements(activated_buttons, True)
                message = "To continue, press '" + \
                          self.button_get_description(activated_buttons[0]) + "'"
                if len(activated_buttons) > 1:
                    for button in activated_buttons[1:]:
                        message += ", or '" + self.button_get_description(button) + "'"
                message += ", or exit the program with 'Quit'."
                self.write_status_bar(message, "red")
            else:
                self.write_status_bar("Load new jobs, or quit.", "red")
                # self.ui.statusBar.setStyleSheet('color: red')

        self.show_batch_progress_widgets(self.job_number > 1)
        if self.job_number > 0:
            self.ui.progressBar_batch.setValue(int(100*self.job_index/self.job_number))

    def button_get_description(self, element):
        """
        Get the textual description of a button. Different methods must be used for "pushButton" and
        comboBoxes.

        :param element: Button object
        :return: Text written on the button (string)
        """

        try:
            description = element.text()
            return description
        except:
            pass
        try:
            description = element.currentText()
            return description
        except:
            return ""

    def activate_gui_elements(self, elements, enable):
        """
        Enable / Disable selected GUI elements.

        :param elements: List of GUI elements.
        :param enable: If "True", enable the elements; otherwise disable them.
        :return: -
        """

        for element in elements:
            element.setEnabled(enable)

    @QtCore.pyqtSlot(str, str)
    def write_status_bar(self, message, color):
        """
        Set the text in the status bar to "message".

        :param message: Text to be displayed
        :param color: Color in which the text is to be displayed (string)
        :return: -
        """

        self.ui.statusBar.showMessage(message)
        self.ui.statusBar.setStyleSheet('color: ' + color)

    def append_status_bar(self, append_message):
        """
        Append text to the current status bar message.

        :param append_message: Text to be appended.
        :return: -
        """

        self.ui.statusBar.showMessage(self.ui.statusBar.currentMessage() + append_message)

    def closeEvent(self, event=None):
        """
        This event is triggered when the user closes the main window by clicking on the cross in
        the window corner, or selects 'Quit' in the file menu.

        :param event: event object
        :return: -
        """

        # Ask the user for confirmation.
        quit_msg = "Are you sure you want to exit?"
        reply = QtWidgets.QMessageBox.question(self, 'Message', quit_msg,
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        # Positive reply: Do it.
        if reply == QtWidgets.QMessageBox.Yes:
            if event:
                event.accept()

            # If there is a image log file still open, close it.
            if self.workflow.attached_log_file:
                self.workflow.attached_log_file.close()

            # Store the geometry of main window, so it is placed the same at next program start.
            if self.windowState() != QtCore.Qt.WindowMaximized:
                (x0, y0, width, height) = self.geometry().getRect()
                self.configuration.hidden_parameters_main_window_x0 = x0
                self.configuration.hidden_parameters_main_window_y0 = y0
                self.configuration.hidden_parameters_main_window_width = width
                self.configuration.hidden_parameters_main_window_height = height

            # Write the current configuration to the ".ini" file in the user's home directory.
            self.configuration.write_config()
            exit(0)
        else:
            # No confirmation by the user: Don't stop program execution.
            if event:
                event.ignore()


if __name__ == "__main__":
    # The following steps are required to get the program running on Linux:
    #
    # - Add "export QT_XKB_CONFIG_ROOT=/usr/share/X11/xkb" to file .bashrc.
    #
    # - There is still a problem with fonts: PyInstaller seems to hardcode the path to fonts
    #   which do not make sense on another computer. This leads to error messages
    #   "Fontconfig error: Cannot load default config file", and a standard font is used
    #   instead.
    #
    # To run the PyInstaller, open a Terminal in PyCharm and enter
    # "pyinstaller planetary_system_stacker.spec" on Windows, or
    # "pyinstaller planetary_system_stacker_linux.spec" on Linux
    #

    app = QtWidgets.QApplication(argv)
    myapp = PlanetarySystemStacker()
    myapp.show()
    exit(app.exec_())
