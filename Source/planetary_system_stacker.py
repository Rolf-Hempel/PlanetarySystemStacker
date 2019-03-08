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

import sys
from pathlib import Path

from PyQt5 import QtWidgets, QtCore

from main_gui import Ui_MainWindow

from configuration import Configuration
from configuration_editor import ConfigurationEditor
from job_editor import JobEditor
from workflow import Workflow


class PlanetarySystemStacker(QtWidgets.QMainWindow):
    """
    This class is the main class of the "Planetary System Stacker" software. It implements
    the main GUI for the communication with the user. It creates the workflow thread which controls
    all program activities asynchronously.

    """

    signal_frames = QtCore.pyqtSignal(str, str, bool)
    signal_rank_frames = QtCore.pyqtSignal()
    signal_align_frames = QtCore.pyqtSignal(bool, int, int, int, int)
    signal_set_roi = QtCore.pyqtSignal(int, int, int, int)
    signal_set_alignment_points = QtCore.pyqtSignal()
    signal_compute_frame_qualities = QtCore.pyqtSignal()
    signal_stack_frames = QtCore.pyqtSignal()
    signal_save_stacked_image = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        """
        Initialize the Planetary System Stacker environment.

        :param parent: None
        """

        # The (generated) QtGui class is contained in module main_gui.py.
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

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


        self.ui.actionQuit.triggered.connect(self.closeEvent)

        self.ui.pushButton_start.clicked.connect(self.play)
        self.ui.actionLoad_video_directory.triggered.connect(self.load_video_directory)
        self.ui.actionEdit_configuration.triggered.connect(self.edit_configuration)
        self.ui.actionLoad_config.triggered.connect(self.load_config_file)
        self.ui.actionSave_config.triggered.connect(self.save_config_file)


        # Create the workflow thread and start it.
        self.thread = QtCore.QThread()
        self.workflow = Workflow(self)
        self.workflow.moveToThread(self.thread)
        self.workflow.work_next_task_signal.connect(self.work_next_task)
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

        # Insert the photo viewer into the main GUI.
        # self.ImageWindow = PhotoViewer(self)
        # self.ImageWindow.setObjectName("ImageWindow")
        # self.ui.verticalLayout_3.insertWidget(1, self.ImageWindow, stretch=1)

        # Initialize status variables
        self.automatic = self.ui.box_automatic.isChecked()
        self.job_number = 0
        self.job_index = 0
        self.job_names = []
        self.job_types = []
        self.activities = ['Read frames', 'Rank frames', 'Align frames', 'Set ROI',
                           'Set alignment points', 'Compute frame qualities', 'Stack frames',
                           'Save stacked image', 'Next job']
        self.activity = 'Read frames'

        # Initialize the "backwards" combobox: The user can only go back to those program steps
        # which have been executed already.
        self.set_previous_actions_button(self.activity)

        # Deactivate GUI elements which do not make sense yet.
        self.activate_gui_elements(
            [self.ui.comboBox_back, self.ui.pushButton_start, self.ui.pushButton_stop,
             self.ui.pushButton_next_job, self.ui.actionSave, self.ui.actionSave_as,
             self.ui.actionEdit_postproc_config], False)
        self.show_current_progress_widgets(False)
        self.show_batch_progress_widgets(False)

        # If the configuration was not read in from a previous run (i.e. only default values have
        # been set so far), open the configuration editor GUI to let the user make adjustments if
        # necessary.
        if not self.configuration.config_file_exists:
            self.edit_configuration()

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
                os.remove(str(my_file))
            self.configuration.write_config(file_name=str(my_file))

    def load_video_directory(self):
        """
        This method is invoked by selecting "Open" from the "file" menu.

        :return: -
        """

        # Open the job editor widget.
        self.display_widget(JobEditor(self))

    def play(self):
        """
        This method is invoked when the "Start / Cont." button is pressed.
        :return:
        """

        # Start the next task.
        self.work_next_task(self.activity)

    def work_next_task(self, next_activity):
        """
        This is the central place where all activities are scheduled. Depending on the
         "next_activity" chosen, the appropriate activity is started on the workflow thread.

        :param next_activity: Activity to be performed next.
        :return: -
        """

        self.activity = next_activity

        # Depending on the current activity, the "previous action button" presents different
        # choices. One can only go back to activities which have been performed already.
        self.set_previous_actions_button(self.activity)

        # Activate / Deactivate GUI elements depending on the current situation.
        self.update_status()

        # Start workflow activities. When a workflow method terminates, it invokes this method on
        # the GUI thread, with "next_activity" denoting the next step in the processing chain.
        if self.activity == "Read frames":
            # For the first activity (reading all frames from the file system) there is no
            # GUI interaction. Start the workflow action immediately.
            self.signal_frames.emit(self.job_names[self.job_index],
                                    self.job_types[self.job_index], False)
        if self.activity == "Rank frames":
            # If batch mode is deselected, start GUI activity.
            if not self.automatic:
                pass
            # Now start the corresponding action on the workflow thread.
            self.signal_rank_frames.emit()
        elif self.activity == "Align frames":
            if not self.automatic:
                pass
            self.signal_align_frames.emit(True, 0, 0, 0, 0)
        elif self.activity == "Set ROI":
            if not self.automatic:
                pass
            self.signal_set_roi.emit()
        elif self.activity == "Set alignment points":
            if not self.automatic:
                pass
            self.signal_set_alignment_points.emit()
        elif self.activity == "Compute frame qualities":
            if not self.automatic:
                pass
            self.signal_compute_frame_qualities.emit()
        elif self.activity == "Stack frames":
            if not self.automatic:
                pass
            self.signal_stack_frames.emit()
        elif self.activity == "Save stacked image":
            if not self.automatic:
                pass
            self.signal_save_stacked_image.emit()
        elif self.activity == "Next job":
            # If there are more jobs on the batch list, increment the job counter and reset the
            # "self.activity" variable to the beginning of the processing workflow.
            self.job_index += 1
            if self.job_index < self.job_number:
                self.activity = "Read frames"
                if not self.automatic:
                    pass
                self.signal_frames.emit(self.job_names[self.job_index],
                                        self.job_types[self.job_index], False)

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

    def show_batch_progress_widgets(self, show):
        """
        Show or hide the GUI widgets showing the progress of the batch.

        :param show: If "True", show the widgets, otherwise hide them.
        :return: -
        """

        if show:
            self.ui.progressBar_batch.show()
            self.ui.label_batch_progress.show()
        else:
            self.ui.progressBar_batch.hide()
            self.ui.label_batch_progress.hide()

    def set_previous_actions_button(self, next_activity):
        """
        Initialize the "backwards" combobox: The user can only go back to those program steps
        which have been executed already.

        :param next_activity: The next step in the processing workflow.
        :return: -
        """

        self.ui.comboBox_back.clear()
        if self.job_index > 1:
            self.ui.comboBox_back.addItem('Previous job')
        if next_activity == "Read frames":
            self.ui.comboBox_back.addItems(['Read frames'])
        elif next_activity == "Rank frames":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames'])
        elif next_activity == "Align frames":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames'])
        elif next_activity == "Set ROI":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames',
                                            'Set ROI'])
        elif next_activity == "Set alignment points":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames',
                                            'Set ROI', 'Set alignment points'])
        elif next_activity == "Compute frame qualities":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames', 'Set ROI',
                                            'Set alignment points', 'Compute frame qualities'])
        elif next_activity == "Stack frames":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames', 'Set ROI',
                                            'Set alignment points', 'Compute frame qualities',
                                            'Stack frames'])
        elif next_activity == "Save stacked image":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames', 'Set ROI',
                                            'Set alignment points', 'Compute frame qualities',
                                            'Stack frames', 'Save stacked image'])
        elif next_activity == "Next job":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames', 'Set ROI',
                                            'Set alignment points', 'Compute frame qualities',
                                            'Stack frames', 'Save stacked image'])

    def update_status(self):
        """
        Activate / Deactivate GUI elements depending on the current processing status.

        :return: -
        """

        # In batch mode: Deactivate most buttons and menu entries.
        if self.automatic:
            self.activate_gui_elements([self.ui.comboBox_back, self.ui.pushButton_start,
                                        self.ui.pushButton_next_job, self.ui.menuFile], False)
        # In manual mode, activate most buttons and menu entries.
        else:
            self.activate_gui_elements([self.ui.pushButton_start, self.ui.menuFile], True)
            # self.activate_gui_elements([self.ui.comboBox_back], self.job_index > 0)
            self.activate_gui_elements([self.ui.pushButton_next_job],
                                        self.job_index < self.job_number - 1)

        self.show_current_progress_widgets(self.job_number > 0)
        self.show_batch_progress_widgets(self.job_number > 1)

    def activate_gui_elements(self, elements, enable):
        """
        Enable / Disable selected GUI elements.

        :param elements: List of GUI elements.
        :param enable: If "True", enable the elements; otherwise disable them.
        :return: -
        """

        for element in elements:
            element.setEnabled(enable)

    def closeEvent(self, evnt):
        """
        This event is triggered when the user closes the main window by clicking on the cross in
        the window corner.

        :param evnt: event object
        :return: -
        """

        # Store the geometry of main window, so it is placed the same at next program start.
        if self.windowState() != QtCore.Qt.WindowMaximized:
            (x0, y0, width, height) = self.geometry().getRect()
            self.configuration.hidden_parameters_main_window_x0 = x0
            self.configuration.hidden_parameters_main_window_y0 = y0
            self.configuration.hidden_parameters_main_window_width = width
            self.configuration.hidden_parameters_main_window_height = height

        # Write the current configuration to the ".ini" file in the user's home directory.
        self.configuration.write_config()
        sys.exit(0)


if __name__ == "__main__":
    # The following four lines are a workaround to make PyInstaller work. Remove them when the
    # PyInstaller issue is fixed. Additionally, the following steps are required to get the
    # program running on Linux:
    #
    # - Add "export QT_XKB_CONFIG_ROOT=/usr/share/X11/xkb" to file .bashrc.
    #
    # - There is still a problem with fonts: PyInstaller seems to hardcode the path to fonts
    #   which do not make sense on another computer. This leads to error messages
    #   "Fontconfig error: Cannot load default config file", and a standard font is used
    #   instead.
    #
    # To run the PyInstaller, open a Terminal in PyCharm and enter
    # "pyinstaller moon_panorama_maker_windows.spec" on Windows, or
    # "pyinstaller moon_panorama_maker_linux.spec" on Linux
    #
    import os

    if getattr(sys, 'frozen', False):
        here = os.path.dirname(sys.executable)
        sys.path.insert(1, here)

    app = QtWidgets.QApplication(sys.argv)
    myapp = PlanetarySystemStacker()
    # myapp.setGeometry(200, 200, 1200, 800)
    myapp.show()
    sys.exit(app.exec_())
