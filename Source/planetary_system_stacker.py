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
from time import sleep

from PyQt5 import QtWidgets, QtCore

from main_gui import Ui_MainWindow

from configuration import Configuration
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

        # Initialize the path to the home directory.
        self.current_dir = str(Path.home())

        # Create configuration object and set configuration parameters to standard values.
        self.configuration = Configuration()

        # Write the program version into the window title.
        self.setWindowTitle(self.configuration.version)

        self.ui.comboBox_back.addItems(['Previous Job'])
        self.ui.actionQuit.triggered.connect(self.closeEvent)

        # Create the workflow thread and start it.
        self.thread = QtCore.QThread()
        self.workflow = Workflow(self)
        self.workflow.moveToThread(self.thread)
        self.workflow.work_next_task_signal.connect(self.work_next_task)
        # self.workflow.set_status_signal.connect(self.set_status)
        # self.workflow.set_error_signal.connect(self.show_error_message)
        self.thread.start()

        self.signal_frames.connect(self.workflow.execute_frames)
        self.signal_rank_frames.connect(self.workflow.execute_rank_frames)
        self.signal_align_frames.connect(self.workflow.execute_align_frames)
        self.signal_set_roi.connect(self.workflow.execute_set_roi)
        self.signal_compute_frame_qualities.connect(self.workflow.execute_compute_frame_qualities)
        self.signal_stack_frames.connect(self.workflow.execute_stack_frames)
        self.signal_save_stacked_image.connect(self.workflow.execute_save_stacked_image)

        # Insert the photo viewer into the main GUI.
        # self.ImageWindow = PhotoViewer(self)
        # self.ImageWindow.setObjectName("ImageWindow")
        # self.ui.verticalLayout_3.insertWidget(1, self.ImageWindow, stretch=1)

        self.show_current_progress_widgets(False)
        self.show_batch_progress_widgets(False)

        # Initialize status variables
        self.automatic = self.ui.box_automatic.isChecked()
        self.job_number = 0
        self.job_index = 0
        self.job_names = []
        self.job_types = []
        self.current_activity = None

    def work_next_task(self, activity):
        self.set_previous_actions_button(activity)
        self.update_status()
        if activity == "frames":
            self.signal_frames.emit(self.job_names[self.job_index],
                                    self.job_types[self.job_index], False)
        if activity == "rank_frames":
            if not self.automatic:
                pass
            self.signal_rank_frames.emit()
        elif activity == "align_frames":
            if not self.automatic:
                pass
            self.signal_align_frames.emit()
        elif activity == "set_roi":
            if not self.automatic:
                pass
            self.signal_set_roi.emit()
        elif activity == "compute_frame_qualities":
            if not self.automatic:
                pass
            self.signal_compute_frame_qualities.emit()
        elif activity == "stack_frames":
            if not self.automatic:
                pass
            self.signal_stack_frames.emit()
        elif activity == "save_stacked_image":
            if not self.automatic:
                pass
            self.signal_save_stacked_image.emit()
        elif activity == "next_job":
            self.job_index += 1
            if self.job_index < self.job_number:
                if not self.automatic:
                    pass
                self.signal_frames.emit(self.job_names[self.job_index],
                                        self.job_types[self.job_index], False)

    def show_current_progress_widgets(self, show):
        if show:
            self.ui.progressBar_current.show()
            self.ui.label_current_progress.show()
        else:
            self.ui.progressBar_current.hide()
            self.ui.label_current_progress.hide()

    def show_batch_progress_widgets(self, show):
        if show:
            self.ui.progressBar_batch.show()
            self.ui.label_batch_progress.show()
        else:
            self.ui.progressBar_batch.hide()
            self.ui.label_batch_progress.hide()

    def set_previous_actions_button(self, next_activity):
        self.ui.comboBox_back.clear()
        if self.job_index > 1:
            self.ui.comboBox_back.addItem('Previous job')
        if next_activity == "rank_frames":
            self.ui.comboBox_back.addItems(['Read frames'])
        elif next_activity == "align_frames":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames'])
        elif next_activity == "set_roi":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames'])
        elif next_activity == "compute_frame_qualities":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames',
                                            'Set ROI'])
        elif next_activity == "stack_frames":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames', 'Set ROI',
                                            'Compute frame qualities'])
        elif next_activity == "save_stacked_image":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames', 'Set ROI',
                                            'Compute frame qualities', 'Stack frames'])
        elif next_activity == "next_job":
            self.ui.comboBox_back.addItems(['Read frames', 'Rank frames', 'Align frames', 'Set ROI',
                                            'Compute frame qualities', 'Stack frames',
                                            'Save stacked image'])

    def update_status(self):
        if self.automatic:
            self.activate_gui_elements([self.ui.comboBox_back, self.ui.pushButton_start,
                                        self.ui.pushButton_next_job, self.ui.menuFile], False)
        else:
            self.activate_gui_elements([self.ui.pushButton_start, self.ui.menuFile], True)
            self.activate_gui_elements([self.ui.comboBox_back], self.job_index > 0)
            self.activate_gui_elements([self.ui.pushButton_next_job],
                                        self.job_index < self.job_number - 1)

        self.show_current_progress_widgets(self.job_number > 0)
        self.show_batch_progress_widgets(self.job_number > 1)

    def activate_gui_elements(self, elements, enable):
        for element in elements:
            element.setEnabled(enable)

    def closeEvent(self, evnt):
        """
        This event is triggered when the user closes the main window by clicking on the cross in
        the window corner.

        :param evnt: event object
        :return: -
        """

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
    myapp.showMaximized()
    sys.exit(app.exec_())
