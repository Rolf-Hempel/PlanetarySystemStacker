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

        # Create the workflow thread and start it.
        self.thread = QtCore.QThread()
        self.workflow = Workflow(self)
        self.workflow.moveToThread(self.thread)
        self.workflow.work_task_finished_signal.connect(self.work_task_finished)
        # self.workflow.set_status_signal.connect(self.set_status)
        # self.workflow.set_error_signal.connect(self.show_error_message)
        self.thread.start()

        # Insert the photo viewer into the main GUI.
        # self.ImageWindow = PhotoViewer(self)
        # self.ImageWindow.setObjectName("ImageWindow")
        # self.ui.verticalLayout_3.insertWidget(1, self.ImageWindow, stretch=1)

        self.show_current_progress_widgets(False)
        self.show_batch_progress_widgets(False)

    def work_task_finished(self, activity):
        if activity == "frames":
            pass
        elif activity == "rank_frames":
            pass
        elif activity == "align_frames":
            pass
        elif activity == "set_roi":
            pass
        elif activity == "compute_frame_qualities":
            pass
        elif activity == "stack_frames":
            pass
        elif activity == "save_stacked_image":
            pass

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
