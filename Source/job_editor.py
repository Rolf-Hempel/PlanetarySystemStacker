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
from pathlib import Path

from PyQt5 import QtWidgets, QtCore

from job_dialog import Ui_JobDialog


class FileDialog(QtWidgets.QFileDialog):
    """
    This is a variant of the regular FileDialog class which allows to choose files and directories
    at the same time.
    """

    signal_dialog_ready = QtCore.pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super(FileDialog, self).__init__(*args, **kwargs)
        # Do not use the native dialog of the OS. Otherwise the selection model tree is not
        # available as expected.
        self.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        self.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        self.tree = self.findChild(QtWidgets.QTreeView)

    def accept(self):
        """
        When the selection of files and directories is acknowledged, construct a list of strings
        with their names and send them via a signal to the job editor.

        :return: -
        """

        inds = self.tree.selectionModel().selectedIndexes()
        files = []
        for i in inds:
            if i.column() == 0:
                files.append(os.path.join(str(self.directory().absolutePath()), str(i.data())))
        self.signal_dialog_ready.emit(files)
        self.close()


class JobEditor(QtWidgets.QFrame, Ui_JobDialog):
    """
    Manage the list of jobs. Each item is either the name of a video file (.avi) or a directory
    containing image files of the same shape. Ask the user to add jobs to the list, or to remove
    existing entries. The interaction with the user is through the JobDialog class.
    """

    def __init__(self, parent_gui, parent=None):
        """
        Initialize the job editor. The widget has a fixed size and is rendered as a QFrame.

        :param parent_gui: GUI object by which the editor is invoked.
        :param parent: Parent object
        """

        QtWidgets.QFrame.__init__(self, parent)
        self.setupUi(self)

        self.setFrameShape(QtWidgets.QFrame.Panel)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setObjectName("configuration_editor")

        self.setFixedSize(900, 600)

        self.parent_gui = parent_gui
        self.configuration = self.parent_gui.configuration

        # Get a copy of the job names constructed so far. The editor only works on the copy, so
        # that in the case of "cancel" the original list is not changed.
        self.job_names = self.parent_gui.job_names.copy()

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.button_remove_jobs.clicked.connect(self.remove_job_list)
        self.button_add_jobs.clicked.connect(self.add_jobs)

        # Populate the job list widget with the current job list.
        self.populate_job_list()

    def populate_job_list(self):
        """
        Fill the central QListWidget with the current list of job names.

        :return: -
        """
        self.job_list_widget.clear()
        for job in self.job_names:
            item = QtWidgets.QListWidgetItem(job)
            self.job_list_widget.addItem(item)

    def add_jobs(self):
        """
        Open a file dialog for entering additional job names. Entries can either be video files or
        directories.

        :return: -
        """

        options = QtWidgets.QFileDialog.Options()
        message = "Select video file(s) or/and directories containing image files"
        self.file_dialog = FileDialog(self, message,
                                      self.configuration.hidden_parameters_current_dir,
                                      "Videos (*.avi)", options=options)

        # The list of strings with the new job names is sent by the FileDialog via the signal.
        self.file_dialog.signal_dialog_ready.connect(self.get_input_names)
        self.file_dialog.exec_()

    def get_input_names(self, input_names):
        """
        Receive the list of new job names from the FileDialog.

        :param input_names: List of strings with job names
        :return: -
        """
        if input_names:
            for entry in input_names:
                if entry not in self.job_names:
                    self.job_names.append(entry)

            # Save the current directory location. The next dialog will open at this position.
            self.configuration.hidden_parameters_current_dir = str(Path(input_names[0]).parents[0])
        self.populate_job_list()

    def remove_job_list(self):
        """
        Remove the job entries which are currently selected.

        :return: -
        """

        # Get the selected items from the central job list widget.
        items = self.job_list_widget.selectedItems()
        remove_list = []
        for item in items:
            remove_list.append(str(item.text()))
        input_names = []
        for item in self.job_names:
            if item not in remove_list:
                input_names.append(item)

        # Update the current job name list, and re-draw the job list widget.
        self.job_names = input_names
        self.populate_job_list()

    def accept(self):
        """
        If the OK button is clicked and the job list has been changed update the job list in the
        parent object.

        :return: -
        """

        self.parent_gui.job_names = self.job_names
        self.close()

    def reject(self):
        """
        The Cancel button is pressed, discard the changes and close the GUI window.
        :return: -
        """

        self.close()

    def closeEvent(self, event):
        """
        Remove the job editor widget from the parent GUI and close the editor.

        :param event: Close event object
        :return: -
        """

        self.parent_gui.display_widget(None, display=False)
        self.close()
