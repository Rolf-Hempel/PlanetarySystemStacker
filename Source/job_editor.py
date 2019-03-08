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

from pathlib import Path
from PyQt5 import QtWidgets, QtCore
import os

from job_dialog import Ui_JobDialog

class FileDialog(QtWidgets.QFileDialog):

    signal_dialog_ready = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(FileDialog, self).__init__(*args, **kwargs)
        self.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        self.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        self.tree = self.findChild(QtWidgets.QTreeView)

    def accept(self):
        inds = self.tree.selectionModel().selectedIndexes()
        files = []
        for i in inds:
            if i.column() == 0:
                files.append(os.path.join(str(self.directory().absolutePath()), str(i.data())))
        self.selectedFiles = files
        self.selected_files = files
        print("Selected files: " + str(self.selectedFiles))
        self.hide()
        self.signal_dialog_ready.emit()

class JobEditor(QtWidgets.QFrame, Ui_JobDialog):
    """
    Manage the list of jobs. Each item is either the name of a video file (.avi) or a directory
    containing image files of the same shape. Ask the user to add jobs to the list, or to remove
    existing entries. The interaction with the user is through the JobDialog class.
    """

    def __init__(self, parent_gui, parent=None):
        QtWidgets.QFrame.__init__(self, parent)
        self.setupUi(self)

        self.setFrameShape(QtWidgets.QFrame.Panel)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setObjectName("configuration_editor")

        self.setFixedSize(900, 600)

        self.parent_gui = parent_gui
        self.configuration = self.parent_gui.configuration
        self.job_names = self.parent_gui.job_names.copy()

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.button_remove_jobs.clicked.connect(self.remove_job_list)
        self.button_add_jobs.clicked.connect(self.add_jobs)

        self.populate_job_list()

    def populate_job_list(self):
        self.job_list_widget.clear()
        for job in self.job_names:
            item = QtWidgets.QListWidgetItem(job)
            self.job_list_widget.addItem(item)

    def add_jobs(self):
        options = QtWidgets.QFileDialog.Options()
        message = "Select video file(s) or/and directories containing image files"
        self.file_dialog = FileDialog(self, message,
                                               self.configuration.hidden_parameters_current_dir,
                                               "Videos (*.avi)", options=options)
        self.file_dialog.signal_dialog_ready.connect(self.get_input_names)
        self.file_dialog.exec_()

    def get_input_names(self):
        input_names = self.file_dialog.selected_files
        if input_names:
            for entry in input_names:
                if entry not in self.job_names:
                    self.job_names.append(entry)
            self.configuration.hidden_parameters_current_dir = str(Path(input_names[0]).parents[0])
        self.populate_job_list()


    def remove_job_list(self):
        items = self.job_list_widget.selectedItems()
        remove_list = []
        for item in items:
            remove_list.append(str(item.text()))
        print(remove_list)
        input_names = []
        for item in self.job_names:
            if item not in remove_list:
                input_names.append(item)
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

        self.parent_gui.display_widget(None, display=False)
        self.close()

