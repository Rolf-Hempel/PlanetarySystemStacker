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

from copy import deepcopy
from pathlib import Path

from PyQt6 import QtWidgets, QtCore, QtGui

from exceptions import InternalError
from job_dialog import Ui_JobDialog

# The following lists define the allowed file extensions for still images and videos.
image_extensions = ['.tif', '.tiff', '.fit', '.fits', '.jpg', '.png']
video_extensions = ['.avi', '.mov', '.mp4', '.ser']
extensions = image_extensions + video_extensions


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
        self.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        self.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
        self.tree = self.findChild(QtWidgets.QTreeView)

    def accept(self):
        """
        When the selection of files and directories is acknowledged, construct a list of strings
        with their names and send them via a signal to the job editor.

        :return: -
        """

        files = [self.directory().filePath(str(i.data())) for i in
                 self.tree.selectionModel().selectedIndexes() if i.column() == 0]
        self.signal_dialog_ready.emit(files)
        self.close()


class Job(object):
    """
    Objects of this class encapsulate all information describing a PSS job.
    """

    def __init__(self, job_name):
        """
        Initialize a Job object, given its name. The following instance variables are set:
           - name: Path name string coming from the file chooser.
           - file_name: File name string without path.
           - type: Either 'video', or 'image' for stacking jobs, or 'postproc' for postprocessing.
           - bayer_option_selected: Initialized to 'Auto detect color' for file types for which
                                    debayering is supported. Otherwise None.
           - bayer_pattern: Initialized to None

        :param job_name: Name of the job (str)
        """

        self.name = job_name
        path = Path(self.name)
        self.file_name = path.name

        # Bayer patterns are only defined for type 'video'.
        self.bayer_pattern = None
        self.bayer_option_selected = None

        # Set the type of the job based on the file name extension.
        if path.is_file():
            extension = path.suffix.lower()
            if extension in video_extensions:
                self.type = 'video'
                self.bayer_option_selected = 'Auto detect color'
            elif extension in image_extensions:
                self.type = 'postproc'
            else:
                raise InternalError(
                    "Unsupported file type '" + extension + "' specified for job")
        elif path.is_dir():
            self.type = 'image'
            self.bayer_option_selected = 'Auto detect color'
        else:
            raise InternalError("Cannot decide if input file is video or image directory")


class JoblistWidget(QtWidgets.QListWidget):
    """
    This is a customized version of the standard QListWidget which accepts "drag and drop" of
    picture files and directories. Dropped items are sent to the higher-level class as a list
    via the signal "signal_job_entries".
    """

    signal_job_entries = QtCore.pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setAcceptDrops(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """
        Handle a drop event if its mime type fits.

        :param event: Drop event
        :return: -
        """

        if event.mimeData().hasUrls():
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()

            # Construct a list with the job names received by the drop event.
            links = []
            for url in event.mimeData().urls():
                # Only local files (with appropriate extensions) and directories are accepted.
                if url.isLocalFile():
                    file_path = Path(str(url.toLocalFile()))
                    if file_path.is_dir() or file_path.suffix in extensions:
                        links.append(str(url.toLocalFile()))
            # Send the job name list to the upper-level method "get_input_names".
            self.signal_job_entries.emit(links)
        else:
            event.ignore()


class JobEditor(QtWidgets.QFrame, Ui_JobDialog):
    """
    Manage the list of jobs. Each item is either the name of a video file (.avi .mov .mp4 .ser) or a directory
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
        self.setWindowIcon(QtGui.QIcon('../PSS-Icon-64.ico'))

        # Replace the standard QListWidget with a customized version which implements
        # "drag and drop". It sends dropped items via its signal "signal_job_entries".
        self.job_list_widget = JoblistWidget()
        self.gridLayout.addWidget(self.job_list_widget, 0, 0, 1, 6)
        self.job_list_widget.signal_job_entries.connect(self.get_input_names)

        self.setFrameShape(QtWidgets.QFrame.Shape.Panel)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.setObjectName("configuration_editor")

        # The following line was deactivated. Otherwise the instructions under the joblist
        # would not show completely on full HD monitors.
        # self.setFixedSize(900, 600)

        self.parent_gui = parent_gui
        self.configuration = self.parent_gui.configuration

        # De-activate main gui menu entries while the job editor is active. They will be
        # re-activated upon leaving the editor with "OK" or "Cancel".
        self.parent_gui.activate_gui_elements(
            [self.parent_gui.ui.menuFile, self.parent_gui.ui.menuEdit,
             self.parent_gui.ui.menuCalibrate], False)

        # Set the window icon to the PSS icon.
        self.setWindowIcon(QtGui.QIcon(self.configuration.window_icon))

        # Get a copy of the jobs constructed so far. The editor only works on the copy, so
        # that in the case of "cancel" the original list is not changed.
        self.jobs = deepcopy(self.parent_gui.jobs)

        self.messageLabel.setStyleSheet('color: red')

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.button_remove_jobs.clicked.connect(self.remove_job_list)
        self.button_add_jobs.clicked.connect(self.add_jobs)

        # Install an event filter on the job list to enable the context menu.
        self.job_list_widget.installEventFilter(self)

        # Populate the job list widget with the current job list.
        self.populate_job_list()

        # If the job list is empty, open the input file dialog.
        if not self.jobs:
            self.add_jobs()

    def populate_job_list(self):
        """
        Fill the central QListWidget with the current list of job names.

        :return: -
        """
        self.job_list_widget.clear()
        for job in self.jobs:
            item = QtWidgets.QListWidgetItem(job.name)
            self.job_list_widget.addItem(item)

    def add_jobs(self):
        """
        Open a file dialog for entering additional job names. Entries can either be video files or
        directories for stacking, or single image files for postprocessing.

        :return: -
        """

        #options = QtWidgets.QFileDialog.Option
        message = "Select video file(s)/folders with image files for stacking, and/or " \
                  "image files for postprocessing"

        self.file_dialog = FileDialog(self, message,
                                      self.configuration.hidden_parameters_current_dir,
                                      "Videos (*.avi *.mov *.mp4 *.ser)")
        self.file_dialog.setNameFilters(["Still image folders / video files for stacking (*.avi *.mov *.mp4 *.ser)",
                                         "Images for postprocessing (*.tiff *.tif *.fit *.fits *.png *.jpg)"])
        self.file_dialog.selectNameFilter("Still image folders / video files for stacking (*.avi *.mov *.mp4 *.ser)")
        #self.file_dialog.setOptions(QtWidgets.QFileDialog.Option.ShowDirsOnly)

        # The list of strings with the new job names is sent by the FileDialog via the signal.
        self.file_dialog.signal_dialog_ready.connect(self.get_input_names)
        self.file_dialog.exec()

    def get_input_names(self, input_names):
        """
        Receive the list of new job names from the FileDialog.

        :param input_names: List of strings with job names
        :return: -
        """
        if input_names:
            for input_name in input_names:
                if input_name not in [job.name for job in self.jobs]:
                    try:
                        new_job = Job(input_name)
                        new_job.bayer_option_selected = self.configuration.frames_debayering_default
                        self.jobs.append(new_job)
                    except:
                        pass

            # Save the current directory location. The next dialog will open at this position.
            self.configuration.hidden_parameters_current_dir = str(Path(input_names[0]).parents[0])
        self.populate_job_list()

    def remove_job_list(self):
        """
        Remove the job entries which are currently selected.

        :return: -
        """

        # Get the selected items from the central job list widget.
        remove_list = [str(item.text()) for item in self.job_list_widget.selectedItems()]

        # Update the current job list, and re-draw the job list widget.
        self.jobs = [job for job in self.jobs if job.name not in remove_list]
        self.populate_job_list()

    def eventFilter(self, source, event):
        """
        The event filter intercepts context menu events on job list items. It is used to specify
        Bayer patterns explicitly.

        :param source: The widget for which the filter is activated
                       (in this case "self.job_list_widget")
        :param event: The event type for this filter ("QtCore.QEvent.ContextMenu")
        :return: True
        """

        # Restrict the selected items to the subset for which debayering is supported.
        selected_items = [item for item in self.job_list_widget.selectedItems() if
                          self.jobs[source.row(item)].type == 'video']

        self.pattern = None

        # If a context menu item is pressed, remember the pattern.
        def action1_triggered(state):
            self.pattern = 'Auto detect color'

        def action2_triggered(state):
            self.pattern = 'RGB'

        def action3_triggered(state):
            self.pattern = 'BGR'

        def action4_triggered(state):
            self.pattern = 'Grayscale'

        def action5_triggered(state):
            self.pattern = 'Force Bayer RGGB'

        def action6_triggered(state):
            self.pattern = 'Force Bayer GRBG'

        def action7_triggered(state):
            self.pattern = 'Force Bayer GBRG'

        def action8_triggered(state):
            self.pattern = 'Force Bayer BGGR'

        # The context menu is opened on a job list entry.
        if (event.type() == QtCore.QEvent.Type.ContextMenu and
                source is self.job_list_widget):

            # Show the context menu only if job items are selected for which debayering is
            # supported.
            if selected_items:
                # Create a list of patterns which are checked by the selected items initially.
                checked_patterns = [self.jobs[source.row(item)].bayer_option_selected
                                    for item in selected_items]

                # Create the context menu. Mark those patterns checked which have been set for at least
                # one selected job list entry.
                menu = QtWidgets.QMenu()
                action1 = QtWidgets.QAction('Auto detect color', menu, checkable=True)
                action1.triggered.connect(action1_triggered)
                if 'Auto detect color' in checked_patterns:
                    action1.setChecked(True)
                menu.addAction(action1)
                menu.addSeparator()
                action2 = QtWidgets.QAction('RGB', menu, checkable=True)
                action2.triggered.connect(action2_triggered)
                if 'RGB' in checked_patterns:
                    action2.setChecked(True)
                menu.addAction(action2)
                action3 = QtWidgets.QAction('BGR', menu, checkable=True)
                action3.triggered.connect(action3_triggered)
                if 'BGR' in checked_patterns:
                    action3.setChecked(True)
                menu.addAction(action3)
                action4 = QtWidgets.QAction('Grayscale', menu, checkable=True)
                action4.triggered.connect(action4_triggered)
                if 'Grayscale' in checked_patterns:
                    action4.setChecked(True)
                menu.addAction(action4)
                action5 = QtWidgets.QAction('Force Bayer RGGB', menu, checkable=True)
                action5.triggered.connect(action5_triggered)
                if 'Force Bayer RGGB' in checked_patterns:
                    action5.setChecked(True)
                menu.addAction(action5)
                action6 = QtWidgets.QAction('Force Bayer GRBG', menu, checkable=True)
                action6.triggered.connect(action6_triggered)
                if 'Force Bayer GRBG' in checked_patterns:
                    action6.setChecked(True)
                menu.addAction(action6)
                action7 = QtWidgets.QAction('Force Bayer GBRG', menu, checkable=True)
                action7.triggered.connect(action7_triggered)
                if 'Force Bayer GBRG' in checked_patterns:
                    action7.setChecked(True)
                menu.addAction(action7)
                action8 = QtWidgets.QAction('Force Bayer BGGR', menu, checkable=True)
                action8.triggered.connect(action8_triggered)
                if 'Force Bayer BGGR' in checked_patterns:
                    action8.setChecked(True)
                menu.addAction(action8)

                # Identify the selected items and their locations in the job list. Set the selected
                # Bayer pattern in the corresponding job objects.
                if menu.exec_(event.globalPos()) and self.pattern is not None:
                    for item in selected_items:
                        row = source.row(item)
                        self.jobs[row].bayer_option_selected = self.pattern
                return True

        return super(JobEditor, self).eventFilter(source, event)

    def keyPressEvent(self, event):
        """
        The + and - keys are used for zooming.

        :param event: event object
        :return: -
        """

        # If the "+" key is pressed, zoom in. If "-" is pressed, zoom out.
        if event.key() == QtCore.Qt.Key_Delete:
            self.remove_job_list()
        else:
            super(JobEditor, self).keyPressEvent(event)

    def accept(self):
        """
        If the OK button is clicked and the job list has been changed update the job list in the
        parent object.

        :return: -
        """

        # Update the job list and reset the current job index to the first entry.
        self.parent_gui.jobs = self.jobs
        self.parent_gui.job_number = len(self.jobs)
        self.parent_gui.job_index = 0
        self.parent_gui.activity = "Read frames"

        # Re-activate the main GUI menus, and activate the "Automatic" checkbox.
        self.parent_gui.activate_gui_elements(
            [self.parent_gui.ui.menuFile, self.parent_gui.ui.menuEdit,
             self.parent_gui.ui.menuCalibrate, self.parent_gui.ui.box_automatic], True)
        self.parent_gui.update_status()
        self.close()

    def reject(self):
        """
        The Cancel button is pressed, discard the changes and close the GUI window.
        :return: -
        """

        # Re-activate the main GUI menus.
        self.parent_gui.activate_gui_elements(
            [self.parent_gui.ui.menuFile, self.parent_gui.ui.menuEdit,
             self.parent_gui.ui.menuCalibrate], True)
        self.close()

    def closeEvent(self, event):
        """
        Remove the job editor widget from the parent GUI and close the editor.

        :param event: Close event object
        :return: -
        """

        self.parent_gui.display_widget(None, display=False)
        self.close()
