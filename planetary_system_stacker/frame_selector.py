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

Part of this module (in class "AlignmentPointEditor" was copied from
https://stackoverflow.com/questions/35508711/how-to-enable-pan-and-zoom-in-a-qgraphicsview

"""
# The following PyQt5 imports must precede any matplotlib imports. This is a workaround
# for a Matplotlib 2.2.2 bug.

from glob import glob
from sys import argv, exit
from time import sleep

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from configuration import Configuration
from exceptions import Error
from frame_selector_gui import Ui_frame_selector
from frame_viewer import FrameViewer
from frames import Frames
from rank_frames import RankFrames
from miscellaneous import Miscellaneous


class VideoFrameSelector(FrameViewer):
    """
    This widget implements a frame viewer for frames in a video file. Panning and zooming is
    implemented by using the mouse and scroll wheel.

    """

    resized = QtCore.pyqtSignal()

    def __init__(self, frames, index_included, frame_index=0):
        super(VideoFrameSelector, self).__init__()
        self.frames = frames
        self.index_included = index_included
        self.frame_index = frame_index

        self.setPhoto(self.frame_index)

    def setPhoto(self, index):
        """
        Convert a grayscale image to a pixmap and assign it to the photo object. If the image is
        marked as excluded from the stacking workflow, place a crossed-out red circle in the
        upper left image corner.

        :param index: Index into the frame list. Frames are assumed to be grayscale image in format
                      float32.
        :return: -
        """

        # Indicate that an image is being loaded.
        self.image_loading_busy = True

        image = self.frames.frames_mono(index)

        super(VideoFrameSelector, self).setPhoto(image,
                                                overlay_exclude_mark=not self.index_included[index])


class FrameSelectorWidget(QtWidgets.QFrame, Ui_frame_selector):
    """
    This widget implements a frame viewer together with control elements to visualize frame
    qualities, and to manipulate the stack limits.
    """

    frame_player_start_signal = QtCore.pyqtSignal()

    def __init__(self, parent_gui, configuration, frames, rank_frames, stacked_image_log_file,
                 signal_finished):
        """
        Initialization of the widget.

        :param parent_gui: Parent GUI object
        :param configuration: Configuration object with parameters
        :param frames: Frames object with all video frames
        :param rank_frames: RankFrames object with global quality ranks (between 0. and 1.,
                            1. being optimal) for all frames
        :param stacked_image_log_file: Log file to be stored with results, or None.
        :param signal_finished: Qt signal to trigger the next activity when the viewer exits.
        """

        super(FrameSelectorWidget, self).__init__(parent_gui)
        self.setupUi(self)

        # Keep references to upper level objects.
        self.parent_gui = parent_gui
        self.configuration = configuration
        self.stacked_image_log_file = stacked_image_log_file
        self.signal_finished = signal_finished
        self.frames = frames
        self.index_included = frames.index_included.copy()
        self.quality_sorted_indices = rank_frames.quality_sorted_indices
        self.rank_indices = rank_frames.rank_indices

        # Start with ordering frames by quality. This can be changed by the user using a radio
        # button.
        self.frame_ordering = "quality"

        # Initialize the frame list selection.
        self.items_selected = None
        self.indices_selected = None

        # Set colors for the frame list.
        self.background_included = QtGui.QColor(130, 255, 130)
        self.foreground_included = QtGui.QColor(0, 0, 0)
        self.background_excluded = QtGui.QColor(120, 120, 120)
        self.foreground_excluded = QtGui.QColor(255, 255, 255)

        self.addButton.clicked.connect(self.use_triggered)
        self.removeButton.clicked.connect(self.not_use_triggered)

        # Be careful: Indices are counted from 0, while widget contents are counted from 1 (to make
        # it easier for the user.
        self.quality_index = 0
        self.frame_index = self.quality_sorted_indices[self.quality_index]

        # Set up the frame selector and put it in the upper left corner.
        self.frame_selector = VideoFrameSelector(self.frames, self.index_included, self.frame_index)
        self.frame_selector.setObjectName("frame_selector")
        self.gridLayout.addWidget(self.frame_selector, 0, 0, 2, 3)

        # Initialize the list widget.
        self.fill_list_widget()
        self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listWidget.installEventFilter(self)
        self.listWidget.itemClicked.connect(self.select_items)
        self.listWidget.currentRowChanged.connect(self.synchronize_slider)

        # Group widget elements which are to be blocked during player execution in a list.
        self.widget_elements = [self.listWidget,
                                self.slider_frames,
                                self.addButton,
                                self.removeButton,
                                self.pushButton_play,
                                self.GroupBox_frame_sorting]

        # Initialize a variable for communication with the frame_player object later.
        self.run_player = False

        # Create the frame player thread and start it. The player displays frames in succession.
        # It is pushed on a different thread because otherwise the user could not stop it before it
        # finishes.
        self.player_thread = QtCore.QThread()
        self.frame_player = FramePlayer(self)
        self.frame_player.setParent(self)
        self.frame_player.moveToThread(self.player_thread)
        self.frame_player.block_widgets_signal.connect(self.block_widgets)
        self.frame_player.unblock_widgets_signal.connect(self.unblock_widgets)
        self.frame_player.set_photo_signal.connect(self.frame_selector.setPhoto)
        self.frame_player.set_slider_value.connect(self.slider_frames.setValue)
        self.frame_player_start_signal.connect(self.frame_player.play)
        self.player_thread.start()

        # Initialization of GUI elements
        self.slider_frames.setMinimum(1)
        self.slider_frames.setMaximum(self.frames.number)
        self.slider_frames.setValue(self.quality_index + 1)
        self.radioButton_quality.setChecked(True)

        self.gridLayout.setColumnStretch(0, 7)
        self.gridLayout.setColumnStretch(1, 0)
        self.gridLayout.setColumnStretch(2, 0)
        self.gridLayout.setColumnStretch(3, 0)
        self.gridLayout.setColumnStretch(4, 1)
        self.gridLayout.setRowStretch(0, 0)
        self.gridLayout.setRowStretch(1, 0)

        # Connect signals with slots.
        self.buttonBox.accepted.connect(self.done)
        self.buttonBox.rejected.connect(self.reject)
        self.slider_frames.valueChanged.connect(self.slider_frames_changed)
        self.pushButton_play.clicked.connect(self.pushbutton_play_clicked)
        self.pushButton_stop.clicked.connect(self.pushbutton_stop_clicked)
        self.radioButton_quality.toggled.connect(self.radiobutton_quality_changed)

        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("+++ Start selecting frames +++", self.stacked_image_log_file)

    def block_widgets(self):
        """
        Disable GUI elements to prevent unwanted user interaction while the frame player is running.

        :return: -
        """

        # De-select all items in the listWidget when the player starts.
        self.listWidget.clearSelection()

        for element in self.widget_elements:
            element.setDisabled(True)

    def unblock_widgets(self):
        """
        Enable GUI elements which were temporarily blocked when the player stops.

        :return: -
        """

        for element in self.widget_elements:
            element.setDisabled(False)

        self.listWidget.setFocus()

    def fill_list_widget(self):
        """
        Initialize the list widget with frames, ordered as selected (by rank or chronological).
        Set the colors of each item according to its current inclusion / exclusion state.

        :return: -
        """

        # Initialize the inclusion / exclusion state of frames in the frame list.
        self.listWidget.clear()
        for i in range(self.frames.number_original):
            if self.frame_ordering == "quality":
                frame_number = self.quality_sorted_indices[i]
            else:
                frame_number = i

            if self.index_included[frame_number]:
                item = QtWidgets.QListWidgetItem("Frame %i included" % (frame_number + 1))
                item.setBackground(self.background_included)
                item.setForeground(self.foreground_included)
            else:
                item = QtWidgets.QListWidgetItem("Frame %i excluded" % (frame_number + 1))
                item.setBackground(self.background_excluded)
                item.setForeground(self.foreground_excluded)
            self.listWidget.addItem(item)

        # Set the list widget to the current position.
        if self.frame_ordering == "quality":
            self.listWidget.setCurrentRow(self.quality_index,
                                          QtCore.QItemSelectionModel.SelectCurrent)
        else:
            self.listWidget.setCurrentRow(self.frame_index,
                                          QtCore.QItemSelectionModel.SelectCurrent)

    def select_items(self):
        """
        If a list item or a range of items is selected, store the items and corresponding indices.
        Synchronize the frame slider and frame viewer with the the current list position.

        :return: -
        """

        self.listWidget.currentItem().setSelected(True)
        self.items_selected = self.listWidget.selectedItems()

        if self.frame_ordering == "quality":
            self.indices_selected = [self.quality_sorted_indices[self.listWidget.row(item)] for item
                                     in self.items_selected]
            self.frame_index = self.indices_selected[0]
            self.quality_index = self.rank_indices[self.frame_index]
        else:
            self.indices_selected = [self.listWidget.row(item) for item in self.items_selected]
            self.frame_index = self.indices_selected[0]
            self.quality_index = self.rank_indices[self.frame_index]

        self.synchronize_slider()

    def synchronize_slider(self):
        """
        Set the frame slider to the value currently selected in the listWidget. Update the photo
        displayed in the viewer.

        :return: -
        """

        # Block slider signals to avoid a shortcut.
        self.slider_frames.blockSignals(True)

        if self.frame_ordering == "quality":
            self.quality_index = self.listWidget.currentRow()
            self.frame_index = self.quality_sorted_indices[self.quality_index]
            self.slider_frames.setValue(self.quality_index + 1)
        else:
            self.frame_index = self.listWidget.currentRow()
            self.quality_index = self.rank_indices[self.frame_index]
            self.slider_frames.setValue(self.frame_index + 1)

        # Unblock the slider signals again.
        self.slider_frames.blockSignals(False)

        # Update the image in the viewer.
        self.frame_selector.setPhoto(self.frame_index)

    def eventFilter(self, source, event):
        """
        This eventFilter is listening for events on the listWidget. List items can be marked as
        included / excluded by either using a context menu, by pressing the "+" or "-" buttons, or
        by pressing the keyboard keys "+" or "-".

        :param source: Source object to listen on.
        :param event: Event found
        :return: -
        """

        if source is self.listWidget:

            # Open a context menu with two choices. Depending on the user's choice, either
            # trigger the "use_triggered" or "not_use_triggered" method below.
            if event.type() == QtCore.QEvent.ContextMenu:
                menu = QtWidgets.QMenu()
                action1 = QtWidgets.QAction('Use for stacking', menu)
                action1.triggered.connect(self.use_triggered)
                menu.addAction((action1))
                action2 = QtWidgets.QAction("Don't use for stacking", menu)
                action2.triggered.connect(self.not_use_triggered)
                menu.addAction((action2))
                menu.exec_(event.globalPos())

            # Do the same as above if the user prefers to use the keyboard keys "+" or "-".
            elif event.type() == QtCore.QEvent.KeyPress:
                if event.key() == Qt.Key_Plus:
                    self.use_triggered()
                elif event.key() == Qt.Key_Minus:
                    self.not_use_triggered()
                elif event.key() == Qt.Key_Escape:
                    self.items_selected = []
                    self.indices_selected = []
                    self.listWidget.clearSelection()

        return super(FrameSelectorWidget, self).eventFilter(source, event)

    def use_triggered(self):
        """
        The user has selected a list item or a range of items to be included in the stacking
        workflow. Change the appearance of the list entry, update the "index_included" values for
        the corresponding frames, and reload the image of the current index. The latter step is
        important to update the overlay mark in the upper left image corner.

        :return: -
        """

        self.select_items()
        if self.items_selected:
            for index, item in enumerate(self.items_selected):
                index_selected = self.indices_selected[index]
                frame_selected = index_selected + 1
                item.setText("Frame %i included" % frame_selected)
                item.setBackground(self.background_included)
                item.setForeground(QtGui.QColor(0, 0, 0))
                self.index_included[index_selected] = True
                self.frame_selector.setPhoto(self.frame_index)

    def not_use_triggered(self):
        """
        Same as above in case the user has de-selected a list item or a range of items.
        :return: -
        """

        self.select_items()
        if self.items_selected:
            for index, item in enumerate(self.items_selected):
                index_selected = self.indices_selected[index]
                frame_selected = index_selected + 1
                item.setText("Frame %i excluded" % frame_selected)
                item.setBackground(self.background_excluded)
                item.setForeground(QtGui.QColor(255, 255, 255))
                self.index_included[index_selected] = False
                self.frame_selector.setPhoto(self.frame_index)

    def slider_frames_changed(self):
        """
        The frames slider is changed by the user. Update the frame in the viewer and scroll the
        frame list to show the current frame index.

        :return: -
        """

        # Again, please note the difference between indexing and GUI displays.
        index = self.slider_frames.value() - 1

        # Differentiate between frame ordering (by quality or chronologically).
        if self.frame_ordering == "quality":
            self.frame_index = self.quality_sorted_indices[index]
            self.quality_index = index

        else:
            self.frame_index = index
            self.quality_index = self.rank_indices[self.frame_index]

        # Adjust the frame list and select the current frame.

        self.listWidget.setCurrentRow(index, QtCore.QItemSelectionModel.SelectCurrent)

        # Update the image in the viewer.
        self.frame_selector.setPhoto(self.frame_index)
        self.listWidget.setFocus()

    def radiobutton_quality_changed(self):
        """
        Toggle back and forth between frame ordering modes. The frame slider is updated to reflect
        the index of the current frame in the new ordering scheme. The frame ordering in the list
        widget is changed.

        :return: -
        """

        # Block listWidget signals. Otherwise, changes to the widget triggered by changing the
        # slider would cause trouble.
        self.listWidget.blockSignals(True)

        if self.frame_ordering == "quality":
            self.frame_ordering = "chronological"
            self.slider_frames.setValue(self.frame_index + 1)
        else:
            self.frame_ordering = "quality"
            self.slider_frames.setValue(self.quality_index + 1)

        self.fill_list_widget()

        # Unblock listWidget signals again.
        self.listWidget.blockSignals(False)

    def pushbutton_play_clicked(self):
        """
        Start the frame player if it is not running already.

        :return: -
        """

        self.frame_player_start_signal.emit()

    def pushbutton_stop_clicked(self):
        """
        When the frame player is running, it periodically checks this variable. If it is set to
        False, the player stops.

        :return: -
        """

        if self.frame_player.run_player:
            self.frame_player.run_player = False

    def done(self):
        """
        On exit from the frame viewer, update the selection status of all frames and send a
        completion signal.

        :return: -
        """

        # Check if the status of frames has changed.
        indices_included = []
        indices_excluded = []
        for index in range(self.frames.number_original):
            if self.index_included[index] and not self.frames.index_included[index]:
                indices_included.append(index)
                self.frames.index_included[index] = True
            elif not self.index_included[index] and self.frames.index_included[index]:
                indices_excluded.append(index)
                self.frames.index_included[index] = False

        # Write the changes in frame selection to the protocol.
        if self.configuration.global_parameters_protocol_level > 1:
            if indices_included:
                Miscellaneous.protocol(
                    "           The user has included the following frames into the stacking "
                    "workflow: " + str(
                        [item + 1 for item in indices_included]), self.stacked_image_log_file,
                    precede_with_timestamp=False)
            if indices_excluded:
                Miscellaneous.protocol(
                    "           The user has excluded the following frames from the stacking "
                    "workflow: " + str(
                        [item + 1 for item in indices_excluded]), self.stacked_image_log_file,
                    precede_with_timestamp=False)
            frames_remaining = sum(self.frames.index_included)
            if frames_remaining != self.frames.number:
                Miscellaneous.protocol("           " + str(
                    frames_remaining) + " frames will be used in the stacking workflow.",
                                       self.stacked_image_log_file, precede_with_timestamp=False)

        # Send a completion message. The "execute_rank_frames" method is triggered on the workflow
        # thread. The signal payload is True if the status was changed for at least one frame.
        # In this case, the index translation table is updated before the frame ranking starts.
        if self.parent_gui is not None:
            self.signal_finished.emit()

        # Close the Window.
        self.player_thread.quit()
        self.close()

    def reject(self):
        """
        If the "cancel" button is pressed, the coordinates are not stored.

        :return: -
        """

        # Send a completion message.
        if self.parent_gui is not None:
            self.signal_finished.emit()

        # Close the Window.
        self.player_thread.quit()
        self.close()


class FramePlayer(QtCore.QObject):
    """
    This class implements a video player using the FrameViewer and the control elements of the
    FrameViewerWidget. The player is started by the widget on a separate thread. This way the user
    can instruct the GUI to stop the running player.

    """
    block_widgets_signal = QtCore.pyqtSignal()
    unblock_widgets_signal = QtCore.pyqtSignal()
    set_photo_signal = QtCore.pyqtSignal(int)
    set_slider_value = QtCore.pyqtSignal(int)

    def __init__(self, frame_selector_widget):
        super(FramePlayer, self).__init__()

        # Store a reference of the frame selector widget and create a list of GUI elements. This
        # makes it easier to perform the same operation on all elements.
        self.frame_selector_widget = frame_selector_widget

        # Set the delay time between frames.
        self.delay_between_frames = 0.1

        # Initialize a variable used to stop the player in the GUI thread.
        self.run_player = False

    def play(self):
        """
        Start the player.
        :return: -
        """

        # Block signals from GUI elements to avoid cross-talk, and disable them to prevent unwanted
        # user interaction.
        self.block_widgets_signal.emit()

        # Set the player running.
        self.run_player = True

        # The frames are ordered by their quality.
        if self.frame_selector_widget.frame_ordering == "quality":

            # The player stops when the end of the video is reached, or when the "run_player"
            # variable is set to False in the GUI thread.
            while self.frame_selector_widget.quality_index < self.frame_selector_widget.frames.number_original \
                    - 1 and self.run_player:

                if not self.frame_selector_widget.frame_selector.image_loading_busy:
                    self.frame_selector_widget.quality_index += 1
                    self.frame_selector_widget.frame_index = \
                        self.frame_selector_widget.quality_sorted_indices[
                            self.frame_selector_widget.quality_index]

                    self.set_slider_value.emit(self.frame_selector_widget.quality_index + 1)
                    self.set_photo_signal.emit(self.frame_selector_widget.frame_index)

                # Insert a short pause to keep the video from running too fast.
                sleep(self.delay_between_frames)

        else:
            # The same for chronological frame ordering.
            while self.frame_selector_widget.frame_index < self.frame_selector_widget.frames.number_original \
                    - 1 and self.run_player:
                if not self.frame_selector_widget.frame_selector.image_loading_busy:
                    self.frame_selector_widget.frame_index += 1
                    self.frame_selector_widget.quality_index = \
                        self.frame_selector_widget.quality_sorted_indices.index(
                            self.frame_selector_widget.frame_index)
                    self.set_slider_value.emit(self.frame_selector_widget.frame_index + 1)
                    self.set_photo_signal.emit(self.frame_selector_widget.frame_index)

                sleep(self.delay_between_frames)

        self.run_player = False

        # Re-set the GUI elements to their normal state.
        self.unblock_widgets_signal.emit()


if __name__ == '__main__':
    # Images can either be extracted from a video file or a batch of single photographs. Select
    # the example for the test run.
    type = 'video'
    if type == 'image':
        names = glob(
            'Images/2012*.tif')  # names = glob.glob('Images/Moon_Tile-031*ap85_8b.tif')  # names
        # = glob.glob('Images/Example-3*.jpg')
    else:
        names = 'Videos/another_short_video.avi'
    print(names)

    # Get configuration parameters.
    configuration = Configuration()
    configuration.initialize_configuration()
    try:
        frames = Frames(configuration, names, type=type)
        print("Number of images: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Error as e:
        print("Error: " + e.message)
        exit()

    # Rank the frames by their overall local contrast.
    rank_frames = RankFrames(frames, configuration)
    rank_frames.frame_score()

    print("Best frame index: " + str(rank_frames.frame_ranks_max_index))

    app = QtWidgets.QApplication(argv)
    window = FrameSelectorWidget(None, configuration, frames, rank_frames, None, None)
    window.setMinimumSize(800, 600)
    # window.showMaximized()
    window.show()
    app.exec_()

    exit()
