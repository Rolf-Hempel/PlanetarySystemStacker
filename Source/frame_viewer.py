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
from time import time, sleep

import matplotlib.pyplot as plt
from numpy import array, full, uint8, uint16
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

from align_frames import AlignFrames
from configuration import Configuration
from exceptions import NotSupportedError, InternalError
from frame_viewer_gui import Ui_frame_viewer
from frames import Frames
from miscellaneous import Miscellaneous
from rank_frames import RankFrames


class MatplotlibWidget(Canvas):
    """
    This widget creates a plot of frame qualities, either sorted chronologically or by quality.

    """

    def __init__(self, configuration, rank_frames, parent=None):
        """
        Initialize the widget.

        :param configuration: Configuration object with parameters
        :param rank_frames: RankFrames object with global quality ranks (between 0. and 1.,
                            1. being optimal) for all frames
        :param parent: Parent object
        """

        super(MatplotlibWidget, self).__init__(Figure())

        self.setParent(parent)
        self.configuration = configuration
        self.rank_frames = rank_frames
        self.line_chronological = None
        self.line_quality = None
        self.dot = None
        self.line_quality_cutoff = None
        self.patch_quality_cutoff = None

        plt.rcParams.update({'font.size': 8})

        self.fig, self.ax = plt.subplots()
        self.ax.invert_xaxis()
        plt.subplots_adjust(left=0.23, right=0.95, top=0.98, bottom=0.12)

    def renew_plot(self, index, frame_ordering, alignment_points_frame_number):
        """
        Update the complete plot, including the quality line, the cutoff line, and the dot at the
        position of the current frame.

        :param index: Index of the current frame (relative to the selected order)
        :param frame_ordering: Ordering of frames, either "chronological" or "quality"
        :param alignment_points_frame_number: Number of frames to be stacked at each AP.
        :return: -
        """

        self.ax.clear()
        self.line_quality_cutoff = None
        self.patch_quality_cutoff = None

        # The quality axis starts with 1. (highest quality).
        self.ax.invert_xaxis()

        # Remember objects drawn into the plot.
        self.line_chronological = None
        self.line_quality = None
        self.dot = None
        self.line_quality_cutoff = None

        # Plot the new data.
        self.plot_data(frame_ordering)
        self.plot_cutoff_lines(frame_ordering, alignment_points_frame_number)
        self.plot_dot(index)

    def plot_data(self, frame_ordering):
        """
        Plot the quality line. Frames (y axis) are ordered as specified with parameter
         "frame_ordering".

        :param frame_ordering: Ordering of frames, either "chronological" or "quality"
        :return: -
        """

        self.ax.set_ylim(0, self.rank_frames.number + 1)
        self.y = array(range(1, self.rank_frames.number + 1))
        if frame_ordering == "chronological":
            if self.line_chronological is not None:
                self.line_chronological.remove()
            self.x = array(self.rank_frames.frame_ranks)
            plt.ylabel('Frame numbers ordered chronologically')
            plt.gca().invert_yaxis()
            plt.xlabel('Quality')
            self.line_chronological, = plt.plot(self.x, self.y, lw=1, color='blue')
            plt.grid(True)
        else:
            if self.line_quality is not None:
                self.line_quality.remove()
            self.x = array(
                [self.rank_frames.frame_ranks[i] for i in self.rank_frames.quality_sorted_indices])
            plt.ylabel('Frame numbers ordered by quality')
            plt.gca().invert_yaxis()
            plt.xlabel('Quality')
            self.line_quality, = plt.plot(self.x, self.y, lw=1, color='green')
            plt.grid(True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_cutoff_lines(self, frame_ordering, alignment_points_frame_number):
        """
        Plot a horizontal (if frames are ordered by quality) or vertical (if frames are ordered
        chronologically) line to separate frames to be stacked from the other ones.

        :param frame_ordering: Ordering of frames, either "chronological" or "quality"
        :param alignment_points_frame_number: Number of frames to be stacked at each AP.
        :return: -
        """
        if frame_ordering == "chronological":
            if self.line_quality_cutoff is not None:
                self.line_quality_cutoff.remove()
            if self.patch_quality_cutoff is not None:
                self.patch_quality_cutoff.remove()
            quality_cutoff = self.rank_frames.frame_ranks[
                self.rank_frames.quality_sorted_indices[alignment_points_frame_number]]
            x_cutoff = full((self.rank_frames.number,), quality_cutoff)
            self.line_quality_cutoff, = plt.plot(x_cutoff, self.y, lw=1, color='orange')
            width = 1. - quality_cutoff
            height = self.rank_frames.number - 1
            xy = (quality_cutoff, 1.)
        else:
            if self.line_quality_cutoff is not None:
                self.line_quality_cutoff.remove()
            if self.patch_quality_cutoff is not None:
                self.patch_quality_cutoff.remove()
            y_cutoff = full((self.rank_frames.number,), alignment_points_frame_number)
            self.line_quality_cutoff, = plt.plot(self.x, y_cutoff, lw=1, color='orange')
            width = 1. - self.x[-1]
            height = alignment_points_frame_number
            xy = (self.x[-1], 0.)
        self.patch_quality_cutoff = patches.Rectangle(xy, width, height, linewidth=None,
                                                      facecolor=(0.5, 0.2, 0., 0.15))
        self.ax.add_patch(self.patch_quality_cutoff)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_dot(self, frame_index):
        """
        Plot a dot on the quality line at the position of the current frame.

        :param frame_index: Frame index
        :return: -
        """

        if self.dot is not None:
            self.dot.remove()
        self.dot = plt.scatter(self.x[frame_index], self.y[frame_index], s=20, color='red')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class FrameViewer(QtWidgets.QGraphicsView):
    """
    This widget implements a frame viewer. Panning and zooming is implemented by using the mouse
    and scroll wheel.

    """

    resized = QtCore.pyqtSignal()

    def __init__(self, frames, align_frames):
        super(FrameViewer, self).__init__()
        self._zoom = 0
        self._empty = True

        self.frames = frames
        self.align_frames = align_frames
        self.frame_index = 0

        # Initialize a flag which indicates when an image is being loaded.
        self.image_loading_busy = False

        # Initialize the scene. This object handles mouse events if not in drag mode.
        self._scene = QtWidgets.QGraphicsScene()
        # Initialize the photo object. No image is loaded yet.
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)

        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.drag_mode = True

        self.setPhoto(self.frame_index)
        self.resized.connect(self.fitInView)

        # Set the focus on the viewer.
        self.setFocus()

    def resizeEvent(self, event):
        self.resized.emit()
        return super(FrameViewer, self).resizeEvent(event)

    def hasPhoto(self):
        return not self._empty

    def fitInView(self):
        """
        Scale the scene such that it fits into the window completely.

        :return: -
        """

        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, index):
        """
        Convert a grayscale image to a pixmap and assign it to the photo object.

        :param index: Index into the frame list. Frames are assumed to be grayscale image in format
                      float32.
        :return: -
        """

        # Indicate that an image is being loaded.
        self.image_loading_busy = True
        image = self.frames.frames_mono(index)[self.align_frames.intersection_shape[0][0] -
                                                    self.align_frames.frame_shifts[index][0]:
                                                    self.align_frames.intersection_shape[0][1] -
                                                    self.align_frames.frame_shifts[index][0],
                                                    self.align_frames.intersection_shape[1][0] -
                                                    self.align_frames.frame_shifts[index][1]:
                                                    self.align_frames.intersection_shape[1][1] -
                                                    self.align_frames.frame_shifts[index][1]]

        # Convert the monochrome image into uint8 format. If the frame type is uint16, values
        # correspond to 16bit resolution.
        if image.dtype == uint16:
            image_uint8 = (image[:, :] / 256.).astype(uint8)
        elif image.dtype == uint8:
            image_uint8 = image.astype(uint8)
        else:
            raise NotSupportedError("Attempt to set a photo in frame viewer with type neither"
                                    " uint8 nor uint16")

        self.shape_y = image_uint8.shape[0]
        self.shape_x = image_uint8.shape[1]
        qt_image = QtGui.QImage(image_uint8, self.shape_x, self.shape_y, self.shape_x,
                                QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(qt_image)

        if pixmap and not pixmap.isNull():
            self._empty = False
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self._photo.setPixmap(QtGui.QPixmap())

        # Release the image loading flag.
        self.image_loading_busy = False

    def wheelEvent(self, event):
        """
        Handle scroll events for zooming in and out of the scene. This is only active when a photo
        is loaded.

        :param event: wheel event object
        :return: -
        """

        if self.drag_mode:
            # Depending of wheel direction, set the direction value to greater or smaller than 1.
            self.zoom(event.angleDelta().y())

        # If not in drag mode, the wheel event is handled at the scene level.
        else:
            self._scene.wheelEvent(event)

    def zoom(self, direction):
        """
        Zoom in or out. This is only active when a photo is loaded

        :param direction: If > 0, zoom in, otherwise zoom out.
        :return: -
        """

        if self.hasPhoto():

            # Depending of direction value, set the zoom factor to greater or smaller than 1.
            if direction > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1

            # Apply the zoom factor to the scene. If the zoom counter is zero, fit the scene
            # to the window size.
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def keyPressEvent(self, event):
        """
        The + and - keys are used for zooming.

        :param event: event object
        :return: -
        """

        # If the "+" key is pressed, zoom in. If "-" is pressed, zoom out.
        if event.key() == QtCore.Qt.Key_Plus and not event.modifiers() & QtCore.Qt.ControlModifier:
            self.zoom(1)
        elif event.key() == QtCore.Qt.Key_Minus and not event.modifiers() & QtCore.Qt.ControlModifier:
            self.zoom(-1)
        else:
            super(FrameViewer, self).keyPressEvent(event)


class FrameViewerWidget(QtWidgets.QFrame, Ui_frame_viewer):
    """
    This widget implements a frame viewer together with control elements to visualize frame
    qualities, and to manipulate the stack limits.
    """

    def __init__(self, parent_gui, configuration, frames, rank_frames, align_frames,
                 stacked_image_log_file, signal_finished, signal_payload):
        """
        Initialization of the widget.

        :param parent_gui: Parent GUI object
        :param configuration: Configuration object with parameters
        :param frames: Frames object with all video frames
        :param rank_frames: RankFrames object with global quality ranks (between 0. and 1.,
                            1. being optimal) for all frames
        :param align_frames: AlignFrames object with global shift information for all frames
        :param stacked_image_log_file: Log file to be stored with results, or None.
        :param signal_finished: Qt signal with signature (str) to trigger the next activity when
                                the viewer exits.
        :param signal_payload: Payload of "signal_finished" (str).
        """

        super(FrameViewerWidget, self).__init__(parent_gui)
        self.setupUi(self)

        # Keep references to upper level objects.
        self.parent_gui = parent_gui
        self.configuration = configuration
        self.stacked_image_log_file = stacked_image_log_file
        self.signal_finished = signal_finished
        self.signal_payload = signal_payload
        self.frames = frames
        self.rank_frames = rank_frames
        self.align_frames = align_frames

        # Set up the frame viewer and put it in the upper left corner.
        self.frame_viewer = FrameViewer(self.frames, self.align_frames)
        self.frame_viewer.setObjectName("framewiever")
        self.grid_layout.addWidget(self.frame_viewer, 0, 0, 4, 3)

        # Initialize variables. The values for "alignment_point_frame_number" and
        # "alignment_points_frame_percent" are held as copies in this object. Only if the user
        # presses "OK" at the end, the values are copied back into the configuration object.
        self.alignment_points_frame_number = self.configuration.alignment_points_frame_number
        self.alignment_points_frame_percent = self.configuration.alignment_points_frame_percent
        if self.alignment_points_frame_number is None or 0 < self.alignment_points_frame_number \
                <= self.frames.number:
            self.alignment_points_frame_number = max(1, int(
                round(self.frames.number * self.alignment_points_frame_percent / 100.)))
        self.frame_ranks = rank_frames.frame_ranks
        self.quality_sorted_indices = rank_frames.quality_sorted_indices

        # Be careful: Indices are counted from 0, while widget contents are counted from 1 (to make
        # it easier for the user.
        self.quality_index = 0
        self.frame_index = self.rank_frames.quality_sorted_indices[self.quality_index] + 1

        # Start with ordering frames by quality. This can be changed by the user using a radio
        # button.
        self.frame_ordering = "quality"

        # Initialize a variable for communication with the frame_player object later.
        self.run_player = False

        # Create the frame player thread and start it. The player displays frames in succession.
        # It is pushed on a different thread because otherwise the user could not stop it before it
        # finishes.
        self.player_thread = QtCore.QThread()
        self.frame_player = FramePlayer(self)
        self.frame_player.moveToThread(self.player_thread)
        self.frame_player.set_photo_signal.connect(self.frame_viewer.setPhoto)
        self.player_thread.start()

        # Initialization of GUI elements
        self.slider_frames.setMinimum(1)
        self.slider_frames.setMaximum(self.frames.number)
        self.slider_frames.setValue(self.quality_index + 1)
        self.spinBox_chronological.setValue(self.quality_index)
        self.spinBox_quality.setValue(self.quality_index + 1)
        self.spinBox_chronological.setMinimum(1)
        self.spinBox_chronological.setMaximum(self.frames.number)
        self.spinBox_quality.setMinimum(1)
        self.spinBox_quality.setMaximum(self.frames.number)
        self.radioButton_quality.setChecked(True)

        self.spinBox_number_frames.setMaximum(self.frames.number)
        self.spinBox_percentage_frames.setValue(self.alignment_points_frame_percent)

        self.spinBox_number_frames.setValue(self.alignment_points_frame_number)

        # Create the Matplotlib widget showing the quality lines.
        self.matplotlib_widget = MatplotlibWidget(self.configuration, self.rank_frames)

        self.grid_layout.addWidget(Canvas(self.matplotlib_widget.fig), 0, 3, 2, 1)

        self.grid_layout.setColumnStretch(0, 5)
        self.grid_layout.setColumnStretch(1, 0)
        self.grid_layout.setColumnStretch(2, 0)
        self.grid_layout.setColumnStretch(3, 1)
        self.grid_layout.setRowStretch(0, 1)
        self.grid_layout.setRowStretch(1, 0)
        self.grid_layout.setRowStretch(2, 0)
        self.grid_layout.setRowStretch(3, 0)

        # Connect signals with slots.
        self.buttonBox.accepted.connect(self.done)
        self.buttonBox.rejected.connect(self.reject)
        self.slider_frames.valueChanged.connect(self.slider_frames_changed)
        self.spinBox_number_frames.valueChanged.connect(self.spinbox_number_frames_changed)
        self.spinBox_percentage_frames.valueChanged.connect(self.spinbox_percentage_frames_changed)
        self.radioButton_quality.toggled.connect(self.radiobutton_quality_changed)
        self.spinBox_chronological.valueChanged.connect(self.spinbox_chronological_changed)
        self.spinBox_quality.valueChanged.connect(self.spinbox_quality_changed)
        self.pushButton_set_stacking_limit.clicked.connect(
            self.pushbutton_set_stacking_limit_clicked)
        self.pushButton_play.clicked.connect(self.frame_player.play)
        self.pushButton_stop.clicked.connect(self.pushbutton_stop_clicked)

        # Initialize the Matplotlib widget contents.
        self.matplotlib_widget.renew_plot(self.quality_index, self.frame_ordering,
                                          self.alignment_points_frame_number)

    def slider_frames_changed(self):
        """
        The frames slider is changed by the user.

        :return: -
        """

        # Again, please note the difference between indexing and GUI displays.
        index = self.slider_frames.value() - 1

        # Differentiate between frame ordering (by quality or chronologically).
        if self.frame_ordering == "quality":
            self.frame_index = self.rank_frames.quality_sorted_indices[index]
            self.quality_index = index

            # Plot a dot on the quality line at the position of the current frame.
            self.matplotlib_widget.plot_dot(self.quality_index)
        else:
            self.frame_index = index
            self.quality_index = self.rank_frames.quality_sorted_indices.index(self.frame_index)
            self.matplotlib_widget.plot_dot(self.frame_index)

        # Block signals temporarily to avoid feedback loops. Then update widget contents.
        self.spinBox_chronological.blockSignals(True)
        self.spinBox_quality.blockSignals(True)
        self.spinBox_chronological.setValue(self.frame_index + 1)
        self.spinBox_quality.setValue(self.quality_index + 1)
        self.spinBox_chronological.blockSignals(False)
        self.spinBox_quality.blockSignals(False)
        self.frame_viewer.setPhoto(self.frame_index)

    def spinbox_number_frames_changed(self):
        """
        The user has changed the number of frames to be stacked at each AP.

        :return: -
        """

        self.alignment_points_frame_number = self.spinBox_number_frames.value()
        self.alignment_points_frame_percent = int(
            round(self.alignment_points_frame_number * 100. / self.frames.number))
        self.spinBox_percentage_frames.blockSignals(True)
        self.spinBox_percentage_frames.setValue(self.alignment_points_frame_percent)
        self.spinBox_percentage_frames.blockSignals(False)
        self.matplotlib_widget.plot_cutoff_lines(self.frame_ordering,
                                                 self.alignment_points_frame_number)

    def spinbox_percentage_frames_changed(self):
        """
        The user has changed the percentage of frames to be stacked at each AP.

        :return:
        """

        self.alignment_points_frame_percent = self.spinBox_percentage_frames.value()
        self.alignment_points_frame_number = int(
            round(self.frames.number * self.alignment_points_frame_percent / 100.))
        self.spinBox_number_frames.blockSignals(True)
        self.spinBox_number_frames.setValue(self.alignment_points_frame_number)
        self.spinBox_number_frames.blockSignals(False)
        self.matplotlib_widget.plot_cutoff_lines(self.frame_ordering,
                                                 self.alignment_points_frame_number)

    def radiobutton_quality_changed(self):
        """
        Toggle back and forth between frame ordering modes. The frame slider is updated to reflect
        the index of the current frame in the new ordering scheme. The Matplotlib widget changes
        its appearance.

        :return: -
        """

        if self.frame_ordering == "quality":
            self.frame_ordering = "chronological"
            self.slider_frames.setValue(self.frame_index + 1)
            self.matplotlib_widget.renew_plot(self.frame_index, self.frame_ordering,
                                              self.alignment_points_frame_number)
        else:
            self.frame_ordering = "quality"
            self.slider_frames.setValue(self.quality_index + 1)
            self.matplotlib_widget.renew_plot(self.quality_index, self.frame_ordering,
                                              self.alignment_points_frame_number)

    def spinbox_chronological_changed(self):
        """
        The user has selected a new frame to be displayed by entering its chronological frame index.
        All other widgets which depend on the current frame index are changed.

        :return: -
        """

        self.frame_index = self.spinBox_chronological.value() - 1
        self.quality_index = self.rank_frames.quality_sorted_indices.index(self.frame_index)
        self.slider_frames.blockSignals(True)
        self.spinBox_quality.blockSignals(True)
        self.spinBox_quality.setValue(self.quality_index + 1)
        if self.frame_ordering == "quality":
            self.slider_frames.setValue(self.quality_index + 1)
            self.matplotlib_widget.plot_dot(self.quality_index)
        else:
            self.slider_frames.setValue(self.frame_index + 1)
            self.matplotlib_widget.plot_dot(self.frame_index)
        self.slider_frames.blockSignals(False)
        self.spinBox_quality.blockSignals(False)
        self.frame_viewer.setPhoto(self.frame_index)

    def spinbox_quality_changed(self):
        """
        The user has selected a new frame to be displayed by entering its quality index.
        All other widgets which depend on the current frame index are changed.

        :return:
        """

        self.quality_index = self.spinBox_quality.value() - 1
        self.frame_index = self.rank_frames.quality_sorted_indices[self.quality_index]
        self.slider_frames.blockSignals(True)
        self.spinBox_chronological.blockSignals(True)
        self.spinBox_chronological.setValue(self.frame_index + 1)
        if self.frame_ordering == "quality":
            self.slider_frames.setValue(self.quality_index + 1)
            self.matplotlib_widget.plot_dot(self.quality_index)
        else:
            self.slider_frames.setValue(self.frame_index + 1)
            self.matplotlib_widget.plot_dot(self.frame_index)
        self.slider_frames.blockSignals(False)
        self.spinBox_chronological.blockSignals(False)
        self.frame_viewer.setPhoto(self.frame_index)

    def pushbutton_set_stacking_limit_clicked(self):
        """
        The current frame defines the stacking limit. Compute the corresponding stack limit
        variables and update the widgets. Since the signals are not deactivated before updating
        the widgets, the Matplotlib widget gets updated as well.
        :return:
        """
        self.alignment_points_frame_number = self.quality_index + 1
        self.alignment_points_frame_percent = int(
            round(self.alignment_points_frame_number * 100. / self.frames.number))
        self.spinBox_number_frames.setValue(self.alignment_points_frame_number)
        self.spinBox_percentage_frames.setValue(self.alignment_points_frame_percent)

    def pushbutton_stop_clicked(self):
        """
        When the frame player is running, it periodically checks this variable. If it is set to
        False, the player stops.

        :return:
        """

        self.frame_player.run_player = False

    def done(self):
        """
        On exit from the frame viewer, update the stack frame size and send a completion signal.

        :return: -
        """

        # Check if a new stack size was selected.
        if (self.configuration.alignment_points_frame_number !=
                self.spinBox_number_frames.value() and
                self.configuration.alignment_points_frame_number is not None) or \
                self.configuration.alignment_points_frame_percent != \
                self.spinBox_percentage_frames.value():
            # Save the (potentially changed) stack size.
            self.configuration.alignment_points_frame_percent = \
                self.spinBox_percentage_frames.value()
            self.configuration.alignment_points_frame_number = self.spinBox_number_frames.value()

            # Write the stack size change into the protocol.
            if self.configuration.global_parameters_protocol_level > 1:
                Miscellaneous.protocol("           The user has selected a new stack size: " +
                    str(self.configuration.alignment_points_frame_number) + " frames (" +
                    str(self.configuration.alignment_points_frame_percent) + "% of all frames).",
                    self.stacked_image_log_file, precede_with_timestamp=False)

        # Send a completion message.
        if self.parent_gui is not None:
            self.signal_finished.emit(self.signal_payload)

        # Close the Window.
        plt.close()
        self.close()

    def reject(self):
        """
        If the "cancel" button is pressed, the coordinates are not stored.

        :return: -
        """

        # Send a completion message.
        if self.parent_gui is not None:
            self.signal_finished.emit(self.signal_payload)

        # Close the Window.
        plt.close()
        self.close()


class FramePlayer(QtCore.QObject):
    """
    This class implements a video player using the FrameViewer and the control elements of the
    FrameViewerWidget. The player is started by the widget on a separate thread. This way the user
    can instruct the GUI to stop the running player.

    """
    set_photo_signal = QtCore.pyqtSignal(int)

    def __init__(self, frame_viewer_widget):
        super(FramePlayer, self).__init__()

        # Store a reference of the frame viewer widget and create a list of GUI elements. This makes
        # it easier to perform the same operation on all elements.
        self.frame_viewer_widget = frame_viewer_widget
        self.frame_viewer_widget_elements = [self.frame_viewer_widget.spinBox_chronological,
                                             self.frame_viewer_widget.spinBox_quality,
                                             self.frame_viewer_widget.slider_frames,
                                             self.frame_viewer_widget.pushButton_play]

        # Initialize a variable used to stop the player in the GUI thread.
        self.run_player = False

    def play(self):
        """
        Start the player.
        :return: -
        """

        # Block signals from GUI elements to avoid cross-talk, and disable them to prevent unwanted
        # user interaction.
        for element in self.frame_viewer_widget_elements:
            element.blockSignals(True)
            element.setDisabled(True)

        # Set the plaer running.
        self.run_player = True

        # The frames are ordered by their quality.
        if self.frame_viewer_widget.frame_ordering == "quality":

            # The player stops when the end of the video is reached, or when the "run_player"
            # variable is set to False in the GUI thread.
            while self.frame_viewer_widget.quality_index < self.frame_viewer_widget.frames.number \
                    - 1 and self.run_player:
                # Go to the next frame only if the viewer has finished loading the previous one.
                if not self.frame_viewer_widget.frame_viewer.image_loading_busy:
                    self.frame_viewer_widget.quality_index += 1
                    self.frame_viewer_widget.frame_index = \
                    self.frame_viewer_widget.rank_frames.quality_sorted_indices[
                        self.frame_viewer_widget.quality_index]
                    self.frame_viewer_widget.spinBox_chronological.setValue(
                        self.frame_viewer_widget.frame_index + 1)
                    self.frame_viewer_widget.spinBox_quality.setValue(
                        self.frame_viewer_widget.quality_index + 1)
                    self.frame_viewer_widget.slider_frames.setValue(
                        self.frame_viewer_widget.quality_index + 1)
                    self.set_photo_signal.emit(self.frame_viewer_widget.frame_index)
                    self.frame_viewer_widget.matplotlib_widget.plot_dot(
                        self.frame_viewer_widget.quality_index)

                # Insert a short pause to keep the video from running too fast.
                sleep(0.1)
                self.frame_viewer_widget.update()
        else:
            # The same for chronological frame ordering.
            while self.frame_viewer_widget.frame_index < self.frame_viewer_widget.frames.number \
                    - 1 and self.run_player:
                if not self.frame_viewer_widget.frame_viewer.image_loading_busy:
                    self.frame_viewer_widget.frame_index += 1
                    self.frame_viewer_widget.quality_index = \
                        self.frame_viewer_widget.rank_frames.quality_sorted_indices.index(
                        self.frame_viewer_widget.frame_index)
                    self.frame_viewer_widget.spinBox_chronological.setValue(
                        self.frame_viewer_widget.frame_index + 1)
                    self.frame_viewer_widget.spinBox_quality.setValue(
                        self.frame_viewer_widget.quality_index + 1)
                    self.frame_viewer_widget.slider_frames.setValue(
                        self.frame_viewer_widget.frame_index + 1)
                    self.set_photo_signal.emit(self.frame_viewer_widget.frame_index)
                    self.frame_viewer_widget.matplotlib_widget.plot_dot(
                        self.frame_viewer_widget.frame_index)
                sleep(0.1)
                self.frame_viewer_widget.update()

        self.run_player = False

        # Re-set the GUI elements to their normal state.
        for element in self.frame_viewer_widget_elements:
            element.blockSignals(False)
            element.setDisabled(False)


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
    try:
        frames = Frames(configuration, names, type=type)
        print("Number of images read: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Exception as e:
        print("Error: " + e.message)
        exit()

    # Rank the frames by their overall local contrast.
    rank_frames = RankFrames(frames, configuration)
    start = time()
    rank_frames.frame_score()
    end = time()
    print('Elapsed time in ranking images: {}'.format(end - start))
    print("Index of maximum: " + str(rank_frames.frame_ranks_max_index))
    print("Frame scores: " + str(rank_frames.frame_ranks))
    print("Frame scores (sorted): " + str(
        [rank_frames.frame_ranks[i] for i in rank_frames.quality_sorted_indices]))
    print("Sorted index list: " + str(rank_frames.quality_sorted_indices))

    # Initialize the frame alignment object.
    align_frames = AlignFrames(frames, rank_frames, configuration)

    if configuration.align_frames_mode == "Surface":
        # Select the local rectangular patch in the image where the L gradient is highest in both x
        # and y direction. The scale factor specifies how much smaller the patch is compared to the
        # whole image frame.
        (y_low_opt, y_high_opt, x_low_opt, x_high_opt) = align_frames.compute_alignment_rect(
            configuration.align_frames_rectangle_scale_factor)

        print("optimal alignment rectangle, y_low: " + str(y_low_opt) + ", y_high: " +
              str(y_high_opt) + ", x_low: " + str(x_low_opt) + ", x_high: " + str(x_high_opt))

    # Align all frames globally relative to the frame with the highest score.
    try:
        align_frames.align_frames()
    except NotSupportedError as e:
        print("Error: " + e.message)
        exit()
    except InternalError as e:
        print("Warning: " + e.message)

    print("Intersection, y_low: " + str(align_frames.intersection_shape[0][0]) + ", y_high: "
          + str(align_frames.intersection_shape[0][1]) + ", x_low: " \
          + str(align_frames.intersection_shape[1][0]) + ", x_high: " \
          + str(align_frames.intersection_shape[1][1]))

    app = QtWidgets.QApplication(argv)
    window = FrameViewerWidget(None, configuration, frames, rank_frames, align_frames, None, None,
                               None)
    window.setMinimumSize(800, 600)
    window.showMaximized()
    app.exec_()

    print("Percentage of frames to be stacked: " + str(
        configuration.alignment_points_frame_percent) + ", number of frames: " + str(
        configuration.alignment_points_frame_number))

    exit()
