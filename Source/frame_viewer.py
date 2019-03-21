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

from PyQt5 import QtCore, QtGui, QtWidgets

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

import glob
import sys
from time import time

import numpy as np

from configuration import Configuration
from frames import Frames
from rank_frames import RankFrames
from frame_viewer_gui import Ui_frame_viewer

class MatplotlibWidget(Canvas):
    """
    This widget creates a plot of frame qualities, either sorted chronologically or by quality.

    """

    def __init__(self, configuration, rank_frames, frame_ordering, frame_index, parent=None):
        super(MatplotlibWidget, self).__init__(Figure())

        self.setParent(parent)
        self.configuration = configuration
        self.rank_frames = rank_frames
        self.line_chronological = None
        self.line_quality = None
        self.dot = None
        self.line_quality_cutoff = None

        self.quality_cutoff = self.rank_frames.frame_ranks[self.rank_frames.quality_sorted_indices[
            self.configuration.alignment_points_frame_number]]

        plt.rcParams.update({'font.size': 8})

        self.fig, self.ax = plt.subplots()
        self.ax.invert_xaxis()
        plt.subplots_adjust(left=0.23, right=0.95, top=0.98, bottom=0.12)

    def renew_plot(self, index, frame_ordering):
        self.ax.clear()
        self.ax.invert_xaxis()
        self.line_chronological = None
        self.line_quality = None
        self.dot = None
        self.line_quality_cutoff = None
        self.plot_data(frame_ordering)
        self.plot_cutoff_lines(frame_ordering)
        self.plot_dot(index)

    def plot_data(self, frame_ordering):
        """
        Plot the data, including a dot for the current frame.

        :return: -
        """

        self.ax.set_ylim(0, self.rank_frames.number+1)
        self.y = np.array(range(1, self.rank_frames.number + 1))
        if frame_ordering == "chronological":
            if self.line_chronological is not None:
                self.line_chronological.remove()
            self.x = np.array(self.rank_frames.frame_ranks)
            plt.ylabel('Frame numbers ordered chronologically')
            plt.gca().invert_yaxis()
            plt.xlabel('Quality')
            self.line_chronological, = plt.plot(self.x, self.y, lw=1)
            plt.grid(True)
        else:
            if self.line_quality is not None:
                self.line_quality.remove()
            self.x = np.array([self.rank_frames.frame_ranks[i]
                          for i in self.rank_frames.quality_sorted_indices])
            plt.ylabel('Frame numbers ordered by quality')
            plt.gca().invert_yaxis()
            plt.xlabel('Quality')
            self.line_quality, = plt.plot(self.x, self.y, lw=1)
            plt.grid(True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_cutoff_lines(self, frame_ordering):
        if frame_ordering == "chronological":
            if self.line_quality_cutoff is not None:
                self.line_quality_cutoff.remove()
            quality_cutoff = self.rank_frames.frame_ranks[
                self.rank_frames.quality_sorted_indices[
                    self.configuration.alignment_points_frame_number]]
            x_cutoff = np.full((self.rank_frames.number,), quality_cutoff)
            self.line_quality_cutoff, = plt.plot(x_cutoff, self.y, lw=1)
        else:
            if self.line_quality_cutoff is not None:
                self.line_quality_cutoff.remove()
            y_cutoff = np.full((self.rank_frames.number,), self.configuration.alignment_points_frame_number)
            self.line_quality_cutoff, = plt.plot(self.x, y_cutoff, lw=1)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_dot(self, frame_index):
        if self.dot is not None:
            self.dot.remove()
        self.dot = plt.scatter(self.x[frame_index], self.y[frame_index], s=20)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class FrameViewer(QtWidgets.QGraphicsView):
    """
    This widget implements a frame viewer. Panning and zooming is implemented by using the mouse
    and scroll wheel.

    """

    resized = QtCore.pyqtSignal()

    def __init__(self, frames):
        super(FrameViewer, self).__init__()
        self._zoom = 0
        self._empty = True

        self.frames = frames
        self.shape_y = self.frames[0].shape[0]
        self.shape_x = self.frames[0].shape[1]
        self.frame_index = 0

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

        self.image = self.frames[index]
        # Convert the float32 monochrome image into uint8 format.
        image_uint8 = self.image.astype(np.uint8)
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


class FrameViewerWidget(QtWidgets.QFrame, Ui_frame_viewer):
    """
    This widget implements a rectangular patch editor, to be used for the selection of the
    stabilization patch and the ROI rectangle.
    """

    def __init__(self, parent_gui, configuration, rank_frames, signal_finished):
        """
        Initialization of the widget.

        :param parent_gui: Parent GUI object
        :param frame: Background image on which the patch is superimposed. Usually, the mean frame
                      is used for this purpose.
        :param message: Message to tell the user what to do.
        :param signal_finished: Qt signal with signature (int, int, int, int) sending the
                                coordinate bounds (y_low, y_high, x_low, x_high) of the patch
                                selected, or (0, 0, 0, 0) if unsuccessful.
        """

        super(FrameViewerWidget, self).__init__(parent_gui)
        self.setupUi(self)

        self.parent_gui = parent_gui
        self.configuration = configuration
        self.signal_finished = signal_finished
        self.frames = rank_frames.frames_mono
        self.rank_frames = rank_frames

        self.frame_viewer = FrameViewer(self.frames)
        self.frame_viewer.setObjectName("framewiever")
        self.grid_layout.addWidget(self.frame_viewer, 0, 0, 4, 3)

        self.number_frames = len(self.frames)
        self.frame_percent = self.configuration.alignment_points_frame_percent
        self.frame_ranks = rank_frames.frame_ranks
        self.quality_sorted_indices = rank_frames.quality_sorted_indices
        self.quality_index = 0
        self.frame_index = self.rank_frames.quality_sorted_indices[self.quality_index]+1
        self.frame_ordering = "quality"

        # Initialization of GUI elements
        self.slider_frames.setMinimum(1)
        self.slider_frames.setMaximum(self.number_frames)
        self.slider_frames.setValue(self.quality_index+1)
        self.spinBox_chronological.setValue(self.quality_index)
        self.spinBox_quality.setValue(self.quality_index+1)
        self.spinBox_chronological.setMinimum(1)
        self.spinBox_chronological.setMaximum(self.number_frames)
        self.spinBox_quality.setMinimum(1)
        self.spinBox_quality.setMaximum(self.number_frames)
        self.radioButton_quality.setChecked(True)

        self.spinBox_percentage_frames.setValue(
            self.configuration.alignment_points_frame_percent)
        if self.configuration.alignment_points_frame_number is None or 0 < \
                self.configuration.alignment_points_frame_number <= self.number_frames:
            self.configuration.alignment_points_frame_number = max(1, int(
                round(self.number_frames * self.frame_percent / 100.)))
        self.spinBox_number_frames.setValue(self.configuration.alignment_points_frame_number)

        self.matplotlib_widget = MatplotlibWidget(self.configuration, self.rank_frames,
                                                  self.frame_ordering, self.quality_index)

        self.grid_layout.addWidget(Canvas(self.matplotlib_widget.fig), 0, 3, 2, 1)

        self.grid_layout.setColumnStretch(0, 5)
        self.grid_layout.setColumnStretch(1, 0)
        self.grid_layout.setColumnStretch(2, 0)
        self.grid_layout.setColumnStretch(3, 1)
        self.grid_layout.setRowStretch(0, 1)
        self.grid_layout.setRowStretch(1, 0)
        self.grid_layout.setRowStretch(2, 0)
        self.grid_layout.setRowStretch(3, 0)

        self.buttonBox.accepted.connect(self.done)
        self.buttonBox.rejected.connect(self.reject)
        self.slider_frames.valueChanged.connect(self.slider_frames_changed)
        self.spinBox_number_frames.valueChanged.connect(self.spinbox_number_frames_changed)
        self.spinBox_percentage_frames.valueChanged.connect(self.spinbox_percentage_frames_changed)
        self.radioButton_quality.toggled.connect(self.radiobutton_quality_changed)
        self.spinBox_chronological.valueChanged.connect(self.spinbox_chronological_changed)
        self.spinBox_quality.valueChanged.connect(self.spinbox_quality_changed)
        self.pushButton_set_stacking_limit.clicked.connect(self.pushbutton_set_stacking_limit_clicked)
        self.pushButton_play.clicked.connect(self.pushbutton_play_clicked)
        self.pushButton_stop.clicked.connect(self.pushbutton_stop_clicked)

        self.matplotlib_widget.renew_plot(self.quality_index, self.frame_ordering)

    def slider_frames_changed(self):
        index = self.slider_frames.value()-1
        if self.frame_ordering == "quality":
            self.frame_index = self.rank_frames.quality_sorted_indices[index]
            self.quality_index = index
            self.matplotlib_widget.plot_dot(self.quality_index)
        else:
            self.frame_index = index
            self.quality_index = self.rank_frames.quality_sorted_indices.index(self.frame_index)
            self.matplotlib_widget.plot_dot(self.frame_index)
        self.spinBox_chronological.blockSignals(True)
        self.spinBox_quality.blockSignals(True)
        self.spinBox_chronological.setValue(self.frame_index+1)
        self.spinBox_quality.setValue(self.quality_index + 1)
        self.spinBox_chronological.blockSignals(False)
        self.spinBox_quality.blockSignals(False)
        self.frame_viewer.setPhoto(self.frame_index)

    def spinbox_number_frames_changed(self):
        pass

    def spinbox_percentage_frames_changed(self):
        pass

    def radiobutton_quality_changed(self):
        if self.frame_ordering == "quality":
            self.frame_ordering = "chronological"
            self.slider_frames.setValue(self.frame_index+1)
            self.matplotlib_widget.renew_plot(self.frame_index, self.frame_ordering)
        else:
            self.frame_ordering = "quality"
            self.slider_frames.setValue(self.quality_index+1)
            self.matplotlib_widget.renew_plot(self.quality_index, self.frame_ordering)

    def spinbox_chronological_changed(self):
        self.frame_index = self.spinBox_chronological.value()-1
        self.quality_index = self.rank_frames.quality_sorted_indices.index(self.frame_index)
        self.slider_frames.blockSignals(True)
        self.spinBox_quality.blockSignals(True)
        self.spinBox_quality.setValue(self.quality_index+1)
        if self.frame_ordering == "quality":
            self.slider_frames.setValue(self.quality_index+1)
            self.matplotlib_widget.plot_dot(self.quality_index)
        else:
            self.slider_frames.setValue(self.frame_index+1)
            self.matplotlib_widget.plot_dot(self.frame_index)
        self.slider_frames.blockSignals(False)
        self.spinBox_quality.blockSignals(False)
        self.frame_viewer.setPhoto(self.frame_index)

    def spinbox_quality_changed(self):
        self.quality_index = self.spinBox_quality.value()-1
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
        pass

    def pushbutton_stop_clicked(self):
        pass

    def pushbutton_play_clicked(self):
        self.spinBox_chronological.blockSignals(True)
        self.spinBox_quality.blockSignals(True)
        self.slider_frames.blockSignals(True)
        self.spinBox_chronological.setDisabled(True)
        self.spinBox_quality.setDisabled(True)
        self.slider_frames.setDisabled(True)
        if self.frame_ordering == "quality":
            while self.quality_index < self.number_frames-1:
                self.quality_index += 1
                self.frame_index = self.rank_frames.quality_sorted_indices[self.quality_index]
                self.spinBox_chronological.setValue(self.frame_index+1)
                self.spinBox_quality.setValue(self.quality_index+1)
                self.slider_frames.setValue(self.quality_index+1)
                self.frame_viewer.setPhoto(self.frame_index)
        else:
            while self.frame_index < self.number_frames-1:
                self.frame_index += 1
                self.quality_index = self.rank_frames.quality_sorted_indices.index(self.frame_index)
                self.spinBox_chronological.setValue(self.frame_index+1)
                self.spinBox_quality.setValue(self.quality_index+1)
                self.slider_frames.setValue(self.quality_index+1)
                self.frame_viewer.setPhoto(self.frame_index)
        self.spinBox_chronological.blockSignals(False)
        self.spinBox_quality.blockSignals(False)
        self.slider_frames.blockSignals(False)
        self.spinBox_chronological.setDisabled(False)
        self.spinBox_quality.setDisabled(False)
        self.slider_frames.setDisabled(False)

    def done(self):
        """
        On exit from the frame viewer, update the stack frame size and send a completion signal.

        :return: -
        """

        # Save the (potentially changed) stack size.
        self.configuration.alignment_points_frame_percent = self.spinBox_percentage_frames.value()
        self.configuration.alignment_points_frame_number = self.spinBox_number_frames.value()

        # Send a completion message.
        if self.parent_gui is not None:
            self.signal_finished.emit()

        # Close the Window.
        self.close()

    def reject(self):
        """
        If the "cancel" button is pressed, the coordinates are not stored.

        :return: -
        """

        # Send a completion message.
        self.signal_finished.emit()
        self.close()


if __name__ == '__main__':
    # Images can either be extracted from a video file or a batch of single photographs. Select
    # the example for the test run.
    type = 'video'
    if type == 'image':
        names = glob.glob('Images/2012*.tif')
        # names = glob.glob('Images/Moon_Tile-031*ap85_8b.tif')
        # names = glob.glob('Images/Example-3*.jpg')
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

    # The whole quality analysis and shift determination process is performed on a monochrome
    # version of the frames. If the original frames are in RGB, the monochrome channel can be
    # selected via a configuration parameter. Add a list of monochrome images for all frames to
    # the "Frames" object.
    start = time()
    frames.add_monochrome(configuration.frames_mono_channel)
    end = time()
    print('Elapsed time in creating blurred monochrome images: {}'.format(end - start))

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

    app = QtWidgets.QApplication(sys.argv)
    window = FrameViewerWidget(None, configuration, rank_frames, None)
    window.setMinimumSize(800, 600)
    window.showMaximized()
    app.exec_()

    print ("Percentage of frames to be stacked: " +
           str(configuration.alignment_points_frame_percent) + ", number of frames: " +
           str(configuration.alignment_points_frame_number))

    sys.exit()

