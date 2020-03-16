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

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from cv2 import NORM_MINMAX, normalize
from numpy import uint8, uint16

from align_frames import AlignFrames
from configuration import Configuration
from exceptions import NotSupportedError, InternalError, Error
from frame_viewer_gui import Ui_frame_viewer
from frames import Frames
from frame_selector_gui import Ui_frame_selector
from miscellaneous import Miscellaneous
from rank_frames import RankFrames


class FrameViewer(QtWidgets.QGraphicsView):
    """
    This widget implements a frame viewer. Panning and zooming is implemented by using the mouse
    and scroll wheel.

    """

    resized = QtCore.pyqtSignal()

    def __init__(self):
        super(FrameViewer, self).__init__()
        self._zoom = 0
        self._empty = True

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

    def setPhoto(self, image):
        """
        Convert a color or grayscale image to a pixmap and assign it to the photo object.

        :param image: Image to be displayed. The image is assumed to be in color or grayscale
                      format of length uint8 or uint16.
        :return: -
        """

        # Indicate that an image is being loaded.
        self.image_loading_busy = True

        # Convert the image into uint8 format. If the frame type is uint16, values correspond to
        # 16bit resolution.
        if image.dtype == uint16:
            image_uint8 = (image / 256.).astype(uint8)
        elif image.dtype == uint8:
            image_uint8 = image.astype(uint8)
        else:
            raise NotSupportedError("Attempt to set a photo in frame viewer with type neither"
                                    " uint8 nor uint16")

        self.shape_y = image_uint8.shape[0]
        self.shape_x = image_uint8.shape[1]

        # Normalize the frame brightness.
        image_uint8 = normalize(image_uint8, None, alpha=0, beta=255, norm_type=NORM_MINMAX)

        # The image is monochrome:
        if len(image_uint8.shape) == 2:
            qt_image = QtGui.QImage(image_uint8, self.shape_x, self.shape_y, self.shape_x,
                                    QtGui.QImage.Format_Grayscale8)
        # The image is RGB color.
        else:
            qt_image = QtGui.QImage(image_uint8, self.shape_x,
                                    self.shape_y, 3*self.shape_x, QtGui.QImage.Format_RGB888)
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

class VideoFrameSelector(FrameViewer):
    """
    This widget implements a frame viewer for frames in a video file. Panning and zooming is
    implemented by using the mouse and scroll wheel.

    """

    resized = QtCore.pyqtSignal()

    def __init__(self, frames, frame_index=0):
        super(VideoFrameSelector, self).__init__()
        self.frames = frames
        self.frame_index = frame_index

        self.setPhoto(self.frame_index)

    def setPhoto(self, index):
        """
        Convert a grayscale image to a pixmap and assign it to the photo object.

        :param index: Index into the frame list. Frames are assumed to be grayscale image in format
                      float32.
        :return: -
        """

        # Indicate that an image is being loaded.
        self.image_loading_busy = True

        image = self.frames.frames_mono(index)

        super(VideoFrameSelector, self).setPhoto(image)


class FrameSelectorWidget(QtWidgets.QFrame, Ui_frame_selector):
    """
    This widget implements a frame viewer together with control elements to visualize frame
    qualities, and to manipulate the stack limits.
    """

    def __init__(self, parent_gui, configuration, frames,
                 stacked_image_log_file, signal_finished, signal_payload):
        """
        Initialization of the widget.

        :param parent_gui: Parent GUI object
        :param configuration: Configuration object with parameters
        :param frames: Frames object with all video frames
        :param stacked_image_log_file: Log file to be stored with results, or None.
        :param signal_finished: Qt signal with signature (str) to trigger the next activity when
                                the viewer exits.
        :param signal_payload: Payload of "signal_finished" (str).
        """

        super(FrameSelectorWidget, self).__init__(parent_gui)
        self.setupUi(self)

        # Keep references to upper level objects.
        self.parent_gui = parent_gui
        self.configuration = configuration
        self.stacked_image_log_file = stacked_image_log_file
        self.signal_finished = signal_finished
        self.signal_payload = signal_payload
        self.frames = frames
        self.index_included = frames.index_included.copy()

        # Initialize the frame list selection.
        self.items_selected = None
        self.indices_selected = []

        # Set colors for the frame list.
        self.background_included = QtGui.QColor(130, 255, 130)
        self.foreground_included = QtGui.QColor(0, 0, 0)
        self.background_excluded = QtGui.QColor(120, 120, 120)
        self.foreground_excluded = QtGui.QColor(255, 255, 255)

        self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        # Initialize the inclusion / exclusion state of frames in the frame list.
        for i in range(frames.number_original):
            if self.index_included[i]:
                item = QtWidgets.QListWidgetItem("Frame %i included" % i)
                item.setBackground(self.background_included)
                item.setForeground(self.foreground_included)
            else:
                item = QtWidgets.QListWidgetItem("Frame %i excluded" % i)
                item.setBackground(self.background_excluded)
                item.setForeground(self.foreground_excluded)
            self.listWidget.addItem(item)

        self.listWidget.installEventFilter(self)
        self.listWidget.itemClicked.connect(self.select_items)
        self.addButton.clicked.connect(self.use_triggered)
        self.removeButton.clicked.connect(self.not_use_triggered)

        # Be careful: Indices are counted from 0, while widget contents are counted from 1 (to make
        # it easier for the user.
        self.frame_index = 0

        # Set up the frame viewer and put it in the upper left corner.
        self.frame_selector = VideoFrameSelector(self.frames, self.frame_index)
        self.frame_selector.setObjectName("framewiever")
        self.gridLayout.addWidget(self.frame_selector, 0, 0, 1, 3)

        # Initialize a variable for communication with the frame_player object later.
        self.run_player = False

        # Create the frame player thread and start it. The player displays frames in succession.
        # It is pushed on a different thread because otherwise the user could not stop it before it
        # finishes.
        self.player_thread = QtCore.QThread()
        self.frame_player = FramePlayer(self)
        self.frame_player.moveToThread(self.player_thread)
        self.frame_player.set_photo_signal.connect(self.frame_selector.setPhoto)
        self.player_thread.start()

        # Initialization of GUI elements
        self.slider_frames.setMinimum(1)
        self.slider_frames.setMaximum(self.frames.number)
        self.slider_frames.setValue(self.frame_index + 1)

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
        self.pushButton_play.clicked.connect(self.frame_player.play)
        self.pushButton_stop.clicked.connect(self.pushbutton_stop_clicked)

    def select_items(self):
        self.items_selected = self.listWidget.selectedItems()
        self.indices_selected = [self.listWidget.row(item) for item in self.items_selected]
        self.frame_index = self.indices_selected[0]

        # Set the slider to the current selection.
        self.slider_frames.blockSignals(True)
        self.slider_frames.setValue(self.frame_index + 1)
        self.slider_frames.blockSignals(False)

        # Update the image in the viewer.
        self.frame_selector.setPhoto(self.frame_index)
        print(self.indices_selected)

    def eventFilter(self, source, event):
        if source is self.listWidget:
            if event.type() == QtCore.QEvent.ContextMenu:
                print("Context menu opened")
                menu = QtWidgets.QMenu()
                action1 = QtWidgets.QAction('Use for stacking', menu)
                action1.triggered.connect(self.use_triggered)
                menu.addAction((action1))
                action2 = QtWidgets.QAction("Don't use for stacking", menu)
                action2.triggered.connect(self.not_use_triggered)
                menu.addAction((action2))
                menu.exec_(event.globalPos())
            elif event.type() == QtCore.QEvent.KeyPress:
                if event.key() == Qt.Key_Plus:
                    self.use_triggered()
                elif event.key() == Qt.Key_Minus:
                    self.not_use_triggered()
                elif event.key() == Qt.Key_Escape:
                    self.items_selected = []
                    self.indices_selected = []
        return super(FrameSelectorWidget, self).eventFilter(source, event)

    def use_triggered(self):
        if self.items_selected:
            for index, item in enumerate(self.items_selected):
                index_selcted = self.indices_selected[index]
                item.setText("Frame %i included" % index_selcted)
                item.setBackground(self.background_included)
                item.setForeground(QtGui.QColor(0, 0, 0))
                self.index_included[index_selcted] = True

    def not_use_triggered(self):
        if self.items_selected:
            for index, item in enumerate(self.items_selected):
                index_selcted = self.indices_selected[index]
                item.setText("Frame %i excluded" % index_selcted)
                item.setBackground(self.background_excluded)
                item.setForeground(QtGui.QColor(255, 255, 255))
                self.index_included[index_selcted] = False

    def slider_frames_changed(self):
        """
        The frames slider is changed by the user. Update the frame in the viewer.

        :return: -
        """

        # Again, please note the difference between indexing and GUI displays.
        self.frame_index = self.slider_frames.value() - 1

        # Adjust the frame list and select the current frame.

        self.listWidget.setCurrentRow(self.frame_index, QtCore.QItemSelectionModel.SelectCurrent)
        self.select_items()

        # Update the image in the viewer.
        self.frame_selector.setPhoto(self.frame_index)

    def pushbutton_stop_clicked(self):
        """
        When the frame player is running, it periodically checks this variable. If it is set to
        False, the player stops.

        :return:
        """

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
        if indices_included and self.configuration.global_parameters_protocol_level > 1:
            Miscellaneous.protocol("           The user has included the following frames into the stacking workflow: " + str(indices_included),
                self.stacked_image_log_file, precede_with_timestamp=False)
        if indices_excluded and self.configuration.global_parameters_protocol_level > 1:
            Miscellaneous.protocol("           The user has excluded the following frames from the stacking workflow: " + str(indices_excluded),
                self.stacked_image_log_file, precede_with_timestamp=False)

        # Send a completion message.
        if self.parent_gui is not None:
            self.signal_finished.emit(self.signal_payload)

        # Close the Window.
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
        self.frame_viewer_widget_elements = [self.frame_viewer_widget.listWidget,
                                             self.frame_viewer_widget.addButton,
                                             self.frame_viewer_widget.removeButton]

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

        # Set the player running.
        self.run_player = True

        while self.frame_viewer_widget.frame_index < self.frame_viewer_widget.frames.number_original \
                - 1 and self.run_player:
            if not self.frame_viewer_widget.frame_selector.image_loading_busy:
                self.frame_viewer_widget.frame_index += 1
                self.frame_viewer_widget.slider_frames.setValue(
                    self.frame_viewer_widget.frame_index + 1)
                self.set_photo_signal.emit(self.frame_viewer_widget.frame_index)
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
    configuration.initialize_configuration()
    try:
        frames = Frames(configuration, names, type=type)
        print("Number of images: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Error as e:
        print("Error: " + e.message)
        exit()

    app = QtWidgets.QApplication(argv)
    window = FrameSelectorWidget(None, configuration, frames, None, None, None)
    window.setMinimumSize(800, 600)
    # window.showMaximized()
    window.show()
    app.exec_()

    exit()
