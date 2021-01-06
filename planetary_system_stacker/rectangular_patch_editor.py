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
from glob import glob
from sys import argv, exit
from time import time

from PyQt5 import QtCore, QtWidgets
from numpy import uint8, uint16, int32

from align_frames import AlignFrames
from alignment_point_editor import SelectionRectangleGraphicsItem
from configuration import Configuration
from exceptions import InternalError, NotSupportedError, Error
from frame_viewer import FrameViewer
from frames import Frames
from rank_frames import RankFrames
from rectangular_patch_editor_gui import Ui_rectangular_patch_editor


class GraphicsScene(QtWidgets.QGraphicsScene):
    """
    Handle mouse events on the graphics scene depicting the image with a rectangular patch
    superimposed. The left mouse button can be used to create the patch.

    """

    def __init__(self, photo_editor, parent=None):
        """
        Initialize the scene object.

        :param photo_editor: the object in which the scene is defined
        :param parent: The parent class
        """

        QtWidgets.QGraphicsScene.__init__(self, parent)
        self.photo_editor = photo_editor
        self.left_button_pressed = False
        self.right_button_pressed = False

        self.remember_rp = None

    def mousePressEvent(self, event):
        """
        A mouse button is pressed.

        :param event: event object
        :return: -
        """

        # The following actions are not performed in drag-and-zoom mode. The switch between both
        # modes is handled in the higher-level object "photo_viewer".
        if not self.photo_editor.drag_mode and self.photo_editor.hasPhoto():

            # If a rectangular patch had been drawn before, remove it from the scene.
            if self.remember_rp:
                self.removeItem(self.remember_rp)
                self.remember_rp = None

            pos = event.lastScenePos()
            x = int(pos.x())
            y = int(pos.y())

            # The left button is pressed.
            if event.button() == QtCore.Qt.LeftButton:
                self.left_button_pressed = True

                # Remember the location.
                self.left_y_start = y
                self.left_x_start = x


    def mouseReleaseEvent(self, event):
        """
        A mouse button is released.

        :param event: event object
        :return: -
        """

        if not self.photo_editor.drag_mode and self.photo_editor.hasPhoto():

            # The left button is released.
            if event.button() == QtCore.Qt.LeftButton:
                self.left_button_pressed = False

                # If a rectangular patch has been defined, store its coordinates.
                if self.remember_rp:
                    self.photo_editor.set_selection_rectangle(self.remember_rp.y_low,
                                                              self.remember_rp.y_high,
                                                              self.remember_rp.x_low,
                                                              self.remember_rp.x_high)
                # If the rectangle is void, set all coordinates to zero.
                else:
                    self.photo_editor.set_selection_rectangle(0, 0, 0, 0)



    def mouseMoveEvent(self, event):
        """
        The mouse is moved while being pressed.

        :param event: event object
        :return: -
        """

        pos = event.lastScenePos()
        x = int(pos.x())
        y = int(pos.y())

        if not self.photo_editor.drag_mode and self.photo_editor.hasPhoto():

            # The mouse is moved with the left button pressed.
            if self.left_button_pressed:

                # Compute the new rectangular patch.
                new_sp = SelectionRectangleGraphicsItem(self.left_y_start, self.left_x_start, y, x)

                # If the rectangle was drawn for a previous location, replace it with the new one.
                if self.remember_rp is not None:
                    self.removeItem(self.remember_rp)
                self.addItem(new_sp)
                self.remember_rp = new_sp

            self.update()


class RectangularPatchEditor(FrameViewer):
    """
    This widget implements an editor for handling a rectangular patch superimposed onto an image.
    It supports two modes:
    - In "drag mode" the mouse can be used for panning, and the scroll wheel for zooming.
    - In "alignment point mode" the mouse is used to create/remove APs, to move them or to change
      their sizes.
    The "cntrl" key is used to switch between the two modes.
    """

    def __init__(self, image):
        super(RectangularPatchEditor, self).__init__()

        self.image = image

        # Initialize the rectangular patch.
        self.y_low = None
        self.y_high = None
        self.x_low = None
        self.x_high = None

        self.setPhoto(self.image)

    def initialize_scene(self):
        """
        Initialize the scene. This object handles mouse events if not in drag mode. In derived
        viewer classes this method is replaced with the instantiation of a custom version of the
        graphics scene.

        :return:
        """

        self._scene = GraphicsScene(self, self)
        self._scene.addItem(self._photo)
        self.setScene(self._scene)

    def set_selection_rectangle(self, y_low, y_high, x_low, x_high):
        """
        This method is invoked from the underlying scene object when the left mouse button is
        released, and thus a rectangular patch is defined.

        :param y_low: Lower y bound of patch
        :param y_high: Upper y bound of patch
        :param x_low: Lower x bound of patch
        :param x_high: Upper x bound of patch
        :return: -
        """

        self.y_low = max(0, y_low)
        self.y_high = min(self.shape_y, y_high)
        self.x_low = max(0, x_low)
        self.x_high = min(self.shape_x, x_high)

    def selection_rectangle(self):
        """
        Return the coordinate bounds of the selected rectangle.

        :return: Tuple with 4 ints: (lower y bound, upper y bound, lower x bound, upper x bound)
        """

        if self.y_low is not None:
            return (self.y_low, self.y_high, self.x_low, self.x_high)
        else:
            return (0, 0, 0, 0)


class RectangularPatchEditorWidget(QtWidgets.QFrame, Ui_rectangular_patch_editor):
    """
    This widget implements a rectangular patch editor, to be used for the selection of the
    stabilization patch and the ROI rectangle.
    """

    def __init__(self, parent_gui, frame, message, signal_finished):
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

        super(RectangularPatchEditorWidget, self).__init__(parent_gui)
        self.setupUi(self)

        self.parent_gui = parent_gui

        # Convert the frame into uint8 format. If the frame type is uint16, values
        # correspond to 16bit resolution.
        if frame.dtype == uint16 or frame.dtype == int32:
            self.frame = (frame / 256).astype(uint8)
        elif frame.dtype == uint8:
            self.frame = frame
        else:
            raise NotSupportedError("Attempt to set a photo in frame viewer with type neither"
                                    " uint8 nor uint16 not int32")

        self.message = message
        self.signal_finished = signal_finished

        self.viewer = RectangularPatchEditor(self.frame)
        self.verticalLayout.insertWidget(0, self.viewer)

        self.messageLabel.setText(self.message)
        self.messageLabel.setStyleSheet('color: red')
        self.buttonBox.accepted.connect(self.done)
        self.buttonBox.rejected.connect(self.reject)
        self.shape_y = None
        self.shape_x = None
        self.y_low = None
        self.y_high = None
        self.x_low = None
        self.x_high = None

    def done(self):
        """
        On exit from the rectangular patch editor, look up the coordinate bounds of the rectangular
        patch.

        :return: -
        """

        if self.viewer.selection_rectangle() is not None:
            self.y_low, self.y_high, self.x_low, self.x_high = self.viewer.selection_rectangle()

        # If the patch was selected successfully, send the patch bounds to the workflow thread.
        # If it was not successful, send (0, 0, 0, 0).
        if self.parent_gui is not None:
            if self.y_low is not None:
                self.signal_finished.emit(self.y_low, self.y_high,
                                                         self.x_low, self.x_high)
            # If the patch is not valid, do frame alignment in automatic mode. This is done if
            # all bounds are zero.
            else:
                self.signal_finished.emit(0, 0, 0, 0)

        # Close the Window.
        self.close()

    def reject(self):
        """
        If the "cancel" button is pressed, the coordinates are not stored.

        :return: -
        """

        # No rectangle was selected. Continue the workflow in automatic mode.
        self.signal_finished.emit(0, 0, 0, 0)
        self.close()


if __name__ == '__main__':
    # Images can either be extracted from a video file or a batch of single photographs. Select
    # the example for the test run.
    type = 'video'
    if type == 'image':
        names = glob('Images/2012*.tif')
        # names = glob.glob('Images/Moon_Tile-031*ap85_8b.tif')
        # names = glob.glob('Images/Example-3*.jpg')
    else:
        names = 'Videos/another_short_video.avi'
    print(names)

    # Get configuration parameters.
    configuration = Configuration()
    configuration.initialize_configuration()
    try:
        frames = Frames(configuration, names, type=type)
        print("Number of images read: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Error as e:
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
        start = time()
        # Select the local rectangular patch in the image where the L gradient is highest in both x
        # and y direction. The scale factor specifies how much smaller the patch is compared to the
        # whole image frame.
        (y_low_opt, y_high_opt, x_low_opt, x_high_opt) = align_frames.compute_alignment_rect(
            configuration.align_frames_rectangle_scale_factor)
        end = time()
        print('Elapsed time in computing optimal alignment rectangle: {}'.format(end - start))
        print("optimal alignment rectangle, x_low: " + str(x_low_opt) + ", x_high: " + str(
            x_high_opt) + ", y_low: " + str(y_low_opt) + ", y_high: " + str(y_high_opt))
        reference_frame_with_alignment_points = frames.frames_mono(
            rank_frames.frame_ranks_max_index).copy()
        reference_frame_with_alignment_points[y_low_opt,
        x_low_opt:x_high_opt] = reference_frame_with_alignment_points[y_high_opt - 1,
                                x_low_opt:x_high_opt] = 255
        reference_frame_with_alignment_points[y_low_opt:y_high_opt,
        x_low_opt] = reference_frame_with_alignment_points[y_low_opt:y_high_opt,
                     x_high_opt - 1] = 255
        # plt.imshow(reference_frame_with_alignment_points, cmap='Greys_r')
        # plt.show()

    # Align all frames globally relative to the frame with the highest score.
    start = time()
    try:
        align_frames.align_frames()
    except NotSupportedError as e:
        print("Error: " + e.message)
        exit()
    except InternalError as e:
        print("Warning: " + e.message)
        for index, frame_number in enumerate(align_frames.failed_index_list):
            print("Shift computation failed for frame " + str(
                align_frames.failed_index_list[index]) + ", minima list: " + str(
                align_frames.dev_r_list[index]))
    end = time()
    print('Elapsed time in aligning all frames: {}'.format(end - start))
    print("Frame shifts: " + str(align_frames.frame_shifts))
    print("Intersection: " + str(align_frames.intersection_shape))

    start = time()
    # Compute the reference frame by averaging the best frames.
    average = align_frames.average_frame()
    border = configuration.align_frames_search_width
    # border = 100

    app = QtWidgets.QApplication(argv)
    window = RectangularPatchEditorWidget(None, average[border:-border, border:-border],
            "With 'ctrl' and the left mouse button pressed, draw a rectangular patch "
            "to be used for frame alignment. Or just press 'OK / Cancel' (automatic selection).",
            None)
    window.setMinimumSize(800, 600)
    window.showMaximized()
    app.exec_()

    print("Rectangle selected, y_low: " + str(border+window.y_low) + ", y_high: " +
           str(border+window.y_high) + ", x_low: " + str(border+window.x_low) + ", x_high: "
           + str(border+window.x_high))

    exit()

