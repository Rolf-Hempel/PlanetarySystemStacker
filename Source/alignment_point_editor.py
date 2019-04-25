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

from numpy import uint8
from PyQt5 import QtCore, QtGui, QtWidgets

from align_frames import AlignFrames
from alignment_point_editor_gui import Ui_alignment_point_editor
from alignment_points import AlignmentPoints
from configuration import Configuration, ConfigurationParameters
from exceptions import InternalError, NotSupportedError
from frames import Frames
from rank_frames import RankFrames


class GraphicsScene(QtWidgets.QGraphicsScene):
    """
    Handle mouse events on the graphics scene depicting the image with the alignment points (APs)
    superimposed. The left mouse button can be used to create APs, to move them, and to change their
    sizes. Using the right mouse button APs can be deleted.

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
        self.moved_ap = None
        self.new_ap = False
        self.remember_ap = None

        # Set the maximum diestance between a right click and a right release up to which they
        # are identified with each other. If the distance is larger, the mouse events are
        # identified with the opening of a rectangular patch between.
        self.single_click_threshold = 5
        # Set the maximum distance up to which a mouse position is identified with an existing
        # alignment point location.
        self.max_match_distance = 5
        # Set the factor by which the AP size is changed with a single mouse wheel stop.
        self.ap_size_change_factor = 1.05

    def mousePressEvent(self, event):
        """
        A nouse button is pressed. This can either be a lft or a right button.

        :param event: event object
        :return: -
        """

        # The following actions are not performed in drag-and-zoom mode. The switch between both
        # modes is handled in the higher-level object "photo_viewer".
        if not self.photo_editor.drag_mode:
            pos = event.lastScenePos()
            x = int(pos.x())
            y = int(pos.y())

            # The left button is pressed.
            if event.button() == QtCore.Qt.LeftButton:
                self.left_button_pressed = True

                # Find the closest AP.
                neighbor_ap, distance = self.photo_editor.aps.find_neighbor(y, x,
                                        self.photo_editor.aps.alignment_points)

                # If the AP list is not empty and the closest distance is very small, assume that
                # the AP is to be moved.
                if neighbor_ap and distance < self.max_match_distance:
                    self.moved_ap = neighbor_ap
                    self.remember_ap = neighbor_ap.copy()

                # Create a new AP.
                else:
                    # Create a preliminary AP with the computed size. It only becomes a real AP when
                    # the mouse is released.
                    self.remember_ap = self.photo_editor.aps.new_alignment_point(y, x, False, False,
                        False, False)
                    if self.remember_ap:
                        self.new_ap = True

            # The right button is pressed.
            elif event.button() == QtCore.Qt.RightButton:
                self.right_button_pressed = True

                # Remember the location and initialize an object which during mouse moving stores
                # the rectangular patch opening between start and end positions.
                self.right_y_start = pos.y()
                self.right_x_start = pos.x()
                self.remember_sr = None

    def mouseReleaseEvent(self, event):
        """
        A mouse button is released.

        :param event: event object
        :return: -
        """

        if not self.photo_editor.drag_mode:
            pos = event.lastScenePos()
            x = int(pos.x())
            y = int(pos.y())

            # The left button is released.
            if event.button() == QtCore.Qt.LeftButton:
                self.left_button_pressed = False

                # An existing AP was moved, replace it with the moved one.
                if self.moved_ap:
                    self.removeItem(self.remember_ap['graphics_item'])
                    self.photo_editor.replace_alignment_point(self.moved_ap,
                                                              self.remember_ap)
                    self.moved_ap = None
                # A new AP was created, and possibly moved. Add it to the list.
                elif self.new_ap:
                    self.removeItem(self.remember_ap['graphics_item'])
                    self.photo_editor.add_alignment_point(self.remember_ap)
                    self.new_ap = False

            # The right button is released.
            elif event.button() == QtCore.Qt.RightButton:
                self.right_button_pressed = False

                # If the mouse was not moved much between press and release, a single AP is deleted.
                if max(abs(y - self.right_y_start),
                       abs(x - self.right_x_start)) < self.single_click_threshold:

                    # Find the closest AP and remove it from the scene and the AP list.
                    ap, dist = self.photo_editor.aps.find_neighbor(y, x,
                               self.photo_editor.aps.alignment_points)
                    self.photo_editor.remove_alignment_points([ap])

                # The mouse was moved between press and release. Remove all APs in the opening
                # rectangular patch, both from the scene and the AP list.
                else:
                    y_low = min(self.right_y_start, self.right_y_end)
                    y_high = max(self.right_y_start, self.right_y_end)
                    x_low = min(self.right_x_start, self.right_x_end)
                    x_high = max(self.right_x_start, self.right_x_end)
                    remove_ap_list = self.photo_editor.aps.find_alignment_points(y_low, y_high,
                                                                                 x_low, x_high)
                    self.removeItem(self.remember_sr)
                    self.photo_editor.remove_alignment_points(remove_ap_list)
            self.update()

    def mouseMoveEvent(self, event):
        """
        The mouse is moved while being pressed.

        :param event: event object
        :return: -
        """

        pos = event.lastScenePos()
        self.x = int(pos.x())
        self.y = int(pos.y())

        if not self.photo_editor.drag_mode:
            # The mouse is moved with the left button pressed.
            if self.left_button_pressed and (self.moved_ap or self.new_ap):

                # Copy the preliminary AP, and apply the changes to the copy only.
                new_ap = self.remember_ap.copy()

                # Move the preliminary AP to the new coordinates.
                new_ap = self.photo_editor.aps.move_alignment_point(new_ap, self.y, self.x)
                # The move was successful. Replace the preliminary AP with the new one.
                if new_ap:
                    # The scene widget corresponding to the preliminary AP was stored with the AP.
                    # Remove it from the scene before the moved AP is drawn.
                    self.removeItem(self.remember_ap['graphics_item'])
                    # Draw the new preliminary AP.
                    new_ap['graphics_item'] = AlignmentPointGraphicsItem(new_ap)
                    self.addItem(new_ap['graphics_item'])
                    self.remember_ap = new_ap

            # The mouse is moved with the right button pressed.
            elif self.right_button_pressed:
                self.right_y_end = self.y
                self.right_x_end = self.x

                # Compute the new rectangle for selecting APs to be removed.
                new_sr = SelectionRectangleGraphicsItem(self.right_y_start, self.right_x_start,
                    self.y, self.x)

                # If the rectangle was drawn for a previous location, replace it with the new one.
                if self.remember_sr is not None:
                    self.removeItem(self.remember_sr)
                self.addItem(new_sr)
                self.remember_sr = new_sr
            self.update()

    def wheelEvent(self, event):
        """
        Handle scroll events. Change the size of the nearest AP.

        :param event: wheel event object
        :return: -
        """

        # Depending of wheel direction, set the factor to greater or smaller than 1.
        self.change_ap_size(event.angleDelta().y())

    def keyPressEvent(self, event):
        """
        Handle key events as an alternative to wheel events. Change the size of the nearest AP.

        :param event: key event
        :return: -
        """

        # This is a workaround: Instead of "93" it should read "QtCore.Qt.Key_Plus", but that
        # returns 43 instead.
        if event.key() == 93 and event.modifiers() & QtCore.Qt.ControlModifier:
            self.change_ap_size(1)
        elif event.key() == QtCore.Qt.Key_Minus and event.modifiers() & QtCore.Qt.ControlModifier:
            self.change_ap_size(-1)

    def change_ap_size(self, direction):
        """
        Change the size of the nearest AP.

        :param direction: If > 0, increase the size by a fixed factor. If < 0, decrease its size.
        :return: -
        """

        # Depending of direction value, set the factor to greater or smaller than 1.
        if direction > 0:
            factor = self.ap_size_change_factor
        else:
            factor = 1. / self.ap_size_change_factor

        # Find the closest AP.
        ap, dist = self.photo_editor.aps.find_neighbor(self.y, self.x,
                                                       self.photo_editor.aps.alignment_points)

        # Copy the AP, and apply the changes to the copy only.
        new_ap = ap.copy()
        if self.photo_editor.aps.resize_alignment_point(new_ap, factor):
            new_ap['graphics_item'] = AlignmentPointGraphicsItem(new_ap)
            # Replace the old AP with the resized version of it.
            self.photo_editor.replace_alignment_point(ap, new_ap)


class AlignmentPointGraphicsItem(QtWidgets.QGraphicsItem):
    """
    This widget represents an AP. It consists of a small red dot at the AP location, a red
    bounding rectangle, and a transparent green filling.
    """

    def __init__(self, ap):
        super(AlignmentPointGraphicsItem, self).__init__()

        # Set the color and transparency of the filling.
        self.color_surface = QtGui.QColor(255, 255, 0, 60)

        # Set the color of the bouding rectangle and central dot.
        self.color_boundary = QtGui.QColor(255, 0, 0)
        self.y = ap["y"]
        self.x = ap["x"]
        self.patch_y_low = ap["patch_y_low"]
        self.patch_y_high = ap["patch_y_high"]
        self.patch_x_low = ap["patch_x_low"]
        self.patch_x_high = ap["patch_x_high"]
        self.pen_boundary = QtGui.QPen(self.color_boundary)
        self.pen_boundary.setStyle(0)
        self.width_x = self.patch_x_high - self.patch_x_low
        self.width_x_external = self.width_x + self.pen_boundary.width()
        self.width_y = self.patch_y_high - self.patch_y_low
        self.width_y_external = self.width_y + self.pen_boundary.width()

        # Set the size of the central dot.
        self.dot_width = max(1, int(self.width_x / 30))

    def boundingRect(self):
        return QtCore.QRectF(self.patch_x_low, self.patch_y_low, self.width_x_external,
                             self.width_y_external)

    def paint(self, painter, option, widget):
        painter.setPen(self.pen_boundary)
        painter.setBrush(self.color_boundary)
        painter.drawEllipse(self.x, self.y, self.dot_width, self.dot_width)
        painter.setBrush(self.color_surface)
        painter.drawRect(self.patch_x_low, self.patch_y_low, self.width_x, self.width_y)


class SelectionRectangleGraphicsItem(QtWidgets.QGraphicsItem):
    """
    This widget represents the selection rectangle opening when the mouse is being moved with the
    right mouse button  pressed.
    """

    def __init__(self, y_start, x_start, y_end, x_end):
        super(SelectionRectangleGraphicsItem, self).__init__()

        # Set the color of the transparent rectangle filling.
        self.color_surface = QtGui.QColor(0, 0, 255, 40)

        # Set the color of the bounding rectangle.
        self.color_boundary = QtGui.QColor(255, 0, 0)

        # Set the coordinate limits of the rectangle. Start and end locations can be anywhere.
        self.y_low = min(y_start, y_end)
        self.x_low = min(x_start, x_end)
        self.y_high = max(y_start, y_end)
        self.x_high = max(x_start, x_end)
        self.pen_boundary = QtGui.QPen(self.color_boundary)
        self.pen_boundary.setWidth(1)
        self.width_y = self.y_high - self.y_low
        self.width_x = self.x_high - self.x_low
        self.width_y_external = self.width_y + self.pen_boundary.width()
        self.width_x_external = self.width_x + self.pen_boundary.width()

    def boundingRect(self):
        return QtCore.QRectF(self.x_low, self.y_low, self.width_x_external, self.width_y_external)

    def paint(self, painter, option, widget):
        painter.setPen(self.pen_boundary)
        painter.setBrush(self.color_surface)
        painter.drawRect(self.x_low, self.y_low, self.width_x, self.width_y)


class AlignmentPointEditor(QtWidgets.QGraphicsView):
    """
    This widget implements an editor for handling APs superimposed onto an image. It supports two
    modes:
    - In "drag mode" the mouse can be used for panning, and the scroll wheel for zooming.
    - In "alignment point mode" the mouse is used to create/remove APs, to move them or to change
      their sizes.
    The "cntrl" key is used to switch between the two modes.
    """

    resized = QtCore.pyqtSignal()

    def __init__(self, image, alignment_points):
        super(AlignmentPointEditor, self).__init__()
        self._zoom = 0
        self._empty = True
        self.image = image
        self.shape_y = None
        self.shape_x = None
        # Initialize the scene. This object handles mouse events if not in drag mode.
        self._scene = GraphicsScene(self, self)
        # Initialize the photo object. No image is loaded yet.
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        # Initialize the undo stack.
        self.undoStack = QtWidgets.QUndoStack(self)

        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.drag_mode = True
        # Initialize the alignment point object.
        self.aps = alignment_points

        # Load the image, and connect it to resizing of this window.
        self.setPhoto(self.image)
        self.resized.connect(self.fitInView)

        # Set the focus on the viewer, so the key event is caught.
        self.setFocus()

    def resizeEvent(self, event):
        self.resized.emit()
        return super(AlignmentPointEditor, self).resizeEvent(event)

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
        Convert a grayscale image to a pixmap and assign it to the photo object.

        :param image: grayscale image in format float32.
        :return: -
        """

        self.image = image
        # Convert the float32 monochrome image into uint8 format.
        image_uint8 = self.image.astype(uint8)
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

    def keyPressEvent(self, event):
        """
        The control key is used to switch between drag and AP modes.

        :param event: event object
        :return: -
        """

        # If the control key is pressed, switch to "no drag mode".
        # Use default handling for other keys.
        if event.key() == QtCore.Qt.Key_Control:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self.drag_mode = False
        elif event.key() == QtCore.Qt.Key_Plus and not event.modifiers() & QtCore.Qt.ControlModifier:
            self.zoom(1)
        elif event.key() == QtCore.Qt.Key_Minus and not event.modifiers() & QtCore.Qt.ControlModifier:
            self.zoom(-1)
        elif event.key() == QtCore.Qt.Key_Z and event.modifiers() & QtCore.Qt.ControlModifier:
            self.undoStack.undo()
        elif event.key() == QtCore.Qt.Key_Y and event.modifiers() & QtCore.Qt.ControlModifier:
            self.undoStack.redo()
        else:
            super(AlignmentPointEditor, self).keyPressEvent(event)

    def keyReleaseEvent(self, event):
        # If the control key is released, switch back to "drag mode".
        # Use default handling for other keys.
        if event.key() == QtCore.Qt.Key_Control:
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self.drag_mode = True
        else:
            super(AlignmentPointEditor, self).keyPressEvent(event)

    def createApGrid(self):
        """
        Create an initial alignment point grid, and display it in the viewer.

        :return: -
        """

        command = CommandCreateApGrid(self)
        self.undoStack.push(command)

    def add_alignment_point(self, ap):
        """
        Add an AP using the undo framework.

        :param ap: AP object in the AP list
        :return: -
        """

        command = CommandAdd(self, self.aps, ap)
        self.undoStack.push(command)

    def remove_alignment_points(self, ap_list):
        """
        Remove an AP using the undo framework.

        :param ap: AP object in the AP list
        :return: -
        """

        command = CommandRemove(self, self.aps, ap_list)
        self.undoStack.push(command)

    def replace_alignment_point(self, ap_old, ap_new):
        """
        Replace an AP with another one using the undo framework.

        :param ap_old: AP object in the AP list to be replaced
        :param ap_new: new AP object to be added to the AP list
        :return: -
        """

        command = CommandReplace(self, self.aps, ap_old, ap_new)
        self.undoStack.push(command)

class CommandCreateApGrid(QtWidgets.QUndoCommand):
    """
    Undoable command to replace all existing APs with a new AP grid.
    """
    def __init__(self, photo_editor):
        super(CommandCreateApGrid, self).__init__()
        self.photo_editor = photo_editor
        # Copy the old ap list before overwriting it with the new AP grid.
        self.old_ap_list = self.photo_editor.aps.alignment_points
        # Create the new AP grid.
        self.photo_editor.aps.create_ap_grid()
        # Copy the new list of APs.
        self.new_ap_list = self.photo_editor.aps.alignment_points
        for ap in self.new_ap_list:
            ap['graphics_item'] = AlignmentPointGraphicsItem(ap)

    def redo(self):
        # Remove all APs from the scene.
        for item in self.photo_editor._scene.items():
            if isinstance(item, AlignmentPointGraphicsItem):
                self.photo_editor._scene.removeItem(item)

        # Copy the AP list from grid generation to the "alignment_point" object.
        self.photo_editor.aps.alignment_points = self.new_ap_list.copy()
        # Draw all new APs and update the scene.
        for ap in self.photo_editor.aps.alignment_points:
            self.photo_editor._scene.addItem(ap['graphics_item'])
        self.photo_editor._scene.update()
        self.photo_editor.setFocus()

    def undo(self):
        # Restore the old AP list.
        self.photo_editor.aps.alignment_points = self.old_ap_list

        # Remove all durrent APs from the scene.
        for item in self.photo_editor._scene.items():
            if isinstance(item, AlignmentPointGraphicsItem):
                self.photo_editor._scene.removeItem(item)
        # Draw all old APs and update the scene.
        for ap in self.photo_editor.aps.alignment_points:
            self.photo_editor._scene.addItem(ap['graphics_item'])
        self.photo_editor._scene.update()
        self.photo_editor.setFocus()


class CommandAdd(QtWidgets.QUndoCommand):
    """
    Undoable command to add an AP to the AP list.
    """
    def __init__(self, photo_editor, aps_object, ap):
        super(CommandAdd, self).__init__()
        self.photo_editor = photo_editor
        self.aps = aps_object
        self.ap = ap
        self.ap['graphics_item'] = AlignmentPointGraphicsItem(ap)

    def redo(self):
        self.aps.add_alignment_point(self.ap)
        self.photo_editor._scene.addItem(self.ap['graphics_item'])
        self.photo_editor._scene.update()
        self.photo_editor.setFocus()

    def undo(self):
        self.photo_editor._scene.removeItem(self.ap['graphics_item'])
        self.aps.remove_alignment_points([self.ap])
        self.photo_editor._scene.update()
        self.photo_editor.setFocus()


class CommandRemove(QtWidgets.QUndoCommand):
    """
    Undoable command to remove an AP from the AP list.
    """
    def __init__(self, photo_editor, aps_object, ap_list):
        super(CommandRemove, self).__init__()
        self.photo_editor = photo_editor
        self.aps = aps_object
        self.ap_list = ap_list

    def redo(self):
        for ap in self.ap_list:
            self.photo_editor._scene.removeItem(ap['graphics_item'])
        self.aps.remove_alignment_points(self.ap_list)
        self.photo_editor._scene.update()
        self.photo_editor.setFocus()

    def undo(self):
        for ap in self.ap_list:
            self.aps.add_alignment_point(ap)
            self.photo_editor._scene.addItem(ap['graphics_item'])
        self.photo_editor._scene.update()
        self.photo_editor.setFocus()


class CommandReplace(QtWidgets.QUndoCommand):
    """
    Undoable command to replace an AP on the AP list with another one.
    """
    def __init__(self, photo_editor, aps_object, ap_old, ap_new):
        super(CommandReplace, self).__init__()
        self.photo_editor = photo_editor
        self.aps = aps_object
        self.ap_old = ap_old
        self.ap_new = ap_new

    def redo(self):
        self.photo_editor._scene.removeItem(self.ap_old['graphics_item'])
        self.aps.replace_alignment_point(self.ap_old, self.ap_new)
        self.photo_editor._scene.addItem(self.ap_new['graphics_item'])
        self.photo_editor._scene.update()
        self.photo_editor.setFocus()

    def undo(self):
        self.photo_editor._scene.removeItem(self.ap_new['graphics_item'])
        self.aps.replace_alignment_point(self.ap_new, self.ap_old)
        self.photo_editor._scene.addItem(self.ap_old['graphics_item'])
        self.photo_editor._scene.update()
        self.photo_editor.setFocus()


class AlignmentPointEditorWidget(QtWidgets.QFrame, Ui_alignment_point_editor):
    """
    This widget implements the AP viewer, to be used as part of the application GUI.
    """
    def __init__(self, parent_gui, configuration, align_frames, alignment_points, signal_finished,
                 parent=None):
        """
        Initialization of the widget.

        :param parent_gui: Parent GUI object
        :param image: Background image on which the APs are superimposed. Usually, the mean frame is
                      used for this purpose.
        :param configuration: Configuration object with parameters
        :param align_frames: AlignFrames object with global shift information for all frames
        :param alignment_points: Alignment point object
        :param signal_finished: Qt signal telling the workflow thread that the APs have been
                                created successfully
        """

        # QtWidgets.QFrame.__init__(self, parent)
        super(AlignmentPointEditorWidget, self).__init__()
        self.setupUi(self)

        self.parent_gui = parent_gui

        # If the mean frame type is not uint8, values correspond to 16bit resolution.
        if align_frames.mean_frame.dtype != uint8:
            self.mean_frame = align_frames.mean_frame[:,:]/256.
        else:
            self.mean_frame = align_frames.mean_frame

        self.configuration = configuration
        self.aps = alignment_points
        self.signal_finished = signal_finished

        # Create the viewer frame and insert it into the window.
        self.viewer = AlignmentPointEditor(self.mean_frame, self.aps)
        self.horizontalLayout_2.insertWidget(0, self.viewer)
        self.horizontalLayout_2.setStretch(1,0)

        # Initialize sliders and their value labels.
        self.initialize_widgets_and_local_parameters(
            self.configuration.alignment_points_half_box_width,
            self.configuration.alignment_points_structure_threshold,
            self.configuration.alignment_points_brightness_threshold)

        # Connect events with activities.
        self.aphbw_slider_value.valueChanged['int'].connect(self.aphbw_changed)
        self.apst_slider_value.valueChanged['int'].connect(self.apst_changed)
        self.apbt_slider_value.valueChanged['int'].connect(self.apbt_changed)
        self.restore_standard_values.clicked.connect(self.restore_standard_parameters)

        self.btnApGrid.clicked.connect(self.viewer.createApGrid)
        self.btnUndo.clicked.connect(self.viewer.undoStack.undo)
        self.btnRedo.clicked.connect(self.viewer.undoStack.redo)
        self.buttonBox.accepted.connect(self.done)
        self.shape_y = None
        self.shape_x = None

    def initialize_widgets_and_local_parameters(self, half_box_width, structure_threshold,
                                                brightness_threshold):
        """
        Initialize GUI widgets with current configuration parameter values.

        :param half_box_width: Half the width of a standard alignment box.
        :param structure_threshold: Minimum structure value for an alignment point
                                    (between 0. and 1.)
        :param brightness_threshold: The brightest pixel must be brighter than this value
                                     (0 < value <256)
        :return: -
        """

        self.aphbw_slider_value.setValue(half_box_width * 2)
        self.aphbw_label_display.setText(str(half_box_width * 2))
        self.apst_slider_value.setValue(int(round(structure_threshold * 100)))
        self.apst_label_display.setText(str(structure_threshold))
        self.apbt_slider_value.setValue(brightness_threshold)
        self.apbt_label_display.setText(str(brightness_threshold))
        self.label_message.setStyleSheet('color: red')


    def aphbw_changed(self, value):
        self.configuration.alignment_points_half_box_width = int(value / 2)
        self.configuration.set_derived_parameters()

    def apst_changed(self, value):
        self.configuration.alignment_points_structure_threshold = value / 100.
        self.apst_label_display.setText(str(self.configuration.alignment_points_structure_threshold))

    def apbt_changed(self, value):
        self.configuration.alignment_points_brightness_threshold = value

    def restore_standard_parameters(self):
        """
        Reset configuration parameters and GUI widget settings to standard values.

        :return: -
        """

        # Create a ConfigurationParameters object with standard values for the three AP parameters.
        config_parameters = ConfigurationParameters()
        config_parameters.set_defaults_ap_editing()

        # Reset configuration parameters to standard values.
        self.configuration.alignment_points_half_box_width = \
            config_parameters.alignment_points_half_box_width
        self.configuration.alignment_points_structure_threshold = \
            config_parameters.alignment_points_structure_threshold
        self.configuration.alignment_points_brightness_threshold = \
            config_parameters.alignment_points_brightness_threshold

        # Initialize sliders and their value labels.
        self.initialize_widgets_and_local_parameters(
            self.configuration.alignment_points_half_box_width,
            self.configuration.alignment_points_structure_threshold,
            self.configuration.alignment_points_brightness_threshold)

    def done(self):
        # On exit from the alignment point editor, allocate buffers for APs which have been
        # changed during editing.
        for ap in self.aps.alignment_points:
            if ap['reference_box'] is None:
                self.aps.set_reference_box(ap, self.mean_frame)

        # Reset the busy status of the higher-level GUI (to enable the "play" button) and
        # update the status of the higher-level GUI (if a reference to it was passed in "init").
        if self.parent_gui is not None:
            self.parent_gui.busy = False
            self.parent_gui.update_status()

            # Tell the workflow thread that the APs have been created successfully.
            self.signal_finished.emit()
        # Close the Window.
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
            align_frames.frame_ranks_max_index).copy()
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

    # Initialize the AlignmentPoints object.
    alignment_points = AlignmentPoints(configuration, frames, rank_frames, align_frames)
    end = time()
    print('Elapsed time in computing average frame: {}'.format(end - start))
    print("Average frame computed from the best " + str(
        align_frames.average_frame_number) + " frames.")
    # plt.imshow(align_frames.mean_frame, cmap='Greys_r')
    # plt.show()

    app = QtWidgets.QApplication(argv)
    window = AlignmentPointEditorWidget(None, configuration, align_frames, alignment_points, None)
    window.setMinimumSize(800,600)
    window.showMaximized()
    app.exec_()

    print("After AP editing, number of APs: " + str(
        len(alignment_points.alignment_points)) + ", aps dropped because too dim: " + str(
        alignment_points.alignment_points_dropped_dim) + ", aps dropped because too little "
                                                              "structure: " + str(
        alignment_points.alignment_points_dropped_structure))

    count_updates = 0
    for ap in alignment_points.alignment_points:
        if ap['reference_box'] is not None:
            continue
        count_updates += 1
        AlignmentPoints.set_reference_box(ap, align_frames.mean_frame)
    print ("Buffers allocated for " + str(count_updates) + " alignment points.")
    exit()

