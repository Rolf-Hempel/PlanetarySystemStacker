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

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


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
            x = pos.x()
            y = pos.y()

            # The left button is pressed.
            if event.button() == QtCore.Qt.LeftButton:
                self.left_button_pressed = True

                # Find the closest AP.
                neighbor_ap, distance = self.photo_editor.aps.find_neighbor(y, x)

                # If the distance is very small, assume that the AP is to be moved.
                if distance < self.max_match_distance:
                    self.moved_ap = neighbor_ap
                    self.remember_ap = neighbor_ap.copy()

                # Create a new AP.
                else:
                    # Compute the size of the AP. Take the standard size and reduce it to fit it
                    # into the frame if necessary.
                    half_patch_width_new = min(self.photo_editor.aps.half_patch_width, y,
                                               self.photo_editor.aps.shape_y - y, x,
                                               self.photo_editor.aps.shape_x - x)
                    # Create a preliminary AP with the computed size. It only becomes a real AP when
                    # the mouse is released.
                    self.remember_ap = self.photo_editor.aps.new_alignment_point(y, x,
                                        self.photo_editor.aps.half_box_width, half_patch_width_new)
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
            x = pos.x()
            y = pos.y()

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
                    ap, dist = self.photo_editor.aps.find_neighbor(y, x)
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

    def mouseMoveEvent(self, event):
        """
        The mouse is moved while being pressed.

        :param event: event object
        :return: -
        """

        if not self.photo_editor.drag_mode:
            pos = event.lastScenePos()
            self.x = pos.x()
            self.y = pos.y()

            # The mouse is moved with the left button pressed.
            if self.left_button_pressed:

                # The scene widget corresponding to the preliminary AP was stored with the AP.
                # Remove it from the scene before the moved AP is drawn.
                self.removeItem(self.remember_ap['graphics_item'])

                # Move the preliminary AP to the new coordinates.
                new_ap = self.photo_editor.aps.move_alignment_point(self.remember_ap, self.y,
                                                                    self.x)
                # Draw the new preliminary AP.
                self.photo_editor.draw_alignment_point(new_ap)
                # Update an area slightly larger than the AP patch.
                x_low = new_ap["patch_x_low"] - 5
                y_low = new_ap["patch_y_low"] - 5
                width = new_ap["patch_x_high"] - new_ap["patch_x_low"] + 10
                height = new_ap["patch_y_high"] - new_ap["patch_y_low"] + 10
                self.update(x_low, y_low, width, height)
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
        ap, dist = self.photo_editor.aps.find_neighbor(self.y, self.x)

        # Copy the AP, and apply the changes to the copy only.
        ap_new = ap.copy()
        self.photo_editor.aps.resize_alignment_point(ap_new, factor)

        # Replace the old AP with the resized version of it.
        self.photo_editor.replace_alignment_point(ap, ap_new)


class AlignmentPointGraphicsItem(QtWidgets.QGraphicsItem):
    """
    This widget represents an AP. It consists of a small red dot at the AP location, a red
    bounding rectangle, and a transparent green filling.
    """

    def __init__(self, ap):
        super(AlignmentPointGraphicsItem, self).__init__()

        # Set the color and transparency of the filling.
        self.color_surface = QtGui.QColor(0, 255, 0, 20)

        # Set the color of the bouding rectangle.
        self.color_boundary = QtGui.QColor(255, 0, 0)
        self.y = ap["y"]
        self.x = ap["x"]
        self.patch_y_low = ap["patch_y_low"]
        self.patch_y_high = ap["patch_y_high"]
        self.patch_x_low = ap["patch_x_low"]
        self.patch_x_high = ap["patch_x_high"]
        self.pen_boundary = QtGui.QPen(self.color_boundary)
        self.pen_boundary.setWidth(1)
        self.width_x = self.patch_x_high - self.patch_x_low
        self.width_x_external = self.width_x + self.pen_boundary.width()
        self.width_y = self.patch_y_high - self.patch_y_low
        self.width_y_external = self.width_y + self.pen_boundary.width()

        # Set the size of the central dot.
        self.dot_width = max(2, int(self.width_x / 50))

        # Store a reference of the widget at the corresponding AP.
        ap['graphics_item'] = self

    def boundingRect(self):
        return QtCore.QRectF(self.patch_x_low, self.patch_y_low, self.width_x_external,
                             self.width_y_external)

    def paint(self, painter, option, widget):
        painter.setPen(self.pen_boundary)
        painter.setBrush(self.color_boundary)
        painter.drawEllipse(self.x - self.dot_width / 2, self.y - self.dot_width / 2,
                            self.dot_width, self.dot_width)
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

    def __init__(self, parent):
        super(AlignmentPointEditor, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        # Initialize the scene. This object handles mouse events if not in drag mode.
        self._scene = GraphicsScene(self, self)
        # Initialize the photo object. No image is loaded yet.
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        # Initialize the udo stack.
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
        self.aps = None
        # Set the focus on the viewer, so the key event is caught.
        self.setFocus()

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

    def setPhoto(self, pixmap=None):
        """
        Assign a pixmap to the photo object.

        :param pixmap: pixmap object containing the photo.
        :return: -
        """

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

    def set_alignment_points(self, aps):
        """
        Store a reference to the object holding the alignment points, and draw all APs.

        :param aps: instance of class AlignmentPoints
        :return: -
        """

        self.aps = aps
        for ap in self.aps.alignment_points:
            self.draw_alignment_point(ap)

    def draw_alignment_point(self, ap):
        """
        Create a widget representing an AP and add it to the scene.

        :param ap: AP object
        :return: AP widget
        """

        ap_graphics_item = AlignmentPointGraphicsItem(ap)
        self._scene.addItem(ap_graphics_item)
        return ap_graphics_item

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


class CommandAdd(QtWidgets.QUndoCommand):
    """
    Undoable command to add an AP to the AP list.
    """
    def __init__(self, photo_editor, aps_object, ap):
        super(CommandAdd, self).__init__()
        self.photo_editor = photo_editor
        self.aps = aps_object
        self.ap = ap

    def redo(self):
        self.aps.add_alignment_point(self.ap)
        self.photo_editor.draw_alignment_point(self.ap)

    def undo(self):
        self.photo_editor._scene.removeItem(self.ap['graphics_item'])
        self.aps.remove_alignment_points([self.ap])


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

    def undo(self):
        for ap in self.ap_list:
            self.aps.add_alignment_point(ap)
            self.photo_editor.draw_alignment_point(ap)


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
        self.photo_editor.draw_alignment_point(self.ap_new)
        self.photo_editor._scene.update()

    def undo(self):
        self.photo_editor._scene.removeItem(self.ap_new['graphics_item'])
        self.aps.replace_alignment_point(self.ap_new, self.ap_old)
        self.photo_editor.draw_alignment_point(self.ap_old)
        self.photo_editor._scene.update()


class AlignmentPoints(object):
    """
    This is a mock-up for the real AlignmentPoints class with simplified behaviour. In the final
    code it is to be replaced with the real class.
    """

    def __init__(self, shape_y, shape_x, step_size, half_box_width, half_patch_width):
        self.alignment_points = []
        self.shape_y = shape_y
        self.shape_x = shape_x
        self.step_size = step_size
        self.half_box_width = half_box_width
        self.half_patch_width = half_patch_width

    def new_alignment_point(self, y, x, half_box_width, half_patch_width):
        ap = {}
        ap["y"] = y
        ap["x"] = x
        ap['box_y_low'] = y - half_box_width
        ap['box_y_high'] = y + half_box_width
        ap['box_x_low'] = x - half_box_width
        ap['box_x_high'] = x + half_box_width
        ap['patch_y_low'] = y - half_patch_width
        ap['patch_y_high'] = y + half_patch_width
        ap['patch_x_low'] = x - half_patch_width
        ap['patch_x_high'] = x + half_patch_width
        ap['graphics_item'] = None
        return ap

    def add_alignment_point(self, ap):
        self.alignment_points.append(ap)

    def remove_alignment_points(self, ap_list):
        aps_new = []
        for ap in self.alignment_points:
            if not ap in ap_list:
                aps_new.append(ap)
        self.alignment_points = aps_new

    def replace_alignment_point(self, ap, ap_new):
        self.alignment_points[self.alignment_points.index(ap)] = ap_new

    def move_alignment_point(self, ap, y_new, x_new):
        shift_y = y_new - ap["y"]
        shift_y = max(shift_y, -ap['patch_y_low'])
        shift_y = min(shift_y, self.shape_y - ap['patch_y_high'])
        shift_x = x_new - ap["x"]
        shift_x = max(shift_x, -ap['patch_x_low'])
        shift_x = min(shift_x, self.shape_x - ap['patch_x_high'])
        ap["y"] += shift_y
        ap["x"] += shift_x
        ap['box_y_low'] += shift_y
        ap['box_y_high'] += shift_y
        ap['box_x_low'] += shift_x
        ap['box_x_high'] += shift_x
        ap['patch_y_low'] += shift_y
        ap['patch_y_high'] += shift_y
        ap['patch_x_low'] += shift_x
        ap['patch_x_high'] += shift_x
        ap['graphics_item'] = None
        return ap

    def resize_alignment_point(self, ap, factor):
        y = ap["y"]
        x = ap["x"]
        patch_y_low = int((ap['patch_y_low'] - y) * factor) + y
        if patch_y_low < 0:
            return ap
        patch_y_high = int((ap['patch_y_high'] - y) * factor) + y
        if patch_y_high > self.shape_y:
            return ap
        patch_x_low = int((ap['patch_x_low'] - x) * factor) + x
        if patch_x_low < 0:
            return ap
        patch_x_high = int((ap['patch_x_high'] - x) * factor) + x
        if patch_x_high > self.shape_x:
            return ap
        ap['patch_y_low'] = patch_y_low
        ap['patch_y_high'] = patch_y_high
        ap['patch_x_low'] = patch_x_low
        ap['patch_x_high'] = patch_x_high
        ap['box_y_low'] = int((ap['box_y_low'] - y) * factor) + y
        ap['box_y_high'] = int((ap['box_y_high'] - y) * factor) + y
        ap['box_x_low'] = int((ap['box_x_low'] - x) * factor) + x
        ap['box_x_high'] = int((ap['box_x_high'] - x) * factor) + x
        ap['graphics_item'] = None
        return ap

    def create_ap_grid(self):
        for y in np.arange(2 * self.half_patch_width, self.shape_y - 2 * self.half_patch_width,
                           self.step_size):
            for x in np.arange(2 * self.half_patch_width, self.shape_x - 2 * self.half_patch_width,
                               self.step_size):
                self.alignment_points.append(
                    self.new_alignment_point(y, x, self.half_box_width, self.half_patch_width))

    def find_neighbor(self, y, x):
        """
        For a given (y, x) position find the closest "real" alignment point.

        :param y: y cocrdinate of location of interest
        :param x: x cocrdinate of location of interest

        :return: (Alignment point object of closest AP, distance to this AP)
        """
        min_distance_squared = 1.e30
        ap_neighbor = None
        for ap in self.alignment_points:
            distance_squared = (ap['y'] - y) ** 2 + (ap['x'] - x) ** 2
            if distance_squared < min_distance_squared:
                ap_neighbor = ap
                min_distance_squared = distance_squared
        return ap_neighbor, np.sqrt(min_distance_squared)

    def find_alignment_points(self, y_low, y_high, x_low, x_high):
        """
        Find all alignment points the centers of which are within given (y, x) bounds.

        :param y_low: Lower y pixel coordinate bound
        :param y_high: Upper y pixel coordinate bound
        :param x_low: Lower x pixel coordinate bound
        :param x_high: Upper x pixel coordinate bound
        :return: List of all alignment points with centers within the given coordinate bounds.
                 If no AP satisfies the condition, return an empty list.
        """

        ap_list = []
        for ap in self.alignment_points:
            if y_low <= ap['y'] <= y_high and x_low <= ap['x'] <= x_high:
                ap_list.append(ap)
        return ap_list


class Window(QtWidgets.QWidget):
    def __init__(self, file_name, alignment_points_half_box_width,
                 alignment_points_half_patch_width, alignment_points_search_width,
                 alignment_points_step_size):
        super(Window, self).__init__()
        self.file_name = file_name
        self.alignment_points_half_box_width = alignment_points_half_box_width
        self.alignment_points_half_patch_width = alignment_points_half_patch_width
        self.alignment_points_search_width = alignment_points_search_width
        self.alignment_points_step_size = alignment_points_step_size
        # 'Load image' button
        self.btnLoad = QtWidgets.QToolButton(self)
        self.btnLoad.setText('Load image')
        self.btnLoad.clicked.connect(self.loadImage)
        self.btnApGrid = QtWidgets.QToolButton(self)
        self.btnApGrid.setText('Create AP Grid')
        self.btnApGrid.clicked.connect(self.createApGrid)
        self.viewer = AlignmentPointEditor(self)
        self.btnUndo = QtWidgets.QToolButton(self)
        self.btnUndo.setText('Undo')
        self.btnUndo.clicked.connect(self.viewer.undoStack.undo)
        self.btnRedo = QtWidgets.QToolButton(self)
        self.btnRedo.setText('Redo')
        self.btnRedo.clicked.connect(self.viewer.undoStack.redo)
        self.aps = None
        self.shape_y = None
        self.shape_x = None
        # Arrange layout
        VBlayout = QtWidgets.QVBoxLayout(self)
        VBlayout.addWidget(self.viewer)
        HBlayout = QtWidgets.QHBoxLayout()
        HBlayout.setAlignment(QtCore.Qt.AlignLeft)
        HBlayout.addWidget(self.btnLoad)
        HBlayout.addWidget(self.btnApGrid)
        HBlayout.addWidget(self.btnUndo)
        HBlayout.addWidget(self.btnRedo)
        VBlayout.addLayout(HBlayout)

    def loadImage(self):
        # self.viewer.setPhoto(QtGui.QPixmap('2018-03-24_20-00MEZ_Mond_LRGB.jpg'))
        image = cv2.imread(self.file_name)
        self.shape_y = image.shape[0]
        self.shape_x = image.shape[1]
        qt_image = QtGui.QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3,
                                QtGui.QImage.Format_RGB888).rgbSwapped()
        pixmap = QtGui.QPixmap(qt_image)
        self.viewer.setPhoto(pixmap)
        self.viewer.fitInView()

    def createApGrid(self):
        self.aps = AlignmentPoints(self.shape_y, self.shape_x, self.alignment_points_step_size,
                                   self.alignment_points_half_box_width,
                                   self.alignment_points_half_patch_width)
        self.aps.create_ap_grid()
        self.viewer.set_alignment_points(self.aps)


if __name__ == '__main__':
    import sys

    file_name = 'Images/2018-03-24_20-00MEZ_Mond_LRGB.jpg'
    # file_name = 'Images/Moon_Tile-024_043939_stacked_interpolate_pp.tif'
    alignment_points_half_patch_width = 110
    alignment_points_search_width = 5

    # Compute derived constants.
    alignment_points_half_box_width = min(int(
        round((alignment_points_half_patch_width * 2) / 3)),
        alignment_points_half_patch_width - alignment_points_search_width)
    alignment_points_step_size = int(
        round((alignment_points_half_patch_width * 5) / 3))

    app = QtWidgets.QApplication(sys.argv)
    window = Window(file_name, alignment_points_half_box_width, alignment_points_half_patch_width,
                    alignment_points_search_width, alignment_points_step_size)
    window.setMinimumSize(800,600)
    window.showMaximized()
    sys.exit(app.exec_())
