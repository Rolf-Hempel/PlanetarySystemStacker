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

Part of this module (in class "AlignmentPointViewer" was copied from
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

    def __init__(self, photo_viewer, parent=None):
        """
        Initialize the scene object.

        :param photo_viewer: the object in which the scene is defined
        :param parent: The parent class
        """

        QtWidgets.QGraphicsScene.__init__(self, parent)
        self.photo_viewer = photo_viewer
        self.left_button_pressed = False
        self.right_button_pressed = False

        # Set the maximum diestance between a right click and a right release up to which they
        # are identified with each other. If the distance is larger, the mouse events are
        # identified with the opening of a rectangular patch between.
        self.single_click_threshold = 5
        # Set the maximum distance up to which a mouse position is identified with an existing
        # alignment point location.
        self.max_match_distance = 5
        # Initialize a state variable telling if an AP is being moved.
        self.move_ap = None

    def mousePressEvent(self, event):
        """
        A nouse button is pressed. This can either be a lft or a right button.

        :param event: event object
        :return: -
        """

        pos = event.lastScenePos()
        x = pos.x()
        y = pos.y()

        # The left button is pressed.
        if event.button() == QtCore.Qt.LeftButton:
            self.left_button_pressed = True

            # The following actions are not performed in drag-and-zoom mode. The switch between both
            # modes is handled in the higher-level object "photo_viewer".
            if not self.photo_viewer.drag_mode:

                # Find the closest AP.
                neighbor_ap, distance = self.photo_viewer.aps.find_neighbor(y, x)

                # If the distance is very small, assume that the AP is to be moved.
                if distance < self.max_match_distance:
                    self.remember_ap = neighbor_ap
                    self.move_ap = True

                # Create a new AP.
                else:
                    # Compute the size of the AP. Take the standard size and reduce it to fit it
                    # into the frame if necessary.
                    half_patch_width_new = min(self.photo_viewer.aps.half_patch_width, y,
                                               self.photo_viewer.aps.shape_y - y, x,
                                               self.photo_viewer.aps.shape_x - x)
                    # Create a preliminary AP with the computed size. It only becomes a real AP when
                    # the mouse is released.
                    self.remember_ap = self.photo_viewer.aps.new_alignment_point(y, x,
                                       self.photo_viewer.aps.half_box_width, half_patch_width_new)
                    # Add a widget showing the AP to the scene and remember the current mouse
                    # position.
                    self.photo_viewer.draw_alignment_point(self.remember_ap)
                    self.left_y_start = y
                    self.left_x_start = x
                    self.move_ap = False

        # The right button is pressed.
        elif event.button() == QtCore.Qt.RightButton:
            self.right_button_pressed = True

            # If not in drag-and-zoom mode, remember the location and initialize an object which
            # during mouse moving stores the rectangular patch opening between start and end
            # positions.
            if not self.photo_viewer.drag_mode:
                self.right_y_start = pos.y()
                self.right_x_start = pos.x()
                self.remember_sr = None

    def mouseReleaseEvent(self, event):
        """
        A mouse button is released.

        :param event: event object
        :return: -
        """

        pos = event.lastScenePos()

        # The left button is released.
        if event.button() == QtCore.Qt.LeftButton:
            self.left_button_pressed = False
            if not self.photo_viewer.drag_mode:

                # If a new AP is being created, append the preliminary AP to the list of APs.
                if not self.move_ap:
                    self.photo_viewer.aps.alignment_points.append(self.remember_ap)

        # The right button is released.
        elif event.button() == QtCore.Qt.RightButton:
            self.right_button_pressed = False
            if not self.photo_viewer.drag_mode:
                x = pos.x()
                y = pos.y()

                # If the mouse was not moved much between press and release, a single AP is deleted.
                if max(abs(y - self.right_y_start),
                       abs(x - self.right_x_start)) < self.single_click_threshold:

                    # Find the closest AP and remove it from the scene and the AP list.
                    ap, dist = self.photo_viewer.aps.find_neighbor(y, x)
                    self.removeItem(ap['graphics_item'])
                    self.photo_viewer.aps.remove_alignment_points([ap])

                # The mouse was moved between press and release. Remove all APs in the opening
                # rectangular patch, both from the scene and the AP list.
                else:

                    remove_ap_list = []
                    y_low = min(self.right_y_start, self.right_y_end)
                    y_high = max(self.right_y_start, self.right_y_end)
                    x_low = min(self.right_x_start, self.right_x_end)
                    x_high = max(self.right_x_start, self.right_x_end)
                    remove_ap_list = self.photo_viewer.aps.find_alignment_points(y_low, y_high,
                                                                                 x_low, x_high)
                    for ap in remove_ap_list:
                        self.removeItem(ap['graphics_item'])
                    self.removeItem(self.remember_sr)
                    self.photo_viewer.aps.remove_alignment_points(remove_ap_list)

    def mouseMoveEvent(self, event):
        """
        The mouse is moved while being pressed.

        :param event: event object
        :return: -
        """

        # The mouse is moved with the left button pressed.
        if self.left_button_pressed:
            pos = event.lastScenePos()
            if not self.photo_viewer.drag_mode:
                x_new = pos.x()
                y_new = pos.y()

                # The scene widget corresponding to the preliminary AP was stored with the AP.
                # Remove it from the scene before the moved AP is drawn.
                self.removeItem(self.remember_ap['graphics_item'])

                # Move the preliminary AP to the new coordinates.
                if self.move_ap:
                    new_ap = self.photo_viewer.aps.move_alignment_point(self.remember_ap, y_new,
                                                                        x_new)
                # The preliminary AP stays at the same position, but its size is being increased.
                # Compute the new size and create a new preliminary AP.
                else:
                    half_patch_width_new = max(abs(y_new - self.left_y_start),
                                               abs(x_new - self.left_x_start))
                    half_patch_width_new = min(half_patch_width_new, self.left_y_start,
                                               self.photo_viewer.aps.shape_y - self.left_y_start,
                                               self.left_x_start,
                                               self.photo_viewer.aps.shape_x - self.left_x_start)
                    new_ap = self.photo_viewer.aps.new_alignment_point(self.left_y_start,
                                self.left_x_start, self.photo_viewer.aps.half_box_width,
                                half_patch_width_new)

                # Draw the new preliminary AP.
                self.photo_viewer.draw_alignment_point(new_ap)
                # Update an area slightly larger than the AP patch.
                x_low = new_ap["patch_x_low"] - 5
                y_low = new_ap["patch_y_low"] - 5
                width = new_ap["patch_x_high"] - new_ap["patch_x_low"] + 10
                height = new_ap["patch_y_high"] - new_ap["patch_y_low"] + 10
                self.update(x_low, y_low, width, height)

                self.remember_ap = new_ap

        # The mouse is moved with the right button pressed.
        elif self.right_button_pressed:
            pos = event.lastScenePos()

            if not self.photo_viewer.drag_mode:
                x = pos.x()
                y = pos.y()
                self.right_y_end = y
                self.right_x_end = x

                # Compute the new rectangle for selecting APs to be removed.
                new_sr = SelectionRectangleGraphicsItem(
                    self.right_y_start, self.right_x_start, y, x)

                # If the rectangle was drawn for a previous location, replace it with the new one.
                if self.remember_sr is not None:
                    self.removeItem(self.remember_sr)
                self.addItem(new_sr)
                self.remember_sr = new_sr


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


class AlignmentPointViewer(QtWidgets.QGraphicsView):
    """
    This widget implements a viewer for handling APs superimposed onto an image. It supports two
    modes:
    - In "drag mode" the mouse can be used for panning, and the scroll wheel for zooming.
    - In "alignment point mode" the mouse is used to create/remove APs, or to move them.
    The "cntrl" key is used to switch between the two modes.
    """

    def __init__(self, parent):
        super(AlignmentPointViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        # Initialize the scene. This object handles mouse events if not in drag mode.
        self._scene = GraphicsScene(self, self)
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

    def wheelEvent(self, event):
        """
        Handle scroll events for zooming in and out of the scene. This is only active when a photo
        is loaded.

        :param event: wheel event object
        :return: -
        """

        if self.hasPhoto():
            # Depending of wheel direction, set the zoom factor to greater or smaller than 1.
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1

            # Apply the zoom factor to the scene. If the zoom counter is zero, fit the scene to the
            # window size.
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
        else:
            super(AlignmentPointViewer, self).keyPressEvent(event)

    def keyReleaseEvent(self, event):
        # If the control key is released, switch back to "drag mode".
        # Use default handling for other keys.
        if event.key() == QtCore.Qt.Key_Control:
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self.drag_mode = True
        else:
            super(AlignmentPointViewer, self).keyPressEvent(event)


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

    def remove_alignment_points(self, ap_list):
        aps_new = []
        for ap in self.alignment_points:
            if not ap in ap_list:
                aps_new.append(ap)
        self.alignment_points = aps_new


class Window(QtWidgets.QWidget):
    def __init__(self, file_name, alignment_points_half_patch_width, alignment_points_search_width):
        super(Window, self).__init__()
        self.file_name = file_name
        self.alignment_points_half_patch_width = alignment_points_half_patch_width
        self.alignment_points_search_width = alignment_points_search_width
        # 'Load image' button
        self.btnLoad = QtWidgets.QToolButton(self)
        self.btnLoad.setText('Load image')
        self.btnLoad.clicked.connect(self.loadImage)
        self.btnApGrid = QtWidgets.QToolButton(self)
        self.btnApGrid.setText('Create AP Grid')
        self.btnApGrid.clicked.connect(self.createApGrid)
        self.viewer = AlignmentPointViewer(self)
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
        alignment_points_step_size = int(
            round((self.alignment_points_half_patch_width * 5) / 3))
        alignment_points_half_box_width = min(int(
            round((self.alignment_points_half_patch_width * 2) / 3)),
            self.alignment_points_half_patch_width - self.alignment_points_search_width)
        self.aps = AlignmentPoints(self.shape_y, self.shape_x, alignment_points_step_size,
                                   alignment_points_half_box_width,
                                   self.alignment_points_half_patch_width)
        self.aps.create_ap_grid()
        self.viewer.set_alignment_points(self.aps)


if __name__ == '__main__':
    import sys

    file_name = 'Images/2018-03-24_20-00MEZ_Mond_LRGB.jpg'
    # file_name = 'Images/Moon_Tile-024_043939_stacked_interpolate_pp.tif'
    alignment_point_half_patch_width = 110
    alignment_points_search_width = 5
    app = QtWidgets.QApplication(sys.argv)
    window = Window(file_name, alignment_point_half_patch_width, alignment_points_search_width)
    # window.viewer.setAP(3000, 1500, 400)
    window.setGeometry(500, 300, 800, 600)
    window.showMaximized()
    sys.exit(app.exec_())
