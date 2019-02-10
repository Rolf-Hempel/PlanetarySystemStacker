# Copied from https://stackoverflow.com/questions/35508711/how-to-enable-pan-and-zoom-in-a-qgraphicsview

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


class GraphicsScene(QtWidgets.QGraphicsScene):

    def __init__(self, photo_viewer, parent=None):
        QtWidgets.QGraphicsScene.__init__(self, parent)
        self.photo_viewer = photo_viewer
        self.left_button_pressed = False
        self.right_button_pressed = False
        self.single_click_threshold = 5
        self.max_match_distance = 5
        self.move_ap = None

    def mousePressEvent(self, event):
        pos = event.lastScenePos()
        x = pos.x()
        y = pos.y()
        if event.button() == QtCore.Qt.LeftButton:
            # print ("Left press, x: " + str(x) + ", y: " + str(y))
            self.left_button_pressed = True
            if not self.photo_viewer.drag_mode:
                neighbor_ap, distance = self.photo_viewer.aps.find_neighbor(y, x)
                if distance < self.max_match_distance:
                    self.remember_ap = neighbor_ap
                    self.move_ap = True
                else:
                    half_patch_width_new = min(self.photo_viewer.aps.half_patch_width, y,
                                               self.photo_viewer.aps.shape_y - y, x,
                                               self.photo_viewer.aps.shape_x - x)
                    self.remember_ap = self.photo_viewer.aps.new_alignment_point(y, x,
                                       self.photo_viewer.aps.half_box_width, half_patch_width_new)
                    self.photo_viewer.draw_alignment_point(self.remember_ap)
                    self.left_y_start = y
                    self.left_x_start = x
                    self.move_ap = False

        elif event.button() == QtCore.Qt.RightButton:
            # print("Right press, x: " + str(x) + ", y: " + str(y))
            self.right_button_pressed = True
            if not self.photo_viewer.drag_mode:
                x = pos.x()
                y = pos.y()
                self.right_y_start = y
                self.right_x_start = x
                self.remember_sr = None

    def mouseReleaseEvent(self, event):
        pos = event.lastScenePos()
        if event.button() == QtCore.Qt.RightButton:
            self.right_button_pressed = False
            if not self.photo_viewer.drag_mode:
                x = pos.x()
                y = pos.y()
                if max(abs(y - self.right_y_start),
                       abs(x - self.right_x_start)) < self.single_click_threshold:
                    ap, dist = self.photo_viewer.aps.find_neighbor(y, x)
                    self.removeItem(ap['graphics_item'])
                    self.photo_viewer.aps.remove_alignment_points([ap])
                else:
                    remove_ap_list = []
                    y_low = min(self.right_y_start, self.right_y_end)
                    y_high = max(self.right_y_start, self.right_y_end)
                    x_low = min(self.right_x_start, self.right_x_end)
                    x_high = max(self.right_x_start, self.right_x_end)
                    for ap in self.photo_viewer.aps.alignment_points:
                        if y_low < ap['y'] < y_high and x_low < ap['x'] < x_high:
                            remove_ap_list.append(ap)
                            self.removeItem(ap['graphics_item'])
                    self.removeItem(self.remember_sr)
                    self.photo_viewer.aps.remove_alignment_points(remove_ap_list)

        elif event.button() == QtCore.Qt.LeftButton:
            self.left_button_pressed = False
            if not self.photo_viewer.drag_mode:
                if not self.move_ap:
                    self.photo_viewer.aps.alignment_points.append(self.remember_ap)

    def mouseMoveEvent(self, event):
        if self.left_button_pressed:
            pos = event.lastScenePos()
            if not self.photo_viewer.drag_mode:
                x_new = pos.x()
                y_new = pos.y()
                self.removeItem(self.remember_ap['graphics_item'])
                if self.move_ap:
                    new_ap = self.photo_viewer.aps.move_alignment_point(self.remember_ap, y_new,
                                                                        x_new)
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
                self.photo_viewer.draw_alignment_point(new_ap)
                self.remember_ap = new_ap

        elif self.right_button_pressed:
            pos = event.lastScenePos()
            x = pos.x()
            y = pos.y()
            # print("Right move, x: " + str(x) + ", y: " + str(y))
            if not self.photo_viewer.drag_mode:
                x = pos.x()
                y = pos.y()
                self.right_y_end = y
                self.right_x_end = x
                new_sr = SelectionRectangleGraphicsItem(
                    self.right_y_start, self.right_x_start, y, x)
                if self.remember_sr is not None:
                    self.removeItem(self.remember_sr)
                self.addItem(new_sr)
                self.remember_sr = new_sr


class AlignmentPointGraphicsItem(QtWidgets.QGraphicsItem):
    def __init__(self, ap):
        super(AlignmentPointGraphicsItem, self).__init__()
        self.dot_width = 4
        self.color_surface = QtGui.QColor(0, 255, 0, 20)
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
    def __init__(self, y_1, x_1, y_2, x_2):
        super(SelectionRectangleGraphicsItem, self).__init__()
        self.color_surface = QtGui.QColor(0, 0, 255, 40)
        self.color_boundary = QtGui.QColor(255, 0, 0)
        self.y_low = min(y_1, y_2)
        self.x_low = min(x_1, x_2)
        self.y_high = max(y_1, y_2)
        self.x_high = max(x_1, x_2)
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

    def __init__(self, parent):
        super(AlignmentPointViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = GraphicsScene(self, self)
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
        self.aps = None

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
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
        if pixmap and not pixmap.isNull():
            self._empty = False
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self._photo.setPixmap(QtGui.QPixmap())

    def set_alignment_points(self, aps):
        self.aps = aps
        for ap in self.aps.alignment_points:
            self.draw_alignment_point(ap)

    def draw_alignment_point(self, ap):
        ap_graphics_item = AlignmentPointGraphicsItem(ap)
        self._scene.addItem(ap_graphics_item)
        return ap_graphics_item

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Control:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self.drag_mode = False
        else:
            super(AlignmentPointViewer, self).keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key_Control:
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self.drag_mode = True
        else:
            super(AlignmentPointViewer, self).keyPressEvent(event)


class AlignmentPoints(object):
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
    alignment_point_half_patch_width = 110
    alignment_points_search_width = 10
    app = QtWidgets.QApplication(sys.argv)
    window = Window(file_name, alignment_point_half_patch_width, alignment_points_search_width)
    # window.viewer.setAP(3000, 1500, 400)
    window.setGeometry(500, 300, 800, 600)
    window.show()
    sys.exit(app.exec_())
