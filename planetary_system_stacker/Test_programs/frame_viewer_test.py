from sys import argv

from PyQt5 import QtCore, QtGui, QtWidgets
from cv2 import NORM_MINMAX, normalize, cvtColor, COLOR_GRAY2RGB, circle, line
from numpy import uint8, uint16

from exceptions import NotSupportedError
from frame_viewer_test_gui import Ui_Frame
from frames import Frames


class FrameViewer(QtWidgets.QGraphicsView):
    """
    This widget implements a frame viewer. Panning and zooming is implemented by using the mouse
    and scroll wheel.

    """

    def __init__(self):
        super(FrameViewer, self).__init__()
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

        # Set the focus on the viewer.
        self.setFocus()

    def resizeEvent(self, event):
        """
        This method is called when the window size changes. In this case the image is zoomed so it
        fills the entire view.

        :param event: Resize event.
        :return: -
        """

        # Set the rectangle surrounding the current view.
        self.viewrect = self.viewport().rect()
        self.fitInView()
        return super(FrameViewer, self).resizeEvent(event)

    def hasPhoto(self):
        return not self._empty

    def fitInView(self):
        """
        Scale the scene such that it fits into the window completely.

        :return: -
        """

        if not self.photorect.isNull():
            if self.hasPhoto():
                factor = min(self.viewrect.width() / self.scenerect.width(),
                             self.viewrect.height() / self.scenerect.height())
                self.scale(factor, factor)
                self.scenerect = self.transform().mapRect(self.photorect)

    def set_original_scale(self):
        """
        Scale the scene to the original size of the photo. If the photo has more pixels than the
        current view, only part of the image is displayed.

        :return: -
        """

        if not self.photorect.isNull():
            if self.hasPhoto():
                factor = min(self.photorect.width() / self.scenerect.width(),
                             self.photorect.height() / self.scenerect.height())
                self.scale(factor, factor)
                self.scenerect = self.transform().mapRect(self.photorect)

    def setPhoto(self, image, overlay_exclude_mark=False):
        """
        Convert a color or grayscale image to a pixmap and assign it to the photo object.

        :param image: Image to be displayed. The image is assumed to be in color or grayscale
                      format of length uint8 or uint16.
        :param overlay_exclude_mark: If True, overlay a crossed-out red circle in the upper left
                                     corner of the image.
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

        # Overlay a crossed-out red circle in the upper left image corner.
        if overlay_exclude_mark:
            if len(image_uint8.shape) == 2:
                image_uint8 = cvtColor(image_uint8, COLOR_GRAY2RGB)

            # The position and size of the mark are relative to the image size.
            pos_x = int(round(self.shape_x / 10))
            pos_y = int(round(self.shape_y / 10))
            diameter = int(round(min(pos_x, pos_y) / 4))
            circle(image_uint8, (pos_x, pos_y), diameter, (255, 0, 0), 2)
            line(image_uint8, (pos_x - diameter, pos_y + diameter),
                 (pos_x + diameter, pos_y - diameter), (255, 0, 0), 2)

        # The image is monochrome:
        if len(image_uint8.shape) == 2:
            qt_image = QtGui.QImage(image_uint8, self.shape_x, self.shape_y, self.shape_x,
                                    QtGui.QImage.Format_Grayscale8)
        # The image is RGB color.
        else:
            qt_image = QtGui.QImage(image_uint8, self.shape_x,
                                    self.shape_y, 3 * self.shape_x, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(qt_image)

        if pixmap and not pixmap.isNull():
            self._empty = False
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self._photo.setPixmap(QtGui.QPixmap())

        # Set the new rectangles surrounding the photo and the current scene.
        self.photorect = QtCore.QRectF(self._photo.pixmap().rect())
        self.scenerect = self.transform().mapRect(self.photorect)

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
            print ("Wheel event")

        # If not in drag mode, the wheel event is handled at the scene level.
        else:
            self._scene.wheelEvent(event)

    def zoom(self, direction):
        """
        Zoom in or out. This is only active when a photo is loaded. For small images (smaller than
        the current view), zooming out is limited by the original photo resolution. For larger
        images zooming out stops when the scene fills the entire view.

        :param direction: If > 0, zoom in, otherwise zoom out.
        :return: -
        """

        if self.hasPhoto():
            # Depending of direction value, set the zoom factor to greater or smaller than 1.
            if direction > 0:
                factor = 1.25
            else:
                min_factor = min(
                    min(self.photorect.width(), self.viewrect.width()) / self.scenerect.width(),
                    min(self.photorect.height(), self.viewrect.height()) / self.scenerect.height())
                factor = max(0.8, min_factor)

            self.scale(factor, factor)
            self.scenerect = self.transform().mapRect(self.photorect)

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
        elif event.key() == QtCore.Qt.Key_1 and not event.modifiers() & QtCore.Qt.ControlModifier:
            self.set_original_scale()
        else:
            super(FrameViewer, self).keyPressEvent(event)


class FrameViewerWidget(QtWidgets.QFrame, Ui_Frame):

    def __init__(self, input_image):
        super(FrameViewerWidget, self).__init__()
        self.setupUi(self)
        frame_viewer = FrameViewer()
        frame_viewer.setObjectName("frame_viewer")
        frame_viewer.setFrameShape(QtWidgets.QFrame.Panel)
        frame_viewer.setFrameShadow(QtWidgets.QFrame.Sunken)
        frame_viewer.setMinimumSize(600, 600)
        frame_viewer.setPhoto(input_image)
        self.gridLayout.addWidget(frame_viewer, 0, 0, 1, 1)


if __name__ == '__main__':
    # input_file_name = "D:\SW-Development\Python\PlanetarySystemStacker\Examples\Jupiter_Richard\\2020-07-29-2145_3-L-Jupiter_ALTAIRGP224C_pss_gpp.png"
    input_file_name = "D:\SW-Development\Python\PlanetarySystemStacker\Examples\Jupiter_Richard\\" \
                      "2020-07-29-2145_3-L-Jupiter_ALTAIRGP224C_pss_p70_b48.png"
    # input_file_name = "D:\SW-Development\Python\PlanetarySystemStacker\Examples\Moon_2018-03-24\Moon_Tile-024_043939_pss_drizzle2_gpp.png"
    input_image = Frames.read_image(input_file_name)

    app = QtWidgets.QApplication(argv)

    window = FrameViewerWidget(input_image)
    window.showMaximized()
    app.exec_()
