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

"""

from sys import argv
from time import sleep
from os.path import splitext
from cv2 import imread, cvtColor, COLOR_BGR2GRAY
from copy import deepcopy
from numpy import uint8, uint16

from PyQt5 import QtWidgets, QtCore
from sharpening_layer_widget import Ui_sharpening_layer_widget
from version_manager_widget import Ui_version_manager_widget
from postproc_editor_gui import Ui_postproc_editor
from frame_viewer import FrameViewer
from frames import Frames

class DataObject(object):
    def __init__(self, image_original, name_original, suffix, blinking_period, idle_loop_time):
        self.image_original = image_original
        self.color = len(self.image_original.shape) == 3
        self.file_name_original = name_original
        self.file_name_processed = splitext(name_original)[0] + suffix + '.tiff'
        self.versions = [Version(self.image_original)]
        self.number_versions = 0
        self.version_selected = 0
        self.version_compared = 0
        self.blinking = False
        self.blinking_period = blinking_period
        self.idle_loop_time = idle_loop_time

        initial_version = Version(self.image_original)
        # initial_version = Version((self.image_original / 2.).astype(uint16))
        initial_version.add_layer(Layer(1., 0, False))
        self.add_version(initial_version)

    def add_version(self, version):
        self.versions.append(version)
        self.number_versions += 1
        self.version_selected = self.number_versions

    def remove_version(self, index):
        if 0 < index <= self.number_versions:
            self.versions = self.versions[:index] + self.versions[index + 1:]
            self.number_versions -= 1
            self.version_selected = index -1

class Version(object):
    def __init__(self, image):
        self.image = image
        self.layers = []
        self.number_layers = 0

    def add_layer(self, layer):
        # for index, layer_compare in enumerate(self.layers):
        #     if layer_compare.radius >= layer.radius:
        #         self.layers.insert(index, layer)
        #         return
        self.layers.append(layer)
        self.number_layers += 1

    def remove_layer(self, layer_index):
        if 0 <= layer_index < self.number_layers:
            self.layers = self.layers[:layer_index] + self.layers[layer_index+1:]
            self.number_layers -= 1

class Layer(object):
    def __init__(self, radius, amount, luminance_only):
        self.radius = radius
        self.amount = amount
        self.luminance_only = luminance_only


class SharpeningLayerWidget(QtWidgets.QWidget, Ui_sharpening_layer_widget):
    def __init__(self, layer_index, remove_layer_callback, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)

        self.layer_index = layer_index
        self.title = "Layer " + str(layer_index + 1)
        self.remove_layer_callback = remove_layer_callback

        self.horizontalSlider_radius.valueChanged.connect(self.horizontalSlider_radius_changed)
        self.lineEdit_radius.textChanged.connect(self.lineEdit_radius_changed)
        self.horizontalSlider_amount.valueChanged.connect(self.horizontalSlider_amount_changed)
        self.lineEdit_amount.textChanged.connect(self.lineEdit_amount_changed)
        self.checkBox_luminance.stateChanged.connect(self.checkBox_luminance_toggled)
        self.pushButton_remove.clicked.connect(self.remove_layer)

    def set_values(self, layer):
        self.layer = layer

        self.groupBox_layer.setTitle(self.title)
        self.horizontalSlider_radius.setValue(self.radius_to_int(self.layer.radius))
        self.lineEdit_radius.setText(str(self.layer.radius))
        self.horizontalSlider_amount.setValue(int(round(self.layer.amount)))
        self.lineEdit_amount.setText(str(self.layer.amount))
        self.checkBox_luminance.setChecked(self.layer.luminance_only)

    def radius_to_int(self, radius):
        return max(min(int(round(radius*10.)), 99), 1)

    def int_to_radius(self, int):
        return int / 10.

    def horizontalSlider_radius_changed(self):
        self.layer.radius = self.int_to_radius(self.horizontalSlider_radius.value())
        self.lineEdit_radius.blockSignals(True)
        self.lineEdit_radius.setText(str(self.layer.radius))
        self.lineEdit_radius.blockSignals(False)

    def lineEdit_radius_changed(self):
        try:
            self.layer.radius = max(0.1, min(float(self.lineEdit_radius.text()), 9.9))
            self.horizontalSlider_radius.blockSignals(True)
            self.horizontalSlider_radius.setValue(self.radius_to_int(self.layer.radius))
            self.horizontalSlider_radius.blockSignals(False)
        except:
            pass

    def horizontalSlider_amount_changed(self):
        self.layer.amount = self.horizontalSlider_amount.value()
        self.lineEdit_amount.blockSignals(True)
        self.lineEdit_amount.setText(str(self.layer.amount))
        self.lineEdit_amount.blockSignals(False)

    def lineEdit_amount_changed(self):
        try:
            self.layer.amount = max(0, min(int(round(float(self.lineEdit_amount.text()))), 200))
        except:
            pass
        self.horizontalSlider_amount.blockSignals(True)
        self.horizontalSlider_amount.setValue(self.layer.amount)
        self.horizontalSlider_amount.blockSignals(False)

    def checkBox_luminance_toggled(self):
        self.layer.luminance_only = not self.layer.luminance_only

    def remove_layer(self):
        self.remove_layer_callback(self.layer_index)


class VersionManagerWidget(QtWidgets.QWidget, Ui_version_manager_widget):

    set_photo_signal = QtCore.pyqtSignal(int)
    variant_shown_signal = QtCore.pyqtSignal(bool, bool)

    def __init__(self, data_object, select_version_callback, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)
        self.data_object = data_object
        self.select_version_callback = select_version_callback

        self.spinBox_version.valueChanged.connect(self.select_version)
        self.spinBox_compare.valueChanged.connect(self.select_version_compared)
        self.pushButton_new.clicked.connect(self.new_version)
        self.pushButton_delete.clicked.connect(self.remove_version)
        self.checkBox_blink_compare.stateChanged.connect(self.blinking_toggled)
        self.pushButton_save.clicked.connect(self.save_version)
        self.pushButton_save_as.clicked.connect(self.save_version_as)
        self.variant_shown_signal.connect(self.highlight_variant)

        self.spinBox_version.setMaximum(1)
        self.spinBox_version.setMinimum(0)
        self.spinBox_version.setValue(self.data_object.version_selected)
        self.spinBox_compare.setMaximum(1)
        self.spinBox_compare.setMinimum(0)

    def select_version(self):
        self.data_object.version_selected = self.spinBox_version.value()
        self.select_version_callback(self.data_object.version_selected)

    def select_version_compared(self):
        self.data_object.version_compared = self.spinBox_compare.value()

    def new_version(self):
        new_version = Version(self.data_object.image_original)
        new_version.add_layer(Layer(1., 0, False))
        self.data_object.add_version(new_version)
        # print ("new version")
        self.set_photo_signal.emit(self.data_object.version_selected)
        self.spinBox_version.setMaximum(self.data_object.number_versions)
        self.spinBox_version.setValue(self.data_object.number_versions)
        self.spinBox_compare.setMaximum(self.data_object.number_versions)
        # self.select_version_callback(self.data_object.version_selected)

    def remove_version(self):
        self.data_object.remove_version(self.data_object.version_selected)
        self.set_photo_signal.emit(self.data_object.version_selected)
        self.spinBox_version.setMaximum(self.data_object.number_versions)
        self.spinBox_compare.setMaximum(self.data_object.number_versions)
        self.select_version_callback(self.data_object.version_selected)

    def blinking_toggled(self):
        self.data_object.blinking = not self.data_object.blinking
        if self.data_object.blinking:
            # Create the blink comparator thread and start it.
            self.blink_comparator = BlinkComparator(self.data_object, self.set_photo_signal,
                                                    self.variant_shown_signal)
            self.blink_comparator.setTerminationEnabled(True)
        else:
            self.set_photo_signal.emit(self.data_object.version_selected)
            self.blink_comparator.stop()

    def highlight_variant(self, selected, compare):
        if selected and compare:
            self.spinBox_version.setStyleSheet('color: red')
            self.spinBox_compare.setStyleSheet('color: red')
        elif not selected and not compare:
            self.spinBox_version.setStyleSheet('color: black')
            self.spinBox_compare.setStyleSheet('color: black')
        elif selected:
            self.spinBox_version.setStyleSheet('color: red')
            self.spinBox_compare.setStyleSheet('color: white')
        elif compare:
            self.spinBox_version.setStyleSheet('color: white')
            self.spinBox_compare.setStyleSheet('color: red')


    def save_version(self):
        """
        save the result as 16bit Tiff at the standard location.

        :return: -
        """

        Frames.save_image(self.data_object.file_name_processed,
                          self.data_object.versions[self.data_object.version_selected].image,
                          color=self.data_object.color, avoid_overwriting=False)

    def save_version_as(self):
        """
        save the result as 16bit Tiff at a location selected by the user.

        :return: -
        """

        options = QtWidgets.QFileDialog.Options()
        filename, extension = QtWidgets.QFileDialog.getSaveFileName(self,
            "Save result as 16bit Tiff image", self.data_object.file_name_original ,
            "Image Files (*.tiff)", options=options)

        if filename and extension:
            Frames.save_image(filename,
                              self.data_object.versions[self.data_object.version_selected].image,
                              color=self.data_object.color, avoid_overwriting=False)


class BlinkComparator(QtCore.QThread):

    def __init__(self, data_object, set_photo_signal, variant_shown_signal, parent=None):
        """
        Show two versions of the image in the image viewer alternately.

        :param data_object: Data object with postprocessing data
        """

        QtCore.QThread.__init__(self, parent)
        self.data_object = data_object
        self.set_photo_signal = set_photo_signal
        self.variant_shown_signal = variant_shown_signal
        self.variant_shown_signal.emit(False, False)

        self.start()

    def run(self):
        show_selected_version = True
        while self.data_object.blinking:
            if show_selected_version:
                self.set_photo_signal.emit(self.data_object.version_selected)
                self.variant_shown_signal.emit(True, False)
            else:
                self.set_photo_signal.emit(self.data_object.version_compared)
                self.variant_shown_signal.emit(False, True)
            # Toggle back and forth between first and second image version.
            show_selected_version = not show_selected_version
            # Sleep time inserted to limit CPU consumption by idle looping.
            sleep(self.data_object.blinking_period)

    def stop(self):
        self.variant_shown_signal.emit(False, False)
        self.terminate()


class ImageProcessor(QtCore.QThread):

    set_photo_signal = QtCore.pyqtSignal(int)

    def __init__(self, data_object, parent=None):
        """
        Whenever the parameters of the current version change, compute a new sharpened image.

        :param data_object: Data object with postprocessing data
        """

        QtCore.QThread.__init__(self, parent)
        self.data_object = data_object

        self.last_version_selected = 1
        self.last_layers = [Layer(1., 0, False)]
        self.version_selected = None
        self.layers_selected = None

        self.start()

    def run(self):
        while True:
            self.version_selected = self.data_object.version_selected
            self.layers_selected = deepcopy(self.data_object.versions[
                self.data_object.version_selected].layers)
            if self.start_new_computation() and self.version_selected:

                # Insert computation of a new image here.

                self.set_photo_signal.emit(self.version_selected)
                self.last_version_selected = self.version_selected
                self.last_layers = self.layers_selected
            else:
                sleep(self.data_object.idle_loop_time)

    def start_new_computation(self):
        if self.last_version_selected != self.version_selected:
            return True

        if len(self.last_layers) != len(self.layers_selected):
            return True

        for last_layer, layer_selected in zip(self.last_layers, self.layers_selected):
            if last_layer.radius != layer_selected.radius or last_layer.amount != layer_selected.amount or \
                last_layer.luminance_only != layer_selected.luminance_only:
                # print("layer_selected.radius: " + str(layer_selected.radius))
                return True

        return False

    def stop(self):
        self.terminate()


class PostprocEditorWidget(QtWidgets.QFrame, Ui_postproc_editor):
    """
    This widget implements a frame viewer together with control elements to control the
    postprocessing. Several postprocessing versions can be created and managed, each one using
    up to four sharpening layers.
    """

    def __init__(self, image_original, name_original, suffix, blinking_period, idle_loop_time):
        super(PostprocEditorWidget, self).__init__()
        self.setupUi(self)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.pushButton_add_layer.clicked.connect(self.add_layer)

        self.frame_viewer = FrameViewer()
        self.frame_viewer.setObjectName("framewiever")
        self.gridLayout.addWidget(self.frame_viewer, 0, 0, 7, 1)
        self.data_object = DataObject(image_original, name_original, suffix, blinking_period,
                                      idle_loop_time)

        self.sharpening_layer_widgets = []
        self.max_layers = 4
        for layer in range(self.max_layers):
            sharpening_layer_widget = SharpeningLayerWidget(layer, self.remove_layer)
            self.gridLayout.addWidget(self.frame_viewer, 0, 0, 7, 1)
            self.gridLayout.addWidget(sharpening_layer_widget, layer+1, 1, 1, 1)
            self.sharpening_layer_widgets.append(sharpening_layer_widget)

        self.version_manager_widget = VersionManagerWidget(self.data_object, self.select_version)
        self.gridLayout.addWidget(self.version_manager_widget, 6, 1, 1, 1)
        self.version_manager_widget.set_photo_signal.connect(self.select_image)

        self.select_version(self.data_object.version_selected)

        self.image_processor = ImageProcessor(self.data_object)
        self.image_processor.setTerminationEnabled(True)
        self.image_processor.set_photo_signal.connect(self.select_image)

    def select_version(self, version_index):
        version_selected = self.data_object.versions[version_index]
        for layer_index, layer in enumerate(version_selected.layers):
            self.sharpening_layer_widgets[layer_index].set_values(layer)
            self.sharpening_layer_widgets[layer_index].setHidden(False)
        for layer_index in range(version_selected.number_layers, self.max_layers):
            self.sharpening_layer_widgets[layer_index].setHidden(True)
        self.select_image(version_index)

    def select_image(self, version_index):
        # print ("set Photo")
        self.frame_viewer.setPhoto(self.data_object.versions[version_index].image)

    def add_layer(self):
        version_selected = self.data_object.versions[self.data_object.version_selected]
        num_layers_current = version_selected.number_layers
        if self.data_object.version_selected and num_layers_current < self.max_layers:
            if num_layers_current > 0:
                previous_layer = version_selected.layers[num_layers_current-1]
                new_layer = Layer(2.*previous_layer.radius, 0, previous_layer.luminance_only)
            else:
                new_layer = Layer(1., 0, False)
            version_selected.add_layer(new_layer)
            self.select_version(self.data_object.version_selected)

    def remove_layer(self, layer_index):
        version_selected = self.data_object.versions[self.data_object.version_selected]
        version_selected.remove_layer(layer_index)
        self.select_version(self.data_object.version_selected)

    def accept(self):
        self.image_processor.stop()
        self.close()

    def reject(self):
        self.image_processor.stop()
        self.close()


if __name__ == '__main__':

    input_file_name = "D:\SW-Development\Python\PlanetarySystemStacker\Examples\Moon_2018-03-24\Moon_Tile-024_043939_pss.tiff"
    # input_image = cvtColor(imread(input_file_name), COLOR_BGR2GRAY)
    input_image = imread(input_file_name, -1)
    data_object = DataObject(input_image, input_file_name, "_gpp", 1., 0.1)
    for i in range(3):
        data_object.add_version(Version("image_" + str(i)))

    version_selected = data_object.versions[2]
    for i in range(4):
        version_selected.add_layer(Layer(i, 100+i, True))

    version_selected.remove_layer(2)

    version_selected.add_layer(Layer(30, 1000, False))

    new_layer = Layer(3.5, 65, True)

    # app = QtWidgets.QApplication(argv)
    # window = SharpeningLayerWidget("Layer 4")
    # window.show()
    # app.exec_()

    print("radius: " + str(new_layer.radius))
    print("amount: " + str(new_layer.amount))
    print("luminance only: " + str(new_layer.luminance_only))

    app = QtWidgets.QApplication(argv)
    window = PostprocEditorWidget(input_image, input_file_name, "_gpp", 1., 0.5)
    window.show()
    app.exec_()
