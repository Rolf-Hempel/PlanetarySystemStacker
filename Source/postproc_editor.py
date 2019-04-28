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
from cv2 import imread

from PyQt5 import QtWidgets, QtCore
from sharpening_layer_widget import Ui_sharpening_layer_widget
from version_manager_widget import Ui_version_manager_widget
from postproc_editor_gui import Ui_postproc_editor
from frame_viewer import FrameViewer
from frames import Frames

class DataObject(object):
    def __init__(self, image_original, name_original, suffix, blinking_period, func_display_image):
        self.image_original = image_original
        self.color = len(self.image_original.shape) == 3
        self.file_name_original = name_original
        self.file_name_processed = splitext(name_original)[0] + suffix + '.tiff'
        self.versions = [Version(image_original)]
        self.number_versions = 0
        self.version_selected = 0
        self.version_compared = 0
        self.blinking = False
        self.blinking_period = blinking_period
        self.func_display_image = func_display_image

    def add_version(self, version):
        self.versions.append(version)
        self.number_versions += 1

    def remove_version(self, index):
        if 0 < index <= self.number_versions:
            self.versions = self.versions[:index] + self.versions[index + 1:]
            self.number_versions -= 1

class Version(object):
    def __init__(self, image):
        self.image = image
        self.layers = []
        self.number_layers = 0

    def add_layer(self, layer):
        for index, layer_compare in enumerate(self.layers):
            if layer_compare.radius >= layer.radius:
                self.layers.insert(index, layer)
                return
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
    def __init__(self, title, layer, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)

        self.title = title
        self.set_values(layer)

        self.horizontalSlider_radius.valueChanged.connect(self.horizontalSlider_radius_changed)
        self.lineEdit_radius.textChanged.connect(self.lineEdit_radius_changed)
        self.horizontalSlider_amount.valueChanged.connect(self.horizontalSlider_amount_changed)
        self.lineEdit_amount.textChanged.connect(self.lineEdit_amount_changed)
        self.checkBox_luminance.stateChanged.connect(self.checkBox_luminance_toggled)

    def set_values(self, layer):
        self.layer = layer

        self.groupBox_layer.setTitle(self.title)
        self.horizontalSlider_radius.setValue(self.radius_to_int(self.layer.radius))
        self.lineEdit_radius.setText(str(self.layer.radius))
        self.horizontalSlider_amount.setValue(self.layer.amount)
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


class VersionManagerWidget(QtWidgets.QWidget, Ui_version_manager_widget):
    def __init__(self, data_object, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)
        self.data_object = data_object

        self.spinBox_version.valueChanged.connect(self.select_version)
        self.spinBox_compare.valueChanged.connect(self.select_version_compared)
        self.pushButton_new.clicked.connect(self.new_version)
        self.pushButton_delete.clicked.connect(self.remove_version)
        self.checkBox_blink_compare.stateChanged.connect(self.blinking_toggled)
        self.pushButton_save.clicked.connect(self.save_version)
        self.pushButton_save_as.clicked.connect(self.save_version_as)

    def select_version(self):
        self.data_object.version_selected = self.spinBox_version.value()

    def select_version_compared(self):
        self.data_object.version_compared = self.spinBox_compare.value()

    def new_version(self):
        self.data_object.add_version(Version(self.data_object.image_original))

    def remove_version(self):
        self.data_object.remove_version(self.data_object.version_selected)

    def blinking_toggled(self):
        self.data_object.blinking = not self.data_object.blinking
        if self.data_object.blinking:
            # Create the blink comparator thread and start it.
            self.blink_comparator = BlinkComparator(self.data_object)
            self.blink_comparator.setTerminationEnabled(True)
        else:
            self.blink_comparator.stop()

    def save_version(self):
        """
        save the result as 16bit Tiff at the standard location.

        :return: -
        """

        Frames.save_image(self.file_name_processed, self.data_object.version_selected.image,
                                        color=self.color, avoid_overwriting=False)

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
            Frames.save_image(filename, self.data_object.version_selected.image,
                                color=self.color, avoid_overwriting=False)


class BlinkComparator(QtCore.QThread):

    def __init__(self, data_object, parent=None):
        """
        Show two versions of the image in the image viewer alternately.

        :param data_object: Data object with postprocessing data
        """

        QtCore.QThread.__init__(self, parent)
        self.data_object = data_object

        self.start()

    def run(self):
        show_selected_version = True
        while self.data_object.blinking:
            if show_selected_version:
                self.data_object.func_display_image(self.data_object.version_selected.image)
            else:
                self.data_object.func_display_image(self.data_object.version_compared.image)
            # Toggle back and forth between first and second image version.
            show_selected_version = not show_selected_version
            # Sleep time inserted to limit CPU consumption by idle looping.
            sleep(self.data_object.blinking_period)

    def stop(self):
        self.terminate()


class PostprocEditorWidget(QtWidgets.QFrame, Ui_postproc_editor):
    """
    This widget implements a frame viewer together with control elements to control the
    postprocessing. Several postprocessing versions can be created and managed, each one using
    up to four sharpening layers.
    """

    def __init__(self, image_original, name_original, suffix, blinking_period):
        super(PostprocEditorWidget, self).__init__()
        self.setupUi(self)
        self.frame_viewer = FrameViewer()
        self.data_object = DataObject(image_original, name_original, suffix, blinking_period,
                                      self.frame_viewer.setPhoto)
        self.frame_viewer.setObjectName("framewiever")
        self.gridLayout.addWidget(self.frame_viewer, 0, 0, 7, 1)



if __name__ == '__main__':

    input_file_name = "D:\SW-Development\Python\PlanetarySystemStacker\Examples\Moon_2018-03-24\Moon_Tile-024_043939_pss.tiff"
    input_image = imread(input_file_name)
    data_object = DataObject(input_image, input_file_name, "_gpp", 1., None)
    for i in range(3):
        data_object.add_version(Version("image_" + str(i)))

    version_selected = data_object.versions[2]
    for i in range(4):
        version_selected.add_layer(Layer(i, 100+i, True))

    version_selected.remove_layer(2)

    version_selected.add_layer(Layer(30, 1000, False))

    new_layer = Layer(3.5, 65, True)

    app = QtWidgets.QApplication(argv)
    window = SharpeningLayerWidget("Layer 4", new_layer)
    window.show()
    app.exec_()

    print("radius: " + str(new_layer.radius))
    print("amount: " + str(new_layer.amount))
    print("luminance only: " + str(new_layer.luminance_only))