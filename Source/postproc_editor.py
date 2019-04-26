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

from PyQt5 import QtWidgets
from sharpening_level_widget import Ui_sharpening_level_widget
from version_manager_widget import Ui_version_manager_widget

class DataObject(object):
    def __init__(self, image_original, name, suffix):
        self.versions = [Version(image_original)]
        self.name = name
        self.suffix = suffix

    def add_version(self, version):
        self.versions.append(version)

    def remove_version(self, index):
        self.versions = self.versions[:index] + self.versions[index + 1:]

class Version(object):
    def __init__(self, image):
        self.image = image
        self.levels = []

    def add_level(self, level):
        for index, level_compare in enumerate(self.levels):
            if level_compare.radius >= level.radius:
                self.levels.insert(index, level)
                return
        self.levels.append(level)

    def remove_level(self, level_index):
        self.levels = self.levels[:level_index] + self.levels[level_index+1:]

class Level(object):
    def __init__(self, radius, amount, luminance_only):
        self.radius = radius
        self.amount = amount
        self.luminance_only = luminance_only


class SharpeningLevelWidget(QtWidgets.QWidget, Ui_sharpening_level_widget):
    def __init__(self, title, level, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)

        self.title = title
        self.set_values(level)

        self.horizontalSlider_radius.valueChanged.connect(self.horizontalSlider_radius_changed)
        self.lineEdit_radius.textChanged.connect(self.lineEdit_radius_changed)
        self.horizontalSlider_amount.valueChanged.connect(self.horizontalSlider_amount_changed)
        self.lineEdit_amount.textChanged.connect(self.lineEdit_amount_changed)
        self.checkBox_luminance.stateChanged.connect(self.checkBox_luminance_toggled)

    def set_values(self, level):
        self.level = level

        self.groupBox_level.setTitle(self.title)
        self.horizontalSlider_radius.setValue(self.radius_to_int(self.level.radius))
        self.lineEdit_radius.setText(str(self.level.radius))
        self.horizontalSlider_amount.setValue(self.level.amount)
        self.lineEdit_amount.setText(str(self.level.amount))
        self.checkBox_luminance.setChecked(self.level.luminance_only)

    def radius_to_int(self, radius):
        return max(min(int(round(radius*10.)), 99), 1)

    def int_to_radius(self, int):
        return int / 10.

    def horizontalSlider_radius_changed(self):
        self.level.radius = self.int_to_radius(self.horizontalSlider_radius.value())
        self.lineEdit_radius.blockSignals(True)
        self.lineEdit_radius.setText(str(self.level.radius))
        self.lineEdit_radius.blockSignals(False)

    def lineEdit_radius_changed(self):
        try:
            self.level.radius = max(0.1, min(float(self.lineEdit_radius.text()), 9.9))
            self.horizontalSlider_radius.blockSignals(True)
            self.horizontalSlider_radius.setValue(self.radius_to_int(self.level.radius))
            self.horizontalSlider_radius.blockSignals(False)
        except:
            pass

    def horizontalSlider_amount_changed(self):
        self.level.amount = self.horizontalSlider_amount.value()
        self.lineEdit_amount.blockSignals(True)
        self.lineEdit_amount.setText(str(self.level.amount))
        self.lineEdit_amount.blockSignals(False)

    def lineEdit_amount_changed(self):
        try:
            self.level.amount = max(0, min(int(round(float(self.lineEdit_amount.text()))), 200))
        except:
            pass
        self.horizontalSlider_amount.blockSignals(True)
        self.horizontalSlider_amount.setValue(self.level.amount)
        self.horizontalSlider_amount.blockSignals(False)

    def checkBox_luminance_toggled(self):
        self.level.luminance_only = not self.level.luminance_only


class VersionManagerWidget(QtWidgets.QWidget, Ui_version_manager_widget):
    def __init__(self, data_object, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.setupUi(self)
        self.data_object = data_object


if __name__ == '__main__':

    data_object = DataObject("image", "filename", "_suffix")
    for i in range(3):
        data_object.add_version(Version("image_" + str(i)))

    version_selected = data_object.versions[2]
    for i in range(4):
        version_selected.add_level(Level(i, 100+i, True))

    version_selected.remove_level(2)

    version_selected.add_level(Level(30, 1000, False))

    new_level = Level(3.5, 65, True)

    app = QtWidgets.QApplication(argv)
    window = SharpeningLevelWidget("Level 4", new_level)
    window.show()
    app.exec_()

    print("radius: " + str(new_level.radius))
    print("amount: " + str(new_level.amount))
    print("luminance only: " + str(new_level.luminance_only))