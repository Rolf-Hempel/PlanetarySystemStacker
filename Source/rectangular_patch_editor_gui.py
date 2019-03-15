# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'rectangular_patch_editor_gui.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_rectangular_patch_editor(object):
    def setupUi(self, rectangular_patch_editor):
        rectangular_patch_editor.setObjectName("rectangular_patch_editor")
        rectangular_patch_editor.resize(900, 630)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(rectangular_patch_editor.sizePolicy().hasHeightForWidth())
        rectangular_patch_editor.setSizePolicy(sizePolicy)
        rectangular_patch_editor.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        rectangular_patch_editor.setFrameShape(QtWidgets.QFrame.Panel)
        rectangular_patch_editor.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.verticalLayout = QtWidgets.QVBoxLayout(rectangular_patch_editor)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.buttonLoad = QtWidgets.QPushButton(rectangular_patch_editor)
        self.buttonLoad.setObjectName("buttonLoad")
        self.horizontalLayout.addWidget(self.buttonLoad)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.buttonBox = QtWidgets.QDialogButtonBox(rectangular_patch_editor)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.horizontalLayout.addWidget(self.buttonBox)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(rectangular_patch_editor)
        QtCore.QMetaObject.connectSlotsByName(rectangular_patch_editor)

    def retranslateUi(self, rectangular_patch_editor):
        _translate = QtCore.QCoreApplication.translate
        rectangular_patch_editor.setWindowTitle(_translate("rectangular_patch_editor", "Frame"))
        self.buttonLoad.setText(_translate("rectangular_patch_editor", "Load image"))


