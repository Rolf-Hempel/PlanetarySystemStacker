# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'rectangular_patch_editor_gui.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_rectangular_patch_editor(object):
    def setupUi(self, rectangular_patch_editor):
        rectangular_patch_editor.setObjectName("rectangular_patch_editor")
        rectangular_patch_editor.resize(900, 630)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(rectangular_patch_editor.sizePolicy().hasHeightForWidth())
        rectangular_patch_editor.setSizePolicy(sizePolicy)
        rectangular_patch_editor.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates))
        rectangular_patch_editor.setFrameShape(QtWidgets.QFrame.Shape.Panel)
        rectangular_patch_editor.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.verticalLayout = QtWidgets.QVBoxLayout(rectangular_patch_editor)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.messageLabel = QtWidgets.QLabel(rectangular_patch_editor)
        self.messageLabel.setObjectName("messageLabel")
        self.horizontalLayout.addWidget(self.messageLabel)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.buttonBox = QtWidgets.QDialogButtonBox(rectangular_patch_editor)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.horizontalLayout.addWidget(self.buttonBox)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(rectangular_patch_editor)
        QtCore.QMetaObject.connectSlotsByName(rectangular_patch_editor)

    def retranslateUi(self, rectangular_patch_editor):
        _translate = QtCore.QCoreApplication.translate
        rectangular_patch_editor.setWindowTitle(_translate("rectangular_patch_editor", "Frame"))
        self.messageLabel.setText(_translate("rectangular_patch_editor", "Message to be displayed"))
        self.buttonBox.setToolTip(_translate("rectangular_patch_editor", "Press \'OK\' to save the selection and exit, or \'cancel\' to discard the selection."))
