# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'sharpening_level_widget.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_sharpening_level_widget(object):
    def setupUi(self, sharpening_level_widget):
        sharpening_level_widget.setObjectName("sharpening_level_widget")
        sharpening_level_widget.resize(300, 137)
        self.verticalLayout = QtWidgets.QVBoxLayout(sharpening_level_widget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_level = QtWidgets.QGroupBox(sharpening_level_widget)
        self.groupBox_level.setObjectName("groupBox_level")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_level)
        self.gridLayout.setHorizontalSpacing(12)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalSlider_amount = QtWidgets.QSlider(self.groupBox_level)
        self.horizontalSlider_amount.setMaximum(200)
        self.horizontalSlider_amount.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_amount.setObjectName("horizontalSlider_amount")
        self.gridLayout.addWidget(self.horizontalSlider_amount, 1, 1, 1, 1)
        self.lineEdit_amount = QtWidgets.QLineEdit(self.groupBox_level)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_amount.sizePolicy().hasHeightForWidth())
        self.lineEdit_amount.setSizePolicy(sizePolicy)
        self.lineEdit_amount.setMinimumSize(QtCore.QSize(60, 0))
        self.lineEdit_amount.setMaximumSize(QtCore.QSize(60, 16777215))
        self.lineEdit_amount.setObjectName("lineEdit_amount")
        self.gridLayout.addWidget(self.lineEdit_amount, 1, 2, 1, 1)
        self.horizontalSlider_radius = QtWidgets.QSlider(self.groupBox_level)
        self.horizontalSlider_radius.setMinimum(1)
        self.horizontalSlider_radius.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_radius.setObjectName("horizontalSlider_radius")
        self.gridLayout.addWidget(self.horizontalSlider_radius, 0, 1, 1, 1)
        self.lineEdit_radius = QtWidgets.QLineEdit(self.groupBox_level)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_radius.sizePolicy().hasHeightForWidth())
        self.lineEdit_radius.setSizePolicy(sizePolicy)
        self.lineEdit_radius.setMinimumSize(QtCore.QSize(60, 0))
        self.lineEdit_radius.setMaximumSize(QtCore.QSize(60, 16777215))
        self.lineEdit_radius.setObjectName("lineEdit_radius")
        self.gridLayout.addWidget(self.lineEdit_radius, 0, 2, 1, 1)
        self.label_amount = QtWidgets.QLabel(self.groupBox_level)
        self.label_amount.setObjectName("label_amount")
        self.gridLayout.addWidget(self.label_amount, 1, 0, 1, 1)
        self.label_radius = QtWidgets.QLabel(self.groupBox_level)
        self.label_radius.setObjectName("label_radius")
        self.gridLayout.addWidget(self.label_radius, 0, 0, 1, 1)
        self.pushButton_remove = QtWidgets.QPushButton(self.groupBox_level)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_remove.sizePolicy().hasHeightForWidth())
        self.pushButton_remove.setSizePolicy(sizePolicy)
        self.pushButton_remove.setMinimumSize(QtCore.QSize(60, 0))
        self.pushButton_remove.setMaximumSize(QtCore.QSize(60, 16777215))
        self.pushButton_remove.setObjectName("pushButton_remove")
        self.gridLayout.addWidget(self.pushButton_remove, 2, 2, 1, 1)
        self.checkBox_luminance = QtWidgets.QCheckBox(self.groupBox_level)
        self.checkBox_luminance.setObjectName("checkBox_luminance")
        self.gridLayout.addWidget(self.checkBox_luminance, 2, 0, 1, 2)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setRowStretch(1, 1)
        self.verticalLayout.addWidget(self.groupBox_level)

        self.retranslateUi(sharpening_level_widget)
        QtCore.QMetaObject.connectSlotsByName(sharpening_level_widget)

    def retranslateUi(self, sharpening_level_widget):
        _translate = QtCore.QCoreApplication.translate
        sharpening_level_widget.setWindowTitle(_translate("sharpening_level_widget", "Form"))
        self.groupBox_level.setTitle(_translate("sharpening_level_widget", "Level 1"))
        self.label_amount.setText(_translate("sharpening_level_widget", "Amount"))
        self.label_radius.setText(_translate("sharpening_level_widget", "Radius"))
        self.pushButton_remove.setText(_translate("sharpening_level_widget", "Remove"))
        self.checkBox_luminance.setText(_translate("sharpening_level_widget", "Luminance channel only"))

