# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'shift_distribution_viewer_gui.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_shift_distribution_viewer(object):
    def setupUi(self, shift_distribution_viewer):
        shift_distribution_viewer.setObjectName("shift_distribution_viewer")
        shift_distribution_viewer.resize(900, 530)
        self.verticalLayout = QtWidgets.QVBoxLayout(shift_distribution_viewer)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.failedShiftsLabel = QtWidgets.QLabel(shift_distribution_viewer)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.failedShiftsLabel.setFont(font)
        self.failedShiftsLabel.setObjectName("failedShiftsLabel")
        self.horizontalLayout.addWidget(self.failedShiftsLabel)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.buttonBox = QtWidgets.QDialogButtonBox(shift_distribution_viewer)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.horizontalLayout.addWidget(self.buttonBox)
        self.horizontalLayout.setStretch(1, 1)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(shift_distribution_viewer)
        QtCore.QMetaObject.connectSlotsByName(shift_distribution_viewer)

    def retranslateUi(self, shift_distribution_viewer):
        _translate = QtCore.QCoreApplication.translate
        shift_distribution_viewer.setWindowTitle(_translate("shift_distribution_viewer", "Form"))
        self.failedShiftsLabel.setText(_translate("shift_distribution_viewer", "Failed shift measurements (percent):"))
