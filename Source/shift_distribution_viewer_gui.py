# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'shift_distribution_viewer_gui.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_shift_distribution_viewer(object):
    def setupUi(self, shift_distribution_viewer):
        shift_distribution_viewer.setObjectName("shift_distribution_viewer")
        shift_distribution_viewer.resize(900, 530)
        self.verticalLayout = QtWidgets.QVBoxLayout(shift_distribution_viewer)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.buttonBox = QtWidgets.QDialogButtonBox(shift_distribution_viewer)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.horizontalLayout.addWidget(self.buttonBox)
        self.horizontalLayout.setStretch(0, 1)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(shift_distribution_viewer)
        QtCore.QMetaObject.connectSlotsByName(shift_distribution_viewer)

    def retranslateUi(self, shift_distribution_viewer):
        _translate = QtCore.QCoreApplication.translate
        shift_distribution_viewer.setWindowTitle(_translate("shift_distribution_viewer", "Form"))


