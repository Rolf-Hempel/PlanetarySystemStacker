# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'alignment_point_editor_gui.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_alignment_point_editor(object):
    def setupUi(self, alignment_point_editor):
        alignment_point_editor.setObjectName("alignment_point_editor")
        alignment_point_editor.resize(900, 630)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(alignment_point_editor.sizePolicy().hasHeightForWidth())
        alignment_point_editor.setSizePolicy(sizePolicy)
        alignment_point_editor.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        alignment_point_editor.setFrameShape(QtWidgets.QFrame.Panel)
        alignment_point_editor.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.verticalLayout = QtWidgets.QVBoxLayout(alignment_point_editor)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btnLoad = QtWidgets.QPushButton(alignment_point_editor)
        self.btnLoad.setObjectName("btnLoad")
        self.horizontalLayout.addWidget(self.btnLoad)
        self.btnApGrid = QtWidgets.QPushButton(alignment_point_editor)
        self.btnApGrid.setObjectName("btnApGrid")
        self.horizontalLayout.addWidget(self.btnApGrid)
        self.btnUndo = QtWidgets.QPushButton(alignment_point_editor)
        self.btnUndo.setObjectName("btnUndo")
        self.horizontalLayout.addWidget(self.btnUndo)
        self.btnRedo = QtWidgets.QPushButton(alignment_point_editor)
        self.btnRedo.setObjectName("btnRedo")
        self.horizontalLayout.addWidget(self.btnRedo)
        self.btnDone = QtWidgets.QPushButton(alignment_point_editor)
        self.btnDone.setObjectName("btnDone")
        self.horizontalLayout.addWidget(self.btnDone)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(alignment_point_editor)
        QtCore.QMetaObject.connectSlotsByName(alignment_point_editor)

    def retranslateUi(self, alignment_point_editor):
        _translate = QtCore.QCoreApplication.translate
        alignment_point_editor.setWindowTitle(_translate("alignment_point_editor", "Frame"))
        self.btnLoad.setText(_translate("alignment_point_editor", "Load Image"))
        self.btnApGrid.setText(_translate("alignment_point_editor", "Create AP Grid"))
        self.btnUndo.setText(_translate("alignment_point_editor", "Undo"))
        self.btnRedo.setText(_translate("alignment_point_editor", "Redo"))
        self.btnDone.setText(_translate("alignment_point_editor", "Done"))


