# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'list_view_widget.ui'
#
# Created by: PyQt5 UI code generator 5.11.2
#
# WARNING! All changes made in this file will be lost!

from PyQt6 import QtCore, QtGui, QtWidgets

class Ui_ListViewWidget(object):
    def setupUi(self, ListViewWidget):
        ListViewWidget.setObjectName("ListViewWidget")
        ListViewWidget.resize(264, 341)
        self.horizontalLayout = QtWidgets.QHBoxLayout(ListViewWidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.listWidget = QtWidgets.QListWidget(ListViewWidget)
        self.listWidget.setObjectName("listWidget")
        self.horizontalLayout.addWidget(self.listWidget)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.addButton = QtWidgets.QPushButton(ListViewWidget)
        self.addButton.setObjectName("addButton")
        self.verticalLayout.addWidget(self.addButton)
        self.removeButton = QtWidgets.QPushButton(ListViewWidget)
        self.removeButton.setObjectName("removeButton")
        self.verticalLayout.addWidget(self.removeButton)
        self.horizontalLayout.addLayout(self.verticalLayout)

        self.retranslateUi(ListViewWidget)
        QtCore.QMetaObject.connectSlotsByName(ListViewWidget)

    def retranslateUi(self, ListViewWidget):
        _translate = QtCore.QCoreApplication.translate
        ListViewWidget.setWindowTitle(_translate("ListViewWidget", "Frame"))
        self.addButton.setText(_translate("ListViewWidget", "+"))
        self.removeButton.setText(_translate("ListViewWidget", "-"))
