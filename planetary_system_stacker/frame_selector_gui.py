# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'frame_selector_gui.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_frame_selector(object):
    def setupUi(self, frame_selector):
        frame_selector.setObjectName("frame_selector")
        frame_selector.resize(900, 630)
        frame_selector.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        frame_selector.setFrameShape(QtWidgets.QFrame.Panel)
        frame_selector.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.gridLayout = QtWidgets.QGridLayout(frame_selector)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonBox = QtWidgets.QDialogButtonBox(frame_selector)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 2, 3, 1, 2)
        self.pushButton_play = QtWidgets.QPushButton(frame_selector)
        self.pushButton_play.setObjectName("pushButton_play")
        self.gridLayout.addWidget(self.pushButton_play, 2, 2, 1, 1)
        self.pushButton_stop = QtWidgets.QPushButton(frame_selector)
        self.pushButton_stop.setObjectName("pushButton_stop")
        self.gridLayout.addWidget(self.pushButton_stop, 2, 1, 1, 1)
        self.slider_frames = QtWidgets.QSlider(frame_selector)
        self.slider_frames.setMaximum(1000)
        self.slider_frames.setPageStep(1)
        self.slider_frames.setOrientation(QtCore.Qt.Horizontal)
        self.slider_frames.setObjectName("slider_frames")
        self.gridLayout.addWidget(self.slider_frames, 2, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(frame_selector)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.listWidget = QtWidgets.QListWidget(self.groupBox)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout_2.addWidget(self.listWidget)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.addButton = QtWidgets.QPushButton(self.groupBox)
        self.addButton.setObjectName("addButton")
        self.horizontalLayout.addWidget(self.addButton)
        self.removeButton = QtWidgets.QPushButton(self.groupBox)
        self.removeButton.setObjectName("removeButton")
        self.horizontalLayout.addWidget(self.removeButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout.addWidget(self.groupBox)
        self.gridLayout.addLayout(self.verticalLayout, 0, 4, 1, 1)
        self.GroupBox_frame_sorting = QtWidgets.QGroupBox(frame_selector)
        self.GroupBox_frame_sorting.setObjectName("GroupBox_frame_sorting")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.GroupBox_frame_sorting)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.radioButton_quality = QtWidgets.QRadioButton(self.GroupBox_frame_sorting)
        self.radioButton_quality.setObjectName("radioButton_quality")
        self.horizontalLayout_2.addWidget(self.radioButton_quality)
        self.radioButton_chronological = QtWidgets.QRadioButton(self.GroupBox_frame_sorting)
        self.radioButton_chronological.setObjectName("radioButton_chronological")
        self.horizontalLayout_2.addWidget(self.radioButton_chronological)
        self.gridLayout.addWidget(self.GroupBox_frame_sorting, 1, 4, 1, 1)
        self.gridLayout.setColumnStretch(0, 7)
        self.gridLayout.setColumnStretch(4, 3)

        self.retranslateUi(frame_selector)
        QtCore.QMetaObject.connectSlotsByName(frame_selector)

    def retranslateUi(self, frame_selector):
        _translate = QtCore.QCoreApplication.translate
        frame_selector.setWindowTitle(_translate("frame_selector", "Frame"))
        self.buttonBox.setToolTip(_translate("frame_selector", "Exit the viewer. Press \'OK\' to save the selection, or \'cancel\' to discard changes."))
        self.pushButton_play.setToolTip(_translate("frame_selector", "Start a frame display video."))
        self.pushButton_play.setText(_translate("frame_selector", "Play"))
        self.pushButton_stop.setToolTip(_translate("frame_selector", "Stop the frame display video."))
        self.pushButton_stop.setText(_translate("frame_selector", "Stop"))
        self.slider_frames.setToolTip(_translate("frame_selector", "Use the slider to select the frame to be displayed. As an alternative,\n"
"you can select the frame in the \'Selecct / deselect frames\' list."))
        self.groupBox.setTitle(_translate("frame_selector", "Select / deselect frames"))
        self.listWidget.setToolTip(_translate("frame_selector", "Left-click to mark frame indices / index ranges. Include / exclude selected frames via the context menu, the plus / minus buttons below, or by pressing + / - on the keyboard."))
        self.addButton.setToolTip(_translate("frame_selector", "Use selected frame(s) for stacking."))
        self.addButton.setText(_translate("frame_selector", "+"))
        self.removeButton.setToolTip(_translate("frame_selector", "Don\'t use selected frame(s) for stacking."))
        self.removeButton.setText(_translate("frame_selector", "-"))
        self.GroupBox_frame_sorting.setTitle(_translate("frame_selector", "Frame sorting"))
        self.radioButton_quality.setToolTip(_translate("frame_selector", "Frames are ordered by their overall sharpness."))
        self.radioButton_quality.setText(_translate("frame_selector", "By quality"))
        self.radioButton_chronological.setToolTip(_translate("frame_selector", "Frames are ordered chronologically."))
        self.radioButton_chronological.setText(_translate("frame_selector", "Chronological"))
