# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'job_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_JobDialog(object):
    def setupUi(self, JobDialog):
        JobDialog.setObjectName("JobDialog")
        JobDialog.resize(900, 530)
        self.gridLayout = QtWidgets.QGridLayout(JobDialog)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 2, 1, 1)
        self.job_list_widget = QtWidgets.QListWidget(JobDialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.job_list_widget.sizePolicy().hasHeightForWidth())
        self.job_list_widget.setSizePolicy(sizePolicy)
        self.job_list_widget.setMinimumSize(QtCore.QSize(0, 483))
        self.job_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.job_list_widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.job_list_widget.setObjectName("job_list_widget")
        self.gridLayout.addWidget(self.job_list_widget, 0, 0, 1, 6)
        self.button_remove_jobs = QtWidgets.QPushButton(JobDialog)
        self.button_remove_jobs.setObjectName("button_remove_jobs")
        self.gridLayout.addWidget(self.button_remove_jobs, 1, 1, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(JobDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 1, 5, 1, 1)
        self.button_add_jobs = QtWidgets.QPushButton(JobDialog)
        self.button_add_jobs.setObjectName("button_add_jobs")
        self.gridLayout.addWidget(self.button_add_jobs, 1, 0, 1, 1)
        self.messageLabel = QtWidgets.QLabel(JobDialog)
        self.messageLabel.setObjectName("messageLabel")
        self.gridLayout.addWidget(self.messageLabel, 1, 3, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 1, 4, 1, 1)

        self.retranslateUi(JobDialog)
        self.buttonBox.accepted.connect(JobDialog.accept)
        self.buttonBox.rejected.connect(JobDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(JobDialog)

    def retranslateUi(self, JobDialog):
        _translate = QtCore.QCoreApplication.translate
        JobDialog.setWindowTitle(_translate("JobDialog", "Manage Job List"))
        self.button_remove_jobs.setToolTip(_translate("JobDialog", "Select one or more items in the list. Press this button to remove them from the list."))
        self.button_remove_jobs.setText(_translate("JobDialog", "Remove selected job(s)"))
        self.buttonBox.setToolTip(_translate("JobDialog", "Press \'OK\' to save changes and exit the job editor, or \'Cancel\' to exit without saving."))
        self.button_add_jobs.setToolTip(_translate("JobDialog", "Select job input. This can be video files and / or directories containing image files for stacking, or single image files for postprocessing.\n"
"Each entry is executed as a job, either in batch mode (automatic) or manually. When the list is complete, confirm with pressing \"OK\"."))
        self.button_add_jobs.setText(_translate("JobDialog", "Add job(s)"))
        self.messageLabel.setText(_translate("JobDialog", "Add / remove videos / image folders for stacking, or images for postprocessing, confirm with \'OK\'."))


