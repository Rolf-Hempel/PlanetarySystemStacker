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
        self.button_add_jobs = QtWidgets.QPushButton(JobDialog)
        self.button_add_jobs.setObjectName("button_add_jobs")
        self.gridLayout.addWidget(self.button_add_jobs, 1, 0, 1, 1)
        self.job_list_widget = QtWidgets.QListWidget(JobDialog)
        self.job_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.job_list_widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.job_list_widget.setObjectName("job_list_widget")
        self.gridLayout.addWidget(self.job_list_widget, 0, 0, 1, 3)
        self.buttonBox = QtWidgets.QDialogButtonBox(JobDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 1, 2, 1, 1)
        self.button_remove_jobs = QtWidgets.QPushButton(JobDialog)
        self.button_remove_jobs.setObjectName("button_remove_jobs")
        self.gridLayout.addWidget(self.button_remove_jobs, 1, 1, 1, 1)

        self.retranslateUi(JobDialog)
        self.buttonBox.accepted.connect(JobDialog.accept)
        self.buttonBox.rejected.connect(JobDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(JobDialog)

    def retranslateUi(self, JobDialog):
        _translate = QtCore.QCoreApplication.translate
        JobDialog.setWindowTitle(_translate("JobDialog", "Manage Job List"))
        self.button_add_jobs.setToolTip(_translate("JobDialog", "Select video files and / or directories containing image files.\n"
"Each entry is executed as a job, either in batch mode (automatic) or manually.\n"
"When the list is okay, confirm with pressing the \"OK\" button."))
        self.button_add_jobs.setText(_translate("JobDialog", "Add job(s)"))
        self.button_remove_jobs.setToolTip(_translate("JobDialog", "Select one or more items in the list. Press this button to remove them from the list."))
        self.button_remove_jobs.setText(_translate("JobDialog", "Remove selected job(s)"))


