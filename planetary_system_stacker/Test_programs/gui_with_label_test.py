# -*- coding: utf-8 -*-

from sys import argv

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from gui_with_label import Ui_Form


class LabelGui(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        """
        Initialization of the widget.
        """
        super(LabelGui, self).__init__()
        self.setupUi(self)

        self.label.setText("### Quickstart Guide \n"
            "<br /> Select 'Edit / Edit configuration' to check if the configuration parameters are set properly. The dialog starts with the 'Frame-related Parameters' where the most important choice is the stabilization mode: \n"
            "-   Surface (for extended objects which do not fit into the FoV) \n"
            "-   Planet \n\n"
            "If the view is very unsteady (e.g. caused by wind), increase the 'Stabilization search width'. Check the box 'Dialog to exclude frames from stacking' if some frames may be corrupted by artifacts. Have a look at the other parameter sections as well. \n\n"
            "<br /> Next, press 'File / Open' to specify the jobs for this session. Video files / image folders are stacked, single images are postprocessed. Either use the PSS file chooser, or cancel the chooser and 'drag and drop' objects from a file explorer. \n\n"
            "<br /> Start the processing with 'Start / Cont.'. Jobs are executed in consecutive order, either interactively (standard, with GUI), or automatically (batch). If jobs are similar, use interactive mode for the first job to make processing choices. Then check the box 'Automatic' to have PSS process all the other jobs automatically with the same parameters. You can change back to interactive mode at any time.")

        self.label.setTextFormat(Qt.MarkdownText)


if __name__ == '__main__':
    app = QtWidgets.QApplication(argv)
    window = LabelGui()
    window.show()
    app.exec_()
