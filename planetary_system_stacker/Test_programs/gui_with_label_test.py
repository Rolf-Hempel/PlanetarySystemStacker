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
        self.label.setText("# Example for a Quickstart text\n* First bullet text\n* Secone bullet text\nparagraph after bullets.")
        self.label.setTextFormat(Qt.MarkdownText)


if __name__ == '__main__':
    app = QtWidgets.QApplication(argv)
    window = LabelGui()
    window.show()
    app.exec_()
