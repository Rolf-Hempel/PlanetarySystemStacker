# -*- coding: utf-8 -*-

from sys import argv
from urllib import request

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

        self.label.setTextFormat(Qt.MarkdownText)

        md = self.get_markdown()
        self.label.setText(md)

    def get_markdown(self):
        url = 'file:..\quickstart.md'
        response = request.urlopen(url)
        return response.read().decode()


if __name__ == '__main__':
    app = QtWidgets.QApplication(argv)
    window = LabelGui()
    window.show()
    app.exec_()
