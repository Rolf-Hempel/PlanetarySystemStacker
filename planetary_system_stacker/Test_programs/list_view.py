# -*- coding: utf-8 -*-

from sys import argv

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from list_view_widget import Ui_ListViewWidget


class ListView(QtWidgets.QWidget, Ui_ListViewWidget):
    def __init__(self):
        """
        Initialization of the widget.
        """
        super(ListView, self).__init__()
        self.setupUi(self)

        self.frames_included = []
        self.items_selected = None

        self.background_included = QtGui.QColor(130, 255, 130)
        self.background_excluded = QtGui.QColor(120, 120, 120)

        self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        for i in range(30):
            item = QtWidgets.QListWidgetItem("Frame %i included" % i)
            item.setBackground(self.background_included)
            item.setForeground(QtGui.QColor(0, 0, 0))
            self.listWidget.addItem(item)
            self.frames_included.append(True)

        self.indices_selected = []

        self.listWidget.installEventFilter(self)
        self.listWidget.itemClicked.connect(self.select_items)
        self.addButton.clicked.connect(self.use_triggered)
        self.removeButton.clicked.connect(self.not_use_triggered)

    def select_items(self):
        self.items_selected = self.listWidget.selectedItems()
        self.indices_selected = [self.listWidget.row(item) for item in self.items_selected]
        print(self.indices_selected)

    def eventFilter(self, source, event):
        if source is self.listWidget:
            if event.type() == QtCore.QEvent.ContextMenu:
                print("Context menu opened")
                menu = QtWidgets.QMenu()
                action1 = QtWidgets.QAction('Use for stacking', menu)
                action1.triggered.connect(self.use_triggered)
                menu.addAction((action1))
                action2 = QtWidgets.QAction("Don't use for stacking", menu)
                action2.triggered.connect(self.not_use_triggered)
                menu.addAction((action2))
                menu.exec_(event.globalPos())
            elif event.type() == QtCore.QEvent.KeyPress:
                if event.key() == Qt.Key_Plus:
                    self.use_triggered()
                elif event.key() == Qt.Key_Minus:
                    self.not_use_triggered()
                elif event.key() == Qt.Key_Escape:
                    self.items_selected = []
                    self.indices_selected = []
        return super(ListView, self).eventFilter(source, event)

    def use_triggered(self):
        if self.items_selected:
            for index, item in enumerate(self.items_selected):
                index_selcted = self.indices_selected[index]
                item.setText("Frame %i included" % index_selcted)
                item.setBackground(self.background_included)
                item.setForeground(QtGui.QColor(0, 0, 0))
                self.frames_included[index_selcted] = True

            print(str(self.frames_included))

    def not_use_triggered(self):
        if self.items_selected:
            for index, item in enumerate(self.items_selected):
                index_selcted = self.indices_selected[index]
                item.setText("Frame %i excluded" % index_selcted)
                item.setBackground(self.background_excluded)
                item.setForeground(QtGui.QColor(255, 255, 255))
                self.frames_included[index_selcted] = False

            print(str(self.frames_included))


if __name__ == '__main__':
    app = QtWidgets.QApplication(argv)
    window = ListView()
    window.show()
    app.exec_()
