from sys import argv
from time import sleep

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import QApplication, QMainWindow, QProxyStyle, QStyle

class CustomStyle(QProxyStyle):
    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.SH_SpinBox_KeyPressAutoRepeatRate:
            return 10**6
        elif hint == QStyle.SH_SpinBox_ClickAutoRepeatRate:
            return 10**6
        elif hint == QStyle.SH_SpinBox_ClickAutoRepeatThreshold:
            # You can use only this condition to avoid the auto-repeat,
            # but better safe than sorry ;-)
            return 10**6
        else:
            return super().styleHint(hint, option, widget, returnData)

class SpinBoxTest(QtWidgets.QWidget):
    def __init__(self):
        super(SpinBoxTest, self).__init__()
        self.spinBox = QtWidgets.QSpinBox(self)
        self.spinBox.setGeometry(QtCore.QRect(9, 27, 131, 20))
        self.spinBox.setStyle(CustomStyle())
        self.spinBox.valueChanged.connect(self.spinbox_changed)

    def spinbox_changed(self):
        sleep(0.5)
        print("spinbox has changed, value: " + str(self.spinBox.value()))

if __name__ == '__main__':
    app = QtWidgets.QApplication(argv)
    window = SpinBoxTest()
    window.show()
    app.exec_()
