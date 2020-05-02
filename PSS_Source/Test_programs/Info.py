# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 08:59:44 2019

@author: mpo
"""


import os
import platform
from PyQt5 import QtCore
import psutil
import matplotlib
import numpy as np
import cv2

USER = os.environ['USERNAME']
PC = platform.node()
OS = '{0} {1} {2}'.format(platform.system(), platform.release(), platform.architecture()[0])
PYTHON_VERSION = platform.python_version()
QT_VERSION = QtCore.qVersion()
CPU = psutil.cpu_count()
MEMORY = psutil.virtual_memory()[0]/1024**3
MATPLOTLIB_VERSION = matplotlib.__version__
OPENCV_VERSION = cv2.__version__
NUMPY_VERSION = np.__version__

CONTENT = ('''      PSS version: {0}
System overview: User: {1}
PC: {2}
OS: {3}
Python version: {4}
Qt version: {5}
Matplotlib version: {6}
OpenCV version: {7}
Numpy version: {8}
CPU Cores: {9}
Memory: {10:.1f} [GB]'''.format('0.6.0', USER, PC, OS, PYTHON_VERSION, QT_VERSION, MATPLOTLIB_VERSION, OPENCV_VERSION, NUMPY_VERSION, CPU, MEMORY))

print(CONTENT)
# https://doc.qt.io/qtforpython/PySide2/QtWidgets/QMessageBox.html
