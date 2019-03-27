# -*- coding: utf-8; -*-
"""
Copyright (c) 2019 Rolf Hempel, rolf6419@gmx.de

This file is part of the PlanetarySystemStacker tool (PSS).
https://github.com/Rolf-Hempel/PlanetarySystemStacker

PSS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PSS.  If not, see <http://www.gnu.org/licenses/>.

"""

import matplotlib
matplotlib.use('qt5agg')

import numpy as np
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

from shift_distribution_viewer_gui import Ui_shift_distribution_viewer


class MatplotlibWidget(Canvas):
    """
    This widget creates a plot of frame qualities, either sorted chronologically or by quality.

    """

    def __init__(self, parent=None):
        """
        Initialize the widget.

        :param parent: Parent object
        """

        super(MatplotlibWidget, self).__init__(Figure())

        self.setParent(parent)

        plt.rcParams.update({'font.size': 8})
        self.fig, self.ax = plt.subplots()

    def draw_distribution(self, shift_distribution):
        """
        Draw the shift distribution.

        :param shift_distribution: 1D Numpy array (int) with counts for each shift size.
        :return: -
        """

        # Find the last non-zero entry in the array.
        max_index = [index for index, item in enumerate(shift_distribution) if item != 0][-1] + 1

        pixels = np.arange(max_index)

        plt.bar(pixels, shift_distribution[:max_index], align='center')
        plt.xticks(pixels, pixels)
        plt.xlabel('Warp size (pixels)')
        plt.ylabel('Frequency')
        plt.title('Frequency distribution of local warp sizes at alignment points')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class ShiftDistributionViewerWidget(QtWidgets.QFrame, Ui_shift_distribution_viewer):
    """
    Display the distribution of frame shifts at alignment points. For each AP in each contributing
    frame the total shift (in pixels) is rounded to the next integer value. For every integer number
    up to the maximum frame shift the number of occurrences is displayed.
    """

    def __init__(self, parent_gui, shift_distribution, signal_finished, parent=None):
        """
        Initialize the Viewer. The widget has a fixed size and is rendered as a QFrame.

        :param parent_gui: GUI object by which the viewer is invoked.
        :param parent: Parent object
        :param shift_distribution: 1D Numpy array (int) with counts for each shift size.
        :param signal_finished: Qt signal with signature () telling the workflow thread that the
                                viewer has finished, or None (no signalling).
        """

        QtWidgets.QFrame.__init__(self, parent)
        self.setupUi(self)

        self.setFrameShape(QtWidgets.QFrame.Panel)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setObjectName("shift_distribution_viewer")

        self.setFixedSize(900, 600)

        self.parent_gui = parent_gui
        self.shift_distribution = shift_distribution
        self.signal_finished = signal_finished

        self.buttonBox.accepted.connect(self.accept)

        # Create the Matplotlib widget showing the shift statistics.
        self.matplotlib_widget = MatplotlibWidget()
        self.verticalLayout.insertWidget(0, Canvas(self.matplotlib_widget.fig))

        self.matplotlib_widget.draw_distribution(self.shift_distribution)

    def accept(self):
        """

        :return: -
        """

        # If a signal was passed in initialization, tell the workflow thread that the viewer has
        # finished.
        if self.signal_finished is not None:
            self.signal_finished.emit()

        self.close()

    def closeEvent(self, event):
        """
        Remove the widget from the parent GUI and close the viewer.

        :param event: Close event object
        :return: -
        """

        # Remove the viewer widget from the main GUI and exit.
        self.parent_gui.display_widget(None, display=False)
        self.close()
