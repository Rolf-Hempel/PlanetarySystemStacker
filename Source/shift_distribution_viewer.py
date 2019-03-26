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

from PyQt5 import QtWidgets, QtCore

from shift_distribution_viewer_gui import Ui_shift_distribution_viewer


class ShiftDistributionViewer(QtWidgets.QFrame, Ui_shift_distribution_viewer):
    """
    Display the distribution of frame shifts at alignment points. For each AP in each contributing
    frame the total shift (in pixels) is rounded to the next integer value. For every integer number
    up to the maximum frame shift the number of occurrences is displayed.
    """

    def __init__(self, parent_gui, parent=None):
        """
        Initialize the Viewer. The widget has a fixed size and is rendered as a QFrame.

        :param parent_gui: GUI object by which the viewer is invoked.
        :param parent: Parent object
        """

        QtWidgets.QFrame.__init__(self, parent)
        self.setupUi(self)

        self.setFrameShape(QtWidgets.QFrame.Panel)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setObjectName("shift_distribution_viewer")

        self.setFixedSize(900, 600)

        self.parent_gui = parent_gui
        self.configuration = self.parent_gui.configuration

        self.buttonBox.accepted.connect(self.accept)

    def accept(self):
        """

        :return: -
        """

        self.close()

    def closeEvent(self, event):
        """
        Remove the widget from the parent GUI and close the viewer.

        :param event: Close event object
        :return: -
        """

        self.parent_gui.display_widget(None, display=False)
        self.close()
