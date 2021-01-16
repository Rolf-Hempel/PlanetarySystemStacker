# -*- coding: utf-8; -*-
"""
Copyright (c) 2021 Rolf Hempel, rolf6419@gmx.de

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

from sys import argv
from urllib import request

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from configuration import Configuration
from quickstart_gui import Ui_Form


class DisplayQuickstart(QtWidgets.QWidget, Ui_Form):
    def __init__(self, parent_gui, configuration):
        """
        Initialization of the widget. The quickstart guide is displayed in the viewer window.
        """

        super(DisplayQuickstart, self).__init__()
        self.setupUi(self)

        self.parent_gui = parent_gui
        self.configuration = configuration

        # Get the quickstart guide text from a markdown document.
        self.label.setTextFormat(Qt.MarkdownText)
        response = request.urlopen(self.configuration.global_parameters_quickstart_url)
        self.label.setText(response.read().decode())

        self.dont_show_checkBox.setChecked(not self.configuration.global_parameters_display_quickstart)
        self.dont_show_checkBox.stateChanged.connect(self.checkbox_changed)

    def checkbox_changed(self):
        """
        This method is connected with the checkbox in the quickstart guide and with the menu
        entry where the quickstart guide can be re-activated.

        :return: -
        """
        self.configuration.global_parameters_display_quickstart = \
            not self.configuration.global_parameters_display_quickstart
        if self.parent_gui is not None:
            self.parent_gui.ui.actionShow_Quickstart.setChecked(
                self.configuration.global_parameters_display_quickstart)


if __name__ == '__main__':
    # Get configuration parameters.
    configuration = Configuration()
    configuration.initialize_configuration()

    app = QtWidgets.QApplication(argv)
    window = DisplayQuickstart(None, configuration)
    window.show()
    app.exec_()
