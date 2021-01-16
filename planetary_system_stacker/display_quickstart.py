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

        # Get the quickstart guide text from a markdown document. The original text is in file
        # "quickstart.md". The test program "gui_with_label_test.py" reads that file and converts it
        # into a string. This string is assigned here to variable "markdown_text". This way it can
        # be avoided to access an external file at runtime.
        self.label.setTextFormat(Qt.MarkdownText)
        markdown_text = "### Quickstart Guide\r\n\r\n#### Configuration\r\nSelect 'Edit / Edit configuration' to check if the configuration parameters are set properly. The dialog starts with the 'Frame-related Parameters' where the most important choice is the stabilization mode:\r\n* Surface (for extended objects which do not fit into the FoV)\r\n* Planet (if there is space around the object on all sides)\r\n\r\nIf the view is very unsteady (e.g. caused by wind), increase the 'Stabilization search width'. Check the box 'Dialog to exclude frames from stacking' if some frames may be corrupted by artifacts. Have a look at the other parameter sections as well.\r\n\r\n#### Select jobs\r\nNext, press 'File / Open' to specify the jobs for this session. Video files / image folders are stacked, single images are postprocessed. Either use the PSS file chooser, or cancel the chooser and 'drag and drop' objects from a file explorer.\r\n\r\n#### Job execution (interactive / batch)\r\nStart the processing with 'Start / Cont.'. Jobs are executed in consecutive order, either interactively (default, with GUI), or automatically (batch). If jobs are similar, use interactive mode for the first job to make processing choices. Then check the box 'Automatic' to have PSS process all the other jobs automatically with the same parameters. You can change back to interactive mode at any time by unchecking the 'Automatic' box.\r\n\r\n#### Postprocessing\r\nPostprocessing can follow stacking immediately (if the workflow parameter 'Stacking plus postprocessing' is checked), or be executed separately (if the job input is a single image file). To process several images with the same parameters, adjust the parameters for the first image in interactive mode, and then check 'Automatic' to repeat the same for all other images in batch mocde. \r\n"
        self.label.setText(markdown_text)

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
