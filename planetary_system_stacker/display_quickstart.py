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

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

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
        # "quickstart.md".
        self.label.setTextFormat(Qt.TextFormat.MarkdownText)
        markdown_text = """
### Quickstart Guide

#### Configuration
Select 'Edit / Edit configuration' to check if the configuration parameters are set properly.
The dialog starts with the 'Frame-related Parameters' where the most important choice is the
stabilization mode:
* Surface (for extended objects which do not fit into the FoV)
* Planet (if there is space around the object on all sides)

If the view is very unsteady (e.g. caused by wind), increase the 'Stabilization search width'.
For noisier planetary images, try 'Noise levels' up to the maximum. Check the box
'Dialog to exclude frames from stacking' if some frames may be corrupted by artifacts.
Have a look at the other parameter sections as well. For example, for strongly oversampled
planetary images consider increasing the 'Max alignment search width' (Section 'Multipoint Alignment
Parameters').

#### Select jobs
Next, press 'File / Open' to specify the jobs for this session. Video files / image folders are
stacked, single images are postprocessed. Either use the PSS file chooser, or cancel the chooser and
'drag and drop' objects from a file explorer.

#### Job execution (interactive / batch)
Start the processing with 'Start / Cont.'. Jobs are executed in consecutive order, either
interactively (default, with GUI), or automatically (batch). If jobs are similar, use interactive
mode for the first job to make processing choices. Then check the box 'Automatic' to have PSS
process all the other jobs automatically with the same parameters. You can change back to
interactive mode at any time by unchecking the 'Automatic' box.

The most important dialog is the one where the alignment points are set. It is critical
to experiment with different 'Alignment box widths' (upper right slider). Start with a high
value (e.g. 100), and reduce it until the final image quality does not improve any further.
Too low values may result in stacking artifacts. As an alternative to the automatic AP
grid generation, you can set APs manually by left-clicking into the preview image.

Another important paramter which needs to be adapted to the individual case is the number /
percentage of frames used for stacking. Try different values to find the optimal choice.

#### Postprocessing
Postprocessing can follow stacking immediately (if the workflow parameter 'Stacking plus
postprocessing' is checked), or be executed separately (if the job input is a single image file).
To process several images with the same parameters, adjust the parameters for the first image in
interactive mode, and then check 'Automatic' to repeat the same for all other images in batch mode.
        """

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
