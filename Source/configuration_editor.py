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

from parameter_configuration import Ui_ConfigurationDialog
from configuration import ConfigurationParameters


class ConfigurationEditor(QtWidgets.QDialog, Ui_ConfigurationDialog):
    """
    Update the parameters used by PlanetarySystemStacker which are stored in the configuration
    object. The interaction with the user is through the ConfigurationDialog class.
    """

    def __init__(self, configuration, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)

        self.configuration = configuration
        self.configuration.configuration_changed = False

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        # Create a ConfigurationParameters object and set it to the current parameters.
        self.config_copy = ConfigurationParameters()
        self.config_copy.copy_from_config_object(self.configuration)
