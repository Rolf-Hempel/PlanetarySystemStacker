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

import sys

from PyQt5 import QtWidgets, QtCore

from configuration import ConfigurationParameters, Configuration
from parameter_configuration import Ui_ConfigurationDialog


class ConfigurationEditor(QtWidgets.QDialog, Ui_ConfigurationDialog):
    """
    Update the parameters used by PlanetarySystemStacker which are stored in the configuration
    object. The interaction with the user is through the ConfigurationDialog class.
    """

    def __init__(self, configuration, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)

        self.configuration = configuration

        # On return set the following variable to the activity to which the workflow has to go back
        # due to parameter changes. If it is None, nothing has to be repeated.
        self.configuration.go_back_to_activity = None
        self.configuration.configuration_changed = False

        # Create a ConfigurationParameters object and set it to the current parameters.
        self.config_copy = ConfigurationParameters()
        self.config_copy.copy_from_config_object(self.configuration)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        # Connect local methods with GUI change events.
        self.fgw_slider_value.valueChanged['int'].connect(self.fgw_changed)
        self.afm_comboBox.addItem('Surface')
        self.afm_comboBox.addItem('Planet')
        self.afm_comboBox.activated[str].connect(self.afm_changed)
        self.afa_checkBox.stateChanged.connect(self.afa_changed)
        self.afrsf_slider_value.valueChanged['int'].connect(self.afrsf_changed)
        self.afsw_slider_value.valueChanged['int'].connect(self.afsw_changed)
        self.afafp_slider_value.valueChanged['int'].connect(self.afafp_changed)
        self.gpwptf_checkBox.stateChanged.connect(self.gpwptf_changed)
        self.gpspwr_checkBox.stateChanged.connect(self.gpspwr_changed)
        self.gppl_spinBox.valueChanged['int'].connect(self.gppl_changed)

        self.aphbw_slider_value.valueChanged['int'].connect(self.aphbw_changed)
        self.apsw_slider_value.valueChanged['int'].connect(self.apsw_changed)
        self.apst_slider_value.valueChanged['int'].connect(self.apst_changed)
        self.apbt_slider_value.valueChanged['int'].connect(self.apbt_changed)
        self.apfp_slider_value.valueChanged['int'].connect(self.apfp_changed)

        self.restore_standard_values.clicked.connect(self.restore_standard_parameters)

        self.initialize_widgets_and_local_parameters()

    def initialize_widgets_and_local_parameters(self):
        # Initialize GUI widgets with current configuration parameter values.
        self.fgw_slider_value.setValue(int((self.config_copy.frames_gauss_width+1)/2))
        self.fgw_label_display.setText(str(self.config_copy.frames_gauss_width))
        index = self.afm_comboBox.findText(self.config_copy.align_frames_mode,
                                           QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.afm_comboBox.setCurrentIndex(index)
        self.afa_checkBox.setChecked(self.config_copy.align_frames_automation)
        self.afrsf_slider_value.setValue(
            int(round(100. / self.config_copy.align_frames_rectangle_scale_factor)))
        self.afrsf_label_display.setText(
            str(int(round(100. / self.config_copy.align_frames_rectangle_scale_factor))))
        self.afsw_slider_value.setValue(self.config_copy.align_frames_search_width)
        self.afsw_label_display.setText(str(self.config_copy.align_frames_search_width))
        self.afafp_slider_value.setValue(self.config_copy.align_frames_average_frame_percent)
        self.afafp_label_display.setText(str(self.config_copy.align_frames_average_frame_percent))
        self.gpwptf_checkBox.setChecked(self.config_copy.global_parameters_write_protocol_to_file)
        self.gpspwr_checkBox.setChecked(
            self.config_copy.global_parameters_store_protocol_with_result)
        self.gppl_spinBox.setValue(self.config_copy.global_parameters_protocol_level)

        self.aphbw_slider_value.setValue(self.config_copy.alignment_points_half_box_width * 2)
        self.aphbw_label_display.setText(str(self.config_copy.alignment_points_half_box_width * 2))
        self.apsw_slider_value.setValue(self.config_copy.alignment_points_search_width)
        self.apsw_label_display.setText(str(self.config_copy.alignment_points_search_width))
        self.apst_slider_value.setValue(
            int(round(self.config_copy.alignment_points_structure_threshold * 100)))
        self.apst_label_display.setText(str(self.config_copy.alignment_points_structure_threshold))
        self.apbt_slider_value.setValue(self.config_copy.alignment_points_brightness_threshold)
        self.apbt_label_display.setText(str(self.config_copy.alignment_points_brightness_threshold))
        self.apfp_slider_value.setValue(self.config_copy.alignment_points_frame_percent)
        self.apfp_label_display.setText(str(self.config_copy.alignment_points_frame_percent))

    def fgw_changed(self, value):
        self.config_copy.frames_gauss_width = 2 * value - 1
        self.fgw_label_display.setText(str(self.config_copy.frames_gauss_width))

    def afm_changed(self, value):
        self.config_copy.align_frames_mode = value

    def afa_changed(self, state):
        self.config_copy.align_frames_automation = (state == QtCore.Qt.Checked)

    def afrsf_changed(self, value):
        self.config_copy.align_frames_rectangle_scale_factor = int(round(100. / value))

    def afsw_changed(self, value):
        self.config_copy.align_frames_search_width = value

    def afafp_changed(self, value):
        self.config_copy.align_frames_average_frame_percent = value

    def gpwptf_changed(self, state):
        self.config_copy.global_parameters_write_protocol_to_file = (state == QtCore.Qt.Checked)

    def gpspwr_changed(self, state):
        self.config_copy.global_parameters_write_protocol_to_file = (state == QtCore.Qt.Checked)

    def gppl_changed(self, value):
        self.config_copy.global_parameters_protocol_level = value

    def aphbw_changed(self, value):
        self.config_copy.alignment_points_half_box_width = int(value / 2)

    def apsw_changed(self, value):
        self.config_copy.alignment_points_search_width = value

    def apst_changed(self, value):
        self.config_copy.alignment_points_structure_threshold = value / 100.
        self.apst_label_display.setText(str(self.config_copy.alignment_points_structure_threshold))

    def apbt_changed(self, value):
        self.config_copy.alignment_points_brightness_threshold = value

    def apfp_changed(self, value):
        self.config_copy.alignment_points_frame_percent = value

    def restore_standard_parameters(self):
        """
        Reset configuration parameters and GUI widget settings to standard values. Mark
        configuration as changed. This may be too pessimistic, if standard values were not changed
        before.

        :return:
        """

        self.config_copy.set_defaults()
        self.initialize_widgets_and_local_parameters()

    def accept(self):
        """
        If the OK button is clicked and the configuration has been changed, check if values have
        been changed and assign the new values to configuration parameters.

        :return: -
        """

        if self.config_copy.global_parameters_protocol_level != \
                self.configuration.global_parameters_protocol_level:
            self.configuration.global_parameters_protocol_level = \
                self.config_copy.global_parameters_protocol_level
            self.configuration.configuration_changed = True

        if self.config_copy.alignment_points_frame_percent != \
                self.configuration.alignment_points_frame_percent:
            self.configuration.alignment_points_frame_percent = \
                self.config_copy.alignment_points_frame_percent
            self.configuration.configuration_changed = True
            self.configuration.go_back_to_activity = 'Compute frame qualities'

        if self.config_copy.alignment_points_brightness_threshold != \
                self.configuration.alignment_points_brightness_threshold:
            self.configuration.alignment_points_brightness_threshold = \
                self.config_copy.alignment_points_brightness_threshold
            self.configuration.configuration_changed = True
            self.configuration.go_back_to_activity = 'Set alignment points'

        if self.config_copy.alignment_points_structure_threshold != \
                self.configuration.alignment_points_structure_threshold:
            self.configuration.alignment_points_structure_threshold = \
                self.config_copy.alignment_points_structure_threshold
            self.configuration.configuration_changed = True
            self.configuration.go_back_to_activity = 'Set alignment points'

        if self.config_copy.alignment_points_search_width != \
                self.configuration.alignment_points_search_width:
            self.configuration.alignment_points_search_width = \
                self.config_copy.alignment_points_search_width
            self.configuration.configuration_changed = True
            self.configuration.go_back_to_activity = 'Set alignment points'

        if self.config_copy.alignment_points_half_box_width != \
                self.configuration.alignment_points_half_box_width:
            self.configuration.alignment_points_half_box_width = \
                self.config_copy.alignment_points_half_box_width
            self.configuration.configuration_changed = True
            self.configuration.go_back_to_activity = 'Set alignment points'

        if self.config_copy.align_frames_average_frame_percent != \
                self.configuration.align_frames_average_frame_percent:
            self.configuration.align_frames_average_frame_percent = \
                self.config_copy.align_frames_average_frame_percent
            self.configuration.configuration_changed = True
            self.configuration.go_back_to_activity = 'Align frames'

        if self.config_copy.align_frames_search_width != \
                self.configuration.align_frames_search_width:
            self.configuration.align_frames_search_width = \
                self.config_copy.align_frames_search_width
            self.configuration.configuration_changed = True
            self.configuration.go_back_to_activity = 'Align frames'

        if self.config_copy.align_frames_rectangle_scale_factor != \
                self.configuration.align_frames_rectangle_scale_factor:
            self.configuration.align_frames_rectangle_scale_factor = \
                self.config_copy.align_frames_rectangle_scale_factor
            self.configuration.configuration_changed = True
            self.configuration.go_back_to_activity = 'Align frames'

        if self.config_copy.align_frames_automation != self.configuration.align_frames_automation:
            self.configuration.align_frames_automation = self.config_copy.align_frames_automation
            self.configuration.configuration_changed = True
            self.configuration.go_back_to_activity = 'Align frames'

        if self.config_copy.align_frames_mode != self.configuration.align_frames_mode:
            self.configuration.align_frames_mode = self.config_copy.align_frames_mode
            self.configuration.configuration_changed = True
            self.configuration.go_back_to_activity = 'Align frames'

        if self.config_copy.frames_gauss_width != self.configuration.frames_gauss_width:
            self.configuration.frames_gauss_width = self.config_copy.frames_gauss_width
            self.configuration.configuration_changed = True
            self.configuration.go_back_to_activity = 'Read frames'

        if self.config_copy.global_parameters_store_protocol_with_result != \
                self.configuration.global_parameters_store_protocol_with_result:
            self.configuration.global_parameters_store_protocol_with_result = \
                self.config_copy.global_parameters_store_protocol_with_result
            self.configuration.configuration_changed = True

        if self.config_copy.global_parameters_write_protocol_to_file != \
                self.configuration.global_parameters_write_protocol_to_file:
            self.configuration.global_parameters_write_protocol_to_file = \
                self.config_copy.global_parameters_write_protocol_to_file
            self.configuration.configuration_changed = True

        if self.config_copy.global_parameters_protocol_level != \
                self.configuration.global_parameters_protocol_level:
            self.configuration.global_parameters_protocol_level = \
                self.config_copy.global_parameters_protocol_level
            self.configuration.configuration_changed = True

    def reject(self):
        """
        The Cancel button is pressed, discard the changes and close the GUI window.
        :return: -
        """

        self.configuration.configuration_changed = False
        self.configuration.go_back_to_activity = None
        self.close()

    def closeEvent(self, event):
        self.close()

if __name__ == '__main__':

    # Get configuration parameters.
    configuration = Configuration()

    app = QtWidgets.QApplication(sys.argv)
    myapp = ConfigurationEditor(configuration)
    myapp.showMaximized()
    sys.exit(app.exec_())