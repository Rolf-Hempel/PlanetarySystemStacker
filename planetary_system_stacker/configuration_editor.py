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

from configuration import ConfigurationParameters
from parameter_configuration import Ui_ConfigurationDialog


class ConfigurationEditor(QtWidgets.QFrame, Ui_ConfigurationDialog):
    """
    Update the parameters used by PlanetarySystemStacker which are stored in the configuration
    object. The interaction with the user is through the ConfigurationDialog class.
    """

    def __init__(self, parent_gui, parent=None):
        QtWidgets.QFrame.__init__(self, parent)
        self.setupUi(self)

        self.setFrameShape(QtWidgets.QFrame.Panel)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setObjectName("configuration_editor")

        self.setFixedSize(900, 600)

        self.parent_gui = parent_gui
        self.configuration = parent_gui.configuration

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
        self.fdb_comboBox.addItem('Auto detect color')
        self.fdb_comboBox.addItem('Grayscale')
        self.fdb_comboBox.addItem('RGB')
        self.fdb_comboBox.addItem('BGR')
        self.fdb_comboBox.addItem('Force Bayer RGGB')
        self.fdb_comboBox.addItem('Force Bayer GRBG')
        self.fdb_comboBox.addItem('Force Bayer GBRG')
        self.fdb_comboBox.addItem('Force Bayer BGGR')
        self.fdb_comboBox.activated[str].connect(self.fdb_changed)
        self.fdbm_comboBox.addItem('Bilinear')
        self.fdbm_comboBox.addItem('Variable Number of Gradients')
        self.fdbm_comboBox.addItem('Edge Aware')
        self.fdbm_comboBox.activated[str].connect(self.fdbm_changed)
        self.fn_checkBox.stateChanged.connect(self.fn_changed)
        self.fnt_slider_value.valueChanged['int'].connect(self.fnt_changed)
        self.afm_comboBox.addItem('Surface')
        self.afm_comboBox.addItem('Planet')
        self.afm_comboBox.activated[str].connect(self.afm_changed)
        self.afa_checkBox.stateChanged.connect(self.afa_changed)
        self.afrsf_slider_value.valueChanged['int'].connect(self.afrsf_changed)
        self.afsw_slider_value.valueChanged['int'].connect(self.afsw_changed)
        self.afafp_slider_value.valueChanged['int'].connect(self.afafp_changed)
        self.efs_checkBox.stateChanged.connect(self.efs_changed)
        self.fco_checkBox.stateChanged.connect(self.fco_changed)
        self.gpwptf_checkBox.stateChanged.connect(self.gpwptf_changed)
        self.gpspwr_checkBox.stateChanged.connect(self.gpspwr_changed)
        self.gppl_spinBox.valueChanged['int'].connect(self.gppl_changed)
        self.gpbl_combobox.addItem('auto')
        self.gpbl_combobox.addItem('0')
        self.gpbl_combobox.addItem('1')
        self.gpbl_combobox.addItem('2')
        self.gpbl_combobox.addItem('3')
        self.gpbl_combobox.addItem('4')
        self.gpbl_combobox.activated[str].connect(self.gpbl_changed)
        self.gpif_comboBox.addItem('png')
        self.gpif_comboBox.addItem('tiff')
        self.gpif_comboBox.addItem('fits')
        self.gpif_comboBox.activated[str].connect(self.gpif_changed)
        self.aphbw_slider_value.valueChanged['int'].connect(self.aphbw_changed)
        self.apsw_slider_value.valueChanged['int'].connect(self.apsw_changed)
        self.apst_slider_value.valueChanged['int'].connect(self.apst_changed)
        self.apbt_slider_value.valueChanged['int'].connect(self.apbt_changed)
        self.apfp_comboBox.addItem('Percent of frames to be stacked')
        self.apfp_comboBox.addItem('Number of frames to be stacked')
        self.apfp_comboBox.activated[str].connect(self.apfp_state_changed)
        self.apfp_spinBox.valueChanged['int'].connect(self.apfp_value_changed)
        self.spp_checkBox.stateChanged.connect(self.spp_changed)
        self.ipfn_checkBox.stateChanged.connect(self.ipfn_changed)
        self.nfs_checkBox.stateChanged.connect(self.nfs_changed)
        self.pfs_checkBox.stateChanged.connect(self.pfs_changed)
        self.apbs_checkBox.stateChanged.connect(self.apbs_changed)
        self.nap_checkBox.stateChanged.connect(self.nap_changed)
        self.sfdfs_comboBox.addItem('Off')
        self.sfdfs_comboBox.addItem('1.5x')
        self.sfdfs_comboBox.addItem('2x')
        self.sfdfs_comboBox.addItem('3x')
        self.sfdfs_comboBox.activated[str].connect(self.sfdfs_changed)

        self.restore_standard_values.clicked.connect(self.restore_standard_parameters)

        self.initialize_widgets_and_local_parameters()

    def initialize_widgets_and_local_parameters(self):
        """
        Initialize GUI widgets with current configuration parameter values.

        :return: -
        """

        self.fgw_slider_value.setValue(int((self.config_copy.frames_gauss_width + 1) / 2))
        self.fgw_label_display.setText(str(self.config_copy.frames_gauss_width))
        index = self.fdb_comboBox.findText(self.config_copy.frames_debayering_default,
                                           QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.fdb_comboBox.setCurrentIndex(index)
        index = self.fdbm_comboBox.findText(self.config_copy.frames_debayering_method,
                                            QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.fdbm_comboBox.setCurrentIndex(index)
        index = self.afm_comboBox.findText(self.config_copy.align_frames_mode,
                                           QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.afm_comboBox.setCurrentIndex(index)
        self.afm_activate_deactivate_widgets()
        self.afa_checkBox.setChecked(self.config_copy.align_frames_automation)
        self.afrsf_slider_value.setValue(
            int(100. / self.config_copy.align_frames_rectangle_scale_factor))
        self.afrsf_label_display.setText(
            str(int(100. / self.config_copy.align_frames_rectangle_scale_factor)))
        self.afsw_slider_value.setValue(self.config_copy.align_frames_search_width)
        self.afsw_label_display.setText(str(self.config_copy.align_frames_search_width))
        self.afafp_slider_value.setValue(self.config_copy.align_frames_average_frame_percent)
        self.afafp_label_display.setText(str(self.config_copy.align_frames_average_frame_percent))
        self.efs_checkBox.setChecked(self.config_copy.frames_add_selection_dialog)
        self.fco_checkBox.setChecked(self.config_copy.align_frames_fast_changing_object)
        self.gpwptf_checkBox.setChecked(self.config_copy.global_parameters_write_protocol_to_file)
        self.gpspwr_checkBox.setChecked(
            self.config_copy.global_parameters_store_protocol_with_result)
        self.gppl_spinBox.setValue(self.config_copy.global_parameters_protocol_level)
        if self.config_copy.global_parameters_buffering_level != -1:
            self.gpbl_combobox.setCurrentIndex(self.config_copy.global_parameters_buffering_level+1)
        else:
            self.gpbl_combobox.setCurrentIndex(0)
        index = self.gpif_comboBox.findText(self.config_copy.global_parameters_image_format,
                                           QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.gpif_comboBox.setCurrentIndex(index)
        self.ipfn_checkBox.setChecked(self.config_copy.global_parameters_parameters_in_filename)
        self.nfs_checkBox.setChecked(self.config_copy.global_parameters_stack_number_frames)
        self.pfs_checkBox.setChecked(self.config_copy.global_parameters_stack_percent_frames)
        self.apbs_checkBox.setChecked(self.config_copy.global_parameters_ap_box_size)
        self.nap_checkBox.setChecked(self.config_copy.global_parameters_ap_number)

        self.aphbw_slider_value.setValue(self.config_copy.alignment_points_half_box_width * 2)
        self.aphbw_label_display.setText(str(self.config_copy.alignment_points_half_box_width * 2))
        self.apsw_slider_value.setValue(self.config_copy.alignment_points_search_width)
        self.apsw_label_display.setText(str(self.config_copy.alignment_points_search_width))
        self.apst_slider_value.setValue(round(self.config_copy.alignment_points_structure_threshold * 100))
        self.apst_label_display.setText(str(self.config_copy.alignment_points_structure_threshold))
        self.apbt_slider_value.setValue(self.config_copy.alignment_points_brightness_threshold)
        self.apbt_label_display.setText(str(self.config_copy.alignment_points_brightness_threshold))
        if self.config_copy.alignment_points_frame_percent != -1:
            self.apfp_comboBox.setCurrentIndex(0)
            self.apfp_spinBox.setMaximum(100)
            self.apfp_spinBox.setValue(self.config_copy.alignment_points_frame_percent)
        else:
            self.apfp_comboBox.setCurrentIndex(1)
            self.apfp_spinBox.setMaximum(1000000)
            self.apfp_spinBox.setValue(self.config_copy.alignment_points_frame_number)
        self.spp_checkBox.setChecked(self.config_copy.global_parameters_include_postprocessing)
        self.fn_checkBox.setChecked(self.config_copy.frames_normalization)
        self.fn_activate_deactivate_widgets()
        self.fnt_slider_value.setValue(self.config_copy.frames_normalization_threshold)
        self.fnt_label_display.setText(str(self.config_copy.frames_normalization_threshold))
        self.ipfn_activate_deactivate_widgets()
        index = self.sfdfs_comboBox.findText(self.config_copy.stack_frames_drizzle_factor_string,
                                           QtCore.Qt.MatchFixedString)
        if index >= 0:
            self.sfdfs_comboBox.setCurrentIndex(index)

    def fgw_changed(self, value):
        """
        When the widget changes its value, update the corresponding entry in the configuration copy.
        Please note that for some parameters the representations differ.

        The methods following this one do the same for all other configuration parameters.

        :param value: New value sent by widget
        :return: -
        """

        self.config_copy.frames_gauss_width = 2 * value - 1
        self.fgw_label_display.setText(str(self.config_copy.frames_gauss_width))

    def fdb_changed(self, value):
        self.config_copy.frames_debayering_default = value

    def fdbm_changed(self, value):
        self.config_copy.frames_debayering_method = value

    def afm_changed(self, value):
        self.config_copy.align_frames_mode = value
        self.afm_activate_deactivate_widgets()

    def afm_activate_deactivate_widgets(self):
        """
        Special case: Depending on the "frame stabilization mode" selected, some other parameters
        do not make sense, so they are greyed out (for case 'Planet').

        :return:-
        """

        if self.config_copy.align_frames_mode == 'Planet':
            self.afa_checkBox.setEnabled(False)
            self.afrsf_label_display.setEnabled(False)
            self.afrsf_slider_value.setEnabled(False)
            self.afrsf_label_parameter.setEnabled(False)
            self.afsw_label_parameter.setEnabled(False)
            self.afsw_slider_value.setEnabled(False)
            self.afsw_label_display.setEnabled(False)
        else:
            self.afa_checkBox.setEnabled(True)
            self.afrsf_label_display.setEnabled(True)
            self.afrsf_slider_value.setEnabled(True)
            self.afrsf_label_parameter.setEnabled(True)
            self.afsw_label_parameter.setEnabled(True)
            self.afsw_slider_value.setEnabled(True)
            self.afsw_label_display.setEnabled(True)

    def fn_activate_deactivate_widgets(self):
        if self.config_copy.frames_normalization:
            self.fnt_label_display.setEnabled(True)
            self.fnt_slider_value.setEnabled(True)
            self.fnt_label_parameter.setEnabled(True)
        else:
            self.fnt_label_display.setEnabled(False)
            self.fnt_slider_value.setEnabled(False)
            self.fnt_label_parameter.setEnabled(False)

    def ipfn_activate_deactivate_widgets(self):
        if self.config_copy.global_parameters_parameters_in_filename:
            self.nfs_checkBox.setEnabled(True)
            self.pfs_checkBox.setEnabled(True)
            self.apbs_checkBox.setEnabled(True)
            self.nap_checkBox.setEnabled(True)
        else:
            self.nfs_checkBox.setEnabled(False)
            self.pfs_checkBox.setEnabled(False)
            self.apbs_checkBox.setEnabled(False)
            self.nap_checkBox.setEnabled(False)

    def afa_changed(self, state):
        self.config_copy.align_frames_automation = (state == QtCore.Qt.Checked)

    def afrsf_changed(self, value):
        self.config_copy.align_frames_rectangle_scale_factor = 100. / value

    def afsw_changed(self, value):
        self.config_copy.align_frames_search_width = value

    def afafp_changed(self, value):
        self.config_copy.align_frames_average_frame_percent = value

    def efs_changed(self, state):
        self.config_copy.frames_add_selection_dialog = (state == QtCore.Qt.Checked)

    def fco_changed(self, state):
        self.config_copy.align_frames_fast_changing_object = (state == QtCore.Qt.Checked)

    def gpwptf_changed(self, state):
        self.config_copy.global_parameters_write_protocol_to_file = (state == QtCore.Qt.Checked)

    def gpspwr_changed(self, state):
        self.config_copy.global_parameters_store_protocol_with_result = (state == QtCore.Qt.Checked)

    def gppl_changed(self, value):
        self.config_copy.global_parameters_protocol_level = value

    def gpbl_changed(self, value):
        if value == "auto":
            self.config_copy.global_parameters_buffering_level = -1
        else:
            self.config_copy.global_parameters_buffering_level = int(value)

    def gpif_changed(self, value):
        self.config_copy.global_parameters_image_format = value

    def aphbw_changed(self, value):
        self.config_copy.alignment_points_half_box_width = int(value / 2)

    def apsw_changed(self, value):
        self.config_copy.alignment_points_search_width = value

    def apst_changed(self, value):
        self.config_copy.alignment_points_structure_threshold = value / 100.
        self.apst_label_display.setText(str(self.config_copy.alignment_points_structure_threshold))

    def apbt_changed(self, value):
        self.config_copy.alignment_points_brightness_threshold = value

    def apfp_state_changed(self, value):
        self.apfp_spinBox.blockSignals(True)
        if value == 'Percent of frames to be stacked':
            self.apfp_spinBox.setMaximum(100)
            if self.config_copy.alignment_points_frame_percent > 0:
                self.apfp_spinBox.setValue(self.config_copy.alignment_points_frame_percent)
            else:
                self.apfp_spinBox.setValue(0)
                self.apfp_spinBox.clear()
        else:
            self.apfp_spinBox.setMaximum(1000000)
            if self.config_copy.alignment_points_frame_number > 0:
                self.apfp_spinBox.setValue(self.config_copy.alignment_points_frame_number)
            else:
                self.apfp_spinBox.setValue(0)
                self.apfp_spinBox.clear()
        self.apfp_spinBox.blockSignals(False)

    def apfp_value_changed(self, value):
        if self.apfp_comboBox.currentIndex() == 0:
            self.config_copy.alignment_points_frame_percent = value
        else:
            self.config_copy.alignment_points_frame_number = value

    def spp_changed(self, state):
        self.config_copy.global_parameters_include_postprocessing = (state == QtCore.Qt.Checked)

    def fn_changed(self, state):
        self.config_copy.frames_normalization = (state == QtCore.Qt.Checked)
        self.fn_activate_deactivate_widgets()

    def fnt_changed(self, value):
        self.config_copy.frames_normalization_threshold = value
        self.fnt_label_display.setText(str(self.config_copy.frames_normalization_threshold))

    def ipfn_changed(self, state):
        self.config_copy.global_parameters_parameters_in_filename = (state == QtCore.Qt.Checked)
        self.ipfn_activate_deactivate_widgets()

    def nfs_changed(self, state):
        self.config_copy.global_parameters_stack_number_frames = (state == QtCore.Qt.Checked)

    def pfs_changed(self, state):
        self.config_copy.global_parameters_stack_percent_frames = (state == QtCore.Qt.Checked)

    def apbs_changed(self, state):
        self.config_copy.global_parameters_ap_box_size = (state == QtCore.Qt.Checked)

    def nap_changed(self, state):
        self.config_copy.global_parameters_ap_number = (state == QtCore.Qt.Checked)

    def sfdfs_changed(self, value):
        self.config_copy.stack_frames_drizzle_factor_string = value

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

        go_back_to_activities = []

        if self.config_copy.alignment_points_frame_percent != \
                self.configuration.alignment_points_frame_percent or \
                self.config_copy.alignment_points_frame_number != \
                self.configuration.alignment_points_frame_number:
            # Additional to the state of the comboBox check if a valid number has been entered.
            if self.apfp_comboBox.currentIndex() == 0 and \
                    self.config_copy.alignment_points_frame_percent > 0:
                self.configuration.alignment_points_frame_percent = \
                    self.config_copy.alignment_points_frame_percent
                self.configuration.alignment_points_frame_number = -1
                self.configuration.configuration_changed = True
                go_back_to_activities.append('Compute frame qualities')
            elif self.apfp_comboBox.currentIndex() == 1 and \
                    self.config_copy.alignment_points_frame_number > 0:
                self.configuration.alignment_points_frame_percent = -1
                self.configuration.alignment_points_frame_number = \
                    self.config_copy.alignment_points_frame_number
                self.configuration.configuration_changed = True
                go_back_to_activities.append('Compute frame qualities')

        if self.config_copy.alignment_points_brightness_threshold != \
                self.configuration.alignment_points_brightness_threshold:
            self.configuration.alignment_points_brightness_threshold = \
                self.config_copy.alignment_points_brightness_threshold
            self.configuration.configuration_changed = True
            go_back_to_activities.append('Set alignment points')

        if self.config_copy.alignment_points_structure_threshold != \
                self.configuration.alignment_points_structure_threshold:
            self.configuration.alignment_points_structure_threshold = \
                self.config_copy.alignment_points_structure_threshold
            self.configuration.configuration_changed = True
            go_back_to_activities.append('Set alignment points')

        if self.config_copy.alignment_points_search_width != \
                self.configuration.alignment_points_search_width:
            self.configuration.alignment_points_search_width = \
                self.config_copy.alignment_points_search_width
            self.configuration.configuration_changed = True
            go_back_to_activities.append('Set alignment points')

        if self.config_copy.alignment_points_half_box_width != \
                self.configuration.alignment_points_half_box_width:
            self.configuration.alignment_points_half_box_width = \
                self.config_copy.alignment_points_half_box_width
            self.configuration.configuration_changed = True
            go_back_to_activities.append('Set alignment points')

        if self.config_copy.align_frames_average_frame_percent != \
                self.configuration.align_frames_average_frame_percent:
            self.configuration.align_frames_average_frame_percent = \
                self.config_copy.align_frames_average_frame_percent
            self.configuration.configuration_changed = True
            go_back_to_activities.append('Align frames')

        if self.config_copy.align_frames_search_width != \
                self.configuration.align_frames_search_width:
            self.configuration.align_frames_search_width = \
                self.config_copy.align_frames_search_width
            self.configuration.configuration_changed = True
            go_back_to_activities.append('Align frames')

        if self.config_copy.align_frames_rectangle_scale_factor != \
                self.configuration.align_frames_rectangle_scale_factor:
            self.configuration.align_frames_rectangle_scale_factor = \
                self.config_copy.align_frames_rectangle_scale_factor
            self.configuration.configuration_changed = True
            go_back_to_activities.append('Align frames')

        if self.config_copy.align_frames_automation != self.configuration.align_frames_automation:
            self.configuration.align_frames_automation = self.config_copy.align_frames_automation
            self.configuration.configuration_changed = True
            go_back_to_activities.append('Align frames')

        if self.config_copy.align_frames_mode != self.configuration.align_frames_mode:
            self.configuration.align_frames_mode = self.config_copy.align_frames_mode
            self.configuration.configuration_changed = True
            go_back_to_activities.append('Align frames')

        if self.config_copy.frames_gauss_width != self.configuration.frames_gauss_width:
            self.configuration.frames_gauss_width = self.config_copy.frames_gauss_width
            self.configuration.configuration_changed = True
            go_back_to_activities.append('Read frames')

        if self.config_copy.frames_debayering_default != self.configuration.frames_debayering_default:
            self.configuration.frames_debayering_default = self.config_copy.frames_debayering_default
            self.configuration.configuration_changed = True
            go_back_to_activities.append('Read frames')

        if self.config_copy.frames_debayering_method != self.configuration.frames_debayering_method:
            self.configuration.frames_debayering_method = self.config_copy.frames_debayering_method
            self.configuration.configuration_changed = True
            go_back_to_activities.append('Read frames')

        if self.config_copy.frames_normalization != self.configuration.frames_normalization:
            self.configuration.frames_normalization = self.config_copy.frames_normalization
            self.configuration.configuration_changed = True
            go_back_to_activities.append('Compute frame qualities')

        if self.config_copy.frames_normalization_threshold != \
                self.configuration.frames_normalization_threshold:
            self.configuration.frames_normalization_threshold = \
                self.config_copy.frames_normalization_threshold
            self.configuration.configuration_changed = True
            go_back_to_activities.append('Compute frame qualities')

        if self.config_copy.frames_add_selection_dialog != \
                self.configuration.frames_add_selection_dialog:
            self.configuration.frames_add_selection_dialog = \
                self.config_copy.frames_add_selection_dialog
            self.configuration.configuration_changed = True

        if self.config_copy.align_frames_fast_changing_object != \
                self.configuration.align_frames_fast_changing_object:
            self.configuration.align_frames_fast_changing_object = \
                self.config_copy.align_frames_fast_changing_object
            self.configuration.configuration_changed = True
            go_back_to_activities.append('Align frames')

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

        if self.config_copy.global_parameters_buffering_level != \
                self.configuration.global_parameters_buffering_level:
            self.configuration.global_parameters_buffering_level = \
                self.config_copy.global_parameters_buffering_level
            self.configuration.configuration_changed = True

        if self.config_copy.global_parameters_include_postprocessing != \
                self.configuration.global_parameters_include_postprocessing:
            self.configuration.global_parameters_include_postprocessing = \
                self.config_copy.global_parameters_include_postprocessing
            self.configuration.configuration_changed = True

        if self.config_copy.global_parameters_image_format != \
                self.configuration.global_parameters_image_format:
            self.configuration.global_parameters_image_format = \
                self.config_copy.global_parameters_image_format
            self.configuration.configuration_changed = True

        if self.config_copy.global_parameters_parameters_in_filename != \
                self.configuration.global_parameters_parameters_in_filename:
            self.configuration.global_parameters_parameters_in_filename = \
                self.config_copy.global_parameters_parameters_in_filename
            self.configuration.configuration_changed = True

        if self.config_copy.global_parameters_stack_number_frames != \
                self.configuration.global_parameters_stack_number_frames:
            self.configuration.global_parameters_stack_number_frames = \
                self.config_copy.global_parameters_stack_number_frames
            self.configuration.configuration_changed = True

        if self.config_copy.global_parameters_stack_percent_frames != \
                self.configuration.global_parameters_stack_percent_frames:
            self.configuration.global_parameters_stack_percent_frames = \
                self.config_copy.global_parameters_stack_percent_frames
            self.configuration.configuration_changed = True

        if self.config_copy.global_parameters_ap_box_size != \
                self.configuration.global_parameters_ap_box_size:
            self.configuration.global_parameters_ap_box_size = \
                self.config_copy.global_parameters_ap_box_size
            self.configuration.configuration_changed = True

        if self.config_copy.global_parameters_ap_number != \
                self.configuration.global_parameters_ap_number:
            self.configuration.global_parameters_ap_number = \
                self.config_copy.global_parameters_ap_number
            self.configuration.configuration_changed = True

        if self.config_copy.stack_frames_drizzle_factor_string != self.configuration.stack_frames_drizzle_factor_string:
            self.configuration.stack_frames_drizzle_factor_string = self.config_copy.stack_frames_drizzle_factor_string
            self.configuration.configuration_changed = True
            go_back_to_activities.append('Stack frames')

        # Set dependent parameters.
        self.configuration.set_derived_parameters()

        # If the change of parameters require going back in the workflow, find the latest phase
        # which is safe to go back to.
        if go_back_to_activities:
            self.parent_gui.signal_set_go_back_activity.emit(go_back_to_activities)

        self.close()

    def reject(self):
        """
        The Cancel button is pressed, discard the changes and close the GUI window.
        :return: -
        """

        self.configuration.configuration_changed = False
        self.configuration.go_back_to_activity = None
        self.close()

    def closeEvent(self, event):

        # Write ".ini" file if it does not exist yet or parameters have changed.
        if not self.configuration.config_file_exists or self.configuration.configuration_changed:
            self.configuration.write_config()

        self.parent_gui.display_widget(None, display=False)
        self.close()
