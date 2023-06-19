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
from configparser import ConfigParser
from copy import deepcopy
from os.path import expanduser, join, isfile, dirname
from os.path import splitext

import psutil
from numpy import uint8

from exceptions import ArgumentError
from miscellaneous import Miscellaneous

# Set the current software version.
PSS_Version = "PlanetarySystemStacker 0.9.6"
# PSS_Version = "PlanetarySystemStacker"


class ConfigurationParameters(object):
    def __init__(self):
        self.hidden_parameters_current_dir = None
        self.hidden_parameters_main_window_x0 = None
        self.hidden_parameters_main_window_y0 = None
        self.hidden_parameters_main_window_width = None
        self.hidden_parameters_main_window_height = None
        self.hidden_parameters_main_window_maximized = None
        self.global_parameters_display_quickstart = None
        self.global_parameters_version = None
        self.global_parameters_protocol_level = None
        self.global_parameters_write_protocol_to_file = None
        self.global_parameters_store_protocol_with_result = None
        self.global_parameters_buffering_level = None
        self.global_parameters_maximum_memory_active = None
        self.global_parameters_maximum_memory_amount = None
        self.global_parameters_include_postprocessing = None
        self.global_parameters_image_format = None
        self.global_parameters_parameters_in_filename = None
        self.global_parameters_stack_number_frames = None
        self.global_parameters_stack_percent_frames = None
        self.global_parameters_ap_box_size = None
        self.global_parameters_ap_number = None
        self.frames_gauss_width = None
        self.frames_debayering_default = None
        self.frames_debayering_method = None
        self.frames_normalization = None
        self.frames_normalization_threshold = None
        self.frames_add_selection_dialog = None
        self.align_frames_fast_changing_object = None
        self.align_frames_mode = None
        self.align_frames_automation = None
        self.align_frames_rectangle_scale_factor = None
        self.align_frames_search_width = None
        self.align_frames_average_frame_percent = None
        self.alignment_points_half_box_width = None
        self.alignment_points_search_width = None
        self.alignment_points_structure_threshold = None
        self.alignment_points_brightness_threshold = None
        self.alignment_points_frame_percent = None
        self.alignment_points_frame_number = None
        self.stack_frames_drizzle_factor_string = None

    def set_defaults(self):
        self.hidden_parameters_current_dir = expanduser("~")
        self.hidden_parameters_main_window_x0 = 100
        self.hidden_parameters_main_window_y0 = 100
        self.hidden_parameters_main_window_width = 1200
        self.hidden_parameters_main_window_height = 800
        self.hidden_parameters_main_window_maximized = False
        self.global_parameters_display_quickstart = True
        self.global_parameters_version = PSS_Version
        self.global_parameters_protocol_level = 1
        self.global_parameters_write_protocol_to_file = False
        self.global_parameters_store_protocol_with_result = False
        self.global_parameters_buffering_level = -1
        self.global_parameters_maximum_memory_active = False
        self.global_parameters_maximum_memory_amount = \
            max(1, int(dict(psutil.virtual_memory()._asdict())['available'] / 1e9))
        self.global_parameters_include_postprocessing = True
        self.global_parameters_image_format = "png"
        self.global_parameters_parameters_in_filename = False
        self.global_parameters_stack_number_frames = False
        self.global_parameters_stack_percent_frames = False
        self.global_parameters_ap_box_size = False
        self.global_parameters_ap_number = False
        self.frames_gauss_width = 7
        self.frames_debayering_default = 'Auto detect color'
        self.frames_debayering_method = 'Bilinear'
        self.frames_normalization = True
        self.frames_normalization_threshold = 15
        self.frames_add_selection_dialog = False
        self.align_frames_fast_changing_object = True
        self.align_frames_mode = 'Surface'
        self.align_frames_automation = True
        self.align_frames_rectangle_scale_factor = 3.
        self.align_frames_search_width = 34
        self.align_frames_average_frame_percent = 5
        self.alignment_points_search_width = 14
        self.alignment_points_frame_percent = 10
        self.alignment_points_frame_number = -1
        self.stack_frames_drizzle_factor_string = "Off"
        self.set_defaults_ap_editing()

    def set_defaults_ap_editing(self):
        self.alignment_points_half_box_width = 24
        self.alignment_points_structure_threshold = 0.04
        self.alignment_points_brightness_threshold = 10

    def copy_from_config_object(self, configuration_object):
        self.hidden_parameters_current_dir = configuration_object.hidden_parameters_current_dir
        self.hidden_parameters_main_window_x0 = \
            configuration_object.hidden_parameters_main_window_x0
        self.hidden_parameters_main_window_y0 = \
            configuration_object.hidden_parameters_main_window_y0
        self.hidden_parameters_main_window_width = \
            configuration_object.hidden_parameters_main_window_width
        self.hidden_parameters_main_window_height = \
            configuration_object.hidden_parameters_main_window_height
        self.hidden_parameters_main_window_maximized = \
            configuration_object.hidden_parameters_main_window_maximized
        self.global_parameters_display_quickstart = \
            configuration_object.global_parameters_display_quickstart
        self.global_parameters_version = configuration_object.global_parameters_version
        self.global_parameters_protocol_level = \
            configuration_object.global_parameters_protocol_level
        self.global_parameters_write_protocol_to_file = \
            configuration_object.global_parameters_write_protocol_to_file
        self.global_parameters_store_protocol_with_result = \
            configuration_object.global_parameters_store_protocol_with_result
        self.global_parameters_buffering_level = \
            configuration_object.global_parameters_buffering_level
        self.global_parameters_maximum_memory_active = \
            configuration_object.global_parameters_maximum_memory_active
        self.global_parameters_maximum_memory_amount = \
            configuration_object.global_parameters_maximum_memory_amount
        self.global_parameters_include_postprocessing = \
            configuration_object.global_parameters_include_postprocessing
        self.global_parameters_image_format = \
            configuration_object.global_parameters_image_format
        self.global_parameters_parameters_in_filename = \
            configuration_object.global_parameters_parameters_in_filename
        self.global_parameters_stack_number_frames = \
            configuration_object.global_parameters_stack_number_frames
        self.global_parameters_stack_percent_frames = \
            configuration_object.global_parameters_stack_percent_frames
        self.global_parameters_ap_box_size = \
            configuration_object.global_parameters_ap_box_size
        self.global_parameters_ap_number = \
            configuration_object.global_parameters_ap_number
        self.frames_gauss_width = configuration_object.frames_gauss_width
        self.frames_debayering_default = configuration_object.frames_debayering_default
        self.frames_debayering_method = configuration_object.frames_debayering_method
        self.frames_normalization = configuration_object.frames_normalization
        self.frames_normalization_threshold = configuration_object.frames_normalization_threshold
        self.frames_add_selection_dialog = configuration_object.frames_add_selection_dialog
        self.align_frames_fast_changing_object = \
            configuration_object.align_frames_fast_changing_object
        self.align_frames_mode = configuration_object.align_frames_mode
        self.align_frames_automation = configuration_object.align_frames_automation
        self.align_frames_rectangle_scale_factor = \
            configuration_object.align_frames_rectangle_scale_factor
        self.align_frames_search_width = configuration_object.align_frames_search_width
        self.align_frames_average_frame_percent = \
            configuration_object.align_frames_average_frame_percent
        self.alignment_points_half_box_width = configuration_object.alignment_points_half_box_width
        self.alignment_points_search_width = configuration_object.alignment_points_search_width
        self.alignment_points_structure_threshold = \
            configuration_object.alignment_points_structure_threshold
        self.alignment_points_brightness_threshold = \
            configuration_object.alignment_points_brightness_threshold
        self.alignment_points_frame_percent = configuration_object.alignment_points_frame_percent
        self.alignment_points_frame_number = configuration_object.alignment_points_frame_number
        self.stack_frames_drizzle_factor_string = \
            configuration_object.stack_frames_drizzle_factor_string


class Configuration(object):
    def __init__(self):
        # self.global_parameters_version = "PlanetarySystemStacker"
        self.global_parameters_version = PSS_Version

        # Set fixed parameters which are hidden from the user. Hidden parameters which are
        # changeable are stored in the configuration object.

        # Look for PSS icon in several places:
        python_dir = dirname(sys.executable)
        icon_locations = ['PSS-Icon-64.png', join('Icons', 'PSS-Icon-64.png'),
                          join(python_dir, "Lib", "site-packages", 'planetary_system_stacker',
                                'Icons', 'PSS-Icon-64.png')]

        self.window_icon = None
        for location in icon_locations:
            if isfile(location):
                self.window_icon = location
                break

        self.frames_mono_channel = 'panchromatic'
        self.frames_color_difference_threshold = 0
        self.frames_bayer_max_noise_diff_green = 2.
        self.frames_bayer_min_distance_from_blue = 99.5

        self.rank_frames_pixel_stride = 2
        self.rank_frames_method = "Laplace"

        self.align_frames_method = "MultiLevelCorrelation"
        self.align_frames_rectangle_black_threshold = 10240
        self.align_frames_rectangle_min_fraction = 0.7
        self.align_frames_rectangle_stride = 2
        self.align_frames_border_width = 10
        self.align_frames_sampling_stride = 2
        self.align_frames_min_stabilization_patch_fraction = 0.2
        self.align_frames_max_stabilization_patch_fraction = 0.7
        self.align_frames_max_search_width = 150
        self.align_frames_best_frames_window_extension = 2

        self.alignment_points_min_half_box_width = 10
        self.alignment_points_contrast_threshold = 0
        self.alignment_points_dim_fraction_threshold = 0.6
        self.alignment_points_rank_method = "Laplace"
        self.alignment_points_rank_pixel_stride = 2
        self.alignment_points_de_warp = True
        self.alignment_points_method = 'MultiLevelCorrelation'
        self.alignment_points_sampling_stride = 2
        self.alignment_points_local_search_subpixel = False
        self.alignment_points_penalty_factor = 0.00025

        self.stack_frames_suffix = "_pss"
        self.stack_frames_background_blend_threshold = 0.2
        self.stack_frames_background_fraction = 0.3
        self.stack_frames_background_patch_size = 100

        self.postproc_suffix = "_gpp"
        self.postproc_max_layers = 10
        self.postproc_bi_range_standard = 13
        self.postproc_max_shift = 5
        self.postproc_max_ram_percentage = 20.
        self.postproc_blinking_period = 1.
        self.postproc_idle_loop_time = 0.2

        # Initialize the ConfigParser object for parameters which the user can change.
        self.config_parser_object = ConfigParser()

        # Create and initialize the central data object for postprocessing.
        self.postproc_data_object = PostprocDataObject(self.postproc_suffix)

    def initialize_configuration(self, read_from_file=True):
        """
        Initialize the configuration with parameters stored during the previous PSS run. This
        initialization is skipped if PSS is executed from the command line. Finally, set derived
        parameters, i.e. parameters which are computed from other parameters.

        :param read_from_file: If True, read parameters from a configuration file.
                               If False, skip this step.
        :return: -
        """

        if read_from_file:
            # The config file for persistent parameter storage is located in the user's home
            # directory, as is the detailed logfile.
            self.home = expanduser("~")
            self.config_filename = join(self.home, ".PlanetarySystemStacker.ini")
            self.protocol_filename = join(self.home, "PlanetarySystemStacker.log")

            # Determine if there is a configuration file from a previous run.
            self.config_file_exists = isfile(self.config_filename)
        else:
            self.config_file_exists = False

        # If an existing config file is found, read it in. Set flag to indicate if parameters were
        # read from file successfully.
        self.configuration_read = False
        if self.config_file_exists:
            try:
                self.config_parser_object = self.read_config()
                self.configuration_read = True
            except:
                self.configuration_read = False

        # The configuration could not be read from a file, or versions did not match. Create a
        # new set of standard parameters.
        if not self.configuration_read:
            configuration_parameters = ConfigurationParameters()
            configuration_parameters.set_defaults()

            # Set current configuration parameters to the new values.
            self.import_from_configuration_parameters(configuration_parameters)

            # Create and initialize the central data object for postprocessing.
            self.postproc_data_object = PostprocDataObject(self.postproc_suffix)

        # Compute parameters which are derived from other parameters.
        self.set_derived_parameters()

        # Mark the configuration as not changed.
        self.configuration_changed = False
        self.go_back_to_activity = None

    def import_from_configuration_parameters(self, configuration_parameters):
        """
        Set all current parameters to the corresponding values of a ConfigurationParameters object.

        :param configuration_parameters: ConfigurationParameters object with new parameter values.
        :return: -
        """

        self.hidden_parameters_current_dir = configuration_parameters.hidden_parameters_current_dir
        self.hidden_parameters_main_window_x0 = \
            configuration_parameters.hidden_parameters_main_window_x0
        self.hidden_parameters_main_window_y0 = \
            configuration_parameters.hidden_parameters_main_window_y0
        self.hidden_parameters_main_window_width = \
            configuration_parameters.hidden_parameters_main_window_width
        self.hidden_parameters_main_window_height = \
            configuration_parameters.hidden_parameters_main_window_height
        self.hidden_parameters_main_window_maximized = \
            configuration_parameters.hidden_parameters_main_window_maximized
        self.global_parameters_display_quickstart = \
            configuration_parameters.global_parameters_display_quickstart
        self.global_parameters_version = configuration_parameters.global_parameters_version
        self.global_parameters_protocol_level = \
            configuration_parameters.global_parameters_protocol_level
        self.global_parameters_write_protocol_to_file = \
            configuration_parameters.global_parameters_write_protocol_to_file
        self.global_parameters_store_protocol_with_result = \
            configuration_parameters.global_parameters_store_protocol_with_result
        self.global_parameters_buffering_level = \
            configuration_parameters.global_parameters_buffering_level
        self.global_parameters_maximum_memory_active = \
            configuration_parameters.global_parameters_maximum_memory_active
        self.global_parameters_maximum_memory_amount = \
            configuration_parameters.global_parameters_maximum_memory_amount
        self.global_parameters_include_postprocessing = \
            configuration_parameters.global_parameters_include_postprocessing
        self.global_parameters_image_format = \
            configuration_parameters.global_parameters_image_format
        self.global_parameters_parameters_in_filename = \
            configuration_parameters.global_parameters_parameters_in_filename
        self.global_parameters_stack_number_frames = \
            configuration_parameters.global_parameters_stack_number_frames
        self.global_parameters_stack_percent_frames = \
            configuration_parameters.global_parameters_stack_percent_frames
        self.global_parameters_ap_box_size = \
            configuration_parameters.global_parameters_ap_box_size
        self.global_parameters_ap_number = \
            configuration_parameters.global_parameters_ap_number
        self.frames_gauss_width = configuration_parameters.frames_gauss_width
        self.frames_debayering_default = configuration_parameters.frames_debayering_default
        self.frames_debayering_method = configuration_parameters.frames_debayering_method
        self.frames_normalization = configuration_parameters.frames_normalization
        self.frames_normalization_threshold = \
            configuration_parameters.frames_normalization_threshold
        self.frames_add_selection_dialog = \
            configuration_parameters.frames_add_selection_dialog
        self.align_frames_fast_changing_object = \
            configuration_parameters.align_frames_fast_changing_object
        self.align_frames_mode = configuration_parameters.align_frames_mode
        self.align_frames_automation = configuration_parameters.align_frames_automation
        self.align_frames_rectangle_scale_factor = \
            configuration_parameters.align_frames_rectangle_scale_factor
        self.align_frames_search_width = configuration_parameters.align_frames_search_width
        self.align_frames_average_frame_percent = \
            configuration_parameters.align_frames_average_frame_percent
        self.alignment_points_half_box_width = \
            configuration_parameters.alignment_points_half_box_width
        self.alignment_points_search_width = configuration_parameters.alignment_points_search_width
        self.alignment_points_structure_threshold = \
            configuration_parameters.alignment_points_structure_threshold
        self.alignment_points_brightness_threshold = \
            configuration_parameters.alignment_points_brightness_threshold
        self.alignment_points_frame_percent = \
            configuration_parameters.alignment_points_frame_percent
        self.alignment_points_frame_number = \
            configuration_parameters.alignment_points_frame_number
        self.stack_frames_drizzle_factor_string = \
            configuration_parameters.stack_frames_drizzle_factor_string

    def export_to_configuration_parameters(self, configuration_parameters):
        """
        Set all values of a ConfigurarionParameters object to the current parameters.

        :param configuration_parameters: ConfigurarionParameters object to be updated.
        :return: -
        """

        configuration_parameters.hidden_parameters_current_dir = self.hidden_parameters_current_dir
        configuration_parameters.hidden_parameters_main_window_x0 = \
            self.hidden_parameters_main_window_x0
        configuration_parameters.hidden_parameters_main_window_y0 = \
            self.hidden_parameters_main_window_y0
        configuration_parameters.hidden_parameters_main_window_width = \
            self.hidden_parameters_main_window_width
        configuration_parameters.hidden_parameters_main_window_height = \
            self.hidden_parameters_main_window_height
        configuration_parameters.hidden_parameters_main_window_maximized = \
            self.hidden_parameters_main_window_maximized
        configuration_parameters.global_parameters_display_quickstart = \
            self.global_parameters_display_quickstart
        configuration_parameters.global_parameters_version = self.global_parameters_version
        configuration_parameters.global_parameters_protocol_level = \
            self.global_parameters_protocol_level
        configuration_parameters.global_parameters_write_protocol_to_file = \
            self.global_parameters_write_protocol_to_file
        configuration_parameters.global_parameters_store_protocol_with_result = \
            self.global_parameters_store_protocol_with_result
        configuration_parameters.global_parameters_include_postprocessing = \
            self.global_parameters_include_postprocessing
        configuration_parameters.global_parameters_image_format = \
            self.global_parameters_image_format
        configuration_parameters.global_parameters_parameters_in_filename = \
            self.global_parameters_parameters_in_filename
        configuration_parameters.global_parameters_stack_number_frames = \
            self.global_parameters_stack_number_frames
        configuration_parameters.global_parameters_stack_percent_frames = \
            self.global_parameters_stack_percent_frames
        configuration_parameters.global_parameters_ap_box_size = \
            self.global_parameters_ap_box_size
        configuration_parameters.global_parameters_ap_number = \
            self.global_parameters_ap_number

        configuration_parameters.frames_gauss_width = self.frames_gauss_width
        configuration_parameters.frames_debayering_default = self.frames_debayering_default
        configuration_parameters.frames_debayering_method = self.frames_debayering_method
        configuration_parameters.frames_normalization = self.frames_normalization
        configuration_parameters.frames_normalization_threshold = self.frames_normalization_threshold
        configuration_parameters.frames_add_selection_dialog = self.frames_add_selection_dialog

        configuration_parameters.align_frames_fast_changing_object = \
            self.align_frames_fast_changing_object
        configuration_parameters.align_frames_mode = self.align_frames_mode
        configuration_parameters.align_frames_automation = self.align_frames_automation
        configuration_parameters.align_frames_rectangle_scale_factor = \
            self.align_frames_rectangle_scale_factor
        configuration_parameters.align_frames_search_width = self.align_frames_search_width
        configuration_parameters.align_frames_average_frame_percent = \
            self.align_frames_average_frame_percent

        configuration_parameters.alignment_points_half_box_width = \
            self.alignment_points_half_box_width
        configuration_parameters.alignment_points_search_width = self.alignment_points_search_width
        configuration_parameters.alignment_points_structure_threshold = \
            self.alignment_points_structure_threshold
        configuration_parameters.alignment_points_brightness_threshold = \
            self.alignment_points_brightness_threshold
        configuration_parameters.alignment_points_frame_percent = \
            self.alignment_points_frame_percent
        configuration_parameters.alignment_points_frame_number = \
            self.alignment_points_frame_number
        configuration_parameters.stack_frames_drizzle_factor_string = \
            self.stack_frames_drizzle_factor_string

    def get_all_parameters_from_configparser(self, conf):
        """
        All parameters which can be modified by the user are stored in the ConfigParser object.
        This way they can be written to a file or read from there. Read parameter values from a
        ConfigParser object and store them with the configuration object.

        :param conf: ConfigParser object
        :return: -
        """

        # Create an object with default parameters. They are inserted if the corresponding value
        # cannot be read from the config parser object.
        default_conf_obj = ConfigurationParameters()
        default_conf_obj.set_defaults()

        # Read the PSS version with which the ini file was created.
        self.global_parameters_version_imported_from = get_from_conf(conf, 'Global parameters',
            'version', ' version number could not be identified')

        self.hidden_parameters_current_dir = get_from_conf(conf, 'Hidden parameters',
            'current directory', default_conf_obj.hidden_parameters_current_dir)
        self.hidden_parameters_main_window_x0 = get_from_conf(conf, 'Hidden parameters',
            'main window x0', default_conf_obj.hidden_parameters_main_window_x0)
        self.hidden_parameters_main_window_y0 = get_from_conf(conf, 'Hidden parameters',
            'main window y0', default_conf_obj.hidden_parameters_main_window_y0)
        self.hidden_parameters_main_window_width = get_from_conf(conf, 'Hidden parameters',
            'main window width', default_conf_obj.hidden_parameters_main_window_width)
        self.hidden_parameters_main_window_height = get_from_conf(conf, 'Hidden parameters',
            'main window height', default_conf_obj.hidden_parameters_main_window_height)
        self.hidden_parameters_main_window_maximized = get_from_conf(conf, 'Hidden parameters',
            'main window maximized', default_conf_obj.hidden_parameters_main_window_maximized)

        self.global_parameters_display_quickstart = get_from_conf(conf, 'Global parameters',
            'display quickstart', default_conf_obj.global_parameters_display_quickstart)
        self.global_parameters_protocol_level = get_from_conf(conf, 'Global parameters',
            'protocol level', default_conf_obj.global_parameters_protocol_level)
        self.global_parameters_write_protocol_to_file = get_from_conf(conf, 'Global parameters',
            'write protocol to file', default_conf_obj.global_parameters_write_protocol_to_file)
        self.global_parameters_store_protocol_with_result = get_from_conf(conf, 'Global parameters',
            'store protocol with result', default_conf_obj.global_parameters_store_protocol_with_result)
        self.global_parameters_buffering_level = get_from_conf(conf, 'Global parameters',
            'buffering level', default_conf_obj.global_parameters_buffering_level)
        self.global_parameters_maximum_memory_active = get_from_conf(conf, 'Global parameters',
            'maximum memory active', default_conf_obj.global_parameters_maximum_memory_active)
        self.global_parameters_maximum_memory_amount = get_from_conf(conf, 'Global parameters',
            'maximum memory amount', default_conf_obj.global_parameters_maximum_memory_amount)
        self.global_parameters_include_postprocessing = get_from_conf(conf, 'Global parameters',
            'include postprocessing', default_conf_obj.global_parameters_include_postprocessing)
        self.global_parameters_image_format = get_from_conf(conf, 'Global parameters',
            'image format', default_conf_obj.global_parameters_image_format)
        self.global_parameters_parameters_in_filename = get_from_conf(conf, 'Global parameters',
            'parameters in filename', default_conf_obj.global_parameters_parameters_in_filename)
        self.global_parameters_stack_number_frames = get_from_conf(conf, 'Global parameters',
            'stack number frames', default_conf_obj.global_parameters_stack_number_frames)
        self.global_parameters_stack_percent_frames = get_from_conf(conf, 'Global parameters',
            'stack percent frames', default_conf_obj.global_parameters_stack_percent_frames)
        self.global_parameters_ap_box_size = get_from_conf(conf, 'Global parameters',
            'ap box size', default_conf_obj.global_parameters_ap_box_size)
        self.global_parameters_ap_number = get_from_conf(conf, 'Global parameters',
            'ap number', default_conf_obj.global_parameters_ap_number)

        self.frames_gauss_width = get_from_conf(conf, 'Frames',
            'gauss width', default_conf_obj.frames_gauss_width)
        self.frames_debayering_default = get_from_conf(conf, 'Frames',
            'debayering default', default_conf_obj.frames_debayering_default)
        self.frames_debayering_method = get_from_conf(conf, 'Frames',
            'debayering method', default_conf_obj.frames_debayering_method)
        self.frames_normalization = get_from_conf(conf, 'Frames',
            'normalization', default_conf_obj.frames_normalization)
        self.frames_normalization_threshold = get_from_conf(conf, 'Frames',
            'normalization threshold', default_conf_obj.frames_normalization_threshold)
        self.frames_add_selection_dialog = get_from_conf(conf, 'Frames',
            'add selection dialog', default_conf_obj.frames_add_selection_dialog)

        self.align_frames_fast_changing_object = get_from_conf(conf, 'Align frames',
            'fast changing object', default_conf_obj.align_frames_fast_changing_object)
        self.align_frames_mode = get_from_conf(conf, 'Align frames',
            'mode', default_conf_obj.align_frames_mode)
        self.align_frames_automation = get_from_conf(conf, 'Align frames',
            'automation', default_conf_obj.align_frames_automation)
        self.align_frames_rectangle_scale_factor = get_from_conf(conf, 'Align frames',
            'rectangle scale factor', default_conf_obj.align_frames_rectangle_scale_factor)
        self.align_frames_search_width = get_from_conf(conf, 'Align frames',
            'search width', default_conf_obj.align_frames_search_width)
        self.align_frames_average_frame_percent = get_from_conf(conf, 'Align frames',
            'average frame percent', default_conf_obj.align_frames_average_frame_percent)

        self.alignment_points_half_box_width = get_from_conf(conf, 'Alignment points',
            'half box width', default_conf_obj.alignment_points_half_box_width)
        self.alignment_points_search_width = get_from_conf(conf, 'Alignment points',
            'search width', default_conf_obj.alignment_points_search_width)
        self.alignment_points_structure_threshold = get_from_conf(conf, 'Alignment points',
            'structure threshold', default_conf_obj.alignment_points_structure_threshold)
        self.alignment_points_brightness_threshold = get_from_conf(conf, 'Alignment points',
            'brightness threshold', default_conf_obj.alignment_points_brightness_threshold)
        self.alignment_points_frame_percent = get_from_conf(conf, 'Alignment points',
            'frame percent', default_conf_obj.alignment_points_frame_percent)
        self.alignment_points_frame_number = get_from_conf(conf, 'Alignment points',
            'frame number', default_conf_obj.alignment_points_frame_number)

        self.stack_frames_drizzle_factor_string = get_from_conf(conf, 'Stack frames',
            'drizzle factor string', default_conf_obj.stack_frames_drizzle_factor_string)

    def store_all_parameters_to_config_parser(self):
        """
        Write all variable parameters from the current configuration into a ConfigParser object.

        :return: ConfigParser object with all parameters
        """

        # Clear the ConfigParser object.
        sections = self.config_parser_object.sections()
        for section in sections:
            self.config_parser_object.remove_section(section)

        # Copy all current parameters from the current configuration into the ConfigParser object.
        self.config_parser_object.add_section('Hidden parameters')
        self.set_parameter('Hidden parameters', 'current directory',
                           self.hidden_parameters_current_dir)
        self.set_parameter('Hidden parameters', 'main window x0',
                           str(self.hidden_parameters_main_window_x0))
        self.set_parameter('Hidden parameters', 'main window y0',
                           str(self.hidden_parameters_main_window_y0))
        self.set_parameter('Hidden parameters', 'main window width',
                           str(self.hidden_parameters_main_window_width))
        self.set_parameter('Hidden parameters', 'main window height',
                           str(self.hidden_parameters_main_window_height))
        self.set_parameter('Hidden parameters', 'main window maximized',
                           str(self.hidden_parameters_main_window_maximized))
        self.config_parser_object.add_section('Global parameters')
        self.set_parameter('Global parameters', 'display quickstart',
                           str(self.global_parameters_display_quickstart))
        self.set_parameter('Global parameters', 'version', self.global_parameters_version)
        self.set_parameter('Global parameters', 'protocol level',
                           str(self.global_parameters_protocol_level))
        self.set_parameter('Global parameters', 'write protocol to file',
                           str(self.global_parameters_write_protocol_to_file))
        self.set_parameter('Global parameters', 'store protocol with result',
                           str(self.global_parameters_store_protocol_with_result))
        self.set_parameter('Global parameters', 'buffering level',
                           str(self.global_parameters_buffering_level))
        self.set_parameter('Global parameters', 'maximum memory active',
                           str(self.global_parameters_maximum_memory_active))
        self.set_parameter('Global parameters', 'maximum memory amount',
                           str(self.global_parameters_maximum_memory_amount))
        self.set_parameter('Global parameters', 'include postprocessing',
                           str(self.global_parameters_include_postprocessing))
        self.set_parameter('Global parameters', 'image format',
                           self.global_parameters_image_format)
        self.set_parameter('Global parameters', 'parameters in filename',
                           str(self.global_parameters_parameters_in_filename))
        self.set_parameter('Global parameters', 'stack number frames',
                           str(self.global_parameters_stack_number_frames))
        self.set_parameter('Global parameters', 'stack percent frames',
                           str(self.global_parameters_stack_percent_frames))
        self.set_parameter('Global parameters', 'ap box size',
                           str(self.global_parameters_ap_box_size))
        self.set_parameter('Global parameters', 'ap number',
                           str(self.global_parameters_ap_number))

        self.config_parser_object.add_section('Frames')
        self.set_parameter('Frames', 'gauss width', str(self.frames_gauss_width))
        self.set_parameter('Frames', 'debayering default', self.frames_debayering_default)
        self.set_parameter('Frames', 'debayering method', self.frames_debayering_method)
        self.set_parameter('Frames', 'normalization', str(self.frames_normalization))
        self.set_parameter('Frames', 'normalization threshold',
                           str(self.frames_normalization_threshold))
        self.set_parameter('Frames', 'add selection dialog', str(self.frames_add_selection_dialog))

        self.config_parser_object.add_section('Align frames')
        self.set_parameter('Align frames', 'fast changing object',
                           str(self.align_frames_fast_changing_object))
        self.set_parameter('Align frames', 'mode', self.align_frames_mode)
        self.set_parameter('Align frames', 'automation', str(self.align_frames_automation))
        self.set_parameter('Align frames', 'rectangle scale factor',
                           str(self.align_frames_rectangle_scale_factor))
        self.set_parameter('Align frames', 'search width',
                           str(self.align_frames_search_width))
        self.set_parameter('Align frames', 'average frame percent',
                           str(self.align_frames_average_frame_percent))

        self.config_parser_object.add_section('Alignment points')
        self.set_parameter('Alignment points', 'half box width',
                           str(self.alignment_points_half_box_width))
        self.set_parameter('Alignment points', 'search width',
                           str(self.alignment_points_search_width))
        self.set_parameter('Alignment points', 'structure threshold',
                           str(self.alignment_points_structure_threshold))
        self.set_parameter('Alignment points', 'brightness threshold',
                           str(self.alignment_points_brightness_threshold))
        self.set_parameter('Alignment points', 'frame percent',
                           str(self.alignment_points_frame_percent))
        self.set_parameter('Alignment points', 'frame number',
                           str(self.alignment_points_frame_number))

        self.config_parser_object.add_section('Stack frames')
        self.set_parameter('Stack frames', 'drizzle factor string',
                           self.stack_frames_drizzle_factor_string)

    def set_parameter(self, section, name, value):
        """
        Assign a new value to a parameter in the configuration object. The value is not checked for
        validity. Therefore, this method should be used with well-defined values internally only.

        :param section: section name (e.g. 'Global parameters') within the JSON data object
        :param name: name of the parameter (e.g. 'protocol level')
        :param value: new value to be assigned to the parameter (type str)
        :return: True, if the parameter was assigned successfully. False, otherwise.
        """

        try:
            self.config_parser_object.set(section, name, value)
            return True
        except:
            return False

    def set_derived_parameters(self):
        """
        Set parameters which are computed from other parameters.

        :return: -
        """

        # Set the alignment patch size to 1.5 times the box size.
        self.alignment_points_half_patch_width = int(
            round((self.alignment_points_half_box_width * 3) / 2))

        # Set the AP distance per coordinate direction such that adjacent patches overlap by 1/6
        # step on both sides.
        self.alignment_points_step_size = int(
            round((self.alignment_points_half_patch_width * 4.5) / 3))

        # Set the drizzling parameters.
        if self.stack_frames_drizzle_factor_string == "Off":
            self.drizzle_factor = 1
            self.drizzle_factor_is_1_5 = False
        elif self.stack_frames_drizzle_factor_string == "1.5x":
            self.drizzle_factor = 3
            self.drizzle_factor_is_1_5 = True
        elif self.stack_frames_drizzle_factor_string == "2x":
            self.drizzle_factor = 2
            self.drizzle_factor_is_1_5 = False
        elif self.stack_frames_drizzle_factor_string == "3x":
            self.drizzle_factor = 3
            self.drizzle_factor_is_1_5 = False

    def write_config(self, file_name=None):
        """
        Write all variable configuration parameters to a file. If no file name is specified, take
        the standard ".ini" file

        :param file_name: Optional configuration file name.
        :return: -
        """

        # Reset the ConfigParser object, and fill it with the current parameters.
        self.store_all_parameters_to_config_parser()

        # Add the contents of the postprocessing data object to the ConfigParser object.
        self.postproc_data_object.dump_config(self.config_parser_object)

        if not file_name:
            file_name = self.config_filename

        with open(file_name, 'w') as config_file:
            self.config_parser_object.write(config_file)

    def read_config(self, file_name=None):
        """
        Read the configuration from a file. If no name is given, the config is read from the
        standard ".ini" file in the user's home directory.

        :param file_name: Optional configuration file name
        :return: ConfigParser object with configuration parameters
        """

        if not file_name:
            file_name = self.config_filename
        self.config_parser_object.read(file_name)

        # Get all stacking parameters from the parser object.
        self.get_all_parameters_from_configparser(self.config_parser_object)

        # Transfer all postprocessing parameters from the parser object to the postproc object.
        self.postproc_data_object.load_config(self.config_parser_object)

        return self.config_parser_object

def get_from_conf(config_parser_object, section, name, default):
    """
    Try to read a parameter from the config parser object. If it is not found, a default parameter
    is set instead.

    :param config_parser_object: Config parser object with all input parameters.
    :param section: Section name in config parser object.
    :param name: Variable name in section.
    :param default: Default value if variable cannot be read.
    :return: Either the value read, or default.
    """
    try:
        if isinstance(default, str):
            value = config_parser_object.get(section, name)
        elif isinstance(default, bool):
            value = config_parser_object.getboolean(section, name)
        elif isinstance(default, int):
            value = config_parser_object.getint(section, name)
        elif isinstance(default, float):
            value = config_parser_object.getfloat(section, name)
        else:
            raise ArgumentError("Parameter " + name + " in section " + section + " cannot be parsed")

    except:
        value = default

    return value

class PostprocDataObject(object):
    """
    This class implements the central data object used in postprocessing.

    """

    def __init__(self, postproc_suffix):
        """
        Initialize the data object.

        :param postproc_suffix: File suffix used for postprocessing result.
        :return: -
        """

        self.postproc_suffix = postproc_suffix

        # Initialize the postprocessing image versions with the unprocessed image (as version 0).
        self.initialize_versions()

        # Create a first processed version with initial parameters for Gaussian radius. The amount
        # of sharpening is initialized to zero.
        initial_version = self.add_postproc_version()
        initial_version.add_postproc_layer(PostprocLayer("Multilevel unsharp masking", 1., 1., 0.,
                                                         13, 0., False))

        # Initialize the pointer to the currently selected version to 0 (input image).
        # "version_compared" is used by the blink comparator later on. The blink comparator is
        # switched off initially.
        self.blinking = False
        self.version_compared = 0

    def set_postproc_input_image(self, image_original, name_original, image_format):
        """
        Set the input image and associated file name for postprocessing, and set derived variables.

        :param image_original: Image file (16bit Tiff) holding the input for postprocessing
        :param name_original: Path name of the original image.
        :param image_format: Image format, either 'tiff' or 'fits'.
        :return: -
        """

        self.image_original = image_original
        self.color = len(self.image_original.shape) == 3
        self.file_name_original = name_original

        # Set the standard path to the resulting image using the provided file suffix.
        self.file_name_processed = PostprocDataObject.set_file_name_processed(
            self.file_name_original, self.postproc_suffix, image_format)

        # Reset images and shift vectors which still might hold data from the previous job.
        for version in self.versions:
            version.set_image(self.image_original)
            version.input_image_saved = None
            version.input_image_hsv_saved = None
            version.shift_red_saved = None
            version.shift_blue_saved = None
            version.last_rgb_automatic = None

    @staticmethod
    def set_file_name_processed(file_name_original, postproc_suffix, image_format):
        """
        Derive the postprocessing output file name from the name of postprocessing input.

        :param file_name_original: Postprocessing input file name (e.g. result from stacking)
        :param postproc_suffix: Additional suffix to be inserted before file extension.
        :param image_format: Image format, either 'png', 'tiff' or 'fits'.
        :return: Name of postprocessing result.
        """

        return splitext(file_name_original)[0] + postproc_suffix + '.' + image_format

    def initialize_versions(self):
        """
        Initialize the postprocessing image versions with the unprocessed image (as version 0).

        :return: -
        """

        original_version = PostprocVersion()
        self.versions = [original_version]
        self.number_versions = 0
        self.version_selected = 0

    def add_postproc_version(self):
        """
        Add a new postprocessing version, and set the "selected" pointer to it.

        :return: The new version object
        """

        new_version = PostprocVersion()
        self.versions.append(new_version)
        self.number_versions += 1
        self.version_selected = self.number_versions
        return new_version

    def new_postproc_version_from_existing(self):
        """
        Create a new postprocessing version by copying the currently selected one. Append the new
        version to the list of postprocessing versions. Set the current version pointer to the new
        version.
        :return: New postprocessing version.
        """

        new_version = deepcopy(self.versions[self.version_selected])
        self.versions.append(new_version)
        self.number_versions += 1
        self.version_selected = self.number_versions
        return new_version

    def remove_postproc_version(self, index):
        """
        Remove a postprocessing version, and set the "selected" pointer to the previous one. The
        initial version (input image) cannot be removed.

        :param index: Index of the version to be removed.
        :return: -
        """

        if 0 < index <= self.number_versions:
            self.versions = self.versions[:index] + self.versions[index + 1:]
            self.number_versions -= 1
            self.version_selected = index - 1

    def finalize_postproc_version(self, version_index=None):
        """
        If "correction mode" is on, the image stored with the selected version is in 8bit and
        potentially interpolated by the resolution factor. Before the image is saved, it has to be
        processed from the original image using the current shift and layer data. The 16bit image is
        stored with the current version object.

        :param version_index: If an index is specified, process the version with this index.
                              Otherwise, process the version currently selected.
        :return: The processed 16bit image (also stored with the version).
        """

        if version_index is None:
            # The computation is based on the current "version selected".
            version = self.versions[self.version_selected]
        else:
            version = self.versions[version_index]

        # If the dtype is "uint16", nothing is to be done. If it is "uint8", the current image is
        # an intermediate product of the correction mode. In this case the rigorous postprocessing
        # pipeline must be traversed to get the final image.
        if version.image is not None and version.image.dtype == uint8:

            # Compute the complete current shifts, including temporary shift corrections.
            shift_red = (version.shift_red[0] + version.correction_red[0],
                        version.shift_red[1] + version.correction_red[1])
            shift_blue = (version.shift_blue[0] + version.correction_blue[0],
                        version.shift_blue[1] + version.correction_blue[1])

            # Shift the image with the resolution given by the selected interpolation factor.
            interpolation_factor = [1, 2, 4][version.rgb_resolution_index]
            shifted_image = Miscellaneous.shift_colors(self.image_original,
                                                       shift_red, shift_blue,
                                                       interpolate_input=interpolation_factor,
                                                       reduce_output=interpolation_factor)

            # Apply all wavelet layers.
            version.image = Miscellaneous.post_process(shifted_image, version.layers)

        return version.image

    def dump_config(self, config_parser_object):
        """
        Dump all version and layer parameters into a ConfigParser object (except for image data).
        Version 0 is not included because it does not contain any layer info.

        :param config_parser_object: ConfigParser object.
        :return: -
        """

        # First remove old postprocessing sections, if there are any in the ConfigParser object.
        for section in config_parser_object.sections():
            section_items = section.split()
            if section_items[0] == 'PostprocessingVersion' or \
                    section_items[0] == 'PostprocessingInfo':
                config_parser_object.remove_section(section)

        # Store general postprocessing info.
        config_parser_object.add_section('PostprocessingInfo')
        config_parser_object.set('PostprocessingInfo', 'version selected',
                                 str(self.version_selected))

        # For every version create a section with RGB alignment parameters.
        for version_index, version in enumerate(self.versions):
            section_name = "PostprocessingVersion " + str(version_index) + " RGBAlignment"
            config_parser_object.add_section(section_name)
            # Add the parameters of this version.
            config_parser_object.set(section_name, 'rgb automatic', str(version.rgb_automatic))
            config_parser_object.set(section_name, 'rgb gauss width', str(version.rgb_gauss_width))
            config_parser_object.set(section_name, 'rgb resolution index',
                                     str(version.rgb_resolution_index))
            config_parser_object.set(section_name, 'rgb shift red y',
                                     str(version.shift_red[0]))
            config_parser_object.set(section_name, 'rgb shift red x',
                                     str(version.shift_red[1]))
            config_parser_object.set(section_name, 'rgb shift blue y',
                                     str(version.shift_blue[0]))
            config_parser_object.set(section_name, 'rgb shift blue x',
                                     str(version.shift_blue[1]))

            # For every postprocessing layer of this version, create a separate section.
            for layer_index, layer in enumerate(version.layers):
                section_name = "PostprocessingVersion " + str(version_index) + " Layer " + str(
                    layer_index)
                config_parser_object.add_section(section_name)

                # Add the parameters of the layer.
                config_parser_object.set(section_name, 'postprocessing method', layer.postproc_method)
                config_parser_object.set(section_name, 'radius', str(layer.radius))
                config_parser_object.set(section_name, 'amount', str(layer.amount))
                config_parser_object.set(section_name, 'bilateral fraction', str(layer.bi_fraction))
                config_parser_object.set(section_name, 'bilateral range', str(layer.bi_range))
                config_parser_object.set(section_name, 'denoise', str(layer.denoise))
                config_parser_object.set(section_name, 'luminance only', str(layer.luminance_only))



    def load_config(self, config_parser_object):
        """
        Load all postprocessing configuration data from a ConfigParser object. The data replace
        all versions (apart from version 0) and all associated layer info. The image data is taken
        from the current data object and is not restored.

        :param config_parser_object: ConfigParser object.
        :return: -
        """

        standard_version = PostprocVersion()
        standard_layer = PostprocLayer("Multilevel unsharp masking", 1., 1., 0., 20, 0., False)

        # Initialize the postprocessing image versions with the unprocessed image (as version 0).
        self.initialize_versions()

        # Load general postprocessing info.
        try:
            self.version_selected = config_parser_object.getint('PostprocessingInfo',
                                                                'version selected')
        except:
            # the ConfigParser object does not contain postprocessing info. Leave the data object
            # with only the original version stored.
            return

        # Initialize the version index for comparison with an impossible value.
        self.number_versions = 0

        # Go through all sections and find the postprocessing sections.
        for section in config_parser_object.sections():
            section_items = section.split()
            if section_items[0] == 'PostprocessingVersion':

                # For a new version the first section contains the RGB alignment info. Create a new
                # version and store the RGB alignment parameters with it. For the first (neutral)
                # version 0 only RGB shift parameters are stored (no layers). In this case
                # initialize the version list.
                if section_items[2] == 'RGBAlignment':
                    version_index = int(section_items[1])
                    if version_index:
                        new_version = self.add_postproc_version()
                    else:
                        self.initialize_versions()
                        new_version = self.versions[0]
                    new_version.rgb_automatic = get_from_conf(config_parser_object, section,
                        'rgb automatic', standard_version.rgb_automatic)
                    new_version.rgb_gauss_width = get_from_conf(config_parser_object, section,
                        'rgb gauss width', standard_version.rgb_gauss_width)
                    new_version.rgb_resolution_index = get_from_conf(config_parser_object, section,
                        'rgb resolution index', standard_version.rgb_resolution_index)
                    new_version.shift_red = (get_from_conf(config_parser_object, section,
                        'rgb shift red y', standard_version.shift_red[0]),
                        get_from_conf(config_parser_object, section, 'rgb shift red x',
                        standard_version.shift_red[1]))
                    new_version.shift_blue = (get_from_conf(config_parser_object, section,
                        'rgb shift blue y', standard_version.shift_blue[0]),
                        get_from_conf(config_parser_object, section, 'rgb shift blue x',
                        standard_version.shift_blue[1]))

                # A layer section is found. Store it for the current version.
                elif section_items[2] == 'Layer':
                    # Read all parameters of this layer, and add a layer to the current version.
                    method = get_from_conf(config_parser_object, section,
                        'postprocessing method', standard_layer.postproc_method)
                    radius = get_from_conf(config_parser_object, section,
                        'radius', standard_layer.radius)
                    amount = get_from_conf(config_parser_object, section,
                        'amount', standard_layer.amount)
                    bi_fraction = get_from_conf(config_parser_object, section,
                        'bilateral fraction', standard_layer.bi_fraction)
                    bi_range = get_from_conf(config_parser_object, section,
                        'bilateral range', standard_layer.bi_range)
                    denoise = get_from_conf(config_parser_object, section,
                        'denoise', standard_layer.denoise)
                    luminance_only = get_from_conf(config_parser_object, section,
                        'luminance only', standard_layer.luminance_only)
                    new_version.add_postproc_layer(PostprocLayer(method, radius, amount, bi_fraction,
                                                                 bi_range, denoise, luminance_only))

        # Set the selected version again, because not all versions might have been read successfully.
        self.version_selected = min(config_parser_object.getint('PostprocessingInfo',
                                                            'version selected'), self.number_versions)


class PostprocVersion(object):
    """
    Instances of this class hold the data defining a single postprocessing version, including the
    resulting image for the current parameter set.
    """

    def __init__(self):
        """
        Initialize the version object with the input image and an empty set of processing layers.
        :param image: Input image (16bit Tiff) for postprocessing
        """

        self.postproc_method = "Multilevel unsharp masking"
        self.layers = []
        self.number_layers = 0
        self.rgb_automatic = False
        self.last_rgb_automatic = None
        self.rgb_gauss_width = 7
        self.rgb_resolution_index = 1
        self.shift_red = (0., 0.)
        self.shift_blue = (0., 0.)
        self.shift_red_saved = None
        self.shift_blue_saved = None
        self.rgb_correction_mode = False
        self.correction_red = (0., 0.)
        self.correction_blue = (0., 0.)
        self.correction_red_saved = (0., 0.)
        self.correction_blue_saved = (0., 0.)
        self.image = None
        self.images_uncorrected = [None] * 3
        self.input_image_saved = None
        self.input_image_hsv_saved = None

    def set_image(self, image):
        """
        Set the current image for this version.

        :param image: Image file (16bit Tiff) holding the pixel data of this version's image.
        :return: -
        """

        self.image = image

    def add_postproc_layer(self, layer):
        """
        Add a postprocessing layer.
        :param layer: Layer instance to be added to the list of layers.
        :return: -
        """

        self.layers.append(layer)
        self.number_layers += 1

    def remove_postproc_layer(self, layer_index):
        """
        Remove a postprocessing layer from this version. If there is only one layer, do not delete
        it, but reset it to standard values instead. This makes sure a version always has at least
        one layer.

        :param layer_index: Index of the layer to be removed.
        :return: -
        """

        if self.number_layers == 1:
            self.layers = [PostprocLayer("Multilevel unsharp masking", 1., 1., 0., 20, 0., False)]
        else:
            if 0 <= layer_index < self.number_layers:
                self.layers = self.layers[:layer_index] + self.layers[layer_index + 1:]
                self.number_layers -= 1


class PostprocLayer(object):
    """
    Instances of this class hold the parameters which define a postprocessing layer.
    """

    def __init__(self, method, radius, amount, bi_fraction, bi_range, denoise, luminance_only):
        """
        Initialize the Layer instance with values for Gaussian radius, amount of sharpening and a
        flag which indicates on which channel the sharpening is to be applied.

        :param method: Description of the sharpening method.
        :param radius: Radius (in pixels) of the Gaussian sharpening kernel.
        :param amount: Amount of sharpening for this layer.
        :param bi_fraction: Fraction of bilateral vs. Gaussian filter
                            (0.: only Gaussian, 1.: only bilateral).
        :param bi_range: luminosity range parameter of bilateral filter (0 <= bi_range <= 255).
                         Please note that for 16bit images the true range is 256 times as high.
        :param denoise: Fraction of Gaussian blur to be applied to this layer
                        (0.: No Gaussian blur, 1.: Full filter application).
        :param luminance_only: True, if sharpening is to be applied to the luminance channel only.
                               False, otherwise.
        """

        self.postproc_method = method
        self.radius = radius
        self.amount = amount
        self.bi_fraction = bi_fraction
        self.bi_range = bi_range
        self.denoise = denoise
        self.luminance_only = luminance_only
