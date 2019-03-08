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

import configparser
import os.path

from exceptions import IncompatibleVersionsError


class ConfigurationParameters(object):
    def __init__(self):

        self.hidden_parameters_current_dir = None
        self.hidden_parameters_main_window_x0 = None
        self.hidden_parameters_main_window_y0 = None
        self.hidden_parameters_main_window_width = None
        self.hidden_parameters_main_window_height = None
        self.global_parameters_version = None
        self.global_parameters_protocol_level = None
        self.global_parameters_write_protocol_to_file = None
        self.global_parameters_store_protocol_with_result = None
        self.frames_gauss_width = None
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

    def set_defaults(self):
        self.hidden_parameters_current_dir = os.path.expanduser("~")
        self.hidden_parameters_main_window_x0 = 100
        self.hidden_parameters_main_window_y0 = 100
        self.hidden_parameters_main_window_width = 1200
        self.hidden_parameters_main_window_height = 800
        self.global_parameters_version = "Planetary System Stacker 0.5.0"
        self.global_parameters_protocol_level = 1
        self.global_parameters_write_protocol_to_file = True
        self.global_parameters_store_protocol_with_result = False
        self.frames_gauss_width = 7
        self.align_frames_mode = 'Surface'
        self.align_frames_automation = True
        self.align_frames_rectangle_scale_factor = 3.
        self.align_frames_search_width = 20
        self.align_frames_average_frame_percent = 5
        self.alignment_points_half_box_width = 20
        self.alignment_points_search_width = 10
        self.alignment_points_structure_threshold = 0.05
        self.alignment_points_brightness_threshold = 10
        self.alignment_points_frame_percent = 10

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
        self.global_parameters_version = configuration_object.global_parameters_version
        self.global_parameters_protocol_level = \
            configuration_object.global_parameters_protocol_level
        self.global_parameters_write_protocol_to_file = \
            configuration_object.global_parameters_write_protocol_to_file
        self.global_parameters_store_protocol_with_result = \
            configuration_object.global_parameters_store_protocol_with_result
        self.frames_gauss_width = configuration_object.frames_gauss_width
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

class Configuration(object):
    def __init__(self):
        self.global_parameters_version = "Planetary System Stacker 0.5.0"

        # The config file for persistent parameter storage is located in the user's home
        # directory, as is the detailed logfile.
        self.home = os.path.expanduser("~")
        self.config_filename = os.path.join(self.home, ".PlanetarySystemStacker.ini")
        self.protocol_filename = os.path.join(self.home, "PlanetarySystemStacker.log")

        self.config_file_exists = os.path.isfile(self.config_filename)

        # If an existing config file is found, read it in.
        self.configuration_read = False
        if self.config_file_exists:
            try:
                self.read_config()
                # Set flag to indicate that parameters were read from file successfully.
                self.configuration_read = True
            except:
                self.configuration_read = False

        if not self.configuration_read:
            # The configuration could not be read from a file, or versions did not match. Ccreate a
            # new one with standard parameters.

            configuration_parameters = ConfigurationParameters()
            configuration_parameters.set_defaults()

            # Set current configuration parameters to the new values.
            self.import_from_configuration_parameters(configuration_parameters)

        # Set fixed parameters which are hidden from the user. Hidden parameters which are
        # changeable are stored in the configuration object.
        self.frames_mono_channel = 'panchromatic'

        self.rank_frames_pixel_stride = 1
        self.rank_frames_method = "Laplace"

        self.align_frames_method = "SteepestDescent"
        self.align_frames_rectangle_black_threshold = 40
        self.align_frames_border_width = 10
        self.align_frames_sampling_stride = 2

        self.alignment_points_min_half_box_width = 10
        self.alignment_points_contrast_threshold = 0
        self.alignment_points_dim_fraction_threshold = 0.6
        self.alignment_points_adjust_edge_patches = True
        self.alignment_points_rank_method = "Laplace"
        self.alignment_points_rank_pixel_stride = 2
        self.alignment_points_de_warp = True
        self.alignment_points_method = 'SteepestDescent'
        self.alignment_points_sampling_stride = 2
        self.alignment_points_local_search_subpixel = False

        self.stack_frames_background_fraction = 0.3
        self.stack_frames_background_patch_size = 100
        self.stack_frames_gauss_width = 5

        # Compute parameters which are derived from other parameters.
        self.set_derived_parameters()

        # Mark the configuration as not changed.
        self.configuration_changed = False
        self.go_back_to_activity = None

    def import_from_configuration_parameters(self, configuration_parameters):
        """
        Set all current parameters to the corresponding values of a ConfigurarionParameters object.

        :param configuration_parameters: ConfigurarionParameters object with new parameter values.
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
        self.global_parameters_version = configuration_parameters.global_parameters_version
        self.global_parameters_protocol_level = \
            configuration_parameters.global_parameters_protocol_level
        self.global_parameters_write_protocol_to_file = \
            configuration_parameters.global_parameters_write_protocol_to_file
        self.global_parameters_store_protocol_with_result = \
            configuration_parameters.global_parameters_store_protocol_with_result
        self.frames_gauss_width = configuration_parameters.frames_gauss_width
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
        configuration_parameters.global_parameters_version = self.global_parameters_version
        configuration_parameters.global_parameters_protocol_level = \
            self.global_parameters_protocol_level
        configuration_parameters.global_parameters_write_protocol_to_file = \
            self.global_parameters_write_protocol_to_file
        configuration_parameters.global_parameters_store_protocol_with_result = \
            self.global_parameters_store_protocol_with_result

        configuration_parameters.frames_gauss_width = self.frames_gauss_width

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


    def get_all_parameters_from_configparser(self, conf):
        """
        All parameters which can be modified by the user are stored in the ConfigParser object.
        This way they can be written to a file or read from there. Read parameter values from a
        ConfigParser object and store them with the configuration object.

        :param conf: ConfigParser object
        :return: -
        """

        # Check for version compatibility.
        if self.global_parameters_version != conf.get('Global parameters', 'version'):
            raise IncompatibleVersionsError(
                "Error: parameter file read does not match program version")

        self.hidden_parameters_current_dir = conf.get('Hidden parameters', 'current directory')
        self.hidden_parameters_main_window_x0 = conf.getint('Hidden parameters', 'main window x0')
        self.hidden_parameters_main_window_y0 = conf.getint('Hidden parameters', 'main window y0')
        self.hidden_parameters_main_window_width = \
            conf.getint('Hidden parameters', 'main window width')
        self.hidden_parameters_main_window_height = conf.getint('Hidden parameters',
                                                               'main window height')
        self.global_parameters_protocol_level = conf.getint('Global parameters',
                                                                 'protocol level')
        self.global_parameters_write_protocol_to_file = conf.getboolean('Global parameters',
                                                                        'write protocol to file')
        self.global_parameters_store_protocol_with_result = conf.getboolean(
            'Global parameters',
            'store protocol with result')
        self.frames_gauss_width = conf.getint('Frames', 'gauss width')
        self.align_frames_mode = conf.get('Align frames', 'mode')
        self.align_frames_automation = conf.getboolean('Align frames', 'automation')
        self.align_frames_rectangle_scale_factor = conf.getfloat('Align frames',
                                                                    'rectangle scale factor')
        self.align_frames_search_width = conf.getint('Align frames', 'search width')
        self.align_frames_average_frame_percent = conf.getint('Align frames',
                                                                   'average frame percent')
        self.alignment_points_half_box_width = conf.getint('Alignment points',
                                                                'half box width')
        self.alignment_points_search_width = conf.getint('Alignment points', 'search width')
        self.alignment_points_structure_threshold = conf.getfloat('Alignment points',
                                                                       'structure threshold')
        self.alignment_points_brightness_threshold = conf.getint('Alignment points',
                                                                      'brightness threshold')
        self.alignment_points_frame_percent = conf.getint('Alignment points',
                                                                   'frame percent')

    def store_all_parameters_to_config_parser(self):
        """
        Write all variable parameters from the current configuration into a ConfigParser object.

        :return: ConfigParser object with all parameters
        """

        # Create a ConfigParser object.
        conf = configparser.ConfigParser()

        # Copy all current parameters from the current configuration into the ConfigParser object.
        conf.add_section('Hidden parameters')
        self.set_parameter(conf, 'Hidden parameters', 'current directory',
                           self.hidden_parameters_current_dir)
        self.set_parameter(conf, 'Hidden parameters', 'main window x0',
                           str(self.hidden_parameters_main_window_x0))
        self.set_parameter(conf, 'Hidden parameters', 'main window y0',
                           str(self.hidden_parameters_main_window_y0))
        self.set_parameter(conf, 'Hidden parameters', 'main window width',
                           str(self.hidden_parameters_main_window_width))
        self.set_parameter(conf, 'Hidden parameters', 'main window height',
                           str(self.hidden_parameters_main_window_height))
        conf.add_section('Global parameters')
        self.set_parameter(conf, 'Global parameters', 'version', self.global_parameters_version)
        self.set_parameter(conf, 'Global parameters', 'protocol level',
                           str(self.global_parameters_protocol_level))
        self.set_parameter(conf, 'Global parameters', 'write protocol to file',
                           str(self.global_parameters_write_protocol_to_file))
        self.set_parameter(conf, 'Global parameters', 'store protocol with result',
                           str(self.global_parameters_store_protocol_with_result))

        conf.add_section('Frames')
        self.set_parameter(conf, 'Frames', 'gauss width', str(self.frames_gauss_width))

        conf.add_section('Align frames')
        self.set_parameter(conf, 'Align frames', 'mode', self.align_frames_mode)
        self.set_parameter(conf, 'Align frames', 'automation', str(self.align_frames_automation))
        self.set_parameter(conf, 'Align frames', 'rectangle scale factor',
                           str(self.align_frames_rectangle_scale_factor))
        self.set_parameter(conf, 'Align frames', 'search width',
                           str(self.align_frames_search_width))
        self.set_parameter(conf, 'Align frames', 'average frame percent',
                           str(self.align_frames_average_frame_percent))

        conf.add_section('Alignment points')
        self.set_parameter(conf, 'Alignment points', 'half box width',
                           str(self.alignment_points_half_box_width))
        self.set_parameter(conf, 'Alignment points', 'search width',
                           str(self.alignment_points_search_width))
        self.set_parameter(conf, 'Alignment points', 'structure threshold',
                           str(self.alignment_points_structure_threshold))
        self.set_parameter(conf, 'Alignment points', 'brightness threshold',
                           str(self.alignment_points_brightness_threshold))
        self.set_parameter(conf, 'Alignment points', 'frame percent',
                           str(self.alignment_points_frame_percent))

        return conf

    def set_parameter(self, conf, section, name, value):
        """
        Assign a new value to a parameter in the configuration object. The value is not checked for
        validity. Therefore, this method should be used with well-defined values internally only.

        :param conf: ConfigParser object where the parameter is to be set
        :param section: section name (e.g. 'Global parameters') within the JSON data object
        :param name: name of the parameter (e.g. 'protocol level')
        :param value: new value to be assigned to the parameter (type str)
        :return: True, if the parameter was assigned successfully. False, otherwise.
        """

        try:
            conf.set(section, name, value)
            return True
        except:
            return False

    def set_derived_parameters(self):
        """
        Set parameters which are computed from other parameters.

        :return: -
        """

        # Set the alignment patch size to 1.5 times the box size. Between the patch and box
        # borders there must be at least as many pixels as the alignment search width. This way,
        # alignment boxes close to the border never leave the frame.
        self.alignment_points_half_patch_width = max(int(
            round((self.alignment_points_half_box_width * 3) / 2)),
            self.alignment_points_half_box_width + self.alignment_points_search_width)
        # Set the AP distance per coordinate direction such that adjacent patches overlap by 1/6
        # of their width.
        self.alignment_points_step_size = int(
            round((self.alignment_points_half_patch_width * 5) / 3))

    def write_config(self, file_name=None):
        """
        Write all variable configuration parameters to a file. If no file name is specified, the
        standard ".ini" file

        :param file_name: Optional configuration file name
        :return: -
        """

        # Create a ConfigParser object, and set it to the current parameters.
        conf = self.store_all_parameters_to_config_parser()

        if not file_name:
            file_name = self.config_filename

        with open(file_name, 'w') as config_file:
            conf.write(config_file)

    def read_config(self, file_name=None):
        """
        Read the configuration from a file. If no name is given, the config is read from the
        standard ".ini" file in the user's home directory.

        :param file_name: Optional configuration file name
        :return: ConfigParser object with configuration parameters
        """

        # Allocate a new ConfigParser object.
        conf = configparser.ConfigParser()

        if not file_name:
            file_name = self.config_filename
        conf.read(file_name)

        self.get_all_parameters_from_configparser(conf)

        return conf
