# -*- coding: utf-8; -*-
"""
Copyright (c) 2018 Rolf Hempel, rolf6419@gmx.de

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
import sys

from PyQt5 import QtWidgets
# from configuration_editor import ConfigurationEditor

class Configuration(object):
    def __init__(self):
        self.global_parameters_version = "Planetary System Stacker 0.5.0"

        # The config file for persistent parameter storage is located in the user's home
        # directory, as is the detailed logfile.
        self.home = os.path.expanduser("~")
        self.config_filename = os.path.join(self.home, ".PlanetarySystemStacker.ini")
        self.protocol_filename = os.path.join(self.home, "PlanetarySystemStacker.log")

        self.file_new = not os.path.isfile(self.config_filename)

        # Create a ConfigParser object.
        self.conf = configparser.ConfigParser()

        # If an existing config file is found, read it in.
        if not self.file_new:
            self.conf.read(self.config_filename)
            # Get the parameters from the configparser.
            self.get_parameters_from_configparser(self.conf)
            # Set flag to indicate that parameters were read from file.
            self.configuration_read = True

        else:
            # Code to set standard config info. Some parameters will not be displayed in the
            # configuration GUI.
            self.configuration_read = False

            self.conf.add_section('Global parameters')
            self.conf.set('Global parameters', 'version', self.global_parameters_version)
            self.conf.set('Global parameters', 'write protocol to file', 'True')
            self.conf.set('Global parameters', 'protocol level', '1')
            self.conf.set('Global parameters', 'store protocol with result', 'False')

            self.conf.add_section('Frames')
            self.conf.set('Frames', 'gauss width', '7')

            self.conf.add_section('Align frames')
            self.conf.set('Align frames', 'mode', 'Surface')
            self.conf.set('Align frames', 'automation', 'True')
            self.conf.set('Align frames', 'rectangle scale factor', '3')
            self.conf.set('Align frames', 'search width', '20')
            self.conf.set('Align frames', 'average frame percent', '5')

            self.conf.add_section('Alignment points')
            self.conf.set('Alignment points', 'half box width', '20')
            self.conf.set('Alignment points', 'search width', '10')
            self.conf.set('Alignment points', 'structure threshold', '0.05')
            self.conf.set('Alignment points', 'brightness threshold', '10')
            self.conf.set('Alignment points', 'frame percent', '10')

        # Set parameters which are hidden from the user.
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

    def get_parameters_from_configparser(self, conf):
        """
        Read parameter values from a ConfigParser object and store them with the configuration
        object.

        :param conf: ConfigParser object
        :return: -
        """

        # All parameters which can be modified by the user are stored in the ConfigParser object.
        # This way they can be written to a file or read from there.
        self.global_parameters_protocol_level = self.conf.getint('Global parameters',
                                                                 'protocol level')
        self.global_parameters_write_protocol_to_file = self.conf.getboolean('Global parameters',
                                                                             'write protocol to file')
        self.global_parameters_store_protocol_with_result = self.conf.getboolean(
            'Global parameters',
            'store protocol with result')
        self.frames_gauss_width = self.conf.getint('Frames', 'gauss width')
        self.align_frames_mode = self.conf.get('Align frames', 'mode')
        self.align_frames_automation = self.conf.getboolean('Align frames', 'automation')
        self.align_frames_rectangle_scale_factor = self.conf.getint('Align frames',
                                                                    'rectangle scale factor')
        self.align_frames_search_width = self.conf.getint('Align frames', 'search width')
        self.align_frames_average_frame_percent = self.conf.getint('Align frames',
                                                                   'average frame percent')
        self.alignment_points_half_box_width = self.conf.getint('Alignment points',
                                                                'half box width')
        self.alignment_points_search_width = self.conf.getint('Alignment points', 'search width')
        self.alignment_points_structure_threshold = self.conf.getfloat('Alignment points',
                                                                       'structure threshold')
        self.alignment_points_brightness_threshold = self.conf.getint('Alignment points',
                                                                      'brightness threshold')
        self.align_frames_average_frame_percent = self.conf.getint('Alignment points',
                                                                   'frame percent')

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
            self.conf.set(section, name, value)
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

    def set_protocol_level(self):
        """
        Read from the configuration object the level of detail for the session protocol. The
        follwoing levels are supported:
        0:  No session protocol
        1:  Minimal protocol, only high-level activities
        2:  Quantitative information on high-level activities

        :return: -
        """

        self.protocol_level = self.conf.getint('Global parameters', 'protocol level')

    def write_config(self, file_name=None):
        """
        Write the contents of the configuration object to a file. If no file name is specified, the
        standard ".ini" file in the user's home directory is used.

        :param file_name: Optional configuration file name
        :return: -
        """

        if not file_name:
            file_name = self.config_filename

        with open(file_name, 'w') as config_file:
            self.conf.write(config_file)

    def read_config(self, file_name=None):
        """
        Read the configuration from a file. If no name is given, the config is read from the
        standard ".ini" file in the user's home directory.

        :param file_name: Optional configuration file name
        :return: -
        """

        self.conf = configparser.ConfigParser()
        if not file_name:
            file_name = self.config_filename
        self.conf.read(file_name)
        # Set flag to indicate that parameters were read from file.
        self.configuration_read = True