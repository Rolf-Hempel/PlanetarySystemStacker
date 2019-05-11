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

from configparser import ConfigParser
from os.path import expanduser, join, isfile
from os.path import splitext

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
        self.global_parameters_buffering_level = None
        self.global_parameters_include_postprocessing = None
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
        self.hidden_parameters_current_dir = expanduser("~")
        self.hidden_parameters_main_window_x0 = 100
        self.hidden_parameters_main_window_y0 = 100
        self.hidden_parameters_main_window_width = 1200
        self.hidden_parameters_main_window_height = 800
        self.global_parameters_version = "Planetary System Stacker 0.5.0"
        self.global_parameters_protocol_level = 1
        self.global_parameters_write_protocol_to_file = False
        self.global_parameters_store_protocol_with_result = False
        self.global_parameters_buffering_level = 2
        self.global_parameters_include_postprocessing = False
        self.frames_gauss_width = 7
        self.align_frames_mode = 'Surface'
        self.align_frames_automation = True
        self.align_frames_rectangle_scale_factor = 3.
        self.align_frames_search_width = 20
        self.align_frames_average_frame_percent = 5
        self.alignment_points_search_width = 10
        self.alignment_points_frame_percent = 10
        self.set_defaults_ap_editing()

    def set_defaults_ap_editing(self):
        self.alignment_points_half_box_width = 20
        self.alignment_points_structure_threshold = 0.05
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
        self.global_parameters_version = configuration_object.global_parameters_version
        self.global_parameters_protocol_level = \
            configuration_object.global_parameters_protocol_level
        self.global_parameters_write_protocol_to_file = \
            configuration_object.global_parameters_write_protocol_to_file
        self.global_parameters_store_protocol_with_result = \
            configuration_object.global_parameters_store_protocol_with_result
        self.global_parameters_buffering_level = \
            configuration_object.global_parameters_buffering_level
        self.global_parameters_include_postprocessing = \
            configuration_object.global_parameters_include_postprocessing
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
        self.home = expanduser("~")
        self.config_filename = join(self.home, ".PlanetarySystemStacker.ini")
        self.protocol_filename = join(self.home, "PlanetarySystemStacker.log")

        # Set fixed parameters which are hidden from the user. Hidden parameters which are
        # changeable are stored in the configuration object.
        self.window_icon = '../PSS-Icon-64.ico'

        self.frames_mono_channel = 'green'

        self.rank_frames_pixel_stride = 1
        self.rank_frames_method = "Laplace"

        self.align_frames_method = "SteepestDescent"
        self.align_frames_rectangle_black_threshold = 10240
        self.align_frames_rectangle_min_fraction = 0.7
        self.align_frames_rectangle_stride = 2
        self.align_frames_border_width = 10
        self.align_frames_sampling_stride = 2
        self.align_frames_min_stabilization_patch_fraction = 0.2
        self.align_frames_max_stabilization_patch_fraction = 0.7

        self.alignment_points_min_half_box_width = 10
        self.alignment_points_contrast_threshold = 0
        self.alignment_points_dim_fraction_threshold = 0.6
        self.alignment_points_rank_method = "Laplace"
        self.alignment_points_rank_pixel_stride = 2
        self.alignment_points_de_warp = True
        self.alignment_points_method = 'SteepestDescent'
        self.alignment_points_sampling_stride = 2
        self.alignment_points_local_search_subpixel = False

        self.stack_frames_suffix = "_pss"
        self.stack_frames_background_fraction = 0.3
        self.stack_frames_background_patch_size = 100
        self.stack_frames_gauss_width = 5

        self.postproc_suffix = "_gpp"
        self.postproc_blinking_period = 1.
        self.postproc_idle_loop_time = 0.2

        # Initialize the ConfigParser object for parameters which the user can change.
        self.config_parser_object = ConfigParser()

        # Create and initialize the central data object for postprocessing.
        self.postproc_data_object = PostprocDataObject(self.postproc_suffix)

        # Determine if there is a configuration file from a previous run.
        self.config_file_exists = isfile(self.config_filename)

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
        self.global_parameters_buffering_level = \
            configuration_parameters.global_parameters_buffering_level
        self.global_parameters_include_postprocessing = \
            configuration_parameters.global_parameters_include_postprocessing
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
        configuration_parameters.global_parameters_include_postprocessing = \
            self.global_parameters_include_postprocessing

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
            'Global parameters', 'store protocol with result')
        self.global_parameters_buffering_level = conf.getint('Global parameters', 'buffering level')
        self.global_parameters_include_postprocessing = conf.getboolean(
            'Global parameters', 'include postprocessing')
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
        self.alignment_points_frame_percent = conf.getint('Alignment points', 'frame percent')

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
        self.config_parser_object.add_section('Global parameters')
        self.set_parameter('Global parameters', 'version', self.global_parameters_version)
        self.set_parameter('Global parameters', 'protocol level',
                           str(self.global_parameters_protocol_level))
        self.set_parameter('Global parameters', 'write protocol to file',
                           str(self.global_parameters_write_protocol_to_file))
        self.set_parameter('Global parameters', 'store protocol with result',
                           str(self.global_parameters_store_protocol_with_result))
        self.set_parameter('Global parameters', 'buffering level',
                           str(self.global_parameters_buffering_level))
        self.set_parameter('Global parameters', 'include postprocessing',
                           str(self.global_parameters_include_postprocessing))

        self.config_parser_object.add_section('Frames')
        self.set_parameter('Frames', 'gauss width', str(self.frames_gauss_width))

        self.config_parser_object.add_section('Align frames')
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
        # Initialze the number of frames to be stacked. It will be computed from the corresponding
        # percentage. The user, however, can override this value with a (more precise) figure
        # during the workflow.
        self.alignment_points_frame_number = None

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
        initial_version.add_postproc_layer(PostprocLayer("Multilevel unsharp masking", 1., 0, False))

        # Initialize the pointer to the currently selected version to 0 (input image).
        # "version_compared" is used by the blink comparator later on. The blink comparator is
        # switched off initially.
        self.blinking = False
        self.version_compared = 0

    def set_postproc_input_image(self, image_original, name_original):
        """
        Set the input image and associated file name for postprocessing, and set derived variables.

        :param image_original: Image file (16bit Tiff) holding the input for postprocessing
        :param name_original: Path name of the original image.
        :return: -
        """

        self.image_original = image_original
        self.color = len(self.image_original.shape) == 3
        self.file_name_original = name_original

        # Set the standard path to the resulting image using the provided file suffix.
        self.file_name_processed = PostprocDataObject.set_file_name_processed(
            self.file_name_original, self.postproc_suffix)

        for version in self.versions:
            version.set_image(self.image_original)

    @staticmethod
    def set_file_name_processed(file_name_original, postproc_suffix):
        """
        Derive the postprocessing output file name from the name of postprocessing input.

        :param file_name_original: Postprocessing input file name (e.g. result from stacking)
        :param postproc_suffix: Additional suffix to be inserted before file extension.
        :return: Name of postprocessing result.
        """

        return splitext(file_name_original)[0] + postproc_suffix + '.tiff'

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

        # For every version, and for all layers in each version, create a separate section.
        for version_index, version in enumerate(self.versions):
            for layer_index, layer in enumerate(version.layers):
                section_name = "PostprocessingVersion " + str(version_index) + " layer " + str(
                    layer_index)
                config_parser_object.add_section(section_name)

                # Add the four parameters of the layer.
                config_parser_object.set(section_name, 'postprocessing method', layer.postproc_method)
                config_parser_object.set(section_name, 'radius', str(layer.radius))
                config_parser_object.set(section_name, 'amount', str(layer.amount))
                config_parser_object.set(section_name, 'luminance only', str(layer.luminance_only))

    def load_config(self, config_parser_object):
        """
        Load all postprocessing configuration data from a ConfigParser object. The data replace
        all versions (apart from version 0) and all associated layer info. The image data is taken
        from the current data object and is not restored.

        :param config_parser_object: ConfigParser object.
        :return: -
        """

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
        old_version_index = -1
        self.number_versions = 0

        # Go through all sections and find the postprocessing sections.
        for section in config_parser_object.sections():
            section_items = section.split()
            if section_items[0] == 'PostprocessingVersion':
                this_version_index = section_items[1]

                # A layer section with a new version index is found. Allocate a new version.
                if this_version_index != old_version_index:
                    new_version = self.add_postproc_version()
                    old_version_index = this_version_index

                # Read all parameters of this layer, and add a layer to the current version.
                method = config_parser_object.get(section, 'postprocessing method')
                radius = config_parser_object.getfloat(section, 'radius')
                amount = config_parser_object.getfloat(section, 'amount')
                luminance_only = config_parser_object.getboolean(section, 'luminance only')
                new_version.add_postproc_layer(PostprocLayer(method, radius, amount, luminance_only))

        # Set the selected version again, because it may have been changed by reading versions.
        self.version_selected = config_parser_object.getint('PostprocessingInfo',
                                                            'version selected')


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
        Remove a postprocessing layer from this version.

        :param layer_index: Index of the layer to be removed.
        :return: -
        """

        if 0 <= layer_index < self.number_layers:
            self.layers = self.layers[:layer_index] + self.layers[layer_index + 1:]
            self.number_layers -= 1


class PostprocLayer(object):
    """
    Instances of this class hold the parameters which define a postprocessing layer.
    """

    def __init__(self, method, radius, amount, luminance_only):
        """
        Initialize the Layer instance with values for Gaussian radius, amount of sharpening and a
        flag which indicates on which channel the sharpening is to be applied.

        :param method: Description of the sharpening method.
        :param radius: Radius (in pixels) of the Gaussian sharpening kernel.
        :param amount: Amount of sharpening for this layer.
        :param luminance_only: True, if sharpening is to be applied to the luminance channel only.
                               False, otherwise.
        """

        self.postproc_method = method
        self.radius = radius
        self.amount = amount
        self.luminance_only = luminance_only
