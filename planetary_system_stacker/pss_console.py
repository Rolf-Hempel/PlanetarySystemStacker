from argparse import ArgumentParser, ArgumentTypeError
from time import sleep
from glob import glob

from PyQt5 import QtCore

from configuration import Configuration
from exceptions import InternalError
from job_editor import Job
from miscellaneous import Miscellaneous
from workflow import Workflow

# Definition of data types, including value bounds, used in command line argument parsing.
def ram_size_type(x):
    try:
        x = int(x)
    except:
        raise ArgumentTypeError("Maximum RAM size must be an integer > 0")
    if x < 1:
        raise ArgumentTypeError("Maximum RAM size must be an integer > 0")
    return x


def noise_type(x):
    x = int(x)
    if not 0 <= x <= 11:
        raise ArgumentTypeError("Noise level must be between 0 and 11")
    return x


def stab_size_type(x):
    x = int(x)
    if not 5 <= x <= 80:
        raise ArgumentTypeError("Stabilization patch size must be between 5% and 80%")
    return x


def stab_sw_type(x):
    x = int(x)
    if not 5 <= x <= 150:
        raise ArgumentTypeError(
            "Stabilization search width must be between 5 and 150 pixels")
    return x


def rf_percent_type(x):
    x = int(x)
    if not 3 <= x <= 30:
        raise ArgumentTypeError(
            "Percentage of best frames for reference frame computation must be between 3% and 30%")
    return x


def align_box_width_type(x):
    x = int(x)
    if not 20 <= x <= 140:
        raise ArgumentTypeError(
            "Alignment point box width must be between 20 and 140 pixels")
    return x


def align_search_width_type(x):
    x = int(x)
    if not 6 <= x <= 30:
        raise ArgumentTypeError(
            "Alignment point search width must be between 6 and 30 pixels")
    return x


def align_min_struct_type(x):
    x = float(x)
    if not 0.01 <= x <= 0.30:
        raise ArgumentTypeError(
            "Alignment point minimum structure must be between 0.01 and 0.30")
    return x


def align_min_bright_type(x):
    x = int(x)
    if not 2 <= x <= 50:
        raise ArgumentTypeError(
            "Alignment point minimum brightness must be between 2 and 50")
    return x


def stack_percent_type(x):
    x = int(x)
    if not 1 <= x <= 100:
        raise ArgumentTypeError(
            "Percentage of best frames to be stacked must be between 1 and 100")
    return x


def stack_number_type(x):
    x = int(x)
    if not 1 <= x:
        raise ArgumentTypeError(
            "Number of best frames to be stacked must be greater or equal 1")
    return x


def normalize_bco_type(x):
    x = int(x)
    if not 0 <= x <= 40:
        raise ArgumentTypeError(
            "Normalization black cut-off must be between 0 and 40")
    return x


class PssConsole(QtCore.QObject):
    """
    This class replaces the class PlanetarySystemStacker if the program is started from the
    command line. In this case no GUI activity is created, and there is no interactive mode.
    """

    # Define signals which trigger activities on the workflow thread.
    signal_load_master_dark = QtCore.pyqtSignal(str)
    signal_load_master_flat = QtCore.pyqtSignal(str)
    signal_frames = QtCore.pyqtSignal(object)
    signal_rank_frames = QtCore.pyqtSignal(bool)
    signal_align_frames = QtCore.pyqtSignal(int, int, int, int)
    signal_set_roi = QtCore.pyqtSignal(int, int, int, int)
    signal_set_alignment_points = QtCore.pyqtSignal()
    signal_compute_frame_qualities = QtCore.pyqtSignal()
    signal_stack_frames = QtCore.pyqtSignal()
    signal_save_stacked_image = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(PssConsole, self).__init__(parent)

        # Create the configuration object and modify it as specified in command line arguments.
        self.setup_configuration()

        # Start the workflow.
        self.work_next_task("Read frames")

    def setup_configuration(self):
        """
        Parse the command line arguments, initialize the configuration object and update
        configuration parameters with values passed via command line arguments.

        :return: -
        """
        parser = ArgumentParser()
        parser.add_argument("job_input", nargs='+', help="input video files or still image folders")

        parser.add_argument("-p", "--protocol", action="store_true",
                            help="Store protocol with results")
        parser.add_argument("--protocol_detail", type=int, choices=[0, 1, 2], default=1,
                            help="Protocol detail level")
        parser.add_argument("-b", "--buffering_level", choices=["auto", "0", "1", "2", "3", "4"],
                            default="auto", help="Buffering level")
        parser.add_argument("-r", "--ram_size", type=ram_size_type, default=-1,
                            help="Maximum RAM for this job (GBytes)")
        parser.add_argument("--out_format", choices=["png", "tiff", "fits"], default="png",
                            help="Image format for output")
        parser.add_argument("--name_add_f", action="store_true",
                            help="Add number of stacked frames to output file name")
        parser.add_argument("--name_add_p", action="store_true",
                            help="Add percentage of stacked frames to output file name")
        parser.add_argument("--name_add_apb", action="store_true",
                            help="Add alignment point box size (pixels) to output file name")
        parser.add_argument("--name_add_apn", action="store_true",
                            help="Add number of alignment points to output file name")

        parser.add_argument("--debayering",
                            choices=["Auto detect color", "Grayscale", "RGB", "RGB", "BGR",
                                     "Force Bayer RGGB", "Force Bayer GRBG",
                                     "Force Bayer GBRG", "Force Bayer BGGR"],
                            default="Auto detect color", help="Debayering option")
        parser.add_argument("--debayer_method",
                            choices=["Bilinear", "Variable Number of Gradients",
                                     "Edge Aware"],
                            default="Bilinear", help="Debayering method to be used")
        parser.add_argument("--noise", type=noise_type, default=7,
                            help="Noise level (add Gaussian blur)")
        parser.add_argument("-m", "--stab_mode", choices=["Surface", "Planet"], default="Surface",
                            help="Frame stabilization mode")
        parser.add_argument("--stab_size", type=stab_size_type, default=33,
                            help="Stabilization patch size (%% of frame)")
        parser.add_argument("--stab_sw", type=stab_sw_type, default=34,
                            help="Stabilization search width (pixels)")
        parser.add_argument("--rf_percent", type=rf_percent_type, default=5,
                            help="Percentage of best frames for reference frame computation")
        parser.add_argument("--fast_changing_object", action="store_true",
                            help="The object is changing fast during video time span (e.g. Jupiter")
        parser.add_argument("-d", "--dark", help="Image file for dark frame correction")
        parser.add_argument("-f", "--flat", help="Image file for flat frame correction")

        parser.add_argument("-a", "--align_box_width", type=align_box_width_type, default=48,
                            help="Alignment point box width (pixels)")
        parser.add_argument("-w", "--align_search_width", type=align_search_width_type, default=14,
                            help="Alignment point search width (pixels)")
        parser.add_argument("--align_min_struct", type=align_min_struct_type, default=0.04,
                            help="Alignment point minimum structure")
        parser.add_argument("--align_min_bright", type=align_min_bright_type, default=10,
                            help="Alignment point minimum brightness")

        parser.add_argument("-s", "--stack_percent", type=stack_percent_type, default=10,
                            help="Percentage of best frames to be stacked")
        parser.add_argument("--stack_number", type=stack_number_type,
                            help="Number of best frames to be stacked")
        parser.add_argument("-n", "--normalize_bright", action="store_true",
                            help="Normalize frame brightness")
        parser.add_argument("--normalize_bco", type=normalize_bco_type, default=15,
                            help="Normalization black cut-off")
        parser.add_argument("--drizzle", choices=["Off", "1.5x", "2x", "3x"], default="Off",
                            help="Drizzle factor (Off, 1.5x, 2x, 3x)")

        arguments = parser.parse_args()
        # self.print_arguments(arguments)

        # Create and initialize the configuration object. The configuration stored in the .ini file
        # in the user's home directory is ignored in this case. Modifications to standard values
        # come as command line arguments.
        self.configuration = Configuration()
        self.configuration.initialize_configuration(read_from_file=False)

        # In the standard configuration postprocessing is included in the workflow. This does not
        # make sense in command line mode.
        self.configuration.global_parameters_include_postprocessing = False

        # Modify the standard configuration as specified in the command line arguments.
        self.configuration.global_parameters_store_protocol_with_result = arguments.protocol
        self.configuration.global_parameters_protocol_level = arguments.protocol_detail
        if arguments.ram_size == -1:
            self.configuration.global_parameters_maximum_memory_active = False
            if arguments.buffering_level == "auto":
                self.configuration.global_parameters_buffering_level = -1
            else:
                self.configuration.global_parameters_buffering_level = int(arguments.buffering_level)
        else:
            self.configuration.global_parameters_buffering_level = -1
            self.configuration.global_parameters_maximum_memory_active = True
            self.configuration.global_parameters_maximum_memory_amount = arguments.ram_size

        self.configuration.global_parameters_image_format = arguments.out_format
        self.configuration.global_parameters_parameters_in_filename = arguments.name_add_f or \
            arguments.name_add_p or arguments.name_add_apb or arguments.name_add_apn
        self.configuration.global_parameters_stack_number_frames = arguments.name_add_f
        self.configuration.global_parameters_stack_percent_frames = arguments.name_add_p
        self.configuration.global_parameters_ap_box_size = arguments.name_add_apb
        self.configuration.global_parameters_ap_number = arguments.name_add_apn

        self.configuration.frames_debayering_default = arguments.debayering
        self.configuration.frames_debayering_method = arguments.debayer_method
        self.configuration.frames_gauss_width = arguments.noise

        self.configuration.align_frames_mode = arguments.stab_mode
        self.configuration.align_frames_rectangle_scale_factor = 100. / arguments.stab_size
        self.configuration.align_frames_search_width = arguments.stab_sw
        self.configuration.align_frames_average_frame_percent = arguments.rf_percent
        self.configuration.align_frames_fast_changing_object = arguments.fast_changing_object

        self.configuration.alignment_points_half_box_width = int(
            round(arguments.align_box_width / 2))
        self.configuration.alignment_points_search_width = arguments.align_search_width
        self.configuration.alignment_points_structure_threshold = arguments.align_min_struct
        self.configuration.alignment_points_brightness_threshold = arguments.align_min_bright

        self.configuration.alignment_points_frame_percent = arguments.stack_percent
        self.configuration.alignment_points_frame_number = -1

        # If the number of frames to be stacked is given, it has precedence over the percentage.
        if arguments.stack_number is not None:
            self.configuration.alignment_points_frame_number = arguments.stack_number
            self.configuration.alignment_points_frame_percent = -1

        self.configuration.frames_normalization = arguments.normalize_bright
        self.configuration.frames_normalization_threshold = arguments.normalize_bco
        self.configuration.stack_frames_drizzle_factor_string = arguments.drizzle

        # Re-compute derived parameters after the configuration was changed.
        self.configuration.set_derived_parameters()

        # Create the workflow thread and start it.
        self.thread = QtCore.QThread()
        self.workflow = Workflow(self)
        self.workflow.setParent(None)
        self.workflow.moveToThread(self.thread)
        self.workflow.calibration.report_calibration_error_signal.connect(
            self.report_calibration_error)
        self.workflow.work_next_task_signal.connect(self.work_next_task)
        self.workflow.report_error_signal.connect(self.report_error)
        self.workflow.abort_job_signal.connect(self.next_job_after_error)
        self.thread.start()

        # Connect signals to start activities on the workflow thread (e.g. in method
        # "work_next_task").
        self.signal_load_master_dark.connect(self.workflow.calibration.load_master_dark)
        self.signal_load_master_flat.connect(self.workflow.calibration.load_master_flat)
        self.signal_frames.connect(self.workflow.execute_frames)
        self.signal_rank_frames.connect(self.workflow.execute_rank_frames)
        self.signal_align_frames.connect(self.workflow.execute_align_frames)
        self.signal_set_roi.connect(self.workflow.execute_set_roi)
        self.signal_set_alignment_points.connect(self.workflow.execute_set_alignment_points)
        self.signal_compute_frame_qualities.connect(
            self.workflow.execute_compute_frame_qualities)
        self.signal_stack_frames.connect(self.workflow.execute_stack_frames)
        self.signal_save_stacked_image.connect(self.workflow.execute_save_stacked_image)

        # Set "automatic" to True. There is no interactive mode in this case.
        self.automatic = True

        # Create the job objects using the names passed as positional arguments.
        self.jobs = []
        for name in [f for name in arguments.job_input for f in glob(name)]:
            try:
                job = Job(name)

                # Test if the path specifies a stacking job.
                if job.type == 'video' or job.type == 'image':
                    # Override the "Auto detect color" value of the "Job" object with the
                    # command line value.
                    job.bayer_option_selected = self.configuration.frames_debayering_default
                    self.jobs.append(job)
                else:
                    if self.configuration.global_parameters_protocol_level > 0:
                        Miscellaneous.protocol(
                            "Error: '" + name + "' does not contain valid input for a stacking job,"
                                               " continune with next job.\n",
                            self.workflow.attached_log_file)
            except InternalError:
                if self.configuration.global_parameters_protocol_level > 0:
                    Miscellaneous.protocol(
                        "Error: '" + name + "' does not contain valid input for a stacking job,"
                                           " continune with next job.\n",
                        self.workflow.attached_log_file)

        self.job_number = len(self.jobs)
        if self.job_number == 0:
            if self.configuration.global_parameters_protocol_level > 0:
                Miscellaneous.protocol(
                    "Error: No valid job specified, execution halted.",
                    self.workflow.attached_log_file)
            self.stop_execution()

        self.job_index = 0

        # If a dark frame was specified, load it.
        if arguments.dark:
            if self.configuration.global_parameters_protocol_level > 0:
                Miscellaneous.protocol("+++ Loading master dark frame +++",
                                       self.workflow.attached_log_file)
            self.signal_load_master_dark.emit(arguments.dark)

        # If a flat frame was specified, load it.
        if arguments.flat:
            if self.configuration.global_parameters_protocol_level > 0:
                Miscellaneous.protocol("+++ Loading master flat frame +++",
                                       self.workflow.attached_log_file)
            self.signal_load_master_flat.emit(arguments.flat)

    @QtCore.pyqtSlot(str)
    def report_calibration_error(self, message):
        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol("           " + message,
                                   self.workflow.attached_log_file, precede_with_timestamp=False)

    @QtCore.pyqtSlot(str)
    def report_error(self, message):
        """
        This method is triggered by the workflow thread via a signal when an error is to be
        reported. Depending on the protocol level, the error message is written to the
        protocol (file).

        :param message: Error message to be displayed
        :return: -
        """

        if self.configuration.global_parameters_protocol_level > 0:
            Miscellaneous.protocol(message + "\n", self.workflow.attached_log_file)

    @QtCore.pyqtSlot(str)
    def next_job_after_error(self, message):
        """
        This method is triggered by the workflow thread via a signal when an error causes a job to
        be aborted. Depending on the protocol level, the error message is written to the
        protocol (file).

        :param message: Error message to be displayed
        :return: -
        """

        # Report the error.
        self.report_error(message)

        # Abort the current job and go to the next one.
        self.work_next_task("Next job")

    def work_next_task(self, next_activity):
        """
        This is the central place where all activities are scheduled. Depending on the
         "next_activity" chosen, the appropriate activity is started on the workflow thread.

        :param next_activity: Activity to be performed next.
        :return: -
        """

        # Make sure not to process an empty job list, or a job index out of range.
        if not self.jobs or self.job_index >= self.job_number:
            return

        self.activity = next_activity

        # Start workflow activities. When a workflow method terminates, it invokes this method on
        # the GUI thread, with "next_activity" denoting the next step in the processing chain.
        if self.activity == "Read frames":

            # For the first activity (reading all frames from the file system) there is no
            # GUI interaction. Start the workflow action immediately.
            self.signal_frames.emit(self.jobs[self.job_index])

        elif self.activity == "Rank frames":

            # In batch mode no frames can be dropped in the user dialog.
            update_index_translation_table = False
            # Now start the corresponding action on the workflow thread.
            self.signal_rank_frames.emit(update_index_translation_table)

        elif self.activity == "Select frames":

            # The dialog to exclude frames is not to be called. Go to frames alignment
            # immediately.
            self.signal_align_frames.emit(0, 0, 0, 0)

        elif self.activity == "Select stack size":

            # In automatic mode, nothing is to be done in the workflow thread. Start the next
            # activity on the main thread immediately.
            self.workflow.work_next_task_signal.emit("Set ROI")

        elif self.activity == "Set ROI":

            # If all index bounds are set to zero, no ROI is selected.
            self.signal_set_roi.emit(0, 0, 0, 0)

        elif self.activity == "Set alignment points":

            # In automatic mode, compute the AP grid automatically in the workflow thread. In this
            # case, the AlignmentPoints object is created there as well.
            self.signal_set_alignment_points.emit()

        elif self.activity == "Compute frame qualities":
            self.signal_compute_frame_qualities.emit()

        elif self.activity == "Stack frames":
            self.signal_stack_frames.emit()

        elif self.activity == "Save stacked image":
            self.signal_save_stacked_image.emit()

        elif self.activity == "Next job":
            self.job_index += 1
            if self.job_index < self.job_number:
                # If the end of the queue is not reached yet, start with reading frames of next job.
                self.activity = "Read frames"
                self.signal_frames.emit(self.jobs[self.job_index])
            else:
                self.stop_execution()

    def print_arguments(self, arguments):
        """
        This is an auxiliary method for debugging. It prints all arguments passed to the program.

        :param arguments: Arguments object created by the ArgumentParser
        :return: -
        """

        print("Jobs: " + str(arguments.job_input))
        print("Store protocol with results: " + str(arguments.protocol))
        print("Protocol detail level: " + str(arguments.protocol_detail))
        print("Buffering level: " + str(arguments.buffering_level))
        print("Maximum RAM size: " + str(arguments.ram_size))
        print("Image format for output: " + arguments.out_format)
        print("Add number of stacked frames to output file name: " + str(arguments.name_add_f))
        print("Add percentage of stacked frames to output file name: " + str(arguments.name_add_p))
        print(
            "Add alignment point box size (pixels) to output file name: " + str(
                arguments.name_add_apb))
        print("Add number of alignment points to output file name: " + str(arguments.name_add_apn))
        print("")
        print("Debayering option: " + arguments.debayering)
        print("Debayering method: " + arguments.debayer_method)
        print("Noise level: " + str(arguments.noise))
        print("Frame stabilization mode: " + arguments.stab_mode)
        print("Stabilization patch size (% of frame): " + str(arguments.stab_size))
        print("Stabilization search width (pixels): " + str(arguments.stab_sw))
        print("Percentage of best frames for reference frame computation: " + str(
            arguments.rf_percent))
        if arguments.dark:
            print("Image file for dark frame correction: " + arguments.dark)
        if arguments.flat:
            print("Image file for flat frame correction: " + arguments.flat)
        print("")
        print("Alignment point box width (pixels): " + str(arguments.align_box_width))
        print("Alignment point search width (pixels): " + str(arguments.align_search_width))
        print("Alignment point minimum structure: " + str(arguments.align_min_struct))
        print("Alignment point minimum brightness: " + str(arguments.align_min_bright))
        print("")
        print("Percentage of best frames to be stacked: " + str(arguments.stack_percent))
        print("Number of best frames to be stacked: " + str(arguments.stack_number))
        print("Normalize frame brightness: " + str(arguments.normalize_bright))
        print("Normalization black cut-off: " + str(arguments.normalize_bco))

    def stop_execution(self):
        """
        Halt the application. (There might be a more elegant way to do this!)

        :return: -
        """
        # Wait a little before exiting, so that output buffers can be purged.
        sleep(0.2)
        quit(0)
