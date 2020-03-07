import argparse
import sys

print("The command was called with " + str(len(sys.argv)) + " arguments")

if len(sys.argv) <= 1:
    print ("Running PSS in interactive mode, starting GUI.")
    exit()

def noise_type(x):
    x = int(x)
    if not 0 <= x <= 11:
        raise argparse.ArgumentTypeError("Noise level must be between 0 and 11")
    return x


def stab_size_type(x):
    x = int(x)
    if not 5 <= x <= 80:
        raise argparse.ArgumentTypeError("Stabilization patch size must be between 5% and 80%")
    return x


def stab_sw_type(x):
    x = int(x)
    if not 5 <= x <= 150:
        raise argparse.ArgumentTypeError(
            "Stabilization search width must be between 5 and 150 pixels")
    return x


def rf_percent_type(x):
    x = int(x)
    if not 3 <= x <= 30:
        raise argparse.ArgumentTypeError(
            "Percentage of best frames for reference frame computation must be between 3% and 30%")
    return x


def align_box_width_type(x):
    x = int(x)
    if not 20 <= x <= 140:
        raise argparse.ArgumentTypeError(
            "Alignment point box width must be between 20 and 140 pixels")
    return x


def align_search_width_type(x):
    x = int(x)
    if not 6 <= x <= 30:
        raise argparse.ArgumentTypeError(
            "Alignment point search width must be between 6 and 30 pixels")
    return x


def align_min_struct_type(x):
    x = float(x)
    if not 0.01 <= x <= 0.30:
        raise argparse.ArgumentTypeError(
            "Alignment point minimum structure must be between 0.01 and 0.30")
    return x


def align_min_bright_type(x):
    x = int(x)
    if not 2 <= x <= 50:
        raise argparse.ArgumentTypeError(
            "Alignment point minimum brightness must be between 2 and 50")
    return x


def stack_percent_type(x):
    x = int(x)
    if not 1 <= x <= 100:
        raise argparse.ArgumentTypeError(
            "Percentage of best frames to be stacked must be between 1 and 100")
    return x

def stack_number_type(x):
    x = int(x)
    if not 1 <= x:
        raise argparse.ArgumentTypeError(
            "Number of best frames to be stacked must be greater or equal 1")
    return x

def normalize_bco_type(x):
    x = int(x)
    if not 0 <= x <= 40:
        raise argparse.ArgumentTypeError(
            "Normalization black cut-off must be between 0 and 40")
    return x


parser = argparse.ArgumentParser()
parser.add_argument("job_input", nargs='+', help="input video files or still image folders")

parser.add_argument("--protocol", action="store_true",
                    help="Store protocol with results")
parser.add_argument("--protocol_detail", type=int, choices=[0, 1, 2], default=1,
                    help="Protocol detail level")
parser.add_argument("--buffering_level", type=int, choices=[0, 1, 2, 3, 4], default=2,
                    help="Buffering level")
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

parser.add_argument("--debayering", choices=["Auto detect color", "Grayscale", "RGB", "RGB", "BGR",
                                             "Force Bayer RGGB", "Force Bayer GRBG",
                                             "Force Bayer GBRG", "Force Bayer BGGR"],
                    default="Auto detect color", help="Debayering option")
parser.add_argument("--noise", type=noise_type, default=7, help="Noise level (add Gaussian blur)")
parser.add_argument("--stab_mode", choices=["Surface", "Planet"], default="Surface",
                    help="Frame stabilization mode")
parser.add_argument("--stab_size", type=stab_size_type, default=33,
                    help="Stabilization patch size (% of frame)")
parser.add_argument("--stab_sw", type=stab_sw_type, default=34,
                    help="Stabilization search width (pixels)")
parser.add_argument("--rf_percent", type=rf_percent_type, default=5,
                    help="Percentage of best frames for reference frame computation")

parser.add_argument("--align_box_width", type=align_box_width_type, default=48,
                    help="Alignment point box width (pixels)")
parser.add_argument("--align_search_width", type=align_search_width_type, default=14,
                    help="Alignment point search width (pixels)")
parser.add_argument("--align_min_struct", type=align_min_struct_type, default=0.04,
                    help="Alignment point minimum structure")
parser.add_argument("--align_min_bright", type=align_min_bright_type, default=10,
                    help="Alignment point minimum brightness")

parser.add_argument("--stack_percent", type=stack_percent_type, default=10,
                    help="Percentage of best frames to be stacked")
parser.add_argument("--stack_number", type=stack_number_type,
                    help="Number of best frames to be stacked")
parser.add_argument("--normalize_bright", action="store_true",
                    help="Normalize frame brightness")
parser.add_argument("--normalize_bco", type=normalize_bco_type, default=15,
                    help="Normalization black cut-off")

args = parser.parse_args()

print(str(args.job_input))
print("Store protocol with results: " + str(args.protocol))
print("Protocol detail level: " + str(args.protocol_detail))
print("Buffering level: " + str(args.buffering_level))
print("Image format for output: " + args.out_format)
print("Add number of stacked frames to output file name: " + str(args.name_add_f))
print("Add percentage of stacked frames to output file name: " + str(args.name_add_p))
print("Add alignment point box size (pixels) to output file name: " + str(args.name_add_apb))
print("Add number of alignment points to output file name: " + str(args.name_add_apn))
print("")
print("Debayering option: " + args.debayering)
print("Noise level: " + str(args.noise))
print("Frame stabilization mode: " + args.stab_mode)
print("Stabilization patch size (% of frame): " + str(args.stab_size))
print("Stabilization search width (pixels): " + str(args.stab_sw))
print("Percentage of best frames for reference frame computation: " + str(args.rf_percent))
print("")
print("Alignment point box width (pixels): " + str(args.align_box_width))
print("Alignment point search width (pixels): " + str(args.align_search_width))
print("Alignment point minimum structure: " + str(args.align_min_struct))
print("Alignment point minimum brightness: " + str(args.align_min_bright))
print("")
print("Percentage of best frames to be stacked: " + str(args.stack_percent))
print("Number of best frames to be stacked: " + str(args.stack_number))
print("Normalize frame brightness: " + str(args.normalize_bright))
print("Normalization black cut-off: " + str(args.normalize_bco))