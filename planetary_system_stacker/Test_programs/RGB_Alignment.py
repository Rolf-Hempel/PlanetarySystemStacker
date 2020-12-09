from cv2 import cvtColor, imread, COLOR_BGR2RGB, imshow, destroyAllWindows, \
    waitKey, COLOR_RGB2BGR
from os.path import splitext

from miscellaneous import Miscellaneous
from frames import Frames
from exceptions import Error

def test_detailed(input_file_name, shift_red, shift_blue, interpolation_factor, reduction_factor):
    # Change colors to standard RGB
    input_image = cvtColor(imread(input_file_name, -1), COLOR_BGR2RGB)
    image_original = Miscellaneous.shift_colors(input_image, (0, 0), (0, 0),
                                                interpolate_input=interpolation_factor,
                                                reduce_output=reduction_factor)
    imshow('Original image', cvtColor(image_original, COLOR_RGB2BGR))
    waitKey()
    destroyAllWindows()

    print("Input shifts:")
    print("Red channel shift: (" + str(round(shift_red[0], 1)) + ", " +
          str(round(shift_red[1], 1)) + ")")
    print("Blue channel shift: (" + str(round(shift_blue[0], 1)) + ", " +
          str(round(shift_blue[1], 1)) + ")")

    image_shifted = Miscellaneous.shift_colors(input_image, shift_red, shift_blue,
                                               interpolate_input=interpolation_factor)

    imshow('RGB shifted image', cvtColor(image_shifted, COLOR_RGB2BGR))
    waitKey()
    destroyAllWindows()

    max_shift = 5 * interpolation_factor
    blurr_strength = 7
    reference_channel = 1

    print("\nMeasured shifts:")
    channel_red = 0
    try:
        shift_red = Miscellaneous.measure_rgb_shift(image_shifted, channel_red, reference_channel,
                                                    max_shift, blur_strength=blurr_strength)
        print("Red channel shift: (" + str(round(shift_red[0] / interpolation_factor, 1)) + ", " +
              str(round(shift_red[1] / interpolation_factor, 1)) + ")")
    except Error as e:
        print("Error in measuring red shift: " + e.message)
    channel_blue = 2
    try:
        shift_blue = Miscellaneous.measure_rgb_shift(image_shifted, channel_blue, reference_channel,
                                                     max_shift, blur_strength=blurr_strength)
        print("Blue channel shift: (" + str(round(shift_blue[0] / interpolation_factor, 1)) + ", " +
              str(round(shift_blue[1] / interpolation_factor, 1)) + ")")
    except Error as e:
        print("Error in measuring blue shift: " + e.message)

    image_corrected = Miscellaneous.shift_colors(image_shifted, (-shift_red[0], -shift_red[1]),
                                                 (-shift_blue[0], -shift_blue[1]))

    imshow('RGB shifted image', cvtColor(image_corrected, COLOR_RGB2BGR))
    waitKey()
    destroyAllWindows()

def create_shifted_image(input_file_name, shift_red, shift_blue, interpolation_factor):
    # Change colors to standard RGB
    input_image = cvtColor(imread(input_file_name, -1), COLOR_BGR2RGB)
    shifted_image = Miscellaneous.shift_colors(input_image, shift_red, shift_blue,
                               interpolate_input=interpolation_factor,
                               reduce_output=interpolation_factor)

    output_file_name= splitext(input_file_name)[0] + '_rgb-shifted.png'
    Frames.save_image(output_file_name, shifted_image,
                      color=(len(shifted_image.shape) == 3), avoid_overwriting=False,
                      header="PlanetarySystemStacker")

def test_auto_rgbg_align(input_file_name, interpolation_factor, blur_strength):
    # Change colors to standard RGB
    shifted_image = cvtColor(imread(input_file_name, -1), COLOR_BGR2RGB)

    imshow('RGB-aligned image', cvtColor(shifted_image, COLOR_RGB2BGR))
    waitKey()
    destroyAllWindows()

    max_shift = 5
    corrected_image, shift_red, shift_blue = Miscellaneous.auto_rgb_align(shifted_image, max_shift,
        interpolation_factor=interpolation_factor, blur_strenght=blur_strength)

    output_file_name = splitext(input_file_name)[0] + '_corrected.png'
    Frames.save_image(output_file_name, corrected_image,
                      color=(len(corrected_image.shape) == 3), avoid_overwriting=False,
                      header="PlanetarySystemStacker")

    print("Corrections applied:")
    print("Red channel shift: (" + str(round(shift_red[0], 1)) + ", " +
          str(round(shift_red[1], 1)) + ")")
    print("Blue channel shift: (" + str(round(shift_blue[0], 1)) + ", " +
          str(round(shift_blue[1], 1)) + ")")

    imshow('RGB-aligned image', cvtColor(corrected_image, COLOR_RGB2BGR))
    waitKey()
    destroyAllWindows()

input_file_name = "D:\SW-Development\Python\PlanetarySystemStacker\Examples\Jupiter_Richard\\2020-07-29-2145_3-L-Jupiter_ALTAIRGP224C_pss_p70_b48_gpp.png"
input_file_name_shifted = "D:\SW-Development\Python\PlanetarySystemStacker\Examples\Jupiter_Richard\\2020-07-29-2145_3-L-Jupiter_ALTAIRGP224C_pss_p70_b48_gpp_rgb-shifted.png"

# input_file_name = "D:\SW-Development\Python\PlanetarySystemStacker\Examples\Jupiter_Richard\\2020-07-29-2145_3-L-Jupiter_ALTAIRGP224C_pss_p70_b48.png"
# input_file_name_shifted = "D:\SW-Development\Python\PlanetarySystemStacker\Examples\Jupiter_Richard\\2020-07-29-2145_3-L-Jupiter_ALTAIRGP224C_pss_p70_b48_rgb-shifted.png"

interpolation_factor = 4
reduction_factor = 1
blur_strength = 7
shift_red = (1.3, -3.2)
shift_blue = (-1.8, 1.1)

# test_detailed(input_file_name, shift_red, shift_blue, interpolation_factor, reduction_factor)
create_shifted_image(input_file_name, shift_red, shift_blue, interpolation_factor)
test_auto_rgbg_align(input_file_name_shifted, interpolation_factor, blur_strength)