from cv2 import cvtColor, imread, COLOR_BGR2RGB, imshow, destroyAllWindows, \
    waitKey, COLOR_RGB2BGR

from miscellaneous import Miscellaneous
from exceptions import Error

input_file_name = "D:\SW-Development\Python\PlanetarySystemStacker\Examples\Jupiter_Richard\\2020-07-29-2145_3-L-Jupiter_ALTAIRGP224C_pss_p70_b48_gpp.png"
# Change colors to standard RGB
input_image = cvtColor(imread(input_file_name, -1), COLOR_BGR2RGB)

interpolation_factor = 4
image_original = Miscellaneous.shift_colors(input_image, (0,0), (0,0), interpolate_input=interpolation_factor)
imshow('Original image', cvtColor(image_original, COLOR_RGB2BGR))
waitKey()
destroyAllWindows()

shift_red = (1.3, -3.2)
shift_blue = (-2.8, 1.1)

image_shifted = Miscellaneous.shift_colors(input_image, shift_red, shift_blue,
                             interpolate_input=interpolation_factor)

imshow('RGB shifted image', cvtColor(image_shifted, COLOR_RGB2BGR))
waitKey()
destroyAllWindows()

max_shift = 5 * interpolation_factor
blurr_strength = 7
reference_channel = 1

channel_red = 0
try:
    shift_red = Miscellaneous.measure_shift(image_shifted, channel_red, reference_channel, max_shift,
                              blur_strength=blurr_strength)
    print ("Red channel shift: (" + str(round(shift_red[0]/interpolation_factor, 1)) + ", " +
           str(round(shift_red[1]/interpolation_factor, 1)) + ")")
except Error as e:
    print ("Error in measuring red shift: " + e.message)
channel_blue = 2
try:
    shift_blue = Miscellaneous.measure_shift(image_shifted, channel_blue, reference_channel, max_shift,
                               blur_strength=blurr_strength)
    print ("Blue channel shift: (" + str(round(shift_blue[0]/interpolation_factor, 1)) + ", " +
           str(round(shift_blue[1]/interpolation_factor, 1)) + ")")
except Error as e:
    print ("Error in measuring blue shift: " + e.message)

image_corrected = Miscellaneous.shift_colors(image_shifted, (-shift_red[0], -shift_red[1]),
                               (-shift_blue[0], -shift_blue[1]))

imshow('RGB shifted image', cvtColor(image_corrected, COLOR_RGB2BGR))
waitKey()
destroyAllWindows()