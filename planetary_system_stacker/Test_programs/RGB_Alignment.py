from cv2 import resize, cvtColor, imread, COLOR_BGR2RGB, INTER_CUBIC, imshow, destroyAllWindows, \
    waitKey, COLOR_RGB2BGR, GaussianBlur, matchTemplate, TM_CCORR_NORMED, minMaxLoc
from numpy import empty, float32
from exceptions import ArgumentError


def shift_colors(input_image, shift_red, shift_blue, interpolate_input= 1, reduce_output=1):
    """
    Shift the red and blue channel of a color image in y and x direction, and leave the green
    channel unchanged. The shift is sub-pixel accurate if the input is interpolated. Optionally the
    resulting image is reduced in size by a given factor.

    :param input_image: Three-channel RGB image.
    :param shift_red: Tuple (shift_y, shift_x) with shifts in y and x direction for the red channel.
                      After multiplication with the interpolate_input factor (if given) the shifts
                      are rounded to the next integer value. This enables sub-pixel shifts at the
                      original image scale.
    :param shift_blue: Tuple (shift_y, shift_x) with shifts in y and x direction for the blue channel.
    :param interpolate_input: If set, it must be an integer >= 1. The input image is interpolated in
                              both directions by this factor before the shift is applied.
    :param reduce_output: If set, it must be an integer >= 1. The intermediate image is reduced in
                          size by this factor. Please note that interpolate_input = reduce_output
                          assures that the input and output images are the same size.
    :return: Three-channel RGB image with the color shifts applied.
    """

    # Immediately return for monochrome input.
    if len(input_image.shape) != 3:
        return input_image

    # If interpolation is requested, resize the input image and multiply the shift values.
    if interpolate_input != 1:
        dim_y = input_image.shape[0] * interpolate_input
        dim_x = input_image.shape[1] * interpolate_input
        interp_image = resize(input_image, (dim_x, dim_y), interpolation=INTER_CUBIC)
        s_red_y = round(interpolate_input * shift_red[0])
        s_red_x = round(interpolate_input * shift_red[1])
        s_blue_y = round(interpolate_input * shift_blue[0])
        s_blue_x = round(interpolate_input * shift_blue[1])
    else:
        dim_y = input_image.shape[0]
        dim_x = input_image.shape[1]
        interp_image = input_image
        s_red_y = round(shift_red[0])
        s_red_x = round(shift_red[1])
        s_blue_y = round(shift_blue[0])
        s_blue_x = round(shift_blue[1])

    # Remember the data type of the input image. The output image will be of the same type.
    type = input_image.dtype

    # In the following, for both the red and blue channels the index areas are computed for which
    # no shift is beyond the image dimensions. First treat the y coordinate.
    y_low_source_r = -s_red_y
    y_high_source_r = dim_y - s_red_y
    y_low_target_r = 0
    # If the shift reaches beyond the frame, reduce the copy area.
    if y_low_source_r < 0:
        y_low_target_r = -y_low_source_r
        y_low_source_r = 0
    if y_high_source_r > dim_y:
        y_high_source_r = dim_y
    y_high_target_r = y_low_target_r + y_high_source_r - y_low_source_r

    y_low_source_b = -s_blue_y
    y_high_source_b = dim_y - s_blue_y
    y_low_target_b = 0
    # If the shift reaches beyond the frame, reduce the copy area.
    if y_low_source_b < 0:
        y_low_target_b = -y_low_source_b
        y_low_source_b = 0
    if y_high_source_b > dim_y:
        y_high_source_b = dim_y
    y_high_target_b = y_low_target_b + y_high_source_b - y_low_source_b

    # The same for the x coordinate.
    x_low_source_r = -s_red_x
    x_high_source_r = dim_x - s_red_x
    x_low_target_r = 0
    # If the shift reaches beyond the frame, reduce the copy area.
    if x_low_source_r < 0:
        x_low_target_r = -x_low_source_r
        x_low_source_r = 0
    if x_high_source_r > dim_x:
        x_high_source_r = dim_x
    x_high_target_r = x_low_target_r + x_high_source_r - x_low_source_r

    x_low_source_b = -s_blue_x
    x_high_source_b = dim_x - s_blue_x
    x_low_target_b = 0
    # If the shift reaches beyond the frame, reduce the copy area.
    if x_low_source_b < 0:
        x_low_target_b = -x_low_source_b
        x_low_source_b = 0
    if x_high_source_b > dim_x:
        x_high_source_b = dim_x
    x_high_target_b = x_low_target_b + x_high_source_b - x_low_source_b

    # Now the coordinate bounds can be computed for which the output image has entries in all three
    # channels. The green channel is not shifted and can just be copied in place.
    min_y_g = max(y_low_target_r, y_low_target_b)
    max_y_g = min(y_high_target_r, y_high_target_b)
    min_x_g = max(x_low_target_r, x_low_target_b)
    max_x_g = min(x_high_target_r, x_high_target_b)

    # Allocate space for the output image (still not reduced in size), and copy the three color
    # channels with the appropriate shifts applied.
    output_image_interp = empty((max_y_g - min_y_g, max_x_g - min_x_g, 3), dtype=type)
    output_image_interp[:, :, 0] = interp_image[min_y_g - s_red_y:max_y_g - s_red_y,
                                   min_x_g - s_red_x:max_x_g - s_red_x, 0]
    output_image_interp[:, :, 1] = interp_image[min_y_g:max_y_g, min_x_g:max_x_g, 1]
    output_image_interp[:, :, 2] = interp_image[min_y_g - s_blue_y:max_y_g - s_blue_y,
                                   min_x_g - s_blue_x:max_x_g - s_blue_x, 2]

    # If output size reduction is specified, resize the intermediate image.
    if reduce_output != 1:
        output_image = resize(output_image_interp, (round(input_image.shape[1]/reduce_output),
                                                    round(input_image.shape[0]/reduce_output)),
                              interpolation=INTER_CUBIC)
    else:
        output_image = output_image_interp

    return output_image

def measure_shift(image, channel_id, reference_id, max_shift, blur_strength=None):
    """
    Measure the shift between two color channels of a three-color RGB image. Before measuring the
    shift with cross correlation, a Gaussian blur can be applied to both channels to reduce noise.

    :param image: Input image (3 channel RGB).
    :param channel_id: Index of the channel, the shift of which is to be measured against the
                       reference channel.
    :param reference_id: Index of the reference channel, usually 1 for "green".
    :param max_shift: Maximal search space radius in pixels.
    :param blur_strength: Strength of the Gaussian blur to be applied first.
    :return: Tuple (shift_y, shift_x) with coordinate shifts in y and x.
    """

    # Test for invalid channel ids.
    if not (0 <= channel_id <= 2):
        raise ArgumentError("Invalid color channel id: " + str(channel_id))
    if not (0 <= reference_id <= 2):
        raise ArgumentError("Invalid reference channel id: " + str(reference_id))

    # Reduce the size of the window for which the correlation will be computed to make space for the
    # search space around it.
    channel_window = image[max_shift:-max_shift, max_shift:-max_shift, channel_id]
    channel_reference = image[:, :, reference_id]

    # Apply Gaussian blur if requested.
    if blur_strength:
        channel_blurred = GaussianBlur(channel_window, (blur_strength, blur_strength), 0)
        channel_reference_blurred = GaussianBlur(channel_reference, (blur_strength, blur_strength), 0)
    else:
        channel_blurred = channel_window
        channel_reference_blurred = channel_reference

    # Compute the normalized cross correlation.
    result = matchTemplate((channel_reference_blurred).astype(float32),
                           channel_blurred.astype(float32), TM_CCORR_NORMED)

    # Determine the position of the maximum correlation, and compute the corresponding shifts.
    minVal, maxVal, minLoc, maxLoc = minMaxLoc(result)
    return (max_shift - maxLoc[1], max_shift - maxLoc[0])



input_file_name = "D:\SW-Development\Python\PlanetarySystemStacker\Examples\Jupiter_Richard\\2020-07-29-2145_3-L-Jupiter_ALTAIRGP224C_pss_p70_b48_gpp.png"
# Change colors to standard RGB
input_image = cvtColor(imread(input_file_name, -1), COLOR_BGR2RGB)

interpolation_factor = 4
image_original = shift_colors(input_image, (0,0), (0,0), interpolate_input=interpolation_factor)
imshow('Original image', cvtColor(image_original, COLOR_RGB2BGR))
waitKey()
destroyAllWindows()

shift_red = (-2.3, 3.2)
shift_blue = (-2.8, 1.1)

image_shifted = shift_colors(input_image, shift_red, shift_blue,
                             interpolate_input=interpolation_factor)

imshow('RGB shifted image', cvtColor(image_shifted, COLOR_RGB2BGR))
waitKey()
destroyAllWindows()

max_shift = 5 * interpolation_factor
blurr_strength = 7
reference_channel = 1

channel_red = 0
shift_red = measure_shift(image_shifted, channel_red, reference_channel, max_shift,
                          blur_strength=blurr_strength)
print ("Red channel shift: (" + str(round(shift_red[0]/interpolation_factor, 1)) + ", " +
       str(round(shift_red[1]/interpolation_factor, 1)) + ")")
channel_blue = 2
shift_blue = measure_shift(image_shifted, channel_blue, reference_channel, max_shift,
                           blur_strength=blurr_strength)
print ("Blue channel shift: (" + str(round(shift_blue[0]/interpolation_factor, 1)) + ", " +
       str(round(shift_blue[1]/interpolation_factor, 1)) + ")")

image_corrected = shift_colors(image_shifted, (-shift_red[0], -shift_red[1]),
                               (-shift_blue[0], -shift_blue[1]))

imshow('RGB shifted image', cvtColor(image_corrected, COLOR_RGB2BGR))
waitKey()
destroyAllWindows()