from cv2 import resize, cvtColor, imread, COLOR_BGR2RGB, INTER_CUBIC, imshow, destroyAllWindows, \
    waitKey, COLOR_RGB2BGR
from numpy import empty


def shift_colors(input_image, interp_factor, shift_red, shift_blue):
    if len(input_image.shape) != 3:
        return input_image

    dim_y = input_image.shape[0] * interp_factor
    dim_x = input_image.shape[1] * interp_factor
    type = input_image.dtype

    if interp_factor != 1:
        interp_image = resize(input_image, (dim_x, dim_y), interpolation=INTER_CUBIC)
    else:
        interp_image = input_image

    s_red_y = round(interp_factor * shift_red[0])
    s_red_x = round(interp_factor * shift_red[1])
    s_blue_y = round(interp_factor * shift_blue[0])
    s_blue_x = round(interp_factor * shift_blue[1])

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

    min_y_g = max(y_low_target_r, y_low_target_b)
    max_y_g = min(y_high_target_r, y_high_target_b)
    min_x_g = max(x_low_target_r, x_low_target_b)
    max_x_g = min(x_high_target_r, x_high_target_b)

    output_image_interp = empty((max_y_g - min_y_g, max_x_g - min_x_g, 3), dtype=type)
    output_image_interp[:, :, 0] = interp_image[min_y_g - s_red_y:max_y_g - s_red_y,
                                   min_x_g - s_red_x:max_x_g - s_red_x, 0]
    output_image_interp[:, :, 1] = interp_image[min_y_g:max_y_g, min_x_g:max_x_g, 1]
    output_image_interp[:, :, 2] = interp_image[min_y_g - s_blue_y:max_y_g - s_blue_y,
                                   min_x_g - s_blue_x:max_x_g - s_blue_x, 2]

    if interp_factor != 1:
        output_image = resize(output_image_interp, (input_image.shape[1], input_image.shape[0]),
                              interpolation=INTER_CUBIC)
    else:
        output_image = output_image_interp

    return output_image


input_file_name = "D:\SW-Development\Python\PlanetarySystemStacker\Examples\Jupiter_Richard\\2020-07-29-2145_3-L-Jupiter_ALTAIRGP224C_pss_p70_b48_gpp.png"
# Change colors to standard RGB
input_image = cvtColor(imread(input_file_name, -1), COLOR_BGR2RGB)

interpolation_factor = 4
shift_red = (1., 3.2)
shift_blue = (-2.4, 5.)

image_shifted = shift_colors(input_image, interpolation_factor, shift_red, shift_blue)

imshow('RGB shifted image', cvtColor(image_shifted, COLOR_RGB2BGR))
waitKey()
destroyAllWindows()