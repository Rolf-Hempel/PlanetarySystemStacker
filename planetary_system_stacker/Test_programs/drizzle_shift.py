from math import ceil
from time import time

from cv2 import resize, INTER_AREA, INTER_LINEAR, INTER_CUBIC, imshow, waitKey, destroyAllWindows, \
    imread, IMREAD_GRAYSCALE, namedWindow, setWindowProperty, WND_PROP_FULLSCREEN, \
    WINDOW_FULLSCREEN, GaussianBlur
from numpy import arange, zeros, float32

from miscellaneous import Miscellaneous


def drizzle_shift():
    shifts = arange(-3., 3., 0.1, dtype=float)
    for shift in shifts:
        shift_ceil = ceil(shift / 2) * 2
        frac = shift_ceil - shift - 1. / 3.
        frac_shift = int(round(frac * 3. / 2.))

        print("shift: " + str(shift) + ", ceil: " + str(shift_ceil) + ", frac:" + str(frac_shift))


def drizzle_shift_patch(shift, ilo, ihi):
    frame = arange(30.)
    frame_d = zeros((45,))

    shift_ceil_even = ceil((shift - 1. / 3.) / 2) * 2
    frac = shift_ceil_even - shift
    frac_shift_even = int(round(frac * 3. / 2.))
    shift_ceil_odd = ceil(shift / 2) * 2
    frac = shift_ceil_odd - shift - 1. / 3.
    frac_shift_odd = int(round(frac * 3. / 2.))
    if ilo % 2:
        frame_low_even = ilo + 1 + shift_ceil_even
        frame_high_even = ihi + shift_ceil_even
        frame_d_low_even = int((ilo + 1) / 2 * 3 + frac_shift_even)
        frame_d_high_even = int(round(ihi / 2 * 3)) + frac_shift_even
        frame_low_odd = ilo + shift_ceil_odd
        frame_high_odd = ihi + shift_ceil_odd
        frame_d_low_odd = int((ilo + 1. / 3.) / 2 * 3 + frac_shift_odd)
        frame_d_high_odd = int(round(ihi / 2 * 3)) + frac_shift_odd
    else:
        frame_low_even = ilo + shift_ceil_even
        frame_high_even = ihi + shift_ceil_even
        frame_d_low_even = int(ilo / 2 * 3 + frac_shift_even)
        frame_d_high_even = int(ihi / 2 * 3 + frac_shift_even)
        frame_low_odd = ilo + 1 + shift_ceil_odd
        frame_high_odd = ihi + shift_ceil_odd
        frame_d_low_odd = int((ilo + 4. / 3.) / 2 * 3 + frac_shift_odd)
        frame_d_high_odd = int(ihi / 2 * 3 + frac_shift_odd)
    frame_d[frame_d_low_even:frame_d_high_even:3] = frame[frame_low_even:frame_high_even:2]
    frame_d[frame_d_low_odd:frame_d_high_odd:3] = frame[frame_low_odd:frame_high_odd:2]

    pass


def resize_test(image, drizzle_factor, inter, patch_size):
    width = int(image.shape[1] * drizzle_factor)
    height = int(image.shape[0] * drizzle_factor)
    dim = (width, height)
    time_before = time()
    rep_count = 100
    for i in range(rep_count):
        resized = resize(image, dim, interpolation=inter)
    exec_time = (time() - time_before) / rep_count
    print("Time for one resize operation: " + str(exec_time) + ", method: " +
          str(["INTER_LINEAR", "INTER_CUBIC", "INTER_AREA"][inter - 1]))
    # imshow("Resized image", resized)
    # waitKey(0)
    # destroyAllWindows()

    start_y = 350
    start_x = 500
    patch = image[start_y:start_y + patch_size, start_x:start_x + patch_size]
    width = int(patch.shape[1] * drizzle_factor)
    height = int(patch.shape[0] * drizzle_factor)
    dim = (width, height)
    rep_count_patch = int(
        round(rep_count * image.shape[1] * image.shape[0] / (patch.shape[1] * patch.shape[0])))
    time_before = time()
    for i in range(rep_count_patch):
        resized = resize(patch, dim, interpolation=inter)
    exec_time_patched = (time() - time_before) / rep_count
    factor = exec_time_patched / exec_time
    print("Time for one patched resize operation: " + str(exec_time_patched) + ", factor: "
          + str(factor) + str("\n"))


def show_image(window_name, image, fullscreen=False):
    if fullscreen:
        namedWindow(window_name, WND_PROP_FULLSCREEN)
        setWindowProperty(window_name, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN)
    imshow(window_name, image)
    waitKey()
    destroyAllWindows()


def subpixel_shifted_frame(frame, subpixel_shift_y, subpixel_shift_x):
    interpolation_method = INTER_CUBIC
    undersampling_factor = 0.25
    interpol_factor = 10
    width = int(frame.shape[1] * interpol_factor)
    height = int(frame.shape[0] * interpol_factor)
    dim = (width, height)
    frame_extended = resize(frame, dim, interpolation=interpolation_method)
    print("shape original: " + str(frame.shape) + ", shape extended: " + str(
        frame_extended.shape))
    shift_factor = interpol_factor / undersampling_factor
    shift_y = int(round(subpixel_shift_y * shift_factor))
    shift_x = int(round(subpixel_shift_x * shift_factor))
    border_y = abs(shift_y)
    border_x = abs(shift_x)

    frame_cropped_original = frame_extended[border_y:frame_extended.shape[0] - border_y,
                             border_x:frame_extended.shape[1] - border_x]
    frame_cropped_shifted = frame_extended[
                            border_y + shift_y:frame_extended.shape[0] - border_y + shift_y,
                            border_x + shift_x:frame_extended.shape[1] - border_x + shift_x]

    width = int(round(frame_cropped_original.shape[1] / shift_factor))
    height = int(round(frame_cropped_original.shape[0] / shift_factor))
    dim = (width, height)

    frame_resized = resize(frame_cropped_original, dim, interpolation=interpolation_method)
    frame_shifted = resize(frame_cropped_shifted, dim, interpolation=interpolation_method)

    print("shape resized: " + str(frame_resized.shape) + ", shape shifted: " + str(
        frame_shifted.shape))
    return frame_resized, frame_shifted


def drizzle_shift_test():
    drizzle_shift()
    drizzle_shift_patch(-1.4, 2, 7)

def interpolation_timing():
    img = imread('../Images/Moon_Tile-024_043939_stacked_with_blurr_pp.tif', IMREAD_GRAYSCALE)
    # show_image("Original image", img, fullscreen=True)

    drizzle_factor = 3.
    patch_size = 40

    for inter in [INTER_AREA, INTER_LINEAR, INTER_CUBIC]:
        resize_test(img, drizzle_factor, inter, patch_size)

def sub_pixel_shift_test():
    img = imread('../Images/Moon_Tile-024_043939_stacked_with_blurr_pp.tif', IMREAD_GRAYSCALE)
    show_image("Original image", img, fullscreen=True)
    spx_shifty = 5.2
    spx_shiftx = 3.5

    img_resized, img_shifted = subpixel_shifted_frame(img, spx_shifty, spx_shiftx)
    # for i in range(10):
    #     show_image("Image resized", img_resized, fullscreen=True)
    #     show_image("Image shifted", img_shifted, fullscreen=True)

    gauss_width_reference = 15
    gauss_width_frame = 19
    reference_frame_blurred_intermediate = GaussianBlur(img_resized,
                                                        (gauss_width_reference, gauss_width_reference),
                                                        0).astype(float32)
    reference_frame_blurred = GaussianBlur(reference_frame_blurred_intermediate,
                                           (gauss_width_reference, gauss_width_reference), 0).astype(
                                            float32)
    frame_blurred = GaussianBlur(img_shifted, (gauss_width_frame, gauss_width_frame), 0)

    y_ap = 170
    x_ap = 200
    half_box_width = 24
    y_low = y_ap - half_box_width
    y_high = y_ap + half_box_width
    x_low = x_ap - half_box_width
    x_high = x_ap + half_box_width
    reference_box_second_phase = reference_frame_blurred[y_low: y_high, x_low: x_high]
    reference_box_first_phase = reference_box_second_phase[::2, ::2]

    search_width = 10
    shift_y_local_first_phase, shift_x_local_first_phase, success_first_phase, \
    shift_y_local_second_phase, shift_x_local_second_phase, success_second_phase = \
        Miscellaneous.multilevel_correlation(reference_box_first_phase, frame_blurred,
                                             gauss_width_frame,
                                             reference_box_second_phase, y_low, y_high, x_low, x_high,
                                             search_width,
                                             weight_matrix_first_phase=None, subpixel_solve=True)

    print("Shift in y, first phase: " + str(shift_y_local_first_phase) + ", second phase: " + str(
        shift_y_local_second_phase) + ", total: " + str(
        shift_y_local_first_phase + shift_y_local_second_phase))
    print("Shift in x, first phase: " + str(shift_x_local_first_phase) + ", second phase: " + str(
        shift_x_local_second_phase) + ", total: " + str(
        shift_x_local_first_phase + shift_x_local_second_phase))

# drizzle_shift_test()
interpolation_timing()
# sub_pixel_shift_test()
