from math import ceil
from time import time

from cv2 import resize, INTER_AREA, INTER_LINEAR, INTER_CUBIC, imshow, waitKey, destroyAllWindows, \
    imread, IMREAD_UNCHANGED, IMREAD_GRAYSCALE, namedWindow, setWindowProperty, WND_PROP_FULLSCREEN, \
    WINDOW_FULLSCREEN
from numpy import arange, zeros


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
    frame_cropped_shifted = frame_extended[border_y + shift_y:frame_extended.shape[0] - border_y + shift_y,
                            border_x + shift_x:frame_extended.shape[1] - border_x + shift_x]

    width = int(round(frame_cropped_original.shape[1] / shift_factor))
    height = int(round(frame_cropped_original.shape[0] / shift_factor))
    dim = (width, height)

    frame_resized = resize(frame_cropped_original, dim, interpolation=interpolation_method)
    frame_shifted = resize(frame_cropped_shifted, dim, interpolation=interpolation_method)

    print("shape resized: " + str(frame_resized.shape) + ", shape shifted: " + str(
        frame_shifted.shape))
    return frame_resized, frame_shifted


# drizzle_shift()
# drizzle_shift_patch(-1.4, 2, 7)

img = imread('../Images/Moon_Tile-024_043939_stacked_with_blurr_pp.tif', IMREAD_GRAYSCALE)
show_image("Original image", img, fullscreen=True)

# drizzle_factor = 3.
# patch_size = 40
#
# for inter in [INTER_AREA, INTER_LINEAR, INTER_CUBIC]:
#     resize_test(img, drizzle_factor, inter, patch_size)

spx_shifty = 5.3
spx_shiftx = -3.3

img_resized, img_shifted = subpixel_shifted_frame(img, spx_shifty, spx_shiftx)
for i in range(10):
    show_image("Image resized", img_resized, fullscreen=True)
    show_image("Image shifted", img_shifted, fullscreen=True)
