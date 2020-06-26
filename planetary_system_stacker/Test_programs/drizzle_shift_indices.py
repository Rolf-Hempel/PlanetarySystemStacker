from math import ceil
from numpy import ndarray, zeros

from miscellaneous import Miscellaneous


def compute_bounds(x_low, x_high, drizzle_factor, shift):
    """
    Compute the index bounds in the original (non-drizzled) frame from where the AP patch is to be
    copied, and the (drizzled) index bounds in the summation buffer. The data are copied with stride
     "drizzle_factor".

    :param x_low: Lower index bound of AP patch in original frame.
    :param x_high: Upper index bound of AP patch in original frame.
    :param drizzle_factor: (integer) drizzle factor, typically 2 or 3.
    :param shift: Shift (float) between AP patches in summation buffer and original frame,
                  expressed in non-drizzled units.
    :return: Tuple (x_low_from, x_high_from, xd_low_to, xd_high_to, x_offset) with:
             x_low_from: Lower source index bound for copy operation in original frame.
             x_high_from: Upper source index bound for copy operation in original frame (stride 1).
             xd_low_to: Lower target index bound for copy operation in summation buffer.
             xd_high_to: Upper target index bound for copy operation in summation buffer
                         (stride "drizzle_factor").
             x_offset: Index offset in target patch for begin of copy operation
                       (0, 1, ..., drizzle_factor-1).
    """

    # Compute integer shift in drizzled target grid closest to given shift (float).
    shift_d = int(round(drizzle_factor * shift))

    # Translate into original index coordinates (not integer any more).
    shift_rounded = shift_d / drizzle_factor

    # If the shift stays in the original grid, the offset in the drizzled grid patch is zero.
    if shift_rounded.is_integer():
        shift_rounded = int(shift_rounded)
        x_low_from = x_low + shift_rounded
        x_high_from = x_high + shift_rounded
        x_offset = 0

    # Otherwise the target indices in the drizzled grid patch start at a non-zero offset.
    else:
        shift_ceil = ceil(shift)
        x_low_from = x_low + shift_ceil
        x_high_from = x_high + shift_ceil
        x_offset = int(drizzle_factor * shift_ceil - shift_d)

    xd_low_to = drizzle_factor * x_low_from - shift_d
    xd_high_to = drizzle_factor * x_high_from - shift_d

    return (x_low_from, x_high_from, xd_low_to, xd_high_to, x_offset)

def compute_bounds_2d(y_low, y_high, x_low, x_high, shift_y, shift_x, drizzle_factor):
    """
    Compute the index bounds in the original (non-drizzled) frame from where the AP patch is
    to be copied, and the (drizzled) index bounds in the summation buffer. The data are copied with
    stride "drizzle_factor".

    :param y_low: Lower y index bound of AP patch in original frame.
    :param y_high: Upper y index bound of AP patch in original frame.
    :param x_low: Lower x index bound of AP patch in original frame.
    :param x_high: Upper x index bound of AP patch in original frame.
    :param shift_y: Shift (float) in y between AP patches in summation buffer and original frame,
                    expressed in non-drizzled units.
    :param shift_x: Shift (float) in x between AP patches in summation buffer and original frame,
                    expressed in non-drizzled units.
    :param drizzle_factor: (integer) drizzle factor, typically 2 or 3.
    :return: Tuple (y_low_from, y_high_from, x_low_from, x_high_from, y_offset, x_offset) with:
             y_low_from: Lower source y index bound for copy operation in original frame.
             y_high_from: Upper source y index bound for copy operation in original frame (
             stride 1).
             x_low_from: Lower source x index bound for copy operation in original frame.
             x_high_from: Upper source x index bound for copy operation in original frame (
             stride 1).
             y_offset: Index offset in y direction in target patch for begin of copy operation
                       (0, 1, ..., drizzle_factor-1).
             x_offset: Index offset in x direction in target patch for begin of copy operation
                       (0, 1, ..., drizzle_factor-1).
    """

    # Compute integer shift in both directions in drizzled target grid closest to given shift
    # (float).
    shift_d_y = int(round(drizzle_factor * shift_y))
    shift_d_x = int(round(drizzle_factor * shift_x))

    # Translate into original index coordinates (not integer any more).
    shift_rounded_y = shift_d_y / drizzle_factor
    shift_rounded_x = shift_d_x / drizzle_factor

    # If the shift stays in the original grid, the offset in the drizzled grid patch is zero.
    if shift_rounded_y.is_integer():
        shift_rounded_y = int(shift_rounded_y)
        y_low_from = y_low + shift_rounded_y
        y_high_from = y_high + shift_rounded_y
        y_offset = 0

    # Otherwise the target indices in the drizzled grid patch start at a non-zero offset.
    else:
        shift_ceil = ceil(shift_y)
        y_low_from = y_low + shift_ceil
        y_high_from = y_high + shift_ceil
        y_offset = int(drizzle_factor * shift_ceil - shift_d_y)

    # Do the same for the x coordinate direction.
    if shift_rounded_x.is_integer():
        shift_rounded_x = int(shift_rounded_x)
        x_low_from = x_low + shift_rounded_x
        x_high_from = x_high + shift_rounded_x
        x_offset = 0

    # Otherwise the target indices in the drizzled grid patch start at a non-zero offset.
    else:
        shift_ceil = ceil(shift_x)
        x_low_from = x_low + shift_ceil
        x_high_from = x_high + shift_ceil
        x_offset = int(drizzle_factor * shift_ceil - shift_d_x)

    return (y_low_from, y_high_from, x_low_from, x_high_from, y_offset, x_offset)

def equalize_ap_patch(patch, offset_counters, stack_size, drizzle_factor):
    """
    During drizzling the AP patch gets different numbers of frame contributions at different
    positions in the drizzling pattern, depending on the distribution of sup-pixel shift values.
    The array "offset_counters" contains these numbers, with the sum of all those numbers being the
    total stack size.

    Scale all patch entries with "total stack size" / "offset counter". The result are uniform
    patch entries which look as if there had been "total stack size" contributions in all drizzle
    pattern locations.

    Special care has to be taken at locations where not a single frame contributed (called "holes").
    Here insert the average of patch values at all drizzle positions over a circle with minimum
    radius containing at least one non-zero contribution.

    :param patch: AP buffer after stacking with drizzling.
    :param offset_counters: Integer array, size drizzle_factor x drizzle_factor, with contribution
                            counts at all drizzle locations.
    :param stack_size: Overall number of frames stacked
    :param drizzle_factor: Factor by which the number of pixels is multiplied in each direction.
    :return: Number of drizzle locations without frame contributions.
    """

    dim_y, dim_x = patch.shape
    holes = []

    # First normalize the patch locations with non-zero contributions.
    for y_offset in range(drizzle_factor):
        for x_offset in range(drizzle_factor):
            if offset_counters[y_offset, x_offset]:
                normalization_factor = stack_size/offset_counters[y_offset, x_offset]
                patch[y_offset:dim_y:drizzle_factor, x_offset:dim_x:drizzle_factor] *= normalization_factor
            # For locations with zero contributions (holes) remember the location.
            else:
                holes.append((y_offset, x_offset))

    # Now fill the holes with interpolated values from locations close by.
    for (y_offset, x_offset) in holes:

        # Initialize the buffer locations witz zeros.
        patch[y_offset:dim_y:drizzle_factor, x_offset:dim_x:drizzle_factor] = 0.

        # Look for the circle with smallest radius around the hole with at least one non-zero
        # contribution.
        for radius in range(1, drizzle_factor):
            n_success = 0
            for (y, x) in Miscellaneous.circle_around(y_offset, x_offset, radius):
                if 0<=y<drizzle_factor and 0<=x<drizzle_factor and (y, x) not in holes:
                    # A non-zero entry is found, add its contribution to the buffer.
                    patch[y_offset:dim_y:drizzle_factor, x_offset:dim_x:drizzle_factor] += patch[y:dim_y:drizzle_factor, x:dim_x:drizzle_factor]
                    n_success += 1

            # There was at least one non-zero contribution on the circle with this radius. Normalize
            # the buffer with the number of contributions and continue with the next hole.
            if n_success:
                patch[y_offset:dim_y:drizzle_factor, x_offset:dim_x:drizzle_factor] *= 1./n_success
                break

    # Return the number of holes in the drizzle pattern.
    return len(holes)


def test_index_computations():
    # Set parameters and example shift value.
    drizzle_factor = 3
    x_low = 4
    x_high = 6
    shift = -2.52

    # Compute the index bounds and offset.
    x_low_from, x_high_from, xd_low_to, xd_high_to, x_offset = compute_bounds(x_low, x_high,
                                                                              drizzle_factor,
                                                                              shift)

    # Print the results.
    print("x_low_from: " + str(x_low_from) + ", x_high_from: " + str(
        x_high_from) + "\nxd_low_to: " + str(xd_low_to) + ", xd_high_to: " + str(
        xd_high_to) + "\nx_offset: " + str(x_offset))

def test_index_computations_2d():
    # Set parameters and example shift value.
    drizzle_factor = 3
    y_low = 4
    y_high = 6
    shift_y = 3.14
    x_low = 4
    x_high = 6
    shift_x = -2.52

    # Compute the index bounds and offset.
    y_low_from, y_high_from, x_low_from, x_high_from, y_offset, x_offset = \
        compute_bounds_2d(y_low, y_high, x_low, x_high, shift_y, shift_x, drizzle_factor)

    # Print the results.
    print("y_low_from: " + str(y_low_from) + ", y_high_from: " + str(
        y_high_from) + "\ny_offset: " + str(y_offset) + "\nx_low_from: " + str(
        x_low_from) + ", x_high_from: " + str(x_high_from) + "\nx_offset: " + str(x_offset))

def test_remap_rigid_drizzled():
    drizzle_factor = 3
    y_dim = 20
    x_dim = 16
    y_dim_patch = 3
    x_dim_patch = 4
    frame = ndarray(shape=(y_dim, x_dim), dtype=int)
    for y in range(y_dim):
        for x in range(x_dim):
            frame[y, x] = 1000*y + x
    patch = zeros((y_dim_patch*drizzle_factor, x_dim_patch*drizzle_factor), dtype=int)

    y_low = 10
    y_high = y_low + y_dim_patch
    shift_y = 1.26
    x_low = 6
    x_high = x_low + x_dim_patch
    shift_x = -1.6

    y_low_from, y_high_from, x_low_from, x_high_from, y_offset, x_offset = \
        compute_bounds_2d(y_low, y_high, x_low, x_high, shift_y, shift_x, drizzle_factor)

    print("y_low_from: " + str(y_low_from) + ", y_high_from: " + str(
        y_high_from) + "\ny_offset: " + str(y_offset) + "\nx_low_from: " + str(
        x_low_from) + ", x_high_from: " + str(x_high_from) + "\nx_offset: " + str(x_offset))

    patch[y_offset::drizzle_factor, x_offset::drizzle_factor] = \
        frame[y_low_from:y_high_from, x_low_from:x_high_from]

    print ("patch: " + str(patch))

def test_equalize_ap_patch():
    drizzle_factor = 3
    y_dim = 3
    x_dim = 4
    y_dim_patch = y_dim*drizzle_factor
    x_dim_patch = x_dim*drizzle_factor
    patch = zeros(shape=(y_dim_patch, x_dim_patch), dtype=float)
    offset_counters = zeros(shape=(drizzle_factor, drizzle_factor), dtype=int)
    offset_counters[0, 0] = 1
    offset_counters[0, 1] = 2
    offset_counters[1, 0] = 1
    offset_counters[2, 2] = 3
    stack_size = offset_counters.sum()
    for y_offset in range(drizzle_factor):
        for x_offset in range(drizzle_factor):
            counter = offset_counters[y_offset, x_offset]
            if counter:
                for y in range(y_offset, y_dim_patch, drizzle_factor):
                    for x in range(x_offset, x_dim_patch, drizzle_factor):
                        patch[y, x] = counter * (1000*y + x)
                        # patch[y, x] = counter

    print("patch before equalization: \n" + str(patch))
    len_holes = equalize_ap_patch(patch, offset_counters, stack_size, drizzle_factor)
    print("patch after equalization: \n" + str(patch))
    print ("Number of holes in drizzle pattern: " + str(len_holes))


# Main program: Control the test to be performed.
# test_index_computations()
# test_index_computations_2d()
# test_remap_rigid_drizzled()
test_equalize_ap_patch()
