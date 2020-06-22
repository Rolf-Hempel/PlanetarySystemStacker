from math import ceil


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


# Main program: Set parameters and example shift value.
drizzle_factor = 1
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
