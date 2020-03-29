from math import ceil

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


drizzle_shift()
drizzle_shift_patch(-1.4, 2, 7)
