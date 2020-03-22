from numpy import array, modf, argmin, arange, zeros
from numpy import abs as np_abs
from math import ceil

frame = arange(30.)
frame_d = zeros((45,))

shifts = arange(-3., 3., 0.1, dtype=float)

# shift_even = int(round(1.5*shift))
# print (str(shift_even))
# shift_odd = int(round(1.5*(shift-1./3.)))
#
# ilo = 15
# ihi = 26
#
# if ilo%2:
#     frame_d[int((ilo+1)/2*3)+shift_even:int(ihi/2*3)+shift_even:3] = frame[ilo+1:ihi:2]
#     frame_d[int((ilo*3+1)/2) + shift_odd:int(ihi / 2 * 3) + shift_even:3] = frame[
#                                                                                    ilo:ihi:2]
# else:
#     frame_d[int(ilo/2*3)+shift_even:int(ihi/2*3)+shift_even:3] = frame[ilo:ihi:2]

for shift in shifts:
    shift_ceil = ceil((shift - 1. / 3.) / 2) * 2
    frac = shift_ceil - shift
    frac_shift = int(round(frac * 3. / 2.))

    print("shift: " + str(shift) + ", ceil: " + str(shift_ceil) + ", frac:" + str(frac_shift))

shift = 1.3
shift_ceil_even = ceil((shift - 1. / 3.) / 2) * 2
frac = shift_ceil_even - shift
frac_shift_even = int(round(frac * 3. / 2.))

ilo = 15
ihi = 23

if ilo % 2:
    frame_low = ilo + 1 + shift_ceil_even
    frame_high = ihi + shift_ceil_even
    frame_d_low = int((ilo + 1) / 2 * 3 + frac_shift_even)
    frame_d_high = int(round(ihi / 2 * 3)) + frac_shift_even
else:
    frame_low = ilo + shift_ceil_even
    frame_high = ihi + shift_ceil_even
    frame_d_low = int(ilo / 2 * 3 + frac_shift_even)
    frame_d_high = int(ihi / 2 * 3 + frac_shift_even)
frame_d[frame_d_low:frame_d_high:3] = frame[frame_low:frame_high:2]

pass
