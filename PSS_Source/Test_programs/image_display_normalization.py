from cv2 import imread, IMREAD_UNCHANGED, imshow, waitKey, normalize, NORM_MINMAX
from numpy import uint8

# filename = 'Photodisc.png'
# filename = "E:\SW-Development\Python\PlanetarySystemStacker\PSS_Source\Images\\another_short_video.jpg"
filename = "E:\SW-Development\Python\PlanetarySystemStacker\Examples\AVI_Chris-Garry\\20110929_005012_jupiter_gbrg_ULRA_pss_gpp.jpg"
input_image = imread(filename, IMREAD_UNCHANGED)
display_image = normalize(input_image, None, alpha=0, beta=255, norm_type=NORM_MINMAX)
imshow('image',display_image)
waitKey(0)
