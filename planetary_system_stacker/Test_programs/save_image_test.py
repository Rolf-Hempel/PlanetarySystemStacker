from numpy import ndarray
from numpy import uint8, uint16

from frames import Frames

image = ndarray((500, 500), dtype=uint8)
image[:,:] = 128
color = False
file_name = 'D:\SW-Development\Python\PlanetarySystemStacker\planetary_system_stacker\Images/new_file.tiff'
avoid_overwriting = True

# Save the stacked image as 16bit int.
Frames.save_image(file_name, image, color=color, avoid_overwriting=avoid_overwriting)

