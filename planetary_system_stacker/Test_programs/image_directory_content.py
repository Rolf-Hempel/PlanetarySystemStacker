from os import listdir, rename, remove
from os.path import splitext, join, dirname

# The following lists define the allowed file extensions for still images and videos.
image_extensions = ['.tif', '.tiff', '.fit', '.fits', '.jpg', '.png']

dir_name = "D:\SW-Development\Python\PlanetarySystemStacker\Examples\Moon_2011-04-10\South"
names = [join(dir_name, name) for name in listdir(dir_name) if splitext(name)[-1].lower() in image_extensions]

print ("names: " + str(names))

