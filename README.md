# PlanetarySystemStacker
Produce a sharp image of a planetary system object (moon, sun, planets) from many seeing-affected frames using the "lucky imaging" technique.

The program is mainly targeted at extended objects (moon, sun), but it should work as well for the (easier) planet case. A full working prototype written in Python, still without a graphical user interface, has been finished. Results obtained in first tests show the same image quality as the established software AutoStakkert!3.

Input to the program can be either video files or a directories containing still images. The following algorithmic steps are performed:

* First, all frames are ranked by their overall image quality.
* On the best frame, a rectangular patch with the most pronounced structure in x and y is identified automatically.
* Using this patch, all frames are registered relatively to each other.
* A mean image is computed by averaging the best frames.
* An alignment point mesh covering the object is constructed automatically. Points, where the image is too dim, or has too little contrast or structure, are discarded.
* For each alignment point, all frames are ranked by their local contrast in a surrounding image patch.
* The best frames up to a given number are selected for stacking. Note that this list can be different for different points.
* For all frames, local shifts are computed at all alignment points.
* Using those shifts, the alignment point patches of all contributing frames are stacked into a single average image patch.
* Finally, all stacked patches are blended into a global image.

So far, the code is executed by running the main program in module "main_program.py". Parameters are set at the beginning of the
main program, and in the module "configuration.py".

The decision on the programming language and GUI toolkit for the production level software has not been made yet. The goal is to use the compute resources (CPU, RAM, perhaps graphics card) as efficiently as possible, to produce well-maintainable code, and to provide the user with a state-of-the-art interface.
