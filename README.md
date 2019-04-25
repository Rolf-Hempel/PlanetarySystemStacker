# PlanetarySystemStacker
Produce a sharp image of a planetary system object (moon, sun, planets) from many seeing-affected frames using the "lucky imaging" technique.

The program is mainly targeted at extended objects (moon, sun), but it should work as well for the (easier) planet case. A full working prototype written in Python has been finished. Results obtained in first tests show the same image quality as the established software AutoStakkert!3.

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

Program execution is most efficient if the image data and all intermediate results can be keept in memory. This, however, requires much RAM space. Therefore, the level of buffering can be selected in the configuration dialog, ranging from 0 (no buffering) to 4 (maximum buffering).

The program is started by executing the main program in module "planetary_system_stacker.py". As an alternative (for debugging),
the program can be started without a GUI with the main program in module "main_program.py".

The program uses array operations (numpy) wherever possible to speed up execution. Qt5 is used as the GUI toolkit.