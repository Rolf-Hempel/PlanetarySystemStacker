# PlanetarySystemStacker
Produce a sharp image of a planetary system object (moon, sun, planets) from many seeing-affected frames using the "lucky imaging" technique.

The program is mainly targeted at extended objects (moon, sun), but it should work as well for the (easier) planet case. A full working prototype written in Python, sill without a graphical user interface, has been finished. Results obtained in first tests show the same image quality as the established software AutoStakkert!3.

Images can be frames of a video file or a list of still images. The following algorithmic steps are performed:

* First, all frames are ranked by their overall image quality.
* On the best frame, a rectangular patch with the most pronounced structure in x and y is identified automatically.
* Using this patch, all frames are registered relative to each other by global translation.
* A mean image is computed by averaging the best frames.
* A alignment point mesh covering the object is constructed automatically. Points, where the image is too dim, or has too little contrast or structure, are discarded.
* For each alignment point, all frames are ranked by their local contrast in a surrounding image patch.
* The best frames up to a given number are selected for stacking. Note that this list is different for different points.
* For all frames, local shifts are computed at all alignment points.
* Using those shifts, the alignment point patches are stacked into a single average image patch.
* Finally, all stacked patches are blended into a global image.

The decision on the programming language and GUI toolkit for the production level software has not been made yet. The goal is to use the compute resources (CPU, RAM, perhaps graphics card) as efficiently as possible, to produce well-maintainable code, and to provide the user with a state-of-the-art interface.
