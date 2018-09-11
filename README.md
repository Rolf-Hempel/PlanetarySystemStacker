# PlanetarySystemStacker
Produce a sharp image of a planetary system object (moon, sun, planets) from many seeing-affected frames according to the "lucky imaging" technique.

The program is mainly targeted at extended objects (moon, sun), but it should work as well for the (easier) planet case. It is still in an initial state of development.

Images can be frames of a video file or a list of still images. The following algorithmic steps will be performed:

* First, all frames are ranked by their overall image quality.
* On the best frame, a rectangular patch with the most pronounced structure in x and y is identified.
* Using this patch, all frames are registered relative to each other by global translation.
* A mean image is computed by averaging the best frames.
* A rectangular grid of alignment points is constructed. Points, where the image has too little structure, are discarded.
* A rectangular grid of quality areas which cover the entire image is constructed.
* For each quality area, all frames are ranked by their local contrast.
* The best frames up to a given number are selected for stacking. Note that this list is different for different quality areas.
* For all frames, local shifts are computed at all alignment points contained in quality patches to be used for stacking.
* Local shifts are interpolated between alignment points.
* Using those shifts, the quality area patches are "de-warped" prior to stacking.
* Finally, all de-warped quality area patches are stacked.

It is intended to add a graphical user interface once the computational part is finished.
