# PlanetarySystemStacker (PSS)
_Produce a sharp image of a planetary system object (moon, sun, planets) from many seeing-affected frames using the "lucky imaging" technique._

The program is mainly targeted at extended objects (moon, sun), but it works as well for planets. Results obtained in many tests show at least the same image quality as with the established software AutoStakkert!3.

The software is written in Python 3. The program uses array operations (OpenCV, numpy) wherever possible to speed up execution. A modern graphical user interface (implemented using the QT5 toolkit) and good usability were high priorities in designing the software. PSS is platform-independent and can be used where Python 3 is available. The software has been tested successfully on Windows, various Linux distributions, and macOS. Starting with version 0.8.0, PSS can be used either in GUI mode or from the command line, e.g. as part of a large automatic workflow.

Input to the program can be either video files or directories containing still images. The following algorithmic steps are performed:

* First, all frames are ranked by their overall image quality.
* On the best frame, a rectangular patch with the most pronounced structure in x and y is identified automatically. (Alternatively, the user can select the patch manually as well.)
* Using this patch, all frames are aligned globally with each other.
* A mean image is computed by averaging the best frames.
* An alignment point mesh covering the object is constructed automatically. Points, where the image is too dim, or has too little contrast or structure, are discarded. The user can modify the alignment points, or set them all by hand as well.
* For each alignment point, all frames are ranked by their local contrast in a surrounding image patch.
* The best frames up to a given number are selected for stacking. Note that this list can be different for different points.
* For all frames, local shifts are computed at all alignment points.
* Using those shifts, the alignment point patches of all contributing frames are stacked into a single average image patch.
* Finally, all stacked patches are blended into a global image, using the background image in places without alignment points.
* After stacking is completed, the stacked image can be postprocessed (sharpened) either in a final step of the stacking workflow, or in a separate postprocessing job.

Program execution is most efficient if the image data and all intermediate results can be kept in memory. This, however, requires much RAM space. Therefore, the level of buffering can be selected in the configuration dialog, ranging from 0 (no buffering) to 4 (maximum buffering).

There are three ways to install the program:
* For Windows users there is an self-contained [installer](https://github.com/Rolf-Hempel/PlanetarySystemStacker/releases). It installs the complete software and creates a starter on the user's desktop.
* The user installs [Python 3](https://www.python.org/downloads/) (including pip3), and then calls pip3 to install PSS with all its dependencies automatically. This is the preferred way to install PSS in a platform-independent way. Details can be found in the [User Guide](https://github.com/Rolf-Hempel/PlanetarySystemStacker/blob/master/Documentation/PlanetarySystemStacker_User-Guide.pdf).
* Experienced users can install the Python 3 environment and all packets needed by PSS manually, and then clone the PSS source code from Github. Again, the details are found in the Appendix of the User Guide document.

The way to start the program depends on how it was installed: 
* If the Windows installer was used, the program is started via the desktop icon.
* If PSS was installed with PIP, it can be started from the command line by entering "PlanetarySystemStacker".
* The program can be started in the Python 3 interpreter by executing the main program in the module "planetary_system_stacker.py" .

A [discussion platform](https://www.astronomie.de/PSS/GermanBoard/) for all issues concerning this software project has been created in the context of the German amateur astronomy forum [Astronomie.de](https://www.astronomie.de/). Currently, this forum is in German language only, but an English branch is in preparation. Additionally, an extensive discussion on the subject can be found on the [Cloudy Nights forum](https://www.cloudynights.com/topic/645890-new-stacking-software-project-planetarysystemstacker/).
