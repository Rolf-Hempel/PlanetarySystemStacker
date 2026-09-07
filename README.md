# PlanetarySystemStacker (PSS)
_Produce a sharp image of a planetary system object (moon, sun, planets) from many seeing-affected frames using the "lucky imaging" technique._

> This is a maintained fork of [Rolf-Hempel/PlanetarySystemStacker](https://github.com/Rolf-Hempel/PlanetarySystemStacker). Upstream has been unmaintained since 2023; this fork lands the outstanding pull requests (PyQt6 migration, Python 3.9–3.12 support) and continues bug fixes.

The program is mainly targeted at extended objects (moon, sun), but it works as well for planets. Results obtained in many tests show at least the same image quality as with the established software AutoStakkert!3.

The software is written in Python 3. The program uses array operations (OpenCV, numpy) wherever possible to speed up execution. A modern graphical user interface (implemented using the Qt6 toolkit) and good usability were high priorities in designing the software. PSS is platform-independent and can be used where Python 3 is available. The software has been tested successfully on Windows, various Linux distributions, and macOS. Starting with version 0.8.0, PSS can be used either in GUI mode or from the command line, e.g. as part of a large automatic workflow.

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

## Installation

Requires Python 3.9 or newer. This fork does not (yet) ship prebuilt binaries or a PyPI release — install from source.

```bash
git clone https://github.com/timing/PlanetarySystemStacker.git
cd PlanetarySystemStacker

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

**macOS note:** if the OpenCV or scikit-image wheels aren't available for your Python version and pip falls back to building from source, install CMake first:
```bash
brew install cmake
```

## Running

With the venv activated, either invoke the console script or run the module directly:
```bash
PlanetarySystemStacker
# or
python planetary_system_stacker/planetary_system_stacker.py
```

For CLI (headless) use, pass a config file:
```bash
PlanetarySystemStacker --config_file path/to/config.pss
```

## Upstream binaries (may be outdated)

Rolf-Hempel's original repo ships a Windows installer and a PyPI package (`planetary-system-stacker`). Both predate the PyQt6 migration and Python 3.12 support, and installing them on modern systems is where most of the [upstream install issues](https://github.com/Rolf-Hempel/PlanetarySystemStacker/issues) come from. Use them at your own risk; the source install above is the maintained path.

* Original User Guide (still mostly accurate for algorithm/UI): [PlanetarySystemStacker_User-Guide.pdf](https://github.com/Rolf-Hempel/PlanetarySystemStacker/blob/master/Documentation/PlanetarySystemStacker_User-Guide.pdf)

A [discussion platform](https://www.astronomie.de/PSS/GermanBoard/) for all issues concerning this software project has been created in the context of the German amateur astronomy forum [Astronomie.de](https://www.astronomie.de/). Currently, this forum is in German language only, but an English branch is in preparation. Additionally, an extensive discussion on the subject can be found on the [Cloudy Nights forum](https://www.cloudynights.com/topic/645890-new-stacking-software-project-planetarysystemstacker/).
