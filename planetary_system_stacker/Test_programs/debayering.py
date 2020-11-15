import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cv2 import cvtColor, COLOR_RGB2GRAY, COLOR_BGR2RGB, imread, IMREAD_UNCHANGED
from numpy import ndarray
from numpy import uint8, uint16

from frames import debayer_frame


def apply_bayer(frame_in, bayer_pattern):
    """
    Convert a color (3D) frame into a 2D one by applying a Bayer pattern.

    :param frame_in: Color (three channel) image, either 8 or 16 bit unsigned int.
    :param bayer_pattern: Pattern used to convert the input image into the output image. One out
                          of 'Force Bayer RGGB', 'Force Bayer GRBG', 'Force Bayer GBRG',
                          'Force Bayer BGGR'
    :return: B/W (one channel) image with the same type and pixel dimensions as frame_in.
    """

    # Create the data structure of the output frame (one channel).
    pixel_y, pixel_x, dim = frame_in.shape
    frame_out = ndarray((pixel_y, pixel_x), dtype=frame_in.dtype)

    if dim != 3:
        raise Exception("Error: Input frame expected to be 3D (color)")

    # Apply the Bayer pattern. Just take the appropriate color channel at those places where
    # the pattern is set to a given color. There is no interpolation.
    if bayer_pattern == 'Force Bayer RGGB':
        channels = (0, 1, 1, 2)
    elif bayer_pattern == 'Force Bayer GRBG':
        channels = (1, 0, 2, 1)
    elif bayer_pattern == 'Force Bayer GBRG':
        channels = (1, 2, 0, 1)
    elif bayer_pattern == 'Force Bayer BGGR':
        channels = (2, 1, 1, 0)
    else:
        raise Exception("Invalid Bayer transformation: " + str(bayer_pattern))

    frame_out[0:pixel_y:2, 0:pixel_x:2] = frame_in[0:pixel_y:2, 0:pixel_x:2, channels[0]]
    frame_out[0:pixel_y:2, 1:pixel_x:2] = frame_in[0:pixel_y:2, 1:pixel_x:2, channels[1]]
    frame_out[1:pixel_y:2, 0:pixel_x:2] = frame_in[1:pixel_y:2, 0:pixel_x:2, channels[2]]
    frame_out[1:pixel_y:2, 1:pixel_x:2] = frame_in[1:pixel_y:2, 1:pixel_x:2, channels[3]]

    return frame_out


def show_image(frame, comment):
    """
    Use Matplotlib to show a given frame.

    :param frame: Color or B/W image.
    :param comment: Text string (to be displayed as window title)
    :return: -
    """
    print(comment + ", shape: " + str(frame.shape) + ", type: " + str(frame.dtype))

    # If image is 16bit, reduce it to 8bit for display.
    if frame.dtype == uint16:
        frame_shown = (frame >> 8).astype(uint8)
    else:
        frame_shown = frame

    plt.title(comment)
    if len(frame_shown.shape) == 3:
        plt.imshow(frame_shown)
    else:
        plt.imshow(frame_shown, cmap='gray')
    plt.show()


filename = 'Photodisc.png'
input_image = imread(filename, IMREAD_UNCHANGED)

DEPTH = 8

if input_image is None:
    raise IOError("Cannot read image file. Possible cause: Path contains non-ascii characters")

if DEPTH == 16:
    input_image = input_image.astype(uint16) << 8

# If color image, convert to RGB mode.
if len(input_image.shape) == 3:
    image_color = cvtColor(input_image, COLOR_BGR2RGB)
    image_grayscale = cvtColor(image_color, COLOR_RGB2GRAY)
else:
    image_grayscale = input_image
    pixel_y = input_image.shape[0]
    pixel_x = input_image.shape[1]
    image_color = ndarray((pixel_y, pixel_x, 3))
    image_color[:, :, 0] = input_image
    image_color[:, :, 1] = input_image
    image_color[:, :, 2] = input_image

show_image(image_color, "Color image read")

# Test Bayer encoding RGGB.
image_bayer_encoded = apply_bayer(image_color, 'Force Bayer RGGB')

show_image(image_bayer_encoded, "Image Bayer encoded RGGB")
image_bayer_decoded = debayer_frame(image_bayer_encoded, 'Force Bayer RGGB')
show_image(image_bayer_decoded, "Image Bayer decoded RGGB")

# Test Bayer encoding GRBG.
image_bayer_encoded = apply_bayer(image_color, 'Force Bayer GRBG')
image_bayer_decoded = debayer_frame(image_bayer_encoded, 'Force Bayer GRBG')
show_image(image_bayer_decoded, "Image Bayer decoded GRBG")

# Test Bayer encoding GBRG.
image_bayer_encoded = apply_bayer(image_color, 'Force Bayer GBRG')
image_bayer_decoded = debayer_frame(image_bayer_encoded, 'Force Bayer GBRG')
show_image(image_bayer_decoded, "Image Bayer decoded GBRG")

# Test Bayer encoding BGGR.
image_bayer_encoded = apply_bayer(image_color, 'Force Bayer BGGR')
image_bayer_decoded = debayer_frame(image_bayer_encoded, 'Force Bayer BGGR')
show_image(image_bayer_decoded, "Image Bayer decoded BGGR")

# Test "Force Grayscale" with color input.
image_bayer_decoded = debayer_frame(image_color, 'Grayscale')
show_image(image_bayer_decoded, "Image Grayscale")

# Test conversion to BGR with color input.
image_bgr = debayer_frame(image_color, 'BGR')
show_image(image_bgr, "Image BGR")
