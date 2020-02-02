from cv2 import cvtColor, COLOR_RGB2GRAY, CV_8U, CV_16U, COLOR_BayerBG2RGB, COLOR_BayerGB2RGB, \
    COLOR_BayerGR2RGB, COLOR_BayerRG2RGB, COLOR_GRAY2RGB, COLOR_BGR2RGB, imread, IMREAD_UNCHANGED

from numpy import uint8, uint16

import matplotlib.pyplot as plt

from numpy import ndarray

def frame_decode(frame_in, debayer_pattern='Auto detect color'):
    """
    Process a given input frame "frame_in", either containing one layer (B/W) or three layers
    (color) into an output frame "frame_out" as specified by the parameter "debayer_pattern".

    The rules for this transformation are:
    - If the "debayer_pattern" is "Auto detect color", the input frame is not changed, i.e. the
      output frame is identical to the input frame. The same applies if the input frame is of type
      color, and the "debayer_pattern" is "RGB", or if the input frame is of type "B/W" and the
      "debayer_pattern" is "Grayscale".
    - If the input frame is of type "color" and "debayer_pattern" is "Grayscale", the RGB image is
      converted into a B/W one.
    - If the input frame is of type "Grayscale" and "debayer_pattern" is "RGB", the result is a
      three-channel RGB image where all channels are the same.
    - If a non-standard "debayer_pattern" (i.e. "RGGB", "GRBG", "GBRG", "BGGR") is specified and the
      input is a B/W image, decode the image using the given Bayer pattern. If the input image is
      of type three-channel RGB, first convert it into grayscale and then decode the image as in the
      B/W case.

    :param frame_in: Input image, either 2D (grayscale) or 3D (color). The type is either 8 or 16
                     bit unsigned int.
    :param debayer_pattern: Pattern used to convert the input image into the output image. One out
                            of 'Grayscale', 'RGB', 'Force Bayer RGGB', 'Force Bayer GRBG',
                            'Force Bayer GBRG', 'Force Bayer BGGR'
    :return: (frame_out, color_out) with frame_out: output image (see above)
                                         color_out: True, if three-channel RGB. False otherwise.
    """

    debayer_codes = {
        'Force Bayer RGGB': COLOR_BayerBG2RGB,
        'Force Bayer GRBG': COLOR_BayerGB2RGB,
        'Force Bayer GBRG': COLOR_BayerGR2RGB,
        'Force Bayer BGGR': COLOR_BayerRG2RGB
    }

    type_in = frame_in.dtype

    if type_in != uint8 and type_in != uint16 and type_in != CV_8U and type_in != CV_16U:
        raise Exception("Image type " + str(type_in) + " not supported")

    # If the input frame is 3D, it represents a color image.
    color_in = len(frame_in.shape) == 3

    # Case color input image.
    if color_in:
        # Three-channel input, interpret as RGB color and leave it unchanged.
        if debayer_pattern in ['Auto detect color', 'RGB']:
            color_out = True
            frame_out = frame_in

        # Three-channel (color) input, reduce to two-channel (B/W) image.
        elif debayer_pattern in ['Grayscale', 'Force Bayer RGGB', 'Force Bayer GRBG',
                                 'Force Bayer GBRG', 'Force Bayer BGGR']:

            frame_2D = cvtColor(frame_in, COLOR_RGB2GRAY)

            # Output is B/W image.
            if debayer_pattern == 'Grayscale':
                color_out = False
                frame_out = frame_2D

            # Decode the B/W image into a color image using a Bayer pattern.
            else:
                color_out = True
                frame_out = cvtColor(frame_2D, debayer_codes[debayer_pattern])

        # Invalid debayer pattern specified.
        else:
            raise Exception("Debayer pattern " + debayer_pattern + " not supported")

    # Case B/W input image.
    else:
        # Two-channel input, interpret as B/W image and leave it unchanged.
        if debayer_pattern in ['Auto detect color', 'Grayscale']:
            color_out = False
            frame_out = frame_in

        # Transform the one-channel B/W image in an RGB one where all three channels are the same.
        elif debayer_pattern == 'RGB':
            frame_out = cvtColor(frame_in, COLOR_GRAY2RGB)

        # Non-standard Bayer pattern, decode into color image.
        elif debayer_pattern in ['Force Bayer RGGB', 'Force Bayer GRBG',
                                 'Force Bayer GBRG', 'Force Bayer BGGR']:
            color_out = True
            frame_out = cvtColor(frame_in, debayer_codes[debayer_pattern])

        # Invalid Bayer pattern specified.
        else:
            raise Exception("Debayer pattern " + debayer_pattern + " not supported")

    # Return the decoded image and the color flag.
    return frame_out, color_out

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
    print (comment + ", shape: " + str(frame.shape) + ", type: " + str(frame.dtype))
    plt.title(comment)
    plt.imshow(frame)
    plt.show()

filename = 'Photodisc.png'
input_image = imread(filename, IMREAD_UNCHANGED)

if input_image is None:
    raise IOError("Cannot read image file. Possible cause: Path contains non-ascii characters")

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
image_bayer_decoded = frame_decode(image_bayer_encoded, 'Force Bayer RGGB')
show_image(image_bayer_decoded[0], "Image Bayer decoded RGGB")

# Test Bayer encoding GRBG.
image_bayer_encoded = apply_bayer(image_color, 'Force Bayer GRBG')
image_bayer_decoded = frame_decode(image_bayer_encoded, 'Force Bayer GRBG')
show_image(image_bayer_decoded[0], "Image Bayer decoded GRBG")

# Test Bayer encoding GBRG.
image_bayer_encoded = apply_bayer(image_color, 'Force Bayer GBRG')
image_bayer_decoded = frame_decode(image_bayer_encoded, 'Force Bayer GBRG')
show_image(image_bayer_decoded[0], "Image Bayer decoded GBRG")

# Test Bayer encoding BGGR.
image_bayer_encoded = apply_bayer(image_color, 'Force Bayer BGGR')
image_bayer_decoded = frame_decode(image_bayer_encoded, 'Force Bayer BGGR')
show_image(image_bayer_decoded[0], "Image Bayer decoded BGGR")

# Test "Force Grayscale" with color input.
image_bayer_decoded = frame_decode(image_color, 'Grayscale')[0]
show_image(image_bayer_decoded, "Image Grayscale")
