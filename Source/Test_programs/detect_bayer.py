import matplotlib.pyplot as plt
from cv2 import VideoCapture, cvtColor, COLOR_RGB2GRAY
from numpy import uint8, uint16

from frames import detect_bayer


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


filename = 'D:\SW-Development\Python\PlanetarySystemStacker\Examples\AVI_Chris-Garry\\20110929_005012_jupiter_gbrg_DIB.avi'
# Create the VideoCapture object.
cap = VideoCapture(filename)

# Read the first frame.
ret, first_frame_read = cap.read()
# show_image(first_frame_read, "First frame of Video")

debayer_code = detect_bayer(first_frame_read)

print ("Debayer code found: " + debayer_code)