import matplotlib.pyplot as plt
from cv2 import VideoCapture, VideoWriter_fourcc, VideoWriter
from numpy import full, zeros, uint8, clip


def create_dark(dark):
    dark[700:800, 700:800, :] += 15
    return dark


def create_flat(image):
    type = image.dtype
    image[100:200, 100:200, :] = (image[100:200, 100:200, :] * 0.9).astype(type)
    return image


original_video = 'Videos/short_video.avi'
processed_video = 'D:/SW-Development/Python/PlanetarySystemStacker/Examples/Darks_and_Flats/short_video_dark-flat-processed.avi'
video_dark = "D:/SW-Development/Python/PlanetarySystemStacker/Examples/Darks_and_Flats/artificial_dark.avi"
video_flat = "D:/SW-Development/Python/PlanetarySystemStacker/Examples/Darks_and_Flats/artificial_flat.avi"
ny = 960
nx = 1280

dark = create_dark(zeros((ny, nx, 3), uint8))
flat = (create_flat(full((ny, nx, 3), 128)) + dark).astype(uint8)

plt.imshow(dark)
plt.show()
plt.imshow(flat)
plt.show()

# Create the VideoCapture object.
cap = VideoCapture(original_video)
# Define the codec and create VideoWriter object
fourcc = VideoWriter_fourcc(*'XVID')
out = VideoWriter(processed_video, fourcc, 20.0, (nx, ny))
out_dark = VideoWriter(video_dark, fourcc, 20.0, (nx, ny))
out_flat = VideoWriter(video_flat, fourcc, 20.0, (nx, ny))

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret is True:
        frame_flat_applied = create_flat(frame)
        frame_dark_added = frame_flat_applied + dark
        frame_processed = clip(frame_dark_added, 0, 255)
        out.write(frame_processed)
        out_dark.write(dark)
        out_flat.write(flat)
    else:
        break

# When everything done, release the capture
cap.release()
out.release()
out_dark.release()
out_flat.release()
