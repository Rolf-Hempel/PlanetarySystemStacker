import cv2
from os import remove
from pathlib import Path
from cv2 import resize, INTER_CUBIC


def process_video(input_filename, output_filename, max_frames, frames_stride, reduce_frame_size):
    # Open the input video and read metadata.
    cap = cv2.VideoCapture(input_filename)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    convert_rgb = int(cap.get(cv2.CAP_PROP_CONVERT_RGB))
    print("Input video:\nFrame width: " + str(frame_width) + ", frame height: " + str(frame_height))
    print("Fps: " + str(fps))
    print("Number of frames: " + str(frame_count))
    print("Convert to RGB: " + str(convert_rgb))

    # Define the codec and create VideoWriter object
    if Path(output_filename).is_file():
        remove(output_filename)

    # reduce frame size to multiples of reduce_frame_size.
    frame_width = frame_width - frame_width % reduce_frame_size
    frame_height = frame_height - frame_height % reduce_frame_size
    frame_width_output = int(frame_width / reduce_frame_size)
    frame_height_output = int(frame_height / reduce_frame_size)

    print("Shape of output frames: (" + str(frame_width_output) + ", " + str(
        frame_height_output) + ")")

    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width_output, frame_height_output))

    frame_index_read = 0
    frame_index_written = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not frame_index_read % frames_stride:
            output_frame = resize(frame, (frame_width_output, frame_height_output),
                                  interpolation=INTER_CUBIC)
            out.write(output_frame)
            frame_index_written += 1
            if frame_index_written == max_frames:
                break
        frame_index_read += 1
        if frame_index_read == frame_count:
            break

    cap.release()
    out.release()

    print("Number of frames read: " + str(frame_index_read))
    print("Number of frames written: " + str(frame_index_written))


# input_filename = 'D:\SW-Development\Python\PlanetarySystemStacker\Source\Videos\Moon_Tile-013_205538.avi'
input_filename = 'D:\SW-Development\Python\PlanetarySystemStacker\Examples\Moon_2018-03-24\Moon_Tile-024_043939.avi'
output_filename = 'D:\SW-Development\Python\PlanetarySystemStacker\Examples\Moon_2018-03-24\Moon_Tile-024_043939_resampled.avi'
max_frames = 1000
frames_stride = 20
reduce_frame_size = 4

process_video(input_filename, output_filename, max_frames, frames_stride, reduce_frame_size)
