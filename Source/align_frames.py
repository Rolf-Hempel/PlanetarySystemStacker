import glob

import matplotlib.pyplot as plt
from numpy import sqrt, average, diff, unravel_index, argmax
from numpy.fft import fft2, ifft2

from configuration import Configuration
from exceptions import WrongOrderingError
from frames import Frames
from rank_frames import RankFrames


class AlignFrames(object):
    def __init__(self, frames, rank_frames, configuration):
        self.frames_mono = frames.frames_mono
        self.number = frames.number
        self.shape = frames.shape
        self.configuration = configuration
        self.quality_sorted_indices = rank_frames.quality_sorted_indices
        self.frame_ranks_max_index = rank_frames.frame_ranks_max_index
        self.x_low_opt = self.x_high_opt = self.y_low_opt = self.y_high_opt = None

    def select_alignment_rect(self, scale_factor):
        dim_y, dim_x = self.shape[0:2]
        rect_y = int(self.shape[0] / scale_factor)
        incr_y = int(rect_y / 2)
        rect_x = int(self.shape[1] / scale_factor)
        incr_x = int(rect_x / 2)
        x_low = 0
        x_high = x_low + rect_x
        quality = -1.
        while x_high <= dim_x:
            y_low = 0
            y_high = y_low + rect_y
            while y_high <= dim_y:
                new_quality = self.quality_measure(
                    self.frames_mono[self.frame_ranks_max_index][y_low:y_high, x_low:x_high])
                if new_quality > quality:
                    self.x_low_opt = x_low
                    self.x_high_opt = x_high
                    self.y_low_opt = y_low
                    self.y_high_opt = y_high
                    quality = new_quality
                y_low += incr_y
                y_high += incr_y
            x_low += incr_x
            x_high += incr_x
        return (self.x_low_opt, self.x_high_opt, self.y_low_opt, self.y_high_opt)

    def quality_measure(self, frame):
        dx = diff(frame)[1:, :]  # remove the first row
        dy = diff(frame, axis=0)[:, 1:]  # remove the first column
        sharpness_x = average(sqrt(dx ** 2))
        sharpness_y = average(sqrt(dy ** 2))
        sharpness = min(sharpness_x, sharpness_y)
        return sharpness

    def translation(self, frame_0, frame_1, shape):
        """Return translation vector to register images."""

        f0 = fft2(frame_0)
        f1 = fft2(frame_1)
        ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
        ty, tx = unravel_index(argmax(ir), shape)

        if ty > shape[0] // 2:
            ty -= shape[0]
        if tx > shape[1] // 2:
            tx -= shape[1]
        # Invert the signs of ty, tx. A positive value means that a feature is at a larger coordinate value
        # for frame_1 as compared to frame_0.
        return [-ty, -tx]

    def align_frames(self):
        if self.x_low_opt == None:
            raise WrongOrderingError("Method 'align_frames' is called before 'select_alignment_rect'")
        else:
            self.frame_shifts = []
            self.reference_window = self.frames_mono[self.frame_ranks_max_index][self.y_low_opt:self.y_high_opt,
                                    self.x_low_opt:self.x_high_opt]
            self.reference_window_shape = self.reference_window.shape
            for index, frame in enumerate(self.frames_mono):
                if index == self.frame_ranks_max_index:
                    self.frame_shifts.append([0, 0])
                else:
                    frame_window = self.frames_mono[index][self.y_low_opt:self.y_high_opt,
                                   self.x_low_opt:self.x_high_opt]
                    self.frame_shifts.append(
                        self.translation(self.reference_window, frame_window, self.reference_window_shape))


if __name__ == "__main__":
    names = glob.glob('Images/Test-1*.jpg')
    print(names)
    configuration = Configuration()
    try:
        frames = Frames(names, type='image')
        print("Number of images read: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Exception as e:
        print("Error: " + e.message)
        exit()

    rank_frames = RankFrames(frames, configuration)
    rank_frames.frame_score()
    align_frames = AlignFrames(frames, rank_frames, configuration)
    (x_low_opt, x_high_opt, y_low_opt, y_high_opt) = align_frames.select_alignment_rect(
        configuration.alignment_rectangle_scale_factor)

    print("optimal alignment rectangle, x_low: " + str(x_low_opt) + ", x_high: " + str(x_high_opt) + ", y_low: " + str(
        y_low_opt) + ", y_high: " + str(y_high_opt))
    frame = align_frames.frames_mono[align_frames.frame_ranks_max_index]
    frame[y_low_opt, x_low_opt:x_high_opt] = frame[y_high_opt-1, x_low_opt:x_high_opt] = 255
    frame[y_low_opt:y_high_opt, x_low_opt] = frame[y_low_opt:y_high_opt, x_high_opt-1] = 255
    plt.imshow(frame, cmap='Greys_r')
    plt.show()

    align_frames.align_frames()
    print ("Frame shifts: " + str(align_frames.frame_shifts))
