import glob

import matplotlib.pyplot as plt
from numpy import unravel_index, argmax, empty, mean
from numpy.fft import fft2, ifft2

from configuration import Configuration
from exceptions import WrongOrderingError
from frames import Frames
from miscellaneous import quality_measure
from rank_frames import RankFrames


class AlignFrames(object):
    def __init__(self, frames, rank_frames, configuration):
        self.frames_mono = frames.frames_mono
        self.number = frames.number
        self.shape = frames.shape
        self.frame_shifts = None
        self.intersection_shape = None
        self.mean_frame = None
        self.configuration = configuration
        self.quality_sorted_indices = rank_frames.quality_sorted_indices
        self.frame_ranks_max_index = rank_frames.frame_ranks_max_index
        self.x_low_opt = self.x_high_opt = self.y_low_opt = self.y_high_opt = None

    def select_alignment_rect(self, scale_factor):
        dim_y, dim_x = self.shape[0:2]
        rect_y = int(self.shape[0] / scale_factor)
        incr_y = int(rect_y)
        rect_x = int(self.shape[1] / scale_factor)
        incr_x = int(rect_x)
        x_low = 0
        x_high = x_low + rect_x
        quality = -1.
        while x_high <= dim_x:
            y_low = 0
            y_high = y_low + rect_y
            while y_high <= dim_y:
                new_quality = quality_measure(
                    self.frames_mono[self.frame_ranks_max_index][y_low:y_high, x_low:x_high])
                if new_quality > quality:
                    (self.x_low_opt, self.x_high_opt, self.y_low_opt, self.y_high_opt) = (x_low, x_high, y_low, y_high)
                    quality = new_quality
                y_low += incr_y
                y_high += incr_y
            x_low += incr_x
            x_high += incr_x
        return (self.x_low_opt, self.x_high_opt, self.y_low_opt, self.y_high_opt)

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
        # The shift value means that frame_1 must be shifted by this amount to register with frame_0.
        return [ty, tx]

    def align_frames(self):
        if self.x_low_opt == None:
            raise WrongOrderingError("Method 'align_frames' is called before 'select_alignment_rect'")
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
        self.intersection_shape = [
            [max(b[0] for b in self.frame_shifts), min(b[0] for b in self.frame_shifts) + self.shape[0]],
            [max(b[1] for b in self.frame_shifts), min(b[1] for b in self.frame_shifts) + self.shape[1]]]

    def average_frame(self, frames, shifts):
        if self.intersection_shape == None:
            raise WrongOrderingError("Method 'average_frames' is called before 'align_frames'")
        number_frames = len(frames)
        buffer = empty([number_frames, self.intersection_shape[0][1] - self.intersection_shape[0][0],
                        self.intersection_shape[1][1] - self.intersection_shape[1][0]])
        for index, frame in enumerate(frames):
            buffer[index, :, :] = frame[
                                  self.intersection_shape[0][0] - shifts[index][0]:self.intersection_shape[0][1] -
                                                                                shifts[index][0],
                                  self.intersection_shape[1][0] - shifts[index][1]:self.intersection_shape[1][1] -
                                                                                shifts[index][1]]
        self.mean_frame = mean(buffer, axis=0)
        return self.mean_frame


if __name__ == "__main__":
    names = glob.glob('Images/2012*.tif')
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
    frame = align_frames.frames_mono[align_frames.frame_ranks_max_index].copy()
    frame[y_low_opt, x_low_opt:x_high_opt] = frame[y_high_opt - 1, x_low_opt:x_high_opt] = 255
    frame[y_low_opt:y_high_opt, x_low_opt] = frame[y_low_opt:y_high_opt, x_high_opt - 1] = 255
    plt.imshow(frame, cmap='Greys_r')
    plt.show()

    align_frames.align_frames()
    print("Frame shifts: " + str(align_frames.frame_shifts))
    print("Intersection: " + str(align_frames.intersection_shape))

    average = align_frames.average_frame(align_frames.frames_mono, align_frames.frame_shifts)
    plt.imshow(average, cmap='Greys_r')
    plt.show()
