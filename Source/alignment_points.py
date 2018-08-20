import glob
from math import ceil

import matplotlib.pyplot as plt

from align_frames import AlignFrames
from configuration import Configuration
from frames import Frames
from rank_frames import RankFrames


class AlignmentPoints(object):
    def __init__(self, configuration, frames, rank_frames, align_frames):
        self.configuration = configuration
        self.frames = frames
        self.rank_frames = rank_frames
        self.align_frames = align_frames
        self.average_frame_number = max(ceil(frames.number * configuration.average_frame_percent / 100.), 1)
        self.average_frame = self.align_frames.average_frame(
            [self.frames.frames_mono[i] for i in self.rank_frames.quality_sorted_indices[:self.average_frame_number]],
            [self.align_frames.frame_shifts[i] for i in
             self.rank_frames.quality_sorted_indices[:self.average_frame_number]])


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

    alignment_points = AlignmentPoints(configuration, frames, rank_frames, align_frames)
    print ("Average frame computed from the best " + str(alignment_points.average_frame_number) + " frames.")
    plt.imshow(alignment_points.average_frame, cmap='Greys_r')
    plt.show()
