import glob

from configuration import Configuration
from frames import Frames


class RankFrames(object):
    def __init__(self, frames, configuration):
        self.number = frames.number
        self.shape = frames.shape
        self.configuration = configuration
        frames.add_monochrome(self.configuration.mono_channel)
        self.frames_mono = frames.frames_mono
        self.frame_ranks = []

    def frame_score(self):
        for frame in self.frames_mono:
            self.frame_ranks.append(self.local_contrast(frame, self.configuration.local_contrast_stride))

    def local_contrast(self, frame, stride):
        sum_horizontal = sum(sum(abs(frame[::stride, 2::stride] - frame[::stride, :-2:stride]) * (
                    frame[::stride, 1:-1:stride] > self.configuration.local_contrast_threshold)))
        sum_vertical = sum(sum(abs(frame[2::stride, ::stride] - frame[:-2:stride, ::stride]) * (
                    frame[1:-1:stride, ::stride] > self.configuration.local_contrast_threshold)))
        return min(sum_horizontal, sum_vertical)


if __name__ == "__main__":
    names = glob.glob('Images/2012_*.tif')
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
    print("Frame scores: " + str(rank_frames.frame_ranks))
