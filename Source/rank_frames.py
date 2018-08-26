import glob
from time import time

from configuration import Configuration
from frames import Frames
from miscellaneous import local_contrast, quality_measure_sobel


class RankFrames(object):
    def __init__(self, frames, configuration):
        self.number = frames.number
        self.shape = frames.shape
        self.configuration = configuration
        frames.add_monochrome(self.configuration.mono_channel)
        self.frames_mono = frames.frames_mono
        self.quality_sorted_indices = None
        self.frame_ranks = []
        self.frame_ranks_max_index = None
        self.frame_ranks_max_value = None

    def frame_score(self):
        for frame in self.frames_mono:
            self.frame_ranks.append(local_contrast(frame, self.configuration.frame_score_pixel_stride))
            # self.frame_ranks.append(quality_measure_sobel(frame)) # Ten times slower but twice as good
        self.quality_sorted_indices = [b[0] for b in
                                       sorted(enumerate(self.frame_ranks), key=lambda i: i[1], reverse=True)]
        self.frame_ranks_max_index = self.quality_sorted_indices[0]
        self.frame_ranks_max_value = self.frame_ranks[self.frame_ranks_max_index]
        self.frame_ranks /= self.frame_ranks_max_value


if __name__ == "__main__":
    names = glob.glob('Images/Mond*.jpg')
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
    start = time()
    rank_frames.frame_score()
    end = time()
    for index, frame in enumerate(names):
        print ("Frame rank for no. " + str(index) + ": " + str(rank_frames.frame_ranks[index]))
    print('Elapsed time in computing optimal alignment rectangle: {}'.format(end - start))
