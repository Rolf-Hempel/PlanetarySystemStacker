import glob
from time import time

from numpy import arange, ceil

from align_frames import AlignFrames
from alignment_points import AlignmentPoints
from configuration import Configuration
from frames import Frames
from miscellaneous import local_contrast, circle_around
from rank_frames import RankFrames


class QualityAreas(object):
    def __init__(self, configuration, frames, align_frames, alignment_points):
        self.configuration = configuration
        self.frames = frames
        self.align_frames = align_frames
        self.alignment_points = alignment_points
        self.stack_size = None

        mean_frame = self.align_frames.mean_frame
        mean_frame_shape = mean_frame.shape
        self.quality_area_size_y = int(mean_frame_shape[0] / self.configuration.quality_area_number_y)
        self.quality_area_size_x = int(mean_frame_shape[1] / self.configuration.quality_area_number_x)
        self.y_lows = arange(0, mean_frame_shape[0] - self.quality_area_size_y + 1, self.quality_area_size_y)
        self.y_highs = self.y_lows + self.quality_area_size_y
        self.y_highs[-1] = mean_frame_shape[0]
        self.y_dim = len(self.y_lows)
        self.x_lows = arange(0, mean_frame_shape[1] - self.quality_area_size_x + 1, self.quality_area_size_x)
        self.x_highs = self.x_lows + self.quality_area_size_x
        self.x_highs[-1] = mean_frame_shape[1]
        self.x_dim = len(self.x_lows)
        self.quality_areas = []
        for index_y, y_low in enumerate(self.y_lows):
            y_high = self.y_highs[index_y]
            quality_area_row = []
            for index_x, x_low in enumerate(self.x_lows):
                x_high = self.x_highs[index_x]
                quality_area = {}
                quality_area['coordinates'] = [y_low, y_high, x_low, x_high]
                quality_area['alignment_point_indices'] = []
                quality_area['frame_qualities'] = []
                quality_area_row.append(quality_area)
            self.quality_areas.append(quality_area_row)

        for point_index, [box_index, [j, i, y_center, x_center, y_low, y_high, x_low, x_high]] in enumerate(
                self.alignment_points.alignment_points):
            y_index = min(int(y_center / self.quality_area_size_y), self.y_dim - 1)
            x_index = min(int(x_center / self.quality_area_size_x), self.x_dim - 1)
            self.quality_areas[y_index][x_index]['alignment_point_indices'].append(point_index)

    def select_best_frames(self):
        for frame in self.frames.frames_mono:
            for index_y, quality_area_row in enumerate(self.quality_areas):
                for index_x, quality_area in enumerate(quality_area_row):
                    if quality_area['alignment_point_indices']:
                        [y_low, y_high, x_low, x_high] = quality_area['coordinates']
                        quality_area['frame_qualities'].append(
                            local_contrast(frame[y_low:y_high, x_low:x_high],
                                           self.configuration.quality_area_pixel_stride))
        for index_y, quality_area_row in enumerate(self.quality_areas):
            for index_x, quality_area in enumerate(quality_area_row):
                if quality_area['alignment_point_indices']:
                    quality_area['best_frame_indices'] = [b[0] for b in
                                                          sorted(enumerate(quality_area['frame_qualities']),
                                                                 key=lambda i: i[1], reverse=True)]
        for index_y, quality_area_row in enumerate(self.quality_areas):
            for index_x, quality_area in enumerate(quality_area_row):
                if not quality_area['alignment_point_indices']:
                    quality_area['best_frame_indices'] = self.best_frame_indices_in_empty_areas(index_y, index_x)

    def best_frame_indices_in_empty_areas(self, index_y, index_x):
        for distance in arange(1, max(self.y_dim, self.x_dim)):
            circle = circle_around(index_x, index_y, distance)
            for (compare_x, compare_y) in circle:
                if 0 <= compare_x < self.x_dim and 0 <= compare_y < self.y_dim:
                    if self.quality_areas[compare_y][compare_x]['alignment_point_indices']:
                        return self.quality_areas[compare_y][compare_x]['best_frame_indices']
        return []

    def truncate_best_frames(self):
        max_frames = max(int(ceil(self.frames.number * self.configuration.quality_area_frame_percent / 100.)), 1)
        self.stack_size = min(max_frames, min(
            [min([len(self.quality_areas[j][i]['best_frame_indices']) for i in arange(self.x_dim)]) for j in
             arange(self.y_dim)]))
        for index_y, quality_area_row in enumerate(self.quality_areas):
            for index_x, quality_area in enumerate(quality_area_row):
                quality_area['best_frame_indices'] = quality_area['best_frame_indices'][:self.stack_size]


if __name__ == "__main__":
    names = glob.glob('Images/2012*.tif')
    # names = glob.glob('Images/Moon_Tile-031*ap85_8b.tif')
    # names = glob.glob('Images/Example-3*.jpg')
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
    print('Elapsed time in ranking images: {}'.format(end - start))
    print("Index of maximum: " + str(rank_frames.frame_ranks_max_index))
    print("Frame scores: " + str(rank_frames.frame_ranks))
    print("Frame scores (sorted): " + str([rank_frames.frame_ranks[i] for i in rank_frames.quality_sorted_indices]))
    print("Sorted index list: " + str(rank_frames.quality_sorted_indices))

    align_frames = AlignFrames(frames, rank_frames, configuration)
    start = time()
    (x_low_opt, x_high_opt, y_low_opt, y_high_opt) = align_frames.select_alignment_rect(
        configuration.alignment_rectangle_scale_factor)
    end = time()
    print('Elapsed time in computing optimal alignment rectangle: {}'.format(end - start))
    print("optimal alignment rectangle, x_low: " + str(x_low_opt) + ", x_high: " + str(x_high_opt) + ", y_low: " + str(
        y_low_opt) + ", y_high: " + str(y_high_opt))
    reference_frame_with_alignment_points = align_frames.frames_mono[align_frames.frame_ranks_max_index].copy()
    reference_frame_with_alignment_points[y_low_opt, x_low_opt:x_high_opt] = reference_frame_with_alignment_points[
                                                                             y_high_opt - 1, x_low_opt:x_high_opt] = 255
    reference_frame_with_alignment_points[y_low_opt:y_high_opt, x_low_opt] = reference_frame_with_alignment_points[
                                                                             y_low_opt:y_high_opt, x_high_opt - 1] = 255
    # plt.imshow(reference_frame_with_alignment_points, cmap='Greys_r')
    # plt.show()

    start = time()
    align_frames.align_frames()
    end = time()
    print('Elapsed time in aligning all frames: {}'.format(end - start))
    print("Frame shifts: " + str(align_frames.frame_shifts))
    print("Intersection: " + str(align_frames.intersection_shape))

    start = time()
    alignment_points = AlignmentPoints(configuration, frames, rank_frames, align_frames)
    end = time()
    print('Elapsed time in computing average frame: {}'.format(end - start))
    print("Average frame computed from the best " + str(alignment_points.average_frame_number) + " frames.")
    # plt.imshow(align_frames.mean_frame, cmap='Greys_r')
    # plt.show()

    step_size = configuration.alignment_box_step_size
    box_size = configuration.alignment_box_size
    start = time()
    alignment_points.create_alignment_boxes(step_size, box_size)
    end = time()
    print('Elapsed time in alignment box creation: {}'.format(end - start))
    print("Number of alignment boxes created: " + str(len(alignment_points.alignment_boxes)))

    structure_threshold = configuration.alignment_point_structure_threshold
    brightness_threshold = configuration.alignment_point_brightness_threshold
    contrast_threshold = configuration.alignment_point_contrast_threshold
    print("Selection of alignment points, structure threshold: " + str(
        structure_threshold) + ", brightness threshold: " + str(brightness_threshold) + ", contrast threshold: " + str(
        contrast_threshold))
    start = time()
    alignment_points.select_alignment_points(structure_threshold, brightness_threshold, contrast_threshold)
    end = time()
    print('Elapsed time in alignment point selection: {}'.format(end - start))
    print("Number of alignment points selected: " + str(len(alignment_points.alignment_points)))

    for frame_index in range(frames.number):
        frame_with_shifts = reference_frame_with_alignment_points.copy()
        start = time()
        point_shifts, errors, diffphases = alignment_points.compute_alignment_point_shifts(frame_index)
        end = time()
        print("Elapsed time in computing point shifts for frame number " + str(frame_index) + ": " + str(end - start))

    start = time()
    quality_areas = QualityAreas(configuration, frames, align_frames, alignment_points)
    quality_areas.select_best_frames()
    quality_areas.truncate_best_frames()
    end = time()
    print('Elapsed time in quality area creation and frame ranking: {}'.format(end - start))
    print("Number of frames to be stacked for each quality area: " + str(quality_areas.stack_size))