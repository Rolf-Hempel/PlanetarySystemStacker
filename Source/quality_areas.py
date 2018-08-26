from numpy import arange

from miscellaneous import local_contrast, circle_around


class QualityAreas(object):
    def __init__(self, configuration, frames, rank_frames, align_frames, alignment_points):
        self.configuration = configuration
        self.frames = frames
        self.align_frames = align_frames
        self.alignment_points = alignment_points

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
            for (dx, dy) in circle:
                if 0 <= index_x + dx < self.x_dim and 0 <= index_y + dy < self.y_dim:
                    if self.quality_areas[index_y + dy][index_x + dx]['alignment_point_indices']:
                        return  self.quality_areas[index_y + dy][index_x + dx]['best_frame_indices']
        return []

    def truncate_best_frames(self, max_frames):
        common_length = min(max_frames, min(len([self.quality_areas[j][i]['best_frame_indices'] for j in arange(self.y_dim)] for i in arange(self.x_dim)]))
        for index_y, quality_area_row in enumerate(self.quality_areas):
            for index_x, quality_area in enumerate(quality_area_row):
                quality_area['best_frame_indices'] = quality_area['best_frame_indices'][:common_length]