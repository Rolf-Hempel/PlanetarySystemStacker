from numpy import arange


class QualityAreas(object):
    def __init__(self, configuration, frames, rank_frames, align_frames, alignment_points):
        self.configuration = configuration
        self.frames = frames
        self.align_frames = align_frames
        mean_frame = self.align_frames.mean_frame
        mean_frame_shape = mean_frame.shape

        self.quality_area_size_y = int(mean_frame_shape[0] / self.configuration.quality_area_number_y)
        self.quality_area_size_x = int(mean_frame_shape[1] / self.configuration.quality_area_number_x)
        self.y_lows = arange(0, mean_frame_shape[0] - self.quality_area_size_y + 1, self.quality_area_size_y)
        self.y_highs = self.y_lows + self.quality_area_size_y
        self.y_highs[-1] = mean_frame_shape[0]
        self.x_lows = arange(0, mean_frame_shape[1] - self.quality_area_size_x + 1, self.quality_area_size_x)
        self.x_highs = self.x_lows + self.quality_area_size_x
        self.x_highs[-1] = mean_frame_shape[1]
        self.quality_areas = []
        for index_y, y_low in enumerate(self.y_lows):
            y_high = self.y_highs[index_y]
            quality_area_row = []
            for index_x, x_low in enumerate(self.x_lows):
                x_high = self.x_highs[index_x]
                quality_area = {}
                quality_area['coordinates'] = [y_low, y_high, x_low, x_high]
                quality_area['alignment_points'] = []
                quality_area_row.append(quality_area)
            self.quality_areas.append(quality_area_row)