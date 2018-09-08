class Configuration(object):
    def __init__(self):
        self.mono_channel = 'green'
        self.frame_score_pixel_stride = 2
        self.alignment_rectangle_scale_factor = 5
        self.average_frame_percent = 20.
        self.alignment_box_step_size = 100
        self.alignment_box_size = 60
        self.alignment_point_structure_threshold = 0.1
        self.alignment_point_brightness_threshold = 10
        self.alignment_point_contrast_threshold = 15
        self.alignment_point_method = 'LocalSearch'
        self.alignment_point_search_width = 20
        self.quality_area_number_y = 10
        self.quality_area_number_x = 12
        self.quality_area_pixel_stride = 2
        self.quality_area_frame_percent = 10.