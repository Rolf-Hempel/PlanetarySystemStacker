class Configuration(object):
    def __init__(self):
        self.mono_channel = 'green'
        self.frame_score_pixel_stride = 2
        self.alignment_rectangle_scale_factor = 3
        self.average_frame_percent = 20.
        self.alignment_box_step_size = 60
        self.alignment_box_size = 60
        self.alignment_point_structure_threshold = 0.25
        self.alignment_point_brightness_threshold = 10