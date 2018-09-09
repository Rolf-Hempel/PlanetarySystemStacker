# -*- coding: utf-8; -*-
"""
Copyright (c) 2018 Rolf Hempel, rolf6419@gmx.de

This file is part of the PlanetarySystemStacker tool (PSS).
https://github.com/Rolf-Hempel/PlanetarySystemStacker

PSS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PSS.  If not, see <http://www.gnu.org/licenses/>.

"""

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