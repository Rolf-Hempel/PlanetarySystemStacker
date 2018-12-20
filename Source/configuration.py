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
        self.average_frame_percent = 5.
        self.alignment_de_warp= True
        self.alignment_sub_pixel = True
        self.alignment_box_step_size = 50
        self.alignment_box_size = 40
        self.alignment_gauss_width = 5
        self.alignment_box_max_neighbor_distance = 1
        self.alignment_point_structure_threshold = 0.05
        self.alignment_point_brightness_threshold = 10
        self.alignment_point_contrast_threshold = 5
        self.alignment_point_method = 'LocalSearch'
        self.alignment_point_search_width = 20
        self.quality_area_number_y = 5
        self.quality_area_number_x = 6
        self.quality_area_pixel_stride = 2
        self.quality_area_frame_percent = 10.
        self.stacking_own_remap_method = True
        self.stacking_rigid_ap_shift = True

        # Parameters used for optical flow:
        self.stacking_use_optical_flow = False

        # Image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical
        # pyramid, where each next layer is half as large as the previous one.
        self.pyramid_scale = 0.5  # between 0.1 and 0.9
        # Number of pyramid layers including the initial image; levels=1 means that no extra layers
        # are created and only the original images are used.
        self.levels = 1  # between 1 and 10
        # Averaging window size; larger values increase the algorithm robustness to image noise and
        # give more chances for fast motion detection, but yield more blurred motion field.
        self.winsize = 15  # between 5 and 40
        # If optical flow is used, the flow field must be computed on a slightly larger area than
        # the quality areas to avoid artifacts on quality area boundaries.
        self.stacking_optical_flow_overlap = 3
        # Number of iterations the algorithm does at each pyramid level.
        self.iterations = 1  # between 1 and 10
        # Size of the pixel neighborhood used to find polynomial expansion in each pixel; larger
        # values mean that the image will be approximated with smoother surfaces, yielding more
        # robust algorithm and a more blurred motion field, typically poly_n =5 or 7.
        self.poly_n = 5  # between 3 and 10
        # Standard deviation of the Gaussian that is used to smooth derivatives used as a basis
        # for the polynomial expansion; for a neighborhood size of 5, you can choose 1.1; for a size
        # of 7, a good value would be 1.5.
        self.poly_sigma = 1.1  # between 1. and 2.
        # Select if the Gaussian winsize * winsize filter should be used instead of a box filter
        # of the same size for optical flow estimation; usually, this option gives a more accurate
        # flow than with a box filter, at the cost of lower speed; normally, winsize for a Gaussian
        # window should be set to a larger value to achieve the same level of robustness.
        self.use_gaussian_filter = True  # either True or False
