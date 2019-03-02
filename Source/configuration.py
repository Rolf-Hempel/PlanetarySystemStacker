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
        self.version = "Planetary System Stacker 0.5.0"
        self.frames_mono_channel = 'panchromatic'
        self.frames_gauss_width = 7

        self.rank_frames_pixel_stride = 1
        self.rank_frames_method = "Laplace"

        self.align_frames_mode = "Surface"
        self.align_frames_automation = True
        self.align_frames_method = "SteepestDescent"
        self.align_frames_rectangle_scale_factor = 3
        self.align_frames_rectangle_black_threshold = 40
        self.align_frames_search_width = 20
        self.align_frames_border_width = 10
        self.align_frames_sampling_stride = 2
        self.align_frames_average_frame_percent = 5.

        self.alignment_points_half_box_width = 20
        self.alignment_points_min_half_box_width = 10
        self.alignment_points_search_width = 10
        self.alignment_points_structure_threshold = 0.05
        self.alignment_points_brightness_threshold = 10
        self.alignment_points_contrast_threshold = 0
        self.alignment_points_dim_fraction_threshold = 0.6
        self.alignment_points_adjust_edge_patches = True
        self.alignment_points_frame_percent = 10.
        self.alignment_points_rank_method = "Laplace"
        self.alignment_points_rank_pixel_stride = 2
        self.alignment_points_de_warp = True
        self.alignment_points_method = 'SteepestDescent'
        self.alignment_points_sampling_stride = 2
        self.alignment_points_local_search_subpixel = False

        self.stack_frames_background_fraction = 0.3
        self.stack_frames_background_patch_size = 100
        self.stack_frames_gauss_width = 5

        # Derived configuration parameters:
        # Set the alignment patch size to 1.5 times the box size. Between the patch and box
        # borders there must be at least as many pixels as the alignment search width. This way,
        # alignment boxes close to the border never leave the frame.
        self.alignment_points_half_patch_width = max(int(
            round((self.alignment_points_half_box_width * 3) / 2)),
            self.alignment_points_half_box_width + self.alignment_points_search_width)
        # Set the AP distance per coordinate direction such that adjacent patches overlap by 1/6
        # of their width.
        self.alignment_points_step_size = int(
            round((self.alignment_points_half_patch_width * 5) / 3))
