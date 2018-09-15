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

import glob
from time import time

from numpy import arange, ceil, float32, empty
import matplotlib.pyplot as plt

from align_frames import AlignFrames
from alignment_points import AlignmentPoints
from configuration import Configuration
from frames import Frames
from miscellaneous import Miscellaneous
from rank_frames import RankFrames
from exceptions import InternalError


class StackFrames(object):

    def __init__(self, configuration, frames, align_frames, alignment_points, quality_areas):
        self.configuration = configuration
        self.frames = frames
        self.align_frames = align_frames
        self.alignment_points = alignment_points
        self.quality_areas = quality_areas
        self.stack_size = quality_areas.stack_size
        self.alignment_points.ap_mask_initialize()
        if self.frames.color:
            self.stacked_image = empty(
                [self.align_frames.intersection_shape[0][1] -
                 self.align_frames.intersection_shape[0][0],
                 self.align_frames.intersection_shape[1][1] -
                 self.align_frames.intersection_shape[1][0], 3], dtype=float32)
            self.buffer = self.stacked_image.copy()
        else:
            self.stacked_image = empty(
                [self.align_frames.intersection_shape[0][1] -
                 self.align_frames.intersection_shape[0][0],
                 self.align_frames.intersection_shape[1][1] -
                 self.align_frames.intersection_shape[1][0]], dtype=float32)
            self.buffer = self.stacked_image.copy()
        self.pixel_shift_y = self.stacked_image = empty(
                [self.align_frames.intersection_shape[0][1] -
                 self.align_frames.intersection_shape[0][0],
                 self.align_frames.intersection_shape[1][1] -
                 self.align_frames.intersection_shape[1][0]], dtype=float32)
        self.pixel_shift_x = self.pixel_shift_y.copy()

    def stack_frame(self, frame_index):
        self.alignment_points.ap_mask_reset()
        if self.frames.used_quality_areas[frame_index]:
            for [index_y, index_x] in self.frames.used_quality_areas[frame_index]:
                self.alignment_points.ap_mask_set(self.quality_areas.qa_ap_index_y_lows[index_y],
                                                  self.quality_areas.qa_ap_index_y_highs[index_y],
                                                  self.quality_areas.qa_ap_index_x_lows[index_x],
                                                  self.quality_areas.qa_ap_index_x_highs[index_x])
            self.alignment_points.compute_alignment_point_shifts(frame_index, use_ap_mask=True)


