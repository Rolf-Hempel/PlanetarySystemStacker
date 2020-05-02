# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 15:39:35 2018

@author: Michal Powalko m.powalko@gmail.com
"""

import time
import numpy as np


# Currently by Rolf
tic = time.time()
buffer_Rolf = np.empty([200, 1920, 1080])
mem_alloc_Rolf = buffer_Rolf.nbytes / 1024**3
print('Preallocated {0:.3f} [GB] memory for importing the data in {1:.3f} [s]...Rolf'.format(mem_alloc_Rolf, time.time() - tic))

# Proposal from Michal
# "buffer" is a matrix to HOLD "input data" - and this consist of 8bit data = "uint8". Specifing the "dtype"
# as "np.uint8" You can save al lot of mememory and gain speed (in case of many frames)
# The results of "mean/median" will be saved under another variable, where the resolution might be adjustet
# The only disadvantageous would be, when working with input files with greater color resolution than 8bit

tic = time.time()
buffer_Michal = np.empty([200, 1920, 1080], dtype = np.uint8)
mem_alloc_Michal = buffer_Michal.nbytes / 1024**3
print('Preallocated {0:.3f} [GB] memory for importing the data in {1:.3f} [s]...Michal :)'.format(mem_alloc_Michal, time.time() - tic))

#------------------------------------------------------------------------------------------------------------------------

# Currently by Rolf
tic = time.time()
mean_frame_Rolf = np.mean(buffer_Rolf, axis=0)
print('Calculated "mean" from imported frames in {0:.3f} [s]...Rolf'.format(time.time() - tic))


# Proposal from Michal
# Taking advantage of "variable.npCommand" might save some time
# Disadvantage may be luck of some internal check in numpy "sanity check of input data", but I have never had a problem with this
buffer_Rolf2 = np.empty([200, 1920, 1080])
tic = time.time()
mean_frame_Michal = buffer_Rolf2.mean(axis=0)
print('Calculated "mean" from imported frames in {0:.3f} [s]...Michal'.format(time.time() - tic))

#------------------------------------------------------------------------------------------------------------------------

# Proposal from Michal
# Calculating Sum as well
# Having read:   http://www.stark-labs.com/craig/resources/Articles-&-Reviews/BitDepthStacking.pdf
# building a sum from 8bits we can gaint the bit resolution. To prevent the resolution it is necessary to either:
# - saving stacked uint8 frames with "float"resolution (to keep the "after comma values)
# - or to sum to uint>8, like uint16 or even uint32
tic = time.time()
sum_frame = np.sum(buffer_Rolf, axis=0, dtype=np.uint32)
print('Calculated "sum" from imported frames in {0:.3f} [s]...'.format(time.time() - tic))

#------------------------------------------------------------------------------------------------------------------------

# Proposal from Michal
# Calculating Median as well

tic = time.time()
median_frame = np.median(buffer_Rolf, axis=0)
print('Calculated "median" from imported frames in {0:.3f} [s]...'.format(time.time() - tic))

# We can boost performance by setting "overwrite_input=True" --> the original matrix will be "reshaped" instead of creating another one
# "variable.median" is not available :(
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html
buffer_Rolf2 = np.empty([200, 1920, 1080])
tic = time.time()
median_frame2 = np.median(buffer_Rolf2, axis=0, overwrite_input=True)
print('Calculated "median" from imported frames in {0:.3f} [s]...with "overwrite_input=True"'.format(time.time() - tic))


#------------------------------------------------------------------------------------------------------------------------
## TIPS

# setting "dtype=np.uint32" inside numpy function is much faster then "().astype(np.uint32)"

# "variable.npCommand" is slightly faster than "npCommand(variable)"

# Downsampling to 8bit
# This may be tricky --> ".astype(np.uint)" will perform "floor" instead of "round".
# This might lead to situation, where "9,9999" will be "rounded" to "9"
downsample_ipnut = 10 * np.random.rand(3,2)
downsample_wrong = downsample_ipnut.astype(np.uint8)
downsample_good = downsample_ipnut.round().astype(np.uint8)
