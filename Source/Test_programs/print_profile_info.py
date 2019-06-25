# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 19:54:17 2019

@author: Jens.Scheidtmann@gmail.com

Run 

> python -m cProfiler -o profile_info.txt main_program.py

to create profiling information, then run this file to output the top 10 function calls 
(by cumulative time).

Make sure that you have set 'show_results = False' in the 'Specify Test Case' 
section in main_program.py (else some graphic displays are opened, which
wait for the user to close the window)
"""


import pstats
from pstats import SortKey
p = pstats.Stats('profile_info.txt')
p.sort_stats(SortKey.CUMULATIVE).print_stats(10)

# p.strip_dirs().sort_stats(-1).print_stats()