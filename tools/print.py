#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 6  2020

@author: Kimberly McCormack, NGA


various printing tools

"""

import time

import numpy as np

def print_time(step, start, sep=False, CPU=False):
    """Compute time splits"""
    total_time = time.time() - start
    total_min = np.int(total_time//60)
    total_hours = np.int(total_min//60)
    left_min = np.int(total_min % 60)
    total_sec = np.int(total_time % 60)

    if sep:
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
  
    print("{0} -- ({3:02d}:{1:02d}:{2:02d})".format(step,
              left_min, total_sec, total_hours))
    if sep:
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        
def log_time(step, start, sep=False):
    """Compute time splits"""
    total_time = time.time() - start
    total_min = np.int(total_time//60)
    total_hours = np.int(total_min//60)
    left_min = np.int(total_min % 60)
    total_sec = np.int(total_time % 60)
    

    print_str = "{0} -- ({3:02d}:{1:02d}:{2:02d})".format(step,
                                      left_min, total_sec, total_hours)

    if sep:
        log_str = "\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n{0}\n\
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n".format(print_str)
        
    else:
        log_str = "\n {0}".format(print_str)
    
    return log_str
