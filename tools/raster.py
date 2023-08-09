#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 3 2019

@author: Kimberly McCormack

Last edited on: 07/29/2020


Functions to convert to and from numpy arrays and tif format, and resample
tif files to a different resolution.

Built as part of the pydroDEM hydroconditioning software

"""

import os
from osgeo import gdal, gdal_array
import numpy as np


def DEMdifference(orig, new):
    """
    Difference two input rasters

    Parameters
    ----------
    orig : path
        original DEM file
    new : path
        processed DEM file
    outfile : str
        name of difference file to be created
    nodata : float
        nodata value to be passed to npy2tif

    Returns
    -------
    array
        2D numpy array for difference (new - orig)

    Function assumes that the original and new tif files are the same
        1) dimension
        2) location
        3) projection

    If they are not the same dimension, the difference operation will fail.
    If both tifs have the same dimension but are offset or in different
    projections, the function will succeed but the resulting difference map
    will be wrong.
    """

    orig_array = gdal_array.LoadFile(orig)
    new_array = gdal_array.LoadFile(new)

    diff_array = new_array - orig_array

    return diff_array
