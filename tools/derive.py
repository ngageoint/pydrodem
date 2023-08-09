# -*- coding: utf-8 -*-
"""
Created on July 21 2020

@author: Kimberly McCormack

Last edited on: 07/23/2020


Functions to derive various attributes from a DEM, such as
slope, aspect, and curvature

Built as part of the pydroDEM hydroconditioning software

"""

import os
import numpy as np
from scipy import signal


def DEMdifference(orig, new, outfile, nodata):
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

    work_dir = os.path.dirname(new)

    output_path = os.path.join(work_dir, outfile)

    orig_array = gdal_array.LoadFile(orig)
    new_array = gdal_array.LoadFile(new)

    diff_array = new_array - orig_array
    pyct.npy2tif(diff_array, orig, output_path, nodata)

    return diff_array


def apply_convolve_filter(array, kernel, normalize_kernel=False,
                            boundary='symm', mode='same', dtype=np.float32):
    """
    Apply a convolution filter to a 2D array

    Parameters
    ----------
    array : array
        input array to apply convolution to
    kernel : array
        small convolution filter
    boundary : str, optional
        type of boundary to be applied during convolution, by default 'symm'
    mode : str, optional
        convulution mode, by default 'same'
    dtype : numpy datatype, optional
        output datatype, by default np.float32

    Returns
    -------
    array
        2D array - result of convolution
    """

    if normalize_kernel:
        w_sum = np.sum(kernel)
        kernel = (1./w_sum)*kernel
    
    array_smooth = signal.convolve2d(array, kernel,
                                     boundary=boundary, mode=mode)

    array_smooth = array_smooth.astype(dtype)

    return array_smooth


def slope_D2(array, dx, dy, unit='degree'):
    """
    Compute slope magnitude of a 2D array using 2 directions (x and y, no diagonals)

    Parameters
    ----------
    array : array
        input 2D array
    dx : float
        grid size in x-direction
    dy : float
        gris size in y-direction
    unit : str, optional
        slope units, by default 'degree'

    Returns
    -------
    array
        2D array of slope
    """
    """Compute the slope"""
    dZdy, dZdx = np.gradient(array, dy, dx)
    slope = np.sqrt(dZdy**2 + dZdx**2)

    if unit == 'degree':
        """Convert slope to degrees and save as tif"""
        slope = np.rad2deg(np.arctan(slope)).astype(np.int8)

    return slope


def curvature_D2(array, dx, dy, geometric=True):
    """
    Compute the curvature of a 2D array using x and y directions (no diagonals).
    Either laplacian or modified geometric curvature


    Parameters
    ----------
    array : array
        input 2D array
    dx : float
        grid size in x-direction
    dy : float
        gris size in y-direction
    geometric : bool, optional
        Use geometric curvature? if False, Laplacian is used, by default True

    Returns
    -------
    array
        2D array of curvature
    """
    """Compute the slope"""
    dZdy, dZdx = np.gradient(array, dy, dx)

    if geometric:
        slope = np.sqrt(dZdy**2 + dZdx**2)
        dZ_mag_div = slope + 0.1
        dZdy_norm = np.divide(dZdy, dZ_mag_div)
        dZdx_norm = np.divide(dZdx, dZ_mag_div)

        d2Zdy, _dummy = np.gradient(dZdy_norm, dy, dx)
        _dummy, d2Zdx = np.gradient(dZdx_norm, dy, dx)

    else:
        d2Zdy, _dummy = np.gradient(dZdy, dy, dx)
        _dummy, d2Zdx = np.gradient(dZdx, dy, dx)

    curvature = d2Zdy + d2Zdx

    return curvature


def slope_D4(array, dx, dy, unit='degree'):
    """
    Compute slope magnitude as maximum central difference derivative in 4 directions:
    x, y, and 2 diagonals

    Parameters
    ----------
    array : array
        input 2D array
    dx : float
        grid size in x-direction
    dy : float
        gris size in y-direction
    unit : str, optional
        slope units, by default 'degree'

    Returns
    -------
    array
        2D array of slope
    """

    dZ_d4 = gradient_D4(array, dx, dy)
    slope = np.amax(np.abs(dZ_d4), axis=2)

    if unit == 'degree':
        """Convert slope to degrees and save as tif"""
        slope = np.rad2deg(np.arctan(slope)).astype(np.int8)

    return slope


def gradient_D4(array, dx, dy):
    """Compute gradient for 4 directions for a 2D array using central differences

    Parameters
    ----------
    array : array
        input 2D array
    dx : float
        grid size in x-direction
    dy : float
        gris size in y-direction

    Returns
    -------
    array
        stacked 3D array of the gradient : [dZdx, dZdne, dZdy, dZdnw]
    """

    dxy = 1./(2*np.sqrt(dx**2 + dy**2))

    NE_SW_ker, NW_SE_ker = get_diag_kernel(dxy)

    dZdnw = signal.convolve2d(array, NW_SE_ker, boundary='symm', mode='same')
    dZdne = signal.convolve2d(array, NE_SW_ker, boundary='symm', mode='same')

    dZdy, dZdx = np.gradient(array, dy, dx)

    """Set data types"""
    dZdy = dZdy.astype(np.float32)
    dZdx = -1.*dZdx.astype(np.float32)
    dZdne = dZdne.astype(np.float32)
    dZdnw = dZdnw.astype(np.float32)

    dZ_d4 = np.stack((dZdx,
                      dZdne,
                      dZdy,
                      dZdnw), axis=2)

    return dZ_d4


def curvature_D4(array, dx, dy, geometric=False):
    """
    Compute curvature for a 2D array using central difference for all 8 neighbor cells.
    Either laplacian or modified geometric curvature


    Parameters
    ----------
    array : array
        input 2D array
    dx : float
        grid size in x-direction
    dy : float
        gris size in y-direction
    geometric : bool, optional
        Use geometric curvature? if False, Laplacian is used, by default True

    Returns
    -------
    array
        2D array of curvature
    """
    dxy = 1./(2*np.sqrt(dx**2 + dy**2))

    dZ = gradient_D4(array, dx, dy)

    dZdx = dZ[:, :, 0]
    dZdne = dZ[:, :, 1]
    dZdy = dZ[:, :, 2]
    dZdnw = dZ[:, :, 3]

    NE_SW_ker, NW_SE_ker = get_diag_kernel(dxy)

    if geometric:
        """Geometric curvature - normalized by gradient magnitude"""
        mag_XY = np.sqrt(dZdx**2 + dZdy**2) + 0.1
        mag_diag = np.sqrt(dZdne**2 + dZdnw**2) + 0.1

        dZdx = np.divide(dZdx, mag_XY)
        dZdne = np.divide(dZdne, mag_diag)
        dZdy = np.divide(dZdy, mag_XY)
        dZdnw = np.divide(dZdnw, mag_diag)

    d2Zdy, _dummy = np.gradient(dZdy, dy, dx)
    _dummy, d2Zdx = np.gradient(dZdx, dy, dx)

    d2Zdnw = signal.convolve2d(dZdnw, NW_SE_ker, boundary='symm', mode='same')
    d2Zdne = signal.convolve2d(dZdne, NE_SW_ker, boundary='symm', mode='same')

    curvature = d2Zdx + d2Zdy + d2Zdne + d2Zdnw

    return curvature


def slope_aspect_D2(array, dx, dy, unit='degree'):
    """
    Compute slope magnitude and aspect (direction) of a 2D array using 2
    directions (x and y, no diagonals)

    Parameters
    ----------
    array : array
        input 2D array
    dx : float
        grid size in x-direction
    dy : float
        gris size in y-direction
    unit : str, optional
        slope units, by default 'degree'

    Returns
    -------
    array
        2D array of slope
    array
        2D array of aspect
    """

    """Compute the slope"""
    dZdy, dZdx = np.gradient(array, dy, dx)
    slope = np.sqrt(dZdy**2 + dZdx**2)

    """counter clockwise from East"""
    aspect = np.rad2deg(np.arctan2(dZdy, dZdx))
    neg_aspect = np.where((aspect < 0.))[0]
    aspect[neg_aspect] = 360. + aspect[neg_aspect]

    """ save at 10 deg increments to use 8 bit int"""
    aspect = np.round(aspect/10)

    if unit == 'degree':
        """Convert slope to degrees and save as tif"""
        slope = np.rad2deg(np.arctan(slope)).astype(np.int8)

    return slope, aspect


def get_diag_kernel(dxy):

    NE_SW_ker = np.array([[0., 0., dxy],
                          [0., 0., 0.],
                          [-dxy, 0., 0.]])

    NW_SE_ker = np.array([[dxy, 0., 0.],
                          [0., 0., 0.],
                          [0., 0., -dxy]])

    return NE_SW_ker, NW_SE_ker
