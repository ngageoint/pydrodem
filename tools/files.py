#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:42:47 2020

@author: Kimmy McCormack
"""

import glob
import itertools
import os
import shutil

import numpy as np


def erase_files(outpath, search='*'):
    """
    Erases all files in the directories given by outpath and search string

    Parameters
    ----------
    outpath : path
        directory to erase all files within
    search : str, optional
        optional filter string for glob function, by default '*' (delete everything)
    """

    files = glob.glob(os.path.join(outpath, search))
    onlyfiles = [f for f in files if os.path.isfile(f)]
    for f in onlyfiles:
        os.remove(f)


def delete_file_all_subdir(work_dir, searchstr='basin_*', fname=''):
    """
    Delete a specifc filename within all subdirectories

    Parameters
    ----------
    work_dir : path
        directory to search (containing subdirectories containing the files to be deleted)
    searchstr : str, optional
        filter for subdirectory search, by default 'basin_*'
    fname : str, optional
        file name (with extension) to be deleted, by default ''
    """

    basins = glob.glob(os.path.join(work_dir, searchstr))
    for sb in basins:
        inDS = os.path.join(sb, fname)
        if os.path.exists(inDS):
            os.remove(inDS)


def remove_basin_dir(basin_work_dir, basinIDs):

    basin_paths = [os.path.join(
        basin_work_dir, 'basin_{0}'.format(b)) for b in basinIDs]
    """remove basin directories before re-running basin setup"""
    for basin_path in basin_paths:
        if os.path.exists(basin_path):
            shutil.rmtree(basin_path)


def buildpathlist(src_base_path, searchstr="*"):
    """
    Build and save a text file of full path names to
    specfied files in a directory

    Parameters
    ----------
    src_base_path : path [str]
        directory to search
    searchstr : string
        string to filter by


    """

    pathlist = []
    for root, dirs, files in os.walk(src_base_path):
        for d in dirs:
            pathlist.append(glob.glob(os.path.join(root, d, searchstr)))

    pathlist = list(itertools.chain(*pathlist))

    return pathlist


def pathlist_tifs(work_dir, tifname, secondtifname=None):

    output_paths = glob.glob(os.path.join(work_dir, '*'))
    pathlist = [os.path.join(p, f'{tifname}.tif') for p in output_paths]

    return pathlist


def buildFileList(outpath, srcpath, searchstr="*"):
    """
    Builds and saves a .txt file of full file paths within SRCPATH
    (one level) containing "searchstr"

    Parameters
    ----------
    outpath : path (str)
        PATH TO STORE LIST OF FILE PATHS.
    srcpath : path (str)
        PATH TO SEARCH DIRECTORY
    searchstr : string, optional
        STRING TO FILTER FILE NAMES. The default is "*".

    Returns
    -------
    None.

    """

    cell_paths = glob.glob(os.path.join(srcpath, searchstr))

    with open(outpath, 'w') as filehandle:
        for path in cell_paths:
            filehandle.write("{0}\n".format(path))


def buildFileListMultiple(outpath, srcpath, searchstr="*"):
    """
    Builds and saves a .txt file of full file paths within SRCPATH
    (any level down) containing "searchstr"

    Parameters
    ----------
    outpath : path (str)
        PATH TO STORE LIST OF FILE PATHS.
    srcpath : path (str)
        PATH TO SEARCH DIRECTORY
    searchstr : string, optional
        STRING TO FILTER FILE NAMES. The default is "*".

    Returns
    -------
    None.

    """

    cell_paths = []
    for root, dirs, files in os.walk(srcpath):
        for d in dirs:
            cell_paths.append(glob.glob(os.path.join(root, d, searchstr)))

    cell_paths = list(itertools.chain(*cell_paths))

    with open(outpath, 'w') as filehandle:
        for path in cell_paths:
            filehandle.write("{0}\n".format(path))


def tilesearch(tile, searchlines):
    """
    Search through text file of path names to find the path of the cell

    Parameters
    ----------
    tile : string
        Name of cell to search for
    searchlines : list
        lines of open text file containing DEM paths

    Returns
    -------
    tile_path : path
        full path to input cell

    """

    tile_path = None
    """Search txt file for path of cell"""
    for line in searchlines:
        if tile in line:
            tile_path = line
            break

    return tile_path


def findSWOcell(cell, TDT=True):
    """
    Find overlapping 10x10 degree Surface Water Occurence cell for a give TanDEM-X cell

    Parameters
    ----------
    cell : str
        name of TanDEM-X cell
    TDT : bool, optional
        use TanDEM-X as base dataset, by default True

    Returns
    -------
    str
        name of overlapping SWO tif
    """
    if TDT:
        """ Determine which SWO cell overlaps with TDT cell """
        cell_lat = cell[5:7]
        cell_lon = cell[8:11]
        SWO_EW = cell[7]
        SWO_NS = cell[4]
        if SWO_EW == "E":
            SWO_lon = np.int(cell_lon[:]) - np.int(cell_lon[-1])
        elif SWO_EW == "W":
            SWO_lon = np.int(cell_lon[:]) - np.int(cell_lon[-1]) + 10
        if SWO_NS == "N":
            SWO_lat = ((np.int(cell_lat) - 1) // 10) * 10 + 10
        elif SWO_NS == "S":
            SWO_lat = ((np.int(cell_lat) - 1) // 10) * 10
        if SWO_lat == 0:
            SWO_NS = "N"
        SWO_cell = "{0}{1}_{2}{3}".format(SWO_lon, SWO_EW, SWO_lat, SWO_NS)

    else:
        print("Function only works for TDX cells, currently")
        SWO_cell = None

    return SWO_cell


def findOSMcell(cell, TDT=True):
    """
    Find overlapping 5x5 degree Open Street Mpa water raster for a give TanDEM-X cell

    Parameters
    ----------
    cell : str
        name of TanDEM-X cell
    TDT : bool, optional
        use TanDEM-X as base dataset, by default True

    Returns
    -------
    str
        name of overlapping OSM water tif
    """

    if TDT:
        cell_lat = np.int(cell[5:7])
        cell_lon = np.int(cell[8:11])
        OSM_EW = cell[7]
        OSM_NS = cell[4]
        if OSM_EW == "E":
            EW = "e"
            OSM_lon = cell_lon - (cell_lon % 5)
        elif OSM_EW == "W":
            EW = "w"
            OSM_lon = cell_lon + (5 - cell_lon % 5)
        if OSM_NS == "N":
            NS = "n"
            OSM_lat = cell_lat - (cell_lat % 5)
        elif OSM_NS == "S":
            NS = "s"
            OSM_lat = cell_lat + (5 - cell_lat % 5)

        OSM_cell = "{0}{1:02d}{2}{3:03d}".format(NS, OSM_lat, EW, OSM_lon)

    else:
        print("Function only works for TDT cells, currently")
        OSM_cell = None

    return OSM_cell
