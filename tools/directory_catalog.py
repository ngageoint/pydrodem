#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Kimberly McCormack

:Last edited on: 10/28/2021

Script to build catalog of full paths for data directory and search for paths given list of tiles

"""


import glob
import itertools
import os
import shutil
from shapely import geometry
import numpy as np
import geopandas as gpd



def buildpathlist(src_base_path, searchstr, listpath):
    """
    Build and save a text file of full path names to
    specfied files in a directory

    Parameters
    ----------
    src_base_path : path [str]
        directory to search
    searchstr : string
        string to filter by
    listpath : path [str]
        full path to save text file
    """

    cell_paths = []
    for root, dirs, files in os.walk(src_base_path):
        for d in dirs:
            cell_paths.append(glob.glob(os.path.join(root, d, searchstr)))

    cell_paths = list(itertools.chain(*cell_paths))
    with open(listpath, 'w') as filehandle:
        for path in cell_paths:
            if 'lidar' not in path:
                filehandle.write("{0}\n".format(path))



def tile_searchTDT(tile, searchlines):
    """
    search for version 2 first, then version 1
    """
    cell_search = tile +"_02"
    cell_path = ""
    """Search TDT txt file for path of TDT cell"""
    for line in searchlines:
        if cell_search in line:
            cell_path = line
            break
    if len(cell_path) == 0:
        print("TDT version 2 not found, searching for version 1 instead")
        cell_search = tile +"_01"
        for line in searchlines:
            if cell_search in line:
                cell_path = line
                break

    return cell_path



def buildTDTbbox(TDTcells):
    """
    Create geodataframe of bounding box geometries for list of TanDEM-X cells

    Parameters
    ----------
    TDTcells : list of strings
        list of TanDEM-X cell names - e.g. ['N39E047', 'S08W102']

    Returns
    -------
    TDT_poly_gdf : geopandas geodataframe
        gdf with cell name as index and cell bounding box as geometry

    """

    bbox_bottom = np.array([np.int(cell[1:3]) for cell in TDTcells])
    bbox_left = np.array([np.int(cell[-3::]) for cell in TDTcells])

    """Deal with quadrants"""
    for i, cell in enumerate(TDTcells):
        if cell[0] == 'S':
            bbox_bottom[i] = -1*bbox_bottom[i]
        if cell[3] == 'W':
            bbox_left[i] = -1*bbox_left[i]

    """Create geodataframe of TDT bounding boxes"""
    TDT_bbox = [geometry.box(l, b, r, t)
                for l, b, r, t in zip(bbox_left, bbox_bottom,
                                      bbox_left+1, bbox_bottom+1)]
    TDT_poly_gdf = gpd.GeoDataFrame(geometry=TDT_bbox)
    
    TDT_poly_gdf.crs = "EPSG:4326"
    TDT_poly_gdf.index = TDTcells
    TDT_poly_gdf['TILE_ID'] = TDTcells

    return TDT_poly_gdf



def find_duplicates(list1):
 
    # initialize a null list
    unique_list = []
    duplicate_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
        else:
            duplicate_list.append(x)
            
    return duplicate_list

