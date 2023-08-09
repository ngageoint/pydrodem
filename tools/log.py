#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Kimberly McCormack, NGA


Error logging functions for pydroDEM codes

"""

import glob
import os
import traceback

import numpy as np
import pandas as pd
from osgeo import gdal, gdal_array

from tools.files import tilesearch

###################################################################
"""                 file search/logging                         """
###################################################################



def find_unfinished_tiles_list(work_dir, dir_search='*', tifname='DTM.tif'):

    unfinished_cells = []
    cell_dirs = glob.glob(os.path.join(work_dir, dir_search))
    for d in cell_dirs:
        finishedDEM = os.path.join(d, tifname)
        if not os.path.exists(finishedDEM):
            unfinished_cells.append(finishedDEM)

    return unfinished_cells


def find_finished_tiles_list(work_dir, dir_search='*', tifname='DTM.tif'):

    finished_cells = []
    cell_dirs = glob.glob(os.path.join(work_dir, dir_search))
    for d in cell_dirs:
        finishedDEM = os.path.join(d, tifname)
        if os.path.exists(finishedDEM):
            finished_cells.append(finishedDEM)

    return finished_cells


def find_unfinished_tiles_ids(work_dir, dir_search='', tifname='DTM.tif'):

    unfinished = []
    tile_dirs = glob.glob(os.path.join(work_dir, f'*{dir_search}*'))
    for tile in tile_dirs:
        finished_tif = os.path.join(tile, tifname)
        if not os.path.exists(finished_tif):
            unfinished.append(tile)

    unfinished = [t.replace(f'{dir_search}', '') for t in unfinished]

    return unfinished


def find_unfinished_ids(work_dir, unit='tile', dir_search='',
                        tifname='DTM.tif', id_list=None):

    unfinished = []
    if id_list is None:
        dirs = glob.glob(os.path.join(work_dir, f'*{dir_search}*'))
    else:
        dirs = [glob.glob(os.path.join(work_dir, f'*{i}*'))[0]
                for i in id_list]

    if unit == 'tile':
        for tile in dirs:
            finished_tif = os.path.join(tile, tifname)
            if not os.path.exists(finished_tif):
                unfinished.append(os.path.basename(tile))
        unfinished = [t.replace(f'{unit}_', '') for t in unfinished]
    elif unit == 'basin':
        for b in dirs:
            finishedDEM = os.path.join(b, tifname)
            if not os.path.exists(finishedDEM):
                unfinished.append(b)

        unfinished = [os.path.splitext(os.path.basename(b))[0].
                      replace(f'{dir_search}', '') for b in unfinished]
        unfinished = [np.int(ID) for ID in unfinished]

    return unfinished


def find_all_ids(work_dir, unit='tile', dir_search=''):

    dirs = glob.glob(os.path.join(work_dir, f'*{dir_search}*'))
    if unit == 'tile':
        all_ids = [os.path.basename(d).replace(f'{dir_search}', '')
                   for d in dirs]
    elif unit == 'basin':
        all_ids_str = [os.path.splitext(os.path.basename(d))[0].
                       replace(f'{dir_search}', '') for d in dirs]
        all_ids = [np.int(ID) for ID in all_ids_str]

    return all_ids


def find_finished_basins_list(work_dir, dir_search='basin_*', tifname='conditionedDEM.tif'):

    basins = glob.glob(os.path.join(work_dir, dir_search))
    fin_basins = []
    for b in basins:
        finishedDEM = os.path.join(b, tifname)
        if os.path.exists(finishedDEM):
            fin_basins.append(b)

    return fin_basins


def find_finished_basins(work_dir, dir_search='basin_*', tifname='conditionedDEM.tif'):

    basin_paths = glob.glob(os.path.join(work_dir, dir_search))
    fin_basin_paths = []
    for b in basin_paths:
        finishedDEM = os.path.join(b, tifname)
        if os.path.exists(finishedDEM):
            fin_basin_paths.append(b)

    return fin_basin_paths


def find_unfinished_basins(work_dir, unfin_basins_txt,
                           dir_search='basin_*', tifname='conditionedDEM.tif'):

    basins = glob.glob(os.path.join(work_dir, dir_search))
    filehandle = open(unfin_basins_txt, 'a')
    for b in basins:
        finishedDEM = os.path.join(b, tifname)
        if not os.path.exists(finishedDEM):
            filehandle.write("{0}\n".format(b))
    filehandle.close()


def find_unfinished_basins_list(work_dir, dir_search='basin_*', tifname='conditionedDEM.tif'):

    basins = glob.glob(os.path.join(work_dir, dir_search))
    unfin_basin_paths = []
    for b in basins:
        finishedDEM = os.path.join(b, tifname)
        if not os.path.exists(finishedDEM):
            unfin_basin_paths.append(b)

    return unfin_basin_paths


def find_unfinished_basins_IDs(work_dir, dir_search='basin_', tifname='conditionedDEM.tif'):

    basin_paths = glob.glob(os.path.join(work_dir, f'*{dir_search}*'))
    unfin_basin_paths = []
    for b in basin_paths:
        finishedDEM = os.path.join(b, tifname)
        if not os.path.exists(finishedDEM):
            unfin_basin_paths.append(b)

    unfin_basinIDs = [os.path.splitext(os.path.basename(b))[0].
                      replace('basin_', '') for b in unfin_basin_paths]
    unfin_basinIDs = [np.int(ID) for ID in unfin_basinIDs]

    return unfin_basinIDs


def find_finished_basins_IDs(work_dir, dir_search='basin_*', tifname='conditionedDEM.tif'):

    basin_paths = glob.glob(os.path.join(work_dir, dir_search))
    fin_basin_paths = []
    for b in basin_paths:
        finishedDEM = os.path.join(b, tifname)
        if os.path.exists(finishedDEM):
            fin_basin_paths.append(b)

    fin_basinIDs = [os.path.splitext(os.path.basename(b))[0].
                    replace('basin_', '') for b in fin_basin_paths]
    fin_basinIDs = [np.int(ID) for ID in fin_basinIDs]

    return fin_basinIDs


def find_all_basins_IDs(work_dir, dir_search='basin_*'):

    basin_paths = glob.glob(os.path.join(work_dir, dir_search))
    basinIDs = [np.int(os.path.splitext(os.path.basename(b))[0].replace('basin_', ''))
                for b in basin_paths]

    return basinIDs


def find_NAN_basins(subbasin_work_dir, nan_basins_txt, tifname='DEM.tif'):

    subbasins = glob.glob(os.path.join(subbasin_work_dir, 'subbasin_*'))
    filehandle = open(nan_basins_txt, 'a')
    nan_basins = 0
    for i, sb in enumerate(subbasins):

        try:
            inDEM = os.path.join(sb, tifname)
            raster = gdal.Open(inDEM, gdal.GA_ReadOnly)
            band = raster.GetRasterBand(1)
            minmax = band.ComputeRasterMinMax()

        except RuntimeError as err:
            print('NAN basin: ', inDEM)
            nan_basins += 1
            filehandle.write("{0}\n".format(sb))
            pass

    print('NAN basins = {0}/{1}'.format(nan_basins, len(subbasins)))
    filehandle.close()

    return




def find_unsolved_basins(subbasin_work_dir, unsolved_basins_txt,
                         tifname='DFpits.h5'):

    subbasins = glob.glob(os.path.join(subbasin_work_dir, 'basin_*'))
    filehandle_bad = open(unsolved_basins_txt, 'a')
    num_sb = np.around(len(subbasins), decimals=-1)
    divider = num_sb // 10

    for i, sb in enumerate(subbasins):
        try:
            dfpits_file = os.path.join(sb, tifname)
            if not os.path.exists(dfpits_file):
                print('   {0} does not have a {1} file'.format(sb, tifname))
                continue

            df_pits = pd.read_hdf(dfpits_file)
            df_unsolved = df_pits[df_pits.solution > 8000].copy()
            unsolvedpits = df_unsolved.index.values
            dfpits = None

            if len(unsolvedpits) > 0:
                filehandle_bad.write("{0}\n".format(sb))

        except Exception as e:
            traceback_output = traceback.format_exc()
            print(traceback_output)

    filehandle_bad.close()

    return
