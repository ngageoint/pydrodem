#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 15 2021

@author: Kimberly McCormack

script to impose minimum water value in overlapping coastal basins

"""
import os
import warnings
warnings.simplefilter('ignore')
import sys
import traceback
import time
import json
import numpy as np
import geopandas as gpd
from osgeo import gdal, gdal_array
from configparser import ConfigParser, ExtendedInterpolation
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.log as logt
import tools.convert as pyct
from tools.print import print_time, log_time
import tools.files as ft


def delete_temp_files(outpath, basename, in_dem, in_mask):
    """
    If run is successful, delete temp and input files
    """

    ft.erase_files(outpath, search=f'{basename}_*')
    if os.path.exists(in_dem):
        os.remove(in_dem)
    if os.path.exists(in_mask):
        os.remove(in_mask)


def minimize_overlap(input_id, work_dir, base_unit,
                     projectname, buffer, neighbor_gdf,
                     del_temp_files=False, starttime=time.time()):

    try:

       

        basename = 'DEM'

        if base_unit == 'basin':
            usecutline = True
            capstyle = 1
        else:
            usecutline = False
            capstyle = 3  # square caps

        outpath = os.path.join(work_dir, f'{base_unit}_{input_id}')
        
        completed_file = os.path.join(outpath, 'overlap.txt')
        if os.path.exists(completed_file): os.remove(completed_file)

        print(f'   minimize overlaps for {base_unit} {input_id}')
        inDEM = os.path.join(outpath, 'DEMburned.tif')
        water_mask_tif = os.path.join(outpath, 'Water_mask.tif')
        
        
        #re-name input DEM and water mask
        temp_dem_tif = os.path.join(outpath, 'DEMburned_0.tif')
        temp_water_tif = os.path.join(outpath, 'Water_mask_0.tif')
        if not os.path.exists(temp_dem_tif): 
            os.rename(inDEM, temp_dem_tif)
        if not os.path.exists(temp_water_tif): 
            os.rename(water_mask_tif, temp_water_tif)

        #outDEM = os.path.join(outpath, "DEMburned_overlap.tif")
        # water_mask_tif_out = os.path.join(outpath, 'Water_mask_overlap.tif')

        # Create map from mask value to priority rank and back
        wmap = [0, 6, 5, 13, 4, 3, 2, 1, 9, 10, 11, 14, 12, 7, 8, 15]
        wmap_inv = [np.flatnonzero(wmap == i)[0] for i in np.sort(wmap)]

        """Extract raster nodataval"""
        raster = gdal.Open(temp_dem_tif, gdal.GA_ReadOnly)
        DEM_xshape = raster.RasterXSize
        DEM_yshape = raster.RasterYSize
        band = raster.GetRasterBand(1)
        nodataval = band.GetNoDataValue()   # Set nodata value
        raster = None
        band = None

        water_raster = gdal.Open(temp_water_tif, gdal.GA_ReadOnly)
        water_xshape = water_raster.RasterXSize
        water_yshape = water_raster.RasterYSize
        water_raster = None

        water_mask_array = gdal_array.LoadFile(temp_water_tif)
        inDEM_array = gdal_array.LoadFile(temp_dem_tif)

        island_file = os.path.join(outpath, 'island_chain.txt')
        if os.path.exists(island_file):
            print(f'  {input_id} - island chain basin')
            
            #rename input files to outputs
            os.rename(temp_dem_tif, inDEM)
            os.rename(temp_water_tif, water_mask_tif)
            
            """Erase temp files and write out completed .txt"""
            if del_temp_files:
                delete_temp_files(outpath, basename, 
                                    temp_dem_tif, 
                                    temp_water_tif)
            with open(completed_file, 'w') as f:
                f.write('DEM overlap processing complete')
            
            return

        """find overlapping regions"""
        input_buffer_shpfile = os.path.join(
            outpath, f'{base_unit}_{input_id}_buffer.shp')
        input_gdf = neighbor_gdf.loc[
            neighbor_gdf['input_id'] == input_id].copy()
        input_polygon = list(input_gdf.geometry)[0]
        neighbor_gdf = neighbor_gdf.loc[
            neighbor_gdf.geometry.intersects(input_polygon)]
        neighbor_gdf = neighbor_gdf[neighbor_gdf.input_id != input_id]
        
        overlap_gdf = input_gdf.overlay(neighbor_gdf, how='intersection')
        if overlap_gdf.empty:
            print(f'  {input_id} - No overlapping {base_unit}s')
            
            #rename input files to outputs
            os.rename(temp_dem_tif, inDEM)
            os.rename(temp_water_tif, water_mask_tif)
            
            """Erase temp files and write out completed .txt"""
            if del_temp_files:
                delete_temp_files(outpath, basename, 
                                    temp_dem_tif, 
                                    temp_water_tif)
            with open(completed_file, 'w') as f:
                f.write('DEM overlap processing complete')

            return

        overlap_shp = os.path.join(
            outpath, f'{basename}_{base_unit}_{input_id}_overlap.shp')
        overlap_gdf.to_file(overlap_shp)

        """create overlap mask"""
        blankmask_tif = os.path.join(
            outpath, f"{basename}_blank_mask.tif")
        rastermask = np.ones_like(water_mask_array, dtype=np.int8)
        pyct.npy2tif(rastermask, temp_dem_tif, blankmask_tif,
                     nodata=0, dtype=gdal.GDT_Byte)

        overlap_mask = os.path.join(
            outpath, f"{basename}_overlap_mask.tif")
        pyct.cutline_raster_simple(blankmask_tif, overlap_shp, overlap_mask,
                                   nodata=0, outdtype=gdal.GDT_Byte)
        overlap_array = gdal_array.LoadFile(overlap_mask)
        overlap_ind = np.nonzero(overlap_array)

        if not neighbor_gdf.empty:

            overlap_IDs = list(neighbor_gdf['input_id'].values)
            overlap_IDs.append(input_id)

            DEM_overlap = np.empty((len(overlap_ind[0]), len(overlap_IDs)))
            Watermask_overlap = np.empty(
                (len(overlap_ind[0]), len(overlap_IDs)))

            for i, overlap_id in enumerate(overlap_IDs):
                # clip all overlapping DEMS to buffered shape
                inDEM_neighbor = os.path.join(work_dir,
                                              f'{base_unit}_{overlap_id}',
                                              'DEMburned.tif')
                water_mask_neighbor = os.path.join(work_dir,
                                                   f'{base_unit}_{overlap_id}',
                                                   'Water_mask.tif')

                clipDEM = os.path.join(
                    outpath, f"{basename}_overlapDEM_{overlap_id}.tif")
                pyct.create_cutline_raster(inDEM_neighbor,
                                           input_buffer_shpfile,
                                           clipDEM, srcDS=temp_dem_tif,
                                           nodata=np.nan,
                                           usecutline=usecutline)

                nb_DEM_array = gdal_array.LoadFile(clipDEM)
                nb_DEM_array[nb_DEM_array == nodataval] = np.nan
                DEM_overlap[:, i] = nb_DEM_array[overlap_ind]

                clipmask = os.path.join(
                    outpath, f"{basename}_overlapmask_{overlap_id}.tif")
                pyct.create_cutline_raster(water_mask_neighbor,
                                           input_buffer_shpfile,
                                           clipmask, srcDS=temp_dem_tif,
                                           nodata=0,
                                           usecutline=usecutline,
                                           outdtype=gdal.GDT_Byte)

                # map to priority values
                nb_mask_array = gdal_array.LoadFile(clipmask)
                nb_mask_overlap = nb_mask_array[overlap_ind]
                nb_mask_overlap_map = np.array(
                    [wmap[i] for i in nb_mask_overlap])
                nb_mask_overlap_map[nb_mask_overlap_map == 0] = 99
                Watermask_overlap[:, i] = nb_mask_overlap_map
                
            quit()
            

            # find min of mask overlap (min value = highest priority)
            watermask_overlap_min = np.nanmin(Watermask_overlap, axis=1)
            non_water_ind = np.flatnonzero(watermask_overlap_min == 99)
            watermask_overlap_min[non_water_ind] = 0

            # re-map back to orginal values
            watermask_overlap_min = np.array(
                [wmap_inv[int(i)] for i in watermask_overlap_min])

            # find min of DEM overlap
            DEM_array_overlap_min = np.nanmin(DEM_overlap, axis=1)

            """Save output DEM"""
            # DEM_array_overlap_min[non_water_ind] = nodataval
            inDEM_array[overlap_ind] = DEM_array_overlap_min
            inDEM_array[np.isnan(inDEM_array)] = nodataval
            pyct.npy2tif(inDEM_array, temp_dem_tif, inDEM, nodata=nodataval)

            """Save water mask"""
            water_mask_array[overlap_ind] = watermask_overlap_min
            pyct.npy2tif(water_mask_array, temp_dem_tif,
                         water_mask_tif_out, nodata=0, dtype=gdal.GDT_Byte)
                         
            print_time(f'    {input_id} overlap DEM saved', starttime)

        """Erase temp files and write out completed .txt"""
        if del_temp_files:
            delete_temp_files(outpath, basename, 
                                temp_dem_tif, 
                                temp_water_tif)
        with open(completed_file, 'w') as f:
            f.write('DEM overlap processing complete')
            
            
        return
            
      

    except Exception as e:
        outst = "Exception occurred on basin {0}: {1}".format(input_id, e)
        traceback_output = traceback.format_exc()
        print(outst)
        print(traceback_output)


if __name__ == '__main__':

    starttime = time.time()

    """ Load the configuration file """
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="config file")
    ap.add_argument("-l", "--logfile", required=False, help="log file")
    args = vars(ap.parse_args())
    config_file = os.path.abspath(args['config'])
    log_file = args['logfile']

    """read config file"""
    config = ConfigParser(allow_no_value=True,
                          interpolation=ExtendedInterpolation())
    config.read(config_file)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    #################### SET OPTIONS  ####################
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    run_parallel = config.getboolean("job-options", "run_parallel")
    run_multiprocess = config.getboolean("job-options", "run_multiprocess")
    verbose_print = config.getboolean(
        "processes-depression-handling", "verbose_print")
    del_temp_files = config.getboolean(
        "processes-depression-handling", "del_temp_files")

    """parallel options"""
    if run_parallel:
        import mpi4py.rc
        mpi4py.rc.finalize = False
        from mpi4py.futures import MPICommExecutor
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        nnodes = comm.size
    elif run_multiprocess:
        from multiprocessing import Pool, cpu_count

    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    ######### IMPORT REGION AND PATH INFORMATION #########
    """"""""""""""""""""""""""""""""""""""""""""""""""""""

    """ Path to White Box Tools"""
    WBT_path = os.path.join(config.get("paths", "WBT_path"))
    base_unit = config.get("region", "base_unit")

    """locations of input data"""
    SWO_vrt = os.path.join(config.get("paths", "SWO_vrt"))
    OSM_dir = os.path.join(config.get("paths", "OSM_dir"))
    flattened_mask_vrt = os.path.join(
        config.get('outputs', 'water_mask_mosaic'))
    SWO_cutoff = config.getint("parameters-depression-handling", "SWO_cutoff")

    """output directories"""
    projectname = config.get("outputs", "projectname")
    work_dir = config.get("outputs", "dh_work_dir")
    overwrite = config.getboolean("processes-depression-handling", "overwrite")

    buffer = config.getfloat("parameters-depression-handling", "buffer")
    overlap_shp = os.path.join(config.get('outputs', 'overlap_shp'))
    overlap_gdf = gpd.read_file(overlap_shp)

    """find un-minimized ids"""
    if not overwrite:
        unfin_ids = logt.find_unfinished_ids(work_dir,
                                             unit=base_unit,
                                             dir_search=f'{base_unit}_',
                                             tifname='overlapDEM.tif')
    else:
        unfin_ids = logt.find_all_ids(work_dir,
                                      unit=base_unit,
                                      dir_search=f'{base_unit}_')

    n_unfin = len(unfin_ids)

    if run_parallel:
        with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
            if executor is not None:
                print_time(f'Minimizing {n_unfin} overlaps on {nnodes} nodes',
                           starttime, sep=True)
                executor.starmap(minimize_overlap, [(input_id, work_dir,
                                                     base_unit,
                                                     projectname, buffer,
                                                     overlap_gdf,
                                                     del_temp_files)
                                                    for input_id in unfin_ids])

    elif run_multiprocess:
        nthreads = min(n_unfin, 14)
        with Pool(nthreads) as p:
            print_time(f'Minimizing {n_unfin} overlaps on {nthreads} threads',
                       starttime, sep=True)
            p.starmap(minimize_overlap, [(input_id, work_dir, base_unit,
                                          projectname, buffer, overlap_gdf,
                                          del_temp_files)
                                         for input_id in unfin_ids])

    else:
        for input_id in unfin_ids:
            minimize_overlap(input_id, work_dir, base_unit,
                             projectname, buffer, overlap_gdf,
                             del_temp_files=del_temp_files)

        log_str = log_time("Overlap Minimization COMPLETE", starttime)
        with open(log_file, 'w') as f:
            f.write(log_str)
