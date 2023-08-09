#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 2021

@author: Kimberly McCormack

class to check taudem outputs over large areas for nodat values in the flow direction vrt

"""

import os
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings(action='ignore', message='GEOS')
import sys
import time
import traceback
import numpy as np
from osgeo import gdal, gdal_array
from configparser import ConfigParser, ExtendedInterpolation
import argparse
import json
import geopandas as gpd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.log as logt
import tools.shapes as shpt
import tools.convert as pyct
import tools.files as ft
from tools.print import print_time


class TaudemCheck():
    
    def __init__(self, work_dir, basin_work_dir, taudem_dir, projectname):

        self.starttime = time.time()
        

        self.basin_work_dir = basin_work_dir
        self.taudem_dir = taudem_dir
        self.projectname = projectname
        self.taudem_projdir = os.path.join(self.taudem_dir, self.projectname)
        
        self.filled_DEM_vrt = os.path.join(self.taudem_projdir, 'fel.vrtd', 'orig.vrt')
        self.flow_dir_vrt = os.path.join(self.taudem_projdir, 'p.vrtd', 'orig.vrt')
        self.edm_mosaic = os.path.join(os.path.dirname(os.path.dirname(work_dir)), 
                                                      "DTM_output_mosaics",
                                                      "{0}_EDM.tif".format(self.projectname))
        
        
        
    def check_basin(self, basin_id):
        """
        1. Pull the flow direction raster over a basin
        2. check for no data values that don not overlap:
            a. DEM nodata values
            b. endorheic sinks
        3. if nodata values in taudem d8 flow direction, raise river elevations by 1cm
        
        """
        
        """"""""""""""""""""""""""""""""
        """      Load/Setup Data     """
        """"""""""""""""""""""""""""""""
        
        print_time(" Check basin {0}".format(basin_id), self.starttime)
        try:
            outpath_basin = os.path.join(self.basin_work_dir, 'basin_{0}'.format(basin_id))
            ft.erase_files(outpath_basin, search='taudem_*')
            
            basin_bbox_shpfile = os.path.join(outpath_basin, 'basin_{0}_bbox.shp'.format(basin_id))
            if os.path.exists(basin_bbox_shpfile):
                basin_shpfile = basin_bbox_shpfile
            else:
                basin_shpfile = os.path.join(outpath_basin, 'basin_{0}.shp'.format(basin_id))
                
            
            basin_gdf = gpd.read_file(basin_shpfile)
            
            if basin_gdf["COAST"].values[0] == 1:
                print('   {0} - coastal basin - skip'.format(basin_id))
                return
            
            coast_file =  os.path.join(outpath_basin, 'coast.txt')
            if os.path.exists(coast_file):
                print('   {0} - coastal basin - skip'.format(basin_id))
                return
            
            basin_gdf, invalid_flag = shpt.fix_invalid_geom(basin_gdf, fix_w_buffer=False)
            if invalid_flag:
                basin_shpfile = os.path.join(outpath_basin, 'taudem_basin_{0}.shp'.format(basin_id))
                basin_gdf.to_file(basin_shpfile)
                
            
                
            """Load filled DEM and flow direction from taudem outputs"""
            basinDEM_tif = os.path.join(outpath_basin, 'taudem_DEM.tif')
            basin_d8_tif = os.path.join(outpath_basin, 'taudem_d8_flowdir.tif')
            flatmask_tif = os.path.join(outpath_basin, 'Water_flatten_mask.tif')
            basin_flatmask_tif = os.path.join(outpath_basin, 'taudem_flatten_mask.tif')
            edm_mask = os.path.join(outpath_basin, 'taudem_EDM.tif')
            
            pyct.create_cutline_raster_set_res(self.filled_DEM_vrt, basin_shpfile, basinDEM_tif,
                                               usecutline=True)

            pyct.create_cutline_raster(self.flow_dir_vrt, basin_shpfile,
                                       basin_d8_tif, srcDS=basinDEM_tif, nodata=99, 
                                       usecutline=True, outdtype=gdal.GDT_Int16)
            
            pyct.create_cutline_raster(flatmask_tif, basin_shpfile,
                                       basin_flatmask_tif, srcDS=basinDEM_tif, nodata=0, 
                                       usecutline=True, outdtype=gdal.GDT_Byte)

            pyct.create_cutline_raster(self.edm_mosaic, basin_shpfile, edm_mask,
                                       srcDS=basinDEM_tif, usecutline=True,
                                       outdtype=gdal.GDT_Byte) 
            
            edm_array = gdal_array.LoadFile(edm_mask)    
                       
            """Extract DEM raster shape and nodataval"""
            raster = gdal.Open(basinDEM_tif, gdal.GA_ReadOnly)
            xshape = raster.RasterXSize
            yshape = raster.RasterYSize
            band = raster.GetRasterBand(1)
            self.nodataval = band.GetNoDataValue()   # Set nodata value
            raster = None
            band = None
            
            """Extract d8 flow direction raster shape and nodataval"""
            raster = gdal.Open(basin_d8_tif, gdal.GA_ReadOnly)
            xshape = raster.RasterXSize
            yshape = raster.RasterYSize
            band = raster.GetRasterBand(1)
            self.nodataval_d8 = band.GetNoDataValue()   # Set nodata value
            raster = None
            band = None
            
            DEM_array = gdal_array.LoadFile(basinDEM_tif)
            flatten_mask_array = gdal_array.LoadFile(basin_flatmask_tif)
            d8_array = gdal_array.LoadFile(basin_d8_tif)
            d8_nodata_ind = np.where((d8_array==self.nodataval_d8) 
                                     & (DEM_array!=self.nodataval)
                                     & (flatten_mask_array>1)
                                     & (flatten_mask_array<6)
                                     & (edm_array<4)) 
            
            n_nodata_d8 = len(d8_nodata_ind[0])
            
            if n_nodata_d8 == 0:  
                #print('no problem nodatavalues in flow direction!')
                ft.erase_files(outpath_basin, search='taudem_*')
                return
            
            else: 
                """load sink data if present"""
                sink_mask_tif = os.path.join(outpath_basin, 'endo_sink.tif')
                if os.path.exists(sink_mask_tif):
                    #print('endorheic basin')
                    ft.erase_files(outpath_basin, search='taudem_*')
                    return
                
                else:
                    print('  basin {0}: nodata values in flow direction, raise river elevations by 10 cm'.format(
                        basin_id))
                    self.fix_basin(outpath_basin, basinDEM_tif, DEM_array, 
                                   basin_shpfile, flatten_mask_array)
                    #ft.erase_files(outpath_basin, search='taudem_*')
                    

            
        except Exception as e:
            outst = "Exception occurred on basin {0}: {1}".format(basin_id, e)
            traceback_output = traceback.format_exc()
            print(outst)
            print(traceback_output)
            ft.erase_files(outpath_basin, search='taudem_*')
            pass
            
            
    
    def fix_basin(self, outpath_basin, basinDEM_tif, DEM_array, basin_shpfile, flatten_mask_array):
        

        water_ind = np.where((flatten_mask_array>1) & (flatten_mask_array<6))

        DEM_array[water_ind] = DEM_array[water_ind] + 0.10 

        """Output modified DEM"""
        outDEM_tif = os.path.join(outpath_basin, "conditionedDEM_taudem_adjust.tif")
        DEM_array[DEM_array==self.nodataval] = np.nan
        pyct.npy2tif(DEM_array, basinDEM_tif, outDEM_tif,
             nodata=np.nan, dtype=gdal.GDT_Float32)
        
        



def run_taudem_check(basinID):
    tau_check.check_basin(basinID)
    

if __name__ == '__main__':
    
    starttime = time.time()
    
    """ Load the configuration file """
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="config file")
    ap.add_argument("-l", "--logfile", required=False, help="log file")
    args = vars(ap.parse_args())
    log_file = args['logfile']  
    config_file = args['config']  
    config = ConfigParser(allow_no_value=True, interpolation=ExtendedInterpolation())
    config.read(config_file)
    
        
    """Pull paths from config sile"""
    taudem_dir      = config.get("paths-taudem", "taudem_dir")
    projectname     = config.get("outputs", "projectname")
    basin_levels    = json.loads(config.get("parameters-DH-basin", "basin_levels")) 
    work_dir        = os.path.join(config.get("paths-DH", "output_dir"), projectname)
    basin_work_dir  = os.path.join(work_dir, 'basins_lvl{0:02d}'.format(basin_levels[0]))
    run_w_MPI        = config.getboolean("job-options", "run_w_MPI")
    verbose_print    = config.getboolean("processes-DH-all", "verbose_print")
    
    
    """mpi4py options"""
    if run_w_MPI:        
        import mpi4py.rc
        mpi4py.rc.finalize = False
        from mpi4py.futures import MPICommExecutor, as_completed
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank 
        numnodes = comm.size
    else:
        rank = 0
        numnodes = 1


    
    """Initialize class with input parameters""" 
    tau_check = TaudemCheck(work_dir, basin_work_dir, taudem_dir, projectname)
    basinIDs = logt.find_all_basins_IDs(basin_work_dir)
        
    
    if run_w_MPI:
        with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
            if executor is not None:
                print_time('Checking {0} basins on {1} nodes'.format(len(basinIDs),
                                            numnodes), starttime, sep=True)
                executor.map(run_taudem_check, basinIDs) 

    else:
        for basinID in basinIDs:
            run_taudem_check(basinID)

        log_str = "TAUDEM CHECK COMPLETE"
        with open(log_file, 'w') as f: f.write(log_str)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        