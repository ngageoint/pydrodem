# -*- coding: utf-8 -*-
"""
Created on 8/24/2021

author: Heather Levin
"""

import os
import sys
import time
import traceback
import warnings
import glob
import subprocess
import geopandas as gpd
from osgeo import gdal
import argparse
from configparser import ConfigParser, ExtendedInterpolation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.convert as pyct
import tools.files as ft
import tools.log as logt
from tools import endo
from tools.print import print_time, log_time

gdal.UseExceptions()
warnings.filterwarnings(action='ignore', message='GEOS')
# gdal.PushErrorHandler('CPLQuietErrorHandler')

class FindSinks():
    
    def __init__(self, inDS_vrt, endo_work_dir,
                 SWO_vrt, projectname, WBT_path, 
                 verbose_print=True,
                  rank=0):
        
        self.WBT_path = WBT_path
        self.rank = rank
        self.starttime = time.time()
        self.inDS_vrt = inDS_vrt
        self.endo_work_dir = endo_work_dir
        self.SWO_vrt = SWO_vrt
        self.projectname = projectname
        self.verbose_print = verbose_print         
        
        """Load WhiteBox Tools executable"""
        sys.path.append(WBT_path)
        from whitebox_tools import WhiteboxTools
        self.wbt = WhiteboxTools()
        self.wbt.set_verbose_mode(False)
        self.wbt.set_whitebox_dir(WBT_path)# set location of WBT executable
        
        
    def endo_basin_setup(self, input_id, endo_nodata=99999, overwrite=False):
        """
        Builds VRTs clipped to unbuffered level 12 HydroBASINS shapefiles for
        endorheic basins. Identifies a sink point at the lowest elevation pixel
        or the centroid of the lowest elevation pixels. Saves a the sink to a 
        point shapefile inside the subbasin folder.

        Parameters
        ----------
        input_id : integer
            HydroBASINS ID for the level 12 basin.
        endo_nodata : integer, optional
            Value to use as nodata for the endo basin vrt. This is positive to 
            ensure it doesn't interfere with finding the lowest elevations. 
            The default is 99999.

        Returns
        -------
        None.

        """
        print_time(f"   Endo basin {input_id}", self.starttime)

        outpath_endobasin = os.path.join(self.endo_work_dir, 
                                     'basin_{0}'.format(input_id))
        ft.erase_files(outpath_endobasin, search='DEM_*')

        sink_point_shp = os.path.join(outpath_endobasin, "sink.shp")
        
        """skip if sink file exists"""
        if os.path.exists(sink_point_shp) and not overwrite:
            return

        basin_shpfile = os.path.join(outpath_endobasin, 'basin_{0}.shp'.format(input_id))
        endobasin_gdf = gpd.read_file(basin_shpfile)
        
        try: 
            if endobasin_gdf["ENDO"].values[0] == 2:
                #print("endo basin found:", input_id)
                basin_poly = list(endobasin_gdf.geometry)[0]
                
                """ create basin buffer (zero buffer) to fix invalid geomoetries"""
                endobasin_buffer_shp = os.path.join(outpath_endobasin,
                                            'basin_{0}_buffer.shp'.format(input_id))
                if not os.path.exists(endobasin_buffer_shp):
                    endobasin_gdf.reset_index(drop=True, inplace=True)
                    endobasin_gdf["group"] = 1
                    endobasin_gdf_dissolved = endobasin_gdf.dissolve(by="group")                    
                    pyct.buffer_gdf2shp(endobasin_gdf_dissolved, 0.0, endobasin_buffer_shp)


                """Crop VRTs to unbuffered basin"""
                endoDEM = os.path.join(outpath_endobasin, 'DEM_endo.tif')
                
                pyct.cutline_raster_cmdline(self.inDS_vrt, endobasin_buffer_shp, 
                                            endoDEM, dstnodata=endo_nodata)

               
                """ Run fill single cell pits for endoDEM before finding 
                sinks. This will prevent a single cell pit from being 
                hosen as a sink when the surrounding pixel values
                do not support it. """
                endoDEM_sc = os.path.join(outpath_endobasin, 'DEM_endo_sc.tif')
                self.wbt.fill_single_cell_pits(endoDEM, endoDEM_sc)
                
                sink_point_gdf = endo.find_sink(input_id, endoDEM_sc, 
                                                endobasin_buffer_shp,
                                                SWO_vrt_path=self.SWO_vrt, 
                                                nodata=endo_nodata)

                """save sink points to shapefile""" 
                sink_point_gdf.to_file(sink_point_shp)           

                """delete endoDEM file"""
                os.remove(endoDEM)
                os.remove(endoDEM_sc)
    
        except Exception as e:
            outst = "Exception occurred on rank {0}, basin {1}: {2}".format(
                self.rank, input_id, e)
            traceback_output = traceback.format_exc()
            print(outst)
            print(traceback_output)
            pass        
        
        
        
        
        
def run_find_sinks_basin(endo_basinIDs):
    """ Tiny wrapper function to make the mpi executor behave """   
    fSinks.endo_basin_setup(endo_basinIDs)
    

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

    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    #################### SET OPTIONS  ####################
    """"""""""""""""""""""""""""""""""""""""""""""""""""""    
    run_multiprocess   = config.getboolean("job-options", "run_multiprocess")    
    run_parallel       = config.getboolean("job-options", "run_parallel")
    verbose_print      = config.getboolean("processes-depression-handling", "verbose_print")
    overwrite          = config.getboolean("processes-depression-handling", "overwrite")
   
    """mpi4py options"""
    if run_parallel:
        import mpi4py.rc
        mpi4py.rc.finalize = False
        from mpi4py.futures import MPICommExecutor, as_completed
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank 
        numnodes = comm.size
    elif run_multiprocess:
        from multiprocessing import Pool, cpu_count

    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    ######### IMPORT REGION AND PATH INFORMATION #########
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    
    
    """ Path to White Box Tools"""
    WBT_path = os.path.join(config.get("paths", "WBT_path"))

    """locations of input data"""
    SWO_vrt = os.path.join(config.get("paths", "SWO_vrt"))
    OSM_dir = os.path.join(config.get("paths", "OSM_dir"))
    inDS_vrt = os.path.join(config.get("outputs", "dtm_mosaic"))

    """output directories"""
    projectname = config.get("outputs", "projectname")
    endo_work_dir = os.path.join(config.get("paths", "output_dir_cond"),
                                  projectname, 'endo_basins')
                                

    ###############################################
    """          Run sink finding               """
    ###############################################
    
    endo_basin_dirs = glob.glob(endo_work_dir + '/**/basin_*.shp', recursive=True)
    sink_points_shp = os.path.join(endo_work_dir, 'sinks.shp')
    
    # if (len(endo_basin_dirs) > 0) and (not os.path.exists(sink_points_shp)):
        # run_endo_setup = True
        # endo_basinIDs = [path[-14:-4] for path in endo_basin_dirs if path[-5]!='r'] # exclude paths ending in buffer
    # else: 
        # run_endo_setup = False 

    print(endo_work_dir)

    if not overwrite:
        print('find unfinished')
        endo_ids = logt.find_unfinished_ids(endo_work_dir,
                                             unit='basin',
                                             dir_search='basin_',
                                             tifname='sink.shp')
    else:
        print('find all')
        endo_ids = logt.find_all_ids(endo_work_dir,
                                      unit='basin',
                                      dir_search=f'basin_')  
                                      
    n_endo_ids = len(endo_ids)


    """Initialize class with input parameters"""
    fSinks = FindSinks(inDS_vrt, endo_work_dir,
                 SWO_vrt, projectname, WBT_path, 
                 verbose_print=False)



    """Run sink selection"""
    if run_parallel:
        with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
            if executor is not None:
                executor.map(run_find_sinks_basin, endo_ids)

    elif run_multiprocess:

        nthreads = min(len(endo_ids), 14)
        with Pool(nthreads) as p:
            print_time(f'river enforcement for {n_endo_ids}'+
                       f' basins on {nthreads} threads',
                           starttime, sep=True)
            p.map(fSinks.endo_basin_setup, endo_ids)

    else:  
        for endo_id in endo_ids[0:1]:
            fSinks.endo_basin_setup(endo_id)            
            print_time(f"BASIN {endo_id} COMPLETE", starttime)

        log_str = log_time("ENDO SINK FINDING COMPLETE", starttime)
        with open(log_file, 'w') as f: f.write(log_str)
        print(log_str)
   
        