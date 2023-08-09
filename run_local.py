#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 10 2021

@author: Kimberly McCormack

Last edited on: 11/12/2021

Top level script to run entire hydro-conditioning process

"""
import os
import warnings
warnings.simplefilter('ignore')
import sys
import subprocess
import traceback
import time
import glob
import json
import numpy as np
import shutil
import geopandas as gpd
from osgeo import gdal, gdal_array
import argparse


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tools.log as logt
from tools.print import print_time, log_time
from tools.set_min_water_level import set_min_water
from tools import endo
from setup import PydemSetup
from post_processing import set_nodata
starttime = time.time()

def log_step(step_str, sep=False, CPU=False):
    log_str = log_time(step_str, starttime, sep=sep, CPU=CPU)
    with open(hc_setup.log_file, 'a') as f: f.write(log_str)
    print(log_str)

def wait_4_job(description, output_log):
    min_wait = 0
    while not os.path.exists(output_log):   
        if min_wait == 0: print("Waiting for {0} to finish...".format(description), end="")
        else: print(".", end="")
        sys.stdout.flush()
        time.sleep(30) 
        min_wait += 1
    print('')


if __name__ == '__main__':
    

    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    #####    Load config file and initialize setup   #####
    """""""""""""""""""""""""""""""""""""""""""""""""""""" 
    """ Load the configuration file """
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="config file")
    ap.add_argument("-s", "--system", required=False, help="computing system")
    ap.add_argument("-v", "--virtualenv", required=False, help="virtualenv")
    args = vars(ap.parse_args())
    config_file = os.path.abspath(args['config'])
    venv = os.path.abspath(args['virtualenv'])
    comp_sys = args['system']

    """ Initialize setup class, setup region, outputs, and logging file """
    print('Initializing setup class')
    hc_setup = PydemSetup(config_file)
    
    print_time("Setup region/log file/output directories....", starttime)
    hc_setup.setup_region()
    hc_setup.start_log_file()
    hc_setup.setup_outputs()
    pydem_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    """Extract which processes to run"""
    createDTM = hc_setup.config.getboolean("processes-all", "createDTM")
    dephandle_basins = hc_setup.config.getboolean("processes-all", "dephandle_basins")
    dephandle_noflow = hc_setup.config.getboolean("processes-all", "dephandle_noflow")
    run_taudem = hc_setup.config.getboolean("processes-all", "taudem")
    check_taudem = hc_setup.config.getboolean("processes-all", "check_taudem")
    
    log_step("Project Name: {0}".format(hc_setup.projectname), sep=True)   
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    #####  RUN DTM model (veg removal and smoothing) #####
    """""""""""""""""""""""""""""""""""""""""""""""""""""" 
    overwrite_DTM = hc_setup.config.getboolean("processes-DTM", "overwrite")    

    if createDTM:
        log_step("CREATE DTM FROM DSM", sep=True)  

        hc_setup.setup_DTM()
        
        #----------------------------#s
        ######   RUN DTM MODEL  ######
        #----------------------------# 
        finished_cells = logt.find_finished_TDTcells_list(hc_setup.output_DTM_path, 
                                                                  dir_search='*', 
                                                                  tifname='DTM.tif')
        
        if (len(finished_cells) < len(hc_setup.cell_names)) or (overwrite_DTM):
            sub_call, sub_log = hc_setup.build_dtm_call(pydem_path, comp_sys, venv=venv)            
            subprocess.call(sub_call, shell=True)            
            wait_4_job("DTM", sub_log)

        log_step("DTM completed!", CPU=True) 

        #----------------------------#
        ######   CHECK OUTPUTS  ######
        #----------------------------# 
        """Check all cells were correctly processed"""
        unfinished_cells = logt.find_unfinished_TDTcells_list(hc_setup.output_DTM_path, 
                                                              dir_search='*', 
                                                              tifname='DTM.tif')
        log_step("All cells checked: {0} cells unfinished: {1}".format(
                            len(unfinished_cells), unfinished_cells)) 

        #----------------------------#
        ######   BUILD MOSAIC   ######
        #----------------------------#    
        if hc_setup.build_DTM_Mosaic:
            log_step('Creating mosaic - number of cells to mosaic = {0}'.format(
                                len(hc_setup.cell_names))) 
            output_cell_paths =  glob.glob(os.path.join(hc_setup.output_DTM_path, '*'))   
            DTM_paths = [os.path.join(p, 'DTM.tif') for p in output_cell_paths]
            assert len(unfinished_cells) == 0, 'missing DTM.tif files: {0}'.format(unfinished_cells)
                        
            vrt_options = gdal.BuildVRTOptions(outputSRS="EPSG:4326", 
                                                srcNodata=None, VRTNodata=None)  

            vrt_tif = gdal.BuildVRT(hc_setup.output_DTMmosaic, DTM_paths, options=vrt_options)
            vrt_tif = None   
            
            """ Build EDM mosaic """ 
            EDM_paths = [os.path.join(p, 'EDM.tif') for p in output_cell_paths]  
            EDM_mosaic_path = os.path.join(os.path.dirname(hc_setup.output_DTMmosaic), 
                                           "{0}_EDM.tif".format(hc_setup.projectname))
            
            for EDM_path in EDM_paths:
                if not os.path.exists(EDM_path):
                    print('   {0} does not exist'.format(EDM_path))
            
            print("Building EDM mosaic for", len(EDM_paths), "cells:", EDM_mosaic_path)
            edm_vrt = gdal.BuildVRT(EDM_mosaic_path, EDM_paths, options=vrt_options)
            edm_vrt = None              
            
            log_step("DTM mosaic COMPLETE", CPU=True)        

            
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    ##############   ENDORHEIC BASIN SETUP  ##############
    """""""""""""""""""""""""""""""""""""""""""""""""""""" 
        
    run_endo_setup = hc_setup.config.getboolean("processes-DH-all", "run_endo_setup")
    if run_endo_setup:        
        
        """ setup basins """
        log_step("ENDORHIEC BASIN SETUP", sep=True)
        endo_subbasin_work_dir, endo_level = hc_setup.setup_DH_shapes(endo_level=12, use_ogr=True)
        hc_setup.setup_endo_basins(endo_subbasin_work_dir, endo_level)     

        endo_basin_paths = glob.glob(endo_subbasin_work_dir + '/**/basin_*.shp', recursive=True)
        n_endo_basins = 0
        for path in endo_basin_paths:
            if "buffer" not in path: 
                n_endo_basins += 1
        sink_points_shp = os.path.join(endo_subbasin_work_dir, 'sinks.shp')
        
        """ check that we have the expected number of endo basin shapefiles """
        check_endo_count = hc_setup.config.getboolean("processes-DH-all", "check_endo_count") 
        if check_endo_count:
            log_step("Check endo basins...")
            basins12_shp = os.path.join(hc_setup.shapes_DH_dir, 'basins_lvl12.shp')
            basins12_gdf = gpd.read_file(basins12_shp)      
            aoi_gdf = gpd.read_file(hc_setup.aoi_shpfile)

            basins12_within = basins12_gdf[basins12_gdf.within(aoi_gdf.geometry[0])]
            expected_endo = basins12_within[basins12_within["ENDO"] == 2]
            n_expected_endo = len(expected_endo)
            print("# expected endo basins:", n_expected_endo)
            print("# endo basins found:", n_endo_basins)
            assert n_expected_endo <= n_endo_basins, 'missing {0} endo basins'.format(n_expected_endo - n_endo_basins)
        log_step("Endorheic basins created/checked")          
        
        """ find sinks if there are endo basins but no master sink shapefile """        
        if (n_endo_basins > 0):
            overwrite_basin = hc_setup.config.getboolean("processes-DH-all", "overwrite_basin")
            if overwrite_basin or (not os.path.exists(sink_points_shp)):
            
                """ find sinks """
                sub_call, sub_log = hc_setup.build_sinks_call(pydem_path, comp_sys, 
                                                              nbasins=n_endo_basins, venv=venv)
                subprocess.call(sub_call, shell=True)    
                wait_4_job("Find Sinks", sub_log)
                log_step("Sink selection completed.")        

                """ build master sink gdf """
                sink_points_gdf, sink_points_shp = endo.build_master_sink_gdf(endo_subbasin_work_dir)
                log_step("Master sink shapefile created.")  


    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    ###########        DEPRESSION HANDLING       #########
    """"""""""""""""""""""""""""""""""""""""""""""""""""""   
    """ add basin outputs to config file"""
    overwrite_basin = hc_setup.config.getboolean("processes-DH-all", "overwrite_basin") 
    overwrite_coastal = hc_setup.config.getboolean("processes-DH-all", "overwrite_coastal") 
    overwrite_flatten = hc_setup.config.getboolean("processes-DH-all", "overwrite_flatten") 
    hc_setup.update_config_basin()
    
    """ Set up basin shape files"""    
    basin_work_dir, basin_levels = hc_setup.setup_DH_shapes(level='basin')   
    log_step("Basin shape files created/already exist")

    
    if dephandle_basins:
        log_step("BASIN SETUP", sep=True)          

        #----------------------------#
        ######   SETUP BASINS   ######
        #----------------------------#
        
        """ Set up basin directories"""
        run_basin_setup = hc_setup.config.getboolean("processes-DH-all", "run_basin_setup")
        if run_basin_setup:       
            log_step("Creating/checking basin directories".format(basin_work_dir))
            hc_setup.setup_DH_basins(basin_work_dir, basin_levels)
            log_step("Basin directories created/checked")
            hc_setup.build_DH_shpfile(basin_work_dir)
            log_step("project shapefile built")
            
            
        basin_dirs = glob.glob(os.path.join(basin_work_dir,'basin_*'))

        """ Remove flattened DEM files, if overwrite=True"""
        if overwrite_flatten: 
            nremove_flat = 0
            log_step('Removing flattened DEMs')
            for basin_dir in basin_dirs:
                fdem_path = os.path.join(basin_dir, 'DEMflatten.tif')
                if os.path.exists(fdem_path):
                    nremove_flat += 1
                    os.remove(fdem_path)
                maskdem_path = os.path.join(basin_dir, 'Water_flatten_mask.tif')
                if os.path.exists(maskdem_path):
                    os.remove(maskdem_path)

                all_other_tifs = glob.glob(os.path.join(basin_dir,'*.tif'))
                all_other_txt = glob.glob(os.path.join(basin_dir,'*.txt'))
                for rtif in all_other_tifs: 
                    os.remove(rtif)
                for rtxt in all_other_txt: 
                    os.remove(rtxt)
            log_step('{0} flattened DEMs deleted'.format(nremove_flat))
            
     
            
        """ Remove coastal DEM files, if overwrite=True"""
        if overwrite_coastal: 
            nremove_coast = 0
            log_step('Removing coastal DEMs')
            for basin_dir in basin_dirs:
                coast_txt = os.path.join(basin_dir, 'coast.txt') 
                if os.path.exists(coast_txt):
                    nremove_coast += 1
                    ctxt_path = os.path.join(basin_dir, 'DEM_completed.txt')
                    if os.path.exists(ctxt_path): os.remove(ctxt_path)
                    cdem_path = os.path.join(basin_dir, 'conditionedDEM.tif')
                    if os.path.exists(cdem_path): os.remove(cdem_path)
                    odem_path = os.path.join(basin_dir, 'overlapDEM.tif')
                    if os.path.exists(odem_path): os.remove(odem_path)
                    all_other_tifs = glob.glob(os.path.join(basin_dir,'*.tif'))
                    for rtif in all_other_tifs: 
                        os.remove(rtif)

            log_step('{0} coastal DEMs deleted'.format(nremove_coast))
            
            
        """ Remove conditioned DEM files, if overwrite=True"""
        if overwrite_basin: 
            nremove = 0
            log_step('Removing conditioned DEMs')
            for basin_dir in basin_dirs:
                cdem_path = os.path.join(basin_dir, 'conditionedDEM.tif')
                cdiff_path = os.path.join(basin_dir, 'conditionedDEM_DIFF.tif')
                ctxt_path = os.path.join(basin_dir, 'DEM_completed.txt')
                    
                if os.path.exists(cdem_path):
                    nremove += 1
                    os.remove(cdem_path)
                if os.path.exists(cdiff_path): os.remove(cdiff_path)
                if os.path.exists(ctxt_path): os.remove(ctxt_path)
            log_step('{0} conditioned DEMs deleted'.format(nremove))

        nbasins = len(glob.glob(os.path.join(basin_work_dir,'basin_*')))   
        change_basin_setup = hc_setup.config.getboolean("processes-DH-basin", "change_basin_setup")
        unfin_basinIDs = logt.find_unfinished_basins_IDs(basin_work_dir, tifname='DEM_completed.txt')
        n_unfin_basins = len(unfin_basinIDs)
        log_step('{0} of {1} basins remain to be processed'.format(n_unfin_basins, nbasins))
        
        if change_basin_setup:
            log_step("Change setup for unfinished basins...")
            hc_setup.change_basin_setup(unfin_basinIDs, basin_work_dir, basin_levels)
            log_step("new project shapefile built")
            hc_setup.build_DH_shpfile(basin_work_dir)
            log_step("new project shapefile built")
            nbasins = len(glob.glob(os.path.join(basin_work_dir,'basin_*')))   
            unfin_basinIDs = logt.find_unfinished_basins_IDs(basin_work_dir, tifname='DEM_completed.txt')
            print('# basins with different setup: {0}'.format(nbasins))
            

        #---------------------------------#
        ######   RIVER ENFORCEMENT   ######
        #---------------------------------# 
        log_step("RIVER ENFORCEMENT", sep=True)
        flatten_rivers = hc_setup.config.getboolean("processes-DH-basin", "flatten_water")
        if flatten_rivers:
            hc_setup.update_config_flattened()  
            unflat_basins = logt.find_unfinished_basins_IDs(basin_work_dir, tifname='DEMflatten.tif')
            n_unflat_basins = len(unflat_basins)  
            log_step('{0} of {1} basins remain to be flattened'.format(n_unflat_basins, nbasins))
            
            
            
            if n_unflat_basins > 0:
                sub_call, sub_log = hc_setup.build_flatten_rivers_call(pydem_path, comp_sys, 
                                                                       nbasins=n_unflat_basins, 
                                                                       venv=venv, iteration=1)
                subprocess.call(sub_call, shell=True)    
                wait_4_job("River Flattening pass 1/2", sub_log)
                
                
            """ CHECK FOR UNFINISHED BASINS, set overwrite==False """
            unflat_basins = logt.find_unfinished_basins_IDs(basin_work_dir, tifname='DEMflatten.tif')
            n_unflat_basins = len(unflat_basins)
            log_step('{0} of {1} basins remain to be flattened'.format(n_unflat_basins, nbasins))
            if overwrite_basin: #change to False for second pass
                hc_setup.config.set("processes-DH-all", "overwrite_flatten", False)
                with open(hc_setup.config_file_auto, 'w') as f:
                    hc_setup.config.write(f) 

            """Run second pass on fewer nodes for 12 hours"""
            if n_unflat_basins > 0:
                sub_call, sub_log = hc_setup.build_flatten_rivers_call(pydem_path, comp_sys, 
                                                                       nbasins=n_unflat_basins, 
                                                                       venv=venv, iteration=2)
                subprocess.call(sub_call, shell=True)    
                wait_4_job("River Flattening pass 2/2", sub_log)
            
            log_step("River Flattening completed!") 
                
            unflat_basins = logt.find_unfinished_basins_IDs(basin_work_dir, tifname='DEMflatten.tif')
            assert len(unflat_basins) == 0, 'missing Water_flatten_mask.tif files: {0}'.format(unflat_basins)
            
                        
            #--------------------------------#
            ####  MINIMIZE BASIN OVERLAPS ####
            #--------------------------------# 
            log_step("MINIMIZE BASIN OVERLAPS", sep=True)
            
            work_dir  = os.path.join(hc_setup.config.get("paths-DH", "output_dir"), hc_setup.projectname)
            basins_all_gdf = gpd.read_file(hc_setup.basins_all_shp)   
            basinIDs = basins_all_gdf['HYBAS_ID'].values
            unfin_overlap_basins = logt.find_unfinished_basins_IDs(basin_work_dir, tifname='overlapDEM.tif') 
            unfin_overlapmask_basins = logt.find_unfinished_basins_IDs(basin_work_dir,
                                                                       tifname='Water_flatten_mask_overlap.tif')
            unfin_overlap_basins = list(set(list(unfin_overlap_basins) + (unfin_overlapmask_basins)))
            n_unfin_overlap_basins = len(unfin_overlap_basins)
            log_step('{0} of {1} basins remain to be minimized'.format(n_unfin_overlap_basins, 
                                                                               nbasins))
                        
            if n_unfin_overlap_basins > 0:
                """create shape file for all coastal basin buffered outlines"""
                basin_buffer        = hc_setup.config.getfloat("parameters-DH-basin", "water_flatten_buffer")
                coastal_buffer      = hc_setup.config.getfloat("parameters-DH-basin", "coastal_buffer")
                basin_coastal_buffer   = basin_buffer + coastal_buffer
                
                overlap_basins_shp  = os.path.join(work_dir, "shape_files", 'overlap_basins.shp')
                
                if (not os.path.exists(overlap_basins_shp)) or change_basin_setup:
                    print_time('build overlap shapes', starttime)
                    overlap_basins_gdf = None
                    for basin_id in basinIDs:
                        outpath_basin = os.path.join(basin_work_dir, 'basin_{0}'.format(basin_id))
                        basin_gdf = basins_all_gdf.loc[basins_all_gdf['HYBAS_ID']==basin_id].copy()
                        if basin_gdf["COAST"].values[0] == 1:
                            basin_overlap_buffer = basin_coastal_buffer
                        else:
                            basin_overlap_buffer = basin_buffer

                        basin_buffer_shpfile = os.path.join(outpath_basin,'basin_{0}_{1}m_buffer.shp'.format(
                                                                    basin_id, np.int(basin_overlap_buffer)))
                        basin_buffer_gdf = gpd.read_file(basin_buffer_shpfile)   
                        if overlap_basins_gdf is None:
                            overlap_basins_gdf = basin_buffer_gdf.copy()
                        else:
                            overlap_basins_gdf = overlap_basins_gdf.append(basin_buffer_gdf)
                    overlap_basins_gdf.to_file(overlap_basins_shp)
                
                hc_setup.config.set('outputs', 'overlap_basins_shp', overlap_basins_shp)
                with open(hc_setup.config_file_auto, 'w') as f:
                    hc_setup.config.write(f) 

                sub_call, sub_log = hc_setup.build_minimize_call(pydem_path, comp_sys, 
                                                                       nbasins=n_unfin_overlap_basins, 
                                                                       venv=venv)
                subprocess.call(sub_call, shell=True)    
                wait_4_job("Minimize basin overlap", sub_log)
                

                unfin_overlap_basins = logt.find_unfinished_basins_IDs(basin_work_dir, tifname='overlapDEM.tif')
                n_unfin_overlap_basins = len(unfin_overlap_basins)
                assert n_unfin_overlap_basins == 0, 'missing overlapDEM.tif files for level 6: {0}'.format(
                    unfin_overlap_basins)
                

              
            build_flat_mosaic = hc_setup.config.getboolean("processes-DH-all", "build_flat_mosaic")
            if (not os.path.exists(hc_setup.flattened_mosaic) or (build_flat_mosaic)):
                log_step("BUILD WATER-FLATTENED MOSAIC", sep=True)
                #----------------------------#
                ######   BUILD MOSAIC   ######
                #----------------------------#
                log_step('Creating flattened mosaic - basins to mosaic: {0}'.format(nbasins)) 
                basin_paths =  glob.glob(os.path.join(basin_work_dir, 'basin_*'))               
                """Seperate out flattened basins"""
                f0paths  = []  # no stream burning at all
                f1paths  = []  # burned streams - no flattened rivers
                f2paths  = []  # flattened rivers
                cpaths   = []  # coastal basins
                islpaths = []  # island basins
                
                for bpath in basin_paths:
                    try:
                        island_txt = os.path.join(bpath, 'island_chain.txt')
                        if os.path.exists(island_txt): 
                            islpaths.append(bpath)
                            continue                            
                        coast_txt = os.path.join(bpath, 'coast.txt') 
                        if os.path.exists(coast_txt):
                            cpaths.append(bpath)
                            continue
                            
                        flatten_tif = os.path.join(bpath, 'Water_flatten_mask_overlap.tif') 
                        if os.path.exists(flatten_tif):
                            flatten_array = gdal_array.LoadFile(flatten_tif)
                            stream_ind = np.nonzero(flatten_array > 5)                  
                            river_ind = np.nonzero(flatten_array == 2) 
                            if len(river_ind[0] > 0): #river flattened
                                f2paths.append(bpath)
                            elif len(stream_ind[0] > 0): # just stream burned
                                f1paths.append(bpath)
                            else:
                                f0paths.append(bpath)

                    except Exception as e:
                        outst = "Exception occurred: path =  {0}: {1}".format(bpath, e)
                        traceback_output = traceback.format_exc()
                        print(outst)
                        print(traceback_output)
          
                log_step('   Putting {0} island basins on bottom of flattened mosaic'.format(len(islpaths)))
                log_step('   Putting {0} coastal basins on bottom of flattened mosaic'.format(len(cpaths)))
                log_step('   Putting {0} stream-burned basins in middle of mosaic'.format(len(f1paths)))
                log_step('   Putting {0} water-flattened basins on top of mosaic'.format(len(f2paths)))  
                
                """island, then coastal paths, then stream-burned, then flattened"""
                flatten_paths = islpaths
                flatten_paths.extend(cpaths)
                flatten_paths.extend(f0paths)
                flatten_paths.extend(f1paths)
                flatten_paths.extend(f2paths)
                print('   # flattened paths = {0}, # basin Paths = {1}'.format(len(flatten_paths),len(basin_paths)))
                assert len(flatten_paths) == len(basin_paths), 'missing basins for flattened VRT'

                """Input DEMS, then flattened DEMs, then overlap DEMs"""
                DEM_paths = [os.path.join(bp, 'DEM.tif') for bp in flatten_paths]
                fDEM_paths = [os.path.join(bp, 'DEMflatten.tif') for bp in flatten_paths]
                overlapDEM_paths = [os.path.join(bp, 'overlapDEM.tif') for bp in flatten_paths]
                log_step('   Putting {0} base DEMs on bottom of mosaic'.format(len(DEM_paths)))          
                DEM_paths.extend(fDEM_paths)
                DEM_paths.extend(overlapDEM_paths)            

                missing_flat = 0
                for dp in DEM_paths:    
                    if not os.path.exists(dp):
                        print(dp, ' NOT FOUND')
                        missing_flat += 1

                assert missing_flat == 0, 'missing DEM_flattened.tif files for level 6'        
                vrt_options = gdal.BuildVRTOptions(outputSRS="EPSG:4326", srcNodata=None, VRTNodata=None)
                vrt_tif = gdal.BuildVRT(hc_setup.flattened_mosaic, DEM_paths, options=vrt_options)
                vrt_tif = None   
                log_step("Water Flatten DEM mosaic COMPLETE") 

                water_mask_paths = [os.path.join(bp, 'Water_flatten_mask.tif') for bp in flatten_paths]
                water_mask_overlap_paths = [os.path.join(bp, 'Water_flatten_mask_overlap.tif') for bp in flatten_paths] 
                water_mask_paths.extend(water_mask_overlap_paths)
                
                missing_mask = 0
                for p in water_mask_paths:    
                    if not os.path.exists(p):
                        print(p, ' NOT FOUND')
                        missing_mask += 1
                assert missing_mask == 0, 'missing Water_flatten_mask.tif files for level 6' 
                
                """Pull resolution from DEM vrt and apply to watermask VRT"""
                flatDEM_vrt = gdal.Open(hc_setup.flattened_mosaic, gdal.GA_ReadOnly)
                flatDEM_geoT = flatDEM_vrt.GetGeoTransform()
                xshape_vrt = flatDEM_vrt.RasterXSize
                yshape_vrt = flatDEM_vrt.RasterYSize
                xmin_vrt = flatDEM_geoT[0]
                ymax_vrt = flatDEM_geoT[3]
                xres_vrt = flatDEM_geoT[1]
                yres_vrt = flatDEM_geoT[5]
                xres_vrt_abs = np.abs(xres_vrt)
                yres_vrt_abs = np.abs(yres_vrt)
                xmax_vrt = xmin_vrt + xres_vrt * xshape_vrt
                ymin_vrt = ymax_vrt + yres_vrt * yshape_vrt
                outputbounds = [xmin_vrt, ymin_vrt, xmax_vrt, ymax_vrt]
                flatDEM_vrt = None                
                print('   Flattened DEM vrt info: xres/yes = {0}/{1}  xshape/yshape = {2}/{3}'.format(
                                    xres_vrt, yres_vrt, xshape_vrt, yshape_vrt))
                
                vrt_options = gdal.BuildVRTOptions(outputSRS="EPSG:4326", srcNodata=0, VRTNodata=0,
                                                   resolution='user', outputBounds=outputbounds, 
                                                   xRes=xres_vrt_abs, yRes=yres_vrt_abs)
                
                vrt_tif = gdal.BuildVRT(hc_setup.flattened_mask_mosaic, water_mask_paths, options=vrt_options)
                vrt_tif = None  
                
                """Pull resolution from water mask vrt"""
                flatmask_vrt = gdal.Open(hc_setup.flattened_mask_mosaic, gdal.GA_ReadOnly)
                flatmask_geoT = flatmask_vrt.GetGeoTransform()
                xshape_vrt = flatmask_vrt.RasterXSize
                yshape_vrt = flatmask_vrt.RasterYSize
                xres_vrt = flatmask_geoT[1]
                yres_vrt = flatmask_geoT[5]
                flatmask_vrt = None
                print('   Flattened water mask vrt info: xres/yes = {0}/{1}  xshape/yshape = {2}/{3}'.format(
                                    xres_vrt, yres_vrt, xshape_vrt, yshape_vrt))
                
                log_step("Water flatten mask mosaic COMPLETE") 
           
        log_step("River Enforcement completed!") 

        #----------------------------#
        ######    RUN BASINS    ######
        #----------------------------# 
        log_step("RUN DEPRESSION HANDLING", sep=True)
        
        unfin_basinIDs = logt.find_unfinished_basins_IDs(basin_work_dir, tifname='DEM_completed.txt')
        n_unfin_basins = len(unfin_basinIDs)
        log_step('{0} of {1} basins remain to be processed'.format(n_unfin_basins, nbasins))
        
        """Run first pass on lots of nodes for 1 hour to solve most basins"""
        if n_unfin_basins > 0:
            sub_call, sub_log = hc_setup.build_basin_call(pydem_path, comp_sys, 
                                                          nbasins=n_unfin_basins, venv=venv, iteration=1)
            subprocess.call(sub_call, shell=True)    
            wait_4_job("Dep Handling lvl 6 pass 1/3", sub_log)
        log_step("Depression Handling level 6 (pass 1/3) completed!") 

        """ CHECK FOR UNFINISHED BASINS, set overwrite==False """
        unfin_basinIDs = logt.find_unfinished_basins_IDs(basin_work_dir, tifname='DEM_completed.txt')
        n_unfin_basins = len(unfin_basinIDs)
        if overwrite_basin: #change to False 
            hc_setup.config.set("processes-DH-all", "overwrite_basin", False)
            with open(hc_setup.config_file_auto, 'w') as f:
                hc_setup.config.write(f) 

        """Run second pass on fewer nodes for 2 hours"""
        if n_unfin_basins > 0:
            log_step('{0} of {1} basins remain to be processed'.format(n_unfin_basins, nbasins))
            sub_call, sub_log = hc_setup.build_basin_call(pydem_path, comp_sys, 
                                                          nbasins=n_unfin_basins, venv=venv, iteration=2)
            subprocess.call(sub_call, shell=True)    
            wait_4_job("Dep Handling lvl 6 pass 2/3", sub_log)
            log_step("Depression Handling level 6 (pass 2/3) completed!") 

        unfin_basinIDs = logt.find_unfinished_basins_IDs(basin_work_dir, tifname='DEM_completed.txt')
        n_unfin_basins = len(unfin_basinIDs)
        
        
        """Run second pass on fewer nodes for 2 hours"""
        if n_unfin_basins > 0:
            log_step('{0} of {1} basins remain to be processed'.format(n_unfin_basins, nbasins))
            sub_call, sub_log = hc_setup.build_basin_call(pydem_path, comp_sys, 
                                                          nbasins=n_unfin_basins, venv=venv, iteration=3)
            subprocess.call(sub_call, shell=True)    
            wait_4_job("Dep Handling lvl 6 pass 3/3", sub_log)
            log_step("Depression Handling level 6 (pass 3/3) completed!") 
        
        
        unfin_basinIDs_txt = logt.find_unfinished_basins_IDs(basin_work_dir, tifname='DEM_completed.txt')
        unfin_basinIDs = logt.find_unfinished_basins_IDs(basin_work_dir, tifname='conditionedDEM.tif')
        if (len(unfin_basinIDs) > 0) or (len(unfin_basinIDs_txt) > 0):
            log_step('Unfinished basins remain, exiting: ')
            log_step(unfin_basinIDs)
            log_step(unfin_basinIDs_txt)
            sys.exit()

  
        #----------------------------#
        ######   CHECK OUTPUTS  ######
        #----------------------------#  
        rewrite_DHlogs = hc_setup.config.getboolean("processes-DH-all", "rewrite_logs")
        if rewrite_DHlogs:
            """Set up post-run log files"""
            unfin_basins_txt = os.path.join(basin_work_dir, 'unfinished_basins.txt')
            unsolved_basins_txt = os.path.join(basin_work_dir, 'unsolved_basins.txt')
            diff_basins_txt = os.path.join(basin_work_dir, 'diffcheck_basins.txt')
            if os.path.exists(unfin_basins_txt): os.remove(unfin_basins_txt)   
            if os.path.exists(unsolved_basins_txt): os.remove(unsolved_basins_txt)   
            if os.path.exists(diff_basins_txt): os.remove(diff_basins_txt)  
                
            hc_setup.write_basin_logs(basin_work_dir, unfin_basins_txt,
                                    unsolved_basins_txt, diff_basins_txt) 
            
             
        #----------------------------#
        ######   BUILD MOSAIC   ######
        #----------------------------#           
        build_dh_mosaic = hc_setup.config.getboolean("processes-DH-all", "build_mosaic")
        if build_dh_mosaic:
            log_step("BUILD CONDITIONED MOSAIC", sep=True)
            log_step('Creating lvl 6 mosaic - basins to mosaic: {0}'.format(nbasins)) 
            basin_paths_all =  glob.glob(os.path.join(basin_work_dir, 'basin_*'))

            """Seperate out flattened basins"""
            fpaths = []  # flattened rivers
            f2paths = [] # burned streams - no flattened rivers
            coast_paths = [] # coastal basins
            for bpath in basin_paths_all:
                island_txt = os.path.join(bpath, 'island_chain.txt')
                if os.path.exists(island_txt): continue
                coast_file =  os.path.join(bpath, 'coast.txt')
                if os.path.exists(coast_file):
                    coast_paths.append(bpath)
                    continue
                flatten_tif = os.path.join(bpath, 'Water_flatten_mask_overlap.tif')                
                if os.path.exists(flatten_tif):
                    flatten_array = gdal_array.LoadFile(flatten_tif)
                    river_ind = np.nonzero(flatten_array == 1)                  
                    if len(river_ind[0] > 0):
                        fpaths.append(bpath)
                    else:
                        f2paths.append(bpath)                    

            """remove paths that have been flattened and append at the end"""
            log_step('   Putting {0} coastal basins on bottom of mosaic'.format(len(coast_paths)))
            log_step('   Putting {0} stream-burned basins in middle of mosaic'.format(len(f2paths)))
            log_step('   Putting {0} water-flattened basins top of mosaic'.format(len(fpaths)))
            basin_paths = [p for p in basin_paths_all if p not in coast_paths]
            basin_paths = [p for p in basin_paths if p not in fpaths]  
            basin_paths = [p for p in basin_paths if p not in f2paths] 
            basin_paths.extend(fpaths)
            basin_paths.extend(f2paths)
            
#             on_top_basins = [7070768280]
#             on_top_bpaths = [os.path.join(basin_work_dir, 'basin_{0}'.format(b)) for b in on_top_basins]
#             basin_paths.extend(on_top_bpaths)
            
            
            basin_paths = coast_paths + basin_paths
            assert len(basin_paths) >= len(basin_paths_all), 'missing basins for conditioned VRT'            
            basinDEM_paths = [os.path.join(sb, 'conditionedDEM.tif') for sb in basin_paths]   
                      
            vrt_options = gdal.BuildVRTOptions(outputSRS="EPSG:4326", srcNodata='nan', VRTNodata=None)
            vrt_tif = gdal.BuildVRT(hc_setup.basin_mosaic, basinDEM_paths, options=vrt_options)
            vrt_tif = None   
            log_step("Level 6 mosaic COMPLETE")        

            
     

    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    ###########       FLATTEN LARGE LAKES        #########
    """""""""""""""""""""""""""""""""""""""""""""""""""""" 
    reflatten_lakes = hc_setup.config.getboolean("processes-DH-basin", "reflatten_lakes")

    if reflatten_lakes:
        log_step("FLATTEN LARGE LAKES", sep=True)
        log_dir = os.path.join(hc_setup.config.get("paths-all", "output_log_dir"),'temp_logs')
        sub_log = os.path.join(log_dir, "FlattenLakes_{0}.txt".format(hc_setup.projectname))
        flatten_py = os.path.join(pydem_path, 'models', 'flatten_lakes.py')
        sub_call = 'aprun -n 1 -N 1 bwpy-environ python {0} -c {1} -l {2}'.format(
            flatten_py, hc_setup.config_file_auto, sub_log)        
        subprocess.call(sub_call, shell=True)  
        wait_4_job("Lake Flattening", sub_log)
        
      
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    ########      MANUAL MIN WATER VALUE SET      ########
    """"""""""""""""""""""""""""""""""""""""""""""""""""""         
    set_min_water_bool = hc_setup.config.getboolean("set-min-water", "set_min_water")
    mw_DEM_tifs = []
    if set_min_water_bool:
        basin_ids_mw = json.loads(hc_setup.config.get("set-min-water", "mw_basin_ids")) 
        basin_mw_vals = json.loads(hc_setup.config.get("set-min-water", "mw_values")) 
        for basin_id, basin_mw_val in zip(basin_ids_mw, basin_mw_vals):
            mw_tif = set_min_water(basin_id, basin_mw_val, basin_work_dir, starttime=starttime)
            mw_DEM_tifs.append(mw_tif)
            

    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    ########   MOSAIC BASINS - post lake flatten  ########
    """""""""""""""""""""""""""""""""""""""""""""""""""""" 
    lake_dem_dir = os.path.join(hc_setup.output_DH_path, "large_lakes", "DEMs")
    lake_paths =  glob.glob(os.path.join(lake_dem_dir, 'conditionedDEM_*'))
    lake_mosaic_path = (hc_setup.basin_mosaic).replace('.vrt', '_lakeflattened.vrt')     
    if (reflatten_lakes) or (set_min_water_bool):       
                     
        """check for unfinished basins"""
        unfin_basin_paths = logt.find_unfinished_basins_list(basin_work_dir) 
        assert len(unfin_basin_paths) == 0, "Unifinished basins remain... existing mosaic setup"
        basin_paths_all =  glob.glob(os.path.join(basin_work_dir, 'basin_*')) 
        log_step('Creating lake-flattened mosaic for {0} basins'.format(len(basin_paths_all)))

        """Seperate out flattened basins"""
        fpaths = []  # flattened rivers
        f2paths = [] # burned streams - no flattened rivers
        coast_paths = [] # coastal basins
        for bpath in basin_paths_all:
            island_txt = os.path.join(bpath, 'island_chain.txt')
            if os.path.exists(island_txt): continue
            
            coast_file =  os.path.join(bpath, 'coast.txt')
            if os.path.exists(coast_file):
                coast_paths.append(bpath) 
                continue
            flatten_tif = os.path.join(bpath, 'Water_flatten_mask_overlap.tif')                
            if os.path.exists(flatten_tif):
                flatten_array = gdal_array.LoadFile(flatten_tif)
                river_ind = np.nonzero(flatten_array == 1)                  
                if len(river_ind[0] > 0):
                    fpaths.append(bpath)
                else:
                    f2paths.append(bpath)                    

        """remove paths that have been flattened and append at the end"""
        log_step('   Putting {0} coastal basins on bottom of mosaic'.format(len(coast_paths)))
        log_step('   Putting {0} stream-burned basins in middle of mosaic'.format(len(f2paths)))
        log_step('   Putting {0} water-flattened basins top of mosaic'.format(len(fpaths)))
        basin_paths = [p for p in basin_paths_all if p not in coast_paths]
        basin_paths = [p for p in basin_paths if p not in fpaths]  
        basin_paths = [p for p in basin_paths if p not in f2paths] 
        basin_paths.extend(fpaths)    
        basin_paths.extend(f2paths) 
        
#         on_top_basins = [7070768280]
#         on_top_bpaths = [os.path.join(basin_work_dir, 'basin_{0}'.format(b)) for b in on_top_basins]
#         print(on_top_bpaths)
#         basin_paths.extend(on_top_bpaths)
        
        basin_paths = coast_paths + basin_paths
        assert len(basin_paths) >= len(basin_paths_all), 'missing basins for conditioned VRT'
        basinDEM_paths = [os.path.join(sb, 'conditionedDEM.tif') for sb in basin_paths]   

        """Add manually set rivers on top of mosaic"""
        if len(mw_DEM_tifs)>0:
            log_step('   Putting {0} manually-set rivers top of mosaic'.format(len(mw_DEM_tifs)))
            basinDEM_paths.extend(mw_DEM_tifs)
        
        """Add flattened lakes on top of mosaic"""
        if len(lake_paths)>0:
            basinDEM_paths.extend(lake_paths)
            
            
        """Add d8-adjusted basins (if any)""" 
        d8_basins = []
        #d8_basins = logt.find_finished_basins(basin_work_dir, dir_search='basin_*', 
        #                                          tifname="conditionedDEM_taudem_adjust.tif")
        #d8_DEM_paths = [os.path.join(b, "conditionedDEM_taudem_adjust.tif") for b in d8_basins]
        #basinDEM_paths.extend(d8_DEM_paths)


        """Build VRT"""
        vrt_tif = gdal.BuildVRT(lake_mosaic_path, basinDEM_paths, srcNodata='nan')
        vrt_tif = None             
        log_step("   VRT created for basins - {0} basins/{1} lakes/{2} taudem-adjusted basins used".format(
                    len(basin_paths), len(lake_paths), len(d8_basins)))


    
         
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    ##################   RUN TAUDEM  #####################
    """"""""""""""""""""""""""""""""""""""""""""""""""""""     
    if run_taudem:
        log_step("RUN TAUDEM", sep=True)
        """delete any existing taudem files"""
        taudem_dir = hc_setup.config.get("paths-taudem", "taudem_dir")
        taudem_projdir = os.path.join(taudem_dir, hc_setup.projectname)
        if os.path.exists(taudem_projdir):
            shutil.rmtree(taudem_projdir)
        os.makedirs(taudem_projdir)        
        
        """find final mosaic path"""
        if os.path.exists(lake_mosaic_path):
            mosaic_path = lake_mosaic_path 
            log_step("   Using lake flattened VRT for TauDEM")
        else:
            mosaic_path = hc_setup.basin_mosaic    
            log_step("   Using basin VRT for TauDEM")
                
        """set nodata value and copy vrt to taudem folder"""
        taudem_inDEM = os.path.join(taudem_projdir, "dem.tif")
        set_nodata.set_vrt_nodata(mosaic_path, taudem_inDEM, dstNodata=-9999)

        """set up taudem run"""
        taudem_max_alloc = hc_setup.config.getboolean("paths-taudem", "max_allocation")
        sub_call, sub_log = hc_setup.build_taudem_call(taudem_inDEM, pydem_path, comp_sys, 
                                                       venv=venv, max_allocation=taudem_max_alloc)
        subprocess.call(sub_call, shell=True)    
        wait_4_job("TauDEM", sub_log)
        log_step("TauDEM run completed!")

        
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    ################## CHECK TAUDEM  #####################
    """"""""""""""""""""""""""""""""""""""""""""""""""""""     
    if check_taudem:       
        log_step("CHECK TAUDEM OUTPUTS", sep=True)
        nbasins = len(glob.glob(os.path.join(basin_work_dir,'basin_*')))
        
        """delete existing taudem adjust files"""
        old_d8_basins = logt.find_finished_basins(basin_work_dir, dir_search='basin_*', 
                                                  tifname="conditionedDEM_taudem_adjust.tif")
        for basin in old_d8_basins:
            old_tif = os.path.join(basin, 'conditionedDEM_taudem_adjust.tif')
            os.remove(old_tif)
        
        sub_call, sub_log = hc_setup.build_taudem_check_call(pydem_path, comp_sys, 
                                                             nbasins=nbasins, venv=venv)
        subprocess.call(sub_call, shell=True)    
        wait_4_job("TauDEM check", sub_log)
        log_step("TauDEM check run completed!")
        
        d8_basins = logt.find_finished_basins(basin_work_dir, dir_search='basin_*', 
                                                  tifname="conditionedDEM_taudem_adjust.tif")
        d8_basinsIDs = [os.path.splitext(os.path.basename(b))[0].
                        replace('basin_', '') for b in d8_basins]
        
        log_step("{0} problem d8 flow direction basins found: {1}".format(len(d8_basinsIDs), d8_basinsIDs))
                    
        if len(d8_basins) > 0:
            """"""""""""""""""""""""""""""""""""""""""""""""""""""
            ########   MOSAIC BASINS - post taudem check ########
            """"""""""""""""""""""""""""""""""""""""""""""""""""""
            
            d8_DEM_paths = [os.path.join(b, "conditionedDEM_taudem_adjust.tif") for b in d8_basins]
                        
            lake_dem_dir = os.path.join(hc_setup.output_DH_path, "large_lakes", "DEMs")
            lake_paths =  glob.glob(os.path.join(lake_dem_dir, 'conditionedDEM_*'))
            taudem_mosaic_path = (hc_setup.basin_mosaic).replace('.vrt', '_post_taudem.vrt')     

            """check for unfinished basins"""
            unfin_basin_paths = logt.find_unfinished_basins_list(basin_work_dir) 
            assert len(unfin_basin_paths) == 0, "Unifinished basins remain... existing mosaic setup"
            basin_paths_all =  glob.glob(os.path.join(basin_work_dir, 'basin_*')) 
            log_step('Creating flattened mosaic for {0} basins'.format(len(basin_paths_all)))

            """Seperate out flattened basins"""
            fpaths = []  # flattened rivers
            f2paths = [] # burned streams - no flattened rivers
            coast_paths = [] # coastal basins
            for bpath in basin_paths_all:
                island_txt = os.path.join(bpath, 'island_chain.txt')
                if os.path.exists(island_txt): continue

                coast_file =  os.path.join(bpath, 'coast.txt')
                if os.path.exists(coast_file):
                    coast_paths.append(bpath) 
                    continue
                flatten_tif = os.path.join(bpath, 'Water_flatten_mask_overlap.tif')                
                if os.path.exists(flatten_tif):
                    flatten_array = gdal_array.LoadFile(flatten_tif)
                    river_ind = np.nonzero(flatten_array == 1)                  
                    if len(river_ind[0] > 0):
                        fpaths.append(bpath)
                    else:
                        f2paths.append(bpath)                    

            """remove paths that have been flattened and append at the end"""
            log_step('   Putting {0} coastal basins on bottom of mosaic'.format(len(coast_paths)))
            log_step('   Putting {0} stream-burned basins in middle of mosaic'.format(len(f2paths)))
            log_step('   Putting {0} water-flattened basins top of mosaic'.format(len(fpaths)))
            basin_paths = [p for p in basin_paths_all if p not in coast_paths]
            basin_paths = [p for p in basin_paths if p not in fpaths]  
            basin_paths = [p for p in basin_paths if p not in f2paths] 
            basin_paths.extend(fpaths)    
            basin_paths.extend(f2paths) 
            basin_paths = coast_paths + basin_paths
            assert len(basin_paths) == len(basin_paths_all), 'missing basins for conditioned VRT'
            basinDEM_paths = [os.path.join(sb, 'conditionedDEM.tif') for sb in basin_paths]     
            
            if len(lake_paths) > 0: 
                log_step('   Putting {0} flattened lakes on top of mosaic'.format(len(lake_paths)))
                """Add flattened lakes on top of mosaic"""
                basinDEM_paths.extend(lake_paths)

            """Add manually set rivers on top of mosaic"""
            if len(mw_DEM_tifs)>0:
                log_step('   Putting {0} manually-set rivers top of mosaic'.format(len(mw_DEM_tifs)))
                basinDEM_paths.extend(mw_DEM_tifs)


            """Add taudem-adjusted basins on top of mosaic"""
            basinDEM_paths.extend(d8_DEM_paths)

            """Build VRT"""
            vrt_tif = gdal.BuildVRT(taudem_mosaic_path, basinDEM_paths, srcNodata='nan')
            vrt_tif = None             
            log_step("   VRT created for basins - {0} tifs/{1} taudem-adjusted basins".format(
                len(basinDEM_paths), len(d8_basins)))
            
            
            
            """"""""""""""""""""""""""""""""""""""""""""""""""""""
            ################   RE-RUN TAUDEM  ####################
            """"""""""""""""""""""""""""""""""""""""""""""""""""""     

            """delete any existing taudem files"""
            taudem_dir = hc_setup.config.get("paths-taudem", "taudem_dir")
            taudem_projdir = os.path.join(taudem_dir, hc_setup.projectname)
            if os.path.exists(taudem_projdir):
                shutil.rmtree(taudem_projdir)
            os.makedirs(taudem_projdir)        

            mosaic_path = taudem_mosaic_path  

            """set nodata value and copy vrt to taudem folder"""
            taudem_inDEM = os.path.join(taudem_projdir, "dem.tif")
            set_nodata.set_vrt_nodata(mosaic_path, taudem_inDEM, dstNodata=-9999)

            """set up taudem run"""
            sub_call, sub_log = hc_setup.build_taudem_call(taudem_inDEM, pydem_path, comp_sys, venv=venv)
            subprocess.call(sub_call, shell=True)    
            wait_4_job("TauDEM", sub_log)
            log_step("TauDEM re-run completed!")
            

            """"""""""""""""""""""""""""""""""""""""""""""""""""""
            ################   RE-CHECK TAUDEM  ##################
            """"""""""""""""""""""""""""""""""""""""""""""""""""""
            
            
            log_step("CHECK TAUDEM RE-RUN OUTPUTS", sep=True)
            nbasins = len(glob.glob(os.path.join(basin_work_dir,'basin_*')))

            """delete existing taudem adjust files"""
            old_d8_basins = logt.find_finished_basins(basin_work_dir, dir_search='basin_*', 
                                                      tifname="conditionedDEM_taudem_adjust.tif")
            for basin in old_d8_basins:
                old_tif = os.path.join(basin, 'conditionedDEM_taudem_adjust.tif')
                os.remove(old_tif)

            sub_call, sub_log = hc_setup.build_taudem_check_call(pydem_path, comp_sys, 
                                                                 nbasins=nbasins, venv=venv)
            subprocess.call(sub_call, shell=True)    
            wait_4_job("TauDEM check", sub_log)
            log_step("TauDEM check run completed!")

            d8_basins = logt.find_finished_basins(basin_work_dir, dir_search='basin_*', 
                                                      tifname="conditionedDEM_taudem_adjust.tif")
            d8_basinsIDs = [os.path.splitext(os.path.basename(b))[0].
                            replace('basin_', '') for b in d8_basins]

            log_step("{0} problem d8 flow direction basins found after re-run: {1}".format(
                len(d8_basinsIDs), d8_basinsIDs))

                    
    log_step('All DONE!!', sep=True)




