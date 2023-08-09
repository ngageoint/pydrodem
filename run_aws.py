#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 10 2021

@author: Kimberly McCormack

Last edited on: 11/12/2021

Top level script to run entire hydro-conditioning process

"""

import argparse
import shutil
import glob
import time
import subprocess
import sys
import os
import warnings
warnings.simplefilter('ignore')
import geopandas as gpd
from osgeo import gdal


from post_processing import set_nodata
from setup_aws import PydemSetup
from tools import endo
from tools.print import print_time, log_time
import tools.convert as pyct
import tools.depressions as dep
import tools.log as logt
import tools.aws as aws


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
starttime = time.time()


def log_step(step_str, sep=False):
    log_str = log_time(step_str, starttime, sep=sep)
    with open(hc_setup.log_file, 'a') as f:
        f.write(log_str)
    print(log_str)


def wait_4_job(description, output_log):
    min_wait = 0
    while not os.path.exists(output_log):
        if min_wait == 0:
            print("Waiting for {0} to finish...".format(description), end="")
        else:
            print(".", end="")
        sys.stdout.flush()
        time.sleep(30)
        min_wait += 1
    print('')


def build_vrt_local(vrt_out, work_dir, tifname,
                    secondtifname=None, secondtif_list=None,
                    outputSRS="EPSG:4326",
                    srcnodata=None, dstnodata=-9999):

    output_paths = glob.glob(os.path.join(work_dir, '*'))
    init_paths = [os.path.join(p, f'{tifname}.tif') for p in output_paths]

    if secondtifname is not None:
        second_paths = [os.path.join(
            p, f'{secondtifname}.tif') for p in output_paths]
        init_paths.extend(second_paths)

    if secondtif_list is not None:
        init_paths.extend(secondtif_list)

    vrt_options = gdal.BuildVRTOptions(outputSRS=outputSRS,
                                       srcNodata=srcnodata, VRTNodata=dstnodata)

    vrt = gdal.BuildVRT(vrt_out, init_paths, options=vrt_options)
    vrt = None


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
    if args['virtualenv'] is not None:
        venv = os.path.abspath(args['virtualenv'])
    comp_sys = args['system']

    """ Initialize setup class, setup region, outputs, and logging file """
    print('Initializing setup class')
    hc_setup = PydemSetup(config_file)

    print_time("Setup region/log file/output directories....", starttime)
    hc_setup.setup_region()
    print_time("   region created ....", starttime)
    hc_setup.start_log_file()
    hc_setup.setup_outputs()
    pydem_path = os.path.dirname(os.path.abspath(__file__))

    """Extract which processes to run"""
    dem_noise_remove = hc_setup.config.getboolean(
        "processes-all", "dem_noise_removal")
    dephandle = hc_setup.config.getboolean("processes-all", "dephandle")
    run_taudem = hc_setup.config.getboolean("processes-all", "taudem")
    check_taudem = hc_setup.config.getboolean("processes-all", "check_taudem")

    log_step("Project Name: {0}".format(hc_setup.projectname), sep=True)
    tile_names = hc_setup.tile_names
    tile_buffer_names = hc_setup.tile_buffer_names
    overwrite_region = hc_setup.config.getboolean("region", "overwrite")  
    
    # Build FABDEM mosaic
    FAB_mosaic_vrt = os.path.join(hc_setup.output_FABmosaic)
    if (not os.path.exists(FAB_mosaic_vrt)) or (overwrite_region):
        print("Building FABDEM mosaic for",
              len(tile_buffer_names), " tiles")
        fab_bucket = hc_setup.config.get("paths", "FAB_bucket")
        fab_prefix = hc_setup.config.get("paths", "FAB_prefix")

        FAB_form = "TILEID_FABDEM_V1-0.tif"
        aws.build_vrt_s3_tilelist(fab_bucket, fab_prefix,
                                  FAB_mosaic_vrt, tile_buffer_names,
                                  path_form=FAB_form)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    #####  RUN DTM model (veg removal and smoothing) #####
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    overwrite_DTM = hc_setup.config.getboolean(
        "processes-noise-removal", "overwrite")
    base_unit = hc_setup.config.get("region", "base_unit")

    if dem_noise_remove:
        log_step("CREATE DEM-NR FROM DSM", sep=True)

        hc_setup.setup_noise_removal()

        #-------------------------------#
        ######   RUN DEM-NR MODEL  ######
        #-------------------------------#
        finished_cells = logt.find_finished_tiles_list(
            hc_setup.output_DTM_path, dir_search='*', tifname='DEM-NR.tif')

        if (len(finished_cells) < len(hc_setup.tile_names)) or (overwrite_DTM):
            sub_call, sub_log = hc_setup.build_noise_remove_call(
                pydem_path, comp_sys, venv=venv)
            subprocess.call(sub_call, shell=True)
            #wait_4_job("DEM-NR", sub_log)

        log_step("DEM-NR completed!")
        
                

        #----------------------------#
        ######   CHECK OUTPUTS  ######
        #----------------------------#
        """Check all tiles were correctly processed"""
        input_ids = open(hc_setup.tiles_txt).read().splitlines()        
        unfinished_tiles = logt.find_unfinished_ids(hc_setup.output_DTM_path,
                                             unit=base_unit,
                                             id_list=input_ids,
                                             tifname='DEM-NR.tif')
        # unfinished_tiles = logt.find_unfinished_tiles_list(
            # hc_setup.output_DTM_path, dir_search='*', tifname='DEM-NR.tif')
        log_step("All cells checked: {0} cells unfinished".format(
            len(unfinished_tiles)))

        if len(unfinished_tiles) > 0:
            print(unfinished_tiles)
            quit()

        #----------------------------#
        ######   BUILD MOSAIC   ######
        #----------------------------#
        DTM_mosaic_vrt = os.path.join(hc_setup.output_DTMmosaic)
        ntiles = len(hc_setup.tile_names)
        log_step(f'Creating mosaic - number of tiles to mosaic = {ntiles}')
        
        output_paths = [glob.glob(os.path.join(hc_setup.output_DTM_path,
                        f'*{i}*'))[0] for i in input_ids]
        #output_paths = glob.glob(os.path.join(hc_setup.output_DTM_path, '*'))
        DTM_paths = [os.path.join(p, 'DEM-NR.tif') for p in output_paths]
        vrt_options = gdal.BuildVRTOptions(outputSRS="EPSG:4326",
                                           srcNodata=None, VRTNodata=None)
        vrt_tif = gdal.BuildVRT(hc_setup.output_DTMmosaic, DTM_paths,
                                options=vrt_options)
        vrt_tif = None

        """ Build EDM mosaic """
        EDM_mosaic_vrt = os.path.join(hc_setup.output_EDMmosaic)
        EDM_paths = [os.path.join(p, 'EDM.tif') for p in output_paths]
        vrt_tif = gdal.BuildVRT(EDM_mosaic_vrt, EDM_paths)
        vrt_tif = None

        log_step("DEM-NR mosaic COMPLETE")

    else:  # build DTM and data mask mosaic

        DEM_NR_mosaic_vrt = os.path.join(hc_setup.output_DTMmosaic)
        if (not os.path.exists(DEM_NR_mosaic_vrt)) or (overwrite_region):
            print("Building DEM-NR mosaic for",
                  len(tile_buffer_names), " tiles")
            elev_bucket = hc_setup.config.get("paths", "DEM_bucket")
            elev_prefix = hc_setup.config.get("paths", "DEM_prefix")

            DEM_form = "TILEID/U_LIMDIS_TDX_DEM-NR_TILEID_01.tif"
            aws.build_vrt_s3_tilelist(elev_bucket, elev_prefix,
                                      DEM_NR_mosaic_vrt, tile_buffer_names,
                                      searchtxt='DEM', path_form=DEM_form)

        EDM_mosaic_vrt = os.path.join(hc_setup.output_EDMmosaic)
        if (not os.path.exists(EDM_mosaic_vrt)) or (overwrite_region):
            print("Building EDM mosaic for", len(tile_buffer_names), " tiles")
            edm_bucket = hc_setup.config.get("paths", "EDM_bucket")
            edm_prefix = hc_setup.config.get("paths", "EDM_prefix")

            EDM_form = "TDT_TILEID_02/AUXFILES/TDT_TILEID_02_EDM.tif"
            aws.build_vrt_s3_tilelist(edm_bucket, edm_prefix,
                                      EDM_mosaic_vrt, tile_buffer_names,
                                      searchtxt='EDM', path_form=EDM_form)



    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    ##############   ENDORHEIC BASIN SETUP  ##############
    """"""""""""""""""""""""""""""""""""""""""""""""""""""

    run_endo = hc_setup.config.getboolean(
        "processes-depression-handling", "enforce_endo")
    if run_endo:

        """ setup basins """
        log_step("ENDORHIEC BASIN SETUP", sep=True)
        endo_subbasin_work_dir, endo_level = hc_setup.clip_basins(
            endo_level=12)
        hc_setup.setup_endo_basins(endo_subbasin_work_dir, endo_level)

        endo_basin_paths = glob.glob(
            endo_subbasin_work_dir + '/**/basin_*.shp', recursive=True)
        n_endo_basins = 0
        for path in endo_basin_paths:
            if "buffer" not in path:
                n_endo_basins += 1
        sink_points_shp = os.path.join(endo_subbasin_work_dir, 'sinks.shp')

        """ check that we have the expected number of endo basin shapefiles """
        check_endo_count = hc_setup.config.getboolean(
            "processes-depression-handling", "check_endo_count")
        if check_endo_count:
            log_step("Check endo basins...")
            basins12_shp = os.path.join(
                hc_setup.shapes_DH_dir, 'basins_lvl12.shp')
            basins12_gdf = gpd.read_file(basins12_shp)
            aoi_gdf = gpd.read_file(hc_setup.aoi_shpfile)

            basins12_within = basins12_gdf[basins12_gdf.intersects(
                aoi_gdf.geometry[0])]
            expected_endo = basins12_within[basins12_within["ENDO"] == 2]

            n_expected_endo = len(expected_endo)
            print("# expected endo basins:", n_expected_endo)
            print("# endo basins found:", n_endo_basins)
            assert n_expected_endo <= n_endo_basins, 'missing {0} endo basins'.format(
                n_expected_endo - n_endo_basins)
        log_step("Endorheic basins created/checked")

        """ find sinks if there are endo basins but no master sink shapefile """
        if (n_endo_basins > 0):
            overwrite_endo = hc_setup.config.getboolean(
                "processes-depression-handling", "overwrite")
            if overwrite_endo or (not os.path.exists(sink_points_shp)):

                """ find sinks """
                sub_call, sub_log = hc_setup.build_sinks_call(pydem_path, comp_sys,
                                                              nbasins=n_endo_basins, venv=venv)
                subprocess.call(sub_call, shell=True)
                log_step("Sink selection completed.")
                
                input_sinks_shp = hc_setup.config.get("paths", "endo_sinks_shp")
                if len(input_sinks_shp) < 3:
                    input_sinks_shp = None


                """ build master sink gdf """
                sink_points_gdf, sink_points_shp = endo.build_master_sink_gdf(
                    endo_subbasin_work_dir, input_sinks_shp=input_sinks_shp)
                log_step("Master sink shapefile created.")

    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    ###########        DEPRESSION HANDLING       #########
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    """ add basin outputs to config file"""
    overwrite_DH = hc_setup.config.getboolean(
        "processes-depression-handling", "overwrite")
    

    """ Set up basin/tile shape files"""
    if base_unit == 'basin':
        work_dir, basin_levels = hc_setup.clip_basins()
        log_step("Basin shape files created/already exist")

        """ Set up basin directories"""
        run_basin_setup = hc_setup.config.getboolean(
            "processes-depression-handling", "run_basin_setup")
        if run_basin_setup:
            log_step("Creating/checking basin directories")
            hc_setup.setup_DH_basins(work_dir, basin_levels)
            log_step("Basin directories created")
            hc_setup.build_DH_shpfile(work_dir)
            log_step("project shapefile built")

    elif base_unit == 'tile':
        work_dir = os.path.join(hc_setup.output_DH_path, 'tiles')

        hc_setup.setup_tiles(work_dir)
        log_step("Tile directories created")

    hc_setup.update_config_DH(work_dir)

    input_ids = logt.find_all_ids(work_dir, unit=base_unit,
                                  dir_search=f'{base_unit}_',)
    nunits = len(input_ids)

    if dephandle:

        #---------------------------------#
        ######   RIVER ENFORCEMENT   ######
        #---------------------------------#
        log_step("RIVER ENFORCEMENT", sep=True)
        burn_rivers = hc_setup.config.getboolean(
            "processes-depression-handling", "burn_rivers")
        if burn_rivers:
            hc_setup.update_config_water()
            if not overwrite_DH:
                unfin_ids = logt.find_unfinished_ids(work_dir, unit=base_unit,
                                                     dir_search=f'{base_unit}_',
                                                     tifname='DEMburned.tif')
                n_unfin_ids = len(unfin_ids)
            else:
                n_unfin_ids = nunits

            log_step(
                f"{n_unfin_ids} of {nunits} {base_unit} remain to be enforced")

            if n_unfin_ids > 0:
                sub_call, sub_log = hc_setup.build_burn_rivers_call(
                    pydem_path, comp_sys,
                    nbasins=n_unfin_ids,
                    venv=venv)

                subprocess.call(sub_call, shell=True)

            unfin_ids = logt.find_unfinished_ids(work_dir, unit=base_unit,
                                                 dir_search=f'{base_unit}_',
                                                 tifname='DEMburned.tif')
            assert len(
                unfin_ids) == 0, f'missing DEMburned.tif files: {unfin_ids}'
            log_step("River Enforcement completed!")
            
            
            ##--------------------------------#
            #####  MINIMIZE BASIN OVERLAPS ####
            ##--------------------------------#
            log_step("MINIMIZE BASIN OVERLAPS", sep=True)

            if not overwrite_DH:
                unfin_ids = logt.find_unfinished_ids(work_dir, unit=base_unit,
                                                     dir_search=f'{base_unit}_',
                                                     tifname='overlap.txt')
                n_unfin_ids = len(unfin_ids)
            else:
                n_unfin_ids = nunits
            log_step(
                f"{n_unfin_ids} of {nunits} {base_unit} remain to have overlaps minimized")

            if n_unfin_ids > 0:
                """create shape file for all proccessing unit buffered outlines"""
                overlap_shp = os.path.join(hc_setup.shapes_dir, 'overlap.shp')
                if (not os.path.exists(overlap_shp) or (overwrite_region)):
                    print_time('build overlap shapes', starttime)
                    overlap_gdf = None
                    for input_id in input_ids:
                        outpath = os.path.join(
                            work_dir, f'{base_unit}_{input_id}')
                        buffer_shpfile = os.path.join(
                            outpath, f'{base_unit}_{input_id}_buffer.shp')
                        buffer_gdf = gpd.read_file(buffer_shpfile)
                        buffer_gdf['input_id'] = input_id
                        if overlap_gdf is None:
                            overlap_gdf = buffer_gdf.copy()
                        else:
                            overlap_gdf = overlap_gdf.append(buffer_gdf)
                    overlap_gdf.to_file(overlap_shp)

                hc_setup.config.set('outputs', 'overlap_shp', overlap_shp)
                with open(hc_setup.config_file_auto, 'w') as f:
                    hc_setup.config.write(f)

                sub_call, sub_log = hc_setup.build_minimize_call(pydem_path, comp_sys,
                                                                 nbasins=n_unfin_ids,
                                                                 venv=venv)
                subprocess.call(sub_call, shell=True)

            unfin_ids = logt.find_unfinished_ids(work_dir, unit=base_unit,
                                                 dir_search=f'{base_unit}_',
                                                 tifname='overlap.txt')
            assert len(
                unfin_ids) == 0, f'missing overlapDEM.tif files: {unfin_ids}'
            log_step("Overlap Minimization completed!")
            

            #----------------------------#
            ######   BUILD MOSAICS  ######
            #----------------------------#
            build_burn_mosaic = hc_setup.config.getboolean("processes-depression-handling",
                                                           "build_mosaic")

            if (not os.path.exists(hc_setup.burned_mosaic) or (build_burn_mosaic)):
                build_vrt_local(hc_setup.burned_mosaic, work_dir,
                                "DEMburned")

            if (not os.path.exists(hc_setup.water_mask_mosaic) or (build_burn_mosaic)):
                build_vrt_local(hc_setup.water_mask_mosaic, work_dir,
                                "Water_mask")

            log_step('River burned mosaics complete')


        #-----------------------------------------#
        ######    RUN DEPRESSION HANDLING    ######
        #-----------------------------------------#
        log_step("RUN DEPRESSION HANDLING", sep=True)

        if not overwrite_DH:
            unfin_ids = logt.find_unfinished_ids(work_dir, unit=base_unit,
                                                 dir_search=f'{base_unit}_',
                                                 tifname='DEMcompleted.txt')
            n_unfin_ids = len(unfin_ids)
        else:
            n_unfin_ids = nunits

        log_step(f"{n_unfin_ids} of {nunits} {base_unit} remain to be processed")

        if n_unfin_ids > 0:
            sub_call, sub_log = hc_setup.build_dep_handle_call(
                pydem_path, comp_sys,
                nbasins=n_unfin_ids,
                venv=venv)

            subprocess.call(sub_call, shell=True)
        log_step("Depression Handling completed!")
        
        
        quit()



        #----------------------------#
        ######   BUILD MOSAIC   ######
        #----------------------------#
        build_dh_mosaic = hc_setup.config.getboolean(
            "processes-depression-handling", "build_mosaic")

        if (not os.path.exists(hc_setup.dh_mosaic) or (build_dh_mosaic)):
            build_vrt_local(hc_setup.dh_mosaic, work_dir,
                            "conditionedDEM")

        log_step('Fully Conditioned mosaic complete')

        """"""""""""""""""""""""""""""""""""""""""""""""""""""
        ###########       FLATTEN LARGE LAKES        #########
        """"""""""""""""""""""""""""""""""""""""""""""""""""""
        reflatten_lakes = hc_setup.config.getboolean("processes-depression-handling",
                                                     "reflatten_lakes")

        if reflatten_lakes:
            log_step("FLATTEN LARGE LAKES", sep=True)
            log_dir = os.path.join(hc_setup.config.get(
                "paths", "output_log_dir"), 'temp_logs')
            sub_log = os.path.join(
                log_dir, "FlattenLakes_{0}.txt".format(hc_setup.projectname))
            flatten_py = os.path.join(pydem_path, 'models', 'flatten_lakes.py')
            sub_call = f'python {flatten_py} -c {hc_setup.config_file_auto} -l {sub_log}'
            subprocess.call(sub_call, shell=True)

            """"""""""""""""""""""""""""""""""""""""""""""""""""""
            ########   MOSAIC BASINS - post lake flatten  ########
            """"""""""""""""""""""""""""""""""""""""""""""""""""""
            lake_dem_dir = os.path.join(
                hc_setup.output_DH_path, "large_lakes", "DEMs")
            lake_paths = glob.glob(os.path.join(
                lake_dem_dir, 'conditionedDEM_*'))
            lake_mosaic_path = (hc_setup.dh_mosaic).replace(
                '.vrt', '_lakeflattened.vrt')

            build_vrt_local(hc_setup.burned_mosaic, work_dir,
                            "conditionedDEM", secondtif_list=lake_paths)

        # """"""""""""""""""""""""""""""""""""""""""""""""""""""
        # ##################   WBT TEST    #####################
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""
        WBT_test = False
        if WBT_test:
            """ Path to White Box Tools"""
            WBT_path = os.path.join(hc_setup.config.get("paths", "WBT_path"))
            """Load WhiteBox Tools executable"""
            sys.path.append(WBT_path)
            from whitebox_tools import WhiteboxTools
            wbt = WhiteboxTools()
            wbt.set_verbose_mode(False)
            wbt.set_whitebox_dir(WBT_path)  # set location of WBT executable

            log_step('Convert mosaic to geotiff')
            mosaic_full = hc_setup.dh_mosaic.replace('.vrt', '.tif')
            # if not os.path.exists(mosaic_full):
            pyct.vrt2gtiff(hc_setup.dh_mosaic, mosaic_full)

            log_step('Fix flats')
            fill_method = 'WL'
            fill_tif = mosaic_full.replace('.tif', f'_{fill_method}.tif')
            # if not os.path.exists(fill_tif):
            dep.fill_pits(mosaic_full, wbt, fill_method=fill_method, nodataval=-9999,
                          fixflats=True, flat_increment=0.00001,
                          save_tif=True, downcast=False)

            log_step('Flow direction')
            d8_tif = mosaic_full.replace(
                '.tif', f'_{fill_method}_flowdir_wbt.tif')
            wbt.d8_pointer(fill_tif, d8_tif)

            log_step('Flow Accumulation')
            acc_tif = mosaic_full.replace(
                '.tif', f'_{fill_method}_flowacc_wbt.tif')
            wbt.d8_flow_accumulation(d8_tif, acc_tif, pntr=True)

    # """"""""""""""""""""""""""""""""""""""""""""""""""""""
    # ##################   RUN TAUDEM  #####################
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""

    if run_taudem:

        overwrite = False
        log_step("RUN TAUDEM", sep=True)
        taudem_path = "C:\Program Files\TauDEM\TauDEM5Exe"
        mpi_path = "C:\Program Files\Microsoft MPI\Bin\mpiexec"
        nnodes = 12

        """delete any existing taudem files"""
        taudem_dir = hc_setup.config.get("paths-taudem", "taudem_dir")
        taudem_projdir = os.path.join(taudem_dir, hc_setup.projectname)
        if overwrite:
            if os.path.exists(taudem_projdir):
                shutil.rmtree(taudem_projdir)
        if not os.path.exists(taudem_projdir):
            os.makedirs(taudem_projdir)

        """find final mosaic path"""
        mosaic_path = hc_setup.dh_mosaic

        """set nodata value and copy vrt to taudem folder"""
        taudem_inDEM = os.path.join(taudem_projdir, "dem.tif")
        set_nodata.set_vrt_nodata(mosaic_path, taudem_inDEM, dstNodata=-9999)

        """set up taudem run"""
        tau_fill_dem = os.path.join(taudem_projdir, "dem_tau_filled.tif")
        tau_d8 = os.path.join(taudem_projdir, "dem_tau_flowdir.tif")
        tau_slope = os.path.join(taudem_projdir, "dem_tau_slope.tif")
        tau_acc = os.path.join(taudem_projdir, "dem_tau_flowacc.tif")

        # Pit Filling
        log_step("   pit filling")
        taudem_pitremove = os.path.join(taudem_path, 'PitRemove.exe')
        taudem_log = os.path.join(taudem_projdir, "pitremove.log")
        cmd_lst = [f"{mpi_path}", "-n", f"{nnodes}", f"{taudem_pitremove}",
                   "-z", taudem_inDEM, "-fel", tau_fill_dem, ">", taudem_log]
        # if (not os.path.exists(tau_fill_dem)) or (overwrite is True):
        ret = subprocess.call(cmd_lst, shell=True,
                              executable='C:\Windows\SysWOW64\cmd.exe')
        print(ret)

        # Compute Flow Direction
        log_step("   flow direction")

        taudem_flowdir = os.path.join(taudem_path, 'D8FlowDir.exe')
        taudem_log = os.path.join(taudem_projdir, "d8.log")

        cmd_lst = [f"{mpi_path}", "-n", f"{nnodes}", f"{taudem_flowdir}",
                   "-fel", tau_fill_dem, "-p", tau_d8,
                   "-sd8", tau_slope, ">", taudem_log]

        if (not os.path.exists(tau_d8)) or (overwrite is True):
            ret = subprocess.call(cmd_lst, shell=True,
                                  executable='C:\Windows\SysWOW64\cmd.exe')
            print(ret)

        # Compute Flow Accumulation
        log_step("   flow accumulation")
        taudem_acc = os.path.join(taudem_path, 'AreaD8.exe')
        taudem_log = os.path.join(taudem_projdir, "acc.log")

        cmd_lst = [f"{mpi_path}", "-n", f"{nnodes}", f"{taudem_acc}", "-nc",
                   "-p", tau_d8, "-ad8", tau_acc, ">", taudem_log]

        if (not os.path.exists(tau_acc)) or (overwrite is True):
            ret = subprocess.call(cmd_lst, shell=True,
                                  executable='C:\Windows\SysWOW64\cmd.exe')
            print(ret)

       # wait_4_job("TauDEM", sub_log)
        log_step("TauDEM run completed!")

    log_step('All DONE!!', sep=True)
