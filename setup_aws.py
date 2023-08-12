#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 26 2020

@author: Kimberly McCormack

Last edited on: 7/12/2021

Top level function to setup entire hydro-conditioning process

"""

from shapely import geometry
import geopandas as gpd
import pandas as pd
import datetime
import json
from configparser import ConfigParser, ExtendedInterpolation
import numpy as np
import glob
import time
import os
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings(action='ignore', message='GEOS')

from tools import setup_basins
from tools.print import print_time
import tools.files as ft
import tools.convert as pyct
import tools.shapes as shpt
import tools.log as logt
import tools.aws as aws


class PydemSetup():

    def __init__(self, config_file):

        self.config_file = config_file
        self.config = ConfigParser(
            allow_no_value=True, interpolation=ExtendedInterpolation())
        self.config.read(config_file)
        self.starttime = time.time()

        self.WBT_path = os.path.join(self.get_config("paths", "WBT_path"))

    def identify_srcdem(self, input_id, pathlist):
        """
        Identify the source DEM based on the path.
        """
        dem_path = ft.tilesearch(input_id, pathlist)
        if dem_path is None:
            raise ValueError(f"Missing DEM path for {input_id}")
        
        if TDT_STRING in dem_path:
            return TDT_STRING
        elif TDF_STRING in dem_path:
            return TDF_STRING
        else:
            return ""

    def setup_region(self):
        """
        Set up the region based on the provided configuration.
        """
        """ IMPORT REGION AND PATH INFORMATION """
        projectname = self.get_config("region", "projectname")
        basinID = self.get_config("region", "basinID")
        project_shp_file = self.get_config("region", "shp_file")
        bound_box = json.loads(self.get_config("region", "bound_box"))
        overwrite = self.config.getboolean("region", "overwrite")

        if len(basinID) == 10:
            region_type = 'basinID'
            basinID_region = basinID[0]
            basinID_level = basinID[1:3]
            if len(projectname) == 0:
                projectname = basinID
            basinID = np.int(basinID)

        elif os.path.exists(project_shp_file):
            region_type = 'shpfile'
            if len(projectname) == 0:
                projectname = os.path.splitext(
                    os.path.basename(project_shp_file))[0]

        elif len(bound_box) == 4:
            region_type = 'bbox'
            if len(projectname) == 0:
                projectname = 'bbox_[{0:.0f}_{1:.0f}_{2:.0f}_{3:.0f}]'.format(
                    bound_box[0], bound_box[1], bound_box[2], bound_box[3])
        else:
            raise Exception('No name or region information given')

        # Create shape file for AOI and buffer
        self.shapes_dir = os.path.join(self.get_config("paths", "shapes_dir"),
                                       f"{projectname}")
        if not os.path.exists(self.shapes_dir):
            os.makedirs(self.shapes_dir)
        self.aoi_shpfile = os.path.join(self.shapes_dir, f"{projectname}.shp")
        self.aoi_buffer_shpfile = os.path.join(self.shapes_dir,
                                               f"{projectname}_buffer.shp")

        basins_shp_prefix = self.get_config("paths", "basins_shp_prefix")
        bucket = self.get_config("paths", "bucket")
        basins_dir_prefix = f'/vsis3/{bucket}/{basins_shp_prefix}'
        hybas_lvl1_shp = self.get_config("paths", "hybas_lvl1_shp")

        hybas_lvl1_gdf = gpd.read_file(hybas_lvl1_shp)
        hybas_region_dct = {'1': 'af',
                            '2': 'eu',
                            '3': 'si',
                            '4': 'as',
                            '5': 'au',
                            '6': 'sa',
                            '7': 'na',
                            '8': 'ar',
                            '9': 'gr'}

        if region_type == 'basinID':
            hybasin_region_lst = [hybas_region_dct[basinID_region]]
            hybas_shpfile = f'{basins_dir_prefix}/\
            hybas_{hybasin_region_lst[0]}_lev{basinID_level}_v1c.shp'
            basin_gdf = shpt.extract_single_basin_gdf(hybas_shpfile, basinID)
            project_poly = list(basin_gdf.geometry)[0]

        elif region_type == 'shpfile':
            project_shp_gdf = gpd.read_file(project_shp_file)
            project_poly = list(project_shp_gdf.geometry)[0]
            hybas_region_gdf = hybas_lvl1_gdf.loc[hybas_lvl1_gdf.geometry.intersects(
                project_poly)]
            hybasin_region_lst = list(hybas_region_gdf.region.values)

        elif region_type == 'bbox':

            project_poly = geometry.box(bound_box[0], bound_box[1],
                                        bound_box[2], bound_box[3])
            hybas_region_gdf = hybas_lvl1_gdf.loc[hybas_lvl1_gdf.
                                                  geometry.intersects(project_poly)]
            hybasin_region_lst = list(hybas_region_gdf.region.values)

        # buffer polygon and write shape files
        aoi_gdf = gpd.GeoDataFrame(geometry=[project_poly], crs=EPSG_CODE)

        write_shapefile_if_not_exists(self.aoi_shpfile, aoi_gdf, overwrite)

        if (not os.path.exists(self.aoi_buffer_shpfile)) or (overwrite):
            if not region_type == 'bbox':
                aoi_buffer_gdf = pyct.buffer_gdf(aoi_gdf, BUFFER_SIZE)  # 1km buffer
                aoi_buffer_gdf.to_file(self.aoi_buffer_shpfile)
            else:
                self.aoi_buffer_shpfile = self.aoi_shpfile

        # Build txt file of tiles within project
        self.tiles_txt = os.path.join(
            self.get_config("paths", "project_txt_path"),
            f'{projectname}.txt')
        self.project_tile_shp = os.path.join(self.shapes_dir,
                                             f"{projectname}_tiles.shp")
        if not os.path.exists(os.path.dirname(self.tiles_txt)):
            os.makedirs(os.path.dirname(self.tiles_txt))

        DEM_shpfile = os.path.join(self.get_config("paths", "DEM_shpfile"))
        DEM_cells_gdf = gpd.read_file(DEM_shpfile)
        aoi_buffer_gdf = gpd.read_file(self.aoi_buffer_shpfile)
        project_poly = list(aoi_buffer_gdf.geometry)[0]

        if region_type == 'basinID' or region_type == 'bbox':
            project_DEMtiles_gdf = DEM_cells_gdf.loc[
                DEM_cells_gdf.geometry.within(project_poly)]
        else:
            project_DEMtiles_gdf = DEM_cells_gdf.loc[
                DEM_cells_gdf.geometry.intersects(project_poly)]

        project_tiles = list(project_DEMtiles_gdf.TILE_ID)

        with open(self.tiles_txt, 'w') as filehandle:
            for tilename in project_tiles:
                filehandle.write(f"{tilename}\n")

        with open(self.tiles_txt) as f:
            self.tile_names = f.read().splitlines()

        # Build txt file of tiles within buffered project
        self.tiles_buffer_txt = os.path.join(
            self.get_config("paths", "project_txt_path"),
            f'{projectname}_buffer.txt')
        self.project_tile_buffer_shp = os.path.join(
            self.shapes_dir, f"{projectname}_tiles_buffer.shp")
        if not os.path.exists(os.path.dirname(self.tiles_buffer_txt)):
            os.makedirs(os.path.dirname(self.tiles_buffer_txt))

        # if (not os.path.exists(self.tiles_buffer_txt)) or (not os.path.exists(self.project_tile_buffer_shp)):
        DEM_cells_gdf = gpd.read_file(DEM_shpfile)
        aoi_buffer_gdf = gpd.read_file(self.aoi_buffer_shpfile)
        project_poly = list(aoi_buffer_gdf.geometry)[0]
        project_DEMcells_gdf = DEM_cells_gdf.loc[
            DEM_cells_gdf.geometry.intersects(project_poly)]
            
        # add in adjacent tiles to avoid edge effects
        project_aoi_buffer_gdf = pyct.buffer_gdf_simple(
                project_DEMcells_gdf, 0.001, merge=True, projected=False)
        project_poly_buffer = list(project_aoi_buffer_gdf.geometry)[0]
        project_DEMcells_gdf = DEM_cells_gdf.loc[
            DEM_cells_gdf.geometry.intersects(project_poly_buffer)]
                    
            
        project_DEMcells_gdf.to_file(self.project_tile_buffer_shp)
        project_buffer_tiles = list(project_DEMcells_gdf.TILE_ID)

        with open(self.tiles_buffer_txt, 'w') as filehandle:
            for tilename in project_buffer_tiles:
                filehandle.write(f"{tilename}\n")

        # build path list
        base_DEM = self.get_config("parameters-noise-removal", "base_DEM")
        DEM_prefix_list = os.path.join(
            self.get_config("paths", "DEM_prefix_list"))
        dembucket = self.get_config("paths", "DEM_bucket")
        tdtprefix = self.get_config("paths", "TDT_prefix")

        if not os.path.exists(DEM_prefix_list):
            print('  build full prefix list')
            tdx_paths = []
            all_tiles = list(DEM_cells_gdf.TILE_ID)
            tdt_prefix_list = aws.get_dir_list_files(dembucket, tdtprefix,
                                                     searchtxt="DEM.tif",
                                                     excludetxt=None)
            if base_DEM == TDF_STRING:
                tdfprefix = self.get_config("paths", "TDF_prefix")
                tdf_prefix_list = aws.get_dir_list_files(dembucket, tdfprefix,
                                                         searchtxt="DEM.tif",
                                                         excludetxt=None)
                for tile in all_tiles:
                    tile_path = ft.tilesearch(tile, tdf_prefix_list)
                    if tile_path is None:
                        tile_path = ft.tilesearch(tile, tdt_prefix_list)
                    tdx_paths.append(tile_path)
            elif base_DEM == TDT_STRING:
                tdx_paths = tdt_prefix_list

            tdx_vsi_paths = [f"{dembucket}/{file}" for file in tdx_paths]
            with open(DEM_prefix_list, 'w') as filehandle:
                for path in tdx_vsi_paths:
                    filehandle.write("{0}\n".format(path))

        self.tiles_paths_txt = os.path.join(
            self.get_config("paths", "project_txt_path"),
            f'{projectname}_paths.txt')
            
        with open(DEM_prefix_list) as f:
            dem_prefix_list = f.read().splitlines()
                
        #if not os.path.exists(self.tiles_paths_txt):
        print('  build project prefix list')
        tdx_paths = [ft.tilesearch(tile, dem_prefix_list)
                     for tile in project_buffer_tiles]
        tdx_paths = [f"/vsis3/{tif}" for tif in tdx_paths]

        with open(self.tiles_paths_txt, 'w') as filehandle:
            for path in tdx_paths:
                filehandle.write("{0}\n".format(path))

        with open(self.tiles_paths_txt) as f:
            self.tiles_paths_list = f.read().splitlines()

        # add srcDEM attribute to tile shape file
        project_DEMtiles_gdf['srcDEM'] = project_DEMtiles_gdf.TILE_ID.apply(
            lambda d: self.identify_srcdem(d, self.tiles_paths_list))
            
        # # add srcDEM attribute to global tile shape file
        # if not 'scrDEM' in DEM_cells_gdf.columns:
            # print('  add src attribute to global DEM shapefile')
            # DEM_cells_gdf['srcDEM'] = DEM_cells_gdf.TILE_ID.apply(
                # lambda d: self.identify_srcdem(d, dem_prefix_list))           
            # DEM_cells_gdf.to_file(DEM_shpfile)

        project_DEMtiles_gdf.to_file(self.project_tile_shp)
        self.project_tiles_gdf = gpd.read_file(self.project_tile_shp)
        self.project_tiles_buffer_gdf = gpd.read_file(
            self.project_tile_buffer_shp)
        with open(self.tiles_buffer_txt) as f:
            self.tile_buffer_names = f.read().splitlines()

        self.projectname = projectname
        self.region_type = region_type
        self.hybasin_region_lst = hybasin_region_lst

    def start_log_file(self):
        """create log of run"""
        log_files = os.path.join(self.get_config("paths", "output_log_dir"),
                                 'run_logs')
        if not os.path.exists(log_files):
            os.makedirs(log_files)
        timestamp = datetime.datetime.now().strftime("%m-%d-%Y-%H%M")
        self.log_file = os.path.join(
            log_files, f'{self.projectname}_{timestamp}.log')
        log_sections = self.config.sections()
        with open(self.log_file, 'w') as lf:
            lf.write('Configuration options passed in from:  {0}\n \n'.
                     format(self.config_file))
            for log_section in log_sections:
                sec_dict = dict(self.config[log_section])
                lf.write(f"[{log_section}] \n")
                for key in sec_dict:
                    lf.write(f"   {key} : {sec_dict[key]} \n")
                lf.write("\n")

            lf.write('\n-----------------------------------------------\n')
            lf.write(f'project name: {self.projectname}\n')
            lf.write(f'region type used: {self.region_type}\n')

        # update config file
        config_dir = os.path.join(self.get_config("paths", "output_log_dir"),
                                  'auto_config_files')
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        self.config_file_auto = os.path.join(config_dir,
                                             f"config_{self.projectname}_{timestamp}.ini")
        self.config.add_section('logging')
        self.config.set('logging', 'log_file', self.log_file)
        with open(self.config_file_auto, 'w') as f:
            self.config.write(f)

    def setup_outputs(self, EDM=True, FABDEM=True):

        temp_data_path = os.path.join(
            self.get_config("paths", "temp_data_path"))
        self.output_DTM_path = os.path.join(
            self.get_config("paths", "output_DTM_dir"), "tiles")
        # Set path to output DTM mosaic
        self.output_DTMmosaic = os.path.join(
            self.get_config("paths", "output_DTM_dir"),
            'mosaics', f'{self.projectname}_DTM.tif')
        self.input_DEMmosaic = os.path.join(
            self.get_config("paths", "output_DTM_dir"),
            'mosaics', f'{self.projectname}_inputDEM.tif')

        self.output_DH_path = os.path.join(self.get_config("paths",
                                                           "output_dir_cond"),
                                           self.projectname)
        self.shapes_DH_dir = os.path.join(self.output_DH_path, "shape_files")

        # Set up output directories
        if not os.path.exists(self.output_DH_path):
            os.makedirs(self.output_DH_path)
        if not os.path.exists(self.shapes_DH_dir):
            os.makedirs(self.shapes_DH_dir)
        if not os.path.exists(temp_data_path):
            os.makedirs(temp_data_path)
        if not os.path.exists(self.output_DTM_path):
            os.makedirs(self.output_DTM_path)
        if not os.path.exists(os.path.dirname(self.output_DTMmosaic)):
            os.makedirs(os.path.dirname(self.output_DTMmosaic))

        self.run_parallel = self.config.getboolean(
            "job-options", "run_parallel")
        self.run_multiprocess = self.config.getboolean(
            "job-options", "run_multiprocess")

        # Update config file with derived parameters and paths
        self.config.add_section('outputs')
        self.config.set('outputs', 'region_type', self.region_type)
        self.config.set('outputs', 'projectname', self.projectname)
        self.config.set('outputs', 'dtm_mosaic', self.output_DTMmosaic)
        self.config.set('outputs', 'input_dem_mosaic', self.input_DEMmosaic)
        self.config.set('outputs', 'dtm_output_dir', self.output_DTM_path)
        self.config.set('outputs', 'aoi_shapefile', self.aoi_shpfile)

        self.config.set('outputs', 'tile_buffer_shapefile',
                        self.project_tile_buffer_shp)

        hybasin_regions_str = ' '.join(self.hybasin_region_lst)
        self.config.set("outputs", "hybasin_region_lst", hybasin_regions_str)

        if EDM:
            self.output_EDMmosaic = os.path.join(
                self.get_config("paths", "output_DTM_dir"),
                'mosaics', f'{self.projectname}_EDM.tif')
        else:
            self.output_EDMmosaic = ""
        self.config.set('outputs', 'edm_mosaic', self.output_EDMmosaic)

        if FABDEM:
            self.output_FABmosaic = os.path.join(
                self.get_config("paths", "output_DTM_dir"),
                'mosaics', f'{self.projectname}_FABDEM.tif')
        else:
            self.output_FABmosaic = ""
        self.config.set('outputs', 'fabdem_mosaic', self.output_FABmosaic)

        with open(self.config_file_auto, 'w') as f:
            self.config.write(f)

    def setup_noise_removal(self):
        """ Set parameters to be used outside of the class"""

        self.setup_tiles(self.output_DTM_path)

        with open(self.log_file, 'a') as lf:
            lf.write(
                f'Number of DEM cells in region: {len(self.tile_names)}\n')
            lf.write(f'Output noise-removed mosaic: {self.output_DTMmosaic}\n')

    def setup_endo_basins(self, basin_work_dir, basin_level):
        """ Function to set up directories and shapefiles for endorheic basins"""

        # Open shape files
        lvl = basin_level
        basins_shp = os.path.join(
            self.shapes_DH_dir, f'basins_lvl{lvl:02d}.shp')
        basins_gdf = gpd.read_file(basins_shp)
        endo_gdf = basins_gdf[basins_gdf["ENDO"] == 2]
        cutline_field = self.config.get(
            "parameters-depression-handling", "cutline_field")
        if len(cutline_field) == 0:
            cutline_field = "HYBAS_ID"
        basinIDs = list(endo_gdf[cutline_field].values)

        nbasins = len(basinIDs)

        # extract basinIDS without shape files
        unbuiltIDs = [b for b in basinIDs if not os.path.exists(
                            os.path.join(basin_work_dir, f'basin_{b}',
                                         f'basin_{b}.shp'))]
        print("Setting up directories for {0}/{1} endorheic basins".format(
            len(unbuiltIDs), nbasins))
        for i, basinID in enumerate(unbuiltIDs):
            setup_basins.basin_setup_single_lvl(
                basinID, basins_gdf, basin_work_dir)
            if (i + 1) % 500 == 0:
                print(f"   {i+1}/{nbasins} complete")

    def clip_basins(self, endo_level=None):

        if endo_level is None:
            basin_levels = json.loads(self.config.get(
                "parameters-DH", "basin_levels"))
            basin_work_dir = os.path.join(self.output_DH_path,
                                          f'basins_lvl{basin_levels[0]:02d}')
        else:
            basin_levels = [endo_level]
            basin_work_dir = os.path.join(self.output_DH_path, 'endo_basins')

        if not os.path.exists(basin_work_dir):
            os.makedirs(basin_work_dir)

        # Build shape files for all basins in project
        basins_shp_prefix = os.path.join(
            self.get_config("paths", "basins_shp_prefix"))
        bucket = self.get_config("paths", "bucket")
        basins_dir_prefix = f'/vsis3/{bucket}/{basins_shp_prefix}'
        hybas_regions = self.config.get(
            "outputs", "hybasin_region_lst").split(' ')

        for lvl in basin_levels:
            basins_lvl_shp = os.path.join(self.shapes_DH_dir,
                                          f'basins_lvl{lvl:02d}.shp')

            if not os.path.exists(basins_lvl_shp):
                print(f"   Building basin level {lvl} project shapefile")

                basin_gdf = None
                for hybas_region in hybas_regions:
                    basins_input_shpfile = f"{basins_dir_prefix}/"\
                                           + f'hybas_{hybas_region}_lev{lvl:02d}_v1c.shp'

                    temp_gdf = shpt.clip_shp(self.aoi_buffer_shpfile,
                                                     basins_input_shpfile,
                                                     condition='intersect')

                    if basin_gdf is None:
                        basin_gdf = temp_gdf
                    else:
                        basin_gdf = gpd.GeoDataFrame(pd.concat([basin_gdf, temp_gdf],
                                                               ignore_index=True))

                basin_gdf.to_file(basins_lvl_shp)

        if endo_level is not None:
            basin_levels = endo_level

        return basin_work_dir, basin_levels

    def setup_DH_basins(self, basin_work_dir, basin_levels, basinIDs=None):
        """Open shape files"""
        lvl = basin_levels[0]
        basins_shp = os.path.join(
            self.shapes_DH_dir, f'basins_lvl{lvl:02d}.shp')
        basins_gdf = gpd.read_file(basins_shp)
        cutline_field = self.config.get(
            "parameters-depression-handling", "cutline_field")
        if len(cutline_field) == 0:
            cutline_field = "HYBAS_ID"

        if basinIDs is None:
            basinIDs = list(basins_gdf[cutline_field].values)

        if len(basin_levels) == 1:
            nbasins = len(basinIDs)
            # extract basinIDS without shape files
            unbuiltIDs = [b for b in basinIDs if not os.path.exists(
                                os.path.join(basin_work_dir, f'basin_{b}',
                                             f'basin_{b}.shp'))]
            print(
                f"Setting up directories for {len(unbuiltIDs)}/{nbasins} single level basins")
            for i, basinID in enumerate(unbuiltIDs):
                setup_basins.basin_setup_single_lvl(
                    basinID, basins_gdf, basin_work_dir)
                if (i + 1) % 500 == 0:
                    print(f"   {i+1}/{nbasins} complete")

        elif len(basin_levels) >= 3:
            print("   Setting up directories for multi level basins")
            basin_sz_limit = self.config.getfloat(
                "parameters-DH-basin", "basin_sz_limit")
            basins_p1_shp = os.path.join(self.shapes_DH_dir,
                                         f'basins_lvl{basin_levels[1]:02d}.shp')
            basins_p2_shp = os.path.join(self.shapes_DH_dir,
                                         f'basins_lvl{basin_levels[2]:02d}.shp')

            basins_p1_gdf = gpd.read_file(basins_p1_shp)
            basins_p2_gdf = gpd.read_file(basins_p2_shp)

            if len(basin_levels) == 4:
                basins_p3_shp = os.path.join(self.shapes_DH_dir,
                                             f'basins_lvl{basin_levels[3]:02d}.shp')
                basins_p3_gdf = gpd.read_file(basins_p3_shp)
            else:
                basins_p3_gdf = None

            for basinID in basinIDs:
                setup_basins.basin_setup_multi_lvl(basins_gdf, basinID, basins_p1_gdf, basins_p2_gdf,
                                                   basin_work_dir, levels=basin_levels,
                                                   threshold=basin_sz_limit, basins_p3_gdf=basins_p3_gdf)

            check_setup = self.config.get(
                "processes-DH-basin", "check_basin_setup")

            if check_setup:
                missing_basins = setup_basins.check_setup(basins_gdf, basins_p1_gdf,
                                                          basins_p2_gdf, basin_work_dir,
                                                          basins_p3_gdf=basins_p3_gdf,
                                                          cutline_field=cutline_field)

                if len(missing_basins) > 0:
                    print('     setup missing basins')
                    for basinID in missing_basins:
                        setup_basins.basin_setup_multi_lvl(basins_gdf, basinID,
                                                           basins_p1_gdf, basins_p2_gdf,
                                                           basin_work_dir, levels=basin_levels,
                                                           basins_p3_gdf=basins_p3_gdf,
                                                           threshold=basin_sz_limit)

                    missing_basins = setup_basins.check_setup(basins_gdf, basins_p1_gdf,
                                                              basins_p2_gdf, basin_work_dir,
                                                              basins_p3_gdf=basins_p3_gdf,
                                                              cutline_field=cutline_field)

                    if len(missing_basins) > 0:
                        raise('basins still missing in setup, abort')

        else:
            raise ("basin setup functions only work for 1, 3 or 4 basin levels")

    def setup_tiles(self, work_dir):

        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        tiles = self.tile_names
        tiles_gdf = self.project_tiles_gdf

        for tile in tiles:

            outpath = os.path.join(work_dir, f'tile_{tile}')
            if not os.path.exists(outpath):
                os.mkdir(outpath)

            tile_shpfile = os.path.join(outpath, f'tile_{tile}.shp')
            if not os.path.exists(tile_shpfile):
                tile_gdf = tiles_gdf[tiles_gdf.TILE_ID == tile].copy()
                tile_gdf.crs = EPSG_CODE
                tile_gdf.to_file(tile_shpfile)

    def build_DH_shpfile(self, basin_work_dir):
        """create shape file for all project basins"""
        basin_paths = glob.glob(os.path.join(basin_work_dir, 'basin_*'))
        basinIDs = [os.path.splitext(os.path.basename(b))[0].
                    replace('basin_', '') for b in basin_paths]
        basinIDs = [np.int(ID) for ID in basinIDs]

        basins_all_gdf = None
        for basin_id in basinIDs:
            outpath_basin = os.path.join(basin_work_dir, f'basin_{basin_id}')
            basin_shpfile = os.path.join(
                outpath_basin, f'basin_{basin_id}.shp')
            basin_gdf = gpd.read_file(basin_shpfile)
            if basins_all_gdf is None:
                basins_all_gdf = basin_gdf.copy()
            else:
                basins_all_gdf = basins_all_gdf.append(basin_gdf)

        basins_all_gdf.to_file(self.basins_all_shp)

        basins_all_gdf['group'] = 1
        basins_all_gdf = basins_all_gdf.dissolve(by='group')
        basins_all_gdf.to_file(self.basins_merge_shp)

        basins_all_gdf = pyct.buffer_gdf(basins_all_gdf, 10)
        basins_all_gdf = shpt.close_holes_gdf(basins_all_gdf)
        basins_all_gdf.to_file(self.basins_dissolve_shp)

    def change_basin_setup(self, unfin_basinIDs, basin_work_dir, basin_levels):
        """
        Can run the unfinished basins with a different setup. Use a
        smaller area threshold for the leftover basins if they are taking forever to solve.
        This will break them up more and improve computation times"""

        ft.remove_basin_dir(basin_work_dir, unfin_basinIDs)
        self.setup_DH_basins(basin_work_dir, basin_levels,
                             basinIDs=unfin_basinIDs)

    def update_config_DH(self, work_dir):

        base_unit = self.get_config("region", "base_unit")

        # Set path to output  mosaic
        self.dh_mosaic = os.path.join(self.output_DH_path, 'mosaics',
                                      f'{self.projectname}_dh.vrt')
        if not os.path.exists(os.path.dirname(self.dh_mosaic)):
            os.makedirs(os.path.dirname(self.dh_mosaic))

        self.config.set('outputs', 'dh_mosaic', self.dh_mosaic)
        self.config.set('outputs', 'dh_work_dir', work_dir)

        if base_unit == 'basin':
            self.basins_all_shp = os.path.join(self.shapes_DH_dir,
                                               f'{self.projectname}_basins.shp')
            self.basins_merge_shp = os.path.join(self.shapes_DH_dir,
                                                 f'{self.projectname}_basins_merged.shp')
            self.basins_dissolve_shp = os.path.join(self.shapes_DH_dir,
                                                    f'{self.projectname}_basins_dissolved.shp')

            self.config.set('outputs', 'basins_all_shp', self.basins_all_shp)
            self.config.set('outputs', 'basins_merge_shp',
                            self.basins_merge_shp)
            self.config.set('outputs', 'basins_dissolve_shp',
                            self.basins_dissolve_shp)

        with open(self.config_file_auto, 'w') as f:
            self.config.write(f)

        with open(self.log_file, 'a') as lf:
            lf.write(f'\nOutput DH mosaic: \n    {self.dh_mosaic}\n')

    def update_config_water(self):

        project = self.projectname

        # Set path to output subbasin mosaic
        self.burned_mosaic = os.path.join(self.output_DH_path, 'mosaics',
                                          f'{project}_river_enforced.vrt')

        self.water_mask_mosaic = os.path.join(self.output_DH_path, 'mosaics',
                                              f'{project}_water_mask.vrt')

        self.config.set('outputs', 'water_mask_mosaic', self.water_mask_mosaic)
        self.config.set('outputs', 'burned_mosaic', self.burned_mosaic)
        with open(self.config_file_auto, 'w') as f:
            self.config.write(f)

        with open(self.log_file, 'a') as lf:
            lf.write(f'\nOutput Burned mosaic: \n    {self.burned_mosaic}\n')

    def write_basin_logs(self, basin_work_dir, unfin_basins_txt,
                         unsolved_basins_txt, diff_basins_txt):

        logt.find_unfinished_basins(basin_work_dir, unfin_basins_txt)
        with open(unfin_basins_txt) as f:
            unfin_subpaths = f.read().splitlines()
        print(f'{len(unfin_subpaths)} subbasins remain to be processed')

        # Find basins with unsolved pits
        if os.path.exists(unsolved_basins_txt):
            os.remove(unsolved_basins_txt)
        logt.find_unsolved_basins(basin_work_dir, unsolved_basins_txt)
        with open(unsolved_basins_txt) as f:
            unsolved_subpaths = f.read().splitlines()
        print_time(
            f'{len(unsolved_subpaths)} basins have unsolved pits', self.starttime)

        # Find basins with suspicious difference maps
        if os.path.exists(diff_basins_txt):
            os.remove(diff_basins_txt)
        logt.find_diff_basins(basin_work_dir, diff_basins_txt,
                              searchstr='basin_*', tifname='conditionedDEM_DIFF.tif')
        with open(diff_basins_txt) as f:
            diff_subpaths = f.read().splitlines()
        print_time(f'{len(diff_subpaths)} basins have bad difference masks',
                   self.starttime)

    def build_noise_remove_call(self, pydem_path, comp_sys, venv=''):

        if self.run_parallel:  # run in parallel

            raise "Not currently configured to run in parallel"

        else:
            log_dir = os.path.join(self.config.get(
                "paths", "output_log_dir"), 'temp_logs')
            sub_log = os.path.join(
                log_dir, f"dem_noise_removal_{self.projectname}.txt")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            if os.path.exists(sub_log):
                os.remove(sub_log)
            noise_removal_file = os.path.join(
                pydem_path, 'models', 'dem_noise_removal.py')
            sub_call = f"python {noise_removal_file} -c {self.config_file_auto} -l {sub_log}"

        return sub_call, sub_log

    def build_burn_rivers_call(self, pydem_path, comp_sys, nbasins=1, venv=''):

        if self.run_parallel:  # run in parallel
            raise "Not currently configured to run in parallel"

        else:  # run on a single node
            log_dir = os.path.join(self.config.get(
                "paths", "output_log_dir"), 'temp_logs')
            sub_log = os.path.join(
                log_dir, "EnforceRivers_{0}.txt".format(self.projectname))
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            if os.path.exists(sub_log):
                os.remove(sub_log)
            flatten_file = os.path.join(pydem_path, 'models', 'burn_rivers.py')
            sub_call = "python {1} -c {0} -l {2}".format(
                self.config_file_auto, flatten_file, sub_log)

        return sub_call, sub_log

    def build_minimize_call(self, pydem_path, comp_sys, nbasins=1, venv=''):

        if self.run_parallel:  # run in parallel
            raise "Not currently configured to run in parallel"

        else:  # run on a single node
            log_dir = os.path.join(self.config.get(
                "paths", "output_log_dir"), 'temp_logs')
            sub_log = os.path.join(
                log_dir, "overlap_minimize_{0}.txt".format(self.projectname))
            if os.path.exists(sub_log):
                os.remove(sub_log)
            min_overlap_py = os.path.join(
                pydem_path, 'models', 'minimize_overlap.py')
            sub_call = f"python {min_overlap_py} -c {self.config_file_auto} -l {sub_log}"

        return sub_call, sub_log

    def build_sinks_call(self, pydem_path, comp_sys, nbasins=1, venv=''):

        if self.run_parallel:  # run in parallel
            raise "Not currently configured to run in parallel"

        else:  # run on a single node
            log_dir = os.path.join(self.config.get(
                "paths", "output_log_dir"), 'temp_logs')
            sub_log = os.path.join(
                log_dir, "find_sinks_{0}.txt".format(self.projectname))
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            if os.path.exists(sub_log):
                os.remove(sub_log)
            find_sinks_file = os.path.join(
                pydem_path, 'models', 'find_sinks.py')
            sub_call = f"python {find_sinks_file} -c {self.config_file_auto} -l {sub_log}"

        return sub_call, sub_log

    def build_dep_handle_call(self, pydem_path, comp_sys, nbasins=1, venv=''):

        if self.run_parallel:  # run in parallel
            raise "Not currently configured to run in parallel"

        else:  # run on a single node
            log_dir = os.path.join(self.config.get(
                "paths", "output_log_dir"), 'temp_logs')
            sub_log = os.path.join(
                log_dir, "dephandle_{0}.txt".format(self.projectname))
            if os.path.exists(sub_log):
                os.remove(sub_log)
            dep_handle_py = os.path.join(
                pydem_path, 'models', 'depression_handling.py')
            sub_call = f"python {dep_handle_py} -c {self.config_file_auto} -l {sub_log}"

        return sub_call, sub_log

    def build_taudem_check_call(self, pydem_path, comp_sys, nbasins=1, venv=''):

        if self.run_parallel:  # run in parallel
            raise "Not currently configured to run in parallel"

        else:  # run on a single node
            log_dir = os.path.join(self.config.get(
                "paths", "output_log_dir"), 'temp_logs')
            sub_log = os.path.join(
                log_dir, "taudem_check_{0}.txt".format(self.projectname))
            if os.path.exists(sub_log):
                os.remove(sub_log)
            tau_check_py = os.path.join(
                pydem_path, 'models', 'taudem_check.py')
            sub_call = f"python {tau_check_py} -c {self.config_file_auto} -l {sub_log}"

        return sub_call, sub_log
