#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 26 2020

@author: Kimberly McCormack

Last edited on: 7/12/2021

Top level function to setup entire hydro-conditioning process

"""

import os
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings(action='ignore', message='GEOS')
import time
import glob
import numpy as np
from configparser import ConfigParser, ExtendedInterpolation
import json
import datetime
import geopandas as gpd
from shapely import geometry

import tools.log as logt
import tools.shapes as shpt
import tools.convert as pyct
import tools.files as ft
from tools.print import print_time
from tools import setup_basins


class PydemSetup():
    
    def __init__(self, config_file):

        self.config_file = config_file
        self.config = ConfigParser(allow_no_value=True, interpolation=ExtendedInterpolation())
        self.config.read(config_file)
        self.starttime = time.time()

    def setup_region(self):
        
        """ IMPORT REGION AND PATH INFORMATION """
        projectname = self.config.get("region", "projectname") 
        basinID = self.config.get("region", "basinID")
        project_shp_file = self.config.get("region", "shp_file")
        bound_box = json.loads(self.config.get("region", "bound_box"))
        
        if len(basinID) == 10:
            region_type = 'basinID'
            basinID_region = basinID[0]
            basinID_level = basinID[1:3]
            if len(projectname) == 0: projectname = basinID
            basinID = np.int(basinID)

        elif os.path.exists(project_shp_file):
            region_type = 'shpfile'        
            if len(projectname) == 0: 
                projectname = os.path.splitext(os.path.basename(project_shp_file))[0]

        elif len(bound_box) == 4:
            region_type = 'bbox'
            if len(projectname) == 0:  
                projectname = 'bbox_[{0:.0f}_{1:.0f}_{2:.0f}_{3:.0f}]'.format(bound_box[0],
                                              bound_box[1], bound_box[2], bound_box[3])        
        else:
            raise Exception('No name or region information given') 

        """list of all DEM paths"""
        DEMpath_list = os.path.join(self.config.get("paths-all", "DEMpath_list"))
        if not os.path.exists(DEMpath_list):
            print('{0} does not exist - building'.format(DEMpath_list))
            ft.buildpathlist(DEM_srcpath, '*DEM.tif', DEMpath_list)
        
        """ Create shape file for AOI and buffer"""
        self.shapes_dir = os.path.join(self.config.get("paths-DTM", "shapes_dir"))
        if not os.path.exists(self.shapes_dir): 
            os.makedirs(self.shapes_dir)  
        self.aoi_shpfile = os.path.join(self.shapes_dir, "{0}.shp".format(projectname))
        self.aoi_buffer_shpfile = os.path.join(self.shapes_dir,
                                               "{0}_buffer.shp".format(projectname))

        basins_shp_path = os.path.join(self.config.get("paths-all", "basins_shp_path"))
        hybas_lvl1_shp = os.path.join(basins_shp_path, 'hybas_lvl1_merged.shp')
        hybas_lvl1_gdf = gpd.read_file(hybas_lvl1_shp)
        hybas_region_dct = {'1':'af',
                            '2':'eu',
                            '3':'si',
                            '4':'as',
                            '5':'au',
                            '6':'sa',
                            '7':'na',
                            '8':'ar',
                            '9':'gr'}    
        
        if region_type == 'basinID':
            hybasin_region  = hybas_region_dct[basinID_region]             
            hybas_shpfile = os.path.join(basins_shp_path,
                                   'hybas_{1}_lev{0}_v1c'.format(
                                       basinID_level, hybasin_region),
                                   'hybas_{1}_lev{0}_v1c.shp'.format(
                                       basinID_level, hybasin_region))
            basin_gdf = shpt.extract_single_basin_gdf(hybas_shpfile, basinID)
            project_poly = list(basin_gdf.geometry)[0]

        elif region_type == 'shpfile': 
            project_shp_gdf = gpd.read_file(project_shp_file)
            project_poly = list(project_shp_gdf.geometry)[0]                
            hybas_region_gdf = hybas_lvl1_gdf.loc[hybas_lvl1_gdf.geometry.intersects(project_poly)]
            hybasin_region = hybas_region_gdf.region.values
            if len(hybasin_region) == 1:
                hybasin_region = hybasin_region[0]
            else:
                raise('Shape file provided overlaps more than one (or zero) hybas regions')

        elif region_type == 'bbox':
            project_poly = geometry.box(bound_box[0], bound_box[1], 
                                                bound_box[2], bound_box[3])
            hybas_region_gdf = hybas_lvl1_gdf.loc[hybas_lvl1_gdf.
                                                  geometry.intersects(project_poly)]
            hybasin_region = hybas_region_gdf.region.values
            if len(hybasin_region) == 1:
                hybasin_region = hybasin_region[0]
            else:
                raise('Bounding box provided overlaps more than one (or zero) hybas regions')
           
        """buffer polygon and write shape files"""
        aoi_gdf = gpd.GeoDataFrame(geometry=[project_poly], crs="EPSG:4326")
        
        if not os.path.exists(self.aoi_shpfile):
            aoi_gdf.to_file(self.aoi_shpfile)
        
        if not os.path.exists(self.aoi_buffer_shpfile):
            aoi_buffer_gdf = pyct.buffer_gdf(aoi_gdf, 1e3) #1km buffer  
            aoi_buffer_gdf.to_file(self.aoi_buffer_shpfile)
            
        self.projectname = projectname
        self.region_type = region_type   
        self.hybasin_region = hybasin_region 

            
    def start_log_file(self):
        
        """create log of run"""
        log_files = os.path.join(self.config.get("paths-all", "output_log_dir"),'run_logs')
        if not os.path.exists(log_files): os.makedirs(log_files)
        timestamp = datetime.datetime.now().strftime("%m-%d-%Y-%H%M")
        self.log_file = os.path.join(log_files, '{1}_{0}.log'.format(timestamp, 
                                                                self.projectname))
        log_sections = self.config.sections()
        with open(self.log_file, 'w') as lf:
            lf.write('Configuration options passed in from:  {0}\n \n'.
                             format(self.config_file))
            for log_section in log_sections:
                sec_dict = dict(self.config[log_section])
                lf.write("[{0}] \n".format(log_section))
                for key in sec_dict:
                    lf.write("   {0} : {1} \n".format(key,sec_dict[key]))
                lf.write("\n")

            lf.write('\n-----------------------------------------------\n')
            lf.write('project name: {0}\n'.format(self.projectname))
            lf.write('region type used: {0}\n'.format(self.region_type))

        """ update config file"""
        config_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'auto_config_files')
        if not os.path.exists(config_dir): os.makedirs(config_dir)
        self.config_file_auto = os.path.join(config_dir,  "config_{1}_{0}.ini".format(timestamp,
                                                                                      self.projectname))
        self.config.add_section('logging')
        self.config.set('logging', 'log_file', self.log_file)
        with open(self.config_file_auto, 'w') as f:
            self.config.write(f) 

        
    def setup_outputs(self):
        
        temp_data_path = os.path.join(self.config.get("paths-all", "temp_data_path"))
        output_DTM_path = os.path.join(self.config.get("paths-DTM", "output_DTM_dir"), 
                                       self.projectname)
        """ Set path to output DTM mosaic"""
        output_DTMmosaic = os.path.join(self.config.get("paths-DTM", "out_mosaic_dir"), 
                                      '{0}_DTM.tif'.format(self.projectname))
        
        self.output_DH_path = os.path.join(self.config.get("paths-DH", "output_dir"),
                                           self.projectname)
        self.shapes_DH_dir = os.path.join(self.output_DH_path, "shape_files")
        
        """Set up output directories"""
        if not os.path.exists(self.output_DH_path): 
            os.makedirs(self.output_DH_path)
        if not os.path.exists(self.shapes_DH_dir):
                os.makedirs(self.shapes_DH_dir)
        if not os.path.exists(temp_data_path): 
            os.makedirs(temp_data_path)  
        if not os.path.exists(output_DTM_path): 
            os.makedirs(output_DTM_path)                            
        if not os.path.exists(os.path.dirname(output_DTMmosaic)):
            os.makedirs(os.path.dirname(output_DTMmosaic))      
            
        """ Set parameters to be used outside of the class"""
        self.output_DTMmosaic = output_DTMmosaic
        self.output_DTM_path = output_DTM_path 
        self.run_w_mpi = self.config.getboolean("job-options", "run_w_MPI")

        """Update config file with derived parameters and paths"""
        self.config.add_section('outputs')       
        self.config.set('outputs', 'region_type', self.region_type)
        self.config.set('outputs', 'projectname', self.projectname)
        self.config.set('outputs', 'hybasin_region', self.hybasin_region)
        self.config.set('outputs', 'dtm_mosaic', output_DTMmosaic)
        self.config.set('outputs', 'dtm_output_dir', output_DTM_path)
        self.config.set('outputs', 'aoi_shapefile', self.aoi_shpfile)
        with open(self.config_file_auto, 'w') as f:
            self.config.write(f) 

            
    def setup_DTM(self):
        
        overwrite_DTM = self.config.getboolean("processes-DTM", "overwrite") 
        
        """path to project text file"""    
        project_cells_txt = os.path.join(self.config.get("paths-DTM", "project_txt_path"),
                                        '{0}.txt'.format(self.projectname))     
        DEM_shpfile = os.path.join(self.config.get("paths-all", "DEM_shpfile"))
        if not os.path.exists(project_cells_txt) or (overwrite_DTM):    
            DEM_cells_gdf = gpd.read_file(DEM_shpfile)
            aoi_buffer_gdf = gpd.read_file(self.aoi_buffer_shpfile)
            project_poly = list(aoi_buffer_gdf.geometry)[0] 
            project_DEMcells_gdf = DEM_cells_gdf.loc[DEM_cells_gdf.geometry.intersects(project_poly)]
            project_cells = list(project_DEMcells_gdf.TILE_ID)

            with open(project_cells_txt, 'w') as filehandle:
                for cellname in project_cells:
                    filehandle.write("{0}\n".format(cellname))        
            
        """ Set parameters to be used outside of the class"""
        self.cell_names = open(project_cells_txt).read().splitlines()                   
        self.checkCells = self.config.getboolean("processes-DTM", "check_cells")
        self.build_DTM_Mosaic = self.config.getboolean("processes-DTM", "build_mosaic")
            
        with open(self.log_file, 'a') as lf:
            lf.write('Number of DEM cells in region: {0}\n'.format(
                len(self.cell_names)))
            lf.write('Output DTM mosaic: {0}\n'.format(self.output_DTMmosaic))

            
    def setup_endo_basins(self, basin_work_dir, basin_level): 
        """ Function to set up directories and shapefiles for endorheic basins"""
        
        """Open shape files"""  
        lvl = basin_level
        basins_shp = os.path.join(self.shapes_DH_dir, 'basins_lvl{0:02d}.shp'.format(lvl))
        basins_gdf = gpd.read_file(basins_shp) 
        endo_gdf = basins_gdf[basins_gdf["ENDO"]==2]
        cutline_field = self.config.get("parameters-DH-all", "cutline_field")
        basinIDs = list(endo_gdf[cutline_field].values)

        nbasins = len(basinIDs)            
        """extract basinIDS without shape files"""
        unbuiltIDs = [b for b in basinIDs if not os.path.exists(
                            os.path.join(basin_work_dir,'basin_{0}'.format(b),
                            'basin_{0}.shp'.format(b)))]                        
        print("Setting up directories for {0}/{1} endorheic basins".format(
                len(unbuiltIDs),nbasins))            
        for i, basinID in enumerate(unbuiltIDs):
            setup_basins.basin_setup_single_lvl(basinID, basins_gdf, basin_work_dir)               
            if (i+1)%500==0: print("   {0}/{1} complete".format(i+1,nbasins))        
            

    def setup_DH_shapes(self, level='basin', endo_level=None, use_ogr=False):
         
        if endo_level is None:
            basin_levels = json.loads(self.config.get(
                            "parameters-DH-{0}".format(level), "basin_levels"))     
            basin_work_dir = os.path.join(self.output_DH_path, 
                              'basins_lvl{0:02d}'.format(basin_levels[0]))            
        else:
            basin_levels = [endo_level]
            basin_work_dir = os.path.join(self.output_DH_path, 'endo_basins')
            
        if not os.path.exists(basin_work_dir): os.makedirs(basin_work_dir)

        """Build shape files for all basins in project"""
        basins_shp_path = os.path.join(self.config.get("paths-all", "basins_shp_path"))
        hybasin_region  = self.config.get("outputs", "hybasin_region")
   
        for lvl in basin_levels:
            basins_lvl_shp = os.path.join(self.shapes_DH_dir, 
                                          'basins_lvl{0:02d}.shp'.format(lvl)) 
            if not os.path.exists(basins_lvl_shp):
                print("   Building basin level {0} project shapefile".format(lvl))
                shpt.build_shapefiles(self.aoi_shpfile,
                                    self.aoi_buffer_shpfile, 
                                    self.shapes_DH_dir,
                                    basins_shp_path, basin_levels,
                                    hybasin_region, use_ogr=use_ogr)  
                
        if endo_level is not None: 
            basin_levels = endo_level
       
        return basin_work_dir, basin_levels

    
    def setup_DH_basins(self, basin_work_dir, basin_levels, basinIDs=None):
        
        """Open shape files"""  
        lvl = basin_levels[0]
        basins_shp = os.path.join(self.shapes_DH_dir, 'basins_lvl{0:02d}.shp'.format(lvl))
        basins_gdf = gpd.read_file(basins_shp) 
        cutline_field = self.config.get("parameters-DH-all", "cutline_field")
        
        if basinIDs is None:
            basinIDs = list(basins_gdf[cutline_field].values)
           

        if len(basin_levels) == 1:
            nbasins = len(basinIDs)            
            """extract basinIDS without shape files"""
            unbuiltIDs = [b for b in basinIDs if not os.path.exists(
                                os.path.join(basin_work_dir,'basin_{0}'.format(b),
                                'basin_{0}.shp'.format(b)))]                        
            print("Setting up directories for {0}/{1} single level basins".format(
                    len(unbuiltIDs),nbasins))            
            for i, basinID in enumerate(unbuiltIDs):
                setup_basins.basin_setup_single_lvl(basinID, basins_gdf, basin_work_dir)               
                if (i+1)%500==0: print("   {0}/{1} complete".format(i+1,nbasins))
                
                
        elif len(basin_levels) >= 3: 
            print("   Setting up directories for multi level basins")
            basin_sz_limit = self.config.getfloat("parameters-DH-basin", "basin_sz_limit")
            basins_p1_shp = os.path.join(self.shapes_DH_dir, 
                            'basins_lvl{0:02d}.shp'.format(basin_levels[1]))
            basins_p2_shp = os.path.join(self.shapes_DH_dir, 
                            'basins_lvl{0:02d}.shp'.format(basin_levels[2]))
            
            basins_p1_gdf = gpd.read_file(basins_p1_shp)
            basins_p2_gdf = gpd.read_file(basins_p2_shp)
            
            if len(basin_levels) == 4:
                basins_p3_shp = os.path.join(self.shapes_DH_dir, 
                            'basins_lvl{0:02d}.shp'.format(basin_levels[3]))
                basins_p3_gdf = gpd.read_file(basins_p3_shp)
            else:
                basins_p3_gdf = None

            for basinID in basinIDs:
                setup_basins.basin_setup_multi_lvl(basins_gdf, basinID, basins_p1_gdf, basins_p2_gdf,
                                      basin_work_dir, levels=basin_levels, 
                                      threshold=basin_sz_limit, basins_p3_gdf=basins_p3_gdf)

                #print_time("   preprocess basin {0} directory COMPLETE".format(basinID),
                #           self.starttime)  
               
            check_setup = self.config.get("processes-DH-basin", "check_basin_setup")
            
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
            
            
    def build_DH_shpfile(self, basin_work_dir):
        """create shape file for all project basins"""
        basin_paths =  glob.glob(os.path.join(basin_work_dir, 'basin_*')) 
        basinIDs = [os.path.splitext(os.path.basename(b))[0].
                      replace('basin_', '') for b in basin_paths]
        basinIDs = [np.int(ID) for ID in basinIDs]

        
        basins_all_gdf = None
        for basin_id in basinIDs:
            outpath_basin = os.path.join(basin_work_dir, 'basin_{0}'.format(basin_id))
            basin_shpfile = os.path.join(outpath_basin,'basin_{0}.shp'.format(basin_id))
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
        self.setup_DH_basins(basin_work_dir, basin_levels, basinIDs=unfin_basinIDs)
               

    def update_config_basin(self):
        
        """ Set path to output subbasin mosaic"""
        self.basin_mosaic = os.path.join(self.output_DH_path, 'mosaics',
                                      '{0}_basin.vrt'.format(self.projectname))
        if not os.path.exists(os.path.dirname(self.basin_mosaic)):
            os.makedirs(os.path.dirname(self.basin_mosaic))
            
        self.basins_all_shp  = os.path.join(self.shapes_DH_dir,
                                            '{0}_basins.shp'.format(self.projectname))
        self.basins_merge_shp  = os.path.join(self.shapes_DH_dir,
                                              '{0}_basins_merged.shp'.format(self.projectname))
        self.basins_dissolve_shp  = os.path.join(self.shapes_DH_dir,
                                              '{0}_basins_dissolved.shp'.format(self.projectname))
                                
        self.config.set('outputs', 'basin_mosaic', self.basin_mosaic)
        self.config.set('outputs', 'basins_all_shp', self.basins_all_shp)
        self.config.set('outputs', 'basins_merge_shp', self.basins_merge_shp)
        self.config.set('outputs', 'basins_dissolve_shp', self.basins_dissolve_shp)
        with open(self.config_file_auto, 'w') as f:
            self.config.write(f) 
            
        with open(self.log_file, 'a') as lf:
            lf.write('\nOutput Basin mosaic: \n    {0}\n'.format(self.basin_mosaic))

                       
    def update_config_flattened(self):
        
        """ Set path to output subbasin mosaic"""
        self.flattened_mosaic = os.path.join(self.output_DH_path, 'mosaics',
                                      '{0}_river_flattened.vrt'.format(self.projectname))
        
        self.flattened_mask_mosaic = os.path.join(self.output_DH_path, 'mosaics',
                                      '{0}_water_flatten_mask.vrt'.format(self.projectname))
                                
        self.config.set('outputs', 'flattened_mask_mosaic', self.flattened_mask_mosaic)
        self.config.set('outputs', 'flattened_mosaic', self.flattened_mosaic)
        with open(self.config_file_auto, 'w') as f:
            self.config.write(f) 
            
        with open(self.log_file, 'a') as lf:
            lf.write('\nOutput Flattened mosaic: \n    {0}\n'.format(self.flattened_mosaic))


    def write_basin_logs(self, basin_work_dir, unfin_basins_txt,
                        unsolved_basins_txt, diff_basins_txt):

        logt.find_unfinished_basins(basin_work_dir, unfin_basins_txt)
        with open(unfin_basins_txt) as f:
            unfin_subpaths = f.read().splitlines()        
        print('{0} subbasins remain to be processed'.format(
            len(unfin_subpaths)))  

        """Find basins with unsolved pits"""
        if os.path.exists(unsolved_basins_txt):
            os.remove(unsolved_basins_txt)            
        logt.find_unsolved_basins(basin_work_dir, unsolved_basins_txt)            
        with open(unsolved_basins_txt) as f:
            unsolved_subpaths = f.read().splitlines()                
        print_time('{0} basins have unsolved pits'.format(
            len(unsolved_subpaths)),self.starttime)              
        proceed = True

        """Find basins with suspicious difference maps"""
        if os.path.exists(diff_basins_txt):
            os.remove(diff_basins_txt)
        logt.find_diff_basins(basin_work_dir, diff_basins_txt, 
                             searchstr='basin_*', tifname='conditionedDEM_DIFF.tif')
        with open(diff_basins_txt) as f:
            diff_subpaths = f.read().splitlines()                 
        print_time('{0} basins have bad difference masks'.format(
            len(diff_subpaths)),self.starttime)   
        
        
    def build_dtm_call(self, pydem_path, comp_sys, venv=''):
        
        
        if self.run_w_mpi: #run in parallel         
            
            if comp_sys == 'BW': # run on Bluewaters
            
                """determine number of nodes/hours needed - 12 nodes/100 TDT cells = ~30min"""
                ncells = len(self.cell_names)
                node_type = self.config.get("job-options", "node_type")
                if node_type == 'xe':
                    nodes = np.int(np.maximum(2, np.round(ncells*(8/60))))
                    nodereq = "nodes={0}:ppn=32:xe".format(nodes)
                    nparts = np.int(nodes*4)
                elif node_type == 'xk':
                    nodes = np.int(np.maximum(3, np.round(ncells*(16/60))))
                    nodereq = "nodes={0}:ppn=16:xk".format(nodes)
                    nparts = np.int(nodes*2)

                request_str = "Requesting {0} {1} nodes ({2} threads) for 2 hour".format(
                    nodes, node_type, nparts)
                print(request_str)              
                with open(self.log_file, 'a') as lf: lf.write('\n{0}\n'.format(request_str))

                pbs_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'pbs_logs')
                sub_log = os.path.join(pbs_dir, "CreateDTM_{0}.txt".format(self.projectname))
                if not os.path.exists(pbs_dir): os.makedirs(pbs_dir)
                if os.path.exists(sub_log): os.remove(sub_log)

                args1 = "configfile='{0}'".format(self.config_file_auto)
                args2 = "venv='{0}'".format(venv)        
                args3 = "nnodes='{0}'".format(nparts)        
                jobname = 'createdtm_{0}'.format(self.projectname)
                email = self.config.get("job-options", "email")           
                job_file = os.path.join(pydem_path,'job_files','create_DTM.job')
                queue = self.config.get("job-options", "queue")

                sub_call = "qsub -v {1},{2},{3} -l {7} -N {4} -M {5} -o {8} {0} -q {6}".format(
                            job_file,args1,args2,args3,jobname,email,queue,nodereq,sub_log)   

            else: # run on a single node
                log_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'temp_logs')
                sub_log = os.path.join(log_dir, "CreateDTM_{0}.txt".format(self.projectname))
                if not os.path.exists(log_dir): os.makedirs(log_dir)
                if os.path.exists(sub_log): os.remove(sub_log)
                create_dtm_file = os.path.join(pydem_path, 'models','CreateDTM.py')
                sub_call = "aprun -n 1 -N 1 bwpy-environ python {1} -c {0} -l {2}".format(
                    self.config_file_auto, create_dtm_file, sub_log)    
                
        else:
            log_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'temp_logs')
            sub_log = os.path.join(log_dir, "CreateDTM_{0}.txt".format(self.projectname))
            if not os.path.exists(log_dir): os.makedirs(log_dir)
            if os.path.exists(sub_log): os.remove(sub_log)
            create_dtm_file = os.path.join(pydem_path, 'models','CreateDTM.py')
            sub_call = "python {1} -c {0} -l {2}".format(
                self.config_file_auto, create_dtm_file, sub_log)    
        
        return sub_call, sub_log
    
       
    def build_flatten_rivers_call(self, pydem_path, comp_sys, nbasins=1, venv='', iteration=1):
                                
        if comp_sys == 'BW': # run on Bluewaters

            if self.run_w_mpi: #run in parallel 
                """determine number of nodes/hours needed"""
                node_type = self.config.get("job-options", "node_type")
                pbs_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'pbs_logs')
                
                
                if iteration == 1:
                    jobname = 'water_flatten_pass1_{0}'.format(self.projectname)
                    job_file = os.path.join(pydem_path,'job_files','flatten_rivers_pass1.job')
                    sub_log = os.path.join(pbs_dir, "FlattenRivers_a_{0}.txt".format(self.projectname))
                    node_div, node_split = 24, 2
                    nhours = 1

                elif iteration == 2:
                    jobname = 'water_flatten_pass2_{0}'.format(self.projectname)
                    job_file = os.path.join(pydem_path,'job_files','flatten_rivers_pass2.job')
                    sub_log = os.path.join(pbs_dir, "FlattenRivers_b_{0}.txt".format(self.projectname))
                    node_div, node_split = 2, 1
                    nhours = 16 
                    node_type = 'xe'
                
                if node_type == 'xe':
                    #if nbasins ==1: 
                    #    nodes = 1
                    #else:
                    nodes = np.int(np.maximum(3, np.round(nbasins/node_div)))
                    nodereq = "nodes={0}:ppn=32:xe".format(nodes)
                    nparts = np.int(nodes*node_split)
                elif node_type == 'xk':
                    nodes = np.int(np.maximum(2, np.round(nbasins/(node_div/2))))
                    #if nodes==2: nodes=1
                    nodereq = "nodes={0}:ppn=16:xk".format(nodes)
                    nparts = np.int(nodes)  
                
                request_str = "Requesting {0} {1} nodes ({2} threads) for {3} hours".format(
                    nodes, node_type, nparts, nhours)
                print(request_str)              
                with open(self.log_file, 'a') as lf: lf.write('\n{0}\n'.format(request_str))

                if not os.path.exists(pbs_dir): os.makedirs(pbs_dir)
                if os.path.exists(sub_log): os.remove(sub_log)

                args1 = "configfile='{0}'".format(self.config_file_auto)
                args2 = "venv='{0}'".format(venv)        
                args3 = "nnodes='{0}'".format(nparts)        
                email = self.config.get("job-options", "email")           
                queue = self.config.get("job-options", "queue")

                sub_call = "qsub -v {1},{2},{3} -l {7} -N {4} -M {5} -o {8} {0} -q {6}".format(
                            job_file,args1,args2,args3,jobname,email,queue,nodereq,sub_log)   

            else: # run on a single node
                log_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'temp_logs')
                sub_log = os.path.join(log_dir, "FindSinks_{0}.txt".format(self.projectname))
                if not os.path.exists(log_dir): os.makedirs(log_dir)
                if os.path.exists(sub_log): os.remove(sub_log)
                flatten_file = os.path.join(pydem_path, 'models','flatten_rivers.py')
                sub_call = "aprun -n 1 -N 1 bwpy-environ python {1} -c {0} -l {2}".format(
                    self.config_file_auto, flatten_file, sub_log) 

        else: # run on a single node
            log_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'temp_logs')
            sub_log = os.path.join(log_dir, "FindSinks_{0}.txt".format(self.projectname))
            if not os.path.exists(log_dir): os.makedirs(log_dir)
            if os.path.exists(sub_log): os.remove(sub_log)
            flatten_file = os.path.join(pydem_path, 'models','flatten_rivers.py')
            sub_call = "python {1} -c {0} -l {2}".format(
                self.config_file_auto, flatten_file, sub_log) 

        return sub_call, sub_log    
    
     
    def build_minimize_call(self, pydem_path, comp_sys, nbasins=1, venv=''):
        
        if comp_sys == 'BW': # run on Bluewaters       
            if self.run_w_mpi: #run in parallel            
                """determine number of nodes/hours needed"""
                node_type = self.config.get("job-options", "node_type")
                pbs_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'pbs_logs')

                jobname = 'overlap_minimize_{0}'.format(self.projectname)
                sub_log = os.path.join(pbs_dir, "overlap_minimize_{0}.txt".format(self.projectname))
                job_file = os.path.join(pydem_path,'job_files', 'minimize_basin_overlap.job')
                node_div, node_split = 30, 1
                nhours = 3

                if node_type == 'xe':
                    nodes = np.int(np.maximum(2, np.round(nbasins/node_div)))
                    nodereq = "nodes={0}:ppn=32:xe".format(nodes)
                    nparts = np.int(nodes*node_split)
                elif node_type == 'xk':
                    nodes = np.int(np.maximum(2, np.round(nbasins/(node_div/2))))
                    #if nodes==2: nodes=1
                    nodereq = "nodes={0}:ppn=16:xk".format(nodes)
                    nparts = np.int(nodes)  
                    
                request_str = "Requesting {0} {1} nodes ({2} threads) for {3} hours".format(
                    nodes, node_type, nparts, nhours)
                print(request_str)              
                with open(self.log_file, 'a') as lf: lf.write('\n{0}\n'.format(request_str))
                
                if not os.path.exists(pbs_dir): os.makedirs(pbs_dir)
                if os.path.exists(sub_log): os.remove(sub_log)
                args1 = "configfile='{0}'".format(self.config_file_auto)
                args2 = "venv='{0}'".format(venv)        
                args3 = "nnodes='{0}'".format(nparts)        
                email = self.config.get("job-options", "email")           
                queue = self.config.get("job-options", "queue")

                sub_call = "qsub -v {1},{2},{3} -l {7} -N {4} -M {5} -o {8} {0} -q {6}".format(
                            job_file,args1,args2,args3,jobname,email,queue,nodereq,sub_log)   

            else: # run on a single node
                log_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'temp_logs')
                sub_log = os.path.join(log_dir, "overlap_minimize_{0}.txt".format(self.projectname))
                if os.path.exists(sub_log): os.remove(sub_log)
                tau_check_py = os.path.join(pydem_path, 'models','minimize_basin_overlap.py')
                sub_call = "aprun -n 1 -N 1 bwpy-environ python {1} -c {0} -l {2}".format(
                    self.config_file_auto, tau_check_py, sub_log)    
                
        else: # run on a single node
            log_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'temp_logs')
            sub_log = os.path.join(log_dir, "overlap_minimize_{0}.txt".format(self.projectname))
            if os.path.exists(sub_log): os.remove(sub_log)
            tau_check_py = os.path.join(pydem_path, 'models','minimize_basin_overlap.py')
            sub_call = "python {1} -c {0} -l {2}".format(
                self.config_file_auto, tau_check_py, sub_log)    
            
            
        return sub_call, sub_log
    

    def build_sinks_call(self, pydem_path, comp_sys, nbasins=1, venv=''):
                
        if comp_sys == 'BW': # run on Bluewaters

            if self.run_w_mpi: #run in parallel 
                """determine number of nodes/hours needed - 12 nodes/100 TDT cells = ~30min"""
                node_type = self.config.get("job-options", "node_type")
                if node_type == 'xe':
                    nodes = np.int(np.maximum(2, np.round(nbasins*(4/60)/12)))
                    #if nodes==2: nodes=1
                    nodereq = "nodes={0}:ppn=32:xe".format(nodes)
                    nparts = np.int(nodes*4)
                elif node_type == 'xk':
                    nodes = np.int(np.maximum(3, np.round(nbasins*(8/60)/12)))
                    #if nodes==2: nodes=1
                    nodereq = "nodes={0}:ppn=16:xk".format(nodes)
                    nparts = np.int(nodes*2)     

                request_str = "Requesting {0} {1} nodes ({2} threads) for 10 min".format(
                    nodes, node_type, nparts)
                print(request_str)              
                with open(self.log_file, 'a') as lf: lf.write('\n{0}\n'.format(request_str))

                pbs_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'pbs_logs')
                sub_log = os.path.join(pbs_dir, "FindSinks_{0}.txt".format(self.projectname))
                if not os.path.exists(pbs_dir): os.makedirs(pbs_dir)
                if os.path.exists(sub_log): os.remove(sub_log)

                args1 = "configfile='{0}'".format(self.config_file_auto)
                args2 = "venv='{0}'".format(venv)        
                args3 = "nnodes='{0}'".format(nparts)        
                jobname = 'find_sinks_{0}'.format(self.projectname)
                email = self.config.get("job-options", "email")           
                job_file = os.path.join(pydem_path,'job_files','find_sinks.job')
                queue = self.config.get("job-options", "queue")

                sub_call = "qsub -v {1},{2},{3} -l {7} -N {4} -M {5} -o {8} {0} -q {6}".format(
                            job_file,args1,args2,args3,jobname,email,queue,nodereq,sub_log)   

            else: # run on a single node
                log_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'temp_logs')
                sub_log = os.path.join(log_dir, "FindSinks_{0}.txt".format(self.projectname))
                if not os.path.exists(log_dir): os.makedirs(log_dir)
                if os.path.exists(sub_log): os.remove(sub_log)
                find_sinks_file = os.path.join(pydem_path, 'models','FindSinks.py')
                sub_call = "aprun -n 1 -N 1 bwpy-environ python {1} -c {0} -l {2}".format(
                    self.config_file_auto, find_sinks_file, sub_log)    
            
        else: # run on a single node
            log_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'temp_logs')
            sub_log = os.path.join(log_dir, "FindSinks_{0}.txt".format(self.projectname))
            if not os.path.exists(log_dir): os.makedirs(log_dir)
            if os.path.exists(sub_log): os.remove(sub_log)
            find_sinks_file = os.path.join(pydem_path, 'models','FindSinks.py')
            sub_call = "python {1} -c {0} -l {2}".format(
                self.config_file_auto, find_sinks_file, sub_log)   


        return sub_call, sub_log    


    def build_basin_call(self, pydem_path, comp_sys, nbasins=1, venv='', iteration=1):
        
        
        if comp_sys == 'BW': # run on Bluewaters       
            if self.run_w_mpi: #run in parallel            
                """determine number of nodes/hours needed"""
                node_type = self.config.get("job-options", "node_type")
                pbs_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'pbs_logs')
            
                if iteration == 1:
                    jobname = 'dephandle_lvl6a_{0}'.format(self.projectname)
                    sub_log = os.path.join(pbs_dir, "dephandle_lvl6a_{0}.txt".format(self.projectname))
                    job_file = os.path.join(pydem_path,'job_files','dep_handling_lvl6_pass1.job')
                    node_div, node_split = 24, 2
                    nhours = 1

                elif iteration == 2:
                    jobname = 'dephandle_lvl6b_{0}'.format(self.projectname)
                    sub_log = os.path.join(pbs_dir,"dephandle_lvl6b_{0}.txt".format(self.projectname))
                    job_file = os.path.join(pydem_path,'job_files', 'dep_handling_lvl6_pass2.job')
                    node_div, node_split = 4, 2
                    nhours = 2         
                
                elif iteration == 3:
                    jobname = 'dephandle_lvl6c_{0}'.format(self.projectname)
                    sub_log = os.path.join(pbs_dir,"dephandle_lvl6c_{0}.txt".format(self.projectname))
                    job_file = os.path.join(pydem_path,'job_files', 'dep_handling_lvl6_pass3.job')
                    node_type = 'xe'
                    node_div, node_split = 2, 1
                    nhours = 24         
                
                if node_type == 'xe':
                    if nbasins == 1: 
                        nodes = 2
                    else:
                        nodes = np.int(np.maximum(3, np.round(nbasins/node_div)))
                    nodereq = "nodes={0}:ppn=32:xe".format(nodes)
                    nparts = np.int(nodes*node_split)
                elif node_type == 'xk':
                    nodes = np.int(np.maximum(2, np.round(nbasins/(node_div/2))))
                    #if nodes==2: nodes=1
                    nodereq = "nodes={0}:ppn=16:xk".format(nodes)
                    nparts = np.int(nodes)  
                    
                request_str = "Requesting {0} {1} nodes ({2} threads) for {3} hours".format(
                    nodes, node_type, nparts, nhours)
                print(request_str)              
                with open(self.log_file, 'a') as lf: lf.write('\n{0}\n'.format(request_str))
                
                if not os.path.exists(pbs_dir): os.makedirs(pbs_dir)
                if os.path.exists(sub_log): os.remove(sub_log)
                args1 = "configfile='{0}'".format(self.config_file_auto)
                args2 = "venv='{0}'".format(venv)        
                args3 = "nnodes='{0}'".format(nparts)        
                email = self.config.get("job-options", "email")           
                queue = self.config.get("job-options", "queue")

                sub_call = "qsub -v {1},{2},{3} -l {7} -N {4} -M {5} -o {8} {0} -q {6}".format(
                            job_file,args1,args2,args3,jobname,email,queue,nodereq,sub_log)   

            else: # run on a single node
                log_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'temp_logs')
                sub_log = os.path.join(log_dir, "dephandle_lvl6_{0}.txt".format(self.projectname))
                if os.path.exists(sub_log): os.remove(sub_log)
                dep_handle_py = os.path.join(pydem_path, 'models','depHandle_lvl6_config.py')
                sub_call = "aprun -n 1 -N 1 bwpy-environ python {1} -c {0} -l {2}".format(
                    self.config_file_auto, dep_handle_py, sub_log)    
                
        else: # run on a single node
            log_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'temp_logs')
            sub_log = os.path.join(log_dir, "dephandle_lvl6_{0}.txt".format(self.projectname))
            if os.path.exists(sub_log): os.remove(sub_log)
            dep_handle_py = os.path.join(pydem_path, 'models','depHandle_lvl6_config.py')
            sub_call = "python {1} -c {0} -l {2}".format(
                self.config_file_auto, dep_handle_py, sub_log)   
            
            
        return sub_call, sub_log
    
    
    def build_taudem_call(self, inDEM, pydem_path, comp_sys, venv='', max_allocation=False):
        
        wdir = os.path.dirname(inDEM)
        
        if comp_sys == 'BW': # run on Bluewaters
            project_cells_txt = os.path.join(self.config.get("paths-DTM", "project_txt_path"),
                                            '{0}.txt'.format(self.projectname))     
            self.cell_names = open(project_cells_txt).read().splitlines()               
            
            """determine number of nodes/hours needed - node-hours ~ cells/4.5"""
            ncells = len(self.cell_names)
            # node_type = self.config.get("job-options", "node_type")
            node_type = 'xe'
            if node_type == 'xe':

                if max_allocation:
                    node_hours = 150
                else:
                    node_hours = ncells/4.5
                hours = 3  # default walltime = 2:00
                minutes = 0
                if node_hours <= 1:
                    nodes = 1
                elif node_hours <= 124:  # increase nodes up to 64
                    nodes = np.int(np.maximum(2, np.round(node_hours)))
                else:  # past 64 node-hours, increase walltime instead of nodes
                    nodes = 124
                    hours_exact = node_hours/nodes
                    hours = np.int(np.ceil(hours_exact) + 3)
#                     minutes = np.int(np.round((hours_exact - hours)*60))
#                 time_formatted = "{0}:{1}".format(hours, str(minutes).zfill(2))
                time_formatted = "{0}:00:00".format(hours)

                nodereq = "nodes={0}:ppn=32:xe".format(nodes)


            request_str = "Requesting {0} {1} nodes for {2}".format(
                nodes, node_type, time_formatted)
            print(request_str)              
            with open(self.log_file, 'a') as lf: lf.write('\n{0}\n'.format(request_str))

            pbs_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'pbs_logs')
            sub_log = os.path.join(pbs_dir, "TauDEM_{0}.txt".format(self.projectname))
            if not os.path.exists(pbs_dir): os.makedirs(pbs_dir)
            if os.path.exists(sub_log): os.remove(sub_log)

            args1 = "configfile='{0}'".format(self.config_file_auto)
            args2 = "venv='{0}'".format(venv)        
            args3 = "wdir='{0}'".format(wdir)
            walltime = "walltime={0}".format(time_formatted)
            jobname = 'taudem_{0}'.format(self.projectname)
            email = self.config.get("job-options", "email")           
            job_file = os.path.join(pydem_path,'job_files','taudem.job')
            queue = self.config.get("job-options", "queue")

            sub_call = "qsub -v {1},{2},{3} -l {4} -l {8} -N {5} -M {6} -o {9} {0} -q {7}".format(
                        job_file,args1,args2,args3, walltime,jobname,email,queue,nodereq,sub_log)   
            
        else:
            raise 'Can currently only be run on Bluewaters...'
        
        return sub_call, sub_log  
    
       
    def build_taudem_check_call(self, pydem_path, comp_sys, nbasins=1, venv=''):
                        
        
        if comp_sys == 'BW': # run on Bluewaters       
            if self.run_w_mpi: #run in parallel            
                """determine number of nodes/hours needed"""
                node_type = self.config.get("job-options", "node_type")
                pbs_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'pbs_logs')

                jobname = 'taudem_check_{0}'.format(self.projectname)
                sub_log = os.path.join(pbs_dir, "taudem_check_{0}.txt".format(self.projectname))
                job_file = os.path.join(pydem_path,'job_files', 'check_taudem.job')
                node_div, node_split = 100, 2
                nhours = 2


                if node_type == 'xe':
                    nodes = np.int(np.maximum(2, np.round(nbasins/node_div)))
                    nodereq = "nodes={0}:ppn=32:xe".format(nodes)
                    nparts = np.int(nodes*node_split)
                elif node_type == 'xk':
                    nodes = np.int(np.maximum(2, np.round(nbasins/(node_div/2))))
                    #if nodes==2: nodes=1
                    nodereq = "nodes={0}:ppn=16:xk".format(nodes)
                    nparts = np.int(nodes)  
                    
                request_str = "Requesting {0} {1} nodes ({2} threads) for {3} hours".format(
                    nodes, node_type, nparts, nhours)
                print(request_str)              
                with open(self.log_file, 'a') as lf: lf.write('\n{0}\n'.format(request_str))
                
                if not os.path.exists(pbs_dir): os.makedirs(pbs_dir)
                if os.path.exists(sub_log): os.remove(sub_log)
                args1 = "configfile='{0}'".format(self.config_file_auto)
                args2 = "venv='{0}'".format(venv)        
                args3 = "nnodes='{0}'".format(nparts)        
                email = self.config.get("job-options", "email")           
                queue = self.config.get("job-options", "queue")

                sub_call = "qsub -v {1},{2},{3} -l {7} -N {4} -M {5} -o {8} {0} -q {6}".format(
                            job_file,args1,args2,args3,jobname,email,queue,nodereq,sub_log)   

            else: # run on a single node
                log_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'temp_logs')
                sub_log = os.path.join(log_dir, "taudem_check_{0}.txt".format(self.projectname))
                if os.path.exists(sub_log): os.remove(sub_log)
                tau_check_py = os.path.join(pydem_path, 'models','taudem_check.py')
                sub_call = "aprun -n 1 -N 1 bwpy-environ python {1} -c {0} -l {2}".format(
                    self.config_file_auto, tau_check_py, sub_log)  

        else: # run on a single node
            log_dir = os.path.join(self.config.get("paths-all", "output_log_dir"),'temp_logs')
            sub_log = os.path.join(log_dir, "taudem_check_{0}.txt".format(self.projectname))
            if os.path.exists(sub_log): os.remove(sub_log)
            tau_check_py = os.path.join(pydem_path, 'models','taudem_check.py')
            sub_call = "python {1} -c {0} -l {2}".format(
                self.config_file_auto, tau_check_py, sub_log)  
            
            
        return sub_call, sub_log