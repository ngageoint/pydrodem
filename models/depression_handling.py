#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 26 2020.

@author: Kimberly McCormack, Heather Levin, Amy Morris

Last edited on: 5/12/2021


"""

import gc
import os
import sys
import json
import time
import traceback
import warnings
import subprocess
import geopandas as gpd
import numpy as np
import argparse
from configparser import ConfigParser, ExtendedInterpolation
from osgeo import gdal, gdal_array
from scipy import ndimage


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.convert as pyct
import tools.depressions as dep
import tools.files as ft
import tools.log as logt
import tools.water as wtr
from tools.print import print_time, log_time
from tools import shapes
from tools import endo


gdal.UseExceptions()
warnings.filterwarnings(action='ignore', message='GEOS')
# gdal.PushErrorHandler('CPLQuietErrorHandler')


class DepHandle():
    """
    Class to fix depression within a DEM.

    Attributes
    ----------
    inDS_vrt : path
        path to mosaic of DEM data (vrt or geotiff format)
    shpfile_project : path
        path to .shp file defining outline of entire region to process
    work_dir : path
        output directory
    projectname : str
        name of mosaic geotiff file
    basin_levels : list
        list of hydrobasin levels to use for basin processing -- e.g [6,12]
    WBT_path : path
        path to whitebox tools directory

    Keyword Arguments
    -----------------
    taudem_dir : path
        output directory for local taudem operations.
        Not output for final Taudem runs.
        Only used for detecting deviations in computed stream
        network and building noflow walls
        Default=None.
    overwrite : boolean
        erase .tif  files for basins. Default=False
    run_parallel : boolean
        run in parallel using mpi4py. Default=True
    verbose_print : boolean
        print processing steps/timestamps? Default=True
    del_temp_files : boolean
        delete temporary depression handling tifs? Default=False
    noflow_bound : boolean
        build temp wall to force no flow boundary? Default=False
    burn_rivers: boolean
        Flatten and burn in streams and rivers based on SWO and OSM?
        Should only be used at one processing level (level 6, preferred)
        Default=False.
    reservoir_barrier: boolean
        build temp wall for reservoir crossings? Default=False.
    fill_h20_adjacent : boolean
        auto fill pits that drain into large water bodies?
        Can help fix depressions along large rivers that are a result
        of overestimating vegetation bias.
        Best used once at smallest basin level. Default = False.
    buffer : int or float
        Buffer (in meters) around basin shape to process. Default=100
    sfill : float
        Threshold (in meters) for depression depth to automatically apply
        fill solution. Default=2.0
    fill_single_cell_pits : boolean
        Fill/carve single cell pits before running depression handling?
        Default=True
    flat_increment : boolean
        gradient to set for flat regions
        Default = 0.0. Reasonable value = 0.00001
    fill_method : str
        WhiteboxTools filling method.
        "SF" for standard fill, "WL" for Wang and Liu. Default="WL"
    carve_method : str
        WhiteboxTools carving method.
        'LC'=least cost path, 'SC' = standard carving. Default='SC'
    maxcost : int or float
        maxcost limit for least cost breaching algorithm.
        Only applied to Least Cost Breaching. Default=200
    min_dist : boolean
        Minimize carve distance in least cost carving?
        Only applied to Least Cost Breaching. Default=False
    carveout : boolean
        Allow carves to leave basin? Default=False
    radius : int
        Search radius (in pixels) for carving algorithms. Default=25
    fill_vol_thresh : int or float
        fill volume threshold to attempt a partial fill solution. Default=12000
    carve_vol_thresh : int or float
        carve volume threshold to attempt a partial fill solution. Default=4000
    max_carve_depth : int or float
        Maximum allowable carving depth (meters). Default=100
    combined_fill_interval : float
        Fill increment for partial fill search [0,1]. Default=0.2




    methods
    -------
    open_shp_files():
       Open basins/subbasins shape files as geodataframes

    fillcarve_single_cell_pits(inDEM, outDEM):
        breach and then fill any single cell pits within the DEM

    dep_handle_preprosses(inDEM, outDEM):
        Run single cell fill and carve algorithm

    basin_setup_single_lvl(input_id, basins_gdf):
        Set up basin directories, shape files and buffered shape files
        for a single basin level

    basin_setup_multi_lvl(input_id, basins_p1_shp, basins_p2_shp):
        Set up basin directories, shape files and buffered shape files
        for a multiple basin levels (3 max)

    depression_handle_basin(input_id):
        Run depression handling process on a basin

    burn_rivers(inDEM, outpath, SWO_tif, flatten_mask_tif)
        Flattens and burns continous river segment

    build_reservoir_barrier(inDEM, outDEM, outpath, SWO_tif):
        build temp wall upstream os large reservoirs to ensure carve through dam

    build_noflow_wall(inDEM, input_id, outpath,
                          buffer_shp, wall_array):
        Create boundary wall at locations with large flow accumulations leaving
        basin boundary, except for outlets/inlets

    remove_wall(fill_tif, inDEM_array, wall_array,
                    solution_mask_array, noflow_wall_ind):
        Remove noflow wall


    """

    def __init__(self, inDS_vrt, work_dir,
                 SWO_vrt, projectname, basin_levels, WBT_path,
                 base_unit='tile', OSM_dir=None, edm_vrt=None,
                 taudem_dir=None, overwrite=False, run_parallel=True,
                 verbose_print=True,
                 del_temp_files=False, noflow_bound=False, burn_rivers=False,
                 flattened_mask_vrt=None,
                 basins_merge_shp=None, basins_dissolve_shp=None,
                 reservoir_barrier=False, fill_h20_adjacent=False,
                 buffer=100, coastal_buffer=0,
                 sfill=2.0, fill_single_cell_pits=False,
                 flat_increment=0.0, fill_method="WL",
                 carve_method='SC', maxcost=200, carveout=False,
                 radius=25, min_dist=False, wall_height=50.0,
                 fill_vol_thresh=12000, carve_vol_thresh=4000,
                 max_carve_depth=100, combined_fill_interval=.2,
                 enforce_endo=False, enforce_monotonicity=True,
                 noflow_points=None, wall_remove_min=0, osm_coast_clip=True):

        self.WBT_path = WBT_path
        self.starttime = time.time()
        self.inDS_vrt = inDS_vrt
        self.work_dir = work_dir
        self.base_unit = base_unit
        self.SWO_vrt = SWO_vrt
        self.OSM_dir = OSM_dir
        self.flattened_mask_vrt = flattened_mask_vrt
        self.taudem_work_dir = taudem_dir
        self.projectname = projectname
        self.basin_levels = basin_levels
        self.overwrite = overwrite
        self.run_parallel = run_parallel
        self.verbose_print = verbose_print
        self.del_temp_files = del_temp_files
        self.noflow_bound = noflow_bound
        self.reservoir_barrier = reservoir_barrier
        self.burn_rivers = burn_rivers
        self.buffer = buffer
        self.basins_merge_shp = basins_merge_shp
        self.basins_dissolve_shp = basins_dissolve_shp
        self.coastal_buffer = coastal_buffer
        self.sfill = sfill
        self.fill_single_cell_pits = fill_single_cell_pits
        self.flat_increment = flat_increment
        self.fill_method = fill_method
        self.carvemethod = carve_method
        self.maxcost = maxcost
        self.carveout = carveout
        self.radius = radius
        self.min_dist = min_dist
        self.wall_height = wall_height
        self.fill_vol_thresh = fill_vol_thresh
        self.carve_vol_thresh = carve_vol_thresh
        self.max_carve_depth = max_carve_depth
        self.combined_fill_interval = combined_fill_interval
        self.fill_h20_adjacent = fill_h20_adjacent
        self.enforce_endo = enforce_endo
        self.enforce_monotonicity = enforce_monotonicity
        self.endo_work_dir = os.path.join(os.path.dirname(self.work_dir),
                                          'endo_basins')
        self.noflow_points = noflow_points
        self.wall_remove_min = wall_remove_min
        self.osm_coast_clip = osm_coast_clip
        self.edm_mosaic = edm_vrt

        """Load WhiteBox Tools executable"""
        sys.path.append(WBT_path)
        from whitebox_tools import WhiteboxTools
        self.wbt = WhiteboxTools()
        self.wbt.set_verbose_mode(False)
        self.wbt.set_whitebox_dir(WBT_path)  # set location of WBT executable

        """"""""""""""""""""""""""""""""""""""""""""""""""""""
        ###########  SETUP OUTPUT DIRECTORIES ################
        """"""""""""""""""""""""""""""""""""""""""""""""""""""
        """Find project mosaic file"""
        self.project_SWOtif = self.SWO_vrt
        if not os.path.exists(self.inDS_vrt):
            print('mosaic does not exist!', self.inDS_vrt)
            sys.exit()

    def open_shp_files(self, basins_shpfile):
        """Open basins/subbasins shape files as geodataframes."""
        self.basins_gdf = gpd.read_file(basins_shpfile)

    def fillcarve_single_cell_pits(self, inDEM, outDEM):
        """Breach single cell pits."""
        self.wbt.breach_single_cell_pits(inDEM, outDEM)
        """Fill single cell pits"""
        self.wbt.fill_single_cell_pits(outDEM, outDEM)

    def run_depression_handling(self, input_id, SWO_cutoff=20,
                                initial_burn=2.0,
                                SWO_burn=2.0, burn_cutoff=6.0, gap_dist=240,
                                apply_relative_min=True):
        """
        Run depressioning handling for a specific basin/tile.

        Parameters
        ----------
        input_id : int
            hydrobasin ID number
        SWO_cutoff : int, optional
            Surface Water occurence value to distiguish land from water
            [0,100], by default 20
        initial_burn : float, optional
            Depth to burn in all river values, by default 2.0
        SWO_burn : float, optional
            Additional burn to apply to high SWO values, with a ramp
            applied from the SWO cutoff up to SWO=80. SWO_burn depth
            is added to any value over 80, by default 2.0
        burn_cutoff : float, optional
            maximum allowable elevation change for river flattening process,
            before initial or SWO-based burn is applied, by default 6.0
        gap_dist : int, optional
            allowable gap between water features to be condidered
            'continuous' (meters), by default 240
        """

        looptime = time.time()
        base_unit = self.base_unit
        print_time(f" process {base_unit} {input_id}", self.starttime)

        if base_unit == 'basin':
            self.usecutline = True
            capstyle = 1
        else:
            self.usecutline = False
            capstyle = 3  # square caps

        """"""""""""""""""""""""""""""""
        """      Load/Setup Data     """
        """"""""""""""""""""""""""""""""
        try:

            self.outpath = os.path.join(self.work_dir,
                                        f'{base_unit}_{input_id}')
            outpath = self.outpath
            self.input_id = input_id

            errorlog = os.path.join(outpath, 'error_log_main.txt')
            if os.path.exists(errorlog):
                os.remove(errorlog)

            completed_file = os.path.join(outpath, 'DEMcompleted.txt')
            if os.path.exists(completed_file): os.remove(completed_file)

            """set outputs"""
            outDEM_tif = os.path.join(outpath, "conditionedDEM.tif")
            if os.path.exists(outDEM_tif):
                os.remove(outDEM_tif)
            self.pits_df_out = os.path.join(outpath, 'DFpits.h5')
            self.level = np.int(str(input_id)[2])

            bbox_shpfile = os.path.join(
                outpath, '{0}_bbox.shp'.format(input_id))
            if os.path.exists(bbox_shpfile):
                input_shp = bbox_shpfile
            else:
                input_shp = os.path.join(
                    outpath, f'{base_unit}_{input_id}.shp')

            self.input_gdf = gpd.read_file(input_shp)
            self.input_gdf = shapes.close_holes_gdf(self.input_gdf)
            buffer = self.buffer
            buffer_outer = buffer + 50.0  # add 50 meters to buffer

            """test for coastal basins"""
            self.coastal = False
            coast_file = os.path.join(outpath, 'coast.txt')
            if os.path.exists(coast_file):
                print(f'  On {input_id} - coastal basin')
                buffer = self.buffer + self.coastal_buffer
                SWO_cutoff = 65
                self.coastal = True

            inDEM = os.path.join(outpath, 'inDEM.tif')
            self.basename = os.path.splitext(os.path.basename(inDEM))[0]
            basename = self.basename
            nodataval = -9999

            if self.overwrite:
                ft.erase_files(outpath, search='*.h5')
            """erase temp files before re-processing"""
            ft.erase_files(outpath, search=f'{basename}*')

            """ create outer buffer"""
            self.buffer_shp = os.path.join(
                outpath, f'{self.basename}_buffer.shp')
            if not os.path.exists(self.buffer_shp):
                pyct.buffer_gdf2shp(self.input_gdf, buffer_outer,
                                    self.buffer_shp, capstyle=capstyle)
            self.buffer_gdf = gpd.read_file(self.buffer_shp)
               
            if not os.path.exists(inDEM):
                pyct.create_cutline_raster_set_res(self.inDS_vrt,
                                                   self.buffer_shp,
                                                   inDEM,
                                                   nodata=nodataval,
                                                   usecutline=self.usecutline)
            inDEM_array = gdal_array.LoadFile(inDEM)
            
            
            """Re-build buffer shape - excluding nodata values"""
            data_mask = np.where(inDEM_array==nodataval, 0, 1)
            data_mask_tif = os.path.join(outpath, f'{self.basename}_data_mask.tif')
            pyct.npy2tif(data_mask, inDEM,
                         data_mask_tif, nodata=0, 
                         dtype=gdal.GDT_Byte)
            self.wbt.raster_to_vector_polygons(data_mask_tif, self.buffer_shp)
            self.buffer_gdf = gpd.read_file(self.buffer_shp)

            """ create basin crop shape from buffered shape"""
            self.crop_shp = os.path.join(
                outpath, f'{self.basename}_crop.shp')
            if not os.path.exists(self.crop_shp):
                pyct.buffer_gdf2shp(self.buffer_gdf, -50,
                                    self.crop_shp, capstyle=capstyle)
            self.crop_gdf = gpd.read_file(self.crop_shp)

            """ create shape inside crop shape for non-mono stream search"""
            self.inner_crop_shp = os.path.join(
                outpath, f'{self.basename}_inner_crop.shp')
            if not os.path.exists(self.inner_crop_shp):
                pyct.buffer_gdf2shp(self.crop_gdf, -50,
                                    self.inner_crop_shp, capstyle=capstyle)
            self.inner_crop_gdf = gpd.read_file(self.inner_crop_shp)


            """create a 50m wide 'donut' outside the buffered basin shape"""
            self.donut_shp = os.path.join(
                outpath, f"{self.basename}_boundary_donut.shp")
            if not os.path.exists(self.donut_shp):
                self.donut_gdf = gpd.overlay(
                    self.buffer_gdf, self.crop_gdf, how='difference')
                self.donut_gdf.reset_index(drop=True, inplace=True)
                self.donut_gdf.to_file(self.donut_shp)
            else:
                self.donut_gdf = gpd.read_file(self.donut_shp)


            """Crop VRTs to buffered basin"""
            SWO_tif = os.path.join(outpath, f'{self.basename}_SWO.tif')
            if not os.path.exists(SWO_tif):
                pyct.create_cutline_raster(self.project_SWOtif,
                                           self.buffer_shp,
                                           SWO_tif, srcDS=inDEM,
                                           srcnodata=255, nodata=0,
                                           usecutline=self.usecutline,
                                           outdtype=gdal.GDT_Byte,
                                           rsAlg=gdal.GRIORA_Bilinear,
                                           paletted=True)
                self.SWO_array = gdal_array.LoadFile(SWO_tif)

            if self.flattened_mask_vrt is not None:
                water_mask_tif = os.path.join(
                    outpath, f"{self.basename}_Water_mask.tif")
                if not os.path.exists(water_mask_tif):
                    pyct.create_cutline_raster(self.flattened_mask_vrt,
                                               self.buffer_shp,
                                               water_mask_tif, srcDS=inDEM,
                                               srcnodata=0, nodata=0,
                                               usecutline=self.usecutline,
                                               outdtype=gdal.GDT_Byte)
                water_mask_array = gdal_array.LoadFile(water_mask_tif)
            else:
                water_mask_array = None

            """"""""""""""""""""""""""""""""""""""
            """      Build boundary masks      """
            """"""""""""""""""""""""""""""""""""""
            self.blankmask_tif = os.path.join(
                outpath, f"{self.basename}_blank_mask.tif")
            rastermask = np.ones_like(inDEM_array, dtype=np.int8)
            pyct.npy2tif(rastermask, inDEM, self.blankmask_tif,
                         nodata=0, dtype=gdal.GDT_Byte)

            """use cutline to isolate pixels along basin boundary"""
            donut_mask = os.path.join(
                outpath, f"{self.basename}_boundary_donut.tif")
            if not os.path.exists(donut_mask):
                pyct.cutline_raster_simple(self.blankmask_tif,
                                           self.donut_shp, donut_mask,
                                           nodata=0, outdtype=gdal.GDT_Byte)

            self.donut_array = gdal_array.LoadFile(donut_mask)
            self.wall_array = self.donut_array
            
            """"""""""""""""""""""""""""""""
            """ Run Depression Handling  """
            """"""""""""""""""""""""""""""""

            outDEM_array, \
                water_mask = self.depression_handling(input_id,
                                                      inDEM,
                                                      SWO_tif,
                                                      water_mask_array)

            """"""""""""""""""""""""""""""""
            """       Final outputs      """
            """"""""""""""""""""""""""""""""
            outDEM_tif_temp = inDEM.replace('.tif', '_conditionedTEMP.tif')
            pyct.npy2tif(outDEM_array, inDEM, outDEM_tif_temp,
                         nodata=self.nodataval, dtype=gdal.GDT_Float32)

            """Run final fill"""
            if self.verbose_print:
                print_time(f"   On {input_id} run final fill", self.starttime)
            ffill_array, fdiff_fill_array = dep.fill_pits(
                outDEM_tif_temp, self.wbt,
                save_tif=False, nodataval=self.nodataval)
            pyct.npy2tif(ffill_array, inDEM, outDEM_tif_temp,
                         nodata=self.nodataval, dtype=gdal.GDT_Float32)

            """ Crop final tif to inner buffer and set nodata value """
            pyct.create_cutline_raster(outDEM_tif_temp, self.crop_shp,
                                       outDEM_tif, srcDS=inDEM,
                                       usecutline=self.usecutline,
                                       nodata=self.nodataval,
                                       outdtype=gdal.GDT_Float32)

            """ Set endorheic sink pixels to nodata in all basins """
            if self.enforce_endo:
                outDEM_array = self.enforce_sinks(input_id,
                                                  outpath,
                                                  outDEM_tif,
                                                  nodataval=self.nodataval)
            else:
                outDEM_array = ffill_array

            """Save Water-flattening mask"""
            flatten_mask_tif = os.path.join(
                outpath, "Water_mask.tif")
            pyct.npy2tif(water_mask_array, inDEM, flatten_mask_tif,
                         nodata=0, dtype=gdal.GDT_Byte)
            pyct.tifcompress(flatten_mask_tif)

            """Difference input and conditioned DEMS"""
            nodata_ind = np.nonzero(outDEM_array == self.nodataval)
            diff_tif_fill = outDEM_tif.replace('.tif', '_DIFF.tif')
            diff_fill_array = outDEM_array - inDEM_array
            #diff_fill_array[nodata_ind] = np.nan
            pyct.npy2tif(diff_fill_array, inDEM, diff_tif_fill,
                         nodata=self.nodataval, dtype=gdal.GDT_Float32)

            """reset nodataval and add sinks in output DEM"""
            os.remove(outDEM_tif)
            #outDEM_array[nodata_ind] = np.nan
            pyct.npy2tif(outDEM_array, inDEM, outDEM_tif,
                         nodata=self.nodataval, dtype=gdal.GDT_Float32)

            """Compress outputs"""
            pyct.tifcompress(diff_tif_fill)
            pyct.tifcompress(outDEM_tif)
            with open(completed_file, 'w') as f:
                f.write('DEM processing complete')

            print_time(f"    Solutions applied for {input_id}", looptime)

            """Erase temp files"""
            if self.del_temp_files:
                ft.erase_files(outpath, search=f"{self.basename}*")

        except Exception as e:
            outst = f"Exception occurred on {input_id}: {e}"
            traceback_output = traceback.format_exc()
            print(outst)
            print(traceback_output)
            pass

    def depression_handling(self, input_id, inDEM, SWO_tif, water_mask_array,
                            SWO_cutoff=20, initial_burn=2.0,
                            SWO_burn=2.0, burn_cutoff=6.0, gap_dist=240,
                            apply_relative_min=True):
        """Extract raster shape and nodataval"""
        raster = gdal.Open(inDEM, gdal.GA_ReadOnly)
        band = raster.GetRasterBand(1)
        self.nodataval = band.GetNoDataValue()   # Set nodata value
        raster = None
        band = None

        """"""""""""""""""""""""""""""""""""""""""""""""""""""
        """     Check for and log flat/nan/water basins    """
        """"""""""""""""""""""""""""""""""""""""""""""""""""""
        outpath = self.outpath
        DEMarray = gdal_array.LoadFile(inDEM)
        SWOarray = gdal_array.LoadFile(SWO_tif)
        real_ind = np.where((DEMarray != self.nodataval))
        DEMarray[DEMarray == self.nodataval] = np.nan
        water_ind = np.where((SWOarray > SWO_cutoff) &
                             (DEMarray != self.nodataval))
        num_water = len(water_ind[0])
        num_real = len(real_ind[0])

        if (num_water > 0.99 * num_real) and (not self.coastal):
            print(f'    On {input_id}: DEM is >99% water')

            return DEMarray, water_mask_array

        """"""""""""""""""""""""""""""""""""""""""""
        """       Fill/carve single cell pits    """
        """"""""""""""""""""""""""""""""""""""""""""
        if self.fill_single_cell_pits:
            inDEMpre = inDEM.replace('.tif', '_sc_fill.tif')
            if not os.path.exists(inDEMpre):
                """preprocess basin"""
                self.fillcarve_single_cell_pits(inDEM, inDEMpre)
            inDEM = inDEMpre

        waterbodymask_tif = None
        if self.fill_h20_adjacent:
            """identify pits that drain into water"""
            watermask_shp = os.path.join(
                outpath, f"{self.basename}_watermask.shp")
            watermask_gdf, water_mask_array = wtr.find_large_waterbodies(
                SWO_tif, watermask_shp, self.wbt,
                SWO_threshold=30, area_threshold=1.0)

            if watermask_gdf is not None:
                waterbodymask_tif = os.path.join(
                    outpath, f"{self.basename}_watermask.tif")
                watermask_gdf.to_file(watermask_shp)
                if not os.path.exists(waterbodymask_tif):
                    pyct.create_cutline_raster(
                        self.blankmask_tif, watermask_shp, waterbodymask_tif,
                        srcDS=SWO_tif, nodata=0, outdtype=gdal.GDT_Byte)

        """"""""""""""""""""""""""""""""""""""
        """    Apply Boundary Conditions   """
        """"""""""""""""""""""""""""""""""""""
        if self.burn_rivers and self.enforce_monotonicity:
            print_time(
                f"    On {input_id} enforce monotonicity", self.starttime)

            """create a 'donut' inside input shape
            - used  to test for non-monotonic streams"""
            non_mono_shp = os.path.join(
                outpath, f"{self.basename}_non_mono.shp")
            self.non_mono_gdf = gpd.overlay(
                self.crop_gdf, self.inner_crop_gdf, how='difference')
            self.non_mono_gdf.reset_index(drop=True, inplace=True)
            self.non_mono_gdf.to_file(non_mono_shp)
            non_mono_mask = os.path.join(
                outpath, f"{self.basename}_non_mono.tif")
            if not os.path.exists(non_mono_mask):
                pyct.cutline_raster_simple(self.blankmask_tif, non_mono_shp,
                                           non_mono_mask,
                                           nodata=0, outdtype=gdal.GDT_Byte)
            non_mono_array = gdal_array.LoadFile(non_mono_mask)

            inDEM_flatten_array, \
                water_mask_array = wtr.enforce_monotonic_rivers(
                    inDEM, outpath, DEMarray, SWOarray, water_mask_array,
                    self.wall_array, self.basename,
                    SWO_cutoff=SWO_cutoff, radius=5 * self.radius,
                    strict_endcap_geom=True, max_r_drop=50.0,
                    inner_boundary=non_mono_array, wbt=self.wbt)

            """Load OSM waterlines tif"""
            waterlines_tif_temp = os.path.join(
                outpath, f"{self.basename}_OSM_waterlines_temp.tif")
            waterlines_tif = os.path.join(
                outpath, f"{self.basename}_OSM_waterlines.tif")
            if os.path.exists(waterlines_tif):
                pyct.create_cutline_raster(waterlines_tif, self.buffer_shp,
                                           waterlines_tif_temp, srcDS=inDEM,
                                           srcnodata=0, nodata=0,
                                           usecutline=self.usecutline,
                                           outdtype=gdal.GDT_Byte)
                waterlines_tif = waterlines_tif_temp
            else:
                waterlines_tif = None

            print_time(
                f"    On {input_id} minimize river centers", self.starttime)
            inDEM_flatten_array = wtr.river_minimize(inDEM_flatten_array,
                                                     water_mask_array,
                                                     self.SWO_array,
                                                     max_burn=2,
                                                     river_filter_size=5,
                                                     nodataval=self.nodataval)

            inDEM_flatten_array[np.isnan(
                inDEM_flatten_array)] = self.nodataval
            DEM_mono_tif = os.path.join(
                outpath, f"{self.basename}_monotonic.tif")
            pyct.npy2tif(inDEM_flatten_array, inDEM, DEM_mono_tif,
                         nodata=self.nodataval, dtype=gdal.GDT_Float32)
            inDEM = DEM_mono_tif

        """ Create upstream barrier for large reservoirs """
        raised_wall_ind = None
        if (self.reservoir_barrier):
            inDEM_wall_tif = os.path.join(
                outpath, f"{self.basename}_wall.tif")

            raised_wall_ind = self.build_reservoir_barrier(
                inDEM, inDEM_wall_tif, outpath, water_mask_array,
                self.wall_array, SWO_tif,
                SWO_cutoff=SWO_cutoff)

            if os.path.exists(inDEM_wall_tif):
                inDEM = inDEM_wall_tif
                if self.verbose_print:
                    print_time(f"    On {input_id} - Reservoir barrier built",
                               self.starttime)

        """ No-flow boundary for large deviations in stream networks"""
        if self.noflow_bound:
            if self.verbose_print:
                print_time(f"    On {input_id} - Build noflow boundaries",
                           self.starttime)
            inDEM_noflow, \
                noflow_wall_ind = self.build_noflow_wall(
                    inDEM, input_id, outpath,
                    self.buffer_shp,
                    self.wall_array,
                    self.noflow_points)

            if noflow_wall_ind is not None:
                inDEM = inDEM_noflow
                if self.verbose_print:
                    print(f'  On {input_id} No flow DEM created')

        """"""""""""""""""""""""""""""""""""""""""""
        """      Save boundary condition masks   """
        """"""""""""""""""""""""""""""""""""""""""""
        if self.noflow_bound:
            if noflow_wall_ind is not None:
                noflow_wall_array = np.zeros_like(
                    self.wall_array, dtype=np.int8)
                noflow_wall_array[noflow_wall_ind] = 1
                wallmask_tif = os.path.join(
                    outpath, "Noflow_wall_mask.tif")
                pyct.npy2tif(noflow_wall_array, inDEM,
                             wallmask_tif, nodata=0, dtype=gdal.GDT_Byte)
        if (self.reservoir_barrier) and (not self.coastal):
            if raised_wall_ind is not None:
                lake_wall_array = np.zeros_like(self.wall_array, dtype=np.int8)
                lake_wall_array[raised_wall_ind] = 1
                wallmask_tif = os.path.join(
                    outpath, "reservoir_barrier_mask.tif")
                pyct.npy2tif(lake_wall_array, inDEM,
                             wallmask_tif, nodata=0, dtype=gdal.GDT_Byte)

        """"""""""""""""""""""""""""""""
        """  Initial fill and carve  """
        """"""""""""""""""""""""""""""""

        modDEM_vec, fill_vec, carve_vec,\
            diff_vec_fill, diff_vec_carve, \
            fillID_vec, carveID_vec, carvefillID_vec,\
            nx, ny = dep.initial_fill_carve(inDEM, self.basename, outpath,
                                            self.fill_method, self.maxcost,
                                            self.radius, self.min_dist,
                                            self.nodataval, self.starttime,
                                            self.wbt, sfill=self.sfill,
                                            flat_increment=self.flat_increment,
                                            watermask=waterbodymask_tif,
                                            water_mask_array=water_mask_array,
                                            carveout=False,
                                            carve_method=self.carvemethod,
                                            verbose_print=self.verbose_print,
                                            del_temp_files=self.del_temp_files)

        if self.verbose_print:
            print_time(
                f"   fill/carve for {input_id} COMPLETE", self.starttime)
            print(
                f"      On {input_id} # pits = {np.max(fillID_vec)}")

        """"""""""""""""""""""""""""""""
        """       Build catalog      """
        """"""""""""""""""""""""""""""""
        DFpits, fillcells_dct, carvecells_only_dct, \
            carvecells_dct, carve_fillcells_dct = dep.build_pit_catalog(
                modDEM_vec, self.nodataval, carve_vec, diff_vec_carve,
                diff_vec_fill, fillID_vec, carveID_vec,
                carvefillID_vec, verbose_print=self.verbose_print)
        if self.verbose_print:
            print_time(
                f"    pit catalog for {input_id} COMPLETE", self.starttime)

        """"""""""""""""""""""""""""""""
        """     Select solutions     """
        """"""""""""""""""""""""""""""""
        if (self.noflow_bound) and (noflow_wall_ind is not None):
            bound_wall_array = self.wall_array
        else:
            bound_wall_array = None

        sink_mask_tif = os.path.join(outpath, f"{self.basename}_endo_sink.tif")
        if os.path.exists(sink_mask_tif):
            sink_mask = gdal_array.LoadFile(sink_mask_tif)
        else:
            sink_mask = None

        DFpits = dep.select_solution(DFpits, fillcells_dct,
                                     carvecells_dct, carvecells_only_dct,
                                     fill_vec, carve_vec,
                                     fillID_vec, carveID_vec, self.fill_vol_thresh,
                                     self.carve_vol_thresh,
                                     self.max_carve_depth, apply_shallow_fill=False,
                                     wall_array=bound_wall_array,
                                     water_mask_array=water_mask_array, sink_array=sink_mask)

        if self.verbose_print:
            print_time(
                f"   select solutions for {input_id} COMPLETE", self.starttime)

        """"""""""""""""""""""""""""""""""""""""""""
        """      Apply solutions - solved pits   """
        """"""""""""""""""""""""""""""""""""""""""""
        excludedpits = list(DFpits[DFpits.solution > 3000].index.values)
        modDEM_vec, solution_mask_vec = dep.apply_solution(DFpits, fillcells_dct, carvecells_dct,
                                                           carve_fillcells_dct, modDEM_vec,
                                                           fill_vec, carve_vec,
                                                           excludedpits=excludedpits)

        """save to tif before running combined solutions"""
        modDEM_array = modDEM_vec.reshape(ny, nx)
        modDEM_tif = inDEM.replace('.tif', '_apply_TEMP.tif')
        pyct.npy2tif(modDEM_array, inDEM, modDEM_tif,
                     nodata=self.nodataval, dtype=gdal.GDT_Float32)

        """"""""""""""""""""""""""""""""
        """  Run combined solutions  """
        """"""""""""""""""""""""""""""""
        DFpits_3 = DFpits[(DFpits.solution >= 4000)
                          & (DFpits.solution < 5000)].copy()
        num_combined = len(DFpits_3.index.values)
        if num_combined > 0:
            if self.verbose_print:
                print_time(
                    f"   On {input_id} Combined pits = {num_combined}",
                    self.starttime)

            modDEM_array = gdal_array.LoadFile(modDEM_tif)
            if waterbodymask_tif is None:
                waterbodymask_array = None
            else:
                waterbodymask_array = gdal_array.LoadFile(
                    waterbodymask_tif)

            wall_vec = self.wall_array.reshape(ny * nx,)

            for pit_id in DFpits_3.index.values:
                solution_name, \
                    sol_df = dep.combined_solution(
                        modDEM_tif, modDEM_array, outpath, nx, ny, DFpits,
                        fillcells_dct, carvecells_dct, carve_fillcells_dct,
                        pit_id, self.combined_fill_interval,
                        fillID_vec, fill_vec, carve_vec,
                        self.radius, self.maxcost,
                        self.min_dist, self.max_carve_depth,
                        self.wbt, nodataval=self.nodataval,
                        flat_increment=self.flat_increment,
                        carve_method=self.carvemethod,
                        water_array=waterbodymask_array,
                        wall_vec=wall_vec,
                        del_temp_files=self.del_temp_files)

                """ Apply combined solution"""
                DFpits.at[pit_id, 'solution'] = solution_name
                combined_footprint_ind = sol_df.footprint.values[0]
                combined_footprint_elevs = sol_df.elevs.values[0]
                modDEM_vec[combined_footprint_ind] = combined_footprint_elevs
                solution_mask_vec[combined_footprint_ind] = solution_name

        modDEMoutput_array = modDEM_vec.reshape(ny, nx)
        

        # """"""""""""""""""""""""""""""""""""""""""""
        # """   Solve pits upstream combined pits  """
        # """"""""""""""""""""""""""""""""""""""""""""

        # pits_unsolved = list(DFpits[DFpits.solution > 9000].index.values)
        # DFpits = dep.solve_leftover_pits(DFpits, modDEM_vec, carvecells_only_dct,
                                         # carvecells_dct, fillcells_dct,
                                         # fillID_vec, fill_vec,
                                         # carve_vec, carveID_vec,
                                         # self.fill_vol_thresh,
                                         # verbose_print=self.verbose_print)

        solutions_grouped = DFpits.groupby(["solution"]).count()["maxfill"]
        if self.verbose_print:
            print(solutions_grouped)
        gc.collect()

        # """"""""""""""""""""""""""""""""""""""""""""
        # """      Apply solutions-unsolved pits   """
        # """"""""""""""""""""""""""""""""""""""""""""
        # DFpits_unsolved = DFpits.loc[pits_unsolved].copy()
        # modDEMoutput_vec, solution_mask_vec = dep.apply_solution(
            # DFpits_unsolved, fillcells_dct, carvecells_dct,
            # carve_fillcells_dct, modDEM_vec, fill_vec, carve_vec,
            # solution_mask_vec=solution_mask_vec)
            
        # inDEM_array = gdal_array.LoadFile(inDEM)
        # modDEMoutput_array = modDEMoutput_vec.reshape(ny, nx)

        """"""""""""""""""""""""""""""""""""""""""""
        """      Re-Apply minimum water values   """
        """"""""""""""""""""""""""""""""""""""""""""
        if self.burn_rivers and apply_relative_min:
            if self.coastal:
                river_raise = 0.05
                stream_raise = 0.20
                canal_raise = 0.10
                max_raise = 1.0
            else:
                river_raise = 0.10
                stream_raise = 1.0
                canal_raise = 0.40
                max_raise = 2.0

            """apply minimum non-SWO river value (above lowest adjacent SWO river value)"""
            modDEMoutput_array = wtr.raise_small_rivers(
                modDEMoutput_array, modDEMoutput_array,
                water_mask_array, SWOarray,
                self.nodataval, SWO_cutoff_low=5,
                SWO_cutoff_high=50, filter_size=31,
                extra_raise=river_raise,
                max_raise=max_raise)

            """apply minimum stream value (above lowest adjacent river value)"""
            modDEMoutput_array = wtr.raise_small_streams(
                DEMarray, modDEMoutput_array,
                water_mask_array, self.nodataval,
                extra_raise=stream_raise,
                max_raise=max_raise)

            # """apply minimum flattened canal value (above lowest adjacent river value)"""
            # if waterlines_tif is not None:
            # inDEM_flatten_array = wtr.raise_canals(modDEMoutput_array, modDEMoutput_array,
            # water_mask_array, SWOarray,
            # waterlines_tif, self.nodataval,
            # extra_raise=canal_raise,
            # max_raise=max_raise)

            """"""""""""""""""""""""""""""""""""""""""""
            """           Save TEMP outputs          """
            """"""""""""""""""""""""""""""""""""""""""""
            """save modified DEM"""
            nodata_ind = np.where(modDEMoutput_array == self.nodataval)

            """save solution mask"""
            solution_mask_array = solution_mask_vec.reshape(ny, nx)
            solution_mask_array[nodata_ind] = -99
            
            if not self.del_temp_files:
                solution_mask_tif = os.path.join(
                    outpath, f'{self.basename}_solution_mask.tif')
                pyct.npy2tif(solution_mask_array, inDEM,
                             solution_mask_tif, nodata=-99, 
                             dtype=gdal.GDT_Int16)

            """save dataframe of pit solutions"""
            DFpits.to_hdf(self.pits_df_out, key='df', mode='w')
            if self.verbose_print:
                print_time(
                    f"   On {input_id} temp DFpits dataframe SAVED",
                    self.starttime)
            gc.collect()

            """"""""""""""""""""""""""""""""
            """  Remove Wall (if needed) """
            """"""""""""""""""""""""""""""""
            if (self.reservoir_barrier):  # and (not self.coastal):
                if raised_wall_ind is not None:
                    modDEMoutput_array[raised_wall_ind] = self.nodataval

            if self.noflow_bound:
                if noflow_wall_ind is not None:
                    crossing_mask_tif = os.path.join(outpath,
                                                     "problem_crossings.tif")
                    crossing_array = gdal_array.LoadFile(crossing_mask_tif)
                    modDEMoutput_array = self.remove_wall(modDEMoutput_array,
                                                          inDEM_array,
                                                          crossing_array,
                                                          solution_mask_array,
                                                          noflow_wall_ind)

        return modDEMoutput_array, water_mask_array

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """       FUNCTIONS TO APPLY/REMOVE BOUNDARY CONDITIONS        """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def enforce_sinks(self, input_id, outpath, outDEM_tif,
                      dem_array=None, nodataval=-9999):
        """Set endorheic sink pixels to nodata in all basins."""
        sink_shp = os.path.join(self.endo_work_dir, 'sinks.shp')

        if dem_array is None:
            dem_array = gdal_array.LoadFile(outDEM_tif)

        if os.path.exists(sink_shp):
            sink_gdf = gpd.read_file(sink_shp)

            """Check if sink shapefile exists and sink raster does not
            exist. Check if any points in sink shapefile intersect the basin.
            If so, create a raster mask for sinks."""
            sink_mask_tif = os.path.join(
                outpath, f"{self.basename}_endo_sink.tif")
            if not os.path.exists(sink_mask_tif):
                sink_points_gdf = gpd.read_file(sink_shp)
                sinks_in_basin = gpd.tools.sjoin(
                    sink_points_gdf, self.input_gdf, how="left")
                sinks_in_basin = sinks_in_basin.dropna()
                if len(sinks_in_basin) > 0:
                    endo.build_sink_mask(sink_shp, outDEM_tif,
                                         sink_points_gdf, sink_mask_tif)

            dem_array = endo.reinforce_sinks(sink_gdf, self.input_gdf,
                                             outDEM_tif, nodata=nodataval)

            # outDEM_tif_sinks = os.path.join(outpath, "DEM_sinks.tif")

            # """ save modified DEM to tif """
            # pyct.npy2tif(dem_array, outDEM_tif,
            #              outDEM_tif_sinks, nodata=self.nodataval)

            # os.remove(outDEM_tif)
            # os.rename(outDEM_tif_sinks, outDEM_tif)

        return dem_array

    def build_reservoir_barrier(self, inDEM, outDEM, outpath,
                                water_mask_array,
                                wall_array, SWO_tif, h_diff=20,
                                modDEM_array=None, SWO_cutoff=20,
                                raised_wall_ind=None, OSM_tif=None):
        """
        Identifies large water bodies that cross the basin boundary and
        are some elevation above the outlet water crossing. If such a water
        body exists, a temporary wall is built outside of the basin along
        this boundary. This was forces the breaching algorithm to
        carve through the dam, even when the entire reservoir is not
        included within the DEM. The walls are then removed and clipped
        off of the final output. False positives (walls built where they
        are not needed) should be rare and have little impact on
        the final output.

        Parameters
        ----------
        inDEM : path
            DEM as .tif
        outDEM : path
            output DEM file
        outpath : path
            path to basin outputs (directory)
        SWO_tif : path
            geotiff of Surface Water Occurence values

        Returns
        -------
        raised_wall_ind : list of indices of raised wall.
        Returns None if conditions were not met.

        """

        """Find lowest/highest elevation lake crossing"""
        if modDEM_array is None:
            inDEM_array = gdal_array.LoadFile(inDEM)
        else:
            inDEM_array = modDEM_array

        # simplified binary river/water mask from water mask array
        river_mask = np.where((water_mask_array > 2) & (
            water_mask_array < 8), 1, 0)
        water_mask = np.where((water_mask_array > 2) & (
            water_mask_array < 12) & (
                self.SWO_array > SWO_cutoff), 1, 0)

        water_shp = os.path.join(
            outpath, f"{self.basename}_barrier_watermask.shp")
        big_water_gdf = wtr.find_possible_reservoir(
            SWO_tif, water_shp, self.wbt,
            OSM_tif=OSM_tif, SWO_limit=SWO_cutoff,
            area_threshold=3.0, fix_w_buffer=False,
            h_diff=h_diff, inDEM_array=inDEM_array,
            water_mask=river_mask)

        if big_water_gdf is None:  # no large enough water bodies
            return raised_wall_ind

        big_water_gdf.reset_index(drop=True, inplace=True)
        lake_outside_bound = gpd.overlay(
            big_water_gdf, self.donut_gdf, how='intersection')

        if lake_outside_bound.empty:  # no interaction with boundary
            return raised_wall_ind

        res_wall_mask = np.zeros_like(self.SWO_array)
        inDEM_wall_array = np.copy(inDEM_array)

        # keep only water bodies that cross boundary
        donut_poly = list(self.donut_gdf.geometry)[0]
        big_water_gdf = big_water_gdf.loc[
            big_water_gdf.geometry.intersects(donut_poly)]
        possible_reservoir_shp = os.path.join(
            outpath, f"{self.basename}_possible_reservoir.shp")
        fid_list = np.arange(1, len(big_water_gdf.index) + 1)
        big_water_gdf['FID'] = fid_list
        big_water_gdf.to_file(possible_reservoir_shp)

        possible_reservoir_tif = os.path.join(
            outpath, f"{self.basename}_possible_reservoir.tif")
        pyct.poly_rasterize(possible_reservoir_shp,
                            self.blankmask_tif,
                            possible_reservoir_tif, attr='FID')

        """buffer each intersection and re-clip to boundary donut"""
        big_water_wall_gdf = pyct.buffer_gdf_simple(big_water_gdf,
                                                    1e3, merge=False)
        big_water_wall_gdf = gpd.overlay(big_water_wall_gdf,
                                         self.donut_gdf,
                                         how='intersection')

        """dissolve/explode all overlapping walls"""
        big_water_wall_gdf['group'] = 1
        big_water_wall_gdf = big_water_wall_gdf.dissolve(by='group')
        big_water_wall_gdf = big_water_wall_gdf.explode()
        fid_wall_list = np.arange(1, len(big_water_wall_gdf.index) + 1)
        big_water_wall_gdf['FID'] = fid_wall_list
        big_water_wall_shp = os.path.join(
            outpath, f"{self.basename}_reservoir_wall_id.shp")
        big_water_wall_gdf.to_file(big_water_wall_shp)
        reservoir_wall_tif = os.path.join(
            outpath, f"{self.basename}_reservoir_wall_id.tif")
        pyct.poly_rasterize(big_water_wall_shp,
                            self.blankmask_tif,
                            reservoir_wall_tif, attr='FID')

        # buffer reservoir shapes
        big_water_buffer_gdf = pyct.buffer_gdf_simple(big_water_gdf,
                                                      500, merge=True)

        fid_buffer_list = np.arange(1, len(big_water_buffer_gdf.index) + 1)
        big_water_buffer_gdf['FID'] = fid_buffer_list
        possible_reservoir_buffer_shp = os.path.join(
            outpath, f"{self.basename}_possible_reservoir_buffer.shp")
        big_water_buffer_gdf.to_file(possible_reservoir_buffer_shp)

        possible_reservoir_buffer_tif = os.path.join(
            outpath, f"{self.basename}_possible_reservoir_buffer.tif")
        pyct.poly_rasterize(possible_reservoir_buffer_shp,
                            self.blankmask_tif,
                            possible_reservoir_buffer_tif, attr='FID')

        res_fid_array = gdal_array.LoadFile(possible_reservoir_tif)
        res_fid_buffer_array = gdal_array.LoadFile(
            possible_reservoir_buffer_tif)
        res_fid_wall_array = gdal_array.LoadFile(reservoir_wall_tif)

        """ check if each reservoir boundary values are upstream or down"""
        for fid in fid_list:
            fid_mask = np.where(res_fid_array == fid, 1, 0)
            fid_buffer = np.unique(res_fid_buffer_array[fid_mask > 0])
            wall_ids = np.unique(res_fid_wall_array[(fid_mask > 0) & (
                res_fid_wall_array > 0)])
                
            print(f"River FID = {fid}")
            print(f"Wall ids = {wall_ids}")
            
            
            
            fid_wall_mask = np.isin(res_fid_wall_array, wall_ids)
            fid_wall_mask = np.where(fid_wall_mask, 1, 0)

            fid_water_ind = np.nonzero((res_fid_buffer_array == fid_buffer) & (
                water_mask > 0) & (inDEM_array != self.nodataval))

            min_fid_water = np.percentile(inDEM_array[fid_water_ind], 0.1)
            max_fid_water = np.percentile(inDEM_array[fid_water_ind], 99)

            if max_fid_water - min_fid_water < h_diff:
                continue

            for wall_id in wall_ids:
                print(f"   Wall id = {wall_id}")
            
                wall_id_ind = np.nonzero(res_fid_wall_array == wall_id)
                wall_id_water_ind = np.nonzero((fid_mask > 0) & (
                    res_fid_wall_array == wall_id) & (water_mask > 0) & (
                        inDEM_array != self.nodataval))
                        
                if len(wall_id_water_ind[0]) < 10:
                    continue
                min_wall_id_water = np.percentile(inDEM_array[
                    wall_id_water_ind], 0.1)

                # lowest lake/river elevation
                if self.verbose_print:
                    print(f'      wallID = {wall_id}.',
                          f'min wall id water elev = {min_wall_id_water:.2f},',
                          f'min fid water elev = {min_fid_water:.2f}')

                if (min_wall_id_water > min_fid_water + h_diff):
                    res_wall_mask[wall_id_ind] = 1

        if np.count_nonzero(res_wall_mask) > 0:
            if self.verbose_print:
                print_time("   build reservoir wall", self.starttime)
            raised_wall_ind = np.nonzero(res_wall_mask)
            inDEM_wall_array[raised_wall_ind] = inDEM_wall_array[
                raised_wall_ind] + self.max_carve_depth
            pyct.npy2tif(inDEM_wall_array, inDEM,
                         outDEM, nodata=self.nodataval)

        return raised_wall_ind

    def build_noflow_wall(self, inDEM, input_id, outpath,
                          buffer_shp, wall_array, prob_crossing_shp):
        """
        Create boundary wall at locations with large flow accumulations leaving
        basin boundary, except for outlets/inlets

        Parameters
        ----------
        inDEM : path
            DEM as .tif
        input_id : int
            hydrobasins id number
        outpath : path
            basin output directory
        buffer_shp : path
            shape file used to clip regional accumulation vrt
        wall_array : numpy array
            mask of potential wall pixels. This is the ring around the
            basin between the defined buffer and inner buffer
        prob_crossing_shp : path
            point shapefile containing locations where the river should
            not cross the basin boundary
        Returns
        -------
        inDEM_noflow : path
            DEM with noflow wall applied
        noflow_wall_ind : list of indices of "no flow" wall.
            Returns None if conditions
        were not met.
        """
        try:
            noflow_wall_ind = None  # empty index
            inDEM_noflow = os.path.join(
                outpath, f"{self.basename}_noflow.tif")

            """ Load problem flow accumulation point shape file.
            buffer and dissolve"""
            if not os.path.exists(prob_crossing_shp):
                return inDEM, None

            """ test whether problem crossing touches donut"""
            crossing_buff = self.buffer + 100
            prob_crossing_buffer = prob_crossing_shp.replace(
                ".shp", f"_{int(crossing_buff)}m_buffer.shp")
            if os.path.exists(prob_crossing_buffer):
                prob_crossing_test_gdf = gpd.read_file(prob_crossing_buffer)
            else:
                prob_crossing_test_gdf = gpd.read_file(prob_crossing_shp)
                prob_crossing_test_gdf = pyct.buffer_gdf_simple(
                    prob_crossing_test_gdf, crossing_buff)
                prob_crossing_test_gdf['group'] = 1
                prob_crossing_test_gdf = prob_crossing_test_gdf.dissolve(
                    by='group')
                prob_crossing_test_gdf.to_file(prob_crossing_buffer)

            """ create intersection of buffered crossing and donut for wall"""
            prob_donut_test_intersect = gpd.overlay(prob_crossing_test_gdf,
                                                    self.donut_gdf, how='intersection')
            prob_donut_test_intersect.reset_index(drop=True, inplace=True)

            """ if the basin donut does not intersect the buffered problem point, skip """
            if len(prob_donut_test_intersect) == 0:
                print("No problem crossings, skipping noflow wall for basin", input_id)
                return inDEM, None
            else:
                print("Problem crossing found for basin", input_id)

            """ rasterize noflow polygon """
            srcDS = gdal.Open(inDEM, gdal.GA_ReadOnly)
            xshape_inDEM = srcDS.RasterXSize
            yshape_inDEM = srcDS.RasterYSize
            geoT_inDEM = srcDS.GetGeoTransform()
            xmin_inDEM = geoT_inDEM[0]
            ymax_inDEM = geoT_inDEM[3]
            xmax_inDEM = xmin_inDEM + geoT_inDEM[1] * xshape_inDEM
            ymin_inDEM = ymax_inDEM + geoT_inDEM[5] * yshape_inDEM
            srcDS = None

            crossing_mask_tif = os.path.join(
                outpath, f"{self.basename}_problem_crossings.tif")
            rstr_command = "gdal_rasterize -a id -ot Byte -ts {0} {1}\
                -te {2} {3} {4} {5} {6} {7}".format(xshape_inDEM, yshape_inDEM, xmin_inDEM,
                                                    ymin_inDEM, xmax_inDEM, ymax_inDEM,
                                                    prob_crossing_shp, crossing_mask_tif)
            subprocess.call(rstr_command, shell=True,
                            stdout=subprocess.DEVNULL)

            crossing_mask_array = gdal_array.LoadFile(crossing_mask_tif)
            inDEM_array = gdal_array.LoadFile(inDEM)
            inDEM_wall_array = np.copy(inDEM_array)
            noflow_wall_ind = np.where((crossing_mask_array > 0) & (
                inDEM_array != self.nodataval))

            """ add wall height within polygon """
            inDEM_wall_array[noflow_wall_ind] = inDEM_wall_array[noflow_wall_ind] + \
                self.wall_height

            if len(noflow_wall_ind[0]) == 0:
                print(f"   On {input_id}: No noflow walls built")
                return inDEM, None

            """ save modified DEM to tif """
            pyct.npy2tif(inDEM_wall_array, inDEM,
                         inDEM_noflow, nodata=self.nodataval)

            del inDEM_wall_array

        except Exception as e:
            outst = f"Exception occurred on {input_id}: {e}"
            traceback_output = traceback.format_exc()
            print(outst)
            print(traceback_output)

        return inDEM_noflow, noflow_wall_ind

    def remove_wall(self, modDEM_array, inDEM_array, wall_array,
                    solution_mask_array, noflow_wall_ind):
        """Remove noflow wall applied for large deviations in stream network. Two cases:

            1. No fill has been applied adjacent to the wall - set pixel to 10cm above original value
            2. Fill has been applied next to the noflow wall - set pixel to 10cm above fill value

        Parameters
        ----------
        modDEM_array : array
            raster of fully conditioned DEM
        inDEM_array : array
            2D array of raster DEM
        wall_array : array
            mask of pixels that are part of the noflow wall
        solution_mask_array : array
            array of solution types applied to each pit (1= pit within noflow wall)
        noflow_wall_ind : list/array
            indices of noflow wall within entire DEM array

        Returns
        -------
        array
            2D DEM with noflow wall removed

        """

        """remove wall pits"""
        modDEM_array[solution_mask_array == 1] = inDEM_array[
            solution_mask_array == 1]
        """copy so not to query a changing array while iterating through pixels"""
        testDEM_array = np.copy(modDEM_array)
        # -9999 --> nan for min/max
        testDEM_array[testDEM_array == self.nodataval] = np.nan
        testDEM_array[self.donut_array == 1] = np.nan

        """ create a dictionary fro wall remove values """
        unique_ids = np.unique(wall_array)[1::]
        #remove_dict = dict(zip(unique_ids, self.wall_remove_min))
        #print("wall removal minimum dictionary:", remove_dict)
        print("unique_ids", unique_ids)

        for wall_id in unique_ids:
            wall_id_ind = np.nonzero(wall_array == wall_id)
            wall_remove_min_value = self.wall_remove_min[wall_id - 1]
            print(" wall ID = {0}... wall_remove_min = {1}".format(
                wall_id, wall_remove_min_value))

            """remove outer wall, raise 10 cm, ensuring wall pixels remain at
            a higher elevation than adjacent interior pixels"""
            for row_ind, col_ind in zip(wall_id_ind[0], wall_id_ind[1]):
                z0 = testDEM_array[row_ind, col_ind] - self.wall_height
                z0 += 0.1  # add 10 cm
                #wall_id = wall_array[row_ind, col_ind]
                #wall_remove_min_value = remove_dict[wall_id]

                """find min within 5x5 pixel window"""
                DEM_window = testDEM_array[row_ind - 2:row_ind + 3,
                                           col_ind - 2:col_ind + 3]
                inDEM_window = inDEM_array[row_ind - 2:row_ind + 3,
                                           col_ind - 2:col_ind + 3]
                wall_array_window = wall_array[row_ind - 2:row_ind + 3,
                                               col_ind - 2:col_ind + 3]
                non_wall_ind = np.where(wall_array_window == 0)

                # all wall pixels - set to original height +10cm
                if len(non_wall_ind[0]) < 1:
                    new_val = max(z0, wall_remove_min_value)

                else:  # non wall pixels in window
                    """
                    Look for pixels that have had any fill applied within the search window -
                    make sure ghost wall is set to 10cm above any adjacent fill. Else, set to
                    10cm above original pixel height.
                    """
                    if any(np.isfinite(DEM_window[non_wall_ind])):
                        """query solution mask to id potential filled pixels that are not
                        wall pixels"""
                        pit_ind_win = np.where(
                            (solution_mask_array[row_ind - 2:row_ind + 3,
                                                 col_ind - 2:col_ind + 3] > 999) &
                            (wall_array_window == 0))
                        fill_diff_win = np.round(
                            (DEM_window[pit_ind_win] - inDEM_window[pit_ind_win]), 3)
                        if len(pit_ind_win[0]) > 0:  # next to pit pixels
                            max_fill_ind = np.argmax(fill_diff_win)
                            # next to filled pixels
                            if (fill_diff_win[max_fill_ind] > 0.05):
                                DEMfill_max_win = DEM_window[pit_ind_win][
                                    max_fill_ind] + 0.1
                                new_val = max(z0, DEMfill_max_win,
                                              wall_remove_min_value)
                            else:  # no filled pit pixels in window
                                new_val = max(z0, wall_remove_min_value)
                        else:  # no pit pixels in window
                            new_val = max(z0, wall_remove_min_value)
                    else:  # all non-wall pixels are nans
                        new_val = max(z0, wall_remove_min_value)

                modDEM_array[row_ind, col_ind] = new_val

        return modDEM_array


""" Tiny wrapper functions to make the mpi executor behave """


def run_depression_handle(input_id, SWO_cutoff, initial_burn, SWO_burn,
                          burn_cutoff, gap_dist):

    dHandle.run_depression_handling(input_id, SWO_cutoff=SWO_cutoff,
                                    initial_burn=initial_burn, SWO_burn=SWO_burn,
                                    burn_cutoff=burn_cutoff, gap_dist=gap_dist)


if __name__ == '__main__':
    starttime = time.time()

    """ Load the configuration file """
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="config file")
    ap.add_argument("-l", "--logfile", required=False, help="log file")
    args = vars(ap.parse_args())
    log_file = args['logfile']
    config_file = args['config']
    config = ConfigParser(allow_no_value=True,
                          interpolation=ExtendedInterpolation())
    config.read(config_file)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    ###################  SET OPTIONS #####################
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    projectname = config.get("outputs", "projectname")
    basin_levels = json.loads(config.get(
        "parameters-depression-handling", "basin_levels"))
    run_parallel = config.getboolean("job-options", "run_parallel")
    run_multiprocess = config.getboolean("job-options", "run_multiprocess")
    overwrite = config.getboolean("processes-depression-handling", "overwrite")
    verbose_print = config.getboolean(
        "processes-depression-handling", "verbose_print")
    buffer = config.getfloat("parameters-depression-handling", "buffer")
    fill_h20_adjacent = config.getboolean(
        "processes-depression-handling", "fill_h20_adjacent")
    del_temp_files = config.getboolean(
        "processes-depression-handling", "del_temp_files")
    overwrite = config.getboolean("processes-depression-handling", "overwrite")
    sfill = config.getfloat("parameters-depression-handling", "sfill")
    fill_method = config.get("parameters-depression-handling", "fill_method")
    carve_method = config.get("parameters-depression-handling", "carve_method")
    maxcost = config.getfloat("parameters-depression-handling", "maxcost")
    radius = config.getint("parameters-depression-handling", "radius")
    min_dist = config.getboolean("parameters-depression-handling", "min_dist")
    fill_vol_thresh = config.getint(
        "parameters-depression-handling", "fill_vol_thresh")
    carve_vol_thresh = config.getint(
        "parameters-depression-handling", "carve_vol_thresh")
    max_carve_depth = config.getint(
        "parameters-depression-handling", "max_carve_depth")
    combined_fill_interval = config.getfloat(
        "parameters-depression-handling", "combined_fill_interval")
    flat_increment = config.getfloat(
        "parameters-depression-handling", "flat_increment")

    burn_rivers = config.getboolean(
        "processes-depression-handling", "burn_rivers")
    enforce_monotonicity = config.getboolean(
        "processes-depression-handling", "enforce_monotonicity")
    reservoir_barrier = config.getboolean(
        "processes-depression-handling", "reservoir_barrier")
    fill_h20_adjacent = config.getboolean(
        "processes-depression-handling", "fill_h20_adjacent")

    enforce_endo = config.getboolean(
        "processes-depression-handling", "enforce_endo")
    noflow_bound = config.getboolean(
        "processes-depression-handling", "noflow_walls")

    wall_height = config.getfloat("parameters-noflow", "wall_height")
    wall_remove_min = json.loads(config.get(
        "parameters-noflow", "wall_remove_min"))

    osm_coast_clip = config.getboolean(
        "processes-depression-handling", "osm_coast_clip")

    """   Water-flattening parameters  """
    SWO_cutoff = config.getint("parameters-depression-handling", "SWO_cutoff")
    initial_burn = config.getfloat(
        "parameters-depression-handling", "initial_burn")
    SWO_burn = config.getfloat("parameters-depression-handling", "SWO_burn")
    burn_cutoff = config.getfloat(
        "parameters-depression-handling", "burn_cutoff")
    gap_dist = config.getint("parameters-depression-handling", "gap_dist")

    if sfill == 0.0:
        sfill = None

    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    ################### DEFINE PATHS #####################
    """"""""""""""""""""""""""""""""""""""""""""""""""""""

    """ Path to White Box Tools"""
    WBT_path = os.path.join(config.get("paths", "WBT_path"))

    """locations of input data"""
    edm_vrt = os.path.join(config.get("outputs", "edm_mosaic"))
    SWO_vrt = os.path.join(config.get("paths", "SWO_vrt"))
    OSM_dir = os.path.join(config.get("paths", "OSM_dir"))
    noflow_points = os.path.join(config.get("paths-noflow", "noflow_points"))

    if burn_rivers:
        inDS_vrt = os.path.join(config.get('outputs', 'burned_mosaic'))
        flattened_mask_vrt = os.path.join(
            config.get('outputs', 'water_mask_mosaic'))
    else:
        inDS_vrt = os.path.join(config.get('outputs', 'dtm_mosaic'))
        flattened_mask_vrt = None

    """output directories"""
    work_dir = config.get("outputs", "dh_work_dir")
    taudem_dir = os.path.join(config.get(
        "paths-taudem", "taudem_dir"), projectname)

    base_unit = config.get("region", "base_unit")
    if base_unit == 'basin':
        basins_merge_shp = os.path.join(
            config.get('outputs', 'basins_merge_shp'))
        basins_dissolve_shp = os.path.join(
            config.get('outputs', 'basins_dissolve_shp'))
    else:
        basins_merge_shp = None
        basins_dissolve_shp = None

    """Set up output directories"""
    if (taudem_dir is not None) and (not os.path.exists(taudem_dir)):
        os.makedirs(taudem_dir)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    ###################   MPI SETUP   ####################
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
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
    ###############  INITIALIZE CLASS ####################
    """"""""""""""""""""""""""""""""""""""""""""""""""""""

    """initialize depression handling class"""
    dHandle = DepHandle(inDS_vrt,
                        work_dir,
                        SWO_vrt,
                        projectname, basin_levels, WBT_path,
                        OSM_dir=OSM_dir,
                        edm_vrt=edm_vrt,
                        taudem_dir=taudem_dir,
                        overwrite=overwrite,
                        run_parallel=run_parallel,
                        verbose_print=verbose_print,
                        del_temp_files=del_temp_files,
                        burn_rivers=burn_rivers,
                        flattened_mask_vrt=flattened_mask_vrt,
                        basins_merge_shp=basins_merge_shp,
                        basins_dissolve_shp=basins_dissolve_shp,
                        reservoir_barrier=reservoir_barrier,
                        fill_h20_adjacent=fill_h20_adjacent,
                        buffer=buffer, sfill=sfill,
                        flat_increment=flat_increment,
                        fill_method=fill_method, carve_method=carve_method,
                        maxcost=maxcost, radius=radius, min_dist=min_dist,
                        fill_vol_thresh=fill_vol_thresh,
                        carve_vol_thresh=carve_vol_thresh,
                        max_carve_depth=max_carve_depth,
                        combined_fill_interval=combined_fill_interval,
                        enforce_endo=enforce_endo,
                        enforce_monotonicity=enforce_monotonicity,
                        noflow_bound=noflow_bound,
                        noflow_points=noflow_points,
                        wall_height=wall_height,
                        wall_remove_min=wall_remove_min,
                        osm_coast_clip=osm_coast_clip)

    if not overwrite:
        inputIDs = logt.find_unfinished_ids(work_dir,
                                            unit=base_unit,
                                            dir_search=f'{base_unit}_',
                                            tifname='DEMcompleted.txt')
    else:
        inputIDs = logt.find_all_ids(work_dir,
                                     unit=base_unit,
                                     dir_search=f'{base_unit}_')

    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    ###########    RUN DEPRESSION HANDLING    ############
    """"""""""""""""""""""""""""""""""""""""""""""""""""""

    """sort in descending order to place smaller basins first"""
    inputIDs = np.sort(inputIDs)[::-1]
    n_unfin = len(inputIDs)

    if run_parallel:

        with MPICommExecutor(MPI.COMM_WORLD, root=0) as ex:
            if ex is not None:
                print_time(f'Depression handling for {n_unfin}'
                           + f' {base_unit}s on {numnodes} nodes',
                           starttime, sep=True)

                ex.starmap(run_depression_handle, [(inputID, SWO_cutoff,
                                                    initial_burn,
                                                    SWO_burn,
                                                    burn_cutoff,
                                                    gap_dist)
                                                   for inputID in inputIDs])

    elif run_multiprocess:

        nthreads = min(n_unfin, 14)
        with Pool(nthreads) as p:
            print_time(f'Depression handling for {n_unfin}'
                       + f' {base_unit}s on {nthreads} threads',
                       starttime, sep=True)
            p.starmap(dHandle.run_depression_handling, [(inputID, SWO_cutoff,
                                                         initial_burn, SWO_burn,
                                                         burn_cutoff, gap_dist)
                                                        for inputID in inputIDs])

    else:

        for inputID in inputIDs:
            run_depression_handle(inputID, SWO_cutoff,
                                  initial_burn, SWO_burn,
                                  burn_cutoff, gap_dist)

        log_str = log_time("Depression handling", starttime)
        with open(log_file, 'w') as f:
            f.write(log_str)
