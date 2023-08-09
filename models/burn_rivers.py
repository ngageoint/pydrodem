#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Kimberly McCormack, Heather Levin, Amy Morris

Last edited on: 9/15/2021

Class to flatten large rivers and burn in streams

"""
import geopandas as gpd
import pandas as pd
import numpy as np
import warnings
import argparse
import traceback
import time
import sys
import os
from configparser import ConfigParser, ExtendedInterpolation
from scipy import ndimage
from osgeo import gdal, gdal_array
from shapely.geometry import box

sys.path.append(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))
from tools import osm
from tools import shapes
import tools.convert as pyct
import tools.files as ft
import tools.log as logt
from tools.derive import apply_convolve_filter
import tools.water as wtr
from tools.files import tilesearch
from tools.print import print_time, log_time  # , print_CPUusage

warnings.filterwarnings(action='ignore')
gdal.UseExceptions()


class EnforceRivers():

    def __init__(self, inDS_vrt, work_dir, SWO_vrt,
                 projectname, WBT_path, input_paths,
                 base_unit='tile', edm_mosaic=None, fabdem_mosaic=None,
                 OSM_dir=None, OSM_tiles=None, del_temp_files=False,
                 verbose_print=True, buffer=100, coastal_buffer=2000,
                 radius=25, SWO_cutoff=15, initial_burn=3.0, SWO_burn=4.0,
                 burn_cutoff=6.0, outlet_dist=1e3, riv_area_cutoff=0.5,
                 gap_dist=240, basins_merge_shp=None,
                 basins_dissolve_shp=None):

        self.WBT_path = WBT_path
        self.input_paths = input_paths
        self.starttime = time.time()
        self.inDS_vrt = inDS_vrt
        self.base_unit = base_unit
        self.edm_mosaic = edm_mosaic
        self.fabdem_mosaic = fabdem_mosaic
        self.work_dir = work_dir
        self.SWO_vrt = SWO_vrt
        self.projectname = projectname
        self.OSM_dir = OSM_dir
        self.OSM_tiles = OSM_tiles
        self.buffer = buffer
        self.coastal_buffer = coastal_buffer
        self.radius = radius
        self.verbose_print = verbose_print
        self.del_temp_files = del_temp_files
        self.SWO_cutoff = SWO_cutoff
        self.initial_burn = initial_burn
        self.SWO_burn = SWO_burn
        self.burn_cutoff = burn_cutoff
        self.outlet_dist = outlet_dist
        self.riv_area_cutoff = riv_area_cutoff
        self.gap_dist = gap_dist
        self.basins_merge_shp = basins_merge_shp
        self.basins_dissolve_shp = basins_dissolve_shp

        # Load WhiteBox Tools executable
        sys.path.append(WBT_path)
        from whitebox_tools import WhiteboxTools
        self.wbt = WhiteboxTools()
        self.wbt.set_verbose_mode(False)
        self.wbt.set_whitebox_dir(WBT_path)  # set location of WBT executable

    def run_river_enforce(self, input_id):

        self.input_id = input_id
        base_unit = self.base_unit
        nodataval = -9999
        
        if base_unit == 'basin':
            self.usecutline = True
            self.joinstyle = 1
        else:
            self.usecutline = False
            self.joinstyle = 2  # square caps

        try:
            print_time(f" {base_unit} {input_id}",
                       self.starttime)
            self.outpath = os.path.join(self.work_dir,
                                        f'{base_unit}_{input_id}')
            outpath = self.outpath
            

            input_shp = os.path.join(outpath,
                                     f'{base_unit}_{input_id}.shp')
            self.input_gdf = gpd.read_file(input_shp)

            self.src_dem = self.input_gdf.srcDEM.values[0]
            buffer = self.buffer
            self.coast = False
            self.exclude_wetland = True

            if base_unit == 'basin' and self.input_gdf["COAST"].values[0] == 1:
                print(f'  On {input_id} - coastal basin -',
                      f' increase buffer by {self.coastal_buffer} m')
                buffer = self.buffer + self.coastal_buffer
                coast_file = os.path.join(outpath, 'coast.txt')
                with open(coast_file, 'w') as f:
                    f.write('coastal basin')
                self.coast = True
                self.SWO_cutoff = 65
                # self.exclude_wetland = False

            """ create buffer"""
            self.input_gdf = shapes.close_holes_gdf(self.input_gdf)
            self.buffer_shp = os.path.join(
                outpath, f'{base_unit}_{input_id}_buffer.shp')
            pyct.buffer_gdf2shp(self.input_gdf, buffer, self.buffer_shp,
                                joinstyle=self.joinstyle)
            self.buffer_gdf = gpd.read_file(self.buffer_shp)

            """Pull DEM and SWO from vrt"""
            self.dem_tif = os.path.join(outpath, 'DEM.tif')
            self.basename = os.path.splitext(os.path.basename(self.dem_tif))[0]
            ft.erase_files(outpath, search=f"{self.basename}_*")
            
            overlapDEM = os.path.join(outpath, 'overlapDEM.tif')
            if os.path.exists(overlapDEM):
                os.remove(overlapDEM)

            if not os.path.exists(self.dem_tif):
                pyct.create_cutline_raster_set_res(self.inDS_vrt,
                                                   self.buffer_shp,
                                                   self.dem_tif, nodata=nodataval,
                                                   usecutline=self.usecutline)
            
            """Re-build buffer shape - excluding nodata values"""
            in_array = gdal_array.LoadFile(self.dem_tif)
            data_mask = np.where(in_array==nodataval, 0, 1)
            data_mask_tif = os.path.join(outpath, f'{self.basename}_data_mask.tif')
            
            if data_mask.min() == 0: #masked out pixels - re-crop DEM
                print(f"    on {input_id}: re-crop DEM")
                pyct.npy2tif(data_mask, self.dem_tif,
                             data_mask_tif, nodata=0, dtype=gdal.GDT_Byte)
                self.wbt.raster_to_vector_polygons(data_mask_tif, self.buffer_shp)
                self.buffer_gdf = gpd.read_file(self.buffer_shp)
                pyct.create_cutline_raster_set_res(self.inDS_vrt,
                                                   self.buffer_shp,
                                                   self.dem_tif,
                                                   nodata=nodataval,
                                                   buffer=0,
                                                   usecutline=self.usecutline)

          
            
            """create inner buffer and donut"""
            self.donut_buffer_shp = os.path.join(
                outpath, f'{self.basename}_{base_unit}_{input_id}_donut.shp')
            inner_buffer_gdf = pyct.buffer_gdf_simple(self.buffer_gdf,
                                                      -50,
                                                      joinstyle=self.joinstyle)
            self.donut_gdf = gpd.overlay(self.buffer_gdf,
                                         inner_buffer_gdf, how='difference')
            self.donut_gdf.reset_index(drop=True, inplace=True)
            self.donut_gdf.to_file(self.donut_buffer_shp)


            if self.edm_mosaic is not None:
                self.edm_tif = os.path.join(outpath, f"{self.basename}_EDM.tif")
                if not os.path.exists(self.edm_tif):
                    pyct.create_cutline_raster(self.edm_mosaic,
                                               input_shp,
                                               self.edm_tif,
                                               srcDS=self.dem_tif,
                                               usecutline=self.usecutline,
                                               nodata=0,
                                               outdtype=gdal.GDT_Byte)
            else:
                self.edm_tif = None

            if self.fabdem_mosaic is not None:
                fab_tif = os.path.join(outpath, f"{self.basename}_FABDEM.tif")
                if not os.path.exists(fab_tif):
                    pyct.create_cutline_raster(self.fabdem_mosaic,
                                               self.buffer_shp,
                                               fab_tif, srcDS=self.dem_tif,
                                               usecutline=self.usecutline,
                                               nodata=nodataval,
                                               rsAlg=gdal.GRIORA_Bilinear)

            else:
                fab_tif = None

            swo_tif = os.path.join(outpath, f"{self.basename}_SWO.tif")
            if not os.path.exists(swo_tif):
                pyct.create_cutline_raster(self.SWO_vrt, self.buffer_shp,
                                           swo_tif, srcDS=self.dem_tif, 
                                           nodata=255,
                                           usecutline=self.usecutline,
                                           outdtype=gdal.GDT_Byte,
                                           rsAlg=gdal.GRIORA_Bilinear,
                                           paletted=True)
                self.SWO_array = gdal_array.LoadFile(swo_tif)
                self.SWO_array[self.SWO_array > 100] = 0
                pyct.npy2tif(self.SWO_array, self.dem_tif,
                             swo_tif, nodata=0, dtype=gdal.GDT_Byte)
            self.SWO_array = gdal_array.LoadFile(swo_tif)
                       
            # check for ocean pixels in tiles - mark as coastal tile
            coast_file = os.path.join(outpath, 'coast.txt')
            if (base_unit == 'tile') and (self.edm_tif is not None):
                self.edm_array = gdal_array.LoadFile(self.edm_tif)
                ocean_pixels = np.nonzero(self.edm_array == 4)
                if len(ocean_pixels[0]) > 1000:
                    print(f'    {input_id} - coastal tile')
                    with open(coast_file, 'w') as f:
                        f.write('coastal tile')
                    self.coast = True
                    # self.exclude_wetland = False
                    self.SWO_cutoff = 65

            DEM_burn_arr, \
                water_mask_array,\
                flatten_flag = self.enforce_rivers(
                    input_id, self.dem_tif, swo_tif,
                    useOSM=True, edm_tif=self.edm_tif,
                    fab_tif=fab_tif, SWO_cutoff=self.SWO_cutoff,
                    initial_burn=self.initial_burn, SWO_burn=self.SWO_burn,
                    burn_cutoff=self.burn_cutoff, gap_dist=self.gap_dist,
                    keep_water_shp=False, floor_river_value=-1.0,
                    require_bound=False, riv_area_cutoff=0.5,
                    apply_relative_min=True)

            """save outputs"""
            flatten_mask_tif = os.path.join(
                outpath, "Water_mask.tif")
            pyct.npy2tif(water_mask_array, self.dem_tif,
                         flatten_mask_tif, nodata=0, dtype=gdal.GDT_Byte)
            pyct.tifcompress(flatten_mask_tif)

            """set non-river pixels to non data"""
            # DEM_burn_arr[water_mask_array == 0] = self.nodata
            DEM_flatten_tif = os.path.join(outpath,
                                           f"{self.basename}burned.tif")
            pyct.npy2tif(DEM_burn_arr, self.dem_tif,
                         DEM_flatten_tif, nodata=self.nodata)
            pyct.tifcompress(DEM_flatten_tif)

            """Erase temp files"""
            if self.del_temp_files:
                ft.erase_files(outpath, search=f'{self.basename}_*')
                ft.erase_files(outpath, search='*temp*')

        except Exception as e:
            outst = f"Exception occurred {e}"
            traceback_output = traceback.format_exc()
            print(outst)
            print(traceback_output)

    def enforce_rivers(self, input_id, inDEM_tif, SWO_tif,
                       useOSM=True, edm_tif=None, fab_tif=None,
                       SWO_cutoff=20, initial_burn=3.0, SWO_burn=2.0,
                       burn_cutoff=6.0, outlet_dist=1e3, riv_area_cutoff=0.5,
                       gap_dist=240, keep_water_shp=False,
                       floor_river_value=-1.0,
                       require_bound=True, apply_relative_min=True):
        """
        Flattens and burns continuous river segments.

        Using Surface Water Occurence and OpenStreetMap.

        Parameters
        ----------
        inDEM_tif : path
            DEM as .tif
        swo_tif : path
            geotiff of Surface Water Occurence values (same as DEM)
        flatten_mask_tif : path
            output tif for mask of water pixels that are modified
        SWO_cutoff : int
            cutoff value to consider pixel "water" [0-100]. Default=30.
        initial_burn : float
            burn value applied to all river pixels (meters). Default=2.0
        burn_cutoff : float
            Maximum allowable burn (meters). Default=6.0

        Returns
        -------
        flatten_flag : bool
            True if flattening was applied.
        DEM_burn_arr : numpy array
            Flattened version of input DEM.

        Water Mask:
            1 : riverbank                           6
            2 : interpolated riverbank              5
            3 : water/lake                          13
            4 : river                               4
            5 : lowered below land                  3
            6 : non-monotonic relative drop         2
            7 : non-monotonic absolute drop         1
            8 : river line                          9
            9 : stream line                         10
            10 : drain/canal line                   11
            11 : FABDEM/EDM river extension         14
            12 : burned stream banks                12
            13 : non-monotonic stream section       7
            14 : interpolated stream bank           8
            15 : interpolated FABDEM edge           15
        """
        flatten_flag = False
        outpath = self.outpath

        """ Load data"""
        basename = self.basename
        inDEM_array = gdal_array.LoadFile(inDEM_tif)
        dem_geot = pyct.get_geoT(inDEM_tif)
        pix_sz_m = dem_geot[5] * 111139  # pixel height deg --> meters
        water_mask_array = np.zeros_like(inDEM_array, dtype=np.int8)
        DEM_burn_arr = np.copy(inDEM_array)

        # Import OpenStreetMap data
        if self.verbose_print:
            print_time(
                f"   On {input_id} rasterize OpenStreetMap", self.starttime)
        self.import_osm(inDEM_tif, outpath)

        """create blank raster mask"""
        self.blankmask_tif = os.path.join(outpath,
                                          f"{self.basename}_blank.tif")
        rastermask = np.ones_like(self.SWO_array, dtype=np.int8)
        pyct.npy2tif(rastermask, inDEM_tif, self.blankmask_tif,
                     nodata=0, dtype=gdal.GDT_Byte)

        donut_mask = os.path.join(outpath, f"{self.basename}_donut.tif")
        pyct.cutline_raster_simple(self.blankmask_tif,
                                   self.donut_buffer_shp, donut_mask,
                                   nodata=0, outdtype=gdal.GDT_Byte)
        self.donut_array = gdal_array.LoadFile(donut_mask)

        """Extract raster nodata"""
        raster = gdal.Open(inDEM_tif, gdal.GA_ReadOnly)
        band = raster.GetRasterBand(1)
        self.nodata = band.GetNoDataValue()   # Set nodata value
        raster = None
        band = None

        self.land_array = None  # set to array of ones
        if self.coast and self.base_unit == 'basin':
            """if coastal, clip interior boundaries to flatten buffer"""
            """create bounding box shape for self.dem_tif"""
            DEM_bbox_gdf = shapes.raster_bounds_gdf(inDEM_tif)
            DEM_bbox_shp = os.path.join(
                outpath, f'{self.basename}_basin_{input_id}_raster_bbox.shp')
            DEM_bbox_gdf.to_file(DEM_bbox_shp)
            basin_interior_lvl2_shp = os.path.join(
                outpath, f"{self.basename}_interior_clip.shp")
            shapes.clip_shp_ogr(DEM_bbox_shp, basin_interior_lvl2_shp,
                                self.basins_merge_shp)
            basins_merge_gdf = gpd.read_file(basin_interior_lvl2_shp)
            interior_bound_gdf = gpd.overlay(basins_merge_gdf,
                                             self.buffer_gdf, how='difference')
            interior_bound_gdf.reset_index(drop=True, inplace=True)
            if not interior_bound_gdf.empty:
                interior_shp = os.path.join(
                    outpath, f"{self.basename}_interior_bound.shp")
                interior_bound_gdf = pyct.buffer_gdf_simple(
                    interior_bound_gdf, -20.0, joinstyle=self.joinstyle)
                interior_bound_gdf, _flag = shapes.fix_invalid_geom(
                    interior_bound_gdf, fix_w_buffer=True)
                interior_bound_gdf.to_file(interior_shp)

                interior_mask = os.path.join(
                    outpath, f"{self.basename}_interior_boundary.tif")
                pyct.cutline_raster_simple(self.blankmask_tif, interior_shp,
                                           interior_mask,
                                           nodata=0, outdtype=gdal.GDT_Byte)
                self.interior_array = gdal_array.LoadFile(interior_mask)
            else:
                self.interior_array = np.zeros_like(self.donut_array)

            """create mask outside of dissolved level 2 shape"""
            basin_exterior_lvl2_shp = os.path.join(
                outpath, f"{self.basename}_exterior_clip.shp")
            shapes.clip_shp_ogr(DEM_bbox_shp,
                                basin_exterior_lvl2_shp,
                                self.basins_dissolve_shp)
            land_mask = os.path.join(outpath, f"{self.basename}_land_mask.tif")
            pyct.cutline_raster_simple(self.blankmask_tif,
                                       basin_exterior_lvl2_shp,
                                       land_mask,
                                       nodata=99, outdtype=gdal.GDT_Byte)
            self.land_array = gdal_array.LoadFile(land_mask)

            """mask out ocean pixels"""
            wtr.clip_ocean(inDEM_tif, outpath, self.buffer_shp, self.input_gdf,
                           edm_mosaic=self.edm_mosaic, nodata=self.nodata,
                           basename=basename, interior_mask=None,
                           land_mask=self.land_array,
                           usecutline=self.usecutline)

        elif self.coast and self.base_unit == 'tile':
            """mask out ocean pixels"""
            ocean_z = floor_river_value - 0.5
            ocean_buffer_pix = int(abs(100 / pix_sz_m))
            # print("ocean z = ", ocean_z,
            #       "  ocean buffer = ", ocean_buffer_pix)
            # Create land mask from DEM and EDM
            if self.waterpolys_tif is not None:
                osm_poly_array = gdal_array.LoadFile(self.waterpolys_tif)
            else:
                osm_poly_array = np.zeros_like(inDEM_tif, dtype=np.int8)

            ocean_mask = np.where(
                (((inDEM_array == 0.0) & (self.edm_array == 5)) |
                 (self.edm_array == 4)) &
                (osm_poly_array == 0), 1, 0)

            # close holes in ocean mask and re-test for ocean pixels that
            # are filled (EDM=5) with elev < 0
            hole_s = np.ones((15, 15))
            ocean_mask_closed = ndimage.binary_closing(ocean_mask,
                                                       structure=hole_s)
            ocean_holes_add = np.nonzero((ocean_mask == 0) &
                                        (ocean_mask_closed == 1) &
                                         (inDEM_array < 0.0) &
                                         (self.edm_array >= 4) &
                                         (osm_poly_array == 0))
            ocean_mask[ocean_holes_add] = 1

            # vectorize land mask
            land_mask_tif = os.path.join(
                outpath, f"{self.basename}_land_mask.tif")
            land_mask_shp = os.path.join(
                outpath, f"{self.basename}_land_mask.shp")
            land_buffer_mask_tif = os.path.join(
                outpath, f"{self.basename}_land_buffer_mask.tif")
            land_mask_buffer_shp = os.path.join(
                outpath, f"{self.basename}_land_mask_buffer.shp")

            land_mask = np.where(
                ((ocean_mask == 0) & (inDEM_array != self.nodata)), 1, 0)
            pyct.npy2tif(land_mask, inDEM_tif, land_mask_tif,
                         dtype=gdal.GDT_Byte, nodata=0)

            self.wbt.raster_to_vector_polygons(land_mask_tif,
                                               land_mask_shp)

            # create tight and buffered land mask, close holes
            landmask_gdf = gpd.read_file(land_mask_shp)
            landmask_gdf = pyct.buffer_gdf(landmask_gdf, 10)
            landmask_buffer_gdf = pyct.buffer_gdf(landmask_gdf, 100)
            landmask_buffer_gdf = pyct.buffer_gdf(landmask_buffer_gdf, 0)

            landmask_gdf['land'] = 1
            landmask_buffer_gdf['land'] = 1
            landmask_gdf.to_file(land_mask_shp)
            landmask_buffer_gdf.to_file(land_mask_buffer_shp)

            pyct.poly_rasterize(land_mask_shp, self.blankmask_tif,
                                land_mask_tif, attr='land')
            pyct.poly_rasterize(land_mask_buffer_shp, self.blankmask_tif,
                                land_buffer_mask_tif, attr='land')
            self.land_array = gdal_array.LoadFile(land_mask_tif)
            land_mask_buffer = gdal_array.LoadFile(land_buffer_mask_tif)

            DEM_burn_arr, self.ocean_ind = wtr.clip_ocean(
                inDEM_tif,
                outpath, self.buffer_shp,
                self.input_gdf,
                edm_array=self.edm_array,
                nodata=self.nodata,
                basename=basename,
                osm_tif=self.waterpolys_tif,
                land_mask=land_mask_buffer,
                usecutline=self.usecutline)

            DEM_burn_arr, beach_ind = wtr.set_ocean_elev(DEM_burn_arr,
                                                         self.land_array,
                                                         nodata=self.nodata,
                                                         ocean_z=ocean_z)

            ocean_mask_buffer = np.zeros_like(water_mask_array)
            ocean_mask_buffer[self.land_array == 0] = 99
            self.ocean_mask_buffer = apply_convolve_filter(ocean_mask_buffer,
                                                           np.ones((7, 7)))
            self.ocean_mask_buffer[self.ocean_mask_buffer > 0] = 1

            DEM_ocean_mask_tif = os.path.join(
                outpath, f"{self.basename}_ocean_mask_buffer.tif")
            pyct.npy2tif(self.ocean_mask_buffer, inDEM_tif,
                         DEM_ocean_mask_tif, nodata=0, dtype=gdal.GDT_Byte)
            ocean_mask_shp = os.path.join(
                outpath, f"{self.basename}_ocean_mask_buffer.shp")
            self.wbt.raster_to_vector_polygons(DEM_ocean_mask_tif,
                                               ocean_mask_shp)
            self.oceanmask_gdf = gpd.read_file(ocean_mask_shp)
            self.oceanmask_gdf = pyct.buffer_gdf(self.oceanmask_gdf, 10)

            DEM_ocean_tif = os.path.join(outpath, f"{self.basename}_ocean.tif")
            pyct.npy2tif(DEM_burn_arr, inDEM_tif,
                         DEM_ocean_tif, nodata=self.nodata)
            inDEM_tif = DEM_ocean_tif
            inDEM_array = DEM_burn_arr.copy()

        """     Check for and log all-water basins     """
        real_ind = np.where((inDEM_array != self.nodata))
        water_ind = np.where((self.SWO_array > SWO_cutoff) &
                             (inDEM_array != self.nodata))
        num_water = len(water_ind[0])
        num_real = len(real_ind[0])
        """basins that contain mostly/only water values"""
        if num_water > 0.99 * num_real:
            print(f'    On {input_id}: DEM is >99% water')
            inDEM_array[water_ind] -= initial_burn
            burn_add = np.zeros_like(inDEM_array)
            burn_add = (self.SWO_array - SWO_cutoff) * \
                (SWO_burn / (80 - SWO_cutoff))
            burn_add[(self.SWO_array >= 80)] = SWO_burn
            DEM_burn_arr = inDEM_array - burn_add
            water_mask_array[water_ind] = 3

            return DEM_burn_arr, water_mask_array, flatten_flag

        """Create a water mask from SWO and OSM data"""
        water_mask_array = self.create_water_mask(DEM_burn_arr,
                                                  SWO_tif, self.SWO_array,
                                                  outpath,
                                                  SWO_cutoff=SWO_cutoff,
                                                  riv_area_cutoff=1.0,
                                                  keep_water_shp=False,
                                                  require_bound=False,
                                                  outlet_dist=outlet_dist,
                                                  gap_dist=gap_dist)

        river_burn_ind = np.nonzero(water_mask_array == 4)
        if np.count_nonzero(water_mask_array) > 0:
            burn_river = True
        else:
            burn_river = False

        """Create burn values from raster mask for line features"""
        stream_over_lake = False
        if self.waterlines_gdf is not None:
            if self.verbose_print:
                print_time(f"   On {input_id} create stream burn values",
                           self.starttime)

            streamburn_array, sclass_buffer_array = wtr.compute_vec_burn(
                self.waterlines_tif,
                riverburn=1.0,
                stream_burn=0.5,
                drainburn=0.5)

            streamburn_array[inDEM_array == self.nodata] = 0
            # streamburn_tif = os.path.join(
            #     outpath, f"{self.basename}_streamburn.tif")
            # pyct.npy2tif(streamburn_array, inDEM_tif,
            #              streamburn_tif, dtype=gdal.GDT_Float32)
            if stream_over_lake:
                sburn_ind = np.nonzero((streamburn_array > 0) &
                                       (water_mask_array != 4))
            else:
                sburn_ind = np.nonzero((streamburn_array > 0) &
                                       ((water_mask_array < 3) |
                                       (water_mask_array > 4)))
            water_mask_array[sburn_ind] = 12

            if not stream_over_lake:
                # account for 'lakes' completely contained within streams
                lake_mask = np.where(water_mask_array == 3, 1, 0)
                lake_mask_label, num_lake_feat = ndimage.label(lake_mask)
                lake_label_list = [e for e in np.unique(
                    lake_mask_label) if e > 0]
                for label in lake_label_list:
                    lake_label_ind = np.nonzero(lake_mask_label == label)
                    count_lake = len(lake_label_ind[0])
                    count_stream = np.count_nonzero(
                        sclass_buffer_array[lake_label_ind])
                    perc = 100 * (count_stream / count_lake)
                    if perc > 95:
                        water_mask_array[lake_mask_label == label] = 12
            else: # remove small leftover pieces of lake
                lake_mask = np.where(water_mask_array == 3, 1, 0)
                lake_mask_label, num_lake_feat = ndimage.label(lake_mask)
                lake_label_list = [e for e in np.unique(
                    lake_mask_label) if (e > 0) and (
                        np.count_nonzero(lake_mask_label[
                            lake_mask_label==e]))>100]

        else:
            if self.verbose_print:
                print_time(f"   On {input_id} no waterlines to burn stream from",
                           self.starttime)
            streamburn_array = np.zeros_like(inDEM_array)
            sclass_buffer_array = np.zeros_like(inDEM_array)

        if self.verbose_print:
            print_time(
                f"   On {input_id} - Water mask created", self.starttime)

        # temp_tif = os.path.join(
        #     outpath, f"{self.basename}_water_mask_sburn.tif")
        # pyct.npy2tif(water_mask_array, inDEM_tif,
        #              temp_tif, dtype=gdal.GDT_Byte, nodata=0)

        # return DEM_burn_arr, water_mask_array, flatten_flag

        """Initialize water pixels with FABDEM data"""
        no_stream_mask = None
        if fab_tif is not None:
            fab_river_override = True

            DEM_burn_arr,\
                water_mask_array = self.fabdem_init(
                    DEM_burn_arr, fab_tif,
                    water_mask_array,
                    fab_river_override=fab_river_override)

            river_burn_ind = np.nonzero(water_mask_array == 4)
            # dem_burn_tif = os.path.join(outpath, f"{self.basename}_init.tif")
            # pyct.npy2tif(DEM_burn_arr, inDEM_tif,
            #              dem_burn_tif, nodata=self.nodata)
            if self.verbose_print:
                print_time(f"   On {input_id} - FABDEM values initialized",
                           self.starttime)

            no_stream_mask = np.where(
                (water_mask_array == 11) | (water_mask_array == 15), 1, 0)

        # return DEM_burn_arr, water_mask_array, flatten_flag

        """create river bank"""
        edge_burn_array = np.zeros_like(inDEM_array)
        edge_burn_array[river_burn_ind] = 1
        kernel = np.ones((7, 7))
        kernel = (1. / np.sum(kernel)) * kernel
        edge_burn_array = apply_convolve_filter(
            edge_burn_array, kernel)
        bank_ind = np.where((inDEM_array != self.nodata) &
                           ((water_mask_array == 0) | (water_mask_array >= 11)) &
                            (edge_burn_array > 0.0))
        water_mask_array[bank_ind] = 1

        # re-compute streamburn indices
        sburn_ind = np.nonzero((water_mask_array == 12) |
                               ((sclass_buffer_array > 0) & (water_mask_array > 10)))
        water_mask_array[sburn_ind] = 12

        # remove any stream segments contained within extended river mask
        if no_stream_mask is not None:
            stream_test_mask = np.where(water_mask_array == 12, 1, 0)
            labeled_array, num_features = ndimage.label(
                stream_test_mask, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            clump_IDs = np.unique(labeled_array[no_stream_mask == 1])
            for clump in clump_IDs:  # stream segment within no_stream mask?
                clump_ind = np.nonzero(labeled_array == clump)
                if min(no_stream_mask[clump_ind]) > 0:
                    water_mask_array[clump_ind] = 11
            # reset where stream burn will be applied
            sburn_ind = np.nonzero(water_mask_array == 12)

        # reset coastal boarder/ocean pixels
        if self.coast:
            DEM_burn_arr[beach_ind] = ocean_z
            DEM_burn_arr[self.ocean_ind] = self.nodata
            water_mask_array[beach_ind] = 0

        # return DEM_burn_arr, water_mask_array, flatten_flag

        if initial_burn > 0.0:
            """ Apply initial burn """
            DEM_burn_arr[river_burn_ind] -= initial_burn

            if len(bank_ind[0]) > 0:
                norm_edge_burn = np.percentile(
                    edge_burn_array[bank_ind], 90)
                norm_fac = initial_burn / norm_edge_burn
                edge_burn_array = norm_fac * edge_burn_array
                edge_burn_array[river_burn_ind] = 0
                edge_burn_array[edge_burn_array >
                                initial_burn] = initial_burn
                DEM_burn_arr[bank_ind] -= edge_burn_array[bank_ind]
                DEM_burn_arr[inDEM_array == self.nodata] = self.nodata

        if burn_river:
            if self.verbose_print:
                print_time(
                    "    enforce river below surrounding land", self.starttime)
            """check for river segments above surrounding land"""

            DEM_burn_arr, \
                water_mask_array = self.lower_river_below_land(
                    inDEM_tif, DEM_burn_arr, self.SWO_array,
                    water_mask_array, SWO_cutoff=SWO_cutoff)

            """correct for any no data values"""
            DEM_burn_arr[DEM_burn_arr <
                         - 999] = inDEM_array[DEM_burn_arr < -999]

            if self.verbose_print:
                print_time(f"    On {input_id} enforce river monotonicity",
                           self.starttime)
            DEM_burn_arr, \
                water_mask_array = wtr.enforce_monotonic_rivers(
                    inDEM_tif, outpath, DEM_burn_arr,
                    self.SWO_array, water_mask_array,
                    self.donut_array, self.basename,
                    SWO_cutoff=SWO_cutoff,
                    radius=self.radius,
                    strict_endcap_geom=True,
                    wbt=self.wbt)

            """add SWO-based burn to rivers"""
            burn_add = np.zeros_like(DEM_burn_arr)
            if SWO_cutoff < 80:
                burn_add = (self.SWO_array - SWO_cutoff) * \
                    (SWO_burn / (80 - SWO_cutoff))
                burn_add[(self.SWO_array >= 80)] = SWO_burn
                burn_add[burn_add < 0.0] = 0.0  # no negative burn values
                # no SWO-burn > SWO cutoff
                burn_add[(burn_add > SWO_burn)] = 0
            else:
                _SWO_burn = 1.0
                _SWO_cutoff = 40
                burn_add = (self.SWO_array - _SWO_cutoff) * \
                    (_SWO_burn / (90 - _SWO_cutoff))
                burn_add[(self.SWO_array >= 90)] = _SWO_burn
                burn_add[burn_add < 0.0] = 0.0  # no negative burn values
                burn_add[(burn_add > _SWO_burn)] = 0

            riv_ind = np.nonzero((water_mask_array > 0) &
                                (water_mask_array < 8) &
                                 (inDEM_array != self.nodata))
            DEM_burn_arr[riv_ind] -= burn_add[riv_ind]

            """apply minimum water value
            (either 0 or 5th percentile land value - 1m)"""
            if floor_river_value is not None:
                min_water_apply = floor_river_value
                # print('    min_water_apply = ', min_water_apply)
                water_ind = np.nonzero((water_mask_array > 0) &
                                       (inDEM_array != self.nodata))
                inDEM_tif_flatten_water = DEM_burn_arr[water_ind]
                inDEM_tif_flatten_water[inDEM_tif_flatten_water <
                                        min_water_apply] = min_water_apply
                DEM_burn_arr[water_ind] = inDEM_tif_flatten_water

        """Apply stream burn and add to flatten_mask"""
        if np.count_nonzero(streamburn_array) > 0:
            if self.verbose_print:
                print_time(
                    f"    On {input_id} apply stream burn", self.starttime)

            # apply moving window minimum
            DEM_burn_arr,\
                water_mask_array = wtr.stream_minimize(DEM_burn_arr,
                                                       self.waterlines_tif,
                                                       water_mask_array,
                                                       max_burn=15,
                                                       canal_filter_size=21,
                                                       stream_filter_size=25,
                                                       river_filter_size=31,
                                                       canal_as_river=True,
                                                       nodataval=self.nodata,
                                                       no_vector_mask=None)
            # apply stream burn
            DEM_burn_arr[sburn_ind] = DEM_burn_arr[sburn_ind] - \
                streamburn_array[sburn_ind]
            flatten_flag = True

        # return DEM_burn_arr, water_mask_array, flatten_flag

        if self.verbose_print:
            print_time(
                f"    On {input_id} enforce stream monotonicity",
                self.starttime)
        DEM_burn_arr,\
            water_mask_array = wtr.enforce_monotonic_streams(
                inDEM_tif, outpath, DEM_burn_arr, water_mask_array,
                self.donut_array, self.basename,
                radius=self.radius,
                wbt=self.wbt,
                waterlines_tif=self.waterlines_tif)

        # return DEM_burn_arr, water_mask_array, flatten_flag

        """apply minimum value to streams
        (either -.5 or 10th percentile river value + 50cm)"""
        if floor_river_value is not None:
            floor_stream_value = floor_river_value + 0.5
            land_ind = np.where((water_mask_array == 0) &
                                (inDEM_array != self.nodata))
            if len(land_ind[0]) > 0:
                min_land = np.percentile(inDEM_array[land_ind], 1)
                min_stream_apply = floor_stream_value
                if (min_land < 0.5) & (self.waterpolys_tif is not None):  # coast w/ low lying land
                    riverclass_array = gdal_array.LoadFile(
                        self.waterpolys_tif)
                    riv_ind_temp = np.where((water_mask_array > 2) &
                                           (water_mask_array < 8) &
                                            (riverclass_array > 0))
                    # Use OSM to ignore pure ocean pixels
                    if len(riv_ind_temp[0]) > 0:
                        min_riv_temp = np.percentile(
                            DEM_burn_arr[riv_ind_temp], 5)
                        if min_riv_temp < -1.0:
                            min_stream_apply = min_riv_temp + 0.5
                            print(f"   On {input_id} set min stream value to:",
                                  f"{min_stream_apply:.2f}")

                # print("   min_stream_apply = ", min_stream_apply)
                stream_ind = np.where((((water_mask_array > 7) &
                                       (water_mask_array < 11)) | (water_mask_array == 13)) &
                                      (inDEM_array != self.nodata))
                inDEM_tif_streams = DEM_burn_arr[stream_ind]
                inDEM_tif_streams[inDEM_tif_streams <
                                  min_stream_apply] = min_stream_apply
                DEM_burn_arr[stream_ind] = inDEM_tif_streams

        if apply_relative_min:

            if self.verbose_print:
                print(f"    On {input_id} apply relative water heights")
            """apply minimum stream value (above adjacent river value)"""
            DEM_burn_arr = wtr.raise_small_streams(DEM_burn_arr,
                                                   DEM_burn_arr,
                                                   water_mask_array,
                                                   self.nodata,
                                                   extra_raise=0.40)

            """apply minimum canal value (above adjacent river value)"""
            # DEM_burn_arr = wtr.raise_canals(inDEM_array,
            # DEM_burn_arr,
            # water_mask_array,
            # self.SWO_array, self.waterlines_tif,
            # self.nodata,
            # extra_raise=0.40)

            """apply minimum moving window to SWO>90 river pixels"""
            # DEM_burn_arr = wtr.smooth_rivers(inDEM_array,
            # DEM_burn_arr,
            # water_mask_array,
            # self.SWO_array,
            # self.nodata)

        return DEM_burn_arr, water_mask_array, flatten_flag

    def import_osm(self, inDEM_tif_tif, outpath):

        if self.OSM_dir is not None:

            """Import, reclassify and rasterize OpenStreetMap water data"""
            waterlines_shp = os.path.join(
                outpath, f'{self.basename}_OSM_waterlines.shp')
            waterpolys_shp = os.path.join(
                outpath, f'{self.basename}_OSM_waterpolygons.shp')
            self.waterlines_tif = os.path.join(
                outpath, f'{self.basename}_OSM_waterlines.tif')
            self.waterpolys_tif = os.path.join(
                outpath, f'{self.basename}_OSM_waterpolygons.tif')

            self.waterlines_gdf, \
                self.waterpolys_gdf = osm.OSMclip(inDEM_tif_tif, self.OSM_dir,
                                                  self.OSM_tiles, outpath,
                                                  waterlines_shp, waterpolys_shp,
                                                  basin_clip_shp=self.buffer_shp,
                                                  verboseprint=self.verbose_print)

            if not os.path.exists(self.waterpolys_tif) \
                    or (not os.path.exists(self.waterlines_tif)):
                osm.OSMrasterize(waterlines_shp, waterpolys_shp,
                                 inDEM_tif_tif, self.waterlines_gdf, self.waterpolys_gdf,
                                 self.waterlines_tif, self.waterpolys_tif)

            if not os.path.exists(self.waterpolys_tif):
                self.waterpolys_tif = None
            if not os.path.exists(self.waterlines_tif):
                self.waterlines_tif = None
        else:
            self.waterlines_tif = None
            self.waterpolys_tif = None

    def create_water_mask(self, inDEM_array, SWO_tif,
                          SWO_array, outpath,
                          SWO_cutoff=20, riv_area_cutoff=0.5,
                          keep_water_shp=False, require_bound=False,
                          outlet_dist=1e3, gap_dist=240):

        river_flag = False

        edm_array = self.edm_array.copy()
        if edm_array is not None:
            edm_water_mask = np.where(
                (edm_array == 2) | (edm_array == 3), 1, 0)
        else:
            edm_water_mask = np.zeros_like(SWO_array)

        coast_array = self.land_array
        if self.coast:
            coast_array[(self.ocean_mask_buffer > 0) & (inDEM_array > 0.0)] = 0

            # create ocean mask .shp to remove lakes touching ocean

        """exclude land pixels"""
        if self.waterpolys_tif is not None:
            riverclass_array = gdal_array.LoadFile(self.waterpolys_tif)
            land_ind = np.where(((riverclass_array == 0) &
                                (edm_water_mask == 0) &
                                    (SWO_array < SWO_cutoff)) |
                                ((riverclass_array == 0) &
                                   (edm_water_mask == 1) &
                                 (SWO_array < 3)))
            if self.exclude_wetland:  # remove wetland pixels
                land_ind = np.where(((riverclass_array == 0) &
                                    (SWO_array < SWO_cutoff) &
                                      (edm_water_mask == 0)) |
                                    ((riverclass_array == 0) &
                                        (edm_water_mask == 1) &
                                        (SWO_array < 3)) |
                                    ((riverclass_array == 5) &
                                       (SWO_array < 90)))
        else:
            land_ind = np.where(SWO_array < SWO_cutoff)

        water_shp = os.path.join(
            outpath, f"{self.basename}_possible_rivers.shp")
        poss_riv_gdf,\
            water_mask_temp = wtr.find_possible_rivers(
                inDEM_array, self.nodata,
                SWO_tif, water_shp, self.wbt,
                OSM_tif=self.waterpolys_tif,
                data_mask_tif=self.edm_tif,
                SWO_threshold=SWO_cutoff,
                area_threshold=riv_area_cutoff,
                fix_w_buffer=False,
                river_buffer=30,
                exclude_wetland=self.exclude_wetland,
                keep_water_shp=keep_water_shp,
                coast_mask=coast_array)

        s = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
        water_donut_ind = np.nonzero(
            (water_mask_temp == 1) & (self.donut_array == 1))
        labeled_array = np.zeros_like(inDEM_array)
        labeled_array[water_donut_ind] = 1
        labeled_array, num_water_edge_feat = ndimage.label(
            labeled_array, structure=s)
        if self.verbose_print:
            print('       # of border water features total = ',
                  num_water_edge_feat)

        if num_water_edge_feat > 100:
            print(f"    On {self.input_id} - Raise SWO and re-find rivers",
                  f"({num_water_edge_feat} edge features)")
            SWO_cutoff = 40
            poss_riv_gdf,\
                water_mask_temp = wtr.find_possible_rivers(
                    inDEM_array, self.nodata,
                    SWO_tif, water_shp, self.wbt,
                    OSM_tif=self.waterpolys_tif,
                    SWO_threshold=SWO_cutoff,
                    area_threshold=riv_area_cutoff,
                    fix_w_buffer=False,
                    river_buffer=1.0,
                    exclude_wetland=self.exclude_wetland,
                    exclude_water=True,
                    keep_water_shp=keep_water_shp,
                    coast_mask=coast_array)

        if poss_riv_gdf is not None:  # large enough water bodies
            """ Buffer, Dissolve and then separate polygons
            - allows continous rivers to span small gaps such as bridges """
            poss_riv_buf_gdf = pyct.buffer_gdf_simple(
                poss_riv_gdf, gap_dist / 2)
            poss_riv_buf_gdf['group'] = 1
            poss_riv_buf_gdf = poss_riv_buf_gdf.dissolve(by='group')
            poss_riv_buf_gdf = poss_riv_buf_gdf.explode()
            poss_riv_buf_gdf = poss_riv_buf_gdf.reset_index()
            poss_riv_buf_gdf = shapes.calculate_area(poss_riv_buf_gdf)
            poss_riv_buf_gdf['FID'] = list(poss_riv_buf_gdf.index)

            water_buffer_shp = os.path.join(
                outpath, f"{self.basename}_possible_rivers_buffer.shp")
            poss_riv_buf_gdf.to_file(water_buffer_shp)

            if self.verbose_print:
                print_time(
                    f"   Loop through {len(poss_riv_buf_gdf.index)} possible river segments", self.starttime)

            """Loop through each possible water body, see if it qualifies as a continous river"""
            burn_river_gdf = None
            for index, row in poss_riv_buf_gdf.iterrows():
                river_flag = None
                test_river_gdf = poss_riv_buf_gdf.loc[[index], :].copy()
                FID = test_river_gdf.FID.values[0]
                test_river_gdf = test_river_gdf.reset_index()
                test_river_gdf = test_river_gdf.filter(
                    ['index', 'geometry', 'area'], axis=1)
                test_river_poly = list(test_river_gdf.geometry)[0]

                river_crossing_gdf = gpd.overlay(
                    test_river_gdf, self.donut_gdf, how='intersection')
                river_crossing_gdf = river_crossing_gdf.filter(
                    ['index', 'geometry'], axis=1)
                river_crossing_gdf = river_crossing_gdf.explode()

                # un-buffered river segment(s)
                burn_segment_gdf = gpd.overlay(test_river_gdf,
                                               poss_riv_gdf,
                                               how='intersection')
                burn_segment_gdf = shapes.calculate_area(burn_segment_gdf)

                if self.waterpolys_gdf is not None:  # check for OSM river
                    osm_intersect_gdf = gpd.overlay(
                        self.waterpolys_gdf,
                        burn_segment_gdf, how='intersection')
                    if not osm_intersect_gdf.empty:
                        osm_intersect_gdf = shapes.calculate_area(
                            osm_intersect_gdf)
                        if 'river' in osm_intersect_gdf['fclass'].values:
                            river_flag = True
                        elif 'lake' in osm_intersect_gdf['fclass'].values:
                            # if test river is >90% OSM lake and 0% OSM river
                            osm_lake_gdf = osm_intersect_gdf[
                                osm_intersect_gdf.fclass=='lake'].copy()
                            osm_lake_area = osm_lake_gdf.area.sum()
                            test_river_area = burn_segment_gdf.area.sum()
                            area_frac = osm_lake_area/test_river_area
                            # print(f'   OSM lake = {100*area_frac:.0f}% (FID={FID})')
                            if area_frac > 0.9:
                                river_flag = False

                if len(river_crossing_gdf.index) == 0:  # no river crossing boundary
                    if (require_bound) and (river_flag is not True):
                        # print(f"   FID={FID} no boundary crossing")
                        continue

                # inlet AND outlet in same water body
                if len(river_crossing_gdf.index) > 1 and (river_flag is not False):
                    """buffer to ensure crossings are not too close to one another"""
                    cross_buff = outlet_dist / 2  # at least x km apart
                    river_crossing_gdf = pyct.buffer_gdf_simple(
                        river_crossing_gdf, cross_buff)
                    river_crossing_gdf = gpd.overlay(
                        river_crossing_gdf, self.donut_gdf, how='intersection')
                    river_crossing_gdf = river_crossing_gdf.filter(
                        ['index', 'geometry'], axis=1)
                    river_crossing_gdf['group'] = 1
                    river_crossing_gdf = river_crossing_gdf.dissolve(
                        by='group')
                    river_crossing_gdf = river_crossing_gdf.explode()

                    """Check that crossings include river pixels"""
                    for kindex, row in river_crossing_gdf.iterrows():  # Looping over all polygons
                        test_crossing_gdf = river_crossing_gdf.loc[[
                            kindex], :].copy()
                        test_crossing_poly = list(
                            test_crossing_gdf.geometry)[0]
                        if not test_crossing_poly.intersects(test_river_poly):
                            river_crossing_gdf.drop([kindex], inplace=True)

                if len(river_crossing_gdf.index) < 2:
                    if (require_bound) and (river_flag is not True):
                        # print(f"   FID={FID} one boundary crossing")
                        continue

                elif (len(river_crossing_gdf.index) > 1) and (river_flag is not False):
                    river_flag = True

                """pull out river segment shapes that overlap buffered shape"""
                burn_segment_gdf = gpd.overlay(test_river_gdf,
                                               poss_riv_gdf, how='intersection')
                if burn_segment_gdf.empty:
                    # print(f"   FID={FID} no burn segment")
                    continue

                """river segments that cross boundary more than 4 times, require OSM too"""
                n_riv_boundary = len(river_crossing_gdf.index)
                if (n_riv_boundary > 4) and (not self.coast):  # require OSM too
                    if self.waterpolys_gdf is not None:
                        if river_flag is not False:
                            burn_segment_gdf_copy = shapes.calculate_area(
                                burn_segment_gdf)
                            burn_segment_area = burn_segment_gdf_copy['area'].sum()

                            osm_intersect_gdf = gpd.overlay(
                                burn_segment_gdf, self.waterpolys_gdf,
                                how='intersection')
                            osm_intersect_gdf = shapes.calculate_area(
                                osm_intersect_gdf)
                            osm_area = osm_intersect_gdf['area'].sum()

                            if osm_area < 0.1 * burn_segment_area:
                                river_flag = False
                    else:
                        # print(f"   FID={FID} no no waterpolys")
                        continue

                if river_flag:
                    burn_segment_gdf['watertype'] = 2
                else:
                    burn_segment_gdf['watertype'] = 1

                if burn_river_gdf is None:
                    burn_river_gdf = burn_segment_gdf
                else:
                    burn_river_gdf = gpd.GeoDataFrame(pd.concat(
                        [burn_river_gdf, burn_segment_gdf],
                        ignore_index=True), crs="EPSG:4326")

            if burn_river_gdf is not None:

                # remove lakes touching ocean
                if self.coast:
                    for ocean_poly in list(self.oceanmask_gdf.geometry):
                        ocean_bool = burn_river_gdf.intersects(ocean_poly)
                        burn_river_gdf['ocean'] = ocean_bool
                        burn_river_gdf = burn_river_gdf.loc[~((
                            burn_river_gdf['ocean']) & (
                                burn_river_gdf['watertype'] == 1))]

                burnriver_shp = os.path.join(
                    outpath, f"{self.basename}_river.shp")
                burn_river_gdf.to_file(burnriver_shp)
                burnriver_mask = os.path.join(
                    outpath, f"{self.basename}_river_mask.tif")

                pyct.poly_rasterize(burnriver_shp, self.blankmask_tif,
                                    burnriver_mask, attr='watertype')

                river_array = gdal_array.LoadFile(burnriver_mask)
                if not keep_water_shp:
                    river_array[land_ind] = 0
                river_array[inDEM_array == self.nodata] = 0
                river_array[water_mask_temp == 0] = 0

                river_flag = True

        """Output mask of flattened indices"""
        if river_flag:
            water_mask_array = np.zeros_like(water_mask_temp)
            water_mask_array[river_array == 1] = 3
            water_mask_array[river_array == 2] = 4
            # if self.edm_tif is not None:  # add rivers from TDF only
            #     if self.src_dem == "TDF":
            #         water_mask_array[self.edm_array == 3] = 4
            water_mask_array
            water_mask_array[land_ind] = 0
        else:
            water_mask_array = np.zeros_like(inDEM_array, dtype=np.int8)

        return water_mask_array

    def fabdem_init(self, DEM_arr, fab_tif, water_mask_array,
                    fab_river_override=True):

        replace_EDM_water = False
        if self.src_dem == "TDT":
            replace_EDM_water = True

        water_mask_edit = water_mask_array.copy()
        DEM_out_arr = DEM_arr.copy()

        fabdem_array = gdal_array.LoadFile(fab_tif)
        fabdem_array[DEM_arr == self.nodata] = self.nodata
        river_ind = np.where(water_mask_array == 4)
        river_orig_array = np.zeros_like(water_mask_array)
        river_orig_array[river_ind] = 1
        river_label, num_riv_feat = ndimage.label(river_orig_array)
        riv_labels, riv_counts = np.unique(
            river_label, return_counts=True)
        riv_feat_list = [e for e in riv_labels if (
            e > 0) and (riv_counts[riv_labels == e] > 1000)]
        num_riv_feat = len(riv_feat_list)

        # temp_tif = os.path.join(
        #     self.outpath, f"{self.basename}_river_label.tif")
        # pyct.npy2tif(river_label, self.dem_tif, temp_tif, nodata=0)

        if (replace_EDM_water) and (self.edm_tif is not None):

            if self.src_dem == "TDT": # include fill connected to rivers
                edm_water_ind = np.where((self.edm_array == 2) |  # lake
                                     (self.edm_array == 3) |  # river
                                     (self.edm_array == 5))  # fill
            else:
                edm_water_ind = np.where((self.edm_array == 2) |  # lake
                                     (self.edm_array == 3))  # river
                                     
            edm_riv_ind = np.where(((self.edm_array == 2) |  # lake
                                   (self.edm_array == 3)) &  # river
                                   ((water_mask_array == 4) |
                                      (water_mask_array == 12)))

            """for river segments, include overlapping EDM lakes
            and rivers in fabdem initialization"""
            riv_edm_array = np.zeros_like(water_mask_array)
            riv_edm_array[edm_water_ind] = 1
            edm_mask_label, num_edm_feat = ndimage.label(riv_edm_array)
            edm_riv_label = edm_mask_label[edm_riv_ind]
            edm_riv_label = [e for e in np.unique(
                edm_riv_label) if e > 0]
            riv_edm_mask = np.isin(edm_mask_label,
                                   np.unique(edm_riv_label))

        """Extract FABDEM river mask.
        Conditioned rivers in fabdem have 0.5m increment values,
        identify large continuous segments"""
        fab_diff = fabdem_array % 0.5  # isolate probable water values
        fab_mask = np.zeros_like(water_mask_array)
        fab_mask[fab_diff == 0.0] = 1
        fab_mask[self.SWO_array < 2] = 0  # only consider SWO water pixels
        # erase single pixel mask points
        fab_mask = apply_convolve_filter(
            fab_mask, np.ones((3, 3)), normalize_kernel=True)
        fab_mask[fab_mask < 1] = 0
        fab_mask = apply_convolve_filter(fab_mask, np.ones((3, 3)))
        fab_mask[fab_mask > 0] = 1

        # remove ocean-adjacent pixels
        if self.coast:
            fab_mask[self.land_array == 0] = 0
            fab_mask[(self.ocean_mask_buffer > 0) & (DEM_arr > 0.0)] = 0

        """mask of all contiguous clumps of fabdem water
        that intersect river pixels. """
        fab_mask_label, num_fab_feat = ndimage.label(fab_mask)
        fab_riv_label = fab_mask_label[river_ind]
        fab_riv_label = [e for e in np.unique(fab_riv_label) if e > 0]
        fab_extend_mask = np.isin(fab_mask_label,
                                  np.unique(fab_riv_label))

        # temp_tif = os.path.join(
        #     self.outpath, f"{self.basename}_fabmask_label.tif")
        # pyct.npy2tif(fab_mask_label, self.dem_tif, temp_tif, nodata=0)

        """single mask of all if the indices to be initialized with
        FABDEM, buffer by one pixel for edge effects, and apply that"""
        dem_fab_mask = np.zeros_like(water_mask_array)
        dem_fab_mask[fab_extend_mask] = 1  # FABDEM river pixels
        dem_fab_mask = apply_convolve_filter(dem_fab_mask,
                                             np.ones((3, 3)))
        # buffered FABDEM river pixels
        dem_fab_mask[dem_fab_mask > 0] = 1
        fab_only_ind = np.nonzero(dem_fab_mask)
        fab_edm_extend_mask = np.copy(dem_fab_mask)
        # fab_edm_extend_mask[river_ind] = 1 # add in original river mask
        if replace_EDM_water:  # overlapping EDM water pixels
            fab_edm_extend_mask[riv_edm_mask] = 1
        apply_fab_ind = np.nonzero(fab_edm_extend_mask)
        DEM_out_arr[apply_fab_ind] = fabdem_array[apply_fab_ind]

        # interopolate edges
        fab_edge_mask = apply_convolve_filter(fab_edm_extend_mask,
                                              np.ones((5, 5)))
        fab_edge_ind = np.nonzero((fab_edge_mask > 0.0) &
                                  (fab_edm_extend_mask == 0))  # edge pixels

        DEM_out_arr[fab_edge_ind] = self.nodata
        edge_tif0 = os.path.join(self.outpath, f"{self.basename}_fabedge.tif")
        pyct.npy2tif(DEM_out_arr, self.dem_tif,
                     edge_tif0, nodata=self.nodata)
        edge_tif2 = os.path.join(
            self.outpath, f"{self.basename}_fabedge_interp.tif")
        self.wbt.fill_missing_data(edge_tif0, edge_tif2,
                                   filter=11, weight=2, no_edges=False)

        DEM_burn_edge_arr = gdal_array.LoadFile(edge_tif2)
        DEM_out_arr[fab_edge_ind] = DEM_burn_edge_arr[fab_edge_ind]
        DEM_out_arr[DEM_arr == self.nodata] = self.nodata

        """mark pixels that are rivers only in FABDEM/EDM"""
        fab_edm_extend_mask[river_ind] = 1  # add in original river mask
        if not fab_river_override:
            fab_edm_extend_mask[river_ind] = 0
        water_mask_edit[np.nonzero(fab_edm_extend_mask)] = 11
        water_mask_edit[fab_edge_ind] = 15

        # reset lake mask
        water_mask_edit[water_mask_array == 3] = 3

        # For lakes, use lowest value between FABDEM and TDX
        lake_ind = np.nonzero(water_mask_array == 3)
        fab_lake_elev = fabdem_array[lake_ind]
        dem_lake_elev = DEM_out_arr[lake_ind]
        lake_min_elev = np.fmin(fab_lake_elev, dem_lake_elev)
        DEM_out_arr[lake_ind] = lake_min_elev

        if fab_river_override:
            """Re-draw river mask"""
            new_riv_array = np.zeros_like(water_mask_array)
            new_riv_array[(water_mask_array == 11) &
                          (self.SWO_array >= self.SWO_cutoff)] = 1
            new_riv_array[fab_only_ind] = 1

            # remove ocean-adjacent pixels
            if self.coast:
                new_riv_array[self.land_array == 0] = 0
                new_riv_array[self.ocean_ind] = 0
                new_riv_array[(self.ocean_mask_buffer > 0) &
                              (DEM_arr > 0.0)] = 0

            new_riv_label, num_nriv_feat = ndimage.label(new_riv_array)
            # temp_tif = os.path.join(
            #     self.outpath, f"{self.basename}_new_river_label.tif")
            # pyct.npy2tif(new_riv_label, self.dem_tif,
            #              temp_tif, nodata=0)
            if self.verbose_print:
                print(f"   # new river features = {num_nriv_feat}")
                print(f"   riv_feat_list = {riv_feat_list}")
            if num_nriv_feat > 1:
                for riv_feat in riv_feat_list:
                    riv_count = riv_counts[np.argwhere(
                        riv_labels == riv_feat)][0][0]
                    #print(f"    river feature = {riv_feat}:",
                    #      f"pixels = {riv_count}")
                    riv_feat_ind = np.nonzero(river_label == riv_feat)
                    fab_riv_label = new_riv_label[riv_feat_ind]
                    riv_feat_labels, riv_feat_counts = np.unique(
                        fab_riv_label, return_counts=True)
                    fab_riv_label = [e for e in riv_feat_labels
                                     if (e > 0) and (riv_feat_counts[
                                         riv_feat_labels == e] > 1e3)]
                    new_riv_mask = np.isin(new_riv_label,
                                           np.unique(fab_riv_label))
                    riv_feat_label = new_riv_label[new_riv_mask]
                    labels, counts = np.unique(riv_feat_label,
                                               return_counts=True)

                    if labels.size == 0:
                        #print(f'     {self.input_id} keep old river mask')
                        water_mask_edit[riv_feat_ind] = 4
                    else:
                        biggest_feat = labels[np.argmax(counts)]
                        big_feat_count = np.max(counts)
                        per_new_riv = 100 * (big_feat_count / riv_count)
                        if self.verbose_print:
                            print("   % new river vs old river =",
                                  f"{per_new_riv:.0f}%  (#pixels={riv_count})")
                        # water_mask_edit[new_riv_label ==
                        #                biggest_feat] = 4

                        if per_new_riv >= 50 and per_new_riv <= 200:
                            # print('     apply new river mask')
                            water_mask_edit[new_riv_label
                                            == biggest_feat] = 4
                        else:
                            # print('     keep old river mask')
                            water_mask_edit[riv_feat_ind] = 4
            else:
                water_mask_edit[np.nonzero(new_riv_array)] = 4

        water_mask_array = water_mask_edit

        return DEM_out_arr, water_mask_array

    def lower_river_below_land(self, inDEM_tif, inDEM_array, SWO_array, water_mask_array,
                               SWO_cutoff=20):

        riv_ind = np.where((water_mask_array == 4) |
                           (water_mask_array == 3))

        water_mask_edit = water_mask_array.copy()
        DEM_river = inDEM_array[riv_ind]

        if DEM_river.size < 100:  # very few river pixels
            print(
                f"        {self.input_id} - too few river pixels to lower below land")
            return inDEM_array, water_mask_array

        min_river_h = np.maximum(-2.0, np.percentile(DEM_river, 3))
        min_river_h = np.floor(min_river_h) - 1.25
        max_river_h = np.maximum(0, np.percentile(DEM_river, 99))
        max_river_h = np.ceil(max_river_h) + 0.25

        raised_step = 0.5
        if (max_river_h - min_river_h) > 10:
            raised_step = 1.0
        if (max_river_h - min_river_h) > 100:
            raised_step = 2.0
        if (max_river_h - min_river_h) > 200:
            raised_step = 5.0
        river_h_steps = np.arange(
            min_river_h, max_river_h + raised_step, raised_step)
        n_riv_pixels = len(riv_ind[0])
        if self.verbose_print:
            print(
                f'      river h = {min_river_h}/{max_river_h}, step={raised_step}')

        river_array = np.zeros_like(inDEM_array)
        river_array[riv_ind] = inDEM_array[riv_ind]
        left_array = np.ones_like(water_mask_array)
        interp_mask = np.zeros_like(inDEM_array)
        edge_interp = False
        s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

        for h in river_h_steps:
            h_max = h + raised_step
            riv_h_ind = np.where((river_array >= h) &
                                 (river_array <= h_max))
            labeled_array = np.zeros_like(inDEM_array)
            labeled_array[riv_h_ind] = 1
            labeled_array[left_array == 0] = 0
            labeled_array, num_features = ndimage.label(labeled_array,
                                                        structure=s)
            clump_ind = np.nonzero((labeled_array > 0))
            clump_IDs = np.unique(labeled_array[clump_ind])
            # print_time(f"     number clumps within {h}-{h_max} = {len(clump_IDs)}", starttime)
            # clump_tif = os.path.join(self.outpath, "{0}_lbl_clump_{1}.tif".format(self.basename, h))
            # pyct.npy2tif(labeled_array, inDEM_tif, clump_tif, nodata=0, dtype=gdal.GDT_Int16)

            for clumpID in clump_IDs:
                clumpID_ind_temp = np.nonzero(
                    labeled_array[clump_ind] == clumpID)[0]
                n_clump_ind = len(clumpID_ind_temp)
                if n_clump_ind < 50:
                    continue

                clumpID_ind = (clump_ind[0][clumpID_ind_temp],
                               clump_ind[1][clumpID_ind_temp])
                DEM_clump = inDEM_array[clumpID_ind]
                min_r_clump = np.percentile(DEM_clump, 25)

                # clip to buffered box around clump
                ind0_min, ind0_max = clumpID_ind[0].min(
                ) - 19, clumpID_ind[0].max() + 20
                ind1_min, ind1_max = clumpID_ind[1].min(
                ) - 19, clumpID_ind[1].max() + 20
                clump_arr = labeled_array[ind0_min:ind0_max,
                                          ind1_min:ind1_max]
                clump_mask = np.zeros_like(clump_arr)
                clump_mask[clump_arr == clumpID] = 99

                # find adjacent land pixels
                DEM_array_temp = inDEM_array[ind0_min:ind0_max,
                                             ind1_min:ind1_max]
                flatten_mask_temp = water_mask_array[ind0_min:ind0_max,
                                                     ind1_min:ind1_max]
                SWO_array_temp = SWO_array[ind0_min:ind0_max,
                                           ind1_min:ind1_max]

                # check boarder to ensure clump touches 2 river banks
                c_border_array = apply_convolve_filter(
                    clump_mask, np.ones((3, 3)))
                c_border_bank_ind = np.where((DEM_array_temp != self.nodata) &
                                            (clump_mask < 99) &
                                             (c_border_array > 0.0) &
                                             (flatten_mask_temp < 3))
                n_border_bank = len(c_border_bank_ind[0])
                test_bank_arr = np.zeros_like(clump_arr)
                test_bank_arr[c_border_bank_ind] = 1
                labeled_arr, num_bank_clump = ndimage.label(
                    test_bank_arr, structure=s)

                c_border_riv_ind = np.where((DEM_array_temp != self.nodata) &
                                           (clump_mask < 99) &
                                            (c_border_array > 0.0) &
                                            (flatten_mask_temp > 2))
                n_border_riv = len(c_border_riv_ind[0])

                if n_border_bank == 0:  # completely surrounded by water
                    continue

                if n_border_riv > 0:
                    per_bank_clump = 100 * \
                        (n_border_bank / (n_border_riv + n_border_bank))
                    if (num_bank_clump < 2) or (per_bank_clump < 20):
                        continue

                # buffer to find nearby land pixels
                adj_land_array = apply_convolve_filter(
                    clump_mask, np.ones((31, 31)))
                adj_land_ind_temp = np.where((DEM_array_temp != self.nodata) &
                                            (flatten_mask_temp < 3) &
                                             (SWO_array_temp < SWO_cutoff) &
                                             (adj_land_array > 0.0) &
                                             (clump_mask < 99))

                if len(adj_land_ind_temp[0]) == 0:  # no nearby land
                    continue

                # shift back to whole array
                adj_land_ind = (adj_land_ind_temp[0] + ind0_min,
                                adj_land_ind_temp[1] + ind1_min)
                min_adj_land = np.percentile(
                    inDEM_array[adj_land_ind], 5)

                r_drop = min_r_clump - (min_adj_land - 0.5)
                if r_drop > 30.0:
                    min_adj_land = np.percentile(
                        inDEM_array[adj_land_ind], 10)
                    r_drop = min_r_clump - (min_adj_land - 1.0)

                if r_drop > 0.0:  # river above nearby land
                    # print(f"      {input_id}-Lower seg {clumpID} {r_drop} meters")
                    buffer_clump_temp = apply_convolve_filter(
                        clump_mask, np.ones((7, 7)))
                    buf_clump_ind = np.where(((clump_arr == clumpID) |
                                             (clump_arr == 0)) &
                                             (flatten_mask_temp > 2) &
                                             (buffer_clump_temp > 0.0) &
                                             (DEM_array_temp >=
                                                (min_r_clump - r_drop)))

                    # if number of river pixels in buffered clump is
                    # >2.5x n_clump_ind, likely a narrow river bank strip, skip
                    n_riv_buffer_ind = len(buf_clump_ind[0])
                    if n_riv_buffer_ind >= 2.5 * n_clump_ind:
                        continue

                    drop_ind = clumpID_ind
                    inDEM_array[drop_ind] = inDEM_array[drop_ind] - r_drop
                    left_array[drop_ind] = 0
                    water_mask_edit[drop_ind] = 5
                    min_allowed = min_r_clump - r_drop - 1.0
                    too_low_temp = np.where(
                        inDEM_array[drop_ind] < min_allowed)[0]
                    too_low_ind = (
                        drop_ind[0][too_low_temp], drop_ind[1][too_low_temp])
                    inDEM_array[too_low_ind] = min_allowed

                    if r_drop > 3.0:
                        if self.verbose_print:
                            print(f"       {self.input_id}-Lowering river segment" +
                                  f"{clumpID} additional {r_drop:.2f} meters")

                        # buffer to find bank pixels
                        riv_bank_ind_temp = np.where((DEM_array_temp != self.nodata) &
                                                    (flatten_mask_temp < 2) &
                                                     (buffer_clump_temp > 0.0))
                        # adjacent bank pixels
                        if len(riv_bank_ind_temp[0]) > 0:
                            riv_bank_ind = (riv_bank_ind_temp[0] + ind0_min,
                                            riv_bank_ind_temp[1] + ind1_min)
                            interp_mask[riv_bank_ind] = 2
                            edge_interp = True

                    # buffer to find up/downstream river pixels
                    adj_riv_ind_temp = np.where((DEM_array_temp != self.nodata) &
                                               (clump_mask == 0.0) &
                                                (flatten_mask_temp > 2) &
                                                (flatten_mask_temp < 6) &
                                                (buffer_clump_temp > 0.0))

                    # adjacent river pixels
                    if len(adj_riv_ind_temp[0]) > 0:
                        adj_riv_ind = (adj_riv_ind_temp[0] + ind0_min,
                                       adj_riv_ind_temp[1] + ind1_min)
                        interp_mask[adj_riv_ind] = 1
                        edge_interp = True

        # reset lake mask to avoid monotonicity enforcement
        water_mask_edit[water_mask_array == 3] = 3

        if edge_interp:  # Interpolate removed edge pixels
            interp_ind = np.nonzero(interp_mask)
            inDEM_array[interp_ind] = self.nodata
            edge_tif0 = os.path.join(
                self.outpath, f"{self.basename}_riveredge.tif")
            pyct.npy2tif(inDEM_array, inDEM_tif,
                         edge_tif0, nodata=self.nodata)
            edge_tif2 = os.path.join(
                self.outpath, f"{self.basename}_riveredge_interp.tif")
            self.wbt.fill_missing_data(
                edge_tif0, edge_tif2, filter=17, weight=2)

            DEM_burn_edge_arr = gdal_array.LoadFile(edge_tif2)
            inDEM_array[interp_ind] = DEM_burn_edge_arr[interp_ind]
            water_mask_edit[interp_mask == 2] = 2

        return inDEM_array, water_mask_edit


def run_enforce_river(inputID):
    """ Tiny wrapper function to make the mpi executor behave """
    rivEnforce.run_river_enforce(inputID)


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
    #################### SET OPTIONS  ####################
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    run_multiprocess = config.getboolean("job-options", "run_multiprocess")
    run_parallel = config.getboolean("job-options", "run_parallel")
    verbose_print = config.getboolean(
        "processes-depression-handling", "verbose_print")
    overwrite = config.getboolean("processes-depression-handling", "overwrite")
    del_temp_files = config.getboolean(
        "processes-depression-handling", "del_temp_files")

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
    base_unit = config.get("region", "base_unit")

    """locations of input data"""
    OSM_dir = config.get("paths", "OSM_dir")
    OSM_tiles = config.get("paths", "OSM_shp")

    SWO_vrt = os.path.join(config.get("paths", "SWO_vrt"))
    inDS_vrt = os.path.join(config.get("outputs", "dtm_mosaic"))
    buffer = config.getfloat("parameters-depression-handling", "buffer")
    radius = config.getint("parameters-depression-handling", "radius")
    if base_unit == 'basin':
        basins_merge_shp = os.path.join(
            config.get('outputs', 'basins_merge_shp'))
        basins_dissolve_shp = os.path.join(
            config.get('outputs', 'basins_dissolve_shp'))
    else:
        basins_merge_shp = config.get("paths", "hybas_lvl1_shp")
        basins_dissolve_shp = None

    """   Water-flattening parameters  """
    SWO_cutoff = config.getint("parameters-depression-handling", "SWO_cutoff")
    initial_burn = config.getfloat(
        "parameters-depression-handling", "initial_burn")
    SWO_burn = config.getfloat("parameters-depression-handling", "SWO_burn")
    burn_cutoff = config.getfloat(
        "parameters-depression-handling", "burn_cutoff")
    gap_dist = config.getint("parameters-depression-handling", "gap_dist")

    """output directories"""
    projectname = config.get("outputs", "projectname")
    work_dir = config.get("outputs", "dh_work_dir")
    project_paths_txt = os.path.join(config.get("paths", "project_txt_path"),
                                     f'{projectname}_paths.txt')

    edm_mosaic = os.path.join(config.get("outputs", "edm_mosaic"))
    fabdem_mosaic = os.path.join(config.get("outputs", "fabdem_mosaic"))

    ###############################################
    """          Run river flattening           """
    ###############################################

    """Initialize class with input parameters"""
    rivEnforce = EnforceRivers(inDS_vrt, work_dir,
                               SWO_vrt, projectname,
                               WBT_path, project_paths_txt,
                               OSM_dir=OSM_dir, OSM_tiles=OSM_tiles,
                               verbose_print=verbose_print,
                               del_temp_files=del_temp_files,
                               outlet_dist=1e3,
                               buffer=buffer, coastal_buffer=1e3,
                               radius=radius, SWO_cutoff=SWO_cutoff,
                               initial_burn=initial_burn, SWO_burn=SWO_burn,
                               burn_cutoff=burn_cutoff, gap_dist=gap_dist,
                               edm_mosaic=edm_mosaic,
                               fabdem_mosaic=fabdem_mosaic,
                               basins_merge_shp=basins_merge_shp,
                               basins_dissolve_shp=basins_dissolve_shp)

    """find unenforced basins"""
    if not overwrite:
        unfin_ids = logt.find_unfinished_ids(work_dir,
                                             unit=base_unit,
                                             dir_search=f'{base_unit}_',
                                             tifname='DEMburned.tif')
    else:
        unfin_ids = logt.find_all_ids(work_dir,
                                      unit=base_unit,
                                      dir_search=f'{base_unit}_')

    # unfin_ids = ["N24E067"]
    n_unfin = len(unfin_ids)

    """Run river flattening"""
    if run_parallel:
        if rank == 0:
            print_time(
                f'river enforcement for {n_unfin} {base_unit}s', starttime)
        with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
            if executor is not None:

                print_time(
                    f'Water enforce {len(unfin_ids)} {base_unit}s on {numnodes} nodes', starttime, sep=True)
                executor.map(run_enforce_river, unfin_ids)

    elif run_multiprocess:

        nthreads = min(n_unfin, 8)
        with Pool(nthreads) as p:
            print_time(f'river enforcement for {n_unfin}'
                       + f' {base_unit}s on {nthreads} threads',
                       starttime, sep=True)
            p.map(rivEnforce.run_river_enforce, unfin_ids)

    else:
        print_time(f'{len(unfin_ids)} {base_unit}s remain', starttime)
        for input_id in unfin_ids:
            rivEnforce.run_river_enforce(input_id)
            print_time(f"{base_unit} {input_id} COMPLETE", starttime)

        log_str = log_time("RIVER ENFORCEMENT COMPLETE", starttime)
        with open(log_file, 'w') as f:
            f.write(log_str)
        print(log_str)
