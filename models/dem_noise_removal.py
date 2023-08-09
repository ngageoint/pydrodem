#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Kimberly McCormack

:Last edited on: 07/20/2020


"""

import os
import sys
import geopandas as gpd
import time
import traceback
import warnings
warnings.simplefilter('ignore')
import glob
import json
import scipy
import numpy as np
import pandas as pd
from osgeo import gdal, gdal_array
import argparse
from configparser import ConfigParser, ExtendedInterpolation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.build_operators as bop
import tools.convert as pyct
import tools.derive as pdt
import tools.log as logt
import tools.veg_remove as pvt
from tools.files import tilesearch
from tools.find_neighbors import findNeighbors, get_nb_paths
from tools.print import print_time, log_time
from tools import aws
import tools.files as ft


class CreateDTM():

    def __init__(self, input_ids, input_paths, input_shps, filtertype,
                 slopebreaks, curvebreaks, SWOlevel,
                 WBT_path, tile_buffer=100,
                 model_file_path=None, landcover_data_path=None,
                 SWO_vrt=None, treecover_vrt=None,
                 treecover_fill_vrt=None,
                 treecover_fill2_vrt=None,
                 treeheight_vrt=None, 
                 fabdem_mosaic=None,
                 temp_data_path=None,
                 output_path=None,
                 veg_remove=True, veg_model='3Dtable',
                 base_DEM='TDT', gdir=4, usegeometric=True, computeD8=False,
                 save_veg_files=None, overwrite=False, buffercell=True,
                 pull_crosssections=False):
        """
        Smooth and apply vegetation removal for a DEM

        Parameters
        ----------
        project_tiles_txt : path
            txt file containing list of tandem-x cells to be processed
        DEMpath_list : path
            text file with full paths to all DEM tiles
        filtertype : str
            type of smoothing filter to apply. 'DW'=distance weighted (linear)
        slopebreaks : list
            list of slope values (degrees) to at which to apply different filter sizes
        curvebreaks : list
            list of geometric curvature values (as factor of standard deviation) to apply
            smallest (3x3) or no filter to
        SWOlevel : int
            Surface water occurence level to consider 'water' pixel
        tile_buffer : int
            buffer (in pixels) around DEM
        model_file_path : path, optional
            directory of vegetation model .h5 files, by default None
        landcover_data_path : path, optional
            directory of vegetation biome data files, by default None
        veg_remove : bool, optional
            Remove vegetation bias?, by default True
        veg_model : str, optional
            'poly' = polymonial model. 'table' = lookup table, by default 'table'
        base_DEM : str, optional
            DEM base_DEM, by default 'TDT'
        gdir : int, optional
            number of gradient directions for slope computation [2 or 4] , by default 4
        usegeometric : bool, optional
            use geometric curvature? If False, Laplacian curvature is used, by default True
        computeD8 : bool, optional
            compute D8 flow direction? This implementation can cause memory issues.
            Recommend using taudem, whiteboxtools or another software, by default False
        save_veg_files : str, optional
            save vegtation bias tifs?, [None, 'bias', 'all'] default None
        overwrite : bool, optional
            overwrite existing output files?, by default False
        buffercell : bool, optional
            use 'tile_buffer' param to process buffered cell?, by default True
        pull_crosssections : bool, optional
            extract and save cross sections of output DEMs?, by default False

        """
        self.starttime = time.time()
        self.success = False

        """ Imported Options """
        self.WBT_path = WBT_path
        self.input_ids = input_ids
        self.input_paths = input_paths
        self.input_shps = input_shps
        self.filtertype = filtertype  # filter_type
        self.slopebreaks = slopebreaks  # slope_thresholds
        self.curvebreaks = curvebreaks  # [fill thres, curve_thres]
        self.usegeometric = usegeometric  # Use geometric curvature. default is Laplacian
        self.SWOlevel = SWOlevel  # water_occ_threshold
        self.buffer = tile_buffer  # nbb
        self.overwrite = overwrite
        self.gdir = gdir
        self.computeD8 = computeD8
        self.buffercell = buffercell
        self.veg_remove = veg_remove
        self.veg_model = veg_model  # polynomial or lookup table
        self.save_veg_files = save_veg_files
        self.pull_crosssections = pull_crosssections
        self.model_file_path = model_file_path
        self.landcover_data_path = landcover_data_path

        self.SWO_vrt = SWO_vrt
        self.treecover_vrt = treecover_vrt
        self.treecover_fill_vrt = treecover_fill_vrt
        self.treecover_fill2_vrt = treecover_fill2_vrt
        self.treeheight_vrt = treeheight_vrt
        self.fabdem_mosaic = fabdem_mosaic
        self.temp_data_path = temp_data_path
        self.output_path = output_path

        """Load WhiteBox Tools executable"""
        sys.path.append(WBT_path)
        from whitebox_tools import WhiteboxTools
        self.wbt = WhiteboxTools()
        self.wbt.set_verbose_mode(False)
        self.wbt.set_whitebox_dir(WBT_path)  # set location of WBT executable

        """ Other Options - generally static"""
        self.s0ind = 2  # 0=9x9, 1=7x7, 2=5x5, 3=3x3

        if len(self.input_ids) == 0:
            raise Exception("no cells in text file {0}".format(
                self.project_tiles_txt))

        """Set the weights based on filter type"""
        self.fweights, self.fsizes = self.setfweights()

    def run_tile(self, input_id, mpi=False):
        """
        Apply smoothing and vegetation removal to an input DEM

        Parameters
        ----------
        input_id : str
            name of tanDEM-X cell
        mpi : bool, optional
            run in parallel?, by default True
        """
        base_unit = "tile"
        nodataval = -9999

        if mpi:
            from mpi4py import MPI
            print_time("On {0} process {1}".format(MPI.COMM_WORLD.rank,
                                                   input_id), self.starttime)
        else:
            print_time("Start processing {0}".format(
                input_id), self.starttime)

        self.outpath = os.path.join(self.output_path,
                               f'{base_unit}_{input_id}')

        """Check if DEM has been smoothed"""
        outDEMfile = os.path.join(self.outpath, 'DEM-NR.tif')
        if os.path.exists(outDEMfile) and self.overwrite is False:
            print('   {0} already processed'.format(input_id))
            return

        #--------------------------#
        ######## LOAD DATA #########
        #--------------------------#
        input_shp = os.path.join(self.outpath,
                                 f'{base_unit}_{input_id}.shp')
        input_gdf = gpd.read_file(input_shp)

        """ create buffer"""
        if self.buffercell:
            self.buffer_shp = os.path.join(self.outpath,
                                           f'{base_unit}_{input_id}_buffer.shp')
            pyct.buffer_gdf2shp(input_gdf, self.buffer, self.buffer_shp,
                                joinstyle=2)
        else:
            self.buffer_shp = input_shp

        with open(self.input_paths) as f:
            dem_path_list = f.read().splitlines()

        clip_pixels = (self.fsizes[0] // 2) + 1
        dem_tif = os.path.join(self.outpath, 'DEM.tif')
        self.basename = os.path.splitext(os.path.basename(dem_tif))[0]

        src_dict = self.crop_w_buffer(input_id, dem_tif, 
                                      dem_path_list,
                                      clip_pixels, nodataval,
                                      outformat="VRT")
                     
        """Re-build buffer shape"""
        buffer_gdf = pyct.build_bbox_gdf(dem_tif)
        buffer_gdf.to_file(self.buffer_shp)
        
        # if tile is along meridian, import fabdem to replace artifact
        fab_array = None
        if self.fabdem_mosaic is not None:
            fab_tif = os.path.join(self.outpath, 'FABDEM.tif')
            if not os.path.exists(fab_tif):
                pyct.create_cutline_raster(self.fabdem_mosaic,
                                           self.buffer_shp,
                                           fab_tif, srcDS=dem_tif,
                                           usecutline=False,
                                           nodata=nodataval,
                                           outformat="VRT",
                                           rsAlg=gdal.GRIORA_Bilinear)
        
        if os.path.exists(fab_tif):
            fab_array = gdal_array.LoadFile(fab_tif)
                                           

        """Load EDM"""
        input_tif = tilesearch(input_id, dem_path_list)
        EDM_tif = input_tif.replace('/DEM/', '/AUXFILES/')
        EDM_tif = EDM_tif.replace('_DEM.tif', '_EDM.tif')

        # load EDM array, if TDF, map to TDT EDM values for consistency
        edm_array = gdal_array.LoadFile(EDM_tif)
        if edm_array.max() > 5:
            edm_array_map = np.where(
                edm_array == 2, 5, 0)  # fill values
            edm_array_map = np.where(
                edm_array == 8, 4, edm_array_map)  # ocean
            edm_array_map = np.where(
                edm_array == 9, 2, edm_array_map)  # lake
            edm_array_map = np.where(
                edm_array == 10, 3, edm_array_map)  # river
        else:
            edm_array_map = edm_array

        edm_tif = os.path.join(self.outpath, "EDM.tif")
        pyct.npy2tif(edm_array_map, input_tif,
                     edm_tif, nodata=0, dtype=gdal.GDT_Byte)

        """load data from tifs"""
        dem_array = gdal_array.LoadFile(dem_tif).astype(np.float32)
        edm_array_temp = gdal_array.LoadFile(edm_tif).astype(np.int8)
        self.edm_array = np.zeros_like(dem_array, dtype=np.int8)
        self.edm_array[clip_pixels:-clip_pixels,
                  clip_pixels:-clip_pixels] = edm_array_temp
                  
        """load geotransform data from DEM tif"""
        geoT = pyct.get_geoT(dem_tif)

        """get grid size/shape/step"""
        self.nx, self.ny = dem_array.shape[1], dem_array.shape[0]
        self.dlon, self.dlat = np.abs(geoT[1]), np.abs(geoT[5])
        latmax = geoT[3] - (0.5 * self.dlat)
        latmin = latmax - self.dlat * (self.ny - 1)
        mid_lat = latmin + 0.5 * (latmax - latmin)
        self.dx = self.dlon * (111 * 1e3) * np.cos(mid_lat * (np.pi / 180))
        self.dy = self.dlat * (111 * 1e3)

        artifact_mask = self.correct_tdt_artifact(
                            dem_array, input_id, 
                            src_dict, clip_pixels)
                            
        if np.count_nonzero(artifact_mask) > 0:
            artifact_tif = os.path.join(self.outpath, 
                                        f"{self.basename}_artifact_mask.tif")
            pyct.npy2tif(artifact_mask, dem_tif,
                         artifact_tif, nodata=0, dtype=gdal.GDT_Byte)
            dem_array[artifact_mask==1] = fab_array[artifact_mask==1] 
                              
        
        if abs(latmin) < 79.5:  # above 80 degrees, don't remove veg bias
            """Surface Water Occurence"""
            swo_tif = os.path.join(self.outpath, f"{self.basename}_SWO.tif")
            if self.SWO_vrt is not None:
                if not os.path.exists(swo_tif):
                    pyct.create_cutline_raster(self.SWO_vrt, self.buffer_shp,
                                               swo_tif, srcDS=dem_tif,
                                               nodata=255,
                                               usecutline=False,
                                               outdtype=gdal.GDT_Byte,
                                               rsAlg=gdal.GRIORA_Bilinear,
                                               paletted=True)

                SWO_array = gdal_array.LoadFile(swo_tif).astype(np.int8)

            else:
                SWO_array = np.empty((self.ny, self.nx))
        else:
            SWO_array = np.empty((self.ny, self.nx))

        if self.veg_remove:
            temp_format = "VRT"
            """Treecover % """
            if self.treecover_vrt is not None:
                treecover_tif = os.path.join(self.temp_data_path,
                                             f"{input_id}_treecover.tif")
                pyct.resample(self.treecover_vrt, treecover_tif,
                              outformat=temp_format,
                              srcDS=dem_tif,
                              rsAlg=gdal.GRIORA_Bilinear,
                              dtype=gdal.GDT_Byte)

            """Tree Height """
            if self.treeheight_vrt is not None:
                treeheight_tif = os.path.join(self.temp_data_path,
                                              f"{input_id}_treeheight.tif")
                pyct.resample(self.treeheight_vrt, treeheight_tif,
                              outformat=temp_format,
                              srcDS=dem_tif,
                              rsAlg=gdal.GRIORA_Bilinear,
                              dtype=gdal.GDT_Byte)

        else:
            SWO_array = np.zeros((self.nyb, self.nxb))
            self.veg_remove = False

        """Pull out watermask indices"""
        watermaskSWO = np.where(SWO_array > self.SWOlevel)
        watermaskTX = np.where((self.edm_array == 2) | (self.edm_array == 3))
        self.ocean_mask = np.where(self.edm_array == 4, 1, 0)

        #---------------------------------------#
        ######## REMOVE VEGETATION BIAS #########
        #---------------------------------------#
        if self.veg_remove:

            if self.veg_model == '3Dtable':

                bias_array,\
                    slope_veg_array,\
                    max_treeheight = self.compute_veg_bias_3D(
                        treecover_tif,
                        treeheight_tif,
                        input_id, dem_array)
            else:
                bias_array,\
                    slope_veg_array = self.compute_veg_bias_2D(treecover_tif,
                                                               input_id,
                                                               dem_array)

            """Apply no bias correction over watermask"""
            bias_array[watermaskTX] = 0.0

            """Apply correction to raw input DEM"""
            dem_array = dem_array - bias_array

            if len(self.save_veg_files) > 0:

                if self.save_veg_files == 'all' or 'bias':
                    bias_array = 1e2 * bias_array  # convert to cm integers

                    outfile = os.path.join(self.outpath, 'veg_bias.tif')
                    pyct.npy2tif(bias_array, dem_tif, outfile,
                                 dtype=gdal.GDT_Float32)
                    pyct.tifcompress(outfile)

                if self.save_veg_files == 'all':
                    outfile = os.path.join(self.outpath, 'slope_veg.tif')
                    pyct.npy2tif(slope_veg_array, dem_tif, outfile,
                                 dtype=gdal.GDT_Byte)

                    outfile = os.path.join(self.outpath, 'treecover.tif')
                    pyct.npy2tif(self.TC_array, dem_tif, outfile,
                                 dtype=gdal.GDT_Byte)

                    if self.veg_model == '3Dtable':
                        Theight_array = gdal_array.LoadFile(treeheight_tif)
                        # Theight_array[Theight_array > max_treeheight] = max_treeheight
                        Theight_array = Theight_array.astype(np.int)
                        outfile = os.path.join(self.outpath, 'treeheight.tif')
                        pyct.npy2tif(Theight_array, dem_tif,
                                     outfile, dtype=gdal.GDT_Byte)

        #---------------------------#
        ######## SMOOTH DEM #########
        #---------------------------#

        """Smooth DEM with uniform kernels"""
        Zs_list = []
        for weights in self.fweights:
            kernel = bop.build_kernel(
                weights, dtype=np.float32, normalize=True)
            Zs_list.append(pdt.apply_convolve_filter(dem_array, kernel))

        """Compute initial slope/curvature"""
        if self.gdir == 2:
            slope_0 = pdt.slope_D2(Zs_list[self.s0ind], self.dx, self.dy)
        elif self.gdir == 4:
            slope_0 = pdt.slope_D4(Zs_list[self.s0ind], self.dx, self.dy)

        curvature = pdt.curvature_D2(Zs_list[self.s0ind], self.dx, self.dy)

        """Find peaks/valleys"""
        std_geoC = np.std(curvature)
        cbreak = [c * std_geoC for c in self.curvebreaks]
        peaks = np.where((curvature < -cbreak[1]))
        peakfill = np.where((curvature >= -cbreak[1]) &
                            (curvature < -cbreak[0]))
        valleys = np.where((curvature > cbreak[1]))
        valleyfill = np.where((curvature <= cbreak[1]) &
                              (curvature > cbreak[0]))

        """Boundaries --> no filter applied"""
        row_steep, col_steep = np.where(slope_0 >= self.slopebreaks[-1])
        bound_rows = list(valleys[0]) + list(peaks[0]) + list(row_steep)
        bound_cols = list(valleys[1]) + list(peaks[1]) + list(col_steep)

        """ Pixels for 3x3 filter"""
        fill_rows = list(valleyfill[0]) + list(peakfill[0])
        fill_cols = list(valleyfill[1]) + list(peakfill[1])

        """Initialize smoothed elevation array with largest filter data"""
        Zf = Zs_list[0].copy()
        filter_array = 9 * \
            np.ones((Zf.shape[0], Zf.shape[1])).astype(np.int8)

        """Loop through filter sizes and apply pixels to final elevation"""
        for i, Zs_uniform in enumerate(Zs_list[1::]):

            filtersize = np.int(2 * len(self.fweights[i + 1]) - 1)
            min_slope = self.slopebreaks[i + 1]
            max_slope = self.slopebreaks[i + 2]
            rows, cols = np.where((slope_0 >= min_slope) &
                                  (slope_0 <= max_slope))
            if filtersize == 3:
                """add bound fill to rows for 3x3 filter"""
                rows = list(rows) + fill_rows
                cols = list(cols) + fill_cols

            """assign filter size and elevation to correct indices"""
            filter_array[rows, cols] = filtersize
            Zf[rows, cols] = Zs_uniform[rows, cols]

        """Apply boundaries to correct indices"""
        filter_array[row_steep, col_steep] = 1
        filter_array[peaks] = 2
        filter_array[valleys] = 0
        Zf[bound_rows, bound_cols] = dem_array[bound_rows, bound_cols]

        """Apply 3x3 filter to SWO watermask"""
        filter_array[watermaskSWO] = 3
        Zf[watermaskSWO] = Zs_list[-1][watermaskSWO]

        """Apply 3x3 filter to DEM fill pixels (unless larger filter was applied)"""
        DEMfill = np.where((self.edm_array == 5) & (filter_array < 3))
        filter_array[DEMfill] = 3
        Zf[DEMfill] = Zs_list[-1][DEMfill]

        """Apply no filter to TanDEM-X watermask, and artifact fill"""
        filter_array[watermaskTX] = 1
        Zf[watermaskTX] = dem_array[watermaskTX]
        Zf[artifact_mask==1] = dem_array[artifact_mask==1]     

        """Compute final slope/curvature"""
        if self.gdir == 2:
            slope_array = pdt.slope_D2(Zf, self.dx, self.dy)
        elif self.gdir == 4:
            slope_array = pdt.slope_D4(Zf, self.dx, self.dy)

        # curvature_array = pdt.curvature_D2(Zf, self.dx, self.dy)

        """Clip buffer"""
        Zf = self.clipbuffer_arr(Zf, clip_pixels, set_nodata=nodataval)
        filter_array = self.clipbuffer_arr(filter_array, clip_pixels,
                                           set_nodata=99)
        slope_array = self.clipbuffer_arr(slope_array, clip_pixels,
                                          set_nodata=99)
        # curvature_array = self.clipbuffer_arr(curvature_array, self.buffer)
        # curvature_array *= 1e3

        #----------------------------#
        ######## SAVE OUTPUTS ########
        #----------------------------#

        """list of layers to output"""
        save_list = [Zf, filter_array, slope_array]
        output_dtypes = [np.float32, np.uint8, np.uint8]
        output_nodata = [-9999, 99, 99]
        output_names = ['DEM-NR', 'Filtermask', 'Slope']

        for array, name, arr_dtype, nodata in zip(
                save_list, output_names, output_dtypes, output_nodata):

            """Set data type in GDAL based on numpy dtype"""
            if arr_dtype == np.uint8:
                out_dtype = gdal.GDT_Byte
            elif arr_dtype == np.float32:
                out_dtype = gdal.GDT_Float32

            outfile = os.path.join(
                self.outpath, '{0}.tif'.format(name))
            pyct.npy2tif(array, dem_tif, outfile,
                         dtype=out_dtype, nodata=nodata)
            pyct.tifcompress(outfile)

        if self.pull_crosssections:
            """Save cross sections as arrays"""
            crossX_idx = 4500
            crossY_idx = np.int(self.nx / 2)

            DEM_crossX = dem_array[crossX_idx, :]
            DEM_crossY = dem_array[:, crossY_idx]
            DEM_smooth_crossX = save_list[0][crossX_idx, :]
            DEM_smooth_crossY = save_list[0][:, crossY_idx]
            filter_crossX = save_list[2][crossX_idx, :]
            filter_crossY = save_list[2][:, crossY_idx]

            np.save(os.path.join(self.outpath, 'crossX.npy'), DEM_crossX)
            np.save(os.path.join(self.outpath, 'crossY.npy'), DEM_crossY)
            np.save(os.path.join(self.outpath,
                    'crossX_smooth.npy'), DEM_smooth_crossX)
            np.save(os.path.join(self.outpath,
                    'crossY_smooth.npy'), DEM_smooth_crossY)
            np.save(os.path.join(self.outpath,
                    'filter_crossX.npy'), filter_crossX)
            np.save(os.path.join(self.outpath,
                    'filter_crossY.npy'), filter_crossY)

        #----------------------------#
        ###### CLEAR TEMP FILES ######
        #----------------------------#
        ft.erase_files(self.outpath, search=f'{self.basename}_*')
        temp_files = glob.glob(os.path.join(
            self.temp_data_path, "{0}_*.tif".format(input_id)))
        for tf in temp_files:
            os.remove(tf)

        return

    ##############################################################
    #################     FUNCTIONS    ###########################
    ##############################################################

    def printMinMax(self, var):
        print("{0} min/max: {1}/{2}".format("Z", var.min(), var.max()))

    def compute_veg_bias_2D(self, treecover_tif, input_id, Zarr):
        """Compute vegetation bias for an input DEM with a given bias model and Treecover
        input

        Parameters
        ----------
        treecover_tif : path
            treecover input tif clipped to DEM area
        input_id : str
            name of TanDEM-x cell
        Zarr : array
            2D array of DEM data

        Returns
        -------
        array
            2D array of vegetetion bias model (meters)
        """

        poly_coeff_file = os.path.join(
            self.model_file_path, 'VegModel_coeff.h5')
        lookup_table_file = os.path.join(
            self.model_file_path, 'Veg_lookup_tables.h5')

        TC_array = gdal_array.LoadFile(treecover_tif)

        TC_array[TC_array > 100] = 0
        TC_array = TC_array.astype(np.int)

        # Slope of smooth data
        weights = [1., 0.8, 0.6, 0.4, 0.2]  # 9x9 smoothing filter
        kernel = bop.build_kernel(weights, dtype=np.float32, normalize=True)
        Zarr_smooth9 = pdt.apply_convolve_filter(Zarr, kernel)
        slope_veg_array = pdt.slope_D2(Zarr_smooth9, self.dx, self.dy,
                                       unit='degree')

        """Apply vegetation bias correction"""
        bias_array = np.zeros((self.nyb, self.nxb))

        if self.model_file_path is None:
            print('Missing vegetation model files, skipping vegetation removal')
            return

        if self.veg_model == 'poly':

            coeff_df = pd.read_hdf(poly_coeff_file)

            model_coeff = [0.03789534968886686, 0.1395174341829395,
                           -0.019562663304545865, -0.005218094300390786,
                           0.0015878676683811661, 0.0050266251217852215]
            bias_array = pvt.veg_bias_polynomial(TC_array, slope_veg_array,
                                                 mod_coeff=model_coeff)

        elif self.veg_model == '2Dtable':

            df_tables = pd.read_hdf(lookup_table_file)
            npts = self.nxb * self.nyb
            TC_vec = TC_array.reshape(npts,)
            slope_veg_vec = slope_veg_array.reshape(npts,)
            # set to range of lookup table
            slope_veg_vec[slope_veg_vec > 50] = 50

            df_LC = df_tables[df_tables.LC_type == 999]
            slope_table_vec = df_LC.slope_smooth.values.astype(np.int)
            TC_table_vec = df_LC.treecover.values.astype(np.int)
            data = df_LC.pivot(index='slope_smooth',
                               columns='treecover', values='vegbias')
            bias_table = data.values

            """use slope and treecover values as 2D array indices"""
            bias_vec = [bias_table[S, TC]
                        for S, TC in zip(slope_veg_vec, TC_vec)]
            bias_vec = np.asarray(bias_vec)
            bias_array = bias_vec.reshape(self.nyb, self.nxb)

            return bias_array, slope_veg_array

    def compute_veg_bias_3D(self, treecover_tif, treeheight_tif, input_id, Zarr):
        """
        Compute 3d vegetation bias for an input DEM with a given bias model and Treecover
        input

        Parameters
        ----------
        treecover_tif : path
            treecover input tif clipped to DEM area
        treeheight_tif : path
            treeheight input tif clipped to DEM area
        input_id : str
            name of TanDEM-x cell
        Zarr : array
            2D array of DEM data

        Returns
        -------
        array
            2D array of vegetetion bias model (meters)
        """

        #lookup_table_file = os.path.join(self.model_file_path, 'veg_bias_3d.json')
        lookup_table_file = f"{self.model_file_path}/veg_bias_3d.json"
        if self.model_file_path is None:
            print('Missing vegetation model files, skipping vegetation removal')
            return

        # Read data from file:
        data = json.load(open(lookup_table_file))
        sample_arr = np.array(data.get(list(data.keys())[0]))
        slope_axis = np.array(list(data.keys())).astype(int)
        max_treeheight = sample_arr.shape[1] - 1
        max_treecover = sample_arr.shape[0] - 1

        bias_cube = np.empty((sample_arr.shape[0],
                              sample_arr.shape[1],
                              len(list(data.keys()))))

        """re build array """
        for s in slope_axis:
            bias_cube[:, :, s] = np.array(data.get(str(int(s))))

        # set 0 treecover and tree height to 0 bias
        bias_cube[0, :, :] = 0.0  # tree cover
        bias_cube[:, 0, :] = 0.0  # tree height

        """smooth bias lookup table along slope"""
        weights = np.array([.25, .5, 1.0, .5, .25])
        weights = (1 / np.sum(weights)) * weights
        bias_cube = scipy.ndimage.convolve1d(bias_cube, weights, axis=2)

        TC_array = gdal_array.LoadFile(treecover_tif)
        # fill ocean will 0 TC value
        TC_array[self.ocean_mask == 1] = 0     
        """check for nondata values"""
        TC_nodata = np.nonzero(TC_array > 100)
        interp_filter = 201

        if len(TC_nodata[0]) > 10:  # allow for a few nodata values
            #print_time("   fill treecover nodata values ({1}) - {0}".format(
            #    input_id, len(TC_nodata[0])), self.starttime)

            """Load 2010 dataset"""
            if self.treecover_fill_vrt is not None:

                treecover_fill_tif = os.path.join(
                    self.temp_data_path, f"{input_id}_treecover_fill.tif")
                pyct.resample(self.treecover_fill_vrt, treecover_fill_tif,
                              outformat="VRT", srcDS=treecover_tif,
                              rsAlg=gdal.GRIORA_Bilinear, dtype=gdal.GDT_Byte)

                TC_fill_array = gdal_array.LoadFile(treecover_fill_tif)
                TC_array[TC_nodata] = TC_fill_array[TC_nodata]
                TC_nodata = np.nonzero(TC_array > 100)

                if len(TC_nodata[0]) > 10:
                    """Load 2005 dataset"""
                    if self.treecover_fill2_vrt is not None:

                        treecover_fill2_tif = os.path.join(
                            self.temp_data_path,
                            f"{input_id}_treecover_fill2.tif")
                        pyct.resample(self.treecover_fill2_vrt,
                                      treecover_fill2_tif, outformat="VRT",
                                      srcDS=treecover_tif,
                                      rsAlg=gdal.GRIORA_Bilinear,
                                      dtype=gdal.GDT_Byte)

                        TC_fill2_array = gdal_array.LoadFile(
                            treecover_fill2_tif)
                        TC_array[TC_nodata] = TC_fill2_array[TC_nodata]
                        TC_nodata = np.nonzero(TC_array > 100)

            if len(TC_nodata[0]) > 10:
                #print_time("   interpolate remaining treecover nodata values" +
                #           f"({len(TC_nodata[0])}) - {input_id}",
                #           self.starttime)

                TC_tif0 = os.path.join(
                    self.temp_data_path, f"{input_id}_treecover_full.tif")
                pyct.npy2tif(TC_array, treecover_tif, TC_tif0,
                             nodata=220, dtype=gdal.GDT_Byte)

                TC_tif2 = os.path.join(
                    self.temp_data_path, f"{input_id}_treecover_interp.tif")
                self.wbt.fill_missing_data(
                    TC_tif0, TC_tif2, filter=interp_filter, weight=2)

                TC_array = gdal_array.LoadFile(TC_tif2)
                TC_nodata = np.nonzero(TC_array > 100)

                # set remainder to mean treecover of surounding pixels
                if len(TC_nodata[0]) > 10:

                    #print_time("   fill LEFTOVER treecover nodata values" +
                    #           f"({len(TC_nodata[0])}) - {input_id}",
                    #           self.starttime)
                    nodata_mask = np.zeros_like(TC_array)
                    nodata_mask[TC_nodata] = 9
                    k_size = 21
                    kernel = np.ones((k_size, k_size))
                    nodata_buffer_mask = pdt.apply_convolve_filter(
                        nodata_mask, kernel)
                    nodata_buffer_ind = np.nonzero(nodata_buffer_mask)
                    nodata_buffer_only_ind = np.where(
                        (nodata_buffer_mask > 0) & (TC_array < 100))
                    avg_buffer_value = np.nanmean(
                        TC_array[nodata_buffer_only_ind])
                    TC_array[nodata_buffer_ind] = avg_buffer_value

        """Smooth treecover/tree height values"""
        k_s = 3
        TC_array = pdt.apply_convolve_filter(TC_array, np.ones((k_s, k_s)),
                                             normalize_kernel=True)

        # filled treecover data
        self.TC_array = TC_array
        TC_array[TC_array > 100] = 0
        TC_array = TC_array.astype(np.int)

        # tree height data
        Theight_array = gdal_array.LoadFile(treeheight_tif)
        Theight_array[Theight_array > max_treeheight] = max_treeheight
        #Theight_array[Theight_array == 0] = 1
        Theight_array = pdt.apply_convolve_filter(Theight_array,
                                                  np.ones((k_s, k_s)),
                                                  normalize_kernel=True)
        Theight_array = Theight_array.astype(np.int)

        # Slope of smooth data
        weights = [1., 0.8, 0.6, 0.4, 0.2]  # 9x9 smoothing filter
        kernel = bop.build_kernel(weights, dtype=np.float32, normalize=True)
        Zarr_smooth9 = pdt.apply_convolve_filter(Zarr, kernel)
        slope_veg_array = pdt.slope_D2(
            Zarr_smooth9, self.dx, self.dy, unit='degree')
        # set to range of lookup table
        slope_veg_array[slope_veg_array > 19] = 19
        # set to range of lookup table
        #slope_veg_array[slope_veg_array == 0] = 1
        slope_veg_array = slope_veg_array.astype(np.int)

        # reshape arrays
        npts = TC_array.size
        TC_vec = TC_array.reshape(npts,)
        Theight_vec = Theight_array.reshape(npts,)
        slope_veg_vec = slope_veg_array.reshape(npts,)

        # use treecover, tree height and slope values as [i,j,k] indices
        bias_vec = [bias_cube[TC, TH, S]
                    for TC, TH, S in zip(TC_vec, Theight_vec, slope_veg_vec)]
        bias_vec = np.asarray(bias_vec)
        bias_array = bias_vec.reshape(TC_array.shape)

        return bias_array, slope_veg_array, max_treeheight


    def correct_tdt_artifact(self, dem_array, input_id, src_dict, 
                             clip_pixels):
                  
        artifact_mask = np.zeros_like(dem_array, dtype=np.int8)
        tdt_cond = src_dict[input_id] == "TDT"
        meridian_w_cond = "W001" in input_id
        meridian_e_cond = "E000" in input_id
        a_width = int(3200/self.dx)
        #print("Artifact width = ", a_width)
        
        tile_NS = int(input_id[1:3])
    
        if meridian_w_cond:
            print(f"     {input_id}: Meridian W tile - check for artifact") 
            if tdt_cond:
                #print("Center tile is TDT")
                artifact_mask[clip_pixels:-clip_pixels,
                              -(a_width+clip_pixels):-clip_pixels] = 1
                              
            # check each neighbor along meridian
            if src_dict[f"N{tile_NS+1}W001"] == "TDT": # North
                #print("north nb is TDT")
                artifact_mask[0:clip_pixels,
                              -(a_width+clip_pixels):-clip_pixels] = 1
                              
            if src_dict[f"N{tile_NS-1}W001"] == "TDT": # South
                #print("south nb is TDT")
                artifact_mask[-clip_pixels::,
                              -(a_width+clip_pixels):-clip_pixels] = 1
                              
            if src_dict[f"N{tile_NS}E000"] == "TDT": # East 
                #print("east nb is TDT")
                artifact_mask[clip_pixels:-clip_pixels,
                              -clip_pixels::] = 1
                              
            if src_dict[f"N{tile_NS+1}E000"] == "TDT": # NorthEast
                #print("NE nb is TDT")
                artifact_mask[0:clip_pixels,
                              -clip_pixels::] = 1
                              
            if src_dict[f"N{tile_NS-1}E000"] == "TDT": # SouthEast
                #print("SE nb is TDT")
                artifact_mask[-clip_pixels::,
                              -clip_pixels::] = 1
                          
        if meridian_e_cond and tdt_cond:
            print(f"    {input_id}: Meridian E tile - check for artifact") 
            if tdt_cond:
                #print("Center tile is TDT")            
                artifact_mask[clip_pixels:-clip_pixels,
                              clip_pixels:clip_pixels+5] = 1
                              
             # check each neighbor along meridian
            if src_dict[f"N{tile_NS+1}E000"] == "TDT": # North
                #print("north nb is TDT")
                artifact_mask[0:clip_pixels,
                              clip_pixels:clip_pixels+5] = 1
                              
            if src_dict[f"N{tile_NS-1}E000"] == "TDT": # South
                #print("south nb is TDT")
                artifact_mask[-clip_pixels::,
                              clip_pixels:clip_pixels+5] = 1
                              
            if src_dict[f"N{tile_NS}W001"] == "TDT": # West 
                #print("west nb is TDT")
                artifact_mask[clip_pixels:-clip_pixels,
                              0:clip_pixels] = 1
                              
            if src_dict[f"N{tile_NS+1}W001"] == "TDT": # NorthWest
                #print("NW nb is TDT")
                artifact_mask[0:clip_pixels,
                              0:clip_pixels] = 1
                              
            if src_dict[f"N{tile_NS-1}E000"] == "TDT": # SouthWest
                #print("SW nb is TDT")
                artifact_mask[-clip_pixels::,
                              0:clip_pixels] = 1
        
        return artifact_mask
        

    def crop_w_buffer(self, input_id, output_tif, path_list,
                      buffer, nodataval, outformat="VRT"):
        """
        Load in  neighboring cells and create buffered cell for given dataset

        Inputs:

            input_id : name of TDT tile to be buffered [str]
            output_tif : path of ouput buffered tif
            buffer : size of buffer (in pixels)

        Outputs:

            datab : 2D buffered array of input data


        """

        input_tif = tilesearch(input_id, path_list)
        input_array = gdal_array.LoadFile(input_tif)
        
        # build dictionary of src dem for tile and neighbor tiles
        src_dem_dct = {}
        
        if "TDT" in input_tif:
            self.src_dem = "TDT"
            src_dem_dct[input_id] = "TDT"
        elif "TDF" in input_tif:
            self.src_dem = "TDF"
            src_dem_dct[input_id] = "TDF"
        else:
            self.src_dem = None
            
        """load geotransform data from DEM tif"""
        geoT = pyct.get_geoT(input_tif)

        """get grid size/shape/step"""
        nx, ny = input_array.shape[1], input_array.shape[0]
        dlon, dlat = np.abs(geoT[1]), np.abs(geoT[5])

        nbb = buffer
        nxb, nyb = np.int(nx + 2 * nbb), np.int(ny + 2 * nbb)

        input_buffer_gdf = gpd.read_file(self.buffer_shp)
        input_poly = list(input_buffer_gdf.geometry)[0]
        tiles_gdf = gpd.read_file(self.input_shps)

        nb_tiles_gdf = tiles_gdf.loc[
            tiles_gdf.geometry.intersects(input_poly)]
        nb_tiles = list(nb_tiles_gdf.TILE_ID)
        nb_paths = [tilesearch(i, path_list) for i in nb_tiles]
        
        # build dictionary of src DEMS
        for nb_tile, nb_path in zip(nb_tiles, nb_paths):
            if "TDF" in nb_path:
                src_dem_dct[nb_tile] = "TDF" 
            elif "TDT" in nb_path:
                src_dem_dct[nb_tile] = "TDT"
            else:
                src_dem_dct[nb_tile] = ""
            
        
        if self.src_dem == "TDF":
            TDT_nb_paths = [p for p in nb_paths if "TDT" in p]
            if len(TDT_nb_paths) > 0:
                TDF_nb_paths = [p for p in nb_paths if "TDF" in p]
                # print(f"   {len(TDT_nb_paths)} TDT neighbors to resample")
                for nb in TDT_nb_paths:
                    fname = os.path.basename(nb)
                    temp_nb = os.path.join(self.outpath, f"{self.basename}_{fname}.tif")
                    pyct.resample(nb, temp_nb, outformat="VRT",
                                  srcDS=nb, dtype=gdal.GDT_Float32)
                    TDF_nb_paths.append(temp_nb)
                nb_paths = TDF_nb_paths
            
        elif self.src_dem == "TDT":
            TDF_nb_paths = [p for p in nb_paths if "TDF" in p]
            if len(TDF_nb_paths) > 0:
                TDT_nb_paths = [p for p in nb_paths if "TDT" in p]
                # print(f"   {len(TDF_nb_paths)} TDF neighbors to resample")
                for nb in TDF_nb_paths:
                    fname = os.path.basename(nb)
                    temp_nb = os.path.join(self.outpath, f"{self.basename}_{fname}.tif")
                    pyct.resample(nb, temp_nb, outformat="VRT",
                                  srcDS=nb, dtype=gdal.GDT_Int16)
                    TDT_nb_paths.append(temp_nb)
                nb_paths = TDT_nb_paths
                        
        vrtfile0 = output_tif.replace('.tif', '_neighbors.tif')

        if os.path.exists(output_tif):
            os.remove(output_tif)

        """Create temp vrt for all 9 cells"""
        pyct.build_vrt(nb_paths, vrtfile0, resolution='user',
                       xres=dlon, yres=dlat, resampleAlg=gdal.GRIORA_Bilinear)

        """Define bounding box from DEM cell geotransform"""
        xmin = geoT[0] - (buffer * dlon)
        xmax = xmin + dlon * (nx + 2 * buffer)
        ymax = geoT[3] + (buffer * dlat)
        ymin = ymax - dlat * (ny + 2 * buffer)
        bbox = [xmin, ymin, xmax, ymax]

        pyct.resample_bbox(vrtfile0,
                           output_tif,
                           bbox,
                           xshape=nxb,
                           yshape=nyb,
                           outformat=outformat,
                           dstnodata=nodataval,
                           srcDS=input_tif,
                           rsAlg=gdal.GRIORA_NearestNeighbour,
                           dtype=gdal.GDT_Float32)
                           
        return src_dem_dct
    

    def makebuffer(self, cell, data, dataname, buffer, nx, ny, geoT,
                   datatype=np.int8, useVRT=True, vrt_out=None):
        """
        Load in  neighboring cells and create buffered cell for given dataset

        Inputs:

            cell : name of TDT cell to be buffered [str]
            data : 2d array of data for cell being buffered
            dataname : Name of the dataset being buffered (DEM, SWO. etc)
            buffer : size of buffer (in pixels)
            nx : X dimension
            ny : Y dimension

        Outputs:

            datab : 2D buffered array of input data

        If a neighbor cell does not exist, the values are left as 0.0. This is
        probably the case along coasts.
        """

        nbb = buffer
        self.nxb, self.nyb = np.int(nx + 2 * nbb), np.int(ny + 2 * nbb)
        datab = np.zeros([self.nyb, self.nxb], dtype=datatype)

        if dataname == 'EDM':

            """Load in center/primary cell"""
            datab[nbb:-nbb, nbb:-nbb] = data[:, :]

            return datab

        if useVRT:

            neighbors = findNeighbors(cell)
            nb_paths_dct = get_nb_paths(neighbors, self.tilepaths_all)
            center_tile_path = nb_paths_dct[cell]

            if not os.path.exists(center_tile_path):
                raise(
                    'make buffer - center cell {0} does not exist'.format(center_tile_path))

            vrtfile0 = vrt_out.replace('.tif', '_neighbors.tif')

            if os.path.exists(vrt_out):
                os.remove(vrt_out)

            nb_paths = []
            for nb in neighbors:
                if dataname == 'DEM':
                    nbtile_path = nb_paths_dct[nb]
                else:
                    nbtile_path = os.path.join(os.path.dirname(
                        nb_paths_dct[nb]), '{0}.tif'.format(dataname))

                # if not os.path.exists(nbtile_path):
                    # print('make buffer - neighbor cell {0} for {1} does not exist'.format(
                    #    nbtile_path, center_tile_path))
                if os.path.exists(nbtile_path):
                    nb_paths.append(nbtile_path)

            """Create temp vrt for all 9 cells"""
            pyct.build_vrt(nb_paths, vrtfile0)

            """Define bounding box from DEM cell geotransform"""
            xmin = geoT[0] - (self.buffer * self.dlon)
            xmax = xmin + self.dlon * (self.nx + 2 * self.buffer)
            ymax = geoT[3] + (self.buffer * self.dlat)
            ymin = ymax - self.dlat * (self.ny + 2 * self.buffer)
            bbox = [xmin, ymin, xmax, ymax]

            pyct.resample_bbox(vrtfile0,
                               vrt_out,
                               bbox,
                               xshape=self.nxb,
                               yshape=self.nyb,
                               outformat="VRT",
                               srcDS=center_tile_path,
                               rsAlg=gdal.GRIORA_NearestNeighbour,
                               dtype=gdal.GDT_Float32)

            datab = gdal_array.LoadFile(vrt_out)

        else:

            neighbors = findNeighbors(cell)[1::]
            nb_paths_dct = get_nb_paths(neighbors,
                                        self.tilepaths_all)
            """Load in center/primary cell"""
            datab[nbb:-nbb, nbb:-nbb] = data[:, :]

            for i, nb in enumerate(neighbors):
                """
                Add buffer from neighbors to current cell

                neighbor list format:
                [name, N_tile, E_tile, S_tile, W_tile,
                 NE_tile, SE_tile, SW_tile, NW_tile]
                """

                if dataname == 'DEM':
                    nbtile_path = nb_paths_dct[nb]
                else:
                    nbtile_path = os.path.join(os.path.dirname(
                        nb_paths_dct[nb]), '{0}.tif'.format(dataname))

                if os.path.exists(nbtile_path):

                    if i == 0:  # north cell
                        nb_slice = gdal_array.LoadFile(
                            nbtile_path, xoff=0, yoff=ny - nbb - 1, xsize=None, ysize=nbb)
                        datab[0:nbb, nbb:-nbb] = nb_slice
                    elif i == 1:  # east cell
                        nb_slice = gdal_array.LoadFile(
                            nbtile_path, xoff=1, yoff=0, xsize=nbb, ysize=None)
                        datab[nbb:-nbb, -nbb::] = nb_slice
                    elif i == 2:  # south cell
                        nb_slice = gdal_array.LoadFile(
                            nbtile_path, xoff=0, yoff=1, xsize=None, ysize=nbb)
                        datab[-nbb::, nbb:-nbb] = nb_slice
                    elif i == 3:  # west cell
                        nb_slice = gdal_array.LoadFile(
                            nbtile_path, xoff=nx - nbb - 1, yoff=0, xsize=nbb, ysize=None)
                        datab[nbb:-nbb, 0:nbb] = nb_slice
                    elif i == 4:  # NE cell
                        nb_slice = gdal_array.LoadFile(
                            nbtile_path, xoff=1, yoff=ny - nbb - 1, xsize=nbb, ysize=nbb)
                        datab[0:nbb, -nbb::] = nb_slice
                    elif i == 5:  # SE cell
                        nb_slice = gdal_array.LoadFile(
                            nbtile_path, xoff=1, yoff=1, xsize=nbb, ysize=nbb)
                        datab[-nbb::, -nbb::] = nb_slice
                    elif i == 6:  # SW cell
                        nb_slice = gdal_array.LoadFile(
                            nbtile_path, xoff=nx - nbb - 1, yoff=1, xsize=nbb, ysize=nbb)
                        datab[-nbb::, 0:nbb] = nb_slice
                    elif i == 7:  # NW cell
                        nb_slice = gdal_array.LoadFile(
                            nbtile_path, xoff=nx - nbb - 1, yoff=ny - nbb - 1, xsize=nbb, ysize=nbb)
                        datab[0:nbb, 0:nbb] = nb_slice

                elif not os.path.exists(nbtile_path):
                    print(nbtile_path, 'does not exist')

        return datab


    def clipbuffer_arr(self, buffered_data, buffer, set_nodata=None):
        """
        Clip off the specified buffer for a 2D array

         Parameter
        ----------
        buffered_data : 2D array
            data to be clipped
        buffer : integer
            Size of buffer to be trimmed from all edges (pixels)

        Returns
        -------
        clipped_data : 2D numpy array of clipped data

        """

        b = buffer
        if set_nodata is not None:
            data_mask = np.zeros_like(buffered_data, dtype=np.int8)
            data_mask[b:-b, b:-b] = 1
            clipped_data = np.where(data_mask == 1, buffered_data, set_nodata)
        else:
            clipped_data = buffered_data[b:-b, b:-b]

        return clipped_data


    def load_LC_tile_array(self, LCfile, dem_tif, cellpath, geoT,
                           LCfile_coarse=None):

        LC_tile_tif = os.path.join(cellpath, "LC_buffered.tif")

        """Define bounding box from DEM cell geotransform"""
        xmin = geoT[0] - (self.buffer * self.dlon)
        xmax = xmin + self.dlon * (self.nx + 2 * self.buffer)
        ymax = geoT[3] + (self.buffer * self.dlat)
        ymin = ymax - self.dlat * (self.ny + 2 * self.buffer)
        LC_bbox = [xmin, ymin, xmax, ymax]

        if not os.path.exists(LC_tile_tif):
            """Resample if landcover tif for cell does not exist"""

            pyct.resample_bbox(LCfile,
                               LC_tile_tif,
                               LC_bbox,
                               xshape=self.nxb,
                               yshape=self.nyb,
                               outformat="VRT",
                               srcDS=dem_tif,
                               rsAlg=gdal.GRIORA_NearestNeighbour,
                               dtype=gdal.GDT_Byte)

        LC_array = gdal_array.LoadFile(LC_tile_tif)
        LC_voids = np.where(LC_array == 255)[0]

        """Use coarse dataset to fill voids, if applicable"""
        if LCfile_coarse is not None and len(LC_voids) > 0:
            LC_coarse_tile_tif = os.path.join(
                cellpath, "LC_coarse_buffered.tif")

            if not os.path.exists(LC_coarse_tile_tif):
                pyct.resample_bbox(LCfile_coarse,
                                   LC_coarse_tile_tif,
                                   LC_bbox,
                                   xshape=self.nxb,
                                   yshape=self.nyb,
                                   outformat="VRT",
                                   srcDS=dem_tif,
                                   rsAlg=gdal.GRIORA_NearestNeighbour,
                                   dtype=gdal.GDT_Byte)

            # Fill voids in 500m dataset with 0.05deg data
            LC_coarse_array = gdal_array.LoadFile(LC_coarse_tile_tif)
            LC_array[LC_voids] = LC_coarse_array[LC_voids]

        return LC_array


    def setfweights(self):
        """
        Set the weights and sizes of the smoothing filter.

        The weights are defined as a list - where the first value is
        the weight applied to the center cell (the cell being smoothed),
        followed by the subsequent weights appplied to the surrounding pixels.

        Example:

            For a filter with weights defined as:

                [1., 0.7, 0.35]

            The resulting 5x5 pixel moving-window smoothing operator is:

                -------------------------------
                | .35 | .35 | .35 | .35 | .35 |
                -------------------------------
                | .35 | .70 | .70 | .70 | .35 |
                -------------------------------
                | .35 | .70 | 1.0 | .70 | .35 |
                -------------------------------
                | .35 | .70 | .70 | .70 | .35 |
                -------------------------------
                | .35 | .35 | .35 | .35 | .35 |
                -------------------------------
        """

        if self.filtertype == "DW":
            """distance weighted"""
            weights_3 = [1., 0.5]
            weights_5 = [1., 0.7, 0.35]
            weights_7 = [1., 0.75, 0.5, 0.25]
            weights_9 = [1., 0.8, 0.6, 0.4, 0.2]

        elif self.filtertype == "mean":
            """mean filter"""
            weights_3 = [1., 1.]
            weights_5 = [1., 1., 1.]
            weights_7 = [1., 1., 1., 1.]
            weights_9 = [1., 1., 1., 1, 1.]

        fweights = [weights_9, weights_7, weights_5, weights_3]
        fsizes = [2 * len(f) - 1 for f in fweights]  # fsize_list

        return fweights, fsizes


    def printpercents(self, npoints, peaks, valleys):

        per_peaks = 1e2 * len(peaks) / npoints
        per_valleys = 1e2 * len(valleys) / npoints
        # per_water = 1e2*len(self.water_all)/npoints
        print("   peaks: {0:.2f}%".format(per_peaks), end="")
        print(" | valleys: {0:.2f}%".format(per_valleys))  # , end="")
        # print(" | water: {0:.2f}%".format(per_water))

    def exit(start):
        print("elapsed time: ", str(time.time() / 60. - start)[0:5], "minutes")
        sys.exit()


def run_tile_mpi(cell):
    """ Tiny wrapper function to make the mpi executor behave """
    cDEM.run_tile(cell)


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

    filtertype = config.get("parameters-noise-removal", "filtertype")
    vegmodel = config.get("parameters-noise-removal", "vegmodel")
    SWOlevel = config.getint("parameters-noise-removal", "SWOlevel")
    curvebreaks = json.loads(config.get(
        "parameters-noise-removal", "curvebreaks"))
    slopebreaks = json.loads(config.get(
        "parameters-noise-removal", "slopebreaks"))
    base_DEM = config.get("parameters-noise-removal", "base_DEM")
    overwrite = config.getboolean("processes-noise-removal", "overwrite")
    save_veg_files = config.get("parameters-noise-removal", "save_veg_files")
    veg_remove = config.getboolean("processes-noise-removal", "veg_remove")
    run_multiprocess = config.getboolean("job-options", "run_multiprocess")
    run_parallel = config.getboolean("job-options", "run_parallel")

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
    projectname = config.get("outputs", "projectname")
    base_DEM = config.get("parameters-noise-removal", "base_DEM")

    DEM_shpfile = os.path.join(config.get("paths", "DEM_shpfile"))

    """path to project text file"""
    project_tiles_txt = os.path.join(config.get("paths", "project_txt_path"),
                                     f'{projectname}.txt')
    input_ids = open(project_tiles_txt).read().splitlines()
    project_paths_txt = os.path.join(config.get("paths", "project_txt_path"),
                                     f'{projectname}_paths.txt')
    project_tile_shp = os.path.join(config.get('outputs',
                                               'tile_buffer_shapefile'))

    """Model file directory for vegetation removal"""
    model_file_path = os.path.join(config.get("paths", "model_file_path"))
    buffer = config.getfloat("parameters-depression-handling", "buffer") + 100

    """Full paths to ancillary data vrts"""
    SWO_vrt = os.path.join(config.get("paths", "SWO_vrt"))
    treecover_vrt = os.path.join(config.get("paths", "treecover_vrt"))
    treecover_fill_vrt = os.path.join(
        config.get("paths", "treecover_fill_vrt"))
    treecover_fill2_vrt = os.path.join(
        config.get("paths", "treecover_fill2_vrt"))
    treeheight_vrt = os.path.join(config.get("paths", "treeheight_vrt"))

    temp_data_path = os.path.join(config.get("paths", "temp_data_path"))
    output_DTM_path = os.path.join(config.get("outputs", "dtm_output_dir"))
    output_DTMmosaic = os.path.join(config.get("outputs", "dtm_mosaic"))
    fabdem_mosaic = os.path.join(config.get("outputs", "fabdem_mosaic"))


    WBT_path = os.path.join(config.get("paths", "WBT_path"))

    ###############################################
    """       Run smoothing algorithm            """
    ###############################################

    """Initialize class with input parameters"""
    cDEM = CreateDTM(input_ids, project_paths_txt, project_tile_shp,
                     filtertype, slopebreaks,
                     curvebreaks, SWOlevel,
                     WBT_path,
                     tile_buffer=buffer,
                     model_file_path=model_file_path,
                     SWO_vrt=SWO_vrt,
                     treecover_vrt=treecover_vrt,
                     treecover_fill_vrt=treecover_fill_vrt,
                     treecover_fill2_vrt=treecover_fill2_vrt,
                     treeheight_vrt=treeheight_vrt,
                     fabdem_mosaic=fabdem_mosaic,
                     temp_data_path=temp_data_path,
                     output_path=output_DTM_path,
                     base_DEM=base_DEM,
                     overwrite=overwrite,
                     veg_remove=veg_remove, veg_model=vegmodel,
                     save_veg_files=save_veg_files)


    """find unprocessed tiles"""
    base_unit = "tile"
    if not overwrite:
        input_ids = logt.find_unfinished_ids(output_DTM_path,
                                             unit=base_unit,
                                             id_list=input_ids,
                                             tifname='DEM-NR.tif')


    n_unfin = len(input_ids)

    """Run cell smoothing"""
    if run_parallel:

        with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
            if executor is not None:

                print_time('Processing {0} cells on {1} nodes'.format(
                    len(input_ids), numnodes), starttime, sep=True)

                executor.map(run_tile_mpi, input_ids)

    elif run_multiprocess:

        nthreads = min(n_unfin, 16)
        with Pool(nthreads) as p:
            print_time(f'Noise Removal for {n_unfin}' +
                       f' {base_unit}s on {nthreads} threads',
                       starttime, sep=True)
            p.map(cDEM.run_tile, input_ids)

    else:
        for tile in input_ids[0:1]:
            cDEM.run_tile(tile, mpi=False)
            print_time("Tile {0} COMPLETE".format(tile), starttime)

        log_str = log_time("NOISE REMOVAL COMPLETE", starttime)
        with open(log_file, 'w') as f:
            f.write(log_str)
        print(log_str)
