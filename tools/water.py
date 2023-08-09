#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 1  2021

@author: Kimberly McCormack, NGA



"""
import os
import sys
import subprocess
import time
import traceback
import warnings
warnings.simplefilter('ignore')

import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import gdal, gdal_array
from shapely.geometry import Polygon, box
from scipy import ndimage


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import tools.convert as pyct
import tools.shapes as shpt
import tools.files as ft
import tools.taudem as pytt
from tools.derive import apply_convolve_filter
from tools.print import print_time


def find_possible_reservoir(SWO_tif, res_mask_shp, wbt,
                            OSM_tif=None, SWO_limit=20, area_threshold=1.0,
                            fix_w_buffer=True, exclude_wetland=True,
                            h_diff=None, inDEM_array=None, water_mask=None):
    """
    Find continuous waterbodies above an area threshold from a raster
    input of surface water occurence and OpenStreetMap

    Parameters
    ----------
    SWO_tif : path
        input tif of surface water occurence value
    res_mask_shp : path
        output shape file of vectorized raster
    wbt : class alias
        White Box Tools alias
    OSM_tif : path, optional
        Catagorical Open Street Map tif. If given, will be used in tandem
        with SWO raster to determine water bodies, by default None
    SWO_threshold : int, optional
        threshold to delineate 'land' pixel (<threshold) from
        water pixel (> threshold), by default 20
    area_threshold : float, optional
        area in km^2, above which is considered large water body, by default 1.0
    fix_w_buffer : bool, optional
        simple method to fix invalid geometries (bowties). If False, a slower
        but more robust method is used, by default True
    exclude_wetland : bool, optional
        If an OSM raster is given, explicitly exclude wetland areas from
        being considered water bodies by default True

    Returns
    -------
    gdf
        geodataframe of large water body geometries
    array
        2D array mask of all water bodies - before area filtering
    """

    SWO_array = gdal_array.LoadFile(SWO_tif)
    outpath = os.path.dirname(res_mask_shp)
    res_mask_gdf = None

    """Build water mask with SWO values and OSM raster(if given)"""
    res_mask_array = np.zeros_like(SWO_array, dtype=np.int8)
    SWO_ind = np.nonzero(SWO_array > SWO_limit)
    res_mask_array[SWO_ind] = 1

    if OSM_tif is not None:
        riverclass_array = gdal_array.LoadFile(OSM_tif)
        river_poly_ind = np.nonzero((riverclass_array == 1) |
                                  (riverclass_array == 2) |
                                  (riverclass_array == 3) |
                                  (riverclass_array == 4))

        res_mask_array[river_poly_ind] = 1

        """exclude land pixels"""
        land_ind = np.nonzero(
            (riverclass_array == 0) & (SWO_array < SWO_limit))
        res_mask_array[land_ind] = 0

        if exclude_wetland:  # remove wetland pixels
            res_mask_array[riverclass_array == 5] = 0

    if water_mask is not None:
        res_mask_array[water_mask == 0] = 0
        res_mask_array[water_mask > 0] = 1

    if np.count_nonzero(res_mask_array) > 20:

        """Skip if total water elevation change is < 30"""
        if (h_diff is not None) and (inDEM_array is not None):
            water_elev = inDEM_array[res_mask_array == 1]
            min_water = np.percentile(water_elev, 1)
            max_water = np.percentile(water_elev, 99)
            water_diff = max_water - min_water
            if water_diff < h_diff:
                print(f'     Total water elevation change < {h_diff}',
                      f'({water_diff:.2f}) exiting reservoir barrier')
                return res_mask_gdf

        watermask_tif = os.path.join(outpath, "watermask_temp.tif")
        if os.path.exists(watermask_tif):
            os.remove(watermask_tif)
        pyct.npy2tif(res_mask_array, SWO_tif, watermask_tif,
                     nodata=0, dtype=gdal.GDT_Byte)

        wbt.raster_to_vector_polygons(watermask_tif, res_mask_shp)
        os.remove(watermask_tif)
        res_mask_gdf = gpd.read_file(res_mask_shp)
        res_mask_gdf['geometry'] = res_mask_gdf.geometry.set_crs(epsg=4326)
        res_mask_gdf = res_mask_gdf.explode()

        if not res_mask_gdf.empty:
            """fix invalid geometries"""
            res_mask_gdf, invalid_flag = shpt.fix_invalid_geom(res_mask_gdf)

            """Filter out small water bodies - use equal area projection"""
            res_mask_gdf['area'] = res_mask_gdf[
                'geometry'].to_crs('epsg:6933').map(lambda p: p.area / 10**6)
            res_mask_gdf = res_mask_gdf[res_mask_gdf['area']
                                          > area_threshold]

            res_mask_gdf = res_mask_gdf.reset_index(drop=True)


        if res_mask_gdf.empty:
            res_mask_gdf = None

    return res_mask_gdf


def find_possible_rivers(inDEM_array, nodataval, SWO_tif, watermask_shp, wbt,
                         OSM_tif=None, data_mask_tif=None, SWO_threshold=20,
                         area_threshold=1.0,
                         fix_w_buffer=True, river_buffer=None,
                         exclude_wetland=True, exclude_water=False,
                         keep_water_shp=False, coast_mask=None):
    """
    Find continuous waterbodies above an area threshold from a raster input of surface water
    occurence

    Parameters
    ----------
    SWO_tif : path
        input tif of surface water occurence value
    watermask_shp : path
        output shape file of vectorized raster
    wbt : class alias
        White Box Tools alias
    OSM_tif : path, optional
        Catagorical Open Street Map tif. If given, will be used in tandem
        with SWo raster to determine water bodies, by default None
    SWO_threshold : int, optional
        threshold to delineate 'land' pixel (<threshold) form
        water pixel (> threshold), by default 30
    area_threshold : float, optional
        area in km^2, above which is considered large water body, by default 1.0
    fix_w_buffer : bool, optional
        simple method to fix invalid geometries (bowties). If False, a slower
        but more robust method is used, by default True
    river_buffer : float, optional
        buffer in meters to add before filtering by area. A small buffer helps
        narrow, linear features (i.e. rivers) be retained, by default None
    exclude_wetland : bool, optional
        If an OSM raster is given, explicitly exclude wetland areas from
        being considered water bodies by default True

    Returns
    -------
    gdf
        geodataframe of large water body geometries
    array
        2D array mask of all water bodies - before area filtering
    """
    start = time.time()

    SWO_array = gdal_array.LoadFile(SWO_tif)
    outpath = os.path.dirname(watermask_shp)
    watermask_gdf = None

    """Build water mask with SWO values and OSM raster(if given)"""
    water_mask_array = np.zeros_like(SWO_array, dtype=np.int8)
    SWO_ind = np.nonzero(SWO_array > SWO_threshold)

    if data_mask_tif is not None:
        edm_array = gdal_array.LoadFile(data_mask_tif)
    else:
        edm_array = np.zeros_like(SWO_array, dtype=np.int8)
    edm_mask = np.where((edm_array == 2) | (edm_array == 3), 1, 0)

    if OSM_tif is not None:
        riverclass_array = gdal_array.LoadFile(OSM_tif)
        if exclude_water:
            river_poly_ind = np.nonzero((riverclass_array == 1) |
                                        (riverclass_array == 2) |
                                        (riverclass_array == 3))
        else:
            river_poly_ind = np.nonzero((riverclass_array == 1) |
                                      (riverclass_array == 2) |
                                      (riverclass_array == 3) |
                                      (riverclass_array == 4))

        water_mask_array[SWO_ind] = 1
        water_mask_array[river_poly_ind] = 1
        water_mask_array[edm_mask == 1] = 1

        """exclude land pixels"""
        land_ind = np.nonzero(((riverclass_array == 0) &
                              (SWO_array < SWO_threshold) &
                              (edm_mask == 0)))
        water_mask_array[land_ind] = 0

        if exclude_wetland:  # remove wetland pixels, leave permenent rivers
            water_mask_array[(riverclass_array == 5) &
                             (SWO_array < 90)] = 0

        if coast_mask is not None:  # ocean tile/basin
            water_mask_array[coast_mask == 0] = 0  # ocean pixels

    else:
        water_mask_array[SWO_ind] = 1
        land_ind = np.nonzero((SWO_array < SWO_threshold))
        if coast_mask is not None:  # ocean tile/basin
            water_mask_array[coast_mask == 0] = 0  # ocean pixels

    """Exclude DEM nodata values"""
    water_mask_array[inDEM_array == nodataval] = 0

    if np.count_nonzero(water_mask_array) >= 20:  # check if water pixels exist

        watermask_tif = os.path.join(outpath, "watermask_temp.tif")

        if keep_water_shp:
            if not os.path.exists(watermask_shp):
                pyct.npy2tif(water_mask_array, SWO_tif, watermask_tif,
                             nodata=0, dtype=gdal.GDT_Byte)
                wbt.raster_to_vector_polygons(watermask_tif, watermask_shp)

            else:
                blankmask_tif = os.path.join(
                    outpath, "DEM_blank_mask_water.tif")
                rastermask = np.ones_like(SWO_array, dtype=np.int8)
                pyct.npy2tif(rastermask, SWO_tif, blankmask_tif,
                             nodata=0, dtype=gdal.GDT_Byte)
                pyct.create_cutline_raster(blankmask_tif, watermask_shp,
                                           watermask_tif, srcDS=SWO_tif,
                                           nodata=0, usecutline=True,
                                           outdtype=gdal.GDT_Byte)
                water_mask_array = gdal_array.LoadFile(watermask_tif)

        else:
            pyct.npy2tif(water_mask_array, SWO_tif, watermask_tif,
                         nodata=0, dtype=gdal.GDT_Byte)
            wbt.raster_to_vector_polygons(watermask_tif, watermask_shp)

        watermask_gdf = gpd.read_file(watermask_shp)
        watermask_gdf['geometry'] = watermask_gdf.geometry.set_crs(epsg=4326)
        watermask_gdf = watermask_gdf.explode()

        if not watermask_gdf.empty:
            """fix invalid geometries"""
            watermask_gdf, invalid_flag = shpt.fix_invalid_geom(watermask_gdf)

            """Filter out small water bodies - use equal area projection"""
            watermask_buffer_gdf = watermask_gdf.copy()
            if river_buffer is not None:
                watermask_buffer_gdf = pyct.buffer_gdf_simple(
                    watermask_buffer_gdf, river_buffer)

            watermask_buffer_gdf['area'] = watermask_buffer_gdf['geometry']\
                .to_crs('epsg:6933')\
                .map(lambda p: p.area / 10**6)
            watermask_gdf = watermask_gdf[watermask_buffer_gdf['area']
                                          > area_threshold]

            if not watermask_gdf.empty:
                watermask_gdf = pyct.buffer_gdf_simple(watermask_gdf, 1.0)
                watermask_gdf['area'] = watermask_gdf['geometry']\
                    .to_crs('epsg:6933').map(lambda p: p.area / 10**6)

                watermask_gdf.to_file(watermask_shp)

        if watermask_gdf.empty:
            watermask_gdf = None

    return watermask_gdf, water_mask_array


def close_holes(poly: Polygon) -> Polygon:
    """
    Close polygon holes by limitation to the exterior ring.
    Args:
        poly: Input shapely Polygon
    Example:
        df.geometry.apply(lambda p: close_holes(p))
    """
    return Polygon(poly.exterior)


def find_large_lakes(OSM_src_path, OSM_tiles, outpath, clip_shp,
                     flatten_lakes_shp, area_threshold=1e3, lake_buffer=50):
    """
    Find very large lakes based on Open Street map data over entire project region.
    Designed to be used after the entire hydro-conditioning process is complete to flatten
    great lakes.

    Parameters
    ----------
    OSM_src_path : path
        directory to entire Open Street Map sahpe file dataset
    outpath : path
        output directory
    clip_shp : path
        shape file of project outline (such as a level 2/3 hydrobasin)
    flatten_lakes_shp : path
        output shape file for large lake polygons
    area_threshold : float, optional
        area threshold for lakes in km^2, by default 1,000 km^2
    lake_buffer : float, optional
        buffer to apply around lkaes (in meters), by default 100

    Returns
    -------
    gdf
        geodataframe of large lake polygons or None is empty
    """

    starttime = time.time()

    osm_bbox_gdf = gpd.read_file(OSM_tiles)

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    input_gdf = gpd.read_file(clip_shp)
    input_poly = list(input_gdf.geometry)[0]
    input_box_poly = box(*input_gdf.bounds.values[0])
    input_box_df = pd.DataFrame({'name': ['bbox'],
                                 'geometry': [input_box_poly]})
    input_box_gdf = gpd.GeoDataFrame(input_box_df, crs="EPSG:4326",
                                     geometry=input_box_df['geometry'])
    input_box_shp = os.path.join(outpath, 'input_bbox.shp')
    input_box_gdf.to_file(input_box_shp)

    osm_clip_gdf = osm_bbox_gdf.loc[osm_bbox_gdf.geometry.intersects(
        input_box_poly)]
    osm_clip_list = list(osm_clip_gdf['index'].values)

    fclass_save = ['water', 'reservoir']

    print("   load in OpenStreetMap for {0} regions".format(
        len(osm_clip_gdf)), end="")

    """load and concatenate overlapping countries"""
    lake_polys_gdf = None
    area_threshold_0 = area_threshold / 10
    for i, tile in enumerate(osm_clip_list[0::]):

        print_time(f'   load region: {tile}', starttime)
        sys.stdout.flush()

        waterpolys_shp = f"{OSM_src_path}/{tile}/{tile}_waterpolygons.shp"

        try:
            waterpolys_temp_gdf = gpd.read_file(waterpolys_shp)
        except:
            print(f'could not load {tile} data')
            continue

        if not waterpolys_temp_gdf.empty:
            lake_polys_temp_gdf = waterpolys_temp_gdf.loc[waterpolys_temp_gdf['fclass'].isin(
                fclass_save)].copy()

            if lake_polys_temp_gdf.empty:
                continue

            """filter by size"""
            lake_polys_temp_gdf['area'] = lake_polys_temp_gdf['geometry'].to_crs('epsg:6933')\
                .map(lambda p: p.area / 10**6)

            # apply area threshold
            lake_polys_temp_gdf = lake_polys_temp_gdf[lake_polys_temp_gdf['area']
                                                      > area_threshold_0]

            """append to lake geodataframe"""
            if not lake_polys_temp_gdf.empty:
                if lake_polys_gdf is None:
                    lake_polys_gdf = lake_polys_temp_gdf.copy()
                else:
                    lake_polys_gdf = lake_polys_gdf.append(lake_polys_temp_gdf)

    print_time('lake data merged', starttime)

    if not lake_polys_gdf.empty:

        """clip to basin shape"""
        lake_polys_gdf = lake_polys_gdf.loc[lake_polys_gdf.geometry.within(
            input_poly)]
        lake_polys_gdf.drop('name', axis=1, inplace=True)

        if not lake_polys_gdf.empty:

            lake_polys_gdf['group'] = 1
            lake_polys_gdf = lake_polys_gdf.dissolve(by='group')
            lake_polys_gdf = lake_polys_gdf.explode()
            lake_polys_gdf['area'] = lake_polys_gdf['geometry'].to_crs('epsg:6933')\
                .map(lambda p: p.area / 10**6)
            lakepolys_big_gdf = lake_polys_gdf[lake_polys_gdf['area']
                                               > area_threshold]

            if not lakepolys_big_gdf.empty:

                """close holes in lakes"""
                lakepolys_big_gdf['geometry'] = lakepolys_big_gdf.geometry.apply(
                    lambda p: close_holes(p))

                """calculate permimeter to area ratio (km/km^2) and filter out river reservoirs"""
                lakepolys_big_gdf['perimeter'] = lakepolys_big_gdf['geometry'].to_crs('epsg:6933')\
                    .map(lambda p: p.length / 10**3)
                lakepolys_big_gdf['pa_ratio'] = lakepolys_big_gdf['perimeter'] / \
                    lakepolys_big_gdf['area']
                lakepolys_big_gdf = lakepolys_big_gdf[lakepolys_big_gdf['pa_ratio'] < 1.0]

                if not lakepolys_big_gdf.empty:

                    """buffer and save to file"""
                    lakepolys_big_gdf = pyct.buffer_gdf_simple(
                        lakepolys_big_gdf, lake_buffer)
                    lakepolys_big_gdf.to_file(flatten_lakes_shp)

        else:
            lakepolys_big_gdf = lake_polys_gdf

    else:
        lakepolys_big_gdf = lake_polys_gdf

    if lakepolys_big_gdf.empty:
        lakepolys_big_gdf = None

    return lakepolys_big_gdf


def flatten_lakes(flatten_lakes_shp, lake_dem_dir, lake_shp_dir,
                  inputDEM, inputSWO, SWO_cutoff, SWO_burn, overwrite_lakes=True):
    """
    Flatten very large lakes (found in find_large_lakes()) post-conditioning to improve
    elevations and stream placement

    Parameters
    ----------
    flatten_lakes_shp : path
        input shap file of large lakes to be flattened
    lake_dem_dir : path
        output directory for intermediate files
    lake_shp_dir : path
        input directory for lake shape files (one shape file per lake)
    inputDEM : path
        input raster (or VRT) of complete, hyro-conditioned DEM
    inputSWO : path
        input raster (or VRT) of Surface Water Occurence values
    SWO_cutoff : int
        Water percent cutoff (0-100) to consider pixel 'water'
    SWO_burn : float
        depth to sink lake by (meters)
    overwrite_lakes : bool, optional
        overwite existing files?, by default True
    """

    starttime = time.time()
    """Iterate through lakes to flatten"""
    large_lakes_gdf = gpd.read_file(flatten_lakes_shp)
    for index, row in large_lakes_gdf.iterrows():

        print_time('  Flattening lake #{0}'.format(index + 1), starttime)
        singlelake_gdf = large_lakes_gdf.loc[[index], :].copy()
        lake_poly = list(singlelake_gdf.geometry)[0]
        single_lake_shp = os.path.join(
            lake_shp_dir, 'large_lake_{0}.shp'.format(index))
        singlelake_gdf.to_file(single_lake_shp)

        min_z = None
        if 'min_z' in singlelake_gdf.columns:  # manual elevation override
            min_z = singlelake_gdf['min_z'].values[0]
            print('    min_z = ', min_z)
            if not np.isnan(min_z):
                print('    manually setting lake to {} meters'.format(min_z))
            else:
                min_z = None

        """Crop DEM and SWO to lake polygon"""
        lakeDEM = os.path.join(lake_dem_dir, "DEM_{0}.tif".format(index))
        lakeSWO_tif = os.path.join(lake_dem_dir, "SWO_{0}.tif".format(index))
        output_tif = os.path.join(
            lake_dem_dir, "conditionedDEM_{0}.tif".format(index))
        if os.path.exists(output_tif):
            os.remove(output_tif)

        if not os.path.exists(lakeDEM) or overwrite_lakes:
            pyct.create_cutline_raster_set_res(inputDEM, single_lake_shp,
                                               lakeDEM, nodata=np.nan, usecutline=True)

        if not os.path.exists(lakeSWO_tif) or overwrite_lakes:
            pyct.create_cutline_raster(inputSWO, single_lake_shp,
                                       lakeSWO_tif, srcDS=lakeDEM, nodata=0,
                                       usecutline=True, outdtype=gdal.GDT_Byte,
                                       rsAlg=gdal.GRIORA_Bilinear)

        raster = gdal.Open(lakeDEM, gdal.GA_ReadOnly)
        xshape = raster.RasterXSize
        yshape = raster.RasterYSize
        band = raster.GetRasterBand(1)
        nodataval = band.GetNoDataValue()   # Set nodata value
        raster = None
        band = None
        DEM_array = gdal_array.LoadFile(lakeDEM)
        DEM_array_out = DEM_array.copy()

        if np.isnan(nodataval):
            nodataval = -9999
            print(
                '    nodata value is NAN, change nodataval to {0}'.format(nodataval))
            no_data_ind = np.nonzero(np.isnan(DEM_array))
            DEM_array_out[no_data_ind] = nodataval
        else:
            no_data_ind = np.nonzero(DEM_array == nodataval)

        SWO_array = gdal_array.LoadFile(lakeSWO_tif)
        real_ind = np.nonzero((DEM_array_out != nodataval))
        water_ind = np.nonzero((SWO_array > SWO_cutoff)
                               & (DEM_array_out != nodataval))
        deep_water_ind = np.nonzero((SWO_array > 80)
                                    & (DEM_array_out != nodataval))

        per_water = 100 * (len(deep_water_ind[0]) / len(real_ind[0]))
        print('      % deep water = {0:.2f}%'.format(per_water))
        if len(deep_water_ind[0]) / len(real_ind[0]) < 0.2:
            print('      Lake less than 50% deep water... skipping')
            continue

        """
        find 10th % elevation of SWO>80 pixels. Set all SWO>cutoff pixels to
        min elevation. Then apply SWO burn to give edges a grade
        """

        if min_z is None:
            water_min = np.nanpercentile(DEM_array_out[deep_water_ind], 10)
        else:
            water_min = min_z

        water_max = np.nanpercentile(DEM_array_out[deep_water_ind], 90)
        water_elevs = DEM_array_out[(DEM_array_out >= water_min) & (
            DEM_array_out <= water_max)]
        water_avg = np.nanmean(water_elevs)

        # if water_max - water_min < 0.01:
        #    print('      Lake already flat, skipping')
        #    continue

        if (water_max - water_min > 20.0) and (water_min < 0.0):
            print('      Probable large data gap - fill with positive lake min value')
            print('      Old lake min value = {0}'.format(water_min))

            if min_z is None:
                water_min = np.nanpercentile(
                    DEM_array_out[DEM_array_out > 0.0], 10)
            else:
                water_min = min_z

            print('      New lake min value = {0}'.format(water_min))
            DEM_array_out[DEM_array > 0.0] = water_min
            DEM_array[DEM_array > 0.0] = water_min

        print('    mean deep water elevation = {0}'.format(water_avg))
        print('    min deep water elevation = {0}'.format(water_min))
        print('    max deep water elevation = {0}'.format(water_max))

        DEM_array_out[water_ind] = water_min

        """add SWO burn if elevation is not manually set"""
        if min_z is None:
            burn_add = np.zeros_like(DEM_array_out)
            burn_add = (SWO_array - SWO_cutoff) * \
                (SWO_burn / (80 - SWO_cutoff))
            burn_add[(SWO_array >= 80)] = SWO_burn
            burn_add[burn_add < 0.0] = 0.0  # no negative burn values
            burn_add[(burn_add > SWO_burn)] = 0
            DEM_array_out[water_ind] -= burn_add[water_ind]

        DEM_array_out[no_data_ind] = np.nan

        DEM_diff = DEM_array_out - DEM_array
        DEM_diff[no_data_ind] = 0.0

        """fix any raised elevations"""
        if min_z is None:
            DEM_array_out[DEM_diff > 0.5] = DEM_array[DEM_diff > 0.5]

        """reset nodataval"""
        DEM_array_out[DEM_array == nodataval] = np.nan

        pyct.npy2tif(DEM_array_out, lakeDEM, output_tif,
                     nodata=np.nan, dtype=gdal.GDT_Float32)


def stream_minimize(inDEM_array, waterlines_tif, water_mask_array,
                    max_burn=30, stream_filter_size=9, river_filter_size=17,
                    canal_filter_size=3, riv_width_size=13,
                    nodataval=-9999, canal_as_river=False,
                    adj_land_diff=3.0, no_vector_mask=None):
    """
    Apply a moving window minimum filter to rasterized stream pixels. This helps
    to cut "through" vegetation around small streams and rivers by applying the minimum
    elevation within a defined pixel region to the stream pixel, whether from an adjacent
    river pixel that does not have vegetation, or a land pixel outside of a small
    riparian zone. Water pixels that were included in the large river flattening process
    are ignored.

    The function treats 3 classes of water features differently:

    rivers        : widened to 3 pixels. moving window filter applied 3 times
    streams       : left at 1 pixel width. moving window filter applied twice
    canals/drains : left at 1 pixel width. filter applied only once

    Parameters
    ----------
    inDEM_array : array-like
        2D array of input DEM values to be modified
    waterlines_tif : path
        tif file of classified water pixels
    water_mask_array : array-like
        2D array mask of pixels that have undergone water-flattening (=1)
    stream_filter_size : int, optional
        window size for stream pixels, by default 9
    river_filter_size : int, optional
        window size for river pixels, by default 9
    canal_filter_size : int, optional
        window size for canal and drain pixles, by default 3
    nodataval : int, optional
        no data value of input DEM, by default -9999

    Returns
    -------
    array
        2D array of modified DEM values
    """

    outDEM_array = inDEM_array.copy()

    streamclass_array = gdal_array.LoadFile(waterlines_tif)
    water_mask_edit = water_mask_array.copy()
    streamclass_array[water_mask_array != 12] = 0

    # create mask of large rivers (==1) to disqualify from stream minimize
    big_river_mask = np.where(
        ((water_mask_array > 2) & (water_mask_array < 12)), 1, 0)
    vector_mask = np.zeros_like(big_river_mask)

    if canal_as_river:
        drain_ind = np.nonzero((streamclass_array > 7)  # & # just drains
                               & (big_river_mask == 0))
        canal_ind = np.nonzero((streamclass_array == 7)  # & # just canals
                               & (big_river_mask == 0))
        vector_mask[drain_ind] = 3
        vector_mask[canal_ind] = 1
    else:
        canal_ind = np.nonzero((streamclass_array > 6)  # & # canals and drains
                               & (big_river_mask == 0))
        vector_mask[canal_ind] = 3

    stream_ind = np.nonzero((streamclass_array == 6)
                            & (big_river_mask == 0))
    river_ind = np.nonzero((streamclass_array == 5)
                           & (big_river_mask == 0))

    vector_mask[stream_ind] = 2
    vector_mask[river_ind] = 1

    if no_vector_mask is not None:
        stream_test_mask = np.where(vector_mask > 0, 1, 0)
        labeled_array, num_features = ndimage.label(
            stream_test_mask, structure=[[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        clump_IDs = np.unique(labeled_array[no_vector_mask == 1])
        for clump in clump_IDs:  # stream segment within no_vector mask?
            clump_ind = np.nonzero(labeled_array == clump)
            if min(no_vector_mask[clump_ind]) > 0:
                vector_mask[clump_ind] = 0
                streamclass_array[clump_ind] = 0

    """Apply a moving minimum filter to buffered water vector features"""
    class_array = np.zeros_like(streamclass_array)
    class_array[vector_mask == 1] = 1
    class_array = ndimage.maximum_filter(class_array, size=3)
    river_ind = np.nonzero(class_array == 1)
    vector_mask[river_ind] = 1

    # Initialize test array and run 3x3 smoothing filter to help fill in outliers
    testDEM_array = inDEM_array.copy()
    kernel = np.array([[.5, .5, .5],
                       [.5, 1.0, .5],
                       [.5, .5, .5]])
    testDEM_array = apply_convolve_filter(
        testDEM_array, kernel, normalize_kernel=True)

    # set all pixels outside river width param of stream/river 1e6
    # (disqualify from minimum filter)
    riv_mask = np.where((streamclass_array > 0) & (big_river_mask == 0), 1, 0)
    riv_adj_mask = ndimage.maximum_filter(riv_mask, size=riv_width_size)
    riv_bank_mask = ndimage.maximum_filter(riv_mask, size=7)
    testDEM_array[riv_adj_mask == 0] = 1e6
    DEM_adj_land_array = np.where((riv_bank_mask == 0), testDEM_array, 1e6)

    # set no data value to high number to avoid minimum filter
    testDEM_array[inDEM_array == nodataval] = 1e6

    # Compute moving minimum for water pixels and adjacent land
    min_wtr_river = ndimage.minimum_filter(
        testDEM_array, size=river_filter_size)
    min_wtr_stream = ndimage.minimum_filter(
        testDEM_array, size=stream_filter_size)
    min_wtr_canal = ndimage.minimum_filter(
        testDEM_array, size=canal_filter_size)

    min_adj_land = ndimage.minimum_filter(
        DEM_adj_land_array, size=riv_width_size)

    # replace high elevation values with testDEM array
    min_adj_land = np.where((min_adj_land > 1e5), testDEM_array, min_adj_land)

    outDEM_array[vector_mask == 3] = min_wtr_canal[vector_mask == 3]
    water_mask_edit[vector_mask == 3] = 10

    outDEM_array[vector_mask == 2] = min_wtr_stream[vector_mask == 2]
    water_mask_edit[vector_mask == 2] = 9

    outDEM_array[vector_mask == 1] = min_wtr_river[vector_mask == 1]
    water_mask_edit[vector_mask == 1] = 8

    # find pixels burned over max limit and correct
    diff_array = inDEM_array - outDEM_array
    overburn_ind = np.nonzero(diff_array > max_burn)
    outDEM_array[overburn_ind] = inDEM_array[overburn_ind] - max_burn

    # find pixels set too far below or above adjacent land
    diff_land_array = np.where(
        (riv_bank_mask == 1), min_adj_land - outDEM_array, 0)
    underburn_ind = np.nonzero((diff_land_array < 0.5) & (water_mask_edit > 0)
                               & (big_river_mask == 0) & (streamclass_array > 0))
    overburn_ind = np.nonzero((diff_land_array > adj_land_diff) & (water_mask_edit > 0)
                              & (big_river_mask == 0) & (water_mask_edit != 12))

    overburn_bank_ind = np.nonzero(
        (diff_land_array > 0.0) & (water_mask_edit == 12))

    outDEM_array[underburn_ind] = min_adj_land[underburn_ind] - 0.5
    outDEM_array[overburn_ind] = min_adj_land[overburn_ind] - adj_land_diff
    outDEM_array[overburn_bank_ind] = min_adj_land[overburn_bank_ind]

    # reset no data value
    outDEM_array[inDEM_array == nodataval] = nodataval

    # edit water mask only where no value currently set (don't overwrite)
    edit_mask_ind = np.nonzero(
        (water_mask_array == 0) | (water_mask_array == 12))
    water_mask_array[edit_mask_ind] = water_mask_edit[edit_mask_ind]

    return outDEM_array, water_mask_array


def river_minimize(inDEM_array, water_mask_array, SWO_array, max_burn=5,
                   river_filter_size=3, nodataval=-9999):
    """
    Apply a moving window minimum filter to rivers.

    Parameters
    ----------
    inDEM_array : array-like
        2D array of input DEM values to be modified
    water_mask_array : array-like
        2D array mask of pixels that have undergone water-flattening
    river_filter_size : int, optional
        window size for river pixels, by default 3
    nodataval : int, optional
        no data value of input DEM, by default -9999

    Returns
    -------
    array
        2D array of modified DEM values
    """

    outDEM_array = inDEM_array.copy()

    """find flattened river centerlines"""
    river_ind = np.nonzero((water_mask_array > 2)
                           & (water_mask_array < 8))

    riverbank_ind = np.nonzero((water_mask_array > 0)
                               & (water_mask_array < 3))

    if len(river_ind[0]) > 0:

        max_burn_edge = max_burn - 1.0

        test_array = np.zeros_like(water_mask_array)
        test_array[river_ind] = 1
        test_array[riverbank_ind] = 2
        test_array = ndimage.maximum_filter(test_array, size=3)
        river_ind_narrow = np.nonzero((test_array == 1)
                                       & (SWO_array > 5))

        # where test_array == 1 is inner part of river, offset from the banks
        testDEM_array = np.where(test_array == 1, inDEM_array, 1e6)

        # set no data value to high number to avoid minimum filter
        testDEM_array[inDEM_array == nodataval] = 1e6

        # moving minimum filter
        minimum_array_river = ndimage.minimum_filter(
            testDEM_array, size=river_filter_size)
        testDEM_array[river_ind_narrow] = minimum_array_river[river_ind_narrow]

        # test for overburned pixels
        diff_array = inDEM_array - testDEM_array
        overburn_ind = np.nonzero(diff_array > max_burn)
        testDEM_array[overburn_ind] = inDEM_array[overburn_ind] - max_burn

        outDEM_array = np.where(test_array == 1, testDEM_array, inDEM_array)
        outDEM_array[outDEM_array == 1e6] = inDEM_array[outDEM_array == 1e6]

    return outDEM_array


def compute_vec_burn(waterlines_tif, riverburn=2.0,
                     flat_riverburn=0.0, stream_burn=1.0, drainburn=1.0,
                     riv_width_size=11):
    """
    Create a map of burn values for different Open Street map water line features.
    Excludes any river segments that have been flattened (water_mask_array > 1).

    To lessen sharp drops imposed by lowering single-pixel wide features, the rasterized
    lines are used as a centerline to create a shallow, linearly-sloped V-shape, where
    the center (bottom) of the V is set to the burn value given for that specific water type.

    Parameters
    ----------
    waterlines_tif : path
        rasterized tif of water line features
    water_mask_array : array
        2D array mask of any flattened water segments
    riverburn : float, optional
        meters to lowerr the center of rivers by, by default 4.0
    flat_riverburn : float, optional
        meters to lower river liine features that have already been flattened, by default 0.0
    stream_burn : float, optional
        meters to lower center of stream and canal features, by default 2.0
    drainburn : float, optional
        meters to lower anny drains, by default 1.0

    Returns
    -------
    array
        2D array of burn values (meters)
    """

    streamclass_array = gdal_array.LoadFile(waterlines_tif)

    sclass_buffer_array = np.where(
        streamclass_array > 0, streamclass_array, 99)
    sclass_buffer_array = ndimage.minimum_filter(
        sclass_buffer_array, size=riv_width_size)
    sclass_buffer_array = np.where(
        sclass_buffer_array < 99, sclass_buffer_array, 0)

    stream_ind = np.nonzero((streamclass_array > 5)
                            & (streamclass_array < 8))

    river_ind = np.nonzero(streamclass_array == 5)

    drain_ind = np.nonzero(streamclass_array == 8)

    kernel = np.array([[.5, .5, .5],
                       [.5, 1.0, .5],
                       [.5, .5, .5]])
    streamburn_array = np.zeros_like(streamclass_array, dtype=np.float32)

    """ set burn values"""
    streamburn_array[river_ind] = riverburn
    streamburn_array[stream_ind] = stream_burn
    streamburn_array[drain_ind] = drainburn

    """apply smoothing kernel 3x to create a shallow V-shape that is ~9 pixels accross"""
    streamburn_array = apply_convolve_filter(
        streamburn_array, kernel, normalize_kernel=True)
    streamburn_array = apply_convolve_filter(
        streamburn_array, kernel, normalize_kernel=True)
    streamburn_array = apply_convolve_filter(
        streamburn_array, kernel, normalize_kernel=True)

    # multiply to line up with channel burn value
    streamburn_array = 3.0 * streamburn_array

    """After multiplying, reset channel pixels to prescribed values"""
    streamburn_array[stream_ind] = stream_burn
    streamburn_array[drain_ind] = drainburn
    streamburn_array[river_ind] = riverburn
    streamburn_array[streamburn_array > riverburn] = riverburn

    return streamburn_array, sclass_buffer_array


def smooth_rivers(inDEM_array, DEM_burn_arr, water_mask_array,
                  SWO_array, nodata, SWO_cut=90, max_lower=10.0,
                  filter_size=21):
    """apply moving minimum river value for pixels SWO>90"""
    big_river_ind = np.nonzero((water_mask_array > 2) & (water_mask_array < 8)
                               & (inDEM_array != nodata) & (SWO_array > SWO_cut))

    if len(big_river_ind[0]):

        """smooth large river test values"""
        kernel = np.ones((5, 5))
        kernel = (1. / np.sum(kernel)) * kernel
        DEM_burn_arr_smooth = apply_convolve_filter(
            DEM_burn_arr, kernel)

        testDEM_array = 1e6 * (np.ones_like(DEM_burn_arr))
        testDEM_array[big_river_ind] = DEM_burn_arr_smooth[big_river_ind]
        testDEM_min_array = ndimage.minimum_filter(
            testDEM_array, size=filter_size)
        testDEM_array[big_river_ind] = testDEM_min_array[big_river_ind]

        test_diff_array = np.zeros_like(DEM_burn_arr)
        test_diff_array[big_river_ind] = DEM_burn_arr[big_river_ind] - \
            testDEM_array[big_river_ind]

        lower_ind = np.nonzero((test_diff_array >= 0.01)
                               & (test_diff_array <= max_lower))
        DEM_burn_arr[lower_ind] = testDEM_array[lower_ind]

    return DEM_burn_arr


def raise_small_streams(inDEM_array, DEM_burn_arr, water_mask_array, nodata,
                        extra_raise=0.50, max_raise=5.0, fsize=51):
    """apply minimum stream value above lowest adjacent river value)"""
    stream_ind = np.nonzero((water_mask_array > 7) & (water_mask_array != 11)
                           & (water_mask_array != 15)
                            & (inDEM_array != nodata))
    river_ind = np.nonzero((water_mask_array > 3) & (water_mask_array < 8)
                           & (inDEM_array != nodata))
    fsize_apply = fsize - 16

    if (len(stream_ind[0]) > 0) and (len(river_ind[0])):  # both rivers and stream present
        """raise elevation above any nearby SWO>cutoff pixels"""

        """Buffer stream centerlines"""
        stream_buffer_array = np.zeros_like(water_mask_array)
        stream_buffer_array[stream_ind] = 1
        stream_buffer_array = ndimage.maximum_filter(
            stream_buffer_array, size=5)
        river_ind = np.nonzero((water_mask_array > 3) & (water_mask_array < 8)
                               & (inDEM_array != nodata) & (stream_buffer_array == 0))

        """smooth large river test values"""
        kernel = np.ones((5, 5))
        kernel = (1. / np.sum(kernel)) * kernel
        DEM_burn_arr_smooth = apply_convolve_filter(inDEM_array, kernel)

        testDEM_array = 1e6 * (np.ones_like(DEM_burn_arr))
        testDEM_array[river_ind] = DEM_burn_arr_smooth[river_ind]
        testDEM_min_array = ndimage.minimum_filter(testDEM_array, size=fsize)

        apply_mask_array = np.zeros_like(water_mask_array)
        apply_mask_array[river_ind] = 1
        apply_mask_array = ndimage.maximum_filter(
            apply_mask_array, size=fsize_apply)
        stream_ind_apply = np.nonzero((water_mask_array > 7) & (water_mask_array != 11)
                                     & (water_mask_array != 15) & (inDEM_array != nodata)
                                      & (apply_mask_array == 1))

        """fix pixels raised too much"""
        testDEM_min_array[apply_mask_array == 0] = -9999
        test_diff_array = 1e6 * (np.ones_like(DEM_burn_arr))
        test_diff_array[stream_ind_apply] = DEM_burn_arr[stream_ind_apply] - \
            testDEM_min_array[stream_ind_apply]
        too_low_ind = np.nonzero((test_diff_array <= extra_raise)
                                 & (test_diff_array >= -max_raise))
        DEM_burn_arr[too_low_ind] = testDEM_min_array[too_low_ind] + extra_raise

    return DEM_burn_arr


def raise_small_rivers(inDEM_array, DEM_burn_arr, water_mask_array, SWO_array,
                       nodata, SWO_cutoff_high=50, SWO_cutoff_low=5, extra_raise=0.10,
                       max_raise=3.0, filter_size=51):
    """apply minimum non-SWO river value (above lowest adjacent SWO river value)"""
    small_riv_ind = np.nonzero((water_mask_array > 0) & (water_mask_array < 8)
                               & (inDEM_array != nodata) & (SWO_array < SWO_cutoff_low))
    big_river_ind = np.nonzero((water_mask_array > 2) & (water_mask_array < 8)
                               & (inDEM_array != nodata) & (SWO_array > SWO_cutoff_high))
    fsize_apply = filter_size - 16

    # both rivers and stream present
    if (len(small_riv_ind[0]) > 0) and (len(big_river_ind[0])):
        """raise elevation above any nearby SWO>cutoff pixels"""
        testDEM_array = 1e6 * (np.ones_like(DEM_burn_arr))
        testDEM_array[big_river_ind] = DEM_burn_arr[big_river_ind]
        testDEM_min_array = ndimage.minimum_filter(
            testDEM_array, size=filter_size)

        sm_riv_mask_array = np.zeros_like(water_mask_array)
        sm_riv_mask_array[small_riv_ind] = 1
        apply_mask_array = np.zeros_like(water_mask_array)
        apply_mask_array[big_river_ind] = 1
        apply_mask_array = ndimage.maximum_filter(
            apply_mask_array, size=fsize_apply)
        sriv_ind_apply = np.nonzero(
            (sm_riv_mask_array == 1) & (apply_mask_array == 1))

        test_diff_array = 1e6 * (np.ones_like(DEM_burn_arr))
        test_diff_array[sriv_ind_apply] = DEM_burn_arr[sriv_ind_apply] - \
            testDEM_min_array[sriv_ind_apply]

        too_low_ind = np.nonzero((test_diff_array <= extra_raise)
                                 & (test_diff_array >= -max_raise))
        DEM_burn_arr[too_low_ind] = testDEM_min_array[too_low_ind] + extra_raise

    return DEM_burn_arr


def raise_canals(inDEM_array, DEM_burn_arr, water_mask_array,
                 SWO_array, waterlines_tif, nodata, extra_raise=0.50,
                 max_raise=5.0, SWO_cutoff=20):
    """apply minimum flattened canal value (above lowest adjacent river value)"""
    if waterlines_tif is not None:
        streamclass_array = gdal_array.LoadFile(waterlines_tif)
        canal_center_ind = np.nonzero((streamclass_array > 6) & (water_mask_array < 8)
                                      & (water_mask_array > 0) & (inDEM_array != nodata))
        river_center_ind = np.nonzero((streamclass_array == 5) & (water_mask_array < 8)
                                      & (water_mask_array > 0) & (inDEM_array != nodata))

        # both rivers and canals present
        if (len(canal_center_ind[0]) > 0) and (len(river_center_ind[0])):
            """buffer canal and river centerlines"""
            riv_center_array = np.zeros_like(DEM_burn_arr)
            riv_center_array[river_center_ind] = 1
            riv_center_array = ndimage.maximum_filter(
                riv_center_array, size=31)
            river_center_ind = np.nonzero(
                (riv_center_array > 0) & (SWO_array > SWO_cutoff))
            if len(river_center_ind[0]) > 0:
                canal_array = np.zeros_like(DEM_burn_arr)
                canal_array[canal_center_ind] = 1
                canal_array = ndimage.maximum_filter(canal_array, size=51)
                canal_ind = np.nonzero((canal_array == 1)
                                       & (riv_center_array == 0))

                """raise elevation above any nearby SWO>cutoff pixels"""
                testDEM_array = 1e6 * (np.ones_like(DEM_burn_arr))
                testDEM_array[river_center_ind] = DEM_burn_arr[river_center_ind]
                testDEM_min_array = ndimage.minimum_filter(
                    testDEM_array, size=71)
                far_river_ind = np.nonzero(testDEM_min_array > 1e5)
                testDEM_min_array[far_river_ind] = -99

                test_diff_array = 1e6 * (np.ones_like(DEM_burn_arr))
                test_diff_array[canal_ind] = DEM_burn_arr[canal_ind] - \
                    testDEM_min_array[canal_ind]
                too_low_ind = np.nonzero((test_diff_array <= extra_raise)
                                         & (test_diff_array >= -max_raise))
                DEM_burn_arr[too_low_ind] = testDEM_min_array[too_low_ind] + extra_raise

    return DEM_burn_arr


def set_ocean_elev(dem_array, land_mask, nodata=-9999,
                   ocean_z=-1.0):
    """ set ocean pixels to ocean elevation value """
    if ocean_z is not None:
        ocean_pixels_all = np.nonzero((land_mask == 0)
                                      & (dem_array != nodata))
        dem_array[ocean_pixels_all] = ocean_z

    return dem_array, ocean_pixels_all


def clip_ocean(DEM_tif, outpath, buffer_shp, input_gdf,
               edm_array=None, nodata=-9999, basename='DEM',
               interior_mask=None,
               land_mask=None, osm_tif=None,
               osm_coast_clip=True, usecutline=True):

    dem_array = gdal_array.LoadFile(DEM_tif)
    output_array = dem_array.copy()

    if edm_array is None:
        edm_array = np.zeros_like(DEM_tif, dtype=np.int8)

    if land_mask is None:
        land_mask = np.zeros_like(DEM_tif, dtype=np.int8)

    """ set exterior ocean pixels to nodata """
    if osm_coast_clip:

        """ read in OSM water polygons """
        if os.path.exists(osm_tif):
            osm_array = gdal_array.LoadFile(
                osm_tif)
        else:
            osm_array = np.zeros_like(edm_array)

        ocean_pixels = np.nonzero((edm_array > 3) &
                                  (osm_array != 1) &
                                  (osm_array != 2) &
                                  (osm_array != 3) &
                                  (land_mask != 1))
        # (crop_donut_array == 1) &
    else:
        ocean_pixels = np.nonzero((edm_array > 3)
                                  & (land_mask != 1))

    output_array[ocean_pixels] = nodata

    if interior_mask is not None:
        output_array[interior_mask == 1] = nodata

    return output_array, ocean_pixels


def enforce_monotonic_streams(DEM_tif, outpath, DEM_burn_arr,
                              water_mask_array, donut_array,
                              basename, radius=25,
                              nodata=-9999,
                              wbt=None, waterlines_tif=None,
                              verbose_print=False):
    """ fix non-monotonic stream segments"""
    starttime = time.time()
    s_stream = [[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]]

    if waterlines_tif is not None:
        waterlines_array = gdal_array.LoadFile(waterlines_tif)
        streamlines_array = np.zeros_like(water_mask_array)
        streamlines_ind = np.nonzero(
            (waterlines_array > 0) & ((water_mask_array < 3) |
                                      ((water_mask_array > 7) & (water_mask_array < 11))))
        streamlines_array[streamlines_ind] = 1

    else:
        streamlines_array = np.zeros_like(water_mask_array)
        streamlines_ind = np.nonzero((water_mask_array > 7)
                                     & (water_mask_array < 11))
        streamlines_array[streamlines_ind] = 1

    burn_stream_ind = streamlines_ind
    raised_area_stream = radius / 4
    n_stream_ind = len(burn_stream_ind[0])

    if n_stream_ind > 50:
        stream_elevations = DEM_burn_arr[burn_stream_ind]
        min_stream_h = np.maximum(0, np.percentile(stream_elevations, 2))
        min_stream_h = np.floor(min_stream_h) + 0.5
        max_stream_h = np.percentile(stream_elevations, 98)
        max_stream_h = np.floor(max_stream_h) + 0.5

        if (max_stream_h - min_stream_h) > 5.0:
            raised_step = 2.0
            if (max_stream_h - min_stream_h) > 20:
                raised_step = 3.0
            if (max_stream_h - min_stream_h) > 100:
                raised_step = 5.0
            if (max_stream_h - min_stream_h) > 500:
                raised_step = 10.0
            stream_h_steps = np.arange(
                min_stream_h, max_stream_h, raised_step)

            edge_interp = False
            interp_mask = np.zeros_like(water_mask_array)

            for h in stream_h_steps:
                """difference h and stream elevation and label non-mono sections"""
                diff_array = np.zeros_like(DEM_burn_arr, dtype=np.float32)
                diff_array[burn_stream_ind] = DEM_burn_arr[burn_stream_ind] - h
                labeled_array = np.zeros_like(DEM_burn_arr)
                labeled_array[diff_array > 0] = 99
                labeled_array, num_features = ndimage.label(
                    labeled_array, structure=s_stream)
                clump_ind = np.nonzero((labeled_array > 0))

                """remove IDs that cross boundary"""
                donut_clump_ind = donut_array[clump_ind]
                donutIDs = np.unique(
                    labeled_array[(donut_array == 1) & (labeled_array > 0)])

                for donutID in donutIDs:
                    labeled_array[labeled_array == donutID] = 0

                """relabel without boundary crossing clumps"""
                clump_ind = np.nonzero(labeled_array)
                raised_IDs = np.unique(labeled_array[clump_ind])

                # print_time("     possible raised stream clumps = {0}".format(len(raised_IDs)), starttime)
                # clump_tif = os.path.join(outpath, "{0}_streamclump_{1}.tif".format(basename, h))
                # pyct.npy2tif(labeled_array, DEM_tif, clump_tif, nodata=0, dtype=gdal.GDT_Int16)

                for clumpID in raised_IDs:
                    clumpID_ind_temp = np.nonzero(
                        labeled_array[clump_ind] == clumpID)[0]
                    clumpID_ind = (clump_ind[0][clumpID_ind_temp],
                                   clump_ind[1][clumpID_ind_temp])

                    if len(clumpID_ind_temp) < raised_area_stream:  # patch too small
                        # print("      {0}- too small ({1})".format(clumpID, len(clumpID_ind_temp)))
                        continue

                    if len(clumpID_ind_temp) > n_stream_ind / 4:  # patch too big
                        # print("      {0}- too big".format(clumpID))
                        continue

                    clump_max_z = np.amax(DEM_burn_arr[clumpID_ind])
                    clump_min_z = np.amin(DEM_burn_arr[clumpID_ind])
                    if (clump_max_z - h) < (raised_step * 0.5):
                        # print('    {0}- too small elevation change'.format(clumpID))
                        continue

                    """ move to chip surrounding raised stream section"""
                    ind0_min, ind0_max = clumpID_ind[0].min(
                    ) - 4, clumpID_ind[0].max() + 5
                    ind1_min, ind1_max = clumpID_ind[1].min(
                    ) - 4, clumpID_ind[1].max() + 5
                    clump_arr = labeled_array[ind0_min:ind0_max,
                                              ind1_min:ind1_max]
                    clump_mask = np.zeros_like(clump_arr)
                    clumpID_ind_chip = np.nonzero(clump_arr == clumpID)
                    clump_mask[clumpID_ind_chip] = 99

                    """crop to chip"""
                    diff_array_temp = diff_array[ind0_min:ind0_max,
                                                 ind1_min:ind1_max]
                    streamlines_array_temp = streamlines_array[ind0_min:ind0_max,
                                                               ind1_min:ind1_max]
                    flatten_mask_temp = water_mask_array[ind0_min:ind0_max,
                                                         ind1_min:ind1_max]
                    DEM_array_temp = DEM_burn_arr[ind0_min:ind0_max,
                                                  ind1_min:ind1_max].copy()

                    clump_lake_n_ind = np.count_nonzero(
                        flatten_mask_temp[clumpID_ind_chip] == 3)
                    if clump_lake_n_ind > 10:
                        # print("       clump in lake, skip")
                        continue

                    """buffer clump to find adjacent river pixels"""
                    clump_buffer_mask = apply_convolve_filter(clump_mask,
                                                              np.ones((5, 5)))
                    clumpID_buffer_ind = np.nonzero(clump_buffer_mask)

                    adj_stream_temp = np.flatnonzero((diff_array_temp[clumpID_buffer_ind] < 0)
                                                    & (clump_mask[clumpID_buffer_ind] == 0)
                                                     & (streamlines_array_temp[clumpID_buffer_ind] == 1))

                    if len(adj_stream_temp) == 0:
                        # print('    {0}- no adjacent stream pixels'.format(clumpID))
                        continue

                    adj_stream_ind = (clumpID_buffer_ind[0][adj_stream_temp],
                                      clumpID_buffer_ind[1][adj_stream_temp])

                    """Check that all adjacent river pixels are lower than clump"""
                    if DEM_array_temp[adj_stream_ind].max() > h:
                        # print('    {0}- adjacent river higher than h'.format(clumpID))
                        continue

                    """clump adjacent river pixels"""
                    adj_stream_array = np.zeros_like(clump_arr)
                    adj_stream_array[adj_stream_ind] = 1
                    labeled_adj_array, num_adj_features = ndimage.label(adj_stream_array,
                                                                        structure=s_stream)
                    if num_adj_features < 2:
                        # print('    {0}- too few adjacent river clumps'.format(clumpID))
                        continue

                    """check that all edge features have 2 pixels"""
                    values, counts = np.unique(labeled_adj_array,
                                               return_counts=True)
                    if np.min(counts) < 2:
                        continue

                    """buffer clump to find adjacent land pixels"""
                    test_land_array = apply_convolve_filter(
                        clump_mask, np.ones((17, 17)))
                    clumpID_land_buffer_ind = np.nonzero(test_land_array)
                    adj_land_temp = np.nonzero((diff_array_temp[clumpID_land_buffer_ind] == 0)
                                               & (flatten_mask_temp[clumpID_land_buffer_ind] < 3))[0]
                    adj_land_ind = (clumpID_land_buffer_ind[0][adj_land_temp],
                                    clumpID_land_buffer_ind[1][adj_land_temp])

                    """Check that most adjacent land pixels are higher than h elevation,
                    and that OSM river is not just offset sideways from channel in DEM"""
                    land_elev = DEM_array_temp[adj_land_ind]
                    low_land_elev = np.flatnonzero(
                        land_elev <= (h + raised_step))
                    if len(low_land_elev) >= len(clumpID_ind_temp):
                        continue

                    """Check that each terminal end has adj stream pixels < h"""
                    test_end_array = np.zeros_like(clump_arr, dtype=np.int)
                    test_end_array[clump_arr == clumpID] = 1
                    test_end_array = apply_convolve_filter(
                        test_end_array, np.ones((3, 3)))
                    test_end_vec = test_end_array[clump_mask > 0]
                    values, counts = np.unique(
                        test_end_vec, return_counts=True)
                    ind_min = np.argmin(values)
                    n_end_pts = counts[ind_min]

                    if n_end_pts > 3:
                        # print('    {0}- invalid Y shape'.format(clumpID))
                        continue

                    if (n_end_pts == 3):
                        if num_adj_features < n_end_pts:
                            # print('    {0}- invalid Y shape'.format(clumpID))
                            continue

                    if (max(values) > 4.0) and (n_end_pts < 3):  # Y shape with too few ends
                        # print('    {0}- invalid Y shape (too few ends)'.format(clumpID))
                        continue

                    if num_adj_features == 2:
                        adj_end_ind0 = np.nonzero(labeled_adj_array == 1)
                        adj_end_ind1 = np.nonzero(labeled_adj_array == 2)
                        end0_ind = np.array([adj_end_ind0[0][0],
                                             adj_end_ind0[1][0]])
                        end1_ind = np.array([adj_end_ind1[0][0],
                                             adj_end_ind1[1][0]])

                        end_dist = np.linalg.norm(end0_ind - end1_ind)
                        dist_min = len(clumpID_ind_temp) / 2
                        if (end_dist < dist_min):
                            if verbose_print:
                                print(
                                    '    {0}- endpoints too close together'.format(clumpID))
                            continue

                    if verbose_print:
                        print(
                            f'    {clumpID}- LOWER NON-MONOTONIC stream segment (stream h = {h})')

                    adj_edge_temp = np.nonzero((clump_buffer_mask[clumpID_buffer_ind] > 0)
                                               & (flatten_mask_temp[clumpID_buffer_ind] != 7))[0]
                    adj_edge_ind = (clumpID_buffer_ind[0][adj_edge_temp],
                                    clumpID_buffer_ind[1][adj_edge_temp])

                    """shift back to whole array"""
                    adj_stream_ind = (adj_stream_ind[0] + ind0_min,
                                      adj_stream_ind[1] + ind1_min)
                    adj_edge_ind = (adj_edge_ind[0] + ind0_min,
                                    adj_edge_ind[1] + ind1_min)

                    adj_river_h = DEM_burn_arr[adj_stream_ind].max()
                    edge_interp = True
                    interp_mask[adj_edge_ind] = 1
                    interp_mask[clumpID_ind] = 2
                    DEM_burn_arr[clumpID_ind] = adj_river_h

            if edge_interp:
                """Interpolate removed edge pixels"""
                interp_edge_ind = np.nonzero(interp_mask == 1)

                edge_tif0 = os.path.join(
                    outpath, "{0}_riveredge.tif".format(basename))
                pyct.npy2tif(DEM_burn_arr, DEM_tif,
                             edge_tif0, nodata=nodata)

                edge_tif2 = os.path.join(
                    outpath, "{0}_riveredge_interp.tif".format(basename))
                wbt.fill_missing_data(
                    edge_tif0, edge_tif2, filter=11, weight=2)

                DEM_burn_edge_arr = gdal_array.LoadFile(edge_tif2)
                DEM_burn_arr[interp_edge_ind] = DEM_burn_edge_arr[interp_edge_ind]
                water_mask_array[interp_edge_ind] = 14
                water_mask_array[interp_mask == 2] = 13

    return DEM_burn_arr, water_mask_array


def enforce_monotonic_rivers(DEM_tif, outpath, DEM_burn_arr, SWO_array,
                             water_mask_array, donut_array,
                             basename, SWO_cutoff=20, radius=25,
                             strict_endcap_geom=True,
                             nodata=-9999, wbt=None,
                             max_r_drop=100.0, inner_boundary=None,
                             verbose_print=False):
    """ fix non-monotonic river segments"""
    starttime = time.time()

    """set pixel distance and area thresholds for non-monotonic river segments
    from carve search radius"""
    # raised_dist = radius/2
    raised_area = np.round(np.pi * ((radius / 10.0)**2))
    s_river = [[0, 1, 0],
               [1, 1, 1],
               [0, 1, 0]]

    flatten_river_ind = np.nonzero((water_mask_array > 3)
                                   & (water_mask_array < 8))

    if len(flatten_river_ind[0]) == 0:
        return DEM_burn_arr, water_mask_array

    if inner_boundary is not None:
        bound_river_ind = np.nonzero((water_mask_array > 3)
                                    & (water_mask_array < 8)
                                     & (inner_boundary == 1))
        river_elevations = DEM_burn_arr[bound_river_ind]
        river_elev, counts = np.unique(np.round(river_elevations),
                                       return_counts=True)
        river_h_steps = river_elev[counts > 100]  # filter outliers
        if verbose_print:
            print(f"      river_h_steps = {river_h_steps}")

    else:
        river_elevations = DEM_burn_arr[flatten_river_ind]
        min_river_h = np.maximum(-3.0,
                                 np.percentile(river_elevations, 1) - 1.0)
        min_river_h = np.floor(min_river_h) - 0.5
        max_river_h = np.percentile(river_elevations, 99)
        max_river_h = np.floor(max_river_h) + 3.5
        raised_step = 0.5
        if (max_river_h - min_river_h) > 50:
            raised_step = 1.0
        if (max_river_h - min_river_h) > 100:
            raised_step = 3.0
        if (max_river_h - min_river_h) > 200:
            raised_step = 5.0
        river_h_steps = np.arange(min_river_h, max_river_h, raised_step)
        n_seg_pixels = len(flatten_river_ind[0])

        if len(river_h_steps) < 2 or n_seg_pixels < 3000:
            return DEM_burn_arr, water_mask_array
        if verbose_print:
            print(
                f'      river h = {min_river_h}/{max_river_h}, step={raised_step}')

    DEM_burn_arr0 = DEM_burn_arr.copy()
    water_mask_array0 = water_mask_array.copy()
    # interp_row_ind,interp_col_ind = [],[]
    non_mono_interp = False
    interp_mask = np.zeros_like(water_mask_array)

    left_array = np.zeros_like(SWO_array)
    left_array[flatten_river_ind] = 1
    i = 0
    for h in river_h_steps:
        if verbose_print:
            print_time(
                "     Iteration = {0}, river h = {1}".format(i, h), starttime)
        i += 1
        if np.count_nonzero(left_array[flatten_river_ind]) < 100:
            break
        diff_array = np.zeros_like(DEM_burn_arr)
        diff_array[flatten_river_ind] = DEM_burn_arr[flatten_river_ind] - h
        labeled_array = np.zeros_like(DEM_burn_arr)
        labeled_array[diff_array > 0] = 1
        labeled_array = apply_convolve_filter(
            labeled_array, np.ones((3, 3)))
        labeled_array[labeled_array > 0] = 1
        labeled_array[diff_array == 0] = 0  # re-restrict to river
        labeled_array[left_array == 0] = 0
        labeled_array, num_features = ndimage.label(
            labeled_array, structure=s_river)
        clump_ind = np.nonzero(labeled_array)

        # diff_tif = os.path.join(outpath, "{0}_diff_{1}.tif".format(basename, h))
        # pyct.npy2tif(diff_array, DEM_tif, diff_tif, nodata=0, dtype=gdal.GDT_Float32)
        # clump_tif = os.path.join(outpath, "{0}_clump_{1}.tif".format(basename, h))
        # pyct.npy2tif(labeled_array, DEM_tif, clump_tif, nodata=0, dtype=gdal.GDT_Int16)

        """remove IDs that cross boundary"""
        donut_clump_ind = donut_array[clump_ind]
        donutIDs = np.unique(labeled_array[(donut_array == 1)
                                           & (labeled_array > 0)])
        # mask out boundary crossing clumps
        donut_mask = np.isin(labeled_array, donutIDs)
        labeled_array[donut_mask] = 0
        clump_ind = np.nonzero(labeled_array)
        raised_IDs, counts = np.unique(labeled_array[clump_ind],
                                       return_counts=True)

        if inner_boundary is not None:
            inner_clump_ind = inner_boundary[clump_ind]
            raised_IDs, counts = np.unique(labeled_array[(inner_boundary == 1)
                                                         & (labeled_array > 0)],
                                           return_counts=True)
            if len(raised_IDs) < 1:
                continue

        # Filter out small raised clumps
        raised_IDs = raised_IDs[counts > raised_area]

        for clumpID in raised_IDs:
            """isolate non_monotonic sections of river"""
            clumpID_ind_temp = np.nonzero(
                labeled_array[clump_ind] == clumpID)[0]
            n_clump_ind = len(clumpID_ind_temp)
            clumpID_ind = (clump_ind[0][clumpID_ind_temp],
                           clump_ind[1][clumpID_ind_temp])

            # chip window indices
            ind0_min = clumpID_ind[0].min() - 4
            ind0_max = clumpID_ind[0].max() + 5
            ind1_min = clumpID_ind[1].min() - 4
            ind1_max = clumpID_ind[1].max() + 5
            clump_arr = labeled_array[ind0_min:ind0_max,
                                      ind1_min:ind1_max]
            clump_mask = np.zeros_like(clump_arr)
            clump_mask[clump_arr == clumpID] = 99

            """find adjacent river pixels"""
            diff_array_temp = diff_array[ind0_min:ind0_max,
                                         ind1_min:ind1_max]
            flatten_mask_temp = water_mask_array0[ind0_min:ind0_max,
                                                  ind1_min:ind1_max]
            SWO_array_temp = SWO_array[ind0_min:ind0_max,
                                       ind1_min:ind1_max]

            """check boarder to ensure clump touches 2 river banks"""
            c_border_array = apply_convolve_filter(
                clump_mask, np.ones((3, 3)))
            c_border_bank_ind = np.nonzero((clump_mask < 99)
                                          & (c_border_array > 0.0)
                                           & (flatten_mask_temp < 3))
            n_border_bank = len(c_border_bank_ind[0])

            c_border_riv_ind = np.nonzero((clump_mask < 99)
                                         & (c_border_array > 0.0)
                                          & (flatten_mask_temp > 2))
            n_border_riv = len(c_border_riv_ind[0])

            if (n_border_bank == 0) and (n_clump_ind < raised_area * 4):
                # print(f'    {clumpID} - surrounded by water ({n_border_bank}/{n_border_riv})')
                # water_buffer_ind = (adj_water_ind[0] + ind0_min,
                # adj_water_ind[1] + ind1_min)
                # interp_mask[clumpID_ind] = 3
                # non_mono_interp = True
                # water_mask_array[clumpID_ind] = 7
                continue

            """buffer clump - 3 pixels"""
            clump_buffer_mask = apply_convolve_filter(
                clump_mask, np.ones((7, 7)))
            clumpID_buffer_ind = np.nonzero(clump_buffer_mask)
            clumpID_ind_chip = np.nonzero(clump_mask)

            adj_river_ind_temp = np.nonzero((clump_mask[clumpID_buffer_ind] == 0)
                                           & (flatten_mask_temp[clumpID_buffer_ind] > 2)
                                            & (flatten_mask_temp[clumpID_buffer_ind] < 7))[0]
            adj_water_ind = (clumpID_buffer_ind[0][adj_river_ind_temp],
                             clumpID_buffer_ind[1][adj_river_ind_temp])

            """if num clump pixels is <3x num adjacent river pixels, likely a narrow
            strip between water bodies"""
            if len(clumpID_ind[0]) < (2 * len(adj_water_ind[0])):
                # print("       {0}-Narrow strip... skipping ({1}/{2})".format(clumpID,
                #                             len(clumpID_ind[0]),len(adj_water_ind[0])))
                continue

            """
            compare river bank/edge pixels to flattened river pixels. Should be >2x
            as many river bank pixels in buffered clump than river flattened pixels
            outside of raised section. Prevents skinny sections running along the edge
            of a river (riverbanks) from being lowered.
            """
            bank_buffer_ind = np.nonzero((flatten_mask_temp[clumpID_buffer_ind] < 3)
                                        & (flatten_mask_temp[clumpID_buffer_ind] > 0)
                                         & (clump_mask[clumpID_buffer_ind] == 0))[0]
            river_buffer_ind = np.nonzero((flatten_mask_temp[clumpID_buffer_ind] > 2)
                                         & (flatten_mask_temp[clumpID_buffer_ind] < 7)
                                          & (clump_mask[clumpID_buffer_ind] == 0))[0]

            if len(bank_buffer_ind) < (1.5 * len(river_buffer_ind)):
                """check SWO - if mostly water, allow to continue"""
                SWO_clump = SWO_array_temp[clumpID_ind_chip]
                if np.percentile(SWO_clump, 80) < SWO_cutoff:
                    # print("       {0}-bank/river buffer < 2... skipping ({1}/{2})".format(clumpID,
                    #                             len(bank_buffer_ind),len(river_buffer_ind)))
                    continue

            """clump eligible edge pixels"""
            edge_array = np.zeros_like(clump_arr)
            edge_array[adj_water_ind] = 1
            # only consider lower river pixels
            edge_array[diff_array_temp > 0] = 0
            labeled_edge_array, num_edge_features = ndimage.label(edge_array,
                                                                  structure=s_river)
            if num_edge_features < 2:
                # print("       {0}- too few edge features)".format(clumpID))
                continue

            minx_clump_pix = np.amin(clumpID_ind_chip[1]) + 10
            maxx_clump_pix = np.amax(clumpID_ind_chip[1]) - 10
            miny_clump_pix = np.amin(clumpID_ind_chip[0]) + 10
            maxy_clump_pix = np.amax(clumpID_ind_chip[0]) - 10

            """remove edge clumps that are too small or interior to raised river section"""
            if strict_endcap_geom:
                edge_clumps = np.unique(
                    labeled_edge_array[labeled_edge_array > 0])
                for edge_clump in edge_clumps:
                    edge_clump_ind = np.nonzero(
                        labeled_edge_array == edge_clump)
                    # adjacent river clump too small
                    if len(edge_clump_ind[0]) < 3:
                        edge_array[edge_clump_ind] = 0
                        continue

                    minx_edge_clump_pix = np.amin(edge_clump_ind[1])
                    maxx_edge_clump_pix = np.amax(edge_clump_ind[1])
                    miny_edge_clump_pix = np.amin(edge_clump_ind[0])
                    maxy_edge_clump_pix = np.amax(edge_clump_ind[0])
                    if (minx_clump_pix < minx_edge_clump_pix) and (maxx_clump_pix > maxx_edge_clump_pix):
                        if (miny_clump_pix < miny_edge_clump_pix) and (maxy_clump_pix > maxy_edge_clump_pix):
                            # print("       {0}-remove interior (bbox) edge".format(clumpID))
                            edge_array[edge_clump_ind] = 0
                            continue

                    edge_clump_test = np.zeros_like(clump_arr)
                    edge_clump_test[labeled_edge_array == edge_clump] = 1
                    edge_clump_test = ndimage.maximum_filter(
                        edge_clump_test, size=3)
                    edge_clump_test_ind = np.nonzero((edge_clump_test > 0)
                                                     & (edge_array == 0))
                    if clump_mask[edge_clump_test_ind].min() > 0:
                        # print("       {0}-remove interior edge".format(clumpID))
                        edge_array[edge_clump_test_ind] = 0

                labeled_edge_array, num_edge_features = ndimage.label(edge_array,
                                                                      structure=s_river)

            if num_edge_features > 3:
                # print("       {0}-too many exterior edge features ({1})".format(clumpID, num_edge_features))
                continue

            adj_river_ind = np.nonzero(labeled_edge_array)
            if num_edge_features < 2:
                continue

            """test distance between edge features"""
            minx_pix = np.argmin(adj_river_ind[1])
            minx_ind = np.array([adj_river_ind[0][minx_pix],
                                 adj_river_ind[1][minx_pix]])
            maxx_pix = np.argmax(adj_river_ind[1])
            maxx_ind = np.array([adj_river_ind[0][maxx_pix],
                                 adj_river_ind[1][maxx_pix]])
            miny_pix = np.argmin(adj_river_ind[0])
            miny_ind = np.array([adj_river_ind[0][miny_pix],
                                 adj_river_ind[1][miny_pix]])
            maxy_pix = np.argmax(adj_river_ind[0])
            maxy_ind = np.array([adj_river_ind[0][maxy_pix],
                                 adj_river_ind[1][maxy_pix]])
            x_dist = np.linalg.norm(maxx_ind - minx_ind)
            y_dist = np.linalg.norm(maxy_ind - miny_ind)

            """check the min max x/y are in different clumps"""
            labeled_edge_array = labeled_edge_array[adj_river_ind]
            if labeled_edge_array[minx_pix] == labeled_edge_array[maxx_pix]:
                x_dist = 0
            if labeled_edge_array[miny_pix] == labeled_edge_array[maxy_pix]:
                y_dist = 0

            """diagonal dist of clump bounding box - determine min dist between edge features"""
            topleft_ind = np.array([ind0_min, ind1_min])
            botright_ind = np.array([ind0_max, ind1_max])
            diag_dist = np.linalg.norm(botright_ind - topleft_ind)
            if strict_endcap_geom:
                dist_min = diag_dist / 2
            else:
                dist_min = diag_dist / 10
            # print("       {0}- dist_min={1:.0f}, x_dist={2:.0f}, y_dist={3:.0f}".format(
            #            clumpID,dist_min,x_dist,y_dist))

            if (x_dist < dist_min) or (y_dist < dist_min):
                continue

            """shift back to whole array"""
            adj_river_ind = (adj_river_ind[0] + ind0_min,
                             adj_river_ind[1] + ind1_min)

            raised_min_h = np.percentile(DEM_burn_arr[clumpID_ind], 10)
            raised_max_h = np.percentile(DEM_burn_arr[clumpID_ind], 95)
            adj_river_h = np.percentile(DEM_burn_arr[adj_river_ind], 20)
            r_drop = raised_min_h - adj_river_h

            # print("       {0}-river drop = {1:.2f} meters".format(clumpID, r_drop))
            if r_drop < 0.0:
                continue

            if ((raised_max_h - raised_min_h) < max_r_drop) and (left_array[clumpID_ind].min() == 2):
                # contained in previously lowered clump
                # raised_burn_add = np.minimum(2*raised_step, 4.0)
                set_value = adj_river_h  # - raised_burn_add
                DEM_burn_arr[clumpID_ind] = set_value
                if verbose_print:
                    print(f"       {clumpID}-raised river section 2nd pass...",
                          f"set to {set_value:.2f} meters")

                """remove and interpolate neighboring pixels"""
                bank_buffer_ind = (clumpID_buffer_ind[0][bank_buffer_ind] + ind0_min,
                                   clumpID_buffer_ind[1][bank_buffer_ind] + ind1_min)
                water_buffer_ind = (adj_water_ind[0] + ind0_min,
                                    adj_water_ind[1] + ind1_min)

                interp_mask[water_buffer_ind] = 2
                interp_mask[bank_buffer_ind] = 1
                non_mono_interp = True
                water_mask_array[clumpID_ind] = 7

            else:  # use a relative drop
                if verbose_print:
                    print(f"       {clumpID}-raised river section... "
                          f"lowering {r_drop:.2f} meters")

                water_buffer_ind = (adj_water_ind[0] + ind0_min,
                                    adj_water_ind[1] + ind1_min)

                interp_mask[water_buffer_ind] = 2
                non_mono_interp = True
                water_mask_array[clumpID_ind] = 6
                left_array[clumpID_ind] = 2
                DEM_burn_arr[clumpID_ind] = DEM_burn_arr[clumpID_ind] - r_drop
                overburn_ind_temp = np.nonzero(
                    diff_array[clumpID_ind] < r_drop)[0]
                overburn_ind = (clumpID_ind[0][overburn_ind_temp],
                                clumpID_ind[1][overburn_ind_temp])
                DEM_burn_arr[overburn_ind] = adj_river_h

            """fix for any raised pixels"""
            diff_clump = DEM_burn_arr - DEM_burn_arr0
            raised_ind = np.nonzero(diff_clump > 0.1)
            DEM_burn_arr[raised_ind] = DEM_burn_arr0[raised_ind]

    if non_mono_interp:

        """Interpolate removed non-monotonic pixels"""
        interp_ind = np.nonzero(interp_mask)

        DEM_burn_water_arr = DEM_burn_arr.copy()
        DEM_burn_water_arr[interp_ind] = nodata

        edge_tif0 = os.path.join(outpath, f"{basename}_non_mono.tif")
        pyct.npy2tif(DEM_burn_water_arr, DEM_tif,
                     edge_tif0, nodata=nodata)
        edge_tif2 = os.path.join(outpath, f"{basename}_non_mono_interp.tif")
        wbt.fill_missing_data(edge_tif0, edge_tif2, filter=17, weight=2)

        DEM_burn_nm_arr = gdal_array.LoadFile(edge_tif2)
        DEM_burn_arr[interp_ind] = DEM_burn_nm_arr[interp_ind]
        water_mask_array[interp_mask == 1] = 2

    return DEM_burn_arr, water_mask_array
