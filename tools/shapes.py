#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to extract intersecting shapes and build shape files based on intersecting shapes
"""

import os
import sys
import time
import warnings
import subprocess
import numpy as np
warnings.simplefilter('ignore')

import geopandas as gpd
import pandas as pd
import shapely.geometry as sgeom
from shapely import geometry
from shapely.geometry import Polygon, box
from shapely.ops import polygonize, unary_union
from osgeo import gdal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.convert as pyct
from tools.print import print_time


def clip_shpfile(clip_shp_file, in_shp_file, out_shp_file, condition='intersect',
                 prj="EPSG:4326"):
    """Extract shapes from a shape file within a given shape file, and saves.

    Parameters
    ----------
    clip_shp_file : path
        input shapefile of clipping shape
    in_shp_file : path
        shape file of input shapes to clip
    out_shp_file : path
        output shape file
    condition : str, optional
        'within or 'interesect' as condition for selecting shapes, by default 'intersect'
    """

    input_gdf = gpd.read_file(in_shp_file)
    clip_gdf = gpd.read_file(clip_shp_file)
    clip_polygon = list(clip_gdf.geometry)[0]

    if condition == 'intersect':
        out_gdf = input_gdf.loc[input_gdf.geometry.intersects(clip_polygon)]

    if condition == 'within':
        out_gdf = input_gdf.loc[input_gdf.geometry.within(clip_polygon)]

    out_gdf.crs = prj
    out_gdf.to_file(out_shp_file)


def clip_shpfile_gdf(in_shp_file, clip_gdf, condition='intersect', prj="EPSG:4326"):
    """Extract shapes from a shape file within a given shape file, and saves.

    Parameters
    ----------
    clip_shp_file : path
        input shapefile of clipping shape
    in_shp_file : path
        shape file of input shapes to clip
    out_shp_file : path
        output shape file
    condition : str, optional
        'within or 'interesect' as condition for selecting shapes, by default 'intersect'
    """

    input_gdf = gpd.read_file(in_shp_file)
    #clip_gdf = gpd.read_file(clip_shp_file)
    clip_polygon = list(clip_gdf.geometry)[0]

    if condition == 'intersect':
        out_gdf = input_gdf.loc[input_gdf.geometry.intersects(clip_polygon)]

    if condition == 'within':
        out_gdf = input_gdf.loc[input_gdf.geometry.within(clip_polygon)]

    out_gdf.crs = prj

    return out_gdf
    
    
def clip_shp(clip_shp_file, input_shpfile, condition='intersect', prj="EPSG:4326"):
    """Extract shapes from a shape file within a given shape file, and saves.

    Parameters
    ----------
    clip_shp_file : path
        input shapefile of clipping shape
    in_shp_file : path
        shape file of input shapes to clip
    out_shp_file : path
        output shape file
    condition : str, optional
        'within or 'interesect' as condition for selecting shapes, by default 'intersect'
    """

    input_gdf = gpd.read_file(input_shpfile)
    clip_gdf = gpd.read_file(clip_shp_file)
    clip_polygon = list(clip_gdf.geometry)[0]

    if condition == 'intersect':
        out_gdf = input_gdf.loc[input_gdf.geometry.intersects(clip_polygon)]

    if condition == 'within':
        out_gdf = input_gdf.loc[input_gdf.geometry.within(clip_polygon)]

    out_gdf.crs = prj

    return out_gdf


def extract_basins_ogr(clip_shp_file, in_shp_file, out_shp_file):
    """Extract shapes from a shape file within another shapefile using an ogr2ogr
    command line call. Prints out a progress ticker while waiting, which is useful
    if you need to keep an interactive compute job or ssh connection alive. Otherwise,
    this has identical behavior to extract_basins() function above.

    Parameters
    ----------
    clip_shp_file : path
        input shapefile of clipping shape
    in_shp_file : path
        shape file of input shapes to clip
    out_shp_file : path
        output shape file

    ogr2ogr -clipsrc clip_polygon.shp output.shp input.shp

    """
    print('    Use ogr2ogr command line call', end="")
    sub_call = "ogr2ogr -skipfailures -clipsrc {0} {1} {2}".format(
        clip_shp_file, out_shp_file, in_shp_file)
    process = subprocess.Popen(sub_call, shell=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # stdout=subprocess.PIPE)

    while process.poll() is None:
        print(".", end="")
        sys.stdout.flush()
        time.sleep(30)

    print(" ")
    sys.stdout.flush()


def extract_single_basin(shp_file, shpID, out_shp_file, searchfield='HYBAS_ID'):
    """
    Extract a single basin from a shape file based on an attribute value

    Parameters
    ----------
    shp_file : path
        input shape file
    shpID : int
        hydrobasins basin ID
    out_shp_file : path
        output shpae file
    searchfield : str, optional
        attribute to search on, by default 'HYBAS_ID'
    """

    basins_gdf = gpd.read_file(shp_file)
    single_gdf = basins_gdf[basins_gdf[searchfield] == shpID].copy()
    single_gdf.to_file(out_shp_file)


def extract_single_basin_gdf(shp_file, shpID, searchfield='HYBAS_ID'):
    """
    Extract a single basin from a shape file based on an attribute value

    Parameters
    ----------
    shp_file : path
        input shape file
    shpID : int
        hydrobasins basin ID
    out_shp_file : path
        output shpae file
    searchfield : str, optional
        attribute to search on, by default 'HYBAS_ID'
    """

    basins_gdf = gpd.read_file(shp_file)
    single_gdf = basins_gdf[basins_gdf[searchfield] == shpID].copy()

    return single_gdf


def clip_shp_ogr(clip_shp, output_shp, input_shp):

    ogr_command = "ogr2ogr -clipsrc {0} {1} {2}".format(
        clip_shp, output_shp, input_shp)
    subprocess.call(ogr_command, shell=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def raster_bounds_gdf(path):
    """Create a geodataframe from the bounds of a raster file"""

    raster = gdal.Open(path)
    ulx, xres, xskew, uly, yskew, yres = raster.GetGeoTransform()
    lrx = ulx + (raster.RasterXSize * xres)
    lry = uly + (raster.RasterYSize * yres)
    bbox = [box(lrx, lry, ulx, uly)]

    gdf = gpd.GeoDataFrame(geometry=bbox, crs="EPSG:4326")

    return gdf


def pathlist_bbox_gdf(path_list):
    """Create a geodataframe from the bounds of a list of raster files"""

    tile_ids = [os.path.splitext(os.path.basename(tif))[0]
                for tif in path_list]
    geom_list = []
    for path in path_list:
        raster = gdal.Open(path)
        ulx, xres, xskew, uly, yskew, yres = raster.GetGeoTransform()
        lrx = ulx + (raster.RasterXSize * xres)
        lry = uly + (raster.RasterYSize * yres)
        bbox = [box(lrx, lry, ulx, uly)]
        geom_list.append(bbox)

    df = pd.dataframe({"TILE_ID": tile_ids,
                       "geometry": geom_list})
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    return gdf


def close_holes_gdf(in_gdf):
    """close any holes in basin polygon"""
    basin_poly = list(in_gdf.geometry)[0]
    if basin_poly.geom_type == 'MultiPolygon':
        basin_gdf_fix = in_gdf.explode()
        basin_gdf_fix['area'] = basin_gdf_fix['geometry'].to_crs('epsg:6933')\
            .map(lambda p: p.area / 10**6)

        for index, row in basin_gdf_fix.iterrows():  # close hole within each polygon
            test_basin_gdf = basin_gdf_fix.loc[[index], :].copy()
            test_basin_gdf['geometry'] = test_basin_gdf.geometry.apply(
                lambda p: close_holes(p))
            basin_gdf_fix.loc[[index], :] = test_basin_gdf

        basin_gdf_fix['group'] = 1
        in_gdf = basin_gdf_fix.dissolve(by='group')

    elif basin_poly.geom_type == 'Polygon':
        in_gdf['geometry'] = in_gdf.geometry.apply(lambda p: close_holes(p))

    return in_gdf


def fix_invalid_geom(in_gdf, fix_w_buffer=False):

    invalid_flag = False

    if not in_gdf.empty:
        input_invalid_gdf = in_gdf.loc[~in_gdf['geometry'].is_valid, :].copy()
        if not input_invalid_gdf.empty:
            invalid_flag = True
            if fix_w_buffer:
                in_gdf['geometry'] = in_gdf['geometry'].buffer(0)
            else:
                """A WHOLE bunch of nonsense to handle crossing polygon gemetries"""
                input_valid_gdf = in_gdf.loc[in_gdf['geometry'].is_valid, :].copy(
                )
                new_geom = []
                """disassemble the polygons and re-draw from the collection of lines"""
                for index, row in input_invalid_gdf.iterrows():  # Looping over all polygons
                    poly_ls = row['geometry'].boundary
                    mls = unary_union(poly_ls)
                    polygons = polygonize(mls)
                    valid_poly = sgeom.MultiPolygon(polygons)
                    new_geom.append(valid_poly)
                input_invalid_gdf['geometry'] = new_geom
                in_gdf = input_valid_gdf.append(input_invalid_gdf)

    in_gdf['geometry'] = in_gdf['geometry'].buffer(0)

    return in_gdf, invalid_flag


def close_holes(poly: Polygon) -> Polygon:
    """
    Close polygon holes by limitation to the exterior ring.

    """
    if poly.interiors:
        return Polygon(list(poly.exterior.coords))
    else:
        return poly

def calculate_area(gdf):
    gdf = pyct.buffer_gdf_simple(gdf, 1.0)
    gdf['area'] = gdf['geometry']\
        .to_crs('epsg:6933').map(lambda p: p.area / 10**6)

    return gdf
