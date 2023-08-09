#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funtions to import, filter, reclassify and rasterize OpenStreetMap data. Designed to extract and
rasterize water layer shape files.


@author: Kimberly McCormack

Last edited on: 04/29/2021

"""
import glob
import os
import sys
import subprocess
import time
import warnings
warnings.simplefilter('ignore')

import geopandas as gpd
import numpy as np
from scipy import ndimage
import pandas as pd
from osgeo import gdal, gdal_array, ogr
from shapely.geometry import box

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.derive import apply_convolve_filter
from tools.print import print_time
from tools.files import erase_files
import tools.convert as pyct


gdal.AllRegister()
gdal.UseExceptions()
starttime = time.time()


def OSMclip(clip_tif, src_path, src_shp, outpath, waterlines_clip_shp,
            waterpolys_clip_shp, basin_clip_shp=None,
            verboseprint=True):
    """
    Find and combine all overlapping waterline and water polygon shape files,
    clip to input tif bounding box


    Parameters
    ----------
    clip_shp : path
        input shape file to clip shapes to
    src_path : path
        directory containing :
            1.  all OSM shapefiles in subdirectories named by region
            2.  a 'OSM_tiles.shp' file with region bounding boxes and names
                that correspond to the subdirectories
    outpath : path
        output directory for temp files
    waterlines_clip_shp : path
        output file for clipped and combined line features
    waterpolys_clip_shp : path
        output file for clipped and combined polygon features
    starttime : float
        time.time() starting time
    verboseprint : bool, optional
        print time stamps/processing steps?, by default True

    Returns
    -------
    gdf
        geodataframe of all reclasified water line features
    gdf
        geodataframe of all reclasified water polygon features
    """

    osm_bbox_gdf = gpd.read_file(src_shp)

    srcDS = gdal.Open(clip_tif, gdal.GA_ReadOnly)
    xshape = srcDS.RasterXSize
    yshape = srcDS.RasterYSize
    projection = srcDS.GetProjectionRef()
    geoT = srcDS.GetGeoTransform()
    xmin = geoT[0]
    ymax = geoT[3]
    xmax = xmin + geoT[1] * xshape
    ymin = ymax + geoT[5] * yshape
    srcDS = None

    clip_bbox_poly = box(xmin, ymin, xmax, ymax)
    box_df = pd.DataFrame({'name': ['bbox'],
                           'geometry': [clip_bbox_poly]})
    clip_bbox_gdf = gpd.GeoDataFrame(box_df, crs="EPSG:4326",
                                     geometry=box_df['geometry'])
    clip_bbox_gdf = pyct.buffer_gdf_simple(clip_bbox_gdf, 1e3, capstyle=3)
    clip_box_shp = os.path.join(outpath, 'osm_clip_temp.shp')
    clip_bbox_gdf.to_file(clip_box_shp)

    tiles_clip_gdf = osm_bbox_gdf.loc[osm_bbox_gdf.geometry.intersects(
        clip_bbox_poly)]
    tiles_clip_list = list(tiles_clip_gdf['index'].values)

    """load and concatenate overlapping tiles"""
    if verboseprint:
        print("       load in OpenStreetMap for {0} tiles".format(
            len(tiles_clip_list)))

    waterlines_all_gdf = None
    waterpolys_all_gdf = None

    waterlines_gdf_list = []
    waterpolys_gdf_list = []

    for i, tile in enumerate(tiles_clip_list[0::]):
        waterlines_shp = f"{src_path}/{tile}/{tile}_waterlines.shp"
        waterpolys_shp = f"{src_path}/{tile}/{tile}_waterpolygons.shp"

        """Clip tile data to shape bounding box"""
        waterlines_clip_temp_shp = waterlines_clip_shp.replace(
            '.shp', '_temp.shp')
        waterpolys_clip_temp_shp = waterpolys_clip_shp.replace(
            '.shp', '_temp.shp')

        ogr_command = "ogr2ogr -clipsrc {0} {1} {2}".format(clip_box_shp,
                                                            waterlines_clip_temp_shp,
                                                            waterlines_shp)
        subprocess.call(ogr_command, shell=True, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

        ogr_command = "ogr2ogr -clipsrc {0} {1} {2}".format(clip_box_shp,
                                                            waterpolys_clip_temp_shp,
                                                            waterpolys_shp)
        subprocess.call(ogr_command, shell=True, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)

        """Append clipped tile shapes to geodataframe"""
        try:
            waterlines_temp_gdf = gpd.read_file(waterlines_clip_temp_shp)
            if not waterlines_temp_gdf.empty:
                waterlines_gdf_list.append(waterlines_temp_gdf)
        except:
            print('could not open ', waterlines_clip_temp_shp)
            continue

        try:
            waterpolys_temp_gdf = gpd.read_file(waterpolys_clip_temp_shp)
            if not waterpolys_temp_gdf.empty:
                waterpolys_gdf_list.append(waterpolys_temp_gdf)
        except:
            continue

    if len(waterlines_gdf_list) > 1:
        waterlines_all_gdf = gpd.GeoDataFrame(pd.concat(waterlines_gdf_list,
                                                        ignore_index=True))
    elif len(waterlines_gdf_list) == 1:
        waterlines_all_gdf = waterlines_gdf_list[0]

    if len(waterpolys_gdf_list) > 1:
        waterpolys_all_gdf = gpd.GeoDataFrame(pd.concat(waterpolys_gdf_list,
                                                        ignore_index=True))
    elif len(waterpolys_gdf_list) == 1:
        waterpolys_all_gdf = waterpolys_gdf_list[0]

    if basin_clip_shp is not None:
        clip_gdf = gpd.read_file(basin_clip_shp)
        clip_poly = list(clip_gdf.geometry)[0]  # buffer?

    if waterlines_all_gdf is not None:
        if not waterlines_all_gdf.empty:
            waterlines_all_gdf.reset_index()
            if basin_clip_shp is not None:
                waterlines_all_gdf = gpd.clip(waterlines_all_gdf, clip_poly)
            if not waterlines_all_gdf.empty:
                waterlines_all_gdf = waterlines_all_gdf.set_crs("EPSG:4326")
                waterlines_all_gdf.to_file(waterlines_clip_shp)
            else:
                waterlines_all_gdf = None

    if waterpolys_all_gdf is not None:
        if not waterpolys_all_gdf.empty:
            waterpolys_all_gdf.reset_index()
            if basin_clip_shp is not None:
                waterpolys_all_gdf = gpd.clip(waterpolys_all_gdf, clip_poly)
            if not waterpolys_all_gdf.empty:
                waterpolys_all_gdf = waterpolys_all_gdf.set_crs("EPSG:4326")
                waterpolys_all_gdf.to_file(waterpolys_clip_shp)
            else:
                waterpolys_all_gdf = None

    """Erase temp files"""
    erase_files(outpath, search='*temp*')

    return waterlines_all_gdf, waterpolys_all_gdf


def OSMrasterize(waterlines_shp, waterpolys_shp, stencil_tif, waterlines_gdf, waterpolys_gdf,
                 waterlines_tif, waterpolys_tif):
    """
    Rasterize water polygons created from Open Street map data. Takes both line and polygon
    water features to form catagorical rasters with the same projections and geotransform as
    the input stencil_tif.

    Parameters
    ----------
    waterlines_shp : path
        shapefile of water line features
    waterpolys_shp : path
        shapefile of water polygon/multipolygon features
    stencil_tif : path
        .tif raster for shape files to be projected/resampled on to
    waterlines_gdf : gdf
        Geodataframe of line features - only used to determine if no lines exist
    waterpolys_gdf : gdf
        Geodataframe of polygon features - only used to determine if no polygons exist
    waterlines_tif : path
        output path for rasterized line features
    waterpolys_tif : path
        output path for rasterized polygon features

    """

    srcDS = gdal.Open(stencil_tif, gdal.GA_ReadOnly)
    xshape = srcDS.RasterXSize
    yshape = srcDS.RasterYSize
    geoT = srcDS.GetGeoTransform()
    xmin = geoT[0]
    ymax = geoT[3]
    xmax = xmin + geoT[1] * xshape
    ymin = ymax + geoT[5] * yshape
    srcDS = None

    if waterlines_gdf is not None:
        rstr_command = "gdal_rasterize -a watertype -ot Byte -ts {0} {1}\
            -te {2} {3} {4} {5} {6} {7}".format(xshape, yshape, xmin,
                                                ymin, xmax, ymax, waterlines_shp, waterlines_tif)
        subprocess.call(rstr_command, shell=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if waterpolys_gdf is not None:
        rstr_command = "gdal_rasterize -a watertype -ot Byte -ts {0} {1}\
            -te {2} {3} {4} {5} {6} {7}".format(xshape, yshape, xmin,
                                                ymin, xmax, ymax, waterpolys_shp, waterpolys_tif)
        subprocess.call(rstr_command, shell=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def OSMreclassify(clip_tif, src_path, outpath, waterlines_clip_shp,
                  waterpolys_clip_shp, starttime, basin_clip_shp=None,
                  verboseprint=True):
    """
    Find and compbine all overlapping waterline and water polygon shape files,
    clip to input tif bounding box, and add a integer "watertype' field based on
    existing fclass field to prepare shapes to be rasterized

    Parameters
    ----------
    clip_tif : path
        input tif to clip shapes to
    src_path : path
        directory containing :
            1.  all OSM shapefiles in subdirectories named by region
            2.  a 'countries_bbox.shp' file with region bounding boxes and names
                that correspond to the subdirectories
    outpath : path
        output directory for temp files
    waterlines_clip_shp : path
        output file for clipped and combined line features
    waterpolys_clip_shp : path
        output file for clipped and combined polygon features
    starttime : float
        time.time() starting time
    verboseprint : bool, optional
        print time stamps/processing steps?, by default True

    Returns
    -------
    gdf
        geodataframe of all reclasified water line features
    gdf
        geodataframe of all reclasified water polygon features
    """

    country_bbox_shp = os.path.join(src_path, 'countries_bbox.shp')

    srcDS = gdal.Open(clip_tif, gdal.GA_ReadOnly)
    xshape = srcDS.RasterXSize
    yshape = srcDS.RasterYSize
    projection = srcDS.GetProjectionRef()
    geoT = srcDS.GetGeoTransform()
    xmin = geoT[0]
    ymax = geoT[3]
    xmax = xmin + geoT[1] * xshape
    ymin = ymax + geoT[5] * yshape
    srcDS = None

    country_bbox_gdf = gpd.read_file(country_bbox_shp)
    basin_box_poly = box(xmin, ymin, xmax, ymax)
    basin_box_df = pd.DataFrame({'name': ['bbox'],
                                 'geometry': [basin_box_poly]})
    basin_box_gdf = gpd.GeoDataFrame(basin_box_df, crs="EPSG:4326",
                                     geometry=basin_box_df['geometry'])
    basin_box_shp = os.path.join(outpath, 'basin_bbox.shp')
    basin_box_gdf.to_file(basin_box_shp)

    countries_clip_gdf = country_bbox_gdf.loc[country_bbox_gdf.geometry.intersects(
        basin_box_poly)]
    countries_clip_list = list(countries_clip_gdf['country'].values)

    """load and concatenate overlapping countries"""
    if verboseprint:
        print_time("       load in OpenStreetMap for {0} regions".format(
            len(countries_clip_list)), starttime)

    waterlines_all_gdf = None
    waterpolys_all_gdf = None

    waterlines_gdf_list = []
    waterpolys_gdf_list = []

    for i, country in enumerate(countries_clip_list[0::]):
        waterlines_shp = os.path.join(src_path, country,
                                      '{0}_waterlines.shp'.format(country))
        waterpolys_shp = os.path.join(src_path, country,
                                      '{0}_waterpolygons.shp'.format(country))

        """Clip country data to shape bounding box"""
        waterlines_clip_temp_shp = waterlines_clip_shp.replace(
            '.tif', '_temp.tif')
        waterpolys_clip_temp_shp = waterpolys_clip_shp.replace(
            '.tif', '_temp.tif')

        ogr_command = "ogr2ogr -clipsrc {0} {1} {2}".format(basin_box_shp,
                                                            waterlines_clip_temp_shp,
                                                            waterlines_shp)
        subprocess.call(ogr_command, shell=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        ogr_command = "ogr2ogr -clipsrc {0} {1} {2}".format(basin_box_shp,
                                                            waterpolys_clip_temp_shp,
                                                            waterpolys_shp)
        subprocess.call(ogr_command, shell=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        """Append clipped country shapes to geodataframe"""
        try:
            waterlines_temp_gdf = gpd.read_file(waterlines_clip_temp_shp)
            if not waterlines_temp_gdf.empty:
                waterlines_gdf_list.append(waterlines_temp_gdf)
        except:
            print('could not open ', waterlines_clip_temp_shp)
            continue

        try:
            waterpolys_temp_gdf = gpd.read_file(waterpolys_clip_temp_shp)
            if not waterpolys_temp_gdf.empty:
                waterpolys_gdf_list.append(waterpolys_temp_gdf)
        except:
            continue

    if len(waterlines_gdf_list) > 1:
        waterlines_all_gdf = gpd.GeoDataFrame(pd.concat(waterlines_gdf_list,
                                                        ignore_index=True))
    elif len(waterlines_gdf_list) == 1:
        waterlines_all_gdf = waterlines_gdf_list[0]

    if len(waterpolys_gdf_list) > 1:
        waterpolys_all_gdf = gpd.GeoDataFrame(pd.concat(waterpolys_gdf_list,
                                                        ignore_index=True))
    elif len(waterpolys_gdf_list) == 1:
        waterpolys_all_gdf = waterpolys_gdf_list[0]

    if basin_clip_shp is not None:
        clip_gdf = gpd.read_file(basin_clip_shp)
        clip_poly = list(clip_gdf.geometry)[0]

    if waterlines_all_gdf is not None:
        if not waterlines_all_gdf.empty:
            waterlines_all_gdf.reset_index()
            if basin_clip_shp is not None:
                waterlines_all_gdf = gpd.clip(waterlines_all_gdf, clip_poly)
            if not waterlines_all_gdf.empty:
                waterlines_all_gdf.to_file(waterlines_clip_shp)
            else:
                waterlines_all_gdf = None

    if waterpolys_all_gdf is not None:
        if not waterpolys_all_gdf.empty:
            waterpolys_all_gdf.reset_index()
            if basin_clip_shp is not None:
                waterpolys_all_gdf = gpd.clip(waterpolys_all_gdf, clip_poly)
            if not waterpolys_all_gdf.empty:
                waterpolys_all_gdf.to_file(waterpolys_clip_shp)
            else:
                waterpolys_all_gdf = None

    """Add integer field to be used as raster value"""
    type_map_poly = {'reservoir': 1,
                     'river': 2,
                     'riverbank': 2,
                     'water': 3,
                     'wetland': 4}

    type_map_lines = {'river': 5,
                      'stream': 6,
                      'canal': 7,
                      'drain': 8}

    driver = ogr.GetDriverByName('ESRI Shapefile')

    """Add attribute to polygons"""
    if (waterpolys_all_gdf is not None) and (not waterpolys_all_gdf.empty):
        dataSource = driver.Open(waterpolys_clip_shp, 1)  # 1 is read/write
        fldDef_type = ogr.FieldDefn(
            'watertype', ogr.OFTInteger)  # define field
        layer = dataSource.GetLayer()
        layer.CreateField(fldDef_type)

        for feat in layer:
            fclass = feat.GetField('fclass')
            water_type = type_map_poly.get(fclass)
            feat.SetField('watertype', water_type)
            layer.SetFeature(feat)
        dataSource = None
        waterpolys_all_gdf = gpd.read_file(waterpolys_clip_shp)

    """Add attribute to lines"""
    if (waterlines_all_gdf is not None) and (not waterlines_all_gdf.empty):
        dataSource = driver.Open(waterlines_clip_shp, 1)  # 1 is read/write
        fldDef_type = ogr.FieldDefn('watertype', ogr.OFTInteger)
        layer = dataSource.GetLayer()
        layer.CreateField(fldDef_type)

        for feat in layer:
            fclass = feat.GetField('fclass')
            water_type = type_map_lines.get(fclass)
            feat.SetField('watertype', water_type)
            layer.SetFeature(feat)
        dataSource = None
        waterlines_all_gdf = gpd.read_file(waterlines_clip_shp)

        if waterlines_all_gdf.empty:
            waterlines_all_gdf = None

    if waterpolys_all_gdf is not None:
        if waterpolys_all_gdf.empty:
            waterpolys_all_gdf = None

    return waterlines_all_gdf, waterpolys_all_gdf


def OSM_unpack(src_path, target_path, overwrite=False):
    """Unpack and clean .shp.zip and .pbf Open Street Map files. Only keep waterlines
    and water polygon files. Create a shapefile containg bounding boxes for each of
    the region subdriectories in 'target_path'.

    Parameters
    ----------
    src_path : path
        location of raw OSM .shp.zip and .pbf files
    target_path : path
        out path for cleaned shape files
    """

    pbf_files = glob.glob(os.path.join(src_path, "*.pbf"))
    for i, pfile in enumerate(pbf_files[0::]):

        fname_whole = os.path.basename(pfile)
        country = fname_whole.split('-latest')[0]

        print_time('extracting {0}'.format(fname_whole), starttime)

        DSdir_temp = os.path.join(src_path, country)
        DSdir = os.path.join(target_path, country)

        lines_shp = os.path.join(DSdir_temp, '{0}_lines.shp'.format(country))
        polygons_shp = os.path.join(
            DSdir_temp, '{0}_polygons.shp'.format(country))

        waterlines_shp = os.path.join(
            DSdir, '{0}_waterlines.shp'.format(country))
        waterpolys_shp = os.path.join(
            DSdir, '{0}_waterpolygons.shp'.format(country))

        if not os.path.exists(DSdir_temp):
            os.makedirs(DSdir_temp)
        if not os.path.exists(DSdir):
            os.makedirs(DSdir)

        """Convert and filter water line features"""
        sql_comm = "-sql 'select osm_id,name,waterway,other_tags from lines where waterway is not null'"
        command = 'ogr2ogr -f "ESRI Shapefile" -skipfailures {2} {0} {1}'.format(
            lines_shp, pfile, sql_comm)

        if not overwrite and not os.path.exists(lines_shp):
            subprocess.call(command, shell=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print_time('   line features converted', starttime)

        if not overwrite and not os.path.exists(waterlines_shp):
            lines_gdf = gpd.read_file(lines_shp)
            waterlines_gdf = lines_gdf.loc[(lines_gdf['waterway'] == 'canal')
                                           | (lines_gdf['waterway'] == 'drain')
                                           | (lines_gdf['waterway'] == 'river')
                                           | (lines_gdf['waterway'] == 'riverbank')
                                           | (lines_gdf['waterway'] == 'stream')]

            # rename column
            waterlines_gdf = waterlines_gdf.rename(
                columns={"waterway": "fclass"})

            # save to shapefile
            waterlines_gdf.to_file(waterlines_shp)
            print_time('   line features filtered and saved', starttime)

        """Convert and filter water polygon features"""
        sql_comm = "-sql 'select osm_id,osm_way_id,name,natural,landuse,other_tags from multipolygons \
            where natural is not null or landuse is not null'"
        command = 'ogr2ogr -f "ESRI Shapefile" -skipfailures {2} {0} {1}'.format(
            polygons_shp, pfile, sql_comm)

        if not overwrite and not os.path.exists(polygons_shp):
            subprocess.call(command, shell=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print_time('   polygon features converted', starttime)

        if not overwrite and not os.path.exists(waterpolys_shp):
            polys_gdf = gpd.read_file(polygons_shp)
            waterpolys_gdf = polys_gdf.loc[(polys_gdf['landuse'] == 'reservoir')
                                           | (polys_gdf['natural'] == 'water')
                                           | (polys_gdf['natural'] == 'wetland')].copy()

            print_time('   polygon features filtered', starttime)

            """ map water types to single fclass column"""
            waterpolys_gdf['fclass'] = waterpolys_gdf.apply(
                lambda row: label_water(row), axis=1)
            waterpolys_gdf = waterpolys_gdf.drop(
                columns=['landuse', 'natural'])
            waterpolys_gdf.to_file(waterpolys_shp)
            print_time('   polygon features reclassified and saved', starttime)

            """Delete temp files"""
            delfiles = glob.glob(os.path.join(DSdir_temp, '*'))
            for df in delfiles:
                os.remove(df)

    zip_files = glob.glob(os.path.join(src_path, "*.zip"))
    for i, zfile in enumerate(zip_files[0::]):

        fname_whole = os.path.basename(zfile)
        country = fname_whole.split('-latest')[0]
        DSdir = os.path.join(target_path, country)

        unzip_com = "unzip {0} -d {1}".format(zfile, DSdir)
        if not os.path.exists(DSdir):
            print('Unzipping {0}'.format(country))
            subprocess.call(unzip_com, shell=True, stdout=subprocess.DEVNULL)
            print_time('   {0} uzipped'.format(country), starttime)

            """Rename water files"""
            waterpolys = glob.glob(os.path.join(
                target_path, country, '*water_*'))
            waterlines = glob.glob(os.path.join(
                target_path, country, '*waterways_*'))
            for wp in waterpolys:
                basename = os.path.splitext(os.path.basename(wp))[0]
                new_path = wp.replace(
                    basename, '{0}_waterpolygons'.format(country))
                os.rename(wp, new_path)

            for wl in waterlines:
                basename = os.path.splitext(os.path.basename(wl))[0]
                new_path = wl.replace(
                    basename, '{0}_waterlines'.format(country))
                os.rename(wl, new_path)

            """Delete everything else"""
            delfiles = glob.glob(os.path.join(target_path, country, 'gis_*'))
            for df in delfiles:
                os.remove(df)
            print_time(
                '   {0} --  non-water files deleted'.format(country), starttime)

    print_time('ALL DONE', starttime)


def OSM_build_bbox_shpfile(src_path):
    """
    Create a shapefile containg bounding boxes for each of
    the region subdirectories in 'src_path'.

    Parameters
    ----------
    src_path : path
        location of cleaned shape files

    """

    """Create shapes around all water features"""
    country_bbox_shp = os.path.join(src_path, 'countries_bbox.shp')
    country_dirs = glob.glob(os.path.join(src_path, "*/"))

    print_time('Build catalog for {0} regions'.format(
        len(country_dirs)), starttime)

    countrys = []
    bbox_geometries = []

    for cdir in country_dirs[0::]:
        country = os.path.basename(os.path.dirname(cdir))

        print_time('   Adding {0} to shape catalogue'.format(
            country), starttime)

        waterlines_shp = os.path.join(
            cdir, '{0}_waterlines.shp'.format(country))
        waterpolys_shp = os.path.join(
            cdir, '{0}_waterpolygons.shp'.format(country))

        waterlines_gdf = gpd.read_file(waterlines_shp)
        waterpolys_gdf = gpd.read_file(waterpolys_shp)

        if waterlines_gdf.empty and waterpolys_gdf.empty:
            continue

        water_all_gdf = gpd.GeoDataFrame(
            pd.concat([waterlines_gdf, waterpolys_gdf]), crs="EPSG:4326")

        bounds = water_all_gdf.total_bounds
        minx, miny = bounds[0], bounds[1]
        maxx, maxy = bounds[2], bounds[3]

        if (maxx - minx) > 180.0:
            print('    ...bounding box crosses antimeridian')
            bounds_gdf = water_all_gdf.bounds
            minx_all = bounds_gdf.minx.values
            minx_pos = minx_all[minx_all > 0]
            minx = np.min(minx_pos)

            maxx_all = bounds_gdf.minx.values
            maxx_neg = maxx_all[maxx_all < 0]
            maxx = np.max(maxx_neg)

            water_all_bbox1 = box(minx, miny, 180.0, maxy)
            water_all_bbox2 = box(-180.0, miny, maxx, maxy)

            countrys.append(country)
            bbox_geometries.append(water_all_bbox1)
            countrys.append(country)
            bbox_geometries.append(water_all_bbox2)

        else:
            water_all_bbox = box(minx, miny, maxx, maxy)
            countrys.append(country)
            bbox_geometries.append(water_all_bbox)

    country_df = pd.DataFrame({'country': countrys,
                               'geometry': bbox_geometries})
    country_gdf = gpd.GeoDataFrame(country_df, crs="epsg:4326",
                                   geometry=country_df['geometry'])

    country_gdf.to_file(country_bbox_shp)

    print_time('ALL DONE'.format(country), starttime)


def label_water(row):

    if row['natural'] == 'wetland':
        return 'wetland'

    if row['landuse'] == 'reservoir':
        return 'reservoir'

    if row['natural'] == 'water' and row['other_tags'] is not None:

        if 'river' in row['other_tags']:
            return 'river'

        if 'riverbank' in row['other_tags']:
            return 'riverbank'

        if 'aquaculture' in row['other_tags']:
            return 'aquaculture'

        if 'pond' in row['other_tags']:
            return 'pond'

        if 'reservoir' in row['other_tags']:
            return 'reservoir'

    if row['natural'] == 'water':
        return 'water'

    else:
        return 'NULL'
