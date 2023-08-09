# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:59:37 2021

@author: Heather Levin
"""

from osgeo import gdal, gdal_array
import os
import sys
import numpy as np
import geopandas as gpd
import glob
import time
import subprocess
from scipy.ndimage import label, generate_binary_structure
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points
start_time = time.time()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from post_processing.set_nodata import set_vrt_nodata
from tools.convert import tifdowncast, npy2tif, create_cutline_raster, buffer_gdf
import tools.convert as pyct


def reinforce_sinks(sink_gdf, input_gdf, dem_tif, nodata=-9999):
    """
    Use buffered basin shapefiles to identify any that intersect points
    in a shapefile with sinks. For those intersecting, set the sink pixel
    in the DEM to nodata using the shapefile.

    Parameters
    ----------
    sink_gdf : geodataframe
        geopandas point geodataframe with sinks for entire project.
    basin_buffer_shp : path
        path to buffered basin shapefile
    input_gdf : geodataframe
        gdf of unbuffered basin
    demName : string
        name of the DEM file to modify
    outnodata : integer
        nodata value for output raster

    Returns
    -------
    None.

    """

    input_gdf = input_gdf.explode()
    input_gdf['group'] = 1
    input_gdf = input_gdf.dissolve(by='group')

    # apply a half-pixel buffer to the basin polygon
    input_buffer_gdf = buffer_gdf(input_gdf, 6.0)

    # determine whether any sink points overlap with the basin
    sinks_in_basin = gpd.tools.sjoin(sink_gdf, input_buffer_gdf, how="left")
    sinks_in_basin = sinks_in_basin.dropna()
    sinks_in_basin = sinks_in_basin.drop("index_right", axis=1)

    # start with the assumption that there is no sink within the basin's buffer
    sink_in_buffer = False
    sink_distance = 0.0

    if len(sinks_in_basin) > 0:

        # read in conditioned DEM as array and find geotransform
        dem_array = gdal_array.LoadFile(dem_tif)
        inDEM = gdal.Open(dem_tif, gdal.GA_ReadOnly)
        in_band = inDEM.GetRasterBand(1)
        srcNodata = in_band.GetNoDataValue()
        dem_array[dem_array == srcNodata] = np.nan
        basin_gt = inDEM.GetGeoTransform()
        xmin, dx, rx, ymax, ry, dy = basin_gt
        inDEM = None

        # Iterate through sinks within the basin
        sinks_in_basin = sinks_in_basin.reset_index()
        sink_distance_all = []
        for sink_id in sinks_in_basin["id"].values:

            """
            If the overlapping sink is outside the basin (within the buffer zone),
            set the adjacent pixels to nodata as well as the sink pixel. This
            ensures that the sink is maintained during mosaicking if there is a
            slight offset between pixels among the basin DEMs.
            """
            point_gdf = sinks_in_basin[sinks_in_basin["id"] == sink_id].copy()
            sink_inner = gpd.tools.sjoin(point_gdf, input_gdf, how="left")
            sink_inner = sink_inner.dropna()
            n_sinks_inner = len(sink_inner)

            if n_sinks_inner > 0:
                # set sink pixel to nodata in array
                dem_array = set_nodata_points(point_gdf, dem_array,
                                              dx, dy, xmin, ymax, nodata)
            else:
                print("     Sinks found within basin buffer for: ",
                      os.path.split(os.path.basename(dem_tif))[1])

    return dem_array


def build_master_sink_gdf(in_dir, input_sinks_shp=None):
    """
    Combines sink shapefiles for individual level 12 basins into a master
    file for the project.

    Parameters
    ----------
    in_dir : path
        path to directory containing level 12 basin folders.

    Returns
    -------
    sink_points : geopandas geodataframe
        gdf containing all sink points for the project.

    """
    # create an empty gdf to fill with sink points
    # sink_points = gpd.GeoDataFrame(crs={'init': 'epsg:4326'})  # old syntax
    sink_points_gdf = gpd.GeoDataFrame(crs="EPSG:4326")

    # find the sink for each basin and add it to the point gdf
    basin_shp_list = glob.glob(in_dir + '/**/sink.shp', recursive=True)
    if len(basin_shp_list) > 0:
        for i, basin_sink_shp in enumerate(basin_shp_list):
            sink_pt = gpd.read_file(basin_sink_shp)
            #sink_pt['id'] = i
            sink_points_gdf = sink_points_gdf.append(sink_pt)


        # Add in input sinks and assign ID number
        if input_sinks_shp is not None:
            input_sinks_gdf = gpd.read_file(input_sinks_shp)
            sink_points_gdf = gpd.GeoDataFrame(pd.concat(
                [sink_points_gdf, input_sinks_gdf], ignore_index=True))
            
            num_sinks = len(sink_points_gdf.index)
            sink_id_list = np.arange(1, num_sinks+1)
            sink_points_gdf['id'] = sink_id_list

        # save sink points to shapefile in the main project folder
        sink_points_shp = os.path.join(in_dir, "sinks.shp")
        sink_points_gdf.to_file(sink_points_shp)
   

    else:
        sink_points_gdf = None
        sink_points_shp = None

    return sink_points_gdf, sink_points_shp


def set_nodata_points(point_gdf, raster_array, dx, dy, xmin, ymax,
                      nodata, set_buffer=False):
    """
    Set raster pixels to a nodata value at points defined by a point gdf.

    Parameters
    ----------
    point_gdf : geopandas geodataframe
        gdf containing a single sink point.
    raster_array : array
        Raster values as an array.
    dx : float
        x pixel resolution.
    dy : float
        y pixel resolution.
    xmin : float
        upper left x coordinate for raster.
    ymax : float
        upper left y coordinate for raster.
    nodata : integer
        Nodata value to set
    set_buffer : boolean
        Option to set the sink pixel's immediate neighbors to nodata as well

    Returns
    -------
    None.

    """

    x_coord = point_gdf.geometry[0].coords[0][0]
    y_coord = point_gdf.geometry[0].coords[0][1]
    col, row = coords_to_pixel(x_coord, y_coord, dx, dy, xmin, ymax)

    raster_array[row][col] = nodata

    return raster_array


def get_xy(r, c, gt):
    """
    Function to geolocate the x,y center coordinates associated with any given
    raster cell (along with the elevation value)

    Parameters
    ----------
    r : integer
        row index.
    c : integer
        column index.

    Returns
    -------
    x, y coordinates

    """
    '''Get (x, y) raster centre coordinate at row, column'''
    x0, dx, rx, y0, ry, dy = gt
    return(x0 + c * dx + dx / 2.0, y0 + r * dy + dy / 2.0)


def coords_to_pixel(x, y, dx, dy, xmin, ymax):
    """
    Convert x and y coordinates to row and column indices

    Parameters
    ----------
    x : float
        x coordinate.
    y : float
        y coordinate.
    dx : float
        x resolution.
    dy : float
        y resolution.
    xmin : float
        upper left x coordinate for raster.
    ymax : float
        upper left y coordinate for raster.

    Returns
    -------
    col : integer
        column index.
    row : integer
        row index.

    """
    col = int((x - xmin) / dx)
    row = int((y - ymax) / dy)
    return col, row


def find_sink(basin_id, basin_vrt_path, cutline_shp, SWO_vrt_path=None,
              nodata=99999, id=0):
    """
    Identify a sink point within a basin and return a gdf with that point.

    Parameters
    ----------
    basin_id : integer
        HydroBASINS ID
    basin_vrt_path : path
        bath to clipped basin DEM file
    SWO_vrt_path : path
        path to global SWO VRT
    nodata : integer
        nodata value used in the DEM. The default is 99999.
    id : integer
        ID number to assign to the sink point in the gdf. The default is 0.

    Returns
    -------
    sink_point_basin : geopandas geodataframe
        gdf containing the sink point
    """

    # create an empty gdf to store sink point
    sink_point_basin = gpd.GeoDataFrame(crs="EPSG:4326")

    # get basin ID and geotransform and open as array
    basin_raster = gdal.Open(basin_vrt_path, gdal.GA_Update)
    basin_gt = basin_raster.GetGeoTransform()
    raster_band = basin_raster.GetRasterBand(1)
    raster_array = raster_band.ReadAsArray()
    raster_band.SetNoDataValue(nodata)

    # Find lowest elevation pixels 1% of pixels
    low_threshold = np.percentile(raster_array, 1, axis=None)
    min_elev_ind = np.where(raster_array <= low_threshold)
    min_elev_rows = min_elev_ind[0]
    min_elev_cols = min_elev_ind[1]
    #print(len(min_elev_rows), "low pixels found")

    """ 
    For the case where there is only a single pixel below the elevation
    threshold, locate the cell and add it to the point shapefile
    """
    if len(min_elev_rows) == 1:
        #print ("Only a single minimum elevation point in the basin")
        row = min_elev_rows[0]
        col = min_elev_cols[0]

        x, y = get_xy(row, col, basin_gt)

        sink_pt = gpd.GeoDataFrame({"id": id,
                                    "basin_id": basin_id,
                                    "geometry": gpd.points_from_xy([x], [y])},
                                   crs="EPSG:4326")
        sink_point_basin = sink_point_basin.append(sink_pt)

    else:
        """
        For the case where there is a cluster of low elevation cells,
        locate the centroid of the largest continguous area. If there are
        overlapping water pixels, require placement within the water. Find
        the minimum elevation pixel with this area, or find the pixel closest
        to the centroid.
        """        
        #print ("Finding largest contiguous low elevation area")

        # find largest continguous area
        structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        minmask = np.zeros(raster_array.shape, dtype=int)
        minmask[min_elev_ind] = 1
        label_array, num_features = label(minmask, structure)
        count_list = []
        for feature in np.arange(1, num_features + 1):
            feature_count = len(np.where(label_array == feature)[0])
            count_list.append(feature_count)
        biggest_feature = np.argmax(count_list) + 1
        min_contig = np.where(label_array == biggest_feature)
        min_contig_rows = min_contig[0]
        min_contig_cols = min_contig[1]

        # create water pixel mask
        if SWO_vrt_path is not None:
            #endo_shp = os.path.join(os.path.dirname(basin_vrt_path), "basin_{0}.shp".format(basin_id))
            endoSWO_tif = os.path.join(os.path.dirname(
                basin_vrt_path), f"SWO_{basin_id}.tif")

            pyct.cutline_raster_cmdline(SWO_vrt_path, cutline_shp, endoSWO_tif,
                                        srcDS=basin_vrt_path,
                                        dstnodata=255,
                                        outdtype='Byte',
                                        rsAlg=None)

            SWO_raster = gdal.Open(endoSWO_tif, gdal.GA_Update)
            SWO_band = SWO_raster.GetRasterBand(1)
            SWO_array = SWO_band.ReadAsArray()
            SWO_mask_ind = np.where(SWO_array >= 80)
            """
            if water pixels are present and intersect with low elevation pixels,
            ensure that sink is placed in water
            """
            sink_area_mask = np.zeros(raster_array.shape, dtype=int)
            min_contig_1D = np.ravel_multi_index(
                min_contig, raster_array.shape)
            SWO_mask_1D = np.ravel_multi_index(
                SWO_mask_ind, raster_array.shape)
            combined_mask_1D = np.intersect1d(min_contig_1D, SWO_mask_1D)
            combined_mask_2D = np.unravel_index(
                combined_mask_1D, raster_array.shape)
            sink_area_mask[combined_mask_2D] = 1
            combined_mask_ind = np.where(sink_area_mask == 1)

            if len(combined_mask_ind[0]) > 0:
                # if there is an overlapping area, use that to find sinks
                min_contig = combined_mask_ind
                min_contig_rows = min_contig[0]
                min_contig_cols = min_contig[1]
            else:
                # if there is no overlapping area, use low elevation pixels only
                sink_area_mask = np.zeros(raster_array.shape, dtype=int)
                sink_area_mask[min_contig] = 1

            SWO_raster = None
            SWO_band = None
            SWO_array = None
            os.remove(endoSWO_tif)
        else:
            # if there is no SWO mask, use low elevation pixels only
            sink_area_mask = np.zeros(raster_array.shape, dtype=int)
            sink_area_mask[min_contig] = 1

        # find min elevation within the low elevation/water area
        min_contig_min = raster_array[min_contig].min()
        min_pixel_ind = np.where((sink_area_mask == 1) &
                                 (raster_array == min_contig_min))
        min_pixel_rows = min_pixel_ind[0]
        min_pixel_cols = min_pixel_ind[1]

        if len(min_pixel_ind[0]) == 1:
            # if there is a single minimum within the area, choose that.
            #print("assigning sink to minimum within selected area")
            row = min_pixel_rows[0]
            col = min_pixel_cols[0]

            x_sink, y_sink = get_xy(row, col, basin_gt)

        else:
            #print("assigning sink to centroid within selected area")
            # find centroid
            x_values = []
            y_values = []
            low_pixels = []
            for i, row in enumerate(min_pixel_rows):
                x_value, y_value = get_xy(row, min_pixel_cols[i], basin_gt)
                x_values.append(x_value)
                y_values.append(y_value)
                low_pixels.append(Point(x_value, y_value))
            x_centroid = sum(x_values) / len(min_pixel_cols)
            y_centroid = sum(y_values) / len(min_pixel_rows)
            #print ("Centroid = " + str(x_centroid) + " , ", str(y_centroid))

            centroid = Point(x_centroid, y_centroid)
            pixels = MultiPoint(low_pixels)

            # add the centroid to the point gdf
            nearest_pixel = nearest_points(centroid, pixels)
            sink_pixel = nearest_pixel[1]
            x_sink = sink_pixel.x
            y_sink = sink_pixel.y

        sink_pt = gpd.GeoDataFrame({"id": id,
                                    "basin_id": basin_id,
                                    "geometry": gpd.points_from_xy([x_sink], [y_sink])},
                                   crs="EPSG:4326")
        sink_point_basin = sink_point_basin.append(sink_pt)
        # print()

    basin_raster = None
    raster_band = None

    return sink_point_basin


def build_sink_mask(sink_shp, stencil_tif, sink_gdf, sink_mask_tif):
    """
    Rasterize sink points to form raster mask with the same projection and
    geotransform as the input stencil_tif.

    Parameters
    ----------
    sink_shp : path
        shapefile of sink point features
    stencil_tif : path
        .tif raster for shape files to be projected/resampled on to
    sink_gdf : gdf
        Geodataframe of point features - only used to determine if no lines exist
    sink_mask_tif : path
        output path for raster mask

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

    if sink_gdf is not None:
        # if not os.path.exists(sink_mask_tif):
        rstr_command = "gdal_rasterize -burn 1 -ot Byte -ts {0} {1}\
            -te {2} {3} {4} {5} {6} {7}".format(xshape, yshape, xmin,
                                                ymin, xmax, ymax,
                                                sink_shp, sink_mask_tif)
        subprocess.call(rstr_command, shell=True,
                        stdout=subprocess.DEVNULL)
