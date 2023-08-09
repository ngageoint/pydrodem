#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 3 2019

@author: Kimberly McCormack

Last edited on: 07/29/2020


Functions to convert to and from numpy arrays and tif format, and resample
tif files to a different resolution.

Built as part of the pydroDEM hydroconditioning software

"""
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from shapely import geometry
from osgeo import gdal, gdal_array, ogr, osr
import subprocess
import numpy as np
import os
import warnings

warnings.simplefilter("ignore")

gdal.UseExceptions()
gdal.PushErrorHandler("CPLQuietErrorHandler")


# ----------------------------------------#
######## FILE FORMAT CONVERSIONS #########
# ----------------------------------------#


def tif2npy_save(tif_file, npypath, fname=None, save_geoT=True, dtype=np.float32):
    """
    Transform an a single geotiff into a numpy array and save as pickled
    .npy file

    Parameters
    ----------
    tif_file : path
        path to tif to convert
    npypath : path
        directory to store converted numpy file
    fname : str, optional
        name of file to be saved (without extension). The default is None.
    save_geoT : Boolean, optional
        save the geotransform as a numpy array. The default is True.
            geoT saves the array:
            [lon_min, lon_step, lon_warp, lat_max, lat_warp, lat_step].
    dtype : numpy data type, optional
        set the output datatype for the numpy array. The default is np.float32.

    Returns
    -------
    None.

    """

    if not os.path.exists(npypath):
        os.mkdir(npypath)

    if not fname:
        """pull basename of .tif to assign to .npy file"""
        fname = os.path.basename(tif_file).replace(".tif", ".npy")

    npy_file = os.path.join(npypath, fname)

    if save_geoT:
        """Pull geotransform data from original tif file"""
        geoT = get_geoT(tif_file)
        out_geoT_npy = os.path.join(npypath, "geoT.npy")
        np.save(out_geoT_npy, np.asarray(geoT))

    """Load tif as numpy array and save"""
    tif_array = gdal_array.LoadFile(tif_file)
    np.save(npy_file, tif_array)


def tif2npy(tif_file, dtype=np.float32):
    """
    Transform an a single geotiff into a numpy array and return array

    Parameters
    ----------
    tif_file : path
        path to tif to convert
    dtype : numpy data type, optional
        set the output datatype for the numpy array. The default is np.float32.

    Returns
    -------
    Numpy array of raster data

    """

    array = gdal_array.LoadFile(tif_file)

    return array.astype(dtype)


def vrt2gtiff(vrtpath, outpath):
    """
    Convert from vrt file into full geotiff

    Parameters
    ----------
    vrtpath : path
        path to input vrt
    outpath : path
        output path
    """

    inDS = gdal.Open(vrtpath, gdal.GA_ReadOnly)
    projection = inDS.GetProjectionRef()

    inDS = gdal.Translate(outpath, inDS, format="GTiff", outputSRS=projection)

    inDS = None

    return


def npy2tif(array, orig_tif, outfile, nodata=None, dtype=gdal.GDT_Float32, outformat="GTiff"):
    """
    Convert an array to a geotiff based on the stencil of another
    geotiff file

    Parameters
    ----------
    array : array-like
        2D array of input data
    orig_tif : path
        stencil geotiff to provide projection, geotransform, etc
    outfile : path
        output .tif path
    nodata : float, optional
        no data value, by default None
    dtype : gdal datatype, optional
        data type, by default gdal.GDT_Float32
    """

    inDS = gdal.Open(orig_tif, gdal.GA_ReadOnly)
    Xshape = inDS.RasterXSize
    Yshape = inDS.RasterYSize

    """Create output driver from original tif"""
    # driver = inDS.GetDriver()
    driver = gdal.GetDriverByName(outformat)
    outDS = driver.Create(outfile, Xshape, Yshape, 1, dtype)
    outBand = outDS.GetRasterBand(1)
    outBand.WriteArray(array, 0, 0)
    if nodata is not None:
        outBand.SetNoDataValue(nodata)
    outDS.SetGeoTransform(inDS.GetGeoTransform())
    outDS.SetProjection(inDS.GetProjection())

    outBand.FlushCache()
    outDS.FlushCache()  # saves to disk

    outBand = None
    outDS = None
    inDS = None

    return


def hdf2tif_subdataset(inDS, outDS, subdataset, dtype=gdal.GDT_Byte):
    """
    unpack a single subdataset from a HDF5 container and write to GeoTiff

    Parameters
    ----------
    inDS : path
        input HDF5 file
    outDS : path
        output tif file
    subdataset : str
        subdataset to extract from hdf file
    dtype : gdal datatype, optional
        output data type, by default gdal.GDT_Byte
    """

    hdf_ds = gdal.Open(inDS, gdal.GA_ReadOnly)
    band_ds = gdal.Open(hdf_ds.GetSubDatasets()[
                        subdataset][0], gdal.GA_ReadOnly)

    """read into numpy array"""
    band_array = band_ds.ReadAsArray()

    """create raster"""
    out_ds = gdal.GetDriverByName("GTiff").Create(
        outDS,
        band_ds.RasterXSize,
        band_ds.RasterYSize,
        1,
        dtype,
        ["COMPRESS=LZW", "TILED=YES"],
    )
    out_ds.SetGeoTransform(band_ds.GetGeoTransform())
    out_ds.SetProjection(band_ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(band_array)

    out_ds = None


def tifdowncast(input_tif, dtype=gdal.GDT_Float32, dstSRS=None, nodataval=None):
    """
    Transform a geotiff from 64bit float to 32bit float

    WhiteBoxTools auto saves all out as 64 bit, which is often unecessary.
    This function makes a lower bit (default 32bit) copy of a geotiff and deletes
    the original 64bit version

    Parameters
    ----------
    input_tif : path
        input tif to downcast and delete
    dtype : gdal data type, optional
        output data type, by default gdal.GDT_Float32
    dstSRS : str, optional
        output projection epsg code. If None, input SRS is used, by default None
    nodataval : float, optional
        output nodata value, by default None
    """

    outfile = os.path.splitext(os.path.basename(input_tif))[0] + "_dc.tif"
    output_path = os.path.join(os.path.dirname(input_tif), outfile)

    data_in = gdal.Open(input_tif, gdal.GA_ReadOnly)
    band = data_in.GetRasterBand(1)

    if nodataval is None:
        nodataval = band.GetNoDataValue()

    if dstSRS is None:
        projection = data_in.GetProjectionRef()
    else:
        projection = dstSRS

    Xshape = data_in.RasterXSize
    Yshape = data_in.RasterYSize
    geoT = data_in.GetGeoTransform()
    xmin = geoT[0]
    ymax = geoT[3]
    xmax = xmin + geoT[1] * Xshape
    ymin = ymax + geoT[5] * Yshape

    data_in = gdal.Translate(
        output_path,
        data_in,
        format="GTiff",
        outputBounds=[xmin, ymax, xmax, ymin],
        width=Xshape,
        height=Yshape,
        noData=nodataval,
        outputSRS=projection,
        outputType=dtype,
    )

    data_in = None

    """delete 64 bit file"""
    os.remove(input_tif)
    os.rename(output_path, input_tif)

    return


def tifcompress(input_tif, dstSRS=None):
    """
    Compress a geotiff

    Parameters
    ----------
    input_tif : path
        input tif to downcast and delete

    """

    outfile = os.path.splitext(os.path.basename(input_tif))[
        0] + "_compressed.tif"
    output_path = os.path.join(os.path.dirname(input_tif), outfile)

    inDS = gdal.Open(input_tif, gdal.GA_ReadOnly)
    band = inDS.GetRasterBand(1)
    nodataval = band.GetNoDataValue()
    if dstSRS is not None:
        proj_out = dstSRS
    else:
        proj_out = inDS.GetProjectionRef()

    Xshape = inDS.RasterXSize
    Yshape = inDS.RasterYSize
    geoT = inDS.GetGeoTransform()
    xmin = geoT[0]
    ymax = geoT[3]
    xmax = xmin + geoT[1] * Xshape
    ymin = ymax + geoT[5] * Yshape

    translate_options = gdal.TranslateOptions(
        format="GTiff",
        outputBounds=[xmin, ymax, xmax, ymin],
        width=Xshape,
        height=Yshape,
        noData=nodataval,
        outputSRS=proj_out,
        creationOptions=["COMPRESS=LZW"],
    )
    inDS = gdal.Translate(output_path, inDS, options=translate_options)
    inDS = None

    """delete uncompressed file"""
    os.remove(input_tif)
    os.rename(output_path, input_tif)


def pbf2shp(inDS_pbf, out_dir, clip_shp=None):
    """
    INCOMPLETE

    Convert from .pbf file (OpenStreetMap compressed format) to a directory of shape files

     Parameters
    ----------
    inDS_pbf:  path
        Path to .pbf file
    out_dir : path
        path to outputs (directory)
    clip_shp : path, optional
        shapefile to clip data to. The default is None.

    """
    if clip_shp is not None:
        ogr_command = "ogr2ogr -clip {0} -skipfailures {1} {2}".format(
            clip_shp, out_dir, inDS_pbf
        )

    else:
        ogr_command = "ogr2ogr -skipfailures {0} {1}".format(out_dir, inDS_pbf)

    # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.call(ogr_command, shell=True)


def build_multi_bbox_shp(input_tif, output_shp, step, poly_buffer=None):
    """
    Create small bounding boxes within a mosaic

    Parameters
    ----------
    input_tif : path
        input mosaic to create bounding boxes within
    output_shp : path
        output shapefile path
    step : float
        size of bounding boxes (degrees, linear)
    poly_buffer : float, optional
        bounding box buffer, by default None
    """

    inDS = gdal.Open(input_tif)
    geoT = inDS.GetGeoTransform()
    proj = inDS.GetProjection()
    nx, ny = inDS.RasterXSize, inDS.RasterYSize
    inDS = None

    """Define bounding box from DEM cell geotransform"""
    dlon, dlat = np.abs(geoT[1]), np.abs(geoT[5])
    latmax = geoT[3] - (0.5 * dlat)
    latmin = latmax - (dlat * (ny - 1))
    lonmin = geoT[0] + (0.5 * dlon)
    lonmax = lonmin + (dlon * (nx - 1))
    lon_steps = (lonmax - lonmin) / step
    lat_steps = (latmax - latmin) / step
    bbox_xmin = np.linspace(lonmin, lonmax, lon_steps, endpoint=False)
    bbox_ymin = np.linspace(latmin, latmax, lat_steps, endpoint=False)

    """Create geodataframe of the shapes to process by"""
    TDT_bbox = []
    TDT_bbox_buffer = []
    for xm in bbox_xmin:
        for ym in bbox_ymin:
            # create bounding box polygon and buffer
            bbox = geometry.box(xm, ym, xm + step, ym + step)
            TDT_bbox.append(bbox)

            if poly_buffer is not None:
                bbox_buffer = bbox.buffer(
                    poly_buffer, cap_style=3, joinstyle=2)
                TDT_bbox_buffer.append(bbox_buffer)

    if poly_buffer is not None:
        TDT_poly_gdf = gpd.GeoDataFrame(geometry=TDT_bbox_buffer)
        TDT_poly_gdf["poly_orig"] = TDT_bbox
        TDT_poly_gdf.crs = "EPSG:4326"
    else:
        TDT_poly_gdf = gpd.GeoDataFrame(geometry=TDT_bbox)
        TDT_poly_gdf.crs = "EPSG:4326"

    # convert it to a shapefile with OGR
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(output_shp)
    targetprj = osr.SpatialReference(wkt=proj)
    layer = ds.CreateLayer("basins", targetprj, ogr.wkbPolygon)
    # Add ID attribute
    layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    defn = layer.GetLayerDefn()
    for s_ind, shape in enumerate(TDT_poly_gdf.geometry):
        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField("id", s_ind)
        # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkb(shape.wkb)
        feat.SetGeometry(geom)
        layer.CreateFeature(feat)
        feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = None


def build_bbox_gdf(input_tif, exclude_nodata=False):
    """
    Create bounding box gdf from a geotiff

    Parameters
    ----------
    input_tif : path
        input mosaic to create bounding boxes within

    """

    inDS = gdal.Open(input_tif)
    band = inDS.GetRasterBand(1)
    nodataval = band.GetNoDataValue()
    
    # load data and check for nodata values
    if exclude_nodata:
        in_array = gdal_array.LoadFile(input_tif)
        data_ind = np.where(in_array!=nodataval)
        
        
        
    
    
    geoT = inDS.GetGeoTransform()
    proj = inDS.GetProjection()
    xshape, yshape = inDS.RasterXSize, inDS.RasterYSize
    inDS = None

    """Define bounding box from DEM cell geotransform"""
    xmin = geoT[0]
    ymax = geoT[3]
    xmax = xmin + geoT[1] * xshape
    ymin = ymax + geoT[5] * yshape

    bbox_poly = geometry.box(xmin, ymin, xmax, ymax)
    bbox_df = pd.DataFrame({"name": ["bbox"], "geometry": [bbox_poly]})
    bbox_gdf = gpd.GeoDataFrame(
        bbox_df, crs=proj, geometry=bbox_df["geometry"])

    return bbox_gdf


# -----------------------------------------------#
######## RESAMPLE / REPROJECT / EXTRACT #########
# -----------------------------------------------#


def getextent(inDS, gdalDriver="ESRI Shapefile"):
    """Extract the extent of a shapefile (minimum bounding rectangle)

    Parameters
    ----------
    inDS : path
        shape file
    gdalDriver : str, optional
        gdal driver, by default "ESRI Shapefile"

    Returns
    -------
    extent
        minimum bounding rectangle coordinates
    """

    inDriver = ogr.GetDriverByName(gdalDriver)
    dataSource = inDriver.Open(inDS, 0)
    layer = dataSource.GetLayer(0)
    extent = layer.GetExtent()
    dataSource = None

    return extent


def get_geoT(tif_file):
    """
    Retrieve the geotransforma data from a tif and return as a list

    Parameters
    ----------
    tif_file : path [str]
        path to tif

    Returns
    -------
    list of geotransform parameters:

        [lon_min, lon_step, lon warp, lat_max, lat warp, lat_step]

    """

    inDS = gdal.Open(tif_file, gdal.GA_ReadOnly)
    geoT = inDS.GetGeoTransform()
    inDS = None

    return geoT


def clip_to_window(
    inDEM_array,
    inDEM_tif,
    outDS_tif,
    footprint_arr,
    clip_buf,
    nx,
    ny,
    dtype=gdal.GDT_Float32,
    nodataval=-9999,
    clipped_ind_arr=None,
):
    """
    Clip a tif based on an array of indices within the array version. This is a very specific
    function used to clip out large pits within a DEM to iterate through partial fill/carve
    solutions

    Parameters
    ----------
    inDEM_array : array
        2D array of full input tif
    inDEM_tif : path
        input tif
    outDS_tif : path
        output path for clipped tif
    footprint_arr : array
        array of indices of smaller footprint within larger array
    clip_buf : int
        buffer around footprint in pixels
    nx : int
        num pixels in x-direction
    ny : int
        num pixels in y-direction
    dtype : gdal datatype, optional
        output data type for tif, by default gdal.GDT_Float32
    nodataval : int, optional
        nodata value, by default -9999

    Returns
    -------
    array
        array of clipped indices within larger array (with buffer)
    int
        new nx
    int
        new ny
    """

    row_min, row_max = footprint_arr[0].min(), footprint_arr[0].max()
    col_min, col_max = footprint_arr[1].min(), footprint_arr[1].max()

    row_min = np.int(np.maximum(0, (row_min - clip_buf)))
    row_max = np.int(np.minimum(ny, (row_max + clip_buf + 1)))
    col_min = np.int(np.maximum(0, (col_min - clip_buf)))
    col_max = np.int(np.minimum(nx, (col_max + clip_buf + 1)))

    if clipped_ind_arr is None:
        combined_mask = np.zeros_like(inDEM_array)
        combined_mask[row_min:row_max, col_min:col_max] = 1
        clipped_ind_arr = np.nonzero(combined_mask)
        del combined_mask

    out_array = inDEM_array[row_min:row_max, col_min:col_max]
    nx_clip = out_array.shape[1]
    ny_clip = out_array.shape[0]

    """ tif stencil from inDEM """
    inDS = gdal.Open(inDEM_tif, gdal.GA_ReadOnly)
    geoT = inDS.GetGeoTransform()
    dlon, dlat = np.abs(geoT[1]), np.abs(geoT[5])
    latmax = geoT[3]
    lonmin = geoT[0]
    x_min = lonmin + (col_min * dlon)
    x_max = lonmin + (col_max * dlon)
    y_min = latmax - (row_max * dlat)
    y_max = latmax - (row_min * dlat)
    clip_geot = (x_min, dlon, 0.0, y_max, 0.0, -dlat)

    driver = inDS.GetDriver()
    outDS = driver.Create(outDS_tif, nx_clip, ny_clip, 1, dtype)
    outBand = outDS.GetRasterBand(1)
    outBand.WriteArray(out_array, 0, 0)
    outBand.SetNoDataValue(nodataval)
    outBand.FlushCache()
    outBand = None
    outDS.SetGeoTransform(clip_geot)
    outDS.SetProjection(inDS.GetProjection())
    inDS = None
    outDS = None
    del out_array

    return clipped_ind_arr, nx_clip, ny_clip


def resample(
    inDS,
    outDS,
    xshape=None,
    yshape=None,
    outformat="Gtiff",
    srcDS=None,
    rsAlg=gdal.GRIORA_NearestNeighbour,
    dstnodata=None,
    dtype=gdal.GDT_Float32,
):
    """
    Resample a geotiff to a different resolution


    Parameters
    ----------
    inDS : path
        path to input geotiff to be resampled.
    outDS : path
        path to store resampled tif.
    xshape : integer, optional
        number of resampled pixels in x dimension. The default is 9001.
    yshape : integer, optional
        number of resampled pixels in y dimension. The default is 9001.
    outformat : GDAL format string, optional
        Set format of the output dataset. The default is "Gtiff".
        "VRT" also works to output a virtual raster
    srcDS : path , optional
        optional source tif to set projection, bounds, and size. The default is None.
    rsAlg : GDAL resampling algorithm, optional
        Set the rasampling algorithm used.
        The default is gdal.GRIORA_NearestNeighbour.

        Default xshape, yshape kwargs are set to the dimensions of a
        TanDEM-X cell (1 degree x 1 degree). Equal to ~12m resoloution

        Resampling algorithm options from gdal:

            GRIORA_NearestNeighbour
            GRIORA_Bilinear
            GRIORA_Cubic:
            GRIORA_CubicSpline:
            GRIORA_Lanczos:
            GRIORA_Average:
            GRIORA_Mode:
            GRIORA_Gauss:

    Returns
    -------
    None.

    """

    inDS = gdal.Open(inDS, gdal.GA_ReadOnly)

    if dstnodata is None:
        band = inDS.GetRasterBand(1)
        dstnodata = band.GetNoDataValue()
        band = None

    if srcDS is not None:
        """
        Get resampling projection and size from a source raster file
        """
        srcDS = gdal.Open(srcDS, gdal.GA_ReadOnly)
        xshape = srcDS.RasterXSize
        yshape = srcDS.RasterYSize
        projection = srcDS.GetProjectionRef()
        geoT = srcDS.GetGeoTransform()
        xmin = geoT[0]
        ymax = geoT[3]
        xmax = xmin + geoT[1] * xshape
        ymin = ymax + geoT[5] * yshape
        srcDS = None

    else:
        """
        Get projection from input file and set size explicitly with kwargs
        """
        Xshape = inDS.RasterXSize
        Yshape = inDS.RasterYSize
        projection = inDS.GetProjectionRef()
        geoT = inDS.GetGeoTransform()
        xmin = geoT[0]
        ymax = geoT[3]
        xmax = xmin + geoT[1] * Xshape
        ymin = ymax + geoT[5] * Yshape

    inDS = gdal.Warp(
        outDS,
        inDS,
        format=outformat,
        outputBounds=[xmin, ymin, xmax, ymax],
        width=xshape,
        height=yshape,
        dstSRS=projection,
        dstNodata=dstnodata,
        resampleAlg=rsAlg,
        outputType=dtype,
    )
    inDS = None


def resample_bbox(
    inDS,
    outDS,
    bbox,
    xshape=9001,
    yshape=9001,
    outformat="Gtiff",
    srcDS=None,
    rsAlg=gdal.GRIORA_NearestNeighbour,
    srcnodata=None,
    dstnodata=None,
    dtype=gdal.GDT_Float32,
):
    """
    Resample a geotiff to a different resolution


    Parameters
    ----------
    inDS : path
        path to input geotiff to be resampled.
    outDS : path
        path to store resampled tif.
    bbox : list
        list of output bounds -- [xmin, ymin, xmax, ymax]
    xshape : integer, optional
        number of resampled pixels in x dimension. The default is 9001.
    yshape : integer, optional
        number of resampled pixels in y dimension. The default is 9001.
    outformat : GDAL format string, optional
        Set format of the output dataset. The default is "Gtiff".
        "VRT" also works to output a virtual raster
    srcDS : path [str], optional
        optional source tif to set projection. The default is None.
    rsAlg : GDAL resampling algorithm, optional
        Set the rasampling algorithm used.
        The default is gdal.GRIORA_NearestNeighbour.

        Default xshape, yshape kwargs are set to the dimensions of a
        TanDEM-X cell (1 degree x 1 degree). Equal to ~12m resoloution

        Resampling algorithm options from gdal:

            GRIORA_NearestNeighbour
            GRIORA_Bilinear
            GRIORA_Cubic:
            GRIORA_CubicSpline:
            GRIORA_Lanczos:
            GRIORA_Average:
            GRIORA_Mode:
            GRIORA_Gauss:

    Returns
    -------
    None.

    """

    inDS = gdal.Open(inDS, gdal.GA_ReadOnly)

    if srcDS is not None:
        """
        Get resampling projection and size from a source raster file
        """
        srcDS = gdal.Open(srcDS, gdal.GA_ReadOnly)
        projection = srcDS.GetProjectionRef()
        srcDS = None

    else:
        """
        Get projection from input file and set size explicitly with kwargs
        """
        projection = inDS.GetProjectionRef()
        geoT = inDS.GetGeoTransform()

    inDS = gdal.Warp(
        outDS,
        inDS,
        format=outformat,
        outputBounds=bbox,
        width=xshape,
        height=yshape,
        srcNodata=srcnodata,
        dstNodata=dstnodata,
        dstSRS=projection,
        resampleAlg=rsAlg,
        outputType=dtype,
    )
    inDS = None


def build_vrt(
    tifs,
    vrt_path,
    srcnodata=None,
    VRTnodata=None,
    resolution='highest',
    xres=None,
    yres=None,
    alignedPixels=True,
    resampleAlg=None,
    outputSRS=None,
):
    """
    Create virtual raster file from and arbitrary number of geotiffs

    Parameters
    ----------
    tifs : list
        list of all tifs to be used to generate the VRT (output of glob.glob())
    vrt_path : path
        path to store virtual raster file

     Keyword arguments for gdal.buildVRT() are (Defaults are None):
            resolution --- 'highest', 'lowest', 'average', 'user'.
            outputBounds --- output bounds as (minX, minY, maxX, maxY) in target SRS.
            xRes, yRes --- output resolution in target SRS.
            targetAlignedPixels --- whether to force output bounds to be multiple of output resolution.
            separate --- whether each source file goes into a separate stacked band in the VRT band.
            bandList --- array of band numbers (index start at 1).
            addAlpha --- whether to add an alpha mask band to the VRT when the source raster have none.
            resampleAlg --- resampling mode.
            outputSRS --- assigned output SRS.
            allowProjectionDifference --- whether to accept input datasets have not the same projection.
                Note: they will *not* be reprojected.
            srcNodata --- source nodata value(s).
            VRTNodata --- nodata values at the VRT band level.
            hideNodata --- whether to make the VRT band not report the NoData value.


    Returns
    -------
    None. Saves a VRT.

    """

    vrt_tif = gdal.BuildVRT(
        vrt_path,
        tifs,
        srcNodata=srcnodata,
        VRTNodata=VRTnodata,
        resolution=resolution,
        xRes=xres,
        yRes=yres,
        resampleAlg=resampleAlg,
        outputSRS=outputSRS,
    )

    vrt_tif = None


def set_vrt_nodata(inDEM, outDEM, dstNodata=-9999):
    """
    Make a VRT with different nodata value than input tif

    Parameters
    ----------
    inDEM : path
        input geotiff (or VRT)
    vrt_path : path
        path to store virtual raster file
    out_nodata : int
        nodata value for output VRT. Default= -9999

    Returns
    -------
    None. Saves a VRT.

    """

    srcDS = gdal.Open(inDEM, gdal.GA_ReadOnly)
    in_band = srcDS.GetRasterBand(1)
    srcNodata = in_band.GetNoDataValue()
    xshape = srcDS.RasterXSize
    yshape = srcDS.RasterYSize
    projection = srcDS.GetProjectionRef()

    srcDS = gdal.Warp(
        outDEM,
        srcDS,
        format="VRT",
        srcNodata=srcNodata,
        dstNodata=dstNodata,
        width=xshape,
        height=yshape,
        dstSRS=projection,
    )

    srcDS = None


def reprojectraster(inDS, outDS, srcNodata=None, dstNodata=None, dstSRS="EPSG:4326"):
    """
    Reproject a raster

    Parameters
    ----------
    inDS : path
        Path to input raster.
    outDS : path
        Path for output raster.
    dstSRS : string of target reference rframe, optional
        The default is "EPSG:4326".

    Returns
    -------
    None.

    """

    inDS = gdal.Open(inDS, gdal.GA_ReadOnly)
    inDS = gdal.Warp(
        outDS,
        inDS,
        format="GTiff",
        srcNodata=srcNodata,
        dstNodata=dstNodata,
        dstSRS=dstSRS,
    )
    # targetAlignedPixels=True)
    inDS = None


def cutline_raster_cmdline(
    inDS,
    shapefile,
    outDS,
    srcnodata=-9999,
    dstnodata=-9999,
    dstSRS="EPSG:4326",
    outdtype="Float32",
    outformat="GTiff",
    rsAlg=None,
    srcDS=None,
):

    cmd_lst = [
        "gdalwarp",
        "-cutline",
        f"{shapefile}",
        "-crop_to_cutline",
        "-srcnodata",
        f"{srcnodata}",
        "-dstnodata",
        f"{dstnodata}",
        "-ot",
        f"{outdtype}",
        "-t_srs",
        f"{dstSRS}",
    ]

    if rsAlg is not None:
        cmd_lst.extend(["-r", f"{rsAlg}"])

    cmd_lst.extend([f"{inDS}", f"{outDS}"])

    ret = subprocess.call(cmd_lst, shell=True, stdout=subprocess.DEVNULL)


def cutline_raster_simple(
    inDS,
    shapefile,
    outDS,
    nodata=-9999,
    dstSRS="EPSG:4326",
    outdtype=gdal.GDT_Float32,
    outformat="GTiff",
    rsAlg=None,
):
    """
    Use a shape file to crop a raster
    """

    """
    Get resampling projection and size from a raster file
    """
    srcDS = gdal.Open(inDS, gdal.GA_ReadOnly)
    xshape = srcDS.RasterXSize
    yshape = srcDS.RasterYSize
    projection = srcDS.GetProjectionRef()
    geoT = srcDS.GetGeoTransform()
    xmin = geoT[0]
    ymax = geoT[3]
    xmax = xmin + geoT[1] * xshape
    ymin = ymax + geoT[5] * yshape
    srcDS = None

    outputbounds = [xmin, ymin, xmax, ymax]

    tempds = gdal.Warp(
        outDS,
        inDS,
        format=outformat,
        outputBounds=outputbounds,
        width=xshape,
        height=yshape,
        dstSRS=projection,
        cutlineDSName=shapefile,
        resampleAlg=rsAlg,
    )

    tempds = None

    return


def create_cutline_raster(
    inDS,
    shapefile,
    outDS,
    polygon=None,
    srcDS=None,
    buffer=0.0005,
    usecutline=True,
    srcnodata=None,
    nodata=-9999,
    dstSRS="EPSG:4326",
    overwrite_cutline=False,
    outdtype=gdal.GDT_Float32,
    outformat="GTiff",
    rsAlg=None,
    paletted=False,
):
    """
    Use a shape file to crop a raster


    Parameters
    ----------
    inDS : path
        input raster
    shapefile : path
        input shapefile
    outDS : path
        output raster clipped to shape
    polygon : shapely polygon, optional
        use polygon to define bounding box, by default None
    srcDS : path, optional
        raster to provide projection,size,etc of output, by default None
    buffer : float, optional
        buffer used around shape, by default 0.0005
    usecutline : bool, optional
        option to clip raster to shape bounding box, but not run cutline tool, by default True
    nodata : int, optional
        no data value for output tif, by default -9999
    dstSRS : str, optional
        output projection, by default "EPSG:4326"
    overwrite_cutline : bool, optional
        overwrite existing outDS?, by default False
    outdtype : gdal data type, optional
        output data type, by default gdal.GDT_Float32
    outformat : str, optional
        output gdal format, by default "GTiff"
    rsAlg : gdal.alg, optional
        gdal resampling algorithm, by default None
    """

    if srcDS is not None:

        """
        Get resampling projection and size from a raster file
        """
        srcDS = gdal.Open(srcDS, gdal.GA_ReadOnly)
        xshape = srcDS.RasterXSize
        yshape = srcDS.RasterYSize
        projection = srcDS.GetProjectionRef()
        geoT = srcDS.GetGeoTransform()
        xmin = geoT[0]
        ymax = geoT[3]
        xmax = xmin + geoT[1] * xshape
        ymin = ymax + geoT[5] * yshape
        srcDS = None

    else:
        if polygon is not None:
            extent = polygon.bounds
            minx, miny, maxx, maxy = polygon.bounds
        else:
            shape_gdf = gpd.read_file(shapefile)
            minx, miny, maxx, maxy = shape_gdf.total_bounds

        """Pull out polygon bounding box. Clip VRT to bbox, then use cutline"""
        xmin = minx - buffer
        ymax = maxy + buffer
        xmax = maxx + buffer
        ymin = miny - buffer
        xshape, yshape = 0, 0
        projection = dstSRS

    projection = dstSRS
    projwin = [xmin, ymax, xmax, ymin]  # [ulx, uly, lrx, lry]
    outputbounds = [xmin, ymin, xmax, ymax]

    if overwrite_cutline:
        if os.path.exists(outDS):
            os.remove(outDS)

    if paletted:
    
        """Pull out polygon bounding box. Clip VRT to bbox, then use cutline"""
        xminb = xmin - buffer
        ymaxb = ymax + buffer
        xmaxb = xmax + buffer
        yminb = ymin - buffer
        projwin_buffer = [xminb, ymaxb, xmaxb, yminb]  # [ulx, uly, lrx, lry]
    
        outDS_temp = outDS.replace(".tif", "_temp.tif")
        tempds = gdal.Translate(
            outDS_temp,
            inDS,
            format="GTiff",
            projWin=projwin_buffer,
            noData=nodata,
            outputType=outdtype,
            projWinSRS=projection,
            outputSRS=projection,
        )

        tempDS = None
        tif_array = gdal_array.LoadFile(outDS_temp)
        outDS_temp2 = outDS.replace(".tif", "_temp2.tif")
        npy2tif(tif_array, outDS_temp, outDS_temp2,
                dtype=outdtype, outformat="GTiff")
        inDS = outDS_temp2

    if usecutline:
        tempformat = "GTiff"
    else:
        tempformat = outformat

    """Clip to polygon bounding box"""
    tempds = gdal.Translate(
        outDS,
        inDS,
        format=tempformat,
        projWin=projwin,
        width=xshape,
        height=yshape,
        noData=nodata,
        outputType=outdtype,
        projWinSRS=projection,
        outputSRS=projection,
        resampleAlg=rsAlg,
    )

    if usecutline:
        """Crop tif with shape"""
        xshape = tempds.RasterXSize
        yshape = tempds.RasterYSize
        tempds = gdal.Warp(
            outDS,
            tempds,
            format=outformat,
            outputBounds=outputbounds,
            width=xshape,
            height=yshape,
            dstSRS=projection,
            cutlineDSName=shapefile,
            resampleAlg=rsAlg,
        )

    tempds = None

    if paletted:
        os.remove(outDS_temp)
        os.remove(outDS_temp2)

    return


def create_cutline_raster_set_res(
    inDS,
    shapefile,
    outDS,
    buffer=5,
    usecutline=True,
    srcnodata=None,
    nodata=-9999,
    dstSRS="EPSG:4326",
    outdtype=gdal.GDT_Float32,
    outformat="GTiff",
    rsAlg=None,
    xres=None,
    yres=None,
):
    """
    Use a shape file to crop a raster with a speficfied resolution, either from kwargs or the
    input vrt/tif. Manually aligns the crop boundary to pixel boundaries


    Parameters
    ----------
    inDS : path
        input raster
    shapefile : path
        input shapefile
    outDS : path
        output raster clipped to shape
    polygon : shapely polygon, optional
        use polygon to define bounding box, by default None
    srcDS : path, optional
        raster to provide projection,size,etc of output, by default None
    buffer : float, optional
        buffer used around shape, by default 0.0005
    usecutline : bool, optional
        option to clip raster to shape bounding box, but not run cutline tool, by default True
    nodata : int, optional
        no data value for output tif, by default -9999
    dstSRS : str, optional
        output projection, by default "EPSG:4326"
    overwrite_cutline : bool, optional
        overwrite existing outDS?, by default False
    outdtype : gdal data type, optional
        output data type, by default gdal.GDT_Float32
    outformat : str, optional
        output gdal format, by default "GTiff"
    rsAlg : gdal.alg, optional
        gdal resampling algorithm, by default None
    """

    srcDS = gdal.Open(inDS, gdal.GA_ReadOnly)
    geoT = srcDS.GetGeoTransform()
    xmin_vrt = geoT[0]
    ymax_vrt = geoT[3]
    xres_vrt = geoT[1]
    yres_vrt = geoT[5]
    srcDS = None

    if xres is None:
        xres = xres_vrt
    if yres is None:
        yres = yres_vrt

    shape_gdf = gpd.read_file(shapefile)
    minx, miny, maxx, maxy = shape_gdf.total_bounds

    """Manually line up cropped raster bounds with vrt pixel boundaries"""
    x_width = np.floor(((maxx - minx) / xres_vrt) + buffer * 2)
    y_height = np.floor(((maxy - miny) / np.abs(yres_vrt)) + buffer * 2)

    x_offset = minx - xmin_vrt
    pix_offset_x = x_offset / xres_vrt
    pix_offset_x_align = np.floor(pix_offset_x) - buffer
    xmin = xmin_vrt + (pix_offset_x_align * xres_vrt)
    xmax = xmin + (x_width * xres_vrt)

    y_offset = ymax_vrt - maxy
    pix_offset_y = np.abs(y_offset / yres_vrt)
    pix_offset_y_align = np.floor(pix_offset_y) - buffer
    ymax = ymax_vrt + (pix_offset_y_align * yres_vrt)
    ymin = ymax + (y_height * yres_vrt)

    outputbounds = [xmin, ymin, xmax, ymax]
    projwin = [xmin, ymax, xmax, ymin]  # [ulx, uly, lrx, lry]
    projection = dstSRS

    """Clip to polygon bounding box"""
    tempds = gdal.Translate(
        outDS,
        inDS,
        format="GTiff",
        projWin=projwin,
        xRes=xres,
        yRes=yres,
        noData=nodata,
        outputType=outdtype,
        projWinSRS=projection,
        outputSRS=projection,
    )

    xshape = tempds.RasterXSize
    yshape = tempds.RasterYSize

    if usecutline:
        """Crop tif with shape"""
        tempds = gdal.Warp(
            outDS,
            tempds,
            format=outformat,
            srcNodata=srcnodata,
            dstNodata=nodata,
            dstSRS=projection,
            outputBounds=outputbounds,
            width=xshape,
            height=yshape,
            cutlineDSName=shapefile,
        )

    tempds = None

    return


def create_cutline_vrt(
    inDS,
    shapefile,
    outDS,
    srcDS=None,
    buffer=5,
    srcnodata=None,
    nodata=-9999,
    dstSRS="EPSG:4326",
    outdtype=gdal.GDT_Float32,
    outformat="VRT",
):
    """
    Use a shape file to crop a raster


    Parameters
    ----------
    inDS : path
        input raster
    shapefile : path
        input shapefile
    outDS : path
        output raster clipped to shape
    srcDS : path, optional
        raster to provide projection,size,etc of output, by default None
    nodata : int, optional
        no data value for output tif, by default -9999
    dstSRS : str, optional
        output projection, by default "EPSG:4326"
    outdtype : gdal data type, optional
        output data type, by default gdal.GDT_Float32
    outformat : str, optional
        output gdal format, by default "VRT"
    rsAlg : gdal.alg, optional
        gdal resampling algorithm, by default None
    """

    if srcDS is not None:

        """
        Get resampling projection and size from a raster file
        """
        srcDS = gdal.Open(srcDS, gdal.GA_ReadOnly)
        xshape = srcDS.RasterXSize
        yshape = srcDS.RasterYSize
        projection = srcDS.GetProjectionRef()
        geoT = srcDS.GetGeoTransform()
        xmin = geoT[0]
        ymax = geoT[3]
        xmax = xmin + geoT[1] * xshape
        ymin = ymax + geoT[5] * yshape
        srcDS = None

    else:

        shape_gdf = gpd.read_file(shapefile)
        minx, miny, maxx, maxy = shape_gdf.total_bounds

        """Pull out polygon bounding box. Clip VRT to bbox, then use cutline"""
        xmin = minx - buffer
        ymax = maxy + buffer
        xmax = maxx + buffer
        ymin = miny - buffer
        xshape, yshape = 0, 0
        projection = dstSRS

    projection = dstSRS

    """Crop vrt with shape"""
    tempds = gdal.Warp(
        outDS,
        inDS,
        format=outformat,
        width=xshape,
        height=yshape,
        srcNodata=srcnodata,
        dstNodata=nodata,
        dstSRS=projection,
        cutlineDSName=shapefile,
    )

    tempds = None

    return


def buffer_gdf2shp(geo_df, buffer_m, shp_file, capstyle=1, joinstyle=1):
    """
    Buffer a geodataframe and save to a shape file

    Parameters
    ----------
    geo_df : geodataframe
    buffer_m : float
        buffer to apply to geodataframe geometries (meters)
    shp_file : path
        output shape file

    """
    # bounds = geo_df.bounds.values[0]
    bounds = geo_df.total_bounds
    latmid = (bounds[1] + bounds[3]) / 2
    lonmid = (bounds[0] + bounds[2]) / 2

    """Transform shapes to UTM for buffering"""
    zone = np.int(np.round((183 + lonmid) / 6))
    if latmid < 0:
        UTMepsg = 32600 + zone
    elif latmid >= 0:
        UTMepsg = 32700 + zone

    buffer_gdf = geo_df.copy()
    buffer_gdf = buffer_gdf.reset_index(drop=True)

    if len(buffer_gdf.index) > 1:  # dissolve to one row
        buffer_gdf["group"] = 1
        buffer_gdf = buffer_gdf.dissolve(by="group")
        buffer_gdf = buffer_gdf.reset_index(drop=True)

    buffer_UTM_geom = buffer_gdf.geometry.to_crs("epsg:{0}".format(UTMepsg))
    basin_poly = list(buffer_UTM_geom.geometry)[0]

    if basin_poly.geom_type == "MultiPolygon":

        buffer_multi_gdf = buffer_gdf.explode()
        buffer_multi_gdf = buffer_multi_gdf[["geometry"]]

        for index, row in buffer_multi_gdf.iterrows():  # buffer each polygon
            test_gdf = buffer_multi_gdf.loc[[index], :].copy()
            test_UTM_geom = test_gdf.geometry.to_crs(
                "epsg:{0}".format(UTMepsg))
            test_UTM_geom_buffer = test_UTM_geom.buffer(
                buffer_m, cap_style=capstyle, join_style=joinstyle
            )
            test_UTM_geom_buffer = test_UTM_geom_buffer.apply(
                lambda p: close_holes(p))
            test_buffer_wgs = test_UTM_geom_buffer.to_crs("epsg:4326")
            buffer_multi_gdf.loc[[index], "geometry"] = test_buffer_wgs

        buffer_multi_gdf["group"] = 1
        buffer_gdf = buffer_multi_gdf.dissolve(by="group")

    elif basin_poly.geom_type == "Polygon":
        buffer_UTM_geom = buffer_UTM_geom.buffer(
            buffer_m, cap_style=capstyle, join_style=joinstyle
        )
        buffer_UTM_geom = buffer_UTM_geom.apply(lambda p: close_holes(p))
        buffer_wgs = buffer_UTM_geom.to_crs("epsg:4326")
        buffer_gdf["geometry"] = buffer_wgs

    buffer_gdf.to_file(shp_file)


def buffer_gdf(geo_df, buffer_m, capstyle=1, joinstyle=1):
    """
    Buffer a geodataframe

    Parameters
    ----------
    geo_df : geodataframe
    buffer_m : float
        buffer to apply to geodataframe geometries (meters)

    Returns
    -------
    geodataframe
        geodataframe with buffer applied to geometries
    """

    # bounds = geo_df.bounds.values[0]
    bounds = geo_df.total_bounds
    latmid = (bounds[1] + bounds[3]) / 2
    lonmid = (bounds[0] + bounds[2]) / 2

    """Transform shapes to UTM for buffering"""
    zone = np.int(np.round((183 + lonmid) / 6))
    if latmid < 0:
        UTMepsg = 32600 + zone
    elif latmid >= 0:
        UTMepsg = 32700 + zone

    buffer_gdf = geo_df.copy()
    buffer_gdf = buffer_gdf.reset_index(drop=True)

    if len(buffer_gdf.index) > 1:  # dissolve to one row
        buffer_gdf["group"] = 1
        buffer_gdf = buffer_gdf.dissolve(by="group")
        buffer_gdf = buffer_gdf.reset_index(drop=True)

    buffer_UTM_geom = buffer_gdf.geometry.to_crs("epsg:{0}".format(UTMepsg))
    basin_poly = list(buffer_UTM_geom.geometry)[0]

    if basin_poly.geom_type == "MultiPolygon":

        buffer_multi_gdf = buffer_gdf.explode()
        buffer_multi_gdf = buffer_multi_gdf[["geometry"]]

        for index, row in buffer_multi_gdf.iterrows():  # buffer each polygon
            test_gdf = buffer_multi_gdf.loc[[index], :].copy()
            test_UTM_geom = test_gdf.geometry.to_crs(
                "epsg:{0}".format(UTMepsg))
            test_UTM_geom_buffer = test_UTM_geom.buffer(
                buffer_m, cap_style=capstyle, join_style=joinstyle
            )
            test_UTM_geom_buffer = test_UTM_geom_buffer.apply(
                lambda p: close_holes(p))
            test_buffer_wgs = test_UTM_geom_buffer.to_crs("epsg:4326")
            buffer_multi_gdf.loc[[index], "geometry"] = test_buffer_wgs

        buffer_multi_gdf["group"] = 1
        buffer_gdf = buffer_multi_gdf.dissolve(by="group")

    elif basin_poly.geom_type == "Polygon":
        buffer_UTM_geom = buffer_UTM_geom.buffer(
            buffer_m, cap_style=capstyle, join_style=joinstyle)
        buffer_UTM_geom = buffer_UTM_geom.apply(lambda p: close_holes(p))
        buffer_wgs = buffer_UTM_geom.to_crs("epsg:4326")
        buffer_gdf["geometry"] = buffer_wgs

    return buffer_gdf


def buffer_gdf_simple(geo_df, buffer_dist, capstyle=1, joinstyle=1, merge=False,
                      projected=True):
    """
    Buffer all geometries in a geodataframe without closing holes, dissolving or exploding

    Parameters
    ----------
    geo_df : geodataframe
    buffer_dist : float
        buffer to apply to geodataframe geometries (meters or degrees)

    Returns
    -------
    geodataframe
        geodataframe with buffer applied to geometries
    """
    buffer_gdf = geo_df.copy()
    if projected:
        # bounds = geo_df.bounds.values[0]
        bounds = geo_df.total_bounds
        latmid = (bounds[1] + bounds[3]) / 2
        lonmid = (bounds[0] + bounds[2]) / 2
        
        """Transform shapes to UTM for buffering"""
        zone = np.int(np.round((183 + lonmid) / 6))
        if latmid < 0:
            UTMepsg = 32600 + zone
        elif latmid >= 0:
            UTMepsg = 32700 + zone
  
        buffer_UTM_geom = buffer_gdf.geometry.to_crs("epsg:{0}".format(UTMepsg))
        buffer_UTM_geom = buffer_UTM_geom.buffer(
            buffer_dist, cap_style=capstyle, join_style=joinstyle)
        buffer_wgs = buffer_UTM_geom.to_crs("epsg:4326")
        buffer_gdf["geometry"] = buffer_wgs
    else:
        buffer_gdf["geometry"] = geo_df["geometry"].buffer(buffer_dist)
        
    if merge:
        buffer_gdf['group'] = 1
        buffer_gdf = buffer_gdf.dissolve(by='group')
        #buffer_gdf = buffer_gdf.explode()
        buffer_gdf.reset_index(drop=True)

    return buffer_gdf


def close_holes(poly: Polygon) -> Polygon:
    """
    Close polygon holes by limitation to the exterior ring.
    Args:
        poly: Input shapely Polygon
    Example:
        df.geometry.apply(lambda p: close_holes(p))
    """
    return Polygon(poly.exterior)


def poly_rasterize(input_shp, stencil_tif, output_tif, attr='type'):

    srcDS = gdal.Open(stencil_tif, gdal.GA_ReadOnly)
    xshape = srcDS.RasterXSize
    yshape = srcDS.RasterYSize
    geoT = srcDS.GetGeoTransform()
    xmin = geoT[0]
    ymax = geoT[3]
    xmax = xmin + geoT[1] * xshape
    ymin = ymax + geoT[5] * yshape
    srcDS = None

    rstr_command = f"gdal_rasterize -a {attr} -ot Byte -ts {xshape} {yshape}\
        -te {xmin} {ymin} {xmax} {ymax} {input_shp} {output_tif}"
    subprocess.call(rstr_command, shell=True,
                    stdout=subprocess.DEVNULL)
