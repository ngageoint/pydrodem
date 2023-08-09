#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
# functions for working with AWS S3 
###############################################################################

import os
import glob
import logging
import boto3
import botocore
from botocore.exceptions import ClientError
from configparser import ConfigParser, ExtendedInterpolation
from multiprocessing import Pool, cpu_count

from osgeo import gdal


def does_key_exist_s3(bucket, key):

    s3 = boto3.resource('s3')
    try:
        s3.Object(bucket, key).load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise
    else:
        return True


def get_dir_list_files(bucket, s3_path, searchtxt=None, excludetxt=None,
                       vsi=False):
    """Get list of file names recursively from AWS S3"""

    if s3_path[-1] != "/":
        s3_path = f"{s3_path}/"

    s3 = boto3.client('s3')

    # override AWS 1000 object limit
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=s3_path)

    files = []
    try:
        for page in pages:
            for f in page['Contents']:
                path = f["Key"]
                filename = path.split("/")[-1]
                if len(filename) > 3:
                    if (searchtxt is None) or (searchtxt in path):
                        if (excludetxt is None) or (excludetxt not in path):
                            files.append(path)

    except KeyError:  # empty prefix
        return files

    if vsi:
        files = [f"/vsis3/{bucket}/{file}" for file in files]

    return files


def get_dir_list_dirs(bucket, s3_path, searchtxt=None, excludetxt=None):
    """Get list of folder names one level deep from AWS S3"""
    if s3_path[-1] != "/":
        s3_path = f"{s3_path}/"

    client = boto3.client('s3')
    # override AWS 1000 object limit
    paginator = client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=s3_path)
    
    dirnames = []
    try:
        for page in pages:
            dirs = page['CommonPrefixes']
            for d in dirs:
                prefix = d['Prefix']
                if (searchtxt is None) or (searchtxt in prefix):
                    if (excludetxt is None) or (excludetxt not in prefix):
                        dirnames.append(prefix)
        
    except KeyError:  # empty prefix
        return dirnames

    return dirnames


def buildprefixlist(bucket, s3_path, list_out, searchtxt=None, excludetxt=None):

    prefix_list = get_dir_list_files(bucket, s3_path,
                                     searchtxt=searchtxt,
                                     excludetxt=excludetxt)
    with open(list_out, 'w') as filehandle:
        for path in prefix_list:
            filehandle.write("{0}\n".format(path))


def download_file_s3(bucket, src_prefix, dst_dir,
                     filename=None, overwrite=False,
                     fsearch=None, verbose=False):
    try:
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        if filename is None:
            filename = src_prefix.split("/")[-1]
            if len(filename) == 0:
                return
        out_f = os.path.join(dst_dir, filename)
        if (not overwrite) and (os.path.exists(out_f)):
            return
        # if fname is None, download all files, else filter by fname
        if (fsearch is None) or (fsearch in src_prefix):
            s3 = boto3.resource('s3')
            s3.meta.client.download_file(bucket, src_prefix, out_f)
    except Exception as e:
        if verbose:
            print(f"ERROR in download function: {e}")


def download_dir_contents_s3(src_path, dst_path, bucket=None,
                             fsearch=None, check_file_sum=False,
                             overwrite=False, threading=False,
                             nthreads=8):
    """
    Ignores directories
    
    Parameters
    ----------
    src_path : str
        s3 prefix to download contents from.
    dst_path : str
        local destination.
    bucket : str, optional
        s3 bucket name. The default is None.
    fname : str, optional
        filter/search term for s3 objects. The default is None (pulls everything).
    check_overwrite_file : str, optional
        specific file, if exists in dst dir, abort. The default is None.
    overwrite : boolean, optional
        overwrite existing files with same name?. The default is False.
        
    Returns
    -------
    None.
    
    """
    
    """ download temp file to local notebook """
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        
    # pull list of file names from directory
    dir_contents = get_dir_list_files(bucket, src_path)
    n_contents = len(dir_contents)
    
    # if an overwrite file is specified and already exists, exit
    if check_file_sum:
        local_files = glob.glob(os.path.join(dst_path, '*'))
        n_local_files = len(local_files)
        
        if n_local_files == n_contents:
            print("     local files already downloaded...")
            return
        
    if threading:
        with Pool(nthreads) as p:
            p.starmap(download_file_s3, [(bucket, f, dst_path, None,
                                          overwrite, fsearch)
                                         for f in dir_contents])
            
    for f in dir_contents:
        download_file_s3(bucket, f, dst_path,
                         overwrite=overwrite, fsearch=fsearch)


def upload_file_s3(file_name, bucket, prefix, object_name=None,
                   acl_policy='bucket-owner-full-control', cred_file=None,
                   overwrite=False):
    """Upload a file to an S3 bucket
    
    Parameters
    -----------
    file_name : str
        File to upload
    bucket: str
        AWS Bucket to upload to
    object_name: str
        S3 object name. If not specified then file_name is used
    """
    
    if os.path.exists(cred_file):  # credential file passed
        config = ConfigParser(allow_no_value=True,
                              interpolation=ExtendedInterpolation())
        config.read(cred_file)
        ACCESS_KEY = config.get("default", "aws_access_key_id")
        SECRET_KEY = config.get("default", "aws_secret_access_key")
        s3_client = boto3.client('s3',
                                 aws_access_key_id=ACCESS_KEY,
                                 aws_secret_access_key=SECRET_KEY)
        
    else:
        s3_client = boto3.client('s3')
        
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)
        
    object_s3_path = f"{prefix}/{object_name}"
    if not overwrite:
        file_exists = does_key_exist_s3(bucket, object_s3_path)
        if file_exists:
            print(f'   {object_s3_path} already exists')
            return
        
    try:
        s3_client.upload_file(file_name, bucket, object_s3_path,
                              ExtraArgs={'ACL': acl_policy})
    except ClientError as e:
        logging.error(e)


def upload_dir_s3(dir_path, bucket, s3_prefix, search=None,
                  del_local=True, cred_file=None, overwrite=False):
    
    if search is None:
        search = "*"
        
    files_2_upload = glob.glob(os.path.join(dir_path, search))
    
    for f in files_2_upload:
        upload_file_s3(f, bucket, s3_prefix, cred_file=cred_file,
                       overwrite=overwrite)
        if del_local:
            os.remove(f)  # delete local copy


def change_owner_s3_object(bucket, obj, owner):

    s3_resource = boto3.resource("s3")
    try:
        """check ownership"""
        obj_acl = s3_resource.ObjectAcl(bucket, obj)
        if obj_acl.owner['DisplayName'] == owner:
            return
        copy_source = {'Bucket': bucket,
                       'Key': obj}
        
        bucket_resource = s3_resource.Bucket(bucket)
        bucket_resource.copy(copy_source, obj)
        
    except Exception as e:
        print(e)
        return


def change_owner_s3_prefix(bucket, s3_prefix, verbose=True,
                           owner='uc2s_content_ops', parallel=False):
    
    s3_files = get_dir_list_files(bucket, s3_prefix)
    print(f'Changing owner for {len(s3_files)} files in {bucket}/{s3_prefix}')
    
    if parallel:
        nthreads = cpu_count()
        
        print(
            f"   change owner for {len(s3_files)} files on {nthreads} threads")
        with Pool(nthreads) as p:
            p.starmap(change_owner_s3_object, [(bucket, obj, owner)
                                               for obj in s3_files])
            
    for obj in s3_files:
        change_owner_s3_object(bucket, obj, owner)


"""Datasest-specific functions"""

def build_vrt_s3(bucket, vrt_dir_prefix, vrt_out,
              searchtxt='tif', excludetxt=None):
    
    tifs = get_dir_list_files(bucket, vrt_dir_prefix,
                              searchtxt=searchtxt, 
                              excludetxt=excludetxt, 
                              vsi=True)
    
    vrt_tif = gdal.BuildVRT(vrt_out, tifs)
    vrt_tif = None


def build_vrt_s3_tilelist(bucket, vrt_dir_prefix, vrt_out, tiles,
              searchtxt='tif', excludetxt=None, path_form=None):
    
    tifs = []
    for tile in tiles:
        
        if path_form:
            tif = path_form.replace('TILEID', f'{tile}')
            tif = f"{vrt_dir_prefix}/{tif}"
            tif_exists = does_key_exist_s3(bucket, tif)
            if not tif_exists: #try version 01
                tif = path_form.replace('_02_', '_01_')
                tif_exists = does_key_exist_s3(bucket, tif)
            if tif_exists:
                tif = f"/vsis3/{bucket}/{tif}"
            else:
                print(f'{tif} DOES NOT EXIST')
                continue
        
        else:
            tile_prefix = get_dir_list_dirs(bucket, vrt_dir_prefix,
                                            searchtxt=f"{tile}")[0]
            tif = get_dir_list_files(bucket, tile_prefix,
                                     searchtxt=searchtxt, 
                                     excludetxt=excludetxt, vsi=True)[0]
            
        tifs.append(tif)
    
    vrt_tif = gdal.BuildVRT(vrt_out, tifs)
    vrt_tif = None


def pull_DEM_tile(tile, save_dir, bucket, s3_path,
                  overwrite=True):
    
    """Download DEM from s3"""
    s3_tile_path = f"{s3_path}/{tile}/"
    tile_files = get_dir_list_files(bucket, s3_tile_path)
        
    dem_tile_s3_obj = get_dir_list_files(bucket, s3_tile_path,
                                         searchtxt="DEM.tif")[0]
    
    fname = f"DEM_{tile}.tif"
    tile_local_tif = os.path.join(save_dir, fname)
    if overwrite or not os.path.exists(tdt_tile_local_tif):
        download_file_s3(bucket, dem_tile_s3_obj, save_dir,
                         filename=fname, overwrite=overwrite)
        
    return tile_local_tif


def pull_worldcover_tile(tile, save_dir, bucket, s3_path,
                        overwrite=True):
    """Download worldcover tile from s3"""
    
    s3_path = f"{s3_path}/LULC/worldcover-esa/map"
    NS, EW = tile[0:1], tile[3:4]
    lat, lon = int(tile[1:3]), int(tile[4:7])
    
    if NS == "S":
        lat = ((lat+2)//3)*3
    else:
        lat = ((lat)//3)*3
    if EW == "W":
        lon = ((lon+2)//3)*3
    else:
        lon = ((lon)//3)*3
        
    wc_fname = f"ESA_WorldCover_10m_2020_v100_{NS}{lat:02}{EW}{lon:03}_Map.tif"
    wc_tile_s3_obj = f"{s3_path}/{wc_fname}"
    wc_tile_local_tif = os.path.join(save_dir, wc_fname)
    if overwrite or not os.path.exists(wc_tile_local_tif):
        download_file_s3(bucket, wc_tile_s3_obj, save_dir,
                         filename=wc_fname, overwrite=overwrite)
        
    return wc_tile_local_tif


def pull_merit_tile(tile, save_dir, bucket, s3_path, overwrite=True):
    """
    Download MERIT DEM tile from S3
    
    Parameters
    ----------
    tile : str
        tile name.
    save_dir : path
        output directory.
    overwrite : boolean, optional
        overwrite existing file?. The default is True.
        
    Returns
    -------
    merit_local_tif : str
        path to output tif
        
    """
    
    s3_path = f"{s3_path}/MERIT-Hydro/elevation"
    NS, EW = tile[0:1], tile[3:4]
    lat, lon = int(tile[1:3]), int(tile[4:7])
    
    if NS == "S":
        lat = ((lat+4)//5)*5
        NS = 's'
    else:
        lat = ((lat)//5)*5
        NS = 'n'
    if EW == "W":
        lon = ((lon+4)//5)*5
        EW = 'w'
    else:
        lon = ((lon)//5)*5
        EW = 'e'
        
    #print(f" MERIT NS, EW = {NS},{EW}...lat/lon = {lat},{lon}")
    fname = f"{NS}{lat:02}{EW}{lon:03}_elv.tif"
    tile_s3_obj = f"{s3_path}/{fname}"
    tile_local_tif = os.path.join(save_dir, fname)
    if overwrite or not os.path.exists(tile_local_tif):
        download_file_s3(bucket, tile_s3_obj, save_dir,
                         overwrite=overwrite)
        
    return tile_local_tif
