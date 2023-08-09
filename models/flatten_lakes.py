#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 30 2021

@author: Kimberly McCormack

script to flatten large lakes 

"""
import os
import warnings
warnings.simplefilter('ignore')
import sys
import time
import geopandas as gpd
from configparser import ConfigParser, ExtendedInterpolation
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.water import find_large_lakes, flatten_lakes
from tools.print import print_time, log_time

starttime = time.time()


""" Load the configuration file """
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True, help="config file")
ap.add_argument("-l", "--logfile", required=True, help="log file")
args = vars(ap.parse_args())
config_file = os.path.abspath(args['config'])
log_file = os.path.abspath(args['logfile'])

"""read config file"""
config = ConfigParser(allow_no_value=True, interpolation=ExtendedInterpolation())
config.read(config_file)

projectname = config.get('outputs', 'projectname')
work_dir = os.path.join(config.get("paths", "output_dir"), 'lake_flatten', projectname)
lake_shp_dir = os.path.join(work_dir, "shapefiles")
lake_dem_dir = os.path.join(work_dir, "DEMs")

overwrite_lakes = config.getboolean("processes-depression-handling", "overwrite_lakes")
SWO_cutoff = config.getint("parameters-depression-handling", "SWO_cutoff")
SWO_burn = config.getfloat("parameters-depression-handling", "SWO_burn")
SWO_vrt = os.path.join(config.get("paths", "SWO_vrt"))
flatten_lakes_shp = os.path.join(lake_shp_dir, 'OSM_large_lakes.shp')
inputDEM = os.path.join(config.get('outputs', 'dh_mosaic'))
aoi_shpfile = os.path.join(config.get('outputs', 'aoi_shapefile'))

"""locations of PSM data"""
OSM_dir = config.get("paths", "OSM_dir")
OSM_tiles = config.get("paths", "OSM_shp")


if not os.path.exists(lake_shp_dir): os.makedirs(lake_shp_dir)        
if not os.path.exists(lake_dem_dir): os.makedirs(lake_dem_dir)

if not os.path.exists(flatten_lakes_shp):
    print_time('   Creating large lake shape file', starttime)
    large_lakes_gdf = find_large_lakes(OSM_dir, OSM_tiles, lake_shp_dir, aoi_shpfile,
                        flatten_lakes_shp, area_threshold=500, lake_buffer=1e2)
else:
    print('   Large lake shapefile exists... loading')
    large_lakes_gdf = gpd.read_file(flatten_lakes_shp)

if large_lakes_gdf is not None:                                
    print('   Large lakes found')
    flatten_lakes(flatten_lakes_shp, lake_dem_dir, lake_shp_dir, 
                  inputDEM, SWO_vrt, SWO_cutoff, 
                  SWO_burn, overwrite_lakes=overwrite_lakes)
    print_time('Large lakes re-flattened', starttime)
else:
    print('   No large lakes found')
    

log_str = log_time("LAKE FLATTENING COMPLETE", starttime)
with open(log_file, 'w') as f: f.write(log_str)

