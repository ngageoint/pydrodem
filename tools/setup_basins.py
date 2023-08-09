#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions to setup directories and basin shapes
"""

import os
import warnings
warnings.simplefilter('ignore')
import traceback
import numpy as np
import geopandas as gpd
from shapely import geometry








def basin_setup_single_lvl(basin_id, basins_gdf, basin_work_dir):
    """
    Set up basin directories, shape files and buffered shape files for 
    each basin for a single hydrobasin level

    Parameters
    ----------
    basin_id : int
        ID number for basin 
    basins_gdf : geodataframe
        geodataframe contains all of the basins for a given hydrobasin level

    """
    try:
        outpath_basin = os.path.join(basin_work_dir, 'basin_{0}'.format(basin_id))
        if not os.path.exists(outpath_basin):
            os.mkdir(outpath_basin)

        basin_shpfile = os.path.join(
            outpath_basin, 'basin_{0}.shp'.format(basin_id))

        if not os.path.exists(basin_shpfile):
            basin_gdf = basins_gdf[basins_gdf.HYBAS_ID == basin_id].copy()
            basin_gdf.crs = "EPSG:4326"
            basin_gdf.to_file(basin_shpfile)

            
    except Exception as e:
        outst = "Exception occurred on basin{0}: {1}".format(basin_id, e)
        traceback_output = traceback.format_exc()
        print(outst)
        print("TRACEBACK : {0}\n".format(traceback_output))


def basin_setup_multi_lvl(basins_gdf, basin_id, basins_p1_gdf, 
                          basins_p2_gdf, basin_work_dir,
                          levels=[6, 7, 8], threshold=6.0,
                          basins_p3_gdf=None, verbose_print=False):
    """
    Set up basin directories, shape files and buffered shape files for each basin.
    If a basin is too large (based on bounding box area), break up into next level basins.

    Designed to use 3 levels of hydrobasins

    Parameters
    ----------
    basin_id : int
        ID number for basin
    basins_p1_shp : path
        shape file containing plus one level basins
    basins_p2_shp : path
        shape file containing plus two level basins
    levels : list, optional
        hydrobasin levels to process, by default [6, 7, 8]
    threshold : float, optional
        size limit (in degree area) for bounding box of basin shape - 
        serves as a proxy for geotiff size

    """

    try:
        
        n_lvl = len(levels)
        bas_lvl = np.int(str(basin_id)[2])
        if bas_lvl == levels[0]:
            basin_gdf = basins_gdf[basins_gdf.HYBAS_ID == basin_id].copy()
            basin_buffer_gdf = basin_gdf.copy()
            basin_buffer_gdf['geometry'] = basin_buffer_gdf.geometry.buffer(0.01)
            bounds = basin_buffer_gdf.bounds.values[0]
            """ if the level 6 basin is too big, break up to next basin level"""
            latrange = np.abs(bounds[1] - bounds[3])
            lonrange = np.abs(bounds[0] - bounds[2])
            deg_area = latrange*lonrange
            
            if deg_area > threshold:
                """find level p1 (plus one) basins within basin"""
                basin_poly = list(basin_buffer_gdf.geometry)[0]
                p1_basins_gdf = basins_p1_gdf.loc[basins_p1_gdf.geometry.within(
                    basin_poly)]
                p1_basins_gdf.crs = "EPSG:4326"
                p1_basins = list(p1_basins_gdf.HYBAS_ID.values)
                basin_gdf = p1_basins_gdf
                
                for p1_basin in p1_basins:
                    p1_basin_gdf = p1_basins_gdf[p1_basins_gdf.HYBAS_ID == p1_basin].copy()
                    p1_basin_gdf['geometry'] = p1_basin_gdf.geometry.buffer(
                        0.01)  # 1km buffer 
                    bounds = p1_basin_gdf.bounds.values[0]
                    latrange = np.abs(bounds[1] - bounds[3])
                    lonrange = np.abs(bounds[0] - bounds[2])
                    deg_area = latrange*lonrange
                    
                    if deg_area > threshold:
                        #print( ' moving to level 8 basins for {0}'.format(p1_basin))
                        """find level p2 (plus two) basins within p1 basin"""
                        p1_basin_poly = list(p1_basin_gdf.geometry)[0]
                        p2_basins_gdf = basins_p2_gdf.loc[basins_p2_gdf.geometry.within(
                            p1_basin_poly)]
                        p2_basins_gdf.crs = "EPSG:4326"
   
                        """drop lvl 7 basin and add its lvl 8 subbasins"""
                        basin_gdf = basin_gdf.drop(
                            basin_gdf[basin_gdf.HYBAS_ID == p1_basin].index)
                        basin_gdf = basin_gdf.append(p2_basins_gdf)
                        
                        if n_lvl == 4:
                            p2_basins = list(p2_basins_gdf.HYBAS_ID.values)
                            for p2_basin in p2_basins:
                                p2_basin_gdf = p2_basins_gdf[p2_basins_gdf.HYBAS_ID == p2_basin].copy()
                                p2_basin_gdf['geometry'] = p2_basin_gdf.geometry.buffer(0.01)
                                bounds = p2_basin_gdf.bounds.values[0]
                                latrange = np.abs(bounds[1] - bounds[3])
                                lonrange = np.abs(bounds[0] - bounds[2])
                                deg_area = latrange*lonrange
                                
                                if deg_area > threshold:
                                    p2_basin_poly = list(p2_basin_gdf.geometry)[0]
                                    p3_basins_gdf = basins_p3_gdf.loc[basins_p3_gdf.geometry.within(
                                        p2_basin_poly)]
                                    p3_basins_gdf.crs = "EPSG:4326"
                                    """drop lvl 8 basin and add its lvl 9 subbasins"""
                                    basin_gdf = basin_gdf.drop(
                                        basin_gdf[basin_gdf.HYBAS_ID == p2_basin].index)
                                    basin_gdf = basin_gdf.append(p3_basins_gdf)
                        

        elif bas_lvl == levels[1]:
            basin_gdf = basins_p1_gdf[basins_p1_gdf.HYBAS_ID == basin_id].copy()
            basin_buffer_gdf = basin_gdf.copy()
            basin_buffer_gdf['geometry'] = basin_buffer_gdf.geometry.buffer(0.01)  # 1km buffer
            bounds = basin_buffer_gdf.bounds.values[0]
            latrange = np.abs(bounds[1] - bounds[3])
            lonrange = np.abs(bounds[0] - bounds[2])
            deg_area = latrange*lonrange
            if deg_area > threshold:
                #print('    moving from 7 to level 8 basins for {0}'.format(basin_id))
                """find level p2 (plus two) basins within p1 basin"""
                p1_basin_poly = list(basin_buffer_gdf.geometry)[0]
                p2_basins_gdf = basins_p2_gdf.loc[basins_p2_gdf.geometry.within(p1_basin_poly)]
                """replace lvl 7 basin with lvl 8 subbasins"""
                basin_gdf = p2_basins_gdf
        elif bas_lvl == levels[2]:
            basin_gdf = basins_p2_gdf[basins_p2_gdf.HYBAS_ID == basin_id].copy()
        elif bas_lvl == levels[3]:
            basin_gdf = basins_p3_gdf[basins_p3_gdf.HYBAS_ID == basin_id].copy()            
        single_basins = list(basin_gdf.HYBAS_ID.values)
        num_basins = len(single_basins)
        if verbose_print:
            print("For basin {0} - processing {1} total basins".format(
                        basin_id, num_basins))
        for s_basin in single_basins:
            basin_setup_single_lvl(s_basin, basin_gdf, basin_work_dir)

    except Exception as e:
        outst = "Exception occurred on basin {0}: {1}".format(basin_id, e)
        traceback_output = traceback.format_exc()
        print(outst)
        print("TRACEBACK : {0}\n".format(traceback_output))


def check_setup(basins_gdf, basins_p1_gdf, basins_p2_gdf, basin_work_dir, basins_p3_gdf=None,
                cutline_field='HYBAS_ID'):
    lvl8_missing = []
    lvl7_missing = []
    lvl6_missing = []
    
    if basins_p3_gdf is not None:
        lvl9_missing = []
    

    lvl6_basinIDs = list(basins_gdf[cutline_field].values)
    nbasins = len(lvl6_basinIDs)
    print('CHECKING BASINS: {0} basins'.format(nbasins))
    for i, lvl6_basinID in enumerate(lvl6_basinIDs):
        basin_path = os.path.join(
            basin_work_dir, 'basin_{0}'.format(lvl6_basinID))
    
        if not os.path.exists(basin_path):
            #print('  {0}/{1}... moving to lvl 7... - {2}'.format(i+1, nbasins, lvl6_basinID))
            lvl6_basin_gdf = basins_gdf[basins_gdf.HYBAS_ID == lvl6_basinID].copy()
            lvl6_basin_gdf['geometry'] = lvl6_basin_gdf.geometry.buffer(
                0.01)  # 1km buffer
            lvl6_basin_poly = list(lvl6_basin_gdf.geometry)[0]
            lvl7_basin_gdf = basins_p1_gdf.loc[basins_p1_gdf.geometry.within(lvl6_basin_poly)]
            lvl7_basinIDs = list(lvl7_basin_gdf.HYBAS_ID.values)
            nbasins7 = len(lvl7_basinIDs)
            lvl7_existing = 0
            lvl7_add = []
            for j, lvl7_basinID in enumerate(lvl7_basinIDs):
                basin7_path = os.path.join(basin_work_dir, 'basin_{0}'.format(lvl7_basinID))
                if os.path.exists(basin7_path):                    
                    lvl7_existing += 1

                if not os.path.exists(basin7_path):                  
                    lvl7_basin_gdf = basins_p1_gdf[basins_p1_gdf.HYBAS_ID == 
                                                     lvl7_basinID].copy()
                    lvl7_basin_gdf['geometry'] = lvl7_basin_gdf.geometry.buffer(0.01)
                    lvl7_basin_poly = list(lvl7_basin_gdf.geometry)[0]
                    lvl8_basin_gdf = basins_p2_gdf.loc[basins_p2_gdf.geometry.within(
                                                        lvl7_basin_poly)]
                    lvl8_basinIDs = list(lvl8_basin_gdf.HYBAS_ID.values)
                    nbasins8 = len(lvl8_basinIDs)

                    lvl8_existing = 0
                    lvl8_add = []
                    for k, lvl8_basinID in enumerate(lvl8_basinIDs):  
                        basin8_path = os.path.join(basin_work_dir, 'basin_{0}'.format(lvl8_basinID))
                        if os.path.exists(basin8_path):
                            lvl8_existing += 1
                        if not os.path.exists(basin8_path):
                            
                            
                            if basins_p3_gdf is not None:
         
                                lvl8_basin_gdf = basins_p2_gdf[basins_p2_gdf.HYBAS_ID == 
                                                     lvl8_basinID].copy()
                                lvl8_basin_gdf['geometry'] = lvl8_basin_gdf.geometry.buffer(0.01)
                                lvl8_basin_poly = list(lvl8_basin_gdf.geometry)[0]



                                lvl9_basin_gdf = basins_p3_gdf.loc[basins_p3_gdf.geometry.within(
                                                                    lvl8_basin_poly)]
                                lvl9_basinIDs = list(lvl9_basin_gdf.HYBAS_ID.values)
                                nbasins9 = len(lvl9_basinIDs)

                                lvl9_add = []
                                for l, lvl9_basinID in enumerate(lvl9_basinIDs):  
                                    basin9_path = os.path.join(basin_work_dir, 'basin_{0}'.format(lvl9_basinID))
                                    if not os.path.exists(basin9_path):
                                        lvl9_add.append(lvl9_basinID)
                                        print('ERROR - basin {0}/{1}/{2}/{3} has no directory'.format(
                                            lvl6_basinID, lvl7_basinID, lvl8_basinID, lvl9_basinID))
                                        
                                
                                if len(lvl9_add) == nbasins9:  # missing all level 9 basins - add level 8
                                    lvl8_add.append(lvl8_basinID)
                                else:
                                    lvl9_missing.extend(lvl9_add)


                            else:
                                lvl8_add.append(lvl8_basinID)
                                print('ERROR - basin {0}/{1}/{2} has no directory'.format(
                                    lvl6_basinID, lvl7_basinID, lvl8_basinID))
         
                    
                    if len(lvl8_add) == nbasins8:  # missing all level 8 basins - add level 7
                        lvl7_add.append(lvl7_basinID)
                    else:
                        lvl8_missing.extend(lvl8_add)

                if len(lvl7_add) == nbasins7:  # missing all level 7 basins - add level 6
                    lvl6_missing.append(lvl6_basinID)
                else:
                    lvl7_missing.extend(lvl7_add)

    lvl6_missing = np.unique(np.array(lvl6_missing))
    lvl7_missing = np.unique(np.array(lvl7_missing))
    lvl8_missing = np.unique(np.array(lvl8_missing))
    print('Level 6 missing ({0} basins): '.format(
        len(lvl6_missing)), lvl6_missing)
    print('  Level 7 missing ({0} basins): '.format(
        len(lvl7_missing)), lvl7_missing)
    print('    Level 8 missing ({0} basins): '.format(
        len(lvl8_missing)), lvl8_missing)
        
    if basins_p3_gdf is not None:
        lvl9_missing = np.unique(np.array(lvl9_missing))
        print('      Level 9 missing ({0} basins): '.format(
                len(lvl9_missing)), lvl9_missing)        
    else:
        lvl9_missing = []
    
    missing_basins = list(lvl6_missing) + list(lvl7_missing) + list(lvl8_missing) + list(lvl9_missing)
    
    return missing_basins

