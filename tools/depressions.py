#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 6  2020

@author: Kimberly McCormack, Heather Levin


Depression handling functions for DEMs. Built as part of the pydroDEM hydroconditioning software


"""
import os, sys
import gc
import time
import traceback

import numpy as np
import pandas as pd
import itertools
from osgeo import gdal, gdal_array
from scipy import ndimage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.convert as pyct
import tools.files as ft
from tools.derive import apply_convolve_filter
from tools.print import print_time

# Stop GDAL printing both warnings and errors to STDERR
gdal.PushErrorHandler('CPLQuietErrorHandler')
# Make GDAL raise python exceptions for errors (warnings won't raise an exception)
gdal.UseExceptions()


###################################################################
"""            Create initial fill and carve solutions          """
###################################################################


def initial_fill_carve(inDEM_tif, basename, outpath, fill_method, maxcost, radius,
                       min_dist, nodataval, starttime, wbt, sfill=None,
                       flat_increment=0.0,
                       watermask=None, water_mask_array=None, 
                       carveout=False, carve_method='SC',
                       verbose_print=True, del_temp_files=False):
    """
    Fills and carves to handle depressions in a DEM using Whitebox Tools. 

    Parameters
    ----------
    inDEM_tif : path
        input DEM tif
    basename : str 
        filename of input DEM (without extension)
    outpath : path 
        directory to save outputs
    fill_method : str
        "SF" for standard fill or "WL" for Wang and Liu
    maxcost : float
        maximum cost for a carve in the LC function. 
        N/A for the standard carve 'SC' function
    radius : int    
        maximum search window in the carving function (pixels)
    min_dist : bool
        True= = minimize distance, False = minimize cost. Only applied in 'LC' carving
    nodataval : float
        nodata value to be passed to npy2tif
    starttime : int
        time process started for timing calculations
    wbt : class
        alias to call WhiteboxTools functions
    sfill : float, optional
        auto-fill pits with max depth greater than sfill. 
        Do not auto fill if None, by default None
    flat_increment : float, optional
        gradient to apply to flat regions 
        (0.00001 is reasonable), by default 0.0 
    watermask : path, optional
        optional tif of water bodies used to fill depressions that drain
        into water to just above the water elevation, by default None
    carveout : bool, optional
        allow carves to leave edge of DEM?, by default False
    carve_method : str, optional
        whitebox tools carving algorithm used 
        'SC' for standard carve, 'LC' for least cost, by default 'SC'
    verbose_print : bool, optional
        print progress/timestamps?, by default True
    del_temp_files : bool, optional
        delete temporary files?, by default False

    Returns
    -------
    modDEM_vec : array
        1D array of reshaped, modified DEM values
    fill_vec : array
        1D array of reshaped, filled DEM values
    carve_vec : array
        1D array of reshaped, carved DEM values
    diff_vec_fill : array
        1D array of difference values btwn filled and input DEM
    diff_vec_carve : array
        1D array of difference values btwn carved and input DEM
    fillID_vec : array
        1D array of pit ID values. Each contiguous depression is assigned an ID
        which is used to track fill/carve volumes, depths and solutions applied
        to each pit. 
    carveID_vec : array
        1D array of carve ID values
    carvefillID_vec : array
        1D array of post-carve fill ID values
    nx : int
        input raster shape in x-direction
    ny : int
        input raster shape in y-direction

    """

    
    """Load inDEM as numpy array"""
    inDEM_array = gdal_array.LoadFile(inDEM_tif)
    ny = inDEM_array.shape[0]
    nx = inDEM_array.shape[1]
    
    if water_mask_array is not None:
        water_mask_vec = np.ravel(water_mask_array)
        water_mask = np.where(water_mask_array>2, 1, 0) #binary water mask
    else:
        water_mask_vec = None
        water_mask = None
    

    """Initial fill to find lake-draining pits and shallow pits"""
    fill_array, diff_fill_array = fill_pits(inDEM_tif, wbt,
                                           fill_method=fill_method,
                                           nodataval=nodataval,
                                           flat_increment=0.0)

    """check for negative values in diff fill"""
    if np.amin(diff_fill_array) < 0.0:  # fill_diff should not be negative
        print_time('     Initial fill diff has negative values', starttime)
        fill_array, diff_fill_array = fix_shifted_data(inDEM_array, 
                                                        fill_array, nodataval)

    if verbose_print:
        print_time('    Initial fill COMPLETE', starttime)

    """Clump filled pixels to create pit IDs"""
    fillID_path = os.path.join(outpath, '{0}_pitID_initial.tif'.format(basename))
    fillID_array = create_clump_array(diff_fill_array, fillID_path, inDEM_tif, wbt,
                                      nodataval=nodataval, condition='greaterthan')
                                      #split_mask=water_mask)
    fillID_vec = fillID_array.reshape(ny*nx,)
    inDEM_vec = inDEM_array.reshape(ny*nx,)
    modDEM_vec = np.copy(inDEM_vec)
    modDEM_vec[np.where(np.isnan(modDEM_vec))] = nodataval
    
    """ Apply optional auto-fill for shallow pits"""
    if sfill is not None:
        """Automatically fill pits shallower than sfill deep (default=2m)"""
        modDEM_vec = fill_shallow_pits(modDEM_vec, sfill, fillID_vec, diff_fill_array,
                                       fill_array, water_vec=water_mask_vec,
                                       verbose_print=verbose_print)

    """Use water mask to raise pixels that drain into large lakes and river above 
    the minimum water height. This is a common artifact of removing too much vegetation
    along riparian zones. Currently only apply at level 12"""
    if watermask is not None:
        if verbose_print:
            print_time('    Adjust pits values below lakes', starttime)

        modDEM_vec, DEMdiff_lake_vec = raise_lake_connected_pits(watermask, fillID_vec,
                                                                 modDEM_vec, inDEM_vec,
                                                                 starttime=starttime,
                                                                 verbose_print=verbose_print)

        DEMdiff_lake_array = DEMdiff_lake_vec.reshape(ny, nx)
        DEMdiff_lake_tif = os.path.join(outpath, f"{basename}_lakefill_DIFF.tif")
        pyct.npy2tif(DEMdiff_lake_array, inDEM_tif, DEMdiff_lake_tif,
                     nodata=nodataval, dtype=gdal.GDT_Float32)
        del DEMdiff_lake_array

    """If needed, Run second fill using shallow and/or lake-adjacent pits filled DEM"""
    if sfill is None and watermask is None:
        if verbose_print:
            print_time('       Skipping second fill', starttime)
        modDEM_tif = inDEM_tif
    else:
        """ save lake modified array to tif and send to initial fill/carve """
        modDEM_array = modDEM_vec.reshape(ny, nx)
        modDEM_tif = os.path.join(outpath, f"{basename}_shallowfill.tif")
        pyct.npy2tif(modDEM_array, inDEM_tif, modDEM_tif,
                     nodata=nodataval, dtype=gdal.GDT_Float32)
                     
        fill_array, diff_fill_array = fill_pits(modDEM_tif, wbt,
                                               fill_method=fill_method,
                                               nodataval=nodataval,
                                               flat_increment=flat_increment)

        """check for negative values in diff fill"""
        if np.amin(diff_fill_array) < 0.0:  # fill_diff should not be negative
            print_time('      Second fill diff has negative values', starttime)
            fill_array, diff_fill_array = fix_shifted_data(modDEM_array, fill_array,
                                                           nodataval)
            print_time('     Second fill diff successfully shifted', starttime)

        if verbose_print:
            print_time('      Second fill complete', starttime)

    """Run clumping function on fill solutions"""
    fillID_path = os.path.join(outpath, '{0}_pitID.tif'.format(basename))
    fillID_array = create_clump_array(diff_fill_array, fillID_path, inDEM_tif, wbt,
                                      nodataval=nodataval, condition='greaterthan', 
                                      clumpbuffer=1) #, split_mask=water_mask)
    fillID_vec = fillID_array.reshape(ny*nx,)
    
    """Create vector of filled values and difference values"""
    fill_vec = fill_array.reshape(ny*nx,)
    diff_vec_fill = diff_fill_array.reshape(ny*nx,)
    del fill_array
    del diff_fill_array

    """BREACH PITS"""
    carve_array, diff_carve_array = breach_pits(modDEM_tif, radius, maxcost, wbt,
                                               min_dist=min_dist, nodataval=nodataval,
                                               method=carve_method, 
                                               flat_increment=flat_increment)
    carve_vec = carve_array.reshape(ny*nx,)
    diff_vec_carve = np.copy(diff_carve_array).reshape(ny*nx,)
    del carve_array

    if verbose_print:
        print_time('      Breach pits complete', starttime)

    """Run clumping function on carve solutions"""
    carveID_path = os.path.join(outpath, '{0}_carveID.tif'.format(basename))
    carve_fillID_path = os.path.join(
        outpath, '{0}_carve_fillID.tif'.format(basename))

    if not carveout:  # include nodata pixels in clump to ID carves that leave tif
        diff_carve_array[inDEM_array == nodataval] = -1.0

    carveID_array = create_clump_array(diff_carve_array, carveID_path, inDEM_tif, wbt,
                                       nodataval=nodataval, condition='lessthan')
                                       #clumpbuffer=1)
    carvefillID_array = create_clump_array(diff_carve_array, carve_fillID_path, inDEM_tif, wbt,
                                           nodataval=nodataval, condition='greaterthan')

    carveID_vec = carveID_array.reshape(ny*nx,)
    carvefillID_vec = carvefillID_array.reshape(ny*nx,)

    del diff_carve_array
    del carveID_array
    del carvefillID_array
    del fillID_array
    del inDEM_array

    """Clear out folder"""
    basename_mod = os.path.splitext(os.path.basename(modDEM_tif))[0]
    if del_temp_files:
        ft.erase_files(outpath, search='{0}_*.tif'.format(basename_mod))

    return modDEM_vec, fill_vec, carve_vec, diff_vec_fill, \
        diff_vec_carve, fillID_vec, carveID_vec, carvefillID_vec, nx, ny


###################################################################
"""              Build catalog of pits and carves               """
###################################################################


def build_pit_catalog(inDEM_vec, nodataval, carve_vec, diff_vec_carve,
                      diff_vec_fill, fillID_vec, carveID_vec,
                      carvefillID_vec, verbose_print=True, carveout=False):
    """
    Build catalog to track elevation and volume changes imposed
    by each fill and carve. The index for tracking is the ID of the
    clumped fills and carves. 

    Parameters
    ----------
    inDEM_vec : array
        1D array of input DEM values
    nodataval : float
        no data value
    carve_vec : array
        1D array of carved DEM values
    diff_vec_carve : array
        1D array for carved difference values
    diff_vec_fill : array
        1D array of filled difference values
    fillID_vec : array
        1D array of pit ID values. Each contiguous depression is assigned an ID
        which is used to track fill/carve volumes, depths and solutions applied
        to each pit. 
    carveID_vec : array
        1D array of carve ID values
    carvefillID_vec : array
        1D array of post-carve fill ID values
    verbose_print : bool, optional
        print progress timestamps?, by default True
    carveout : bool, optional
        allow carves to leave DEM?, by default False

    Returns
    -------
    DFpits : pandas dataframe
        dataframe of catalogued pits
    fillcells_dct : dict
        dictionary of indices of pixels associated with each pit
    carvecells_dct : dict
        dictionary of indices of pixels associated with each carve
    carve_fillcells_dct : dict
        dictionary of indices of pixels associated with each pit filled post-carve 

    """

    tic = time.time()
    """read pitID raster and convert to numpy vectors"""
    carveID_vec_bound = np.where((carveID_vec > 0) & (inDEM_vec != nodataval), carveID_vec, 0)
    carve_ind = np.flatnonzero((carveID_vec > 0) & (inDEM_vec != nodataval))    
    carve_fillind = np.flatnonzero(carvefillID_vec)

    # carve ID of nodata pixels (probably 1)
    if not carveout:
        carveout_ind = carveID_vec[inDEM_vec == nodataval][0]

    """ number of pits in catalog"""
    numpits = fillID_vec.max()
    pitlist = np.arange(1, numpits+1)

    """Initialize empty vectors/dictionaries"""
    fillcells_dct = dict()
    carvecells_dct = dict()
    carve_fillcells_dct = dict()
    carvecells_only_dct = dict()
    maxfill = np.empty((numpits,), dtype=np.float32)
    volfill = np.empty((numpits,), dtype=np.float32)
    maxcarve = np.empty((numpits,), dtype=np.float32)
    volcarve = np.empty((numpits,), dtype=np.float32)
    volfillbehind = np.empty((numpits,), dtype=np.float32)
    maxfillbehind = np.empty((numpits,), dtype=np.float32)
    completecarve = np.empty((numpits,), dtype=np.int8)
    
    """
    Break up into 'chunks' for cataloging. 
    This nested approach greatly speeds up searching 
    """
    chunk_size = 200
    num_chunks = np.int(numpits//chunk_size) + 1  
    print_time('       cataloging pits...', tic)        
    for ci in range(0, num_chunks):  
        # if (ci+1)%10 == 0:
            # print_time('       Cataloging chunk {0}/{1}'.format(ci+1, num_chunks), tic)

        """ find indices that include all pits within chunk"""
        pit_min =  np.maximum(1, np.int(ci*chunk_size))
        pit_max = np.minimum(numpits, (np.int((ci+1)*chunk_size)))       
        pit_chunk_list = np.arange(pit_min, pit_max+1)                            
        if pit_max == numpits:
            chunk_pit_ind = np.flatnonzero(fillID_vec >= pit_min)                       
        else:
            chunk_pit_ind = np.flatnonzero((fillID_vec >= pit_min)
                                            & (fillID_vec <= pit_max)) 
                                  
        fillID_vec_chunk = fillID_vec[chunk_pit_ind]
        carveID_pit_vec_chunk = carveID_vec_bound[chunk_pit_ind]
        carvefillID_vec_chunk = carvefillID_vec[chunk_pit_ind]
        diff_vec_fill_chunk = diff_vec_fill[chunk_pit_ind]
        
        ###find indices of carves that intersect all pits in chunk. (+15s)
        carveIDs_chunk = np.unique(carveID_pit_vec_chunk)    
        if len(carveID_pit_vec_chunk) > 0:
            carve_ID_nonzero_ind = carveID_vec_bound[carve_ind]
            carve_nonzero_ind_chunk = np.flatnonzero(np.isin(carve_ID_nonzero_ind, carveIDs_chunk))
            carve_ind_chunk = carve_ind[carve_nonzero_ind_chunk]
           
        else:
            carve_ind_chunk = []
        
        ### pull carve elevations and diffs for chunk
        carveID_vec_chunk = carveID_vec_bound[carve_ind_chunk]
        carve_vec_chunk = carve_vec[carve_ind_chunk]
        diff_vec_carve_chunk = diff_vec_carve[carve_ind_chunk]
        
        ### stand alone dct for carve cells (+2s)
        for carve_id in carveIDs_chunk:
            carvecells_chunk_id = np.flatnonzero(carveID_vec_chunk == carve_id)
            carvecells_global_id = carve_ind_chunk[carvecells_chunk_id] # global indices of carve 
            carvecells_only_dct[carve_id] = carvecells_global_id

        count_carve_intersect = 0
        count_carve_out = 0

        for pit_id in pit_chunk_list:
            i = pit_id - 1
            
            pitID_ind_chunk = np.flatnonzero(fillID_vec_chunk == pit_id)
            maxfill_add = diff_vec_fill_chunk[pitID_ind_chunk].max()
            volfill_add = np.sum(diff_vec_fill_chunk[pitID_ind_chunk])
            fillbehind_chunk = pitID_ind_chunk[carvefillID_vec_chunk[pitID_ind_chunk] > 0]              
            carve_intersect_chunk = pitID_ind_chunk[carveID_pit_vec_chunk[pitID_ind_chunk] > 0]
       
            """map back to global indices"""
            pitID_ind_global = chunk_pit_ind[pitID_ind_chunk] # global indices of pit
            fillbehind_add = chunk_pit_ind[fillbehind_chunk]
            
            """Find intersecting carves and their total extent outside of the filled pit"""
            carveIDs = np.unique(carveID_pit_vec_chunk[carve_intersect_chunk])
            carvecells_add = []

            # does carve leave basin?
            if (not carveout) and (carveout_ind in list(carveIDs)):
                numcarves = 0
                fillbehind_add = []
                count_carve_out += 1

                
            else:
                for carve in carveIDs:   
                    ### global search (+22s)
                    carvecells_global = carvecells_only_dct[carve]
                    carveID_elevs = carve_vec[carvecells_global]                    
                    carveID_maxind = carvecells_global[np.argmax(carveID_elevs)]
                    carveID_minind = carvecells_global[np.argmin(carveID_elevs)]
                    
                    # intersect (+60s)
                    carveID_ind_intersect = np.intersect1d(carvecells_global,
                                                            pitID_ind_global, 
                                                            assume_unique=True)


                    """Is the highest carve elevation inside the pit OR is the lowest carve elevation
                    outside of the pit??"""  
                    if (carveID_maxind in carveID_ind_intersect):  # max carve within pit
                        carvecells_add.extend(list(carvecells_global))

                    elif (carveID_minind not in carveID_ind_intersect):  # min carve outside pit
                        carvecells_add.extend(list(carvecells_global))

                numcarves = len(carvecells_add)

            """calculate max depth and volume for carve and set carve complete"""
            if numcarves > 0:
                maxcarve_add = abs(diff_vec_carve[carvecells_add].min())
                volcarve_add = abs(np.sum(diff_vec_carve[carvecells_add]))
                volfillbehind_add = abs(np.sum(diff_vec_carve[fillbehind_add]))
            else:
                maxcarve_add = 0
                volcarve_add = 0
                volfillbehind_add = 0
                fillbehind_add = []

            numfillbehind = len(fillbehind_add)

            """calculate max depth of fill behind the carve"""
            if numfillbehind > 0:
                maxfillbehind_add = diff_vec_carve[fillbehind_add].max()
            else:
                maxfillbehind_add = 0.0

            """
            determine if carve was complete - did the lowest pit cell need to be filled?
            UPDATE ---this seems to be a bad metric for the standard carve algorithm. 
            Allow the carve solution to have fill behind it, and compare the total (carve + fill)
            volume (and depth) change to the fully filled volume.   
            """
            if numcarves > 0:
                completecarve_add = 1
            else:
                completecarve_add = 0
             
            # set lists to int32 arrays - list of lists breaks RAM
            pitID_ind_global = (np.array(pitID_ind_global)).astype(np.uint32)
            carvecells_add = (np.array(carvecells_add)).astype(np.uint32)
            fillbehind_add = (np.array(fillbehind_add)).astype(np.uint32)

            """Add to dictionary/vectors"""
            fillcells_dct[pit_id] = pitID_ind_global
            carvecells_dct[pit_id] = carvecells_add
            carve_fillcells_dct[pit_id] = fillbehind_add

            maxfill[i] = maxfill_add
            volfill[i] = volfill_add
            maxcarve[i] = maxcarve_add
            volcarve[i] = volcarve_add
            volfillbehind[i] = volfillbehind_add
            maxfillbehind[i] = maxfillbehind_add
            completecarve[i] = completecarve_add
            
            
    if verbose_print:
        print_time(f"         {numpits} pits cataloged ", tic)

    """Create dataframe from dictionary"""
    pits_dct = {'fillcarveID': pitlist,
                'maxfill': maxfill,
                'volfill': volfill,
                'maxcarve': maxcarve,
                'volcarve': volcarve,
                'volfillbehind': volfillbehind,
                'maxfillbehind': maxfillbehind,
                'completecarve': completecarve}

    DFpits = pd.DataFrame(pits_dct)
    DFpits.index = DFpits['fillcarveID']
    DFpits.drop(labels=['fillcarveID'], axis=1, inplace=True)

    return DFpits, fillcells_dct, carvecells_only_dct, carvecells_dct, carve_fillcells_dct


###################################################################
"""         Select and apply fill or carve solutions            """
###################################################################


def select_solution(DFpits, fillcells_dct, carvecells_dct, carvecells_only_dct,
                    fill_vec, carve_vec, 
                    fillID_vec,  carveID_vec, fill_vol_thresh, carve_vol_thresh,
                    max_carve_depth, maxcarve_factor=1.25,
                    apply_shallow_fill=False,
                    wall_array=None, water_mask_array=None, sink_array=None):
    """

    Use pit catalogue to select optimal solution for each depression, minimizing changes to the DEM

    Parameters
    ----------
    DFpits : pandas dataframe
        dataframe with each pit catalogued
   fillcells_dct : dict
        dictionary of indices of pixels associated with each pit
    carvecells_dct : dict
        dictionary of indices of pixels associated with each carve
    fill_vec : array
        1D array of reshaped, filled DEM values
    carve_vec : array
        1D array of reshaped, carved DEM values
    diff_vec_carve : array
        1D array of difference values btwn carved and input DEM
    fillID_vec : array
        1D array of pit ID values. Each contiguous depression is assigned an ID
        which is used to track fill/carve volumes, depths and solutions applied
        to each pit. 
    carveID_vec : array
        1D array of carve ID values
    fill_vol_thresh : float
        max allowable fill volume before a partial fill solution is attempted
    carve_vol_thresh : float
        maximum allowable carve volume
    max_carve_depth : float
        maximum carve depth
    maxcarve_factor : float, optional
        max carve depth as factor of associated fill depth, by default 1.25
    apply_shallow_fill : bool, optional
        auto fill over shallow, flat pits?, by default False
    wall_array : array, optional
        2D array of temp boundary walls, by default None
    water_mask_array : array, optional
        2D array of mask of burned water values - increases carve preference within 
        water mask, by default None
    sink_array : array, optional
        array of raster mask for endorheic sink points, by default None


    Returns
    -------
    dataframe
        input dataframe updated with solution codes


        SOLUTIONS:

        four digit solution for each pit - FSDP
            F = first and preferred solution
            S = secondary valid solution
            D = connected downstream pit solution
            P = percent fill (x10) or shallow/flat pit flag

        0   : N/A
        1   : carve
        2   : fill
        3   : combined pit (partial fill)
        4   : unsolved combined pit
        5   : shallow pit
        6   : flat pit
        7   : endorheic sink pit
        9   : unsolved/incomplete

    """

    tic = time.time()
    
    
    """loop to select a solution for each pit"""
    pits = DFpits.index
    solutions = 9*np.ones(len(pits),).astype(np.int)

    """read pitID raster and convert to numpy vectors"""
    fill_ind = np.flatnonzero(fillID_vec > 0)
    downstream_pits = np.zeros(len(pits),).astype(np.int)

    if wall_array is not None:
        wall_vec = np.ravel(wall_array)
    
    if water_mask_array is not None:
        water_mask_vec = np.ravel(water_mask_array)

    npits = len(pits)
    for pit in pits:
            
        ipit = pit-1
        fill_cells = fillcells_dct[pit]
        carve_cells = carvecells_dct[pit]
        pit_df = DFpits.loc[pit]
        maxcarve = pit_df.maxcarve
        maxfillbehind = pit_df.maxfillbehind
        maxfill = pit_df.maxfill
        volcarve = pit_df.volcarve
        volfillbehind = pit_df.volfillbehind
        volfill = pit_df.volfill
        complete_carve = pit_df.completecarve

        """Pit within wall - skip"""
        if wall_array is not None:
            if np.min(wall_vec[fill_cells]) == 1:
                solution_name = 1
                solutions[ipit] = solution_name
                continue
            
        """Sink within pit - skip"""
        if sink_array is not None:
            sink_vec = sink_array.reshape(len(fill_vec),)
            if np.max(sink_vec[fill_cells]) > 0:
                solution_name = 7
                solutions[ipit] = solution_name
                continue            

        """
        define valid carve depth as factor of max fill height. If the carve depth is much greater
        than the max fill, the carve is probably not following close to the natural spillpoint
        """
        maxcarve_limit = maxcarve_factor*maxfill

        """If pit contains flattened water values, raise maxcarve limit if carve connects to 
        more flattened water"""
        if (water_mask_vec is not None) and (np.count_nonzero(water_mask_vec[carve_cells])) > 0:
            maxcarve_limit += 5

        if apply_shallow_fill:
            """
            Keep fill solution for pits with: an average fill <1m and maxfill/pitcells < 1cm 
            (big, shallow, often wetlands/void-filled)"""
            avg_fill = maxfill/len(fill_cells)
            avg_vol = volfill/len(fill_cells)
            if avg_fill < 0.01 and avg_vol < 1.0:
                solution_name = 2006
                solutions[ipit] = solution_name
                continue

        """Keep fill solution for pits <1cm deep - accounts for gradient changes"""
        if maxfill < 0.01:
            solution_name = 2005
            solutions[ipit] = solution_name
            continue

        # carve plus fill behind volume/height change
        max_carve_change = np.abs(maxcarve) + np.abs(maxfillbehind)
        volcarve_total = np.abs(volcarve) + np.abs(volfillbehind)

        if (complete_carve == 1) & (max_carve_change <= max_carve_depth) & (maxcarve <= maxcarve_limit):  # valid carve

            """carve completely within flattening mask"""
            if (water_mask_vec is not None) and (np.count_nonzero(water_mask_vec[carve_cells]) >= len(carve_cells)):
                solution_name = 1002

            elif volcarve_total < volfill:  # carve preferred

                if volcarve_total > fill_vol_thresh:  # backfill is still over fill threshold
                
                    avg_fill = maxfill/len(fill_cells)
                    avg_vol = volfill/len(fill_cells)
                    if avg_fill < 0.01 and avg_vol < 1.0:
                        solution_name = 1006

                    elif np.abs(volfillbehind) >= 0.9*volfill:  # no viable carve solution
                        solution_name = 2004
                    else:
                        solution_name = 4000

                elif volcarve < carve_vol_thresh:
                
                    solution_name = 1200

                    # """determine if carve flows into another pit"""
                    # fillID_carve_vec = fillID_vec[carve_cells]
                    # fillID_carve_vec = fillID_carve_vec[np.flatnonzero(fillID_carve_vec)]
                    # pit_intersectIDs = list(np.unique(fillID_carve_vec))
                    # pit_intersectIDs.remove(pit) # remove this pit, leave neighbors 

                    # if len(pit_intersectIDs) == 0:  # does not flow into another pit
                        # """select carve"""
                        # if volfill < fill_vol_thresh:
                            # solution_name = 1200
                        # else:
                            # solution_name = 1000

                    # else:  # flows into another pit
                        # if min(pit_intersectIDs) == 0:
                            # print(f"ERROR: pit intersect ID is ZERO for pit {pit}")
                            # break

                        # solution_name, pit_downstreamID = downstreampit(DFpits,  pit, pit_intersectIDs,
                                                                        # volfill, fill_vol_thresh,
                                                                        # fillID_vec, fill_vec,
                                                                        # carve_cells,
                                                                        # carveID_vec, carve_vec,
                                                                        # carvecells_only_dct, 
                                                                        # carvecells_dct,
                                                                        # fillcells_dct,
                                                                        # solution_vec=solutions)
                        # downstream_pits[ipit] = pit_downstreamID

                elif volfill < fill_vol_thresh:  # select fill
                    solution_name = 2000

                else:  # send to combined fill
                    solution_name = 4000

            elif volcarve_total >= volfill:  # fill prefered
                if volfill < fill_vol_thresh:
                    """select fill"""
                    if volcarve < carve_vol_thresh:
                        solution_name = 2100
                    else:
                        solution_name = 2000
                else:
                    if np.abs(volfillbehind) >= 0.9*volfill:  # no viable carve solution
                        solution_name = 2004
                    else:
                        solution_name = 4000

        else:  # incomplete carve or carve exceeded max_carve_depth
            if volfill < fill_vol_thresh:  # select fill
                    solution_name = 2009
            else:
                solution_name = 4000
        solutions[ipit] = solution_name

    DFpits["solution"] = solutions
    DFpits["downstream_pit"] = downstream_pits    
    
    DFunsolved = DFpits[DFpits.solution == 9000].copy()
    
    if not DFunsolved.empty: #carve unsolved pits
        unsolvedpits = list(DFunsolved.index.values)
        DFpits.at[unsolvedpits, 'solution'] = 1009


    # iteration = 0
    # while len(unsolvedpits) > 0:
        
        # for pit_id in unsolvedpits:
        
            # DFpits.at[pit_id, 'solution'] = solution_name

            # """ pits with solution 9000 have a valid and preferred carve solution"""
            # pit_df = DFpits.loc[pit_id]
            # fill_cells = fillcells_dct[pit_id]
            # carve_cells = carvecells_dct[pit_id]
            # volfill = pit_df.volfill

            # pit_downstream = pit_df.downstream_pit
            # pit_intersectIDs = [pit_downstream] # needs to be list for function call
            # pit_inter_df = DFpits.loc[pit_downstream].copy()

            # """Check for circular pits"""
            # pit_inter_downstreamID = pit_inter_df.downstream_pit
            # if pit_id == pit_inter_downstreamID:
                # """does one carve connect both pits?"""
                # carve_cells_ds = carvecells_dct[pit_inter_downstreamID]
                # non_intersect_carve = list(set(carve_cells) - set(carve_cells_ds))
                # if len(non_intersect_carve) == 0:  # one carve connects both pits - carve
                    # solution_name = 1212
                # else:  # truly circular pits - fill
                    # solution_name = 2022
            # else:
                # solution_name, pit_downstreamID = downstreampit(DFpits, pit_id, pit_intersectIDs,
                                                                # volfill, fill_vol_thresh,
                                                                # fillID_vec, fill_vec, carve_cells,
                                                                # carveID_vec, carve_vec,
                                                                # carvecells_only_dct, carvecells_dct,
                                                                # fillcells_dct)

            # if solution_name != 9000:
                # DFpits.at[pit_id, 'solution'] = solution_name

        # if iteration > 20:
            # print('     Stuck in 9000 pit loop for subbasin, fill remaining pits')
            # DFpits.at[unsolvedpits, 'solution'] = 2009

        # if iteration > 30:
            # print('     ACTUALLY Stuck in 9000 pit loop for subbasin')
            # break

        # DFunsolved = DFpits[DFpits.solution == 9000].copy()
        # unsolvedpits = list(DFunsolved.index.values)
        # iteration += 1

    return DFpits


def combined_solution(inDEM, inDEM_array, outpath, nx, ny, DFpits, fillcells_dct, 
                      carvecells_dct, carve_fillcells_dct, pit_id,
                      combined_fill_interval, fillID_vec, fill_vec, carve_vec, radius,
                      maxcost, min_dist, max_carve_depth, wbt,
                      nodataval=-9999, fixflats=False, flat_increment=0.0, carve_method='SC',
                      water_array=None, wall_vec=None, del_temp_files=True):
    """    
    Generate intermediate fill-carve solutions and apply the one with the 
    minimum volume change. This function is designed to handle cases where the 
    carve solution is incomplete, so a new carve solution is run through WBT 
    after each shallow fill increment and then checked for completeness. Only 
    complete carve solutions are considered when selecting the minimum impact 
    solution. 

    Parameters
    ----------
    inDEM : path
        input DEM that has solution already applied to smaller pits
    outpath : path
        output directory
    nx : int
        raster Xshape
    ny : int
        raster Yshape
    DFpits : pandas dataframe
        dataframe containing catalogue of all pits and their solutions
    fillcells_dct : dict
        dictionary of filled cells for each pitID
    pit_id : ID number of pit to be partially filled
        [description]
    combined_fill_interval : float
        between 0 - 1. Fraction increment to iteratively fill (of total depth)
    fill_vec : array
        1D array of reshaped, filled DEM values
    carve_vec : array
        1D array of reshaped, carved DEM values
    maxcost : float
        maximum cost for a carve in the LC function. 
        N/A for the standard carve 'SC' function
    radius : int    
        maximum search window in the carving function (pixels)
    min_dist : bool
        True= = minimize distance, False = minimize cost. Only applied in 'LC' carving
    max_carve_depth : float
        maximum carve depth
    wbt : class
        alias to call WhiteboxTools functions
    nodataval : int, optional
        no data value, by default -9999
    fixflats : bool, optional
        apply gradient to flat regions after filling, by default False
    flat_increment : float, optional
        gradient to apply to flat regions if fixflat=True. 
        (0.00001 is reasonable), by default 0.0 
    carve_method : str, optional
        whitebox tools carving algorithm used 
        'SC' for standard carve, 'LC' for least cost, by default 'SC'
    watermask : path, optional
        Optional tif of water bodies. Partial fill not allowed to be below water 
        within a pit, by default None
    del_temp_files : bool, optional
        delete temporary files?, by default False

    Returns
    -------
    str 
        solution code for preferred combined fill
    dataframe
        small pandas data frame cataloging fill and carve solutions for given pit

    """

    gc.collect()
    
    """pull out pit parameters from dataframe"""
    pit_df = DFpits.loc[pit_id].copy()
    init_sol = pit_df.solution
    footprint = fillcells_dct[pit_id]
    completecarve = pit_df.completecarve
    combined_sol = init_sol - 1000   # 3000 or 3020

        
    """define footprint of depression and clip to buffered rectangle"""
    combined_path = os.path.join(outpath, "combined_solutions")
    if not os.path.exists(combined_path):
        os.makedirs(combined_path)

    clip_buf = radius+2
    footprint_arr = np.unravel_index(footprint, (ny, nx))  # vec --> array index map
    clippedDEM = os.path.join(combined_path, "{0}.tif".format(pit_id))
    clipped_ind_arr, nx_clipped, ny_clipped = pyct.clip_to_window(inDEM_array, inDEM,
                                                            clippedDEM, footprint_arr,
                                                            clip_buf, nx, ny, nodataval=nodataval)

    """indices of clipped bounding box within whole vector"""
    clipped_ind = np.ravel_multi_index(clipped_ind_arr, (ny, nx))  # array --> vec index map
    inDEM_array_clipped = gdal_array.LoadFile(clippedDEM)
    inDEM_vec_clipped = inDEM_array_clipped.reshape(ny_clipped*nx_clipped,)
      
    fillDEM_array = fill_vec.reshape(ny, nx)
    fillDEM_array_clipped = fillDEM_array[clipped_ind_arr]
    fill_vec_clipped = fillDEM_array_clipped.reshape(ny_clipped*nx_clipped,)
    carveDEM_array = carve_vec.reshape(ny, nx)
    carveDEM_array_clipped = carveDEM_array[clipped_ind_arr]
    carve_vec_orig_clipped = carveDEM_array_clipped.reshape(ny_clipped*nx_clipped,)

    del inDEM_array
    del footprint_arr

    """define fill and carve indices within the clipped DEM"""
    fillID_clipped_vec = fillID_vec[clipped_ind]
    footprint_clipped = np.flatnonzero(fillID_clipped_vec == pit_id)

    if water_array is not None:
        water_vec = water_array.reshape(len(fill_vec),)
        water_vec_clipped = water_vec[clipped_ind]   
        water_ind_clipped_vec = np.where(water_vec_clipped == 1)[0]
        fill_lakecells_clipped = np.intersect1d(water_ind_clipped_vec, 
                                                footprint_clipped,
                                                assume_unique=True)

    """define maximum fill depth and elevation at that location"""
    maxfill_value = pit_df.maxfill

    """build data frame to track volume change for this pit"""
    df_combined = pd.DataFrame(columns=["vol",
                                        "elevs",
                                        "solution",
                                        "footprint"])
    vols = []
    elevs = []
    sol_names = []
    dep_indices = []

    diff_fill_clipped = np.zeros_like(fill_vec_clipped)
    diff_fill_clipped[footprint_clipped] = np.round((fill_vec_clipped[footprint_clipped]
                                                     - inDEM_vec_clipped[footprint_clipped]), 3)
    maxfill_ind_clipped = np.argmax(diff_fill_clipped)
    maxfill_ind_clipped_arr = np.unravel_index(maxfill_ind_clipped,
                                                (ny_clipped, nx_clipped))

    """ Iterate through intermediate solutions, raising the fill level """
    combined_fractions = np.arange(combined_fill_interval, 1, combined_fill_interval)
    fraction_list = np.round(combined_fractions, 1)
    
    for frac in fraction_list:
        """
        Instead of setting fill cells to one value, remove a constant from the full fill,
        solution and retain the gradient imposed in the fix flats algorithm. This seeds the
        carve solution at the fill cell closest to the natural spillpoint (if gradient is applied)
        """
        gc.collect()
        
        SF_vec = np.copy(fill_vec_clipped)
        fill_depth_remove = maxfill_value*(1-frac)
        SF_vec[footprint_clipped] = SF_vec[footprint_clipped] - fill_depth_remove
        SF_diff = np.round((SF_vec - inDEM_vec_clipped), 3)
        belowground_ind = np.where(SF_diff < 0)[0]
        fill_frac_ind = np.where(SF_diff > 0)[0]
        # reset any values now below ground
        SF_vec[belowground_ind] = inDEM_vec_clipped[belowground_ind]
        p_fill_min = (SF_vec[fill_frac_ind]).min()
        carve_fill_min = (carve_vec_orig_clipped[fill_frac_ind]).min()
        
        """partial fill is higher than carve backfill (with complete carve)"""
        if (completecarve == 1) and (p_fill_min > carve_fill_min):
            #print('    partial fill is higher than carve backfill')
            continue


        """convert shallow filled array to raster"""
        modDEM_array = SF_vec.reshape(ny_clipped, nx_clipped)
        clippedDEM_SF = os.path.join(
            combined_path, "{0}_SF_{1}.tif".format(pit_id, frac))
        pyct.npy2tif(modDEM_array, clippedDEM, clippedDEM_SF, nodata=nodataval)
        del modDEM_array
        del SF_vec
        del SF_diff
        
        """carve with fill behind disabled"""
        clippedDEM_carved = os.path.join(combined_path,
                                         "{0}_SF_{1}_{2}.tif".format(pit_id, frac, carve_method))

        if carve_method == 'LC':
            """Breaches the depressions in a DEM using a least-cost pathway method."""
            wbt.breach_depressions_least_cost(clippedDEM_SF, clippedDEM_carved, radius,
                                              max_cost=maxcost, min_dist=min_dist,
                                              flat_increment=flat_increment, fill=False)
            
            """apply WL fill to check whether for complete carve (avoiding edge effects)"""
            clippedDEM_filled = os.path.join(combined_path,
                                             "{0}_SF_{1}_{2}_filled.tif".format(pit_id,
                                                                                frac, carve_method))
            wbt.fill_depressions_wang_and_liu(clippedDEM_carved, clippedDEM_filled,
                                              fix_flats=fixflats, flat_increment=flat_increment)


        else:
            """Breaches all of the depressions in a DEM using Lindsay's (2016) algorithm."""
            wbt.breach_depressions(clippedDEM_SF, clippedDEM_carved, max_length=radius,
                                   max_depth=max_carve_depth, fill_pits=False,
                                   flat_increment=flat_increment, callback=None)
        
        
        """check whether carve is complete using difference"""
        clippedDEM_carved_array = gdal_array.LoadFile(clippedDEM_carved)

        """find difference with original DEM"""
        diff_array_clip = np.round((clippedDEM_carved_array - inDEM_array_clipped), 4)
        diff_vec_clip = diff_array_clip.reshape(ny_clipped*nx_clipped,)

        """Find intersecting carves and their total extent outside of the filled pit"""
        mask_carve_clipped_arr = np.zeros_like(inDEM_array_clipped, dtype=np.int8)
        mask_carve_clipped_arr[diff_array_clip < 0] = 1
        mask_carve_clipped_arr[inDEM_array_clipped == nodataval] = 1
        s_clump = [[1, 1, 1], [1, 1, 1],[1, 1, 1]]
        carveID_clip_array, num_feat = ndimage.label(mask_carve_clipped_arr, 
                                                        structure=s_clump)
 
        # mask_carve_tif = os.path.join(combined_path, "{0}_{1}_carvemask.tif".format(pit_id, frac))
        # pyct.npy2tif(mask_carve_clipped_arr, clippedDEM,
                     # mask_carve_tif, nodata=0, dtype=gdal.GDT_Byte)
        # carveID_clip_tif = os.path.join(combined_path, "{0}_{1}_carveID.tif".format(pit_id, frac))
        # wbt.clump(mask_carve_tif, carveID_clip_tif, zero_back=True)
        # carveID_clip_array = gdal_array.LoadFile(carveID_clip_tif)
                
        carveID_clip_vec = carveID_clip_array.reshape(ny_clipped*nx_clipped,)
        carve_vec_clip = clippedDEM_carved_array.reshape(ny_clipped*nx_clipped,)      
        carve_ind_clipped = np.flatnonzero(diff_vec_clip < 0.0)
        carve_intersect = np.intersect1d(footprint_clipped, 
                                        carve_ind_clipped, assume_unique=True)
        carveIDs_pit = np.unique(carveID_clip_vec[carve_intersect])
        
        """Find intersecting carves through IDs"""
        carve_mask_pit = np.isin(carveID_clip_vec, carveIDs_pit)
        carvecells_add = list(np.flatnonzero(carve_mask_pit))
                
        """check whether carve went off the edge"""
        nodata_clipped_ind = np.flatnonzero(inDEM_vec_clipped == nodataval)
        if len(nodata_clipped_ind > 0):  # clipped tif includes edge
            carveout_ID = carveID_clip_vec[nodata_clipped_ind][0]
            if carveout_ID in list(carveIDs_pit):  # disqualify carve solution
                continue
                
        if len(carvecells_add) < 1:  # no carves from pit, skip
            continue

        """create expanded footprint"""
        expanded_footprint_clipped = list(set(carvecells_add + list(footprint_clipped)))
        
        """convert indices within clipped DEM to indices within original DEM"""
        expanded_footprint = clipped_ind[expanded_footprint_clipped]
        
#         """Check whether solution footprint interacts with boundary wall mask"""
#         if wall_vec is not None:
#             wall_footprint = wall_vec[expanded_footprint]
#             if np.count_nonzero(wall_footprint) > 0:
#                 print('  interacts with wall')
#                 continue
         
        """calculate volume change"""
        volchange = np.sum(abs(diff_vec_clip[expanded_footprint_clipped]))


        """check that max_carve_depth criteria is met in expanded footprint"""
        maxcarvedepth = abs(diff_vec_clip[expanded_footprint_clipped].min())

        if maxcarvedepth <= max_carve_depth:
            """add modified elevations to a list of lists"""
            clippedDEM_filled_vec = clippedDEM_carved_array.reshape(
                ny_clipped*nx_clipped,)
            
            elevs.append(list(clippedDEM_filled_vec[expanded_footprint_clipped]))
            vols.append(volchange)
            dep_indices.append(expanded_footprint.astype(int))

            # assign solution code
            if np.int(frac*10) == 0:
                if combined_sol == 3020:
                    sol_names.append(1020)
                elif combined_sol == 3000:
                    sol_names.append(1003)
            else:
                sol_names.append(np.int(combined_sol+(10*frac)))
                
        """Clear out folder"""
        if del_temp_files:
            ft.erase_files(combined_path, search='{0}_*.tif'.format(pit_id))
            
    """Add initial fill solutions to lists"""
    sol_names.append(2003)
    vols.append(pit_df.volfill)
    dep_indices.append(footprint)
    elevs.append(list(fill_vec[footprint]))
    
    """Add initial complete carve solutions to lists"""
    if completecarve == 1:
        vol_total = pit_df.volcarve + pit_df.volfillbehind
        carve_cells_all = list(set(list(carvecells_dct[pit_id]) 
                                   + list(carve_fillcells_dct[pit_id])
                                   + list(footprint)))      
        if not np.isnan(vol_total):
            sol_names.append(1003)
            vols.append(vol_total)
            dep_indices.append(carve_cells_all)
            elevs.append(list(carve_vec[carve_cells_all]))

    """add lists to columns of df_combined"""
    df_combined["vol"] = vols
    df_combined["elevs"] = elevs
    df_combined["solution"] = sol_names
    df_combined["footprint"] = dep_indices
    
    #print(df_combined)
    

    """find the fraction for which the min volume change occurs with a completed carve solution"""
    solution_name = df_combined.solution[df_combined["vol"].idxmin()]
    sol_df = df_combined.loc[df_combined.solution == solution_name].copy()

    """Clear out folder"""
    #if del_temp_files:
    ft.erase_files(combined_path, search='*.tif')

    return solution_name, sol_df


def apply_solution(DFpits, fillcells_dct, carvecells_dct, carve_fillcells_dct,
                   inDEM_vec, fill_vec, carve_vec,
                   solution_mask_vec=None, excludedpits=None):
    """
    Apply all non-partial fill (combined) solutions. The modified DEM is output. 

    Parameters
    ----------
    DFpits : dataframe
        pit dataframe
    fillcells_dct : dict
        dictionary of indices for each pit ID
    carvecells_dct : dict
        dictionary of indices for each carve ID
    carve_fillcells_dct : dict
        dictionary of iindices for each fillbehind ID 
    inDEM_vec : array
        1D array of reshaped, input DEM values
    fill_vec : array
        1D array of reshaped, filled DEM values
    carve_vec : array
        1D array of reshaped, carved DEM values
    solution_mask_vec : array, optional
        1D array of existing vector of solution codes applied to each pixel. 
        If None, one is created, by default None
    excludedpits : list, optional
        list of pits that are not to have solutions applied (yet unsolved), by default None

    Returns
    -------
    array
        1D array with fill/carve solutions applied
    array
        1D array of solution codes for each pixel
    """

    """create a vector modDEM_vec to be modified for each pit ID"""
    modDEMoutput_vec = np.copy(inDEM_vec)
    if solution_mask_vec is None:
        solution_mask_vec = np.zeros_like(modDEMoutput_vec, dtype=np.int16)
    pits = list(DFpits.index)

    # remove combined pits that have already had solutions applied
    if excludedpits is not None:
        for expit in excludedpits:
            fill_cells = fillcells_dct[expit]
            solution = DFpits.loc[expit].solution
            solution_mask_vec[fill_cells] = solution
            pits.remove(expit)
                
    i_carve = 0

    for pit in pits:
        #pit_df = DFpits.loc[pit].copy()
        fill_cells = fillcells_dct[pit]
        carve_cells = carvecells_dct[pit]
        carvebehind_cells = carve_fillcells_dct[pit]
        solution = DFpits.loc[pit].solution
        solution_mask_vec[fill_cells] = solution

        if solution == 1:  # wall pit
            modDEMoutput_vec[fill_cells] = modDEMoutput_vec[fill_cells]

        if (solution >= 1000) and (solution < 2000):  # apply carve
            i_carve += 1

            if solution == 1023: # fill to downstream pit level
                #try:
                ds_pit = int(DFpits.loc[pit].downstream_pit)
                fill_ds_cells = fillcells_dct[ds_pit]
                min_ds_fill = fill_vec[fill_ds_cells].min()
                min_fill = fill_vec[fill_cells].min()
                if min_fill < min_ds_fill: # ds above pit - fill pit
                    #print(f"     {pit}, {min_fill}...{ds_pit}, {min_ds_fill}")
                    modDEMoutput_vec[fill_cells] = fill_vec[fill_cells]
                else:
                    modDEMoutput_vec[carve_cells] = min_ds_fill
                    below_fill_ind = np.nonzero(inDEM_vec[fill_cells]<min_ds_fill)
                    partial_fill_ind = fill_cells[below_fill_ind]
                    modDEMoutput_vec[partial_fill_ind] = min_ds_fill   
            else:
                modDEMoutput_vec[carve_cells] = carve_vec[carve_cells]
                modDEMoutput_vec[carvebehind_cells] = carve_vec[carvebehind_cells]
            
        elif (solution >= 2000) and (solution < 3000):  # apply fill
            modDEMoutput_vec[fill_cells] = fill_vec[fill_cells]
            
            
    return modDEMoutput_vec, solution_mask_vec


###################################################################
"""                     Helper functions                       """
###################################################################


def fill_pits(inDEM, wbt, fill_method='WL', nodataval=-9999,
              fixflats=False, flat_increment=0.0, save_tif=True,
              downcast=True):
    """
    Fill depressions within a DEM tif using one of two WhiteBoxTools algorithms

    Parameters
    ----------
    inDEM : path
        input DEM raster
    wbt : class
        WhiteBoxTools class alias
    fill_method : str, optional
        'SF' = standard fill, 'WL' = Wang and Lui, by default 'WL'
    nodataval : int, optional
        nodata value of input tif , by default -9999
    fixflats : bool, optional
        apply gradient to flat regions after filling, by default False
    flat_increment : float, optional
        gradient to apply to flat regions if fixflat=True. 
        (0.00001 is reasonable), by default 0.0 

    Returns
    -------
    array
        2D array of filled DEM values
    array
        2D array of differenced (filled - input) values
    path
        path of differenced tif file

    """

    fill_tif = inDEM.replace('.tif', '_{0}.tif'.format(fill_method))
    
    if flat_increment > 0.0:
        fixflats = True

    """Pit filling"""
    if fill_method == "SF":
        """Standard pit filling (SF)"""
        wbt.fill_depressions(inDEM, fill_tif,
                             max_depth=None,
                             fix_flats=fixflats,
                             flat_increment=flat_increment)

    elif fill_method == "WL":
        """ Fill all pits using Wang and Lui algorithm (WL) """
        wbt.fill_depressions_wang_and_liu(inDEM, fill_tif,
                                          fix_flats=fixflats,
                                          flat_increment=flat_increment)
    else:
        raise("   Improper fill method specified")

    # check for proper projection in fill tif
    fill_tif_check = gdal.Open(fill_tif, gdal.GA_ReadOnly)
    projection_check = fill_tif_check.GetProjectionRef()
    fill_tif_check = None

    inDEM_array = gdal_array.LoadFile(inDEM)
    if len(projection_check) == 0:
        print('   bad projection for fill tif, deleting tif')
        fill_array = gdal_array.LoadFile(fill_tif)
        fill_array = fill_array.astype('float32')
        os.remove(fill_tif)
        diff_fill_array = np.round((fill_array - inDEM_array), 3)
        diff_fill_array[(diff_fill_array > 0.0) & (
            diff_fill_array <= 0.002)] = 0.0
    else:
        if downcast: # downcast to 32 bit
            pyct.tifdowncast(fill_tif, dtype=gdal.GDT_Float32)
        fill_array = gdal_array.LoadFile(fill_tif)
        diff_fill_array = np.round((fill_array - inDEM_array), 3)

    if save_tif:
        diff_tif_fill = fill_tif.replace('.tif', '_DIFF.tif')
        pyct.npy2tif(diff_fill_array, inDEM, diff_tif_fill,
                 nodata=nodataval, dtype=gdal.GDT_Float32)
    del inDEM_array

    return fill_array, diff_fill_array


def breach_pits(inDEM_tif, radius, maxcost, wbt, min_dist=False, flat_increment=0.0,
                breach_increment=0.0001, nodataval=-9999, method='SC', fixflats=False,
                save_tif=True):
    """Breach depressions within a DEM using one of two WhiteBoxTools algorithms

    Parameters
    ----------
    inDEM_tif : path
        input raster file
    radius : int    
        maximum search window in the carving function (pixels)
    maxcost : float
        maximum cost for a carve in the LC function. 
        N/A for the standard carve 'SC' function
    wbt : class
        WhiteBoxTools class alias
    min_dist : bool
        True= = minimize distance, False = minimize cost. 
        Only applied in 'LC' carving, by default False
    flat_increment : float, optional
        gradient to apply to flat regions if fixflat=True. 
        (0.00001 is reasonable), by default 0.0 
    breach_increment : float, optional
        gradient applied to carves. Only in 'LC' method, by default 0.0001
    nodataval : int, optional
        nodata value of input DEM, by default -9999
    method : str, optional
        whitebox tools carving algorithm used 
        'SC' for standard carve, 'LC' for least cost, by default 'SC'
    fixflats : bool, optional
        apply gradient to flat regions after filling, by default False

    Returns
    -------
    array
        2D array of carved DEM values
    array
        2D array of differenced (carved - input) values
    path
        path of differenced tif file
    """

    carve_path = inDEM_tif.replace(
        '.tif', '_{1}_{0}.tif'.format(radius, method))
    carvefill_path = inDEM_tif.replace(
        '.tif', '_{1}_{0}_filled.tif'.format(radius, method))
        
    if flat_increment > 0.0:
        fixflats = True

    if not os.path.exists(carve_path):
        if method == 'LC':
            """Breaches the depressions in a DEM using a least-cost pathway method."""
            wbt.breach_depressions_least_cost(inDEM_tif, carve_path, radius,
                                              max_cost=maxcost, min_dist=min_dist,
                                              flat_increment=breach_increment, fill=False)
        else:
            """Breaches all of the depressions in a DEM using Lindsay's (2016) algorithm.
            Note - will auto fill behind using flat increment"""
            wbt.breach_depressions(inDEM_tif, carve_path, max_length=radius,
                                   max_depth=None, fill_pits=True,
                                   callback=None, flat_increment=flat_increment)

    pyct.tifdowncast(carve_path, dtype=gdal.GDT_Float32)

    if method == 'LC':
        wbt.fill_depressions_wang_and_liu(carve_path, carvefill_path, fix_flats=fixflats,
                                          flat_increment=flat_increment)
        pyct.tifdowncast(carvefill_path, dtype=gdal.GDT_Float32)

    else:
        carvefill_path = carve_path

    """Calculate difference"""
    inDEM_array = gdal_array.LoadFile(inDEM_tif)
    carve_array = gdal_array.LoadFile(carvefill_path)
    diff_carve_array = np.round((carve_array - inDEM_array), 3)
    del inDEM_array

    if save_tif:
        diff_tif_carve = carvefill_path.replace('.tif', '_DIFF.tif')
        pyct.npy2tif(diff_carve_array, inDEM_tif, diff_tif_carve,
                 nodata=nodataval, dtype=gdal.GDT_Float32)

    return carve_array, diff_carve_array


def fix_shifted_data(in_array, shift_array, nodataval, 
                     failtest='negative_diff', vprint=False):
    """
    Fix data that have been shifted over by one pixel within a DEM. This occurs in 
    the outputs of WhiteBoxTools functions sometimes, often in basins that are large and 
    have a large longitude span. Likely a bug/rounding error somewhere in the 
    projection/reprojection to and from UTM coordinates. This function is a bit hacky,
    but reliably fixes the issue (which is easy to diagnose from the DEM difference map)

    Parameters
    ----------
    in_array : array
        input DEM array that is shifted
    shift_array : array
        Original DEM that is not shifted
    nodataval : float
        no data value of original DEM
    failtest : str, optional
        Test to apply to difference map to determine if the array shift has 
        been successful. 
            'positive_diff' =  no difference values > 0 
            'negative_diff' = no difference values < 0,
            by default 'negative_diff'
    vprint : bool, optional
        print timestamps?, by default False

    Returns
    -------
    array
        2D array of shifted values
    array
        2D array of new (shifted) difference map
    """

    data_ind = np.nonzero(in_array != nodataval)
    nodata_ind = np.nonzero(in_array == nodataval)
    fill_data_ind = np.nonzero(shift_array != nodataval)

    minx_dataind, maxx_dataind = data_ind[1].min(), data_ind[1].max()
    miny_dataind, maxy_dataind = data_ind[0].min(), data_ind[0].max()
    minx_fill_dataind, maxx_fill_dataind = fill_data_ind[1].min(), fill_data_ind[1].max()
    miny_fill_dataind, maxy_fill_dataind = fill_data_ind[0].min(), fill_data_ind[0].max()

    shift_1 = maxx_fill_dataind - maxx_dataind
    shift_0 = maxy_fill_dataind - maxy_dataind
    ind_diff = np.abs(len(data_ind[0]) - len(fill_data_ind[0]))

    fill_data_ind_shift = (fill_data_ind[0] - shift_0,
                           fill_data_ind[1] - shift_1)

    shift_array[fill_data_ind_shift] = shift_array[fill_data_ind]
    shift_array[nodata_ind] = nodataval

    diff_array = np.round((shift_array - in_array), 3)
    if failtest == 'negative_diff':
        fail_diff = np.nonzero(diff_array < 0.0)
    elif failtest == 'positive_diff':
        fail_diff = np.nonzero(diff_array > 0.0)

    num_fail = len(fail_diff[0])
    assert num_fail <= ind_diff+3, 'Re-mapped fill diff has too many negative values'

    return shift_array, diff_array


def fill_shallow_pits(modDEM_vec, sfill, fillID_vec, diff_fill_array,
                      fill_array, water_vec=None, verbose_print=True):
    """
    Auto-fill any depressions with a maximimum depth less than 'sfill'

    Parameters
    ----------
    modDEM_vec : array
        1D array of input DEM values to be modified
    sfill : float
        max depth of pit to auto-fill
    fillID_vec : array
        1D array of pit ID values. Each contiguous depression is assigned an ID
        which is used to track fill/carve volumes, depths and solutions applied
        to each pit. 
    diff_fill_array : array
        2D array of filled difference values
    fill_array : array
        2D array of filled DEM values
    water_vec : array, optional
        1D array of mask of flattened water values - shallow fill 
        excluded from water mask, by default None
    starttime : float, optional
        starting time, by default time.time()
    verbose_print : bool, optional
        print timestamps?, by default True

    Returns
    -------
    array
        1D array of modified (filled) DEM values
    """
                      
    tic = time.time()

    """Fill all pits shallower than sfill depth"""
    diff_vec_fill = np.ravel(diff_fill_array)
    fill_vec = np.ravel(fill_array)
    numpits = fillID_vec.max()
    print_time(f'    Cataloging {numpits} initial pits. sfill={sfill}', tic)

    deep_fill_ind = np.flatnonzero(diff_vec_fill>sfill)
    deep_fill_ids = np.unique(fillID_vec[deep_fill_ind])
    shallow_fill_pit_mask = np.isin(fillID_vec, deep_fill_ids, invert=True)
    
    #only consider pits outside of water mask
    if water_vec is not None:
        water_pit_ids = np.unique(fillID_vec[np.flatnonzero(water_vec)])
        land_pit_mask = np.isin(fillID_vec, water_pit_ids, invert=True)
        mod_mask = (land_pit_mask) & (shallow_fill_pit_mask)
        
    else:
        mod_mask = shallow_fill_pit_mask

    modDEM_vec[mod_mask] = fill_vec[mod_mask]
    
    if verbose_print:
        n_shallow = len(np.unique(fillID_vec[mod_mask]))
        print_time(f'      number of shallow pits: {n_shallow}', tic)
    
    return modDEM_vec
    
    
   

def raise_lake_connected_pits(watermask, fillID_vec, modDEM_vec, inDEM_vec,
                              starttime=time.time(), verbose_print=True):
    """Auto-fill depressions that drain into large water bodies up to 1cm above 
    the minimum water level

    Parameters
    ----------
    watermask : path
        tif file of large waterbody mask (0=land, 1=water)
    fillID_vec : array
        1D array of pit ID pixels
    modDEM_vec : array
        1D array of input DEM values to be modified
    inDEM_vec : array 
        1D array of original DEM values
    starttime : float, optional
        start time, by default time.time()
    verbose_print : bool, optional
        print timestamps?, by default True

    Returns
    -------
    array
        1D array of modified DEM values
    array
        1D array of difference values
    """

    # indices of flattened water pixels
    water_array = gdal_array.LoadFile(watermask)
    lakeind = np.where(water_array == 1)

    if len(lakeind[0]) > 20:  # only continue if lake pixels are present

        # lake-pit intersection indices
        fill_ind = np.flatnonzero(fillID_vec > 0)
        lake_ind_pit = np.intersect1d(lakeind, fill_ind, assume_unique=True)
        lake_fill_intersectIDs = np.unique(fillID_vec[lake_ind_pit])

        for pitID in lake_fill_intersectIDs:
            pitcells = np.where(fillID_vec == pitID)[0]
            lake_in_pit_ind = np.intersect1d(lakeind, pitcells, assume_unique=True)
            # lowest lake/river elevation within pit
            min_lake = np.amin(inDEM_vec[lake_in_pit_ind])
            belowlake_ind = np.where(inDEM_vec[pitcells] < min_lake)[0]
            modDEM_vec[pitcells[belowlake_ind]] = min_lake + 0.01

        DEMdiff_vec = inDEM_vec - modDEM_vec
        num_lake_adjust = len(np.where(DEMdiff_vec < 0)[0])
        maxlakediff = np.abs(DEMdiff_vec).max()
        if verbose_print:
            print_time(f'      number of cells adjusted: {num_lake_adjust} ',
                        f'Max change: {maxlakediff:.2f}' , starttime)

    return modDEM_vec, DEMdiff_vec


def create_clump_array(input_array, clump_ID_tif, stencil_tif, wbt, nodataval=-9999,
                       noflow_array=None, condition='greaterthan', clumpbuffer=None,
                       split_mask=None, save_tif=True):
    """Create an array of ID numbers for contiguous features based on a difference map.

    Parameters
    ----------
    input_array : array
        difference array to identify contiguous features
    clump_ID_tif : path
        output file for clumped ID tif
    stencil_tif : path
        tif file to provide projection,size,etc
    wbt : class
        WhiteBoxTools() alias
    nodataval : int, optional
        no data value, by default -9999
    noflow_array : array, optional
        mask of boundary condition to include in clumped array 
        (negative diff only), by default None
    condition : str, optional
        condition on which to define features within difference map:
        'greaterthan' : values > 0
        'lessthan' : values < 0
        'nonzeros' : any nonzero difference value, by default 'greaterthan'
    clumpbuffer : int, optional
        buffer (in pixels) around clumped features, by default None

    Returns
    -------
    array
        2D array of ID numbers for contigious features
    """

    clump_array = np.zeros(
        (input_array.shape[0], input_array.shape[1]), dtype=np.int8)
    if condition == 'greaterthan':
        clump_array[input_array > 0] = 1

    if condition == 'lessthan':
        if noflow_array is not None:
            clump_array[noflow_array == 1] = 1
        clump_array[input_array < 0] = 1

    if condition == 'nonzero':
        clump_array[input_array != 0] = 1

    if clumpbuffer is not None:
        sz = (2*clumpbuffer) + 1  # translate buffer to kernel size
        kernel = np.ones((sz, sz))
        clump_array = apply_convolve_filter(clump_array, kernel)
        clump_array[clump_array > 0] = 1    
        #clump_diag = False
        s_clump = [[0, 1, 0], [1, 1, 1], [0, 1, 0]] #no diagonals
    
    else:
        #clump_diag = True
        s_clump = [[1, 1, 1], [1, 1, 1], [1, 1, 1]] #yes diagonals

    if split_mask is not None:
    
        split_ind = np.nonzero(split_mask)
        split_clump_array = np.zeros_like(clump_array)
        split_clump_array[split_ind] = clump_array[split_ind]
        clump_array[split_ind] = 0

        clump_ID_array, num_feat = ndimage.label(clump_array, structure=s_clump)
        split_ID_array, num_split_feat = ndimage.label(split_clump_array, 
                                                        structure=s_clump)
        #change label ids to follow clump IDs in sequence
        split_ind = np.nonzero(split_ID_array)
        split_ID_array[split_ind] += num_feat
        clump_ID_array[split_ind] = split_ID_array[split_ind]

    else:
        """Run clumping tool to identify fills and carves"""
        clump_ID_array, num_feat = ndimage.label(clump_array, structure=s_clump)
      
    if save_tif:
        pyct.npy2tif(clump_ID_array, stencil_tif, clump_ID_tif, nodata=0)
   

    return clump_ID_array

