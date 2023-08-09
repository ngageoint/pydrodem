#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:34:24 2020

Based on code provided from Applied Reseach Labs, Austin TX

@author: Sal Candela, Kimmy McCormack
"""

import sys

import os
import time
import warnings
warnings.simplefilter('ignore')
import random
import numpy as np
import pandas as pd
import traceback
from scipy.interpolate import griddata

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools.build_operators as bop
import tools.derive as pdt
from tools.print import print_time

starttime = time.time()


#########################################################
##########    MODEL BUILD FUNCTIONS    ##################
#########################################################


class ModelPrep():

    def __init__(self, source_path, dst_path, source_filt_method=None,
                 n_tiles=10, spatial_filt=None, bbox=[0, 0, 0, 0],
                 quantile_filt=True, quant=[0.05, 0.99], const_filt=True,
                 filt_value=-10.0, random_enable=False, random_frac=0.5):

        self.source_path = source_path
        self.dst_path = dst_path
        self.source_filt_method = source_filt_method
        self.n_tiles = n_tiles
        self.spatial_filt = spatial_filt
        self.bbox = bbox
        self.quantile_filt = quantile_filt
        self.quant = quant
        self.const_filt = const_filt
        self.filt_value = filt_value
        self.random_enable = random_enable
        self.random_frac = random_frac

    def create_bulk_df(self):

        files_to_process = [os.path.join(d, x)
                            for d, dirs, files in os.walk(self.source_path)
                            for x in files if x.endswith(".h5")]

        # if random tiles are requested, filter the list first
        if self.source_filt_method == 'n_tiles':
            print('Random tiles being selected: ' + str(self.n_tiles))
            files_to_process = random.choices(files_to_process, k=self.n_tiles)

        df_list_all = []
        for file in files_to_process:
            hdf_file = pd.HDFStore(file, mode='r')
            keys = hdf_file.keys()

            df_list = [GetFile(hdf_file, key=key) for key in keys]
            df_list_all.extend(df_list)

        df_combine = pd.concat(df_list_all)


       # deal with gross outliers
        if self.const_filt:
            print("Removing gross outliers (constant method)")

            """Changed from greater than to less than.... double check (Kimmy)"""
            df_combine = df_combine[df_combine.h_diff < self.filt_value]
            print("rows: " + str(df_combine.shape[0]))

        # filter data by additonal methods
        if self.spatial_filt == 'bbox':
            print('additional filter applied: bounding box')
            df_combine = df_combine[(df_combine.lat >= self.bbox[0])
                                    & (df_combine.lat <= self.bbox[1])
                                    & (df_combine.lon <= self.bbox[2])
                                    & (df_combine.lon >= self.bbox[3])]
            print("rows: " + str(df_combine.shape[0]))

        if self.quantile_filt:
            print('Quantile filter selected, removing values between: ' +
                  str(self.quant[0]*100) + '% and ' + str(self.quant[1]*100) + '%')
            df_combine = df_combine[df_combine.h_diff.between(
                df_combine.h_diff.quantile(self.quant[0]),
                df_combine.h_diff.quantile(self.quant[1]))]
            print("rows remaining: " + str(df_combine.shape[0]))

        # randomly select a subsample
        if self.random_enable:
            print('Random sample Enabled: ' + str(self.random_frac*100)+'%')
            df_combine = df_combine.sample(frac=self.random_frac)
            print("remaining rows: " + str(df_combine.shape[0]))

        # reset all the index values
        df_combine.reset_index(drop=True)

        output_hdf = os.path.join(self.dst_path, 'combined_df.h5')
        df_combine.to_hdf(output_hdf, key='combined', mode='w')

    def create_slice_df(self, slice_name, slice_val, col_filter=None):
        """
        Build 2D slices of 3D lookup table

        """
        try:
            # get all the files in the directory that are dataframes
            files_to_process = [os.path.join(d, x)
                                for d, dirs, files in os.walk(self.source_path)
                                for x in files if x.endswith(".h5")]

            #print('number of dataframes to sort and combine = {0}'.format(len(files_to_process)))
            #files_to_process = files_to_process[0:3000]  ## TEMP/TESTING

            # if random tiles are requested, filter the list first
            if self.source_filt_method == 'n_tiles':
                print('Random tiles being selected: ' + str(self.n_tiles))
                files_to_process = random.choices(
                    files_to_process, k=self.n_tiles)

            # #combine all dataframes
            df_list = [GetFile(file,
                               col_filter=col_filter,
                               slice_name=slice_name,
                               slice_val=slice_val) for file in files_to_process]
            df_combine = pd.concat(df_list)
            del df_list

            #print_time('   combined df for slope = {0}'.format(slice_val), starttime, CPU=True)

            # deal with gross outliers
            if self.const_filt:
                #print("Removing gross outliers (constant method)")
                df_combine = df_combine[df_combine.h_diff > self.filt_value]
                #print("rows: " + str(df_combine.shape[0]))

            # filter data by additonal methods
            if self.spatial_filt == 'bbox':
                #print('additional filter applied: bounding box')
                df_combine = df_combine[(df_combine.lat >= self.bbox[0])
                                        & (df_combine.lat <= self.bbox[1])
                                        & (df_combine.lon <= self.bbox[2])
                                        & (df_combine.lon >= self.bbox[3])]
                #print("rows: " + str(df_combine.shape[0]))

            if self.quantile_filt:
                #print('Quantile filter selected, removing values between: ' +
                #      str(self.quant[0]*100) + '% and ' + str(self.quant[1]*100) + '%')
                df_combine = df_combine[df_combine.h_diff.between(
                    df_combine.h_diff.quantile(self.quant[0]),
                    df_combine.h_diff.quantile(self.quant[1]))]
                #print("rows remaining: " + str(df_combine.shape[0]))

            # randomly select a subsample
            if self.random_enable:
                print('Random sample Enabled: ' +
                      str(self.random_frac*100)+'%')
                df_combine = df_combine.sample(frac=self.random_frac)
                #print("remaining rows: " + str(df_combine.shape[0]))

            # reset all the index values
            df_combine.reset_index(drop=True)  # add this into base function

            output_hdf = os.path.join(self.dst_path, '{0}_slice_{1}.h5'.format(slice_name,
                                                                               slice_val))
            df_combine.to_hdf(output_hdf, key='slice', mode='w')

            print_time('   combined and filtered df for slope = {0}'.format(slice_val),
                       starttime, CPU=True)

        except Exception as e:
            outst = "Exception occurred: {1}".format(e)
            traceback_output = traceback.format_exc()
            print(outst)
            print("TRACEBACK : {0}\n".format(traceback_output))


def create_lkup_table(input_df, groupby_names, agg_var, n_cutoff):

    hdf_file = pd.HDFStore(input_df, mode='r')
    keys = hdf_file.keys()

    df_list = [GetFile(hdf_file, key=key) for key in keys]
    df_combine = pd.concat(df_list)

    slope = df_combine['slope_smooth'].min()

    df_grouped = df_combine.groupby(groupby_names)

    """compute the mean of the veg bias"""
    df_mean = df_grouped.agg({agg_var: 'mean'}).reset_index()
    df_mean = df_mean.rename(columns={agg_var: "{0}_mean".format(agg_var)})

    """compute the std of the veg bias, add to df_mean"""
    df_std = df_grouped.agg({agg_var: 'std'}).reset_index()
    df_std = df_std.rename(columns={agg_var: "{0}_std".format(agg_var)})
    df_mean["{0}_std".format(agg_var)] = df_std["{0}_std".format(agg_var)]

    """find the count for each lookup cell"""
    df_mean['count'] = df_grouped.size().values

    """filter by number of observations"""
    df_mean = df_mean[df_mean['count'] > n_cutoff]
    df_lkup_mean = df_mean.pivot(index=groupby_names[0],
                            columns=groupby_names[1],
                            values="{0}_mean".format(agg_var))

    df_lkup_std = df_mean.pivot(index=groupby_names[0],
                            columns=groupby_names[1],
                            values="{0}_std".format(agg_var))

    return df_mean, df_lkup_mean, df_lkup_std, slope


def interp_lkup_table(df_agg, df_lkup, slope_val, treecover_axis, treeheight_axis,
                      treecover_mesh, treeheight_mesh, TC_cutoff=None):


    npts_lkup = len(treecover_axis)*len(treeheight_axis)

    TC_full = df_lkup.index.values.astype(np.int)
    TH_full = df_lkup.columns.values


    """build full lookup table and fill with known values"""
    lkup_mat = np.empty((len(treecover_axis),len(treeheight_axis)))
    lkup_mat.fill(np.nan)

    for tc in TC_full:
        if len(df_lkup.loc[tc].values) > len(treeheight_axis):
            lkup_mat[tc,:] = df_lkup.loc[tc].values[0:len(treeheight_axis)]
        else:
            lkup_mat[tc,TH_full] = df_lkup.loc[tc].values

    bias_orig = lkup_mat.copy()
    bias_cutoff= bias_orig.copy()

    # if TC_cutoff is not None:
    #     bias_cutoff[:,0:TC_cutoff] = 0

    """ apply smoothing filter"""
    weights = [1., .67,  0.33] #5x5 smoothing filter
    weights = [1., .5] #3x3 smoothing filter
    kernel = bop.build_kernel(weights, dtype=np.float32, normalize=True)
    bias_smooth0 = pdt.apply_convolve_filter(bias_cutoff, kernel)



    full_ind = np.where(np.isfinite(bias_smooth0))

    if TC_cutoff is not None:

        max0 = full_ind[0][full_ind[1]==0].max()
        max1 = full_ind[1].max()

        bias_smooth0[0:TC_cutoff,0:max1] = 0
        bias_smooth0[0:max0,0:TC_cutoff-1] = 0

    TC_points = treecover_mesh[full_ind].flatten()
    TH_points = treeheight_mesh[full_ind].flatten()
    bias_points = bias_smooth0[full_ind].flatten()



    """Reshape grids"""
    theight_vec_all = treeheight_mesh.reshape(npts_lkup,)
    TC_vec_all = treecover_mesh.reshape(npts_lkup,)
    bias_vec_smooth = bias_smooth0.copy().reshape(npts_lkup,)
    bias_arr_interp = griddata((TC_points, TH_points),
                               bias_points,
                               (treecover_mesh, treeheight_mesh),
                               method='nearest')


    """Smoothing filter after interpolation"""

    weights = [1., .5] #3x3 smoothing filter
    kernel3 = bop.build_kernel(weights, dtype=np.float32, normalize=True)
    weights = [1., .67,  0.33] #5x5 smoothing filte
    kernel5 = bop.build_kernel(weights, dtype=np.float32, normalize=True)

    #bias_array_final = pdt.apply_convolve_filter(bias_arr_interp, kernel5)
    bias_array_final = bias_arr_interp

    bias_array_final[0,:] = 0
    bias_array_final[:,0] = 0

    bias_array_final = pdt.apply_convolve_filter(bias_array_final, kernel5)

    bias_array_final[0,:] = 0
    bias_array_final[:,0] = 0

    """save lookup table as one dataframe"""
    bias_vec_final = bias_array_final.reshape(npts_lkup,)

    slope_list = [slope_val] * npts_lkup
    """Create dataframe of beam data from dictionary"""
    data_lkup_dct = {'slope_smoth':slope_list,
                     'tree_height_m':theight_vec_all,
                     'treecover':TC_vec_all,
                     'vegbias':bias_vec_final}

    """Build dataframe"""
    lkup_table_df = pd.DataFrame(data_lkup_dct)


    return lkup_table_df, bias_orig, bias_cutoff, bias_arr_interp, bias_array_final




def GetFile(file_name, col_filter=None,
            slice_name=None, slice_val=None, key=None):

    #hack for comprehension list join of multiple frames
    df = pd.read_hdf(file_name, key=key)

    if col_filter is not None:
        df = df[col_filter]

    if slice_name is not None:
        df = df[df[slice_name] == slice_val]

    # Remove NaN values in treecover
    df = df.loc[df['treecover'] < 200]

    return df

