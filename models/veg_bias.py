# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 13:39:32 2020

@author: Sal, Kimmy


Script ot process and plot the vegation bias model based on database
of elevation differences between TanDEM-X DEM and ICESat-2 ground photons.

we build a series of  2D lookup tables with input parameters of:
        1) Slope of the TanDEM-X DEM that has been smoothed with a 9x9 pixel
           distance-weighted filter to remove sharp jumps in slope due to
           small features (buildings, treelines, etc)
        2) Landsat-derived treecover percent - resampled from 30m to 12m
           resolution using a bilinear interpolation
        3) Landsat/GEDI-derived tree height - resampled from 30m to 12m
           resolution using a bilinear interpolation



"""


import os, sys
import time
import numpy as np
import scipy
import glob
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sys.path.append(os.path.dirname(os.getcwd()))
from vegremovetools import create_lkup_table, interp_lkup_table
from tools.print import print_time


plt.close('all')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def plot_slice_tables(slope, bias_raw_array, bias_cut_array,
                      bias_interp, bias_final_array):

    """PLOT lookup table progression"""
    pltmin = 0
    pltmax = 15

    """Plot biome progression"""
    fig = plt.figure(figsize=(15,4))
    fig.canvas.set_window_title('Slope = {0}: Mean Bias (m)'.format(slope))
    fig.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.subplots_adjust(wspace = 0.3, hspace = 0.5 )
    fig.suptitle('Slope = {0}: Mean Bias (m)'.format(slope))
    fig.tight_layout()

    ax0 = fig.add_subplot(1,3,1)
    ax0 = sns.heatmap(bias_raw_array, vmin=pltmin, vmax=pltmax)
    ax0.set_title('Input table')

    #ax1 = fig.add_subplot(1,4,2)
    #ax1 = sns.heatmap(bias_cut_array, vmin=pltmin, vmax=pltmax)
    #ax1.set_title('Cutoff table')

    ax3 = fig.add_subplot(1,3,2)
    ax3 = sns.heatmap(bias_interp, vmin=pltmin, vmax=pltmax)
    ax3.set_title('Interpolated')

    ax4 = fig.add_subplot(1,3,3)
    ax4 = sns.heatmap(bias_final_array, vmin=pltmin, vmax=pltmax)
    ax4.set_title('smoothed')

    ax0.set_ylabel('treecover %')
    ax0.set_xlabel('tree height (m)')
    plt.show()


def plot_lkup_compare(lkup_raw, lkup_final):

    names = list(lkup_raw.keys())

    fig_h = 3
    fig_w = len(names)//fig_h
    if len(names)%fig_h > 0:
        fig_w += 1

    fig_compare, ax_comp = plt.subplots(fig_h, fig_w, figsize=(3*fig_w,3*fig_h))
    fig_compare.canvas.set_window_title('Bias comparison')
    fig_compare.suptitle('Bias comparison')
    fig_compare.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.subplots_adjust(wspace = 0.3, hspace = 0.5 )
    pltmin = 0
    pltmax = 9

    for name, ax in zip(names, ax_comp.flat):

        data = lkup_final[name]
        data_raw = lkup_raw[name]

        data = data[0:40, 0:80]
        ax = sns.heatmap(data, ax=ax, vmin=pltmin, vmax=pltmax)
        zeroline = [0]

        if name == 999:

            data_raw0 = data_raw
            data_raw0[np.isnan(data_raw0)] = -1
            ax.contour(data_raw0, zeroline,
                            colors = "limegreen", linewidths = 1.0)

        else:

            data_raw[np.isnan(data_raw)] = -1

            ax.contour(data_raw, zeroline,
                            colors = "limegreen", linewidths = 1.0)
            ax.contour(data_raw0, zeroline, linestyles ='dashed',
                            colors = "limegreen", linewidths = 0.5)

        ax.set_title('LC Type: {0}'.format(name))


    if len(lkup_final)%fig_h == 2:
        ax_comp.flat[-1].set_visible(False)

    if len(lkup_final)%fig_h == 1:
        ax_comp.flat[-1].set_visible(False)
        ax_comp.flat[-2].set_visible(False)

    return



def plot_model_lines(LCtypes):

    slope = [0,10,20,30,40,50,60,70,80]
    treecover = np.linspace(0, 81, 200)

    fig, axs = plt.subplots(9,sharex = True,sharey=True,gridspec_kw={'hspace': 0})
    fig.suptitle('Bias as a function of treecover')
    axs.set_ylabel('Bias ( m )')
    axs.label_outer()
    for j in range(len(LCtypes)):
        val = mt.GetVegCoeff('biome', j)
        for i in range(len(slope)):
            Y = mt.VegModel(slope[i], treecover, val)
            axs[i].plot(treecover, Y,label = LCtypes[j])

    axs[0].legend(loc="upper right", fancybox= True,shadow= True,bbox_to_anchor = (1.12,1))
    fig.text(0.5, 0.04, 'Treecover %', ha='center')


    # line plots with treecover as a constant
    treecover = [0,10,20,30,40,50,60,70,80]
    slope = np.linspace(0, 81, 200)

    fig, axs = plt.subplots(9,sharex = True,sharey=True,gridspec_kw={'hspace': 0})
    fig.suptitle('Bias as a function of treecover')
    axs.set_ylabel('Bias ( m )')
    axs.label_outer()
    plt.rc('font', size=18)
    for j in range(len(LCtypes)):
        val = mt.GetVegCoeff('biome', j)
        for i in range(len(treecover)):
            Y = mt.VegModel(slope, treecover[i], val)
            axs[i].plot(slope, Y,label = LCtypes[j])

    axs[0].legend(loc="upper right", fancybox= True,shadow= True,bbox_to_anchor = (1.12,1))
    fig.text(0.5, 0.04, 'slope (degrees)', ha='center')

    plt.show()

    return



if __name__=="__main__":

    starttime = time.time()
    plot_tables = True
    n_threshold = 30

    groupby_names = ['treecover','tree_height_m']
    lkup_var = 'h_diff'
    TC_cutoff = 2

    src_path_hdf = os.path.join('/path/to/veg_bias_dfs')

    json_out = os.path.join('output-path',
                            'veg_bias_3d.json')


    # setup the meshgrid - values based on min/max in dataset
    slope_axis = np.linspace(0, 20, 21).astype(int)
    treeheight_axis = np.linspace(0, 30, 31)
    treecover_axis = np.linspace(0, 100, 101)
    treecover_mesh, treeheight_mesh = np.meshgrid(treeheight_axis, treecover_axis)

    """Create lookup tables"""
    bias_array_3D = np.empty((len(treecover_axis),
                             len(treeheight_axis),
                             len(slope_axis)))

    for i, slope in enumerate(slope_axis):

        input_df = glob.glob(os.path.join(src_path_hdf,'*_{0}.h5'.format(slope)))

        if len(input_df) == 0:
            print("slope =  {0} not found, skipping".format(slope))
            continue

        input_df = input_df[0]
        print_time('Creating lookup table for {0}/{1} dataframes'
                   .format(i+1, len(slope_axis)), starttime)

        df_agg, df_lkup_mean, df_lkup_std, slope = create_lkup_table(input_df,
                                           groupby_names, lkup_var, n_threshold)

        lkup_table_df, bias_raw_array, bias_cut_array,\
        bias_interp, bias_final_array = interp_lkup_table(df_agg,
                                                          df_lkup_mean, slope,
                                                           treecover_axis,
                                                           treeheight_axis,
                                                           treecover_mesh,
                                                           treeheight_mesh,
                                                           TC_cutoff=TC_cutoff)

        """add 2d slice to 3d array"""
        bias_array_3D[:,:,slope] = bias_final_array


        if plot_tables:
            plot_slice_tables(slope, bias_raw_array, bias_cut_array,
                      bias_interp, bias_final_array)


    """ 3D smoothing filter (really 3, 1D convolutions)"""
    k1 = np.array([0.5, 1.0, 0.5]) #the kernel along the 1st dimension
    k1 = (1./np.sum(k1))*k1

    # Convolve over all three axes in a for loop
    out = bias_array_3D.copy()
    for i, k in enumerate((k1, k1, k1)):
        out = scipy.ndimage.convolve1d(out, k, axis=i)
    out[0,:,:] = 0
    out[:,0,:] = 0

    """create and save dictionary of arrays"""
    bias_dct = {str(s): bias_array_3D[:,:,s] for s in slope_axis}


    """ Save dict of arrays as .json file"""

    json_dump = json.dumps(bias_dct, cls=NumpyEncoder)


    # Read data from file:
    data = json.load( open( json_out ) )
    sample_arr = np.array( data.get(list(data.keys())[0]) )

    bias_cube = np.empty((sample_arr.shape[0],
                          sample_arr.shape[1],
                          len(list(data.keys()))))

    """re build array """
    for s in slope_axis:
        bias_cube[:,:,s] = np.array(data.get(str(int(s))))






