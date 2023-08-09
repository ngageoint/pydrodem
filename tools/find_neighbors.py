# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:39:30 2019

@author: Kimmy McCormack

Uses the file name of a TDT cell to identify the 8 neighboring cells - allows
for the creation of a buffered cell during processing to elminate edge 
effects between cells

"""
import numpy as np


def get_nb_paths(bound_cells, searchlines):
    """
    Get paths of list of input tiles

    Parameters
    ----------
    bound_cells : list
        list of TDT ceell names
    searchlines : list
        list of all full paths to DEM cells

    Returns
    -------
    dict
        dictionary of cell names and paths
    """

    try:
        cell_paths = []
        for cell in bound_cells:
            cell_search = cell
            cell_path = ''
            """Search TDT txt file for path of TDT cell"""
            for line in searchlines:
                if cell_search in line:
                    cell_path = line
                    break

            cell_paths.append(cell_path)

    except ValueError as err:
        print(err)

    cell_path_dict = dict(zip(bound_cells, cell_paths))

    return cell_path_dict


