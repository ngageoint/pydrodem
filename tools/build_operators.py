"""
Created on Wed Sep 25 08:11:39 2019

@author: Kimmy McCormack
"""

import sys

import numpy as np
import scipy.sparse as sparse
from scipy.sparse import block_diag
from scipy.sparse import diags as spdiags

np.set_printoptions(threshold=sys.maxsize)


def build_grad_operator(dx, dy, nx, ny,
                        gfactor=1e3, gdir=2, step=1,
                        scheme="central", buildDiv=True,
                        gradDtype=np.float32):

    #s = step
    Grad = None
    Div = None

    ddiag = np.sqrt(dx**2 + dy**2)

    if scheme == "central":

        """build d/dy matrix"""
        diag_ty = np.hstack([2*np.ones(nx), np.ones(nx*(ny-2))])
        diag_cy = np.hstack(
            [-2*np.ones(nx), np.zeros(nx*(ny-2)), 2*np.ones(nx)])
        diag_by = np.hstack([-1*np.ones(nx*(ny-2)), -2*np.ones(nx)])
        diagonals_y = [diag_ty, diag_cy, diag_by]
        diffy_matrix = (gfactor/(2.*dy))*spdiags(diagonals_y, [nx, 0, -nx])

        """build block matrix for d/dx"""
        diag_t = np.hstack([-2., -1*np.ones(nx-2)])
        diag_c = np.hstack([2, np.zeros(nx-2), -2])
        diag_b = np.hstack([np.ones(nx-2), 2])
        diagonals = [diag_t, diag_c, diag_b]
        diff_block = spdiags(diagonals, [1, 0, -1])  # .toarray()
        bdiag = [diff_block]*ny
        """place blocks along diagonal to build d/dx matrix"""
        diffx_matrix = (gfactor/(2.*dx))*block_diag(bdiag)

    elif scheme == "forward":

        ones = np.ones(nx)
        nones = -1*np.ones(nx)
        zeros = np.zeros(nx)
        neg_110 = np.hstack([nones[1::], 0])
        pos_neg_11 = np.hstack([ones[1::], -1])
        pos_001 = np.hstack([zeros[1::], 1])

        """ d/dx matrix"""
        tdiag = np.tile(neg_110, ny)[0:-1]
        cdiag = np.tile(pos_neg_11, ny)
        bdiag = np.hstack([pos_001[1::], np.tile(pos_001, ny-1)])
        diffx_matrix = (gfactor/(dx))*spdiags(
            [tdiag, cdiag, bdiag],
            [1, 0, -1])

        """ d/dy matrix"""
        tdiag = np.hstack([ones, np.tile(zeros, ny-2)])
        cdiag = np.hstack([nones, np.tile(ones, ny-1)])
        bdiag = np.tile(nones, ny-1)
        diffy_matrix = (gfactor/(dy))*spdiags(
            [tdiag, cdiag, bdiag],
            [nx, 0, -nx])

    if gdir == 2:
        Grad = sparse.vstack([diffx_matrix, diffy_matrix], dtype=gradDtype)

        if buildDiv:
            Div = sparse.hstack([diffx_matrix, diffy_matrix], dtype=gradDtype)

    elif gdir == 4:
        """
        Build diagnonal central diffence matrices for all 4 directions
        """

        if scheme == "central":
            middle_20 = np.tile(np.hstack([-2, -1*np.ones(nx-2), 0]), (ny-2))
            middle_02 = np.tile(np.hstack([0, np.ones(nx-2), 2]), (ny-2))
            middle_22 = np.tile(np.hstack([2, np.zeros(nx-2), -2]), (ny-2))

            end_neg20 = np.hstack([-2*np.ones(nx-1), 0])
            end_neg02 = np.hstack([0, -2*np.ones(nx-1)])
            end_20 = np.hstack([2*np.ones(nx-1), 0])
            end_02 = np.hstack([0, 2*np.ones(nx-1)])

            """build d/dne matrix"""
            diag_tne = np.hstack([end_02, middle_02, 0])
            diag_cne = np.hstack([end_neg02, middle_22, end_20])
            diag_bne = np.hstack([0, middle_20, end_neg20])
            diagonals_ne = [diag_tne, diag_cne, diag_bne]
            ddne_matrix = (gfactor/(2.*ddiag))*spdiags(
                diagonals_ne,
                [nx-1, 0, -(nx-1)],
                dtype=gradDtype)

            """build d/dse matrix"""
            diag_tse = np.hstack([end_neg20, middle_20[0:-1]])
            diag_cse = np.hstack([end_20, middle_22, end_neg02])
            diag_bse = np.hstack([middle_02[1::], end_02])

            diagonals_se = [diag_tse, diag_cse, diag_bse]
            ddnw_matrix = (-1*gfactor/(2.*ddiag))*spdiags(
                diagonals_se,
                [nx+1, 0, -(nx+1)],
                dtype=gradDtype)

        elif scheme == "forward":

            neg_011 = np.hstack([0, nones[1::]])
            pos_011 = np.hstack([0, ones[1::]])
            pos_110 = np.hstack([ones[1::], 0])
            pos_100 = np.hstack([1, zeros[1::]])
            neg_pos_11 = np.hstack([-1, ones[1::]])

            """ d/dne matrix"""
            tdiag = np.hstack([pos_011, np.tile(pos_001, ny-2), 0])
            cdiag = np.hstack([neg_011, np.tile(pos_neg_11, ny-2), pos_110, 0])
            bdiag = np.hstack([0, np.tile(neg_110, ny-1), 0])
            ddne_matrix = (gfactor/(ddiag))*spdiags(
                [tdiag, cdiag, bdiag],
                [(nx-1), 0, -(nx-1)])

            """ d/dnw matrix"""
            tdiag = np.hstack([pos_110, np.tile(pos_100, ny-2)[0:-1]])
            cdiag = np.hstack([neg_110, np.tile(neg_pos_11, ny-2), pos_011])
            bdiag = np.hstack([nones[1::], np.tile(neg_011, ny-2)])
            ddnw_matrix = (gfactor/(ddiag))*spdiags(
                [tdiag, cdiag, bdiag],
                [(nx+1), 0, -(nx+1)])

        Grad = sparse.vstack([diffx_matrix,
                              ddne_matrix,
                              diffy_matrix,
                              ddnw_matrix], dtype=gradDtype)
        if buildDiv:
            Div = sparse.hstack([diffx_matrix,
                                 ddne_matrix,
                                 diffy_matrix,
                                 ddnw_matrix], dtype=gradDtype)

    Grad.eliminate_zeros()
    if buildDiv:
        Div.eliminate_zeros()

    return Grad, Div


def buildD8_operator(nx, ny, dx, dy, gfactor=1e3, step=1, gradDtype=np.float32):
    """build all 8 forward difference martices for d8 directions"""

    ddiag = np.sqrt(dx**2 + dy**2)
    ones = np.ones(nx)
    nones = -1*np.ones(nx)
    zeros = np.zeros(nx)

    neg_011 = np.hstack([0, nones[1::]])
    pos_011 = np.hstack([0, ones[1::]])
    neg_110 = np.hstack([nones[1::], 0])
    pos_110 = np.hstack([ones[1::], 0])
    pos_neg_11 = np.hstack([ones[1::], -1])
    neg_pos_11 = np.hstack([-1, ones[1::]])
    pos_001 = np.hstack([zeros[1::], 1])
    pos_100 = np.hstack([1, zeros[1::]])

    """ d/dn matrix"""
    tdiag = np.hstack([ones, np.tile(zeros, ny-2)])
    cdiag = np.hstack([nones, np.tile(ones, ny-1)])
    bdiag = np.tile(nones, ny-1)
    ddn_matrix = (gfactor/(dy))*spdiags(
        [tdiag, cdiag, bdiag],
        [nx, 0, -nx])

    """ d/dne matrix"""
    tdiag = np.hstack([pos_011, np.tile(pos_001, ny-2), 0])
    cdiag = np.hstack([neg_011, np.tile(pos_neg_11, ny-2), pos_110, 0])
    bdiag = np.hstack([0, np.tile(neg_110, ny-1), 0])
    ddne_matrix = (gfactor/(ddiag))*spdiags(
        [tdiag, cdiag, bdiag],
        [(nx-1), 0, -(nx-1)])

    """ d/de matrix"""
    tdiag = np.tile(neg_110, ny)[0:-1]
    cdiag = np.tile(pos_neg_11, ny)
    bdiag = np.hstack([pos_001[1::], np.tile(pos_001, ny-1)])
    dde_matrix = (gfactor/(dx))*spdiags(
        [tdiag, cdiag, bdiag],
        [1, 0, -1])

    """ d/dse matrix"""
    tdiag = np.tile(neg_110, ny-1)[0:-1]
    cdiag = np.hstack([pos_110, np.tile(pos_neg_11, ny-2), neg_011])
    bdiag = np.hstack([pos_001[1::], np.tile(pos_001, ny-3), pos_011])
    ddse_matrix = (gfactor/(ddiag))*spdiags(
        [tdiag, cdiag, bdiag],
        [(nx+1), 0, -(nx+1)])

    """ d/ds matrix"""
    tdiag = np.tile(nones, ny-1)
    cdiag = np.hstack([np.tile(ones, ny-1), nones])
    bdiag = np.hstack([np.tile(zeros, ny-2), ones])
    dds_matrix = (gfactor/(dy))*spdiags(
        [tdiag, cdiag, bdiag],
        [nx, 0, -nx])

    """ d/dsw matrix"""
    tdiag = np.hstack([np.tile(neg_011, ny-1), 0])
    cdiag = np.hstack([pos_011, np.tile(neg_pos_11, ny-2), neg_110, 0])
    bdiag = np.hstack([0, np.tile(pos_100, ny-2), pos_110, 0])
    ddsw_matrix = (gfactor/(ddiag))*spdiags(
        [tdiag, cdiag, bdiag],
        [(nx-1), 0, -(nx-1)])

    """ d/dw matrix"""
    tdiag = np.tile(pos_100, ny)[0:-1]
    cdiag = np.tile(neg_pos_11, ny)
    bdiag = np.hstack([nones[1::], np.tile(neg_011, ny-1)])
    ddw_matrix = (gfactor/(dx))*spdiags(
        [tdiag, cdiag, bdiag],
        [1, 0, -1])

    """ d/dnw matrix"""
    tdiag = np.hstack([pos_110, np.tile(pos_100, ny-2)[0:-1]])
    cdiag = np.hstack([neg_110, np.tile(neg_pos_11, ny-2), pos_011])
    bdiag = np.hstack([nones[1::], np.tile(neg_011, ny-2)])
    ddnw_matrix = (gfactor/(ddiag))*spdiags(
        [tdiag, cdiag, bdiag],
        [(nx+1), 0, -(nx+1)])

    D8mat = sparse.vstack([dde_matrix, ddne_matrix,
                          ddn_matrix, ddnw_matrix,
                          ddw_matrix, ddsw_matrix,
                          dds_matrix, ddse_matrix,
                           ], dtype=gradDtype)

    D8mat.eliminate_zeros()

    return D8mat


def build_Laplacian_operator(Grad, Div, g_factor, Dtype=np.float16):

    Lap = ((1./g_factor)*Div*Grad).astype(Dtype)  # Laplacian operator

    return Lap


def apply_boundary_cond(Mat, rows, w_factor):
    for r in rows:
        Mat.data[Mat.indptr[r]:Mat.indptr[r+1]][0] = np.int8(w_factor)
        Mat.data[Mat.indptr[r]:Mat.indptr[r+1]][1::] = np.int8(0)
    Mat.eliminate_zeros()

    return Mat


def smoothing_operator(weights, nx, ny, w_factor, g_factor):
    diag_val = []
    diag_loc = []
    f_size = (len(weights))
    f_size_mat = f_size*2 - 1

    weights = (np.asarray(weights)*w_factor).astype(np.int8)
    w_mat = build_kernel(weights)

    """normalization vector"""
    norm_add = []
    for i, w in enumerate(weights):
        norm_temp = []
        for j in range(0, f_size):
            add = g_factor/np.sum(w_mat[f_size-i-1::, f_size-j-1::])
            norm_temp.append(add)
            norm_piece = np.hstack((norm_temp,
                                    norm_temp[-1]*np.ones(nx-(2*f_size)),
                                    norm_temp[::-1]))
        norm_add.append(norm_piece)

    norm_end = np.asarray(norm_add[0::]).reshape(len(norm_add*nx),)
    norm_middle = np.tile(norm_add[-1], (ny-2*f_size))
    norm_vec = np.hstack((norm_end,
                          norm_middle,
                          norm_end[::-1]))  # .astype(np.float16)

    """sparse diagonal smoothing operator"""
    for i, w in enumerate(weights):
        for j in range(0, f_size):
            if i > 0 and j > 0:
                loc = i*nx - j
                weight = weights[max(i, j)]
                zeros = np.zeros(j)
                ones = np.ones(nx-j)
                diag1 = weight * np.tile(np.hstack((zeros, ones)), ny)
                diag_val.append(diag1)
                diag_val.append(diag1)
                diag_loc.append(loc)
                diag_loc.append(-1*loc)

            loc2 = i*nx + j
            weight = weights[max(i, j)]
            zeros = np.zeros(j)
            ones = np.ones(nx-j)
            diag2 = weight * np.tile(np.hstack((ones, zeros)), ny)
            if loc2 != 0:
                diag_val.append(diag2)
                diag_val.append(diag2)
                diag_loc.append(loc2)
                diag_loc.append(-1*loc2)
            else:
                diag_val.append(diag2)
                diag_loc.append(loc2)

        smooth_matrix = spdiags(diag_val,
                                diag_loc,
                                format="csr",
                                dtype=np.int8)

    return smooth_matrix, norm_vec


def build_kernel(weights, dtype=np.int8, normalize=False):

    f_size = (len(weights))
    f_size_mat = f_size*2 - 1
    """build full smoothing kernel"""
    w_mat = np.zeros((f_size_mat, f_size_mat), dtype=dtype)
    for i, w in enumerate(weights[::-1]):
        if i == 0:
            w_mat[i, :] = w
            w_mat[:, i] = w
            w_mat[-1-i, :] = w
            w_mat[:, -1-i] = w
        elif i == f_size-1:
            w_mat[i, i] = w
        else:
            w_mat[i, i:-1-i] = w
            w_mat[-i-1, i:-1-i] = w
            w_mat[i:-1-i, i] = w
            w_mat[i:-i, -i-1] = w

    if normalize:
        w_sum = np.sum(w_mat)
        w_mat = (1./w_sum)*w_mat

    return w_mat
