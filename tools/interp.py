#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unstructured interpolation using KDTree

Original code source - https://stackoverflow.com/a/3119544

Modified by Kimmy McCormack
    - Changed from just inverse distance weighting (IDW) for interpolating
    points onto points to include bilinear
    interpolation when interpolating from a grid to points.

"""

import numpy as np
from scipy.interpolate import interp2d
from scipy.spatial import cKDTree as KDTree


class InterpTree:

    """
    Usage:

        tree = InterpTree( X, z )  -- data points, values

        INVERSE DISTANCE:
            interp_values = tree.invDist(q, nnear=3, eps=0, p=1, weights=None)
            (interpolates z from the 3 points nearest each query point q)

        BILINEAR (not complete)
            interp_values = tree.bilinear(q, eps=0)


    """

    def __init__(self, X, z, leafsize=20):
        """Initialize and build KDTree for input points

        Parameters
        ----------
        X : tuple
            tuple of arrays for data point locations ([x],[y])
        z : array
            data values at locations
        leafsize : int, optional
            inimum number of points in a given node, by default 20

        """
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree(X, leafsize=leafsize)  # build the tree
        self.z = z
        self.wn = 0
        self.wsum = None

    def invDist(self, q, nnear=6, eps=0, p=1, dmax=np.inf, weights=None):
        """ 
        Inverse-distance-weighted interpolation using KDTree:

            interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None)
            interpolates z from the 3 points nearest each query point q

        Parameters
        ----------
        q : array-like
            query points to interpolate values onto
        nnear : int, optional
            number of neighboring points to use in interpolation, by default 6
        eps : int, optional
            epsilon distance to use for approximating distances.
            dist <= (1 + eps) * true nearest, by default 0
        p : int, optional
            inverse distance power - use 1 / distance**p,  by default 1
        dmax : float, optional
            max distance to search for neighbors, by default np.inf
        weights : array-like, optional
            optional multipliers for 1 / distance**p, of the same shape as q, by default None

        Returns
        -------
        list
            interpolated values for input query points

        """

        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query(q, k=nnear, eps=eps)

        interpol = np.zeros((len(self.distances),) + np.shape(self.z[0]))

        jinterpol = 0
        for dist, ix in zip(self.distances, self.ix):

            if nnear == 1:
                """nearest neighbor"""
                wz = self.z[ix]

            elif dist[0] < 1e-10:
                """query point is very very close to data point, use data point"""
                wz = self.z[ix[0]]

            elif dist[nnear-1] > dmax:
                """check for min dist to ground point. Skip if query point
                does not have nnear data points within dmax"""
                wz = -999

            else:
                """weight points by 1/dist"""
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot(w, self.z[ix])

            interpol[jinterpol] = wz
            jinterpol += 1

        return interpol if qdim > 1 else interpol[0]

    def bilinear(x, y, points):
        """
        Interpolate (x,y) from values associated with four points.

        The four points are a list of four triplets:  (x, y, value).
        The four points can be in any order.  They should form a rectangle.

            >>> bilinear_interpolation(12, 5.5,
            ...                        [(10, 4, 100),
            ...                         (20, 4, 200),
            ...                         (10, 6, 150),
            ...                         (20, 6, 300)])
            165.0

        """
        # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

        points = sorted(points)               # order points by x, then by y
        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

        if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
            raise ValueError('points do not form a rectangle')
        if not x1 <= x <= x2 or not y1 <= y <= y2:
            raise ValueError('(x, y) not within the rectangle')

        return (q11 * (x2 - x) * (y2 - y) +
                q21 * (x - x1) * (y2 - y) +
                q12 * (x2 - x) * (y - y1) +
                q22 * (x - x1) * (y - y1)
                ) / ((x2 - x1) * (y2 - y1) + 0.0)
