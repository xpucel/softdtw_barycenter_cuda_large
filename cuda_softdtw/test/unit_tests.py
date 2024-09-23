#!/usr/bin/env python

"""
Unit tests for the cuda SoftDTW package
Depends on the tslearn library and the standard unittest library
"""

import unittest
import warnings

import numpy as np

from numba import NumbaPerformanceWarning

from cuda_softdtw import softdtw_dist, softdtw_grad
from cuda_softdtw.main.softdtw_dist import merge_chunks
from cuda_softdtw.test._tests import (
    make_xy,
    distance_matrix,
    cpu_softdtw_dist,
    cpu_softdtw_grad,
)

__author__ = "Xavier Pucel"
__copyright__ = "Copyright 2023"
__license__ = "Apache-2.0"
__version__ = "1.0"
__maintainer__ = "Xavier Pucel"
__email__ = "firstname.lastname@onera.fr"
__status__ = "Prototype"

NB_X = 8
NB_Y = 8
LEN_X = 2**11
LEN_Y = 2**11
D = 2
CHUNK_SHAPE = (NB_X, NB_Y, 2**6, 2**6)


class SoftDTWDist(unittest.TestCase):
    def setUp(self):
        self.x, self.y = make_xy(NB_X, NB_Y, LEN_X, LEN_Y, D)
        warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

    def test_softdtw_dist(self):
        D = distance_matrix(self.x, self.y)
        Rref, distref = cpu_softdtw_dist(D)
        R, distances = softdtw_dist(
            self.x, self.y, omega=0, chunk_shape=CHUNK_SHAPE, return_R=True
        )
        R = merge_chunks(self.x, self.y, R)

        self.assertTrue(np.allclose(R, Rref))

    def test_softdtw_grad(self):
        Rchunks, dists = softdtw_dist(
            self.x, self.y, chunk_shape=CHUNK_SHAPE, return_R=True
        )
        E, gx = softdtw_grad(
            self.x, self.y, Rchunks, chunk_shape=CHUNK_SHAPE, return_E=True
        )
        Eref, gxref = cpu_softdtw_grad(self.x, self.y)

        self.assertTrue(np.allclose(E, Eref))
        self.assertTrue(np.allclose(gx, gxref))
