#!/usr/bin/env python

"""
Utility functions for CUDA softdtw implementation
"""

__author__ = "Xavier Pucel"
__copyright__ = "Copyright 2023"
__license__ = "Apache-2.0"
__version__ = "1.0"
__maintainer__ = "Xavier Pucel"
__email__ = "firstname.lastname@onera.fr"
__status__ = "Prototype"


import math

import numpy as np
from numba import cuda


@cuda.jit(device=True)
def squared_euclidean_distance(x, y):
    """
    :param x shape (d,)
    :param y shape (d,)
    returns a float
    """
    tmp = 0.0
    for i in range(x.shape[0]):
        tmp += (x[i] - y[i]) ** 2
    return tmp


@cuda.jit(device=True)
def softmin(a, b, c, gamma, gamma_ninv):
    """Softmin of 3 scalars
    gamma_niv must equal ``-1.0 / gamma``
    """
    na = gamma_ninv * a
    nb = gamma_ninv * b
    nc = gamma_ninv * c
    if na < nb:
        if nb < nc:
            mx = nc
            ea = math.exp(na - nc)
            eb = math.exp(nb - nc)
            ec = 1.0
        else:
            mx = nb
            ea = math.exp(na - nb)
            eb = 1.0
            ec = math.exp(nc - nb)
    else:
        if na < nc:
            mx = nc
            ea = math.exp(na - nc)
            eb = math.exp(nb - nc)
            ec = 1.0
        else:
            mx = na
            ea = 1.0
            eb = math.exp(nb - na)
            ec = math.exp(nc - na)
    sm = -gamma * (math.log(ea + eb + ec) + mx)
    return sm
