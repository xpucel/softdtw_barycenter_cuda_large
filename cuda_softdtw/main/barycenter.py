#!/usr/bin/env python

"""
Computes the barycenter of time series using softdtw barycenter
Uses CUDA for performance
"""

import math
import time
import warnings

import numpy as np
from  numba.core.errors import NumbaPerformanceWarning

from scipy.optimize import minimize
import matplotlib.pyplot as plt

from cuda_softdtw import softdtw_dist, softdtw_grad

import tslearn.barycenters


__author__ = "Xavier Pucel"
__copyright__ = "Copyright 2023"
__license__ = "Apache-2.0"
__version__ = "1.0"
__maintainer__ = "Xavier Pucel"
__email__ = "firstname.lastname@onera.fr"
__status__ = "Prototype"


def softdtw_barycenter(
    X,
    gamma=1.0,
    bandwidth=None,
    omega=0.0,
    init=None,
    method="L-BFGS-B",
    chunk_shape=None,
    **kw,
):
    """
    Computes the softdtw barycenter of a set of signals.
    :param X an array of shape (nx, lx, d) where nx is the number of signals,
    lx their length and d the number of dimensions
    :param gamma the softmax temperature
    :param omega the amercing penalty
    :param bandwidth the Sakoe-Chiba bandwidth, use -1 for no bandwidth
    etc
    """

    nx, lx, d = X.shape

    bary_shape = (1, lx, d)
    barycenter = np.zeros(bary_shape, dtype=X.dtype) if init is None else init

    itnb = 0
    starttime = time.time()
    avgdur = 0

    def cost_jac_fun(Z):
        nonlocal itnb

        Z = Z.reshape(bary_shape)
        R_chunks, distances = softdtw_dist(
            Z, X, gamma, bandwidth, omega, chunk_shape, True
        )
        grad_z = softdtw_grad(Z, X, R_chunks, gamma, chunk_shape)

        itnb += 1
        avgdur = (time.time() - starttime) / itnb
        cost = np.mean(distances)
        # print(
        #     f"\rEvaluation {itnb} avg duration {avgdur:.2f}s, avg cost {cost:.5E}"
        #     + " " * 10,
        #     flush=True,
        #     end="",
        # )
        return np.mean(distances), grad_z.ravel()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NumbaPerformanceWarning)
        result = minimize(
            cost_jac_fun,
            barycenter.ravel(),
            method=method,
            jac=True,
            options=kw,
        )
    # print("\r")
    # print(result)
    return result.x.reshape(bary_shape)
