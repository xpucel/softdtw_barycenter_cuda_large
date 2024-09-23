#!/usr/bin/env python

"""
Reference functions for testing the CUDA implementation
"""

import itertools
import math
import time
import zipfile

import numpy as np

from numba import njit

# def time_fun(fun, *args, **kw):
#     """
#     Measures the time used to calls fun with args *args and **kw,
#     Returns a pair (time, result)
#     """
#     start = time.time()
#     result = fun(*args, **kw)
#     end = time.time()
#     return (end - start), result


def make_xy(nbx, nby, lenx, leny, d):
    """
    Creates a pair of time series with the given number of signals and lengths
    """
    x = np.linspace(10, 20, nbx * lenx * d, dtype=np.float64).reshape((nbx, lenx, d))
    y = np.linspace(15, 25, nby * leny * d, dtype=np.float64).reshape((nby, leny, d))
    fun = lambda z: np.cos(z * np.cos(z))
    x = fun(x)
    y = fun(y)
    return x, y



@njit
def distance_matrix(x, y):
    """CPU-based distance matrix

    :param x, y Matrices of shapes ``(nx, lx, d)`` and ``(ny, ly, d)``
    :return A matrix of shape ``(nx, ny, lx, ly)`` with the squared
    euclidean distance on the last dimension
    """
    d = x[:, None, :, None, :] - y[None, :, None, :, :]
    return np.sum(d**2, axis=4)


@njit
def softmin(a, b, c, gamma=1):
    """
    Computes the softmin of 3 variables
    """
    gamma_inv = -1.0 / gamma
    ra = -a / gamma
    rb = -b / gamma
    rc = -c / gamma
    rmax = max(ra, rb, rc)
    ea = 1 if ra == rmax else math.exp(ra - rmax)
    eb = 1 if rb == rmax else math.exp(rb - rmax)
    ec = 1 if rc == rmax else math.exp(rc - rmax)
    rsum = ea + eb + ec
    softmin = -gamma * (rmax + math.log(rsum))
    return softmin

try :
    import tslearn

    def cpu_softdtw_dist(D, gamma=1):
        """Calls the tslearn Softdtw implementation"""
        nx, ny, lx, ly = D.shape
        R = np.full((nx, ny, lx + 2, ly + 2), np.inf, dtype=D.dtype)
        for inx, iny in itertools.product(range(nx), range(ny)):
            tslearn.metrics.soft_dtw_fast._soft_dtw(D[inx, iny], R[inx, iny], gamma)
        return R[:, :, 1:-1, 1:-1], R[:, :, -1, -1]


    def cpu_softdtw_grad(x, y, gamma=1):
        """Calls the tslearn implementation"""

        gx = np.zeros_like(x)

        nx, lx, d = x.shape
        ny, ly, _ = y.shape

        D = np.empty((nx, ny, lx, ly), dtype=x.dtype)
        E = np.empty_like(D)

        for inx, iny in itertools.product(range(nx), range(ny)):
            dist = tslearn.metrics.SquaredEuclidean(x[inx], y[iny])
            D[inx, iny, :, :] = dist.compute()
            sdtw = tslearn.metrics.SoftDTW(dist, gamma=gamma)
            value = sdtw.compute()
            E[inx, iny] = sdtw.grad()
            gx[inx] += dist.jacobian_product(E[inx, iny])

        return E, gx

except ImportError:

    @njit
    def cpu_softdtw_dist(x, y, D=None, gamma=1, bandwidth=-1, omega=0):
        """
        A Direct implementation of the SoftDTW algorithm
        Returns a pair (R, dist) where R is the cumulative cost matrix
        and dist the distance.
        """
        nx, lx, d = x.shape
        ny, ly, _ = y.shape

        if D is None:
            D = distance_matrix_cpu(x, y)

        if D.shape != (nx, ny, lx, ly):
            raise ValueError(f"Bad D shape, expected {(nx, ny, lx, ly)} got {D.shape}")

        R = np.full((nx, ny, lx + 1, ly + 1), np.inf, dtype=D.dtype)
        R[:, :, 0, 0] = 0

        for inx in range(nx):
            for iny in range(ny):
                for i in range(lx):
                    for j in range(ly):
                        if 0 < bandwidth < abs(i - j):
                            continue
                        s0 = R[inx, iny, i, j]
                        s1 = R[inx, iny, i + 1, j] + omega
                        s2 = R[inx, iny, i, j + 1] + omega
                        sm = softmin(s0, s1, s2)
                        result = D[inx, iny, i, j] + sm
                        R[inx, iny, i + 1, j + 1] = result

        return R, R[:, :, -1, -1]


    @njit
    def cpu_softdtw_grad(x, y, D=None, R=None, gamma=1, bandwidth=-1, omega=0):
        """
        CPU based implementation of the SoftDTW gradient algorithm.
        Returns a tuple (E, gx, gy) where:
          - E is the gradient matrix
          - gx is the gradient of the SoftDTW distance wrt. x
          - gy is the gradient of the SoftDTW distance wrt. y
        """

        nx, lx, d = x.shape
        ny, ly, _ = y.shape

        if D is None:
            D = distance_matrix_cpu(x, y)

        if D.shape != (nx, ny, lx, ly):
            raise ValueError(f"Bad D shape, expected {(nx, ny, lx, ly)} got {D.shape}")

        if R is None:
            R, _ = softdtw_cpu(x, y, D)

        if R.shape != (nx, ny, lx + 1, ly + 1):
            raise ValueError(f"Bad R shape, expected {(nx, ny, lx+1, ly+1)}, got {R.shape}")

        E = np.zeros(R.shape, dtype=R.dtype)
        E[:, :, -1, -1] = np.array([1])

        R[:, :, :-1, :-1] = R[:, :, 1:, 1:]
        R[:, :, -1, :] = -np.inf
        R[:, :, :, -1] = -np.inf
        R[:, :, -1, -1] = R[:, :, -2, -2]

        gradx = np.zeros_like(x)
        grady = np.zeros_like(y)

        for inx in range(nx):
            for iny in range(ny):
                for j in range(ly - 1, -1, -1):
                    for i in range(lx - 1, -1, -1):
                        if 0 < bandwidth < abs(i - j):
                            continue

                        result = 0

                        ei = E[inx, iny, i + 1, j]
                        ej = E[inx, iny, i, j + 1]
                        eij = E[inx, iny, i + 1, j + 1]
                        r = R[inx, iny, i, j]

                        if ei != 0:
                            ri = R[inx, iny, i + 1, j]
                            di = D[inx, iny, i + 1, j] if i + 1 < D.shape[2] else 0
                            result += ei * math.exp((ri - r - di) / gamma)

                        if ej != 0:
                            rj = R[inx, iny, i, j + 1]
                            dj = D[inx, iny, i, j + 1] if j + 1 < D.shape[3] else 0
                            result += ej * math.exp((rj - r - dj) / gamma)

                        if eij != 0:
                            rij = R[inx, iny, i + 1, j + 1]
                            dij = (
                                D[inx, iny, i + 1, j + 1]
                                if i + 1 < D.shape[2] and j + 1 < D.shape[3]
                                else 0
                            )
                            result += eij * math.exp((rij - r - dij) / gamma)

                        E[inx, iny, i, j] = result

                        for k in range(d):
                            grad_inc = result * 2 * (x[inx, i, k] - y[iny, j, k])
                            gradx[inx, i, k] += grad_inc
                            grady[iny, j, k] -= grad_inc

        return E[:, :, :-1, :-1], gradx, grady



