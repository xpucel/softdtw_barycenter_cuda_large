#!/usr/bin/env python

"""
GPU and CPU functions to compute the SoftDTW alignment matrix
"""

from collections import deque
import math
import time
import itertools
import tempfile
import warnings

import numpy as np
from numba import cuda, njit
from numba.core.errors import NumbaPerformanceWarning

import tslearn.metrics

from cuda_softdtw.main._utils import squared_euclidean_distance, softmin


__author__ = "Xavier Pucel"
__copyright__ = "Copyright 2023"
__license__ = "Apache-2.0"
__version__ = "1.0"
__maintainer__ = "Xavier Pucel"
__email__ = "firstname.lastname@onera.fr"
__status__ = "Prototype"


@cuda.jit
def _init_linear_buffers(buf_x, buf_y):
    """Initializes the row and column buffers

    :param buf_x A device array of shape (nx, ny, tx)
    :param buf_y A device array of shape (nx, ny, ty)

    This method should be called with 2 dimensional grid and 1 dimensional block.
    """

    bxini = cuda.blockIdx.x
    bxinc = cuda.gridDim.x
    byini = cuda.blockIdx.y
    byinc = cuda.gridDim.y

    tini = cuda.threadIdx.x
    tinc = cuda.blockDim.x

    for bx in range(bxini, buf_x.shape[0], bxinc):
        for by in range(byini, buf_x.shape[1], byinc):
            for t in range(tini, buf_x.shape[2], tinc):
                buf_x[bx, by, t] = np.inf

    for bx in range(bxini, buf_y.shape[0], bxinc):
        for by in range(byini, buf_y.shape[1], byinc):
            for t in range(tini, buf_y.shape[2], tinc):
                buf_y[bx, by, t] = np.inf


@cuda.jit
def _init_diag_buffer(buf_d):
    """Initializes the diag buffer.

    :param buf_d A device array of shape
    (ceil(tx / SDTW_X_STRIDE), ceil(ty / SDTW_Y_STRIDE))

    This kernel should be called with 2 dimensional grid and 2 dimensional block.
    """

    bxini = cuda.blockIdx.x
    bxinc = cuda.gridDim.x
    byini = cuda.blockIdx.y
    byinc = cuda.gridDim.y

    dxini = cuda.threadIdx.x
    dyini = cuda.threadIdx.y
    dxinc = cuda.blockDim.x
    dyinc = cuda.blockDim.y

    for bx in range(bxini, buf_d.shape[0], bxinc):
        for by in range(byini, buf_d.shape[1], byinc):
            for dx in range(dxini, buf_d.shape[2], dxinc):
                for dy in range(dyini, buf_d.shape[3], dyinc):
                    buf_d[bx, by, dx, dy] = 0 if dx == 0 and dy == 0 else np.inf


_KERNEL_SHARED_ARRAY_SIZE = cuda.get_current_device().MAX_THREADS_PER_BLOCK


@cuda.jit
def _kernel(x, y, gamma, ninv_gamma, omega, n_passes, buf_x, buf_y, buf_d, R):
    """
    Computes part of the soft dtw
    :param x, y Entry signals of shape (nx, len_x, d), and (ny, len_y, d)
    :param gamma The DTW softness parameter (used in softmin)
    :param inv_gamma -1.0 / gamma
    :param omega The amercing penalty
    :param n_passes The number of anti_diagonals
    :param R The cost matrix of shape (nx, ny, len_x, len_y)

    Call with SDTW_GRID_SIZE blocks of SDTW_X_STRIDE threads.
    """

    nx, tx, d = x.shape
    ny, ty, _ = y.shape

    i = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    diag = cuda.shared.array(shape=(_KERNEL_SHARED_ARRAY_SIZE, 2), dtype=R.dtype)

    if bx < nx and by < ny and i < tx:
        jmax = ty
    else:
        jmax = 0

    diag[i, 1] = buf_x[bx, by, i]

    cuda.syncthreads()

    for itnb in range(n_passes):
        j = itnb - i

        if not 0 <= j < jmax:
            rvalue = diag[i, 1]
        else:
            rj = diag[i, 1]
            if i == 0:
                ri = buf_y[bx, by, j]
                if j == 0:
                    rij = buf_d[bx, by, 0, 0]
                else:
                    rij = buf_y[bx, by, j - 1]
            else:
                rij = diag[i - 1, 0]
                ri = diag[i - 1, 1]
            dval = squared_euclidean_distance(x[bx, i], y[by, j])
            smin = softmin(rij, ri + omega, rj + omega, gamma, ninv_gamma)
            rvalue = dval + smin
            R[bx, by, i, j] = rvalue

        cuda.syncthreads()
        diag[i, 0] = diag[i, 1]
        diag[i, 1] = rvalue
        cuda.syncthreads()
    # for itnb

    # update buffers
    for t in range(cuda.threadIdx.x, buf_y.shape[2], cuda.blockDim.x):
        buf_y[bx, by, t] = R[bx, by, -1, t]
    for t in range(cuda.threadIdx.x, buf_x.shape[2], cuda.blockDim.x):
        buf_x[bx, by, t] = R[bx, by, t, -1]
    if i == 0 and buf_d.shape[2] > 1 and buf_d.shape[3] > 1:
        buf_d[bx, by, 1, 1] = R[bx, by, -1, -1]


def softdtw_dist(
    x, y, gamma=1, bandwidth=None, omega=0, chunk_shape=None, return_R=False
):
    """Computes the soft_dtw on the GPU
    :param x, y Entry signals of shape (nx, len_x, d), and (ny, len_y, d)
    :param gamma The DTW softness parameter (used in softmin)
    :param bandwidth The Sakoe-Chiba bandwidth, int (time units) or float
      (in ratio of longest signal)
    :param omega The amercing penalty
    :param chunk_shape a tuple (nx, ny, tx, ty) where
      nx (resp ny) is the number of signals from x (resp y) sent at each gpu computation
      tx (resp ty) is the number of timesteps from x (resp y) sent at each gpu computation
      NOTE : tx is also the number of gpu threads. It must be less than
      cuda.get_current_device().MAX_THREADS_PER_BLOCK
    """

    nx, tx, d = x.shape
    ny, ty, _ = y.shape

    if bandwidth is None:
        bandwidth = np.inf
    elif 0 <= bandwidth < 1:
        bandwidth = int(bandwidth * max(tx, ty))

    # print("bandwidth = ", bandwidth)

    gpu = cuda.get_current_device()

    distances = np.zeros((nx, ny), dtype=x.dtype)

    device_array_type = type(cuda.device_array(()))
    dX = x if type(x) is device_array_type else cuda.to_device(x)
    dY = y if type(y) is device_array_type else cuda.to_device(y)

    if chunk_shape is None:
        chunk_shape = SDTW_GRID_SIZE + (SDTW_X_STRIDE, SDTW_Y_STRIDE)

    grid_dim = (min(chunk_shape[0], nx), min(chunk_shape[1], ny))
    block_dim = (
        min(chunk_shape[2], tx, gpu.MAX_THREADS_PER_BLOCK),
        min(chunk_shape[3], ty),
    )

    chunk_shape = grid_dim + block_dim

    dRker = cuda.device_array(chunk_shape, dtype=x.dtype)
    dRcpy = cuda.device_array(chunk_shape, dtype=x.dtype)

    nb_chunks = (
        math.ceil(x.shape[0] / chunk_shape[0])
        * math.ceil(y.shape[0] / chunk_shape[1])
        * math.ceil(x.shape[1] / chunk_shape[2])
        * math.ceil(y.shape[1] / chunk_shape[3])
    )

    R_chunks = []
    dbuf_x = cuda.device_array((nx, ny, tx), dtype=x.dtype)
    dbuf_y = cuda.device_array((nx, ny, ty), dtype=x.dtype)
    dbuf_d_shape = (
        nx,
        ny,
        math.ceil(tx / block_dim[0]),
        math.ceil(ty / block_dim[1]),
    )
    dbuf_d = cuda.device_array(dbuf_d_shape, dtype=x.dtype)

    nxinc, nyinc = grid_dim
    txinc, tyinc = block_dim

    work_list = deque()

    skipped_chunks = 0
    for txmin, tymin, nxmin, nymin in itertools.product(
        range(0, tx, txinc),
        range(0, ty, tyinc),
        range(0, nx, nxinc),
        range(0, ny, nyinc),
    ):
        dnx = min(x.shape[0] - nxmin, nxinc)
        dny = min(y.shape[0] - nymin, nyinc)
        dtx = min(x.shape[1] - txmin, txinc)
        dty = min(y.shape[1] - tymin, tyinc)
        antidiag_min = txmin - tymin - dty
        antidiag_max = txmin + dtx - tymin
        if antidiag_min > bandwidth / 2 or antidiag_max < -bandwidth / 2:
            skipped_chunks += 1
            continue
        work_list.append((nxmin, dnx, nymin, dny, txmin, dtx, tymin, dty))

    ker_stream = cuda.stream()
    cpy_stream = cuda.stream()
    cpy_chunk = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NumbaPerformanceWarning)

        nb_threads = min(max(*block_dim), gpu.MAX_THREADS_PER_BLOCK)
        _init_linear_buffers[grid_dim, nb_threads](dbuf_x, dbuf_y)

        nb_threads = (
            min(dbuf_d_shape[2], math.floor(math.sqrt(gpu.MAX_THREADS_PER_BLOCK))),
            min(dbuf_d_shape[3], math.floor(math.sqrt(gpu.MAX_THREADS_PER_BLOCK))),
        )
        _init_diag_buffer[grid_dim, nb_threads](dbuf_d)

        while True:
            ker_stream.synchronize()
            cpy_stream.synchronize()

            if work_list:
                # Load the first block
                ker_chunk = work_list.popleft()
                nxmin, dnx, nymin, dny, txmin, dtx, tymin, dty = ker_chunk
                n_passes = dtx + dty - 1

                snx = slice(nxmin, nxmin + dnx)
                sny = slice(nymin, nymin + dny)
                stx = slice(txmin, txmin + dtx)
                sty = slice(tymin, tymin + dty)

                # Compute block on GPU
                _kernel[(dnx, dny), dtx, ker_stream](
                    dX[snx, stx, :],
                    dY[sny, sty, :],
                    gamma,
                    -1.0 / gamma,
                    omega,
                    n_passes,
                    dbuf_x[snx, sny, stx],
                    dbuf_y[snx, sny, sty],
                    dbuf_d[
                        snx,
                        sny,
                        (txmin // txinc) :,
                        (tymin // tyinc) :,
                    ],
                    dRker[:dnx, :dny, :dtx, :dty],
                )

            if cpy_chunk:
                # Copy previous block to host
                nxmin, dnx, nymin, dny, txmin, dtx, tymin, dty = cpy_chunk

                hRcpy = dRcpy.copy_to_host(stream=cpy_stream)
                R_chunks.append((cpy_chunk, hRcpy))

                if txmin + dtx == tx and tymin + dty == ty:
                    # Get the DTW distance
                    snx = slice(nxmin, nxmin + dnx)
                    sny = slice(nymin, nymin + dny)
                    distances[snx, sny] = hRcpy[:dnx, :dny, dtx - 1, dty - 1]

            # Swap dRker and dRcpy
            dRker, dRcpy, cpy_chunk, ker_chunk = dRcpy, dRker, ker_chunk, None

            if cpy_chunk is None:
                break

        # while True
    # with catch warnings

    if return_R:
        return R_chunks, distances
    else:
        return distances


def merge_chunks(x, y, R_chunks):
    """Build R from the chunks returned by softdtw"""
    nx, tx, _ = x.shape
    ny, ty, _ = y.shape
    R = np.full((nx, ny, tx, ty), np.inf, dtype=x.dtype)
    for coords, contents in R_chunks:
        nxmin, dnx, nymin, dny, txmin, dtx, tymin, dty = coords
        nxmax = nxmin + dnx
        nymax = nymin + dny
        txmax = txmin + dtx
        tymax = tymin + dty
        R[nxmin:nxmax, nymin:nymax, txmin:txmax, tymin:tymax] = contents[
            :dnx, :dny, :dtx, :dty
        ]
    return R
