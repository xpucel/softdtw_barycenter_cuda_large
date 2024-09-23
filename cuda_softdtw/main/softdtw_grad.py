#!/usr/bin/env python

"""
CUDA Softdtw gradient algirithm
"""

import os
import math
import time
import itertools
import warnings

import numpy as np
from numba import cuda, jit, njit, prange

from cuda_softdtw.main._utils import squared_euclidean_distance


__author__ = "Xavier Pucel"
__copyright__ = "Copyright 2023"
__license__ = "Apache-2.0"
__version__ = "1.0"
__maintainer__ = "Xavier Pucel"
__email__ = "firstname.lastname@onera.fr"
__status__ = "Prototype"


@cuda.jit
def _init_gradients(xgrad):
    """Set all values to zero
    :param xgrad vector of shape (nx, tx, d)
    Call with 1 dimension grid and 1 dimension block
    """
    nini = cuda.blockIdx.x
    ninc = cuda.gridDim.x
    tini = cuda.threadIdx.x
    tinc = cuda.blockDim.x

    nx, tx, d = xgrad.shape
    for n in range(nini, nx, ninc):
        for t in range(tini, tx, tinc):
            for k in range(d):
                xgrad[n, t, k] = 0


@cuda.jit
def _init_linear_buffers(buf_x, buf_y):
    """Initializes the row and column buffers

    :param buf_x A device array of shape (nx, ny, tx, 3)
    :param buf_y A device array of shape (nx, ny, ty, 3)

    Call with 2 dimensions grid and 1 dimension block
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
                buf_x[bx, by, t, 0] = 0
                buf_x[bx, by, t, 1] = -np.inf
                buf_x[bx, by, t, 2] = 0

    for bx in range(bxini, buf_y.shape[0], bxinc):
        for by in range(byini, buf_y.shape[1], byinc):
            for t in range(tini, buf_y.shape[2], tinc):
                buf_y[bx, by, t, 0] = 0
                buf_y[bx, by, t, 1] = -np.inf
                buf_y[bx, by, t, 2] = 0


@cuda.jit
def _init_diag_buffer(buf_d):
    """Initializes the diag buffer.

    :param buf_d A device array of shape
    (nx, ny, ceil(tx / SDTW_X_STRIDE), ceil(ty / SDTW_Y_STRIDE), 3)

    Call with 2 dimensions grid and 2 dimension block
    """

    bxini = cuda.blockIdx.x
    bxinc = cuda.gridDim.x
    byini = cuda.blockIdx.y
    byinc = cuda.gridDim.y

    dxini = cuda.threadIdx.x
    dyini = cuda.threadIdx.y
    dxinc = cuda.blockDim.x
    dyinc = cuda.blockDim.y

    lastdx = buf_d.shape[2] - 1
    lastdy = buf_d.shape[3] - 1

    for bx in range(bxini, buf_d.shape[0], bxinc):
        for by in range(byini, buf_d.shape[1], byinc):
            for dx in range(dxini, buf_d.shape[2], dxinc):
                for dy in range(dyini, buf_d.shape[3], dyinc):
                    buf_d[bx, by, dx, dy, 0] = 0
                    buf_d[bx, by, dx, dy, 1] = -np.inf
                    buf_d[bx, by, dx, dy, 2] = 0


MAX_NB_THREADS = cuda.get_current_device().MAX_THREADS_PER_BLOCK


@cuda.jit
def _kernel_compute_E(x, y, gamma, n_passes, botright, buf_x, buf_y, buf_d, R, E):
    """
    Computes the SDTW gradient matrix
    :param x a vector of shape (nx, tx, d)
    :param y a vector of shape (ny, ty, d)
    :param gamma the softness parameter
    :param n_passes the number of antidiagonals
    :param botright a boolean indicating whether we are at the bottom right of R and E
    :param buf_x a buffer of shape (nx, ny, tx, 2)
    :param buf_y a buffer of shape (ny, ny, ty, 2)
    :param buf_d a buffer of shape (1+, 1+, 2)
    :param R a chunk of the intermediate cost matrix,
    of shape (nx, ny, tx, ty) as computed by compute_softdtw_host
    :param E the gradient matrix, of shape (nx, ny, tx, ty)
    """

    nx, tx, d = x.shape
    ny, ty, _ = y.shape

    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    i = cuda.threadIdx.x

    diag = cuda.shared.array(shape=(MAX_NB_THREADS, 2, 2), dtype=R.dtype)

    if bx < nx and by < ny and i < tx:
        jmax = ty
    else:
        jmax = 0

    diag[i, 1, 0] = buf_x[bx, by, i, 0]
    diag[i, 1, 1] = buf_x[bx, by, i, 2]

    cuda.syncthreads()

    for itnb in range(n_passes):
        j = ty + tx - i - itnb - 2

        dvalue = squared_euclidean_distance(x[bx, i], y[by, j])
        rvalue = R[bx, by, i, j]

        if not 0 <= j < jmax:
            evalue = diag[i, 1, 1]
        else:
            dj = diag[i, 1, 0]
            if j == ty - 1:
                rj = buf_x[bx, by, i, 1]
            else:
                rj = R[bx, by, i, j + 1]
            ej = diag[i, 1, 1]

            if i == tx - 1:
                di = buf_y[bx, by, j, 0]
                ri = buf_y[bx, by, j, 1]
                ei = buf_y[bx, by, j, 2]
                if j == ty - 1:
                    if botright:
                        dij = 0
                        rij = rvalue
                        eij = 1
                    else:
                        dij = buf_d[bx, by, -1, -1, 0]
                        rij = buf_d[bx, by, -1, -1, 1]
                        eij = buf_d[bx, by, -1, -1, 2]
                else:
                    dij = buf_y[bx, by, j + 1, 0]
                    rij = buf_y[bx, by, j + 1, 1]
                    eij = buf_y[bx, by, j + 1, 2]
            else:
                di = diag[i + 1, 1, 0]
                ri = R[bx, by, i + 1, j]
                ei = diag[i + 1, 1, 1]
                dij = diag[i + 1, 0, 0]
                if j == ty - 1:
                    rij = buf_x[bx, by, i + 1, 1]
                else:
                    rij = R[bx, by, i + 1, j + 1]
                eij = diag[i + 1, 0, 1]

            evalue = 0
            if ei != 0:
                evalue += ei * math.exp((ri - rvalue - di) / gamma)
            if ej != 0:
                evalue += ej * math.exp((rj - rvalue - dj) / gamma)
            if eij != 0:
                evalue += eij * math.exp((rij - rvalue - dij) / gamma)
            # if by == 0:
            # print(bx, by, i, j, ei, ej, eij, rij, rvalue, dij, evalue)
            E[bx, by, i, j] = evalue

            # udpate buffers
            if i == 0:
                buf_y[bx, by, j, 0] = dvalue
                buf_y[bx, by, j, 1] = rvalue
                buf_y[bx, by, j, 2] = evalue

            if j == 0:
                buf_x[bx, by, i, 0] = dvalue
                buf_x[bx, by, i, 1] = rvalue
                buf_x[bx, by, i, 2] = evalue

            if i == 0 and j == 0 and buf_d.shape[2] > 1 and buf_d.shape[3] > 1:
                buf_d[bx, by, -2, -2, 0] = dvalue
                buf_d[bx, by, -2, -2, 1] = rvalue
                buf_d[bx, by, -2, -2, 2] = evalue

        cuda.syncthreads()
        diag[i, 0, 0] = diag[i, 1, 0]
        diag[i, 0, 1] = diag[i, 1, 1]
        diag[i, 1, 0] = dvalue
        diag[i, 1, 1] = evalue
        cuda.syncthreads()
    # for itnb in range(n_passes):


@cuda.jit
def _kernel_compute_gradx(x, y, E, gradx):
    """
    Computes the jacobian product of E with the derivative of the distance matrix
    """
    nx, tx, d = x.shape
    ny, ty, _ = y.shape

    bx = cuda.blockIdx.x
    i = cuda.blockIdx.y
    k = cuda.blockIdx.z

    grad = 0.0
    for by in range(ny):
        for j in range(ty):
            grad += E[bx, by, i, j] * 2 * (x[bx, i, k] - y[by, j, k])
    gradx[bx, i, k] += grad


def softdtw_grad(x, y, R_chunks, gamma=1, chunk_shape=None, return_E=False):
    """
    Computes the SoftDTW gradient multiplied by the jacobian of the distance matrix
    """

    nx, tx, d = x.shape
    ny, ty, _ = y.shape

    gpu = cuda.get_current_device()

    device_array_type = type(cuda.device_array(()))
    dX = x if type(x) is device_array_type else cuda.to_device(x)
    dY = y if type(y) is device_array_type else cuda.to_device(y)

    dgradX = cuda.device_array_like(dX)

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

    dEker = cuda.device_array(chunk_shape, dtype=x.dtype)
    if return_E:
        E = np.zeros((nx, ny, tx, ty), dtype=x.dtype)

    buf_x = cuda.device_array((nx, ny, tx, 3), dtype=x.dtype)
    buf_y = cuda.device_array((nx, ny, ty, 3), dtype=x.dtype)
    buf_d_shape = (
        nx,
        ny,
        math.ceil(tx / block_dim[0]),
        math.ceil(ty / block_dim[1]),
        3,
    )
    buf_d = cuda.device_array(buf_d_shape, dtype=x.dtype)

    ker_stream = cuda.stream()
    cpy_stream = cuda.stream()
    prev_todo = None

    # init gradients
    nb_blocks = min(gpu.MAX_GRID_DIM_X, nx, ny)
    nb_threads = min(gpu.MAX_THREADS_PER_BLOCK, tx, ty)
    _init_gradients[nb_blocks, nb_threads](dgradX)

    # init buffers
    nb_threads = min(gpu.MAX_THREADS_PER_BLOCK, tx, ty)
    _init_linear_buffers[grid_dim, nb_threads](buf_x, buf_y)
    nb_threads = (
        min(buf_d_shape[2], math.floor(math.sqrt(gpu.MAX_THREADS_PER_BLOCK))),
        min(buf_d_shape[2], math.floor(math.sqrt(gpu.MAX_THREADS_PER_BLOCK))),
    )
    _init_diag_buffer[grid_dim, nb_threads](buf_d)

    while True:
        # print("\rSoftDTW gradient :", len(R_chunks), "chunks remaining", end="")
        # print("SoftDTW gradient :", len(R_chunks), "chunks remaining")

        ker_stream.synchronize()
        cpy_stream.synchronize()

        if R_chunks:
            todo = R_chunks.pop()

            (nxmin, dnx, nymin, dny, txmin, dtx, tymin, dty), chunk = todo
            dRcpy.copy_to_device(chunk, stream=cpy_stream)

        if prev_todo:
            (nxmin, dnx, nymin, dny, txmin, dtx, tymin, dty), chunk = prev_todo

            # print(nxmin, dnx, nymin, dny, txmin, dtx, tymin, dty, flush=True)
            # print(chunk)

            snx = slice(nxmin, nxmin + dnx)
            sny = slice(nymin, nymin + dny)
            stx = slice(txmin, txmin + dtx)
            sty = slice(tymin, tymin + dty)
            n_passes = dtx + dty - 1
            botright = tx == stx.stop and ty == sty.stop

            _kernel_compute_E[(dnx, dny), dtx, ker_stream](
                dX[snx, stx],
                dY[sny, sty],
                gamma,
                n_passes,
                botright,
                buf_x[snx, sny, stx],
                buf_y[snx, sny, sty],
                buf_d[
                    snx,
                    sny,
                    : (txmin // block_dim[0]) + 1,
                    : (tymin // block_dim[1]) + 1,
                ],
                dRker,
                dEker,
            )

            _kernel_compute_gradx[(dnx, dtx, d), 1, ker_stream](
                dX[snx, stx], dY[sny, sty], dEker, dgradX[snx, stx, :]
            )

            if return_E:
                ker_stream.synchronize()
                cpy_stream.synchronize()
                if (dnx, dny) == grid_dim and (stx, sty) == block_dim:
                    E[snx, sny, stx, sty] = dEker.copy_to_host()
                else:
                    tmp = dEker[:dnx, :dny, :dtx, :dty].copy_to_host()
                    E[snx, sny, stx, sty] = tmp

        # swap dRker and dRcpy, as well as todo and prev_todo
        dRker, dRcpy, prev_todo, todo = dRcpy, dRker, todo, None

        if not R_chunks and not prev_todo:
            break

    # while True

    ker_stream.synchronize()
    cpy_stream.synchronize()
    gradx = dgradX.copy_to_host()

    if return_E:
        return E, gradx
    else:
        return gradx
