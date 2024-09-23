"""
Test the CUDA barycenter on robot data
"""

import time
import numpy as np
import tslearn
import matplotlib.pyplot as plt
from cuda_softdtw import softdtw_dist, softdtw_barycenter


def time_fun(fun, *args, **kw):
    """
    Measures the time to compute a function.
    Calls fun passing args and kw as arguments, and returns a pair
    (duration, result) where duration is the function execution time
    in seconds, and result is the return value of the function
    """
    start = time.time()
    res = fun(*args, **kw)
    end = time.time()
    return (end - start, res)


def make_xy(nbx, nby, lenx, leny, d):
    """Creates two sets of time series
    nbx : the number of signals in the X time series
    nby : ---------------------------- Y -----------
    lenx: the length of signals in the X time series
    leny: ---------------------------- Y -----------
    d   : the dimension of the time series
    returns a pair of tensors rx, ry of shapes (nbx, lenx, d) and
    (nby, leny, d).
    """
    x = np.arange(nbx * lenx * d).reshape((nbx, lenx, d))
    xfactor = 0.9 + 0.1 * np.random.rand(*x.shape)
    x = np.cos(np.multiply(xfactor, x + np.random.rand(*x.shape)))

    y = np.arange(nby * leny * d).reshape((nby, leny, d))
    yfactor = 0.9 + 0.1 * np.random.rand(*y.shape)
    y = np.cos(np.multiply(yfactor, y + np.random.rand(*y.shape)))

    return x, y


def test_barycenter():
    """
    Simple test
    """
    nbx = 7
    nby = 0
    lenx = 2**12
    leny = 0
    d = 1

    gamma = 1.0
    bandwidth = 0.1  # np.inf
    chunk_shape = (1, nbx, 2**9, 2**10)
    omega = 0

    print("init")
    X, _ = make_xy(nbx, nby, lenx, leny, d)

    init = None

    print("computing cuda barycenter")
    ctime, bcuda = time_fun(
        softdtw_barycenter,
        X,
        omega=omega,
        init=init,
        # init=X[0],
        bandwidth=bandwidth,
        # bandwidth=10,
        method="L-BFGS-B",
        # method="BFGS",
        # disp=True,
        # maxiter=100,
        maxiter=100,
        maxfun=200,
        gtol=1e-8,
        # xrtol=1e-9
        ftol=1e-3,
        chunk_shape=chunk_shape,
    )
    print(f"barycenter computed in {ctime:.2f}s")

    try:
        print(
            "computing tslearn barycenter (press Ctrl-C once to quit tslearn, twice to"
            " exit)"
        )
        ctime, bref = time_fun(
            tslearn.barycenters.softdtw.softdtw_barycenter, X, init=init
        )
        print(f"barycenter computed in {ctime:.2f}s")
    except KeyboardInterrupt:
        print("tslearn computation interrupted, setting barycenter to zero")
        bref = np.zeros_like(bcuda)

    _, axs = plt.subplots(3, 3)
    for Xsplit, ax in zip(np.split(X, len(axs.flat) - 2), axs.flat):
        for x in Xsplit:
            ax.plot(x.ravel(), "k-", alpha=0.2)

    cuda_mean_dist = np.mean(
        softdtw_dist(bcuda, X, gamma, bandwidth, omega, chunk_shape)
    )
    axs.flat[-2].set_title(f"Cuda barycenter {cuda_mean_dist}")
    axs.flat[-2].plot(bcuda.ravel(), "r-", linewidth=2)

    tslearn_mean_dist = np.mean(
        softdtw_dist(bref.reshape(bcuda.shape), X, gamma, bandwidth, omega, chunk_shape)
    )
    axs.flat[-1].set_title(f"TSLearn barycenter {tslearn_mean_dist}")
    axs.flat[-1].plot(bref.ravel(), "r-", linewidth=2)

    for ax in axs.flat:
        ax.label_outer()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_barycenter()
