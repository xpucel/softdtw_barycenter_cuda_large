# CUDA Soft-DTW barycenter for large signals

Library for computing soft DTW barycenters on large signals using cuda GPUs.

This algorithm implements the barycenter part of the WETSAND approach described in the paper [Warped Time Series Anomaly Detection](https://arxiv.org/abs/2404.12134)

## Dependencies

1. Python 3.10+
2. CUDA Enabled GPU
3. [Cuda toolkit](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
4. [Numba Cuda](https://numba.pydata.org/numba-doc/dev/cuda/index.html)
5. [TSLearn library](https://tslearn.readthedocs.io/en/stable/) for running the examples only

And the usual pip packages : numpy, matplotlib, etc.
