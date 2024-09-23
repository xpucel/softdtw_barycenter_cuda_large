#!/usr/bin/env python

"""
The cuda_softdtw package offers methods for computing SoftDTW barycenters
using the CUDA library
"""

from cuda_softdtw.main.softdtw_dist import softdtw_dist
from cuda_softdtw.main.softdtw_grad import softdtw_grad
from cuda_softdtw.main.barycenter import softdtw_barycenter

__author__ = "Xavier Pucel"
__copyright__ = "Copyright 2023"
__license__ = "Apache-2.0"
__version__ = "1.0"
__maintainer__ = "Xavier Pucel"
__email__ = "firstname.lastname@onera.fr"
__status__ = "Prototype"

__all__ = ["softdtw_dist", "softdtw_grad", "softdtw_barycenter"]
