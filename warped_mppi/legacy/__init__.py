"""
Legacy PyCUDA implementation of MPPI.

This module contains the original PyCUDA-based MPPI implementation.
Import only when explicitly needed to avoid CUDA context conflicts with Warp.
"""

from .mppi_pycuda import MPPI as PyCudaMPPI

__all__ = ["PyCudaMPPI"]
