"""
Warped MPPI: High-Performance Model Predictive Path Integral Control

A GPU-accelerated implementation of MPPI control using NVIDIA Warp for
autonomous vehicle navigation and obstacle avoidance.
"""

from .mppi_pycuda import MPPI

# Try to import Warp implementation if available
try:
    from .mppi_warp import WarpMPPI

    WARP_AVAILABLE = True
except ImportError:
    WarpMPPI = None
    WARP_AVAILABLE = False

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = ["MPPI"]

# Add WarpMPPI to exports if available
if WARP_AVAILABLE:
    __all__.append("WarpMPPI")
