"""
Warped MPPI: High-Performance Model Predictive Path Integral Control

A GPU-accelerated implementation of MPPI control using NVIDIA Warp for
autonomous vehicle navigation and obstacle avoidance.
"""

"""
Warped MPPI: High-Performance Model Predictive Path Integral Control

A GPU-accelerated implementation of MPPI control using NVIDIA Warp for
autonomous vehicle navigation and obstacle avoidance.
"""

"""
Warped MPPI: High-Performance Model Predictive Path Integral Control

A GPU-accelerated implementation of MPPI control using NVIDIA Warp for
autonomous vehicle navigation and obstacle avoidance.
"""

# Primary Warp-based implementation
try:
    from .mppi_warp import WarpMPPI, MPPI

    WARP_AVAILABLE = True
except ImportError as e:
    MPPI = None
    WARP_AVAILABLE = False
    raise ImportError(f"Warp not available - cannot import MPPI controller: {e}")

# Legacy PyCUDA implementation available on explicit import
# from warped_mppi.legacy import PyCudaMPPI

__version__ = "0.1.0"
__author__ = "Barry Gilhuly"
__email__ = "barry.gilhuly@uwaterloo.ca"

__all__ = ["MPPI", "WarpMPPI"]
