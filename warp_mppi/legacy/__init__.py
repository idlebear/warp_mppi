"""
Legacy PyCUDA implementation of MPPI.

This module contains the original PyCUDA-based MPPI implementation.
Import only when explicitly needed to avoid CUDA context conflicts with Warp.
"""

try:
    from .mppi_pycuda import MPPI as PyCudaMPPI
except ImportError:
    PyCudaMPPI = None

try:
    from .trajectory_eval_pycuda import evaluate_trajectories_by_entropy_gpu
except ImportError:
    evaluate_trajectories_by_entropy_gpu = None

from .discrete_oce_pycuda import (
    DISCRETE_OCE_SCORING_MODES,
    evaluate_discrete_oce_gpu,
    evaluate_discrete_oce_rollouts_gpu,
    normalize_discrete_oce_scoring_mode,
    score_discrete_oce_components,
)

__all__ = [
    "PyCudaMPPI",
    "evaluate_trajectories_by_entropy_gpu",
    "evaluate_discrete_oce_gpu",
    "evaluate_discrete_oce_rollouts_gpu",
    "DISCRETE_OCE_SCORING_MODES",
    "normalize_discrete_oce_scoring_mode",
    "score_discrete_oce_components",
]
