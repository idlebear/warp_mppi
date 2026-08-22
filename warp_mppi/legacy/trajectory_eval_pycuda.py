"""PyCUDA OCE trajectory evaluator for the legacy MPPI backend.

This module is vendored from the information-gain OCE implementation and patched
to run under the CUDA context managed by ``mppi_pycuda.py``. The scoring path is
selected through ``entropy_space`` / ``oce_config.oce_entropy_space`` so MPPI can
swap between supported OCE objective variants without changing the controller.
"""

import copy
import atexit
import contextlib
import os
import warnings
from dataclasses import dataclass, field, fields
from typing import Iterable, Optional

import numpy as np

try:
    from numba import njit as _numba_njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _NUMBA_AVAILABLE = False

    def _numba_njit(*args, **kwargs):
        _ = (args, kwargs)

        def _decorator(func):
            return func

        return _decorator


from .planning import trajectory_eval_cpu as _cpu_eval
from .planning.continuous_partition import (
    _parse_kde_entropy_space,
    mode_information_at_time,
)

try:
    import pycuda.driver as _cuda
    from pycuda.compiler import SourceModule as _CudaSourceModule

    _PYCUDA_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent
    _cuda = None
    _CudaSourceModule = None
    _PYCUDA_AVAILABLE = False


EGO_STATE_DIM = 3
TARGET_STATE_DIM = 3
BACKEND_NAME = "cuda"
P4_CUDA_AVAILABLE = bool(_PYCUDA_AVAILABLE)
P2_EXPERIMENTAL_OCC_VIS_CUDA_ENABLED = os.environ.get(
    "OCE_EXPERIMENTAL_CUDA_VISIBILITY", ""
).strip().lower() in {"1", "true", "yes", "on"}

_MODULE_CONTEXT = None
_owns_module_context_ref = False
_module_context_pushed = False


def _establish_module_context():
    """Ensure a CUDA context exists for OCE kernels and device buffers."""

    global _MODULE_CONTEXT, _owns_module_context_ref, _module_context_pushed
    if _MODULE_CONTEXT is not None:
        return _MODULE_CONTEXT
    if not _PYCUDA_AVAILABLE:
        raise RuntimeError("PyCUDA is not available for OCE CUDA execution.")

    _cuda.init()
    device = None
    try:
        current_context = _cuda.Context.get_current()
        if current_context is not None:
            device = _cuda.Context.get_device()
    except _cuda.LogicError:
        pass

    if device is None:
        device = _cuda.Device(0)

    try:
        _MODULE_CONTEXT = device.retain_primary_context()
        _MODULE_CONTEXT.push()
        _owns_module_context_ref = True
    except AttributeError:
        _MODULE_CONTEXT = device.make_context()
        _owns_module_context_ref = True

    _module_context_pushed = True
    return _MODULE_CONTEXT


def _pop_module_context_after_compile():
    global _module_context_pushed
    if _module_context_pushed and _MODULE_CONTEXT is not None:
        _MODULE_CONTEXT.pop()
        _module_context_pushed = False


@contextlib.contextmanager
def _active_cuda_context(context=None):
    """Temporarily make the OCE CUDA context current on this thread."""

    if context is None:
        context = _establish_module_context()
        _pop_module_context_after_compile()
    context.push()
    try:
        yield
    finally:
        context.pop()


def _cleanup_context_atexit():
    global _MODULE_CONTEXT, _owns_module_context_ref, _module_context_pushed
    if _owns_module_context_ref and _MODULE_CONTEXT is not None:
        try:
            if _module_context_pushed:
                _MODULE_CONTEXT.pop()
                _module_context_pushed = False
            _MODULE_CONTEXT.detach()
            _MODULE_CONTEXT = None
            _owns_module_context_ref = False
        except Exception:
            pass


atexit.register(_cleanup_context_atexit)


def _read_nonnegative_env_float(name, default):
    raw = os.environ.get(name, "")
    if raw is None:
        return float(default)
    text = str(raw).strip()
    if not text:
        return float(default)
    try:
        value = float(text)
    except ValueError:
        return float(default)
    if not np.isfinite(value) or value < 0.0:
        return float(default)
    return float(value)


def _read_env_str(name, default):
    raw = os.environ.get(name, "")
    if raw is None:
        return str(default)
    text = str(raw).strip()
    return text if text else str(default)


def _cuda_owner_min_contrib_eps():
    """Minimum dynamic contribution for assigning a cell owner in CUDA occupancy build."""

    return _read_nonnegative_env_float("OCE_CUDA_OWNER_MIN_CONTRIB_EPS", 1e-6)


_P4_CUDA_SOURCE = r"""
extern "C" {
    #include <math.h>

    const int MAX_LOCAL_MODES = 32;
    const int NUM_SIGMA_POINTS = 5;

    const float LOG_2PI_F = 1.8378770664093453f;
    const float LOG_2PIE_F = 2.8378770664093453f;

    __device__ __forceinline__ float clamp_prob(float x, float lo, float hi) {
        if (x < lo) return lo;
        if (x > hi) return hi;
        return x;
    }

    __device__ __forceinline__ float discrete_entropy_from_raw_weights(
        const float* raw_weights,
        int num_modes,
        float eps
    ) {
        float total = 0.0f;
        for (int m = 0; m < num_modes; ++m) {
            total += fmaxf(raw_weights[m], 0.0f);
        }
        if (total <= eps) {
            return 0.0f;
        }

        float entropy = 0.0f;
        for (int m = 0; m < num_modes; ++m) {
            float p = fmaxf(raw_weights[m], 0.0f) / total;
            if (p > eps) {
                entropy -= p * logf(p);
            }
        }
        return entropy;
    }

    __device__ __forceinline__ float growth_factor(
        int occ_steps,
        int growth_mode,
        float exp_rate
    ) {
        float s = (float)(occ_steps > 0 ? occ_steps : 1);
        if (growth_mode == 0) {        // linear
            return s;
        } else if (growth_mode == 1) { // sqrt
            return sqrtf(s);
        } else if (growth_mode == 2) { // square
            return s * s;
        } else if (growth_mode == 3) { // exp
            return expf(exp_rate * (s - 1.0f));
        } else if (growth_mode == 4) { // constant
            return 1.0f;
        } else { // none
            return 0.0f;
        }
    }


// ****************

    __device__ __forceinline__ void regularize_covariance_2x2(
        float a,
        float b,
        float d,
        float variance_floor,
        float* out_a,
        float* out_b,
        float* out_d
    ) {
        float trace = a + d;
        float diff = a - d;
        float disc = sqrtf(fmaxf(diff * diff + 4.0f * b * b, 0.0f));
        float lambda1 = 0.5f * (trace + disc);
        float lambda2 = 0.5f * (trace - disc);
        lambda1 = fmaxf(lambda1, variance_floor);
        lambda2 = fmaxf(lambda2, variance_floor);

        float c = 1.0f;
        float s = 0.0f;
        float vx = lambda1 - d;
        float vy = b;
        float norm = sqrtf(vx * vx + vy * vy);
        if (norm > 1e-12f) {
            c = vx / norm;
            s = vy / norm;
        } else if (a < d) {
            c = 0.0f;
            s = 1.0f;
        }

        *out_a = c * c * lambda1 + s * s * lambda2;
        *out_b = c * s * (lambda1 - lambda2);
        *out_d = s * s * lambda1 + c * c * lambda2;
    }

    __device__ __forceinline__ void invert_covariance_2x2(
        float a,
        float b,
        float d,
        float variance_floor,
        float eps,
        float* inv_a,
        float* inv_b,
        float* inv_d
    ) {
        float reg_a = a;
        float reg_b = b;
        float reg_d = d;
        regularize_covariance_2x2(
            reg_a, reg_b, reg_d, variance_floor, &reg_a, &reg_b, &reg_d
        );
        float det = reg_a * reg_d - reg_b * reg_b;
        if (det <= eps) {
            reg_a += variance_floor;
            reg_d += variance_floor;
            det = reg_a * reg_d - reg_b * reg_b;
        }
        det = fmaxf(det, eps);
        float inv_det = 1.0f / det;
        *inv_a = reg_d * inv_det;
        *inv_b = -reg_b * inv_det;
        *inv_d = reg_a * inv_det;
    }

    __device__ __forceinline__ float partition_det_proxy_2d(
        const float* unnormalized_weights,
        int num_modes,
        const float* x_at_k,
        const float* y_at_k,
        float sigma00,
        float sigma01,
        float sigma11,
        float variance_floor,
        float eps
    ) {
        float weight_sum = 0.0f;
        for (int m = 0; m < num_modes; ++m) {
            weight_sum += fmaxf(unnormalized_weights[m], 0.0f);
        }
        if (weight_sum <= eps) {
            return 0.0f;
        }

        float state_cov_00 = sigma00;
        float state_cov_01 = sigma01;
        float state_cov_11 = sigma11;
        regularize_covariance_2x2(
            state_cov_00,
            state_cov_01,
            state_cov_11,
            variance_floor,
            &state_cov_00,
            &state_cov_01,
            &state_cov_11
        );

        float mean_x = 0.0f;
        float mean_y = 0.0f;
        for (int m = 0; m < num_modes; ++m) {
            float w = fmaxf(unnormalized_weights[m], 0.0f) / weight_sum;
            mean_x += w * x_at_k[m];
            mean_y += w * y_at_k[m];
        }

        float center_00 = 0.0f;
        float center_01 = 0.0f;
        float center_11 = 0.0f;
        for (int m = 0; m < num_modes; ++m) {
            float w = fmaxf(unnormalized_weights[m], 0.0f) / weight_sum;
            float dx = x_at_k[m] - mean_x;
            float dy = y_at_k[m] - mean_y;
            center_00 += w * dx * dx;
            center_01 += w * dx * dy;
            center_11 += w * dy * dy;
        }

        float total_cov_00 = state_cov_00;
        float total_cov_01 = state_cov_01;
        float total_cov_11 = state_cov_11;
        float center_trace = center_00 + center_11;
        if (center_trace > eps) {
            float inv_00 = 0.0f;
            float inv_01 = 0.0f;
            float inv_11 = 0.0f;
            invert_covariance_2x2(
                state_cov_00,
                state_cov_01,
                state_cov_11,
                variance_floor,
                eps,
                &inv_00,
                &inv_01,
                &inv_11
            );

            float weighted_mahal_sq = 0.0f;
            for (int m = 0; m < num_modes; ++m) {
                float w_m = fmaxf(unnormalized_weights[m], 0.0f) / weight_sum;
                for (int n = 0; n < num_modes; ++n) {
                    float w_n = fmaxf(unnormalized_weights[n], 0.0f) / weight_sum;
                    float dx = x_at_k[n] - x_at_k[m];
                    float dy = y_at_k[n] - y_at_k[m];
                    float mahal_sq =
                        inv_00 * dx * dx + 2.0f * inv_01 * dx * dy + inv_11 * dy * dy;
                    weighted_mahal_sq += w_m * w_n * mahal_sq;
                }
            }
            weighted_mahal_sq *= 0.5f;
            float avg_mahal_sq = weighted_mahal_sq * 0.5f; // state_dim == 2
            float spread_scale = avg_mahal_sq * (state_cov_00 + state_cov_11);
            float spread_gain = spread_scale / center_trace;
            total_cov_00 += center_00 * spread_gain;
            total_cov_01 += center_01 * spread_gain;
            total_cov_11 += center_11 * spread_gain;
        }

        regularize_covariance_2x2(
            total_cov_00,
            total_cov_01,
            total_cov_11,
            variance_floor,
            &total_cov_00,
            &total_cov_01,
            &total_cov_11
        );
        float det = fmaxf(total_cov_00 * total_cov_11 - total_cov_01 * total_cov_01, eps);
        float logdet = logf(det);
        return 0.5f * (2.0f * LOG_2PIE_F + logdet);
    }

    __device__ __forceinline__ float gaussian_entropy_from_covariance_2d(
        float cov_00,
        float cov_01,
        float cov_11,
        float variance_floor,
        float eps
    ) {
        regularize_covariance_2x2(
            cov_00,
            cov_01,
            cov_11,
            variance_floor,
            &cov_00,
            &cov_01,
            &cov_11
        );
        float det = cov_00 * cov_11 - cov_01 * cov_01;
        if (det <= eps) {
            cov_00 += variance_floor;
            cov_11 += variance_floor;
            det = cov_00 * cov_11 - cov_01 * cov_01;
        }
        det = fmaxf(det, eps);
        return 0.5f * (2.0f * LOG_2PIE_F + logf(det));
    }

    __device__ __forceinline__ void accumulate_component_moments_2d(
        const float* raw_weights,
        int num_modes,
        const float* x_at_k,
        const float* y_at_k,
        float cov_00,
        float cov_01,
        float cov_11,
        float variance_floor,
        float eps,
        float* mean_num_x,
        float* mean_num_y,
        float* second_num_00,
        float* second_num_01,
        float* second_num_11
    ) {
        regularize_covariance_2x2(
            cov_00,
            cov_01,
            cov_11,
            variance_floor,
            &cov_00,
            &cov_01,
            &cov_11
        );

        float weight_sum = 0.0f;
        for (int m = 0; m < num_modes; ++m) {
            weight_sum += fmaxf(raw_weights[m], 0.0f);
        }
        if (weight_sum <= eps) {
            return;
        }

        *second_num_00 += weight_sum * cov_00;
        *second_num_01 += weight_sum * cov_01;
        *second_num_11 += weight_sum * cov_11;
        for (int m = 0; m < num_modes; ++m) {
            float w = fmaxf(raw_weights[m], 0.0f);
            if (w <= 0.0f) {
                continue;
            }
            float x = x_at_k[m];
            float y = y_at_k[m];
            *mean_num_x += w * x;
            *mean_num_y += w * y;
            *second_num_00 += w * x * x;
            *second_num_01 += w * x * y;
            *second_num_11 += w * y * y;
        }
    }

    __device__ __forceinline__ float entropy_from_component_moment_sums_2d(
        float total_weight,
        float mean_num_x,
        float mean_num_y,
        float second_num_00,
        float second_num_01,
        float second_num_11,
        float variance_floor,
        float eps
    ) {
        if (total_weight <= eps) {
            return 0.0f;
        }

        float inv_total = 1.0f / total_weight;
        float mean_x = mean_num_x * inv_total;
        float mean_y = mean_num_y * inv_total;
        float cov_00 = second_num_00 * inv_total - mean_x * mean_x;
        float cov_01 = second_num_01 * inv_total - mean_x * mean_y;
        float cov_11 = second_num_11 * inv_total - mean_y * mean_y;
        return gaussian_entropy_from_covariance_2d(
            cov_00,
            cov_01,
            cov_11,
            variance_floor,
            eps
        );
    }

    __device__ __forceinline__ float partition_moment_entropy_2d(
        const float* raw_weights,
        int num_modes,
        const float* x_at_k,
        const float* y_at_k,
        float cov_00,
        float cov_01,
        float cov_11,
        float variance_floor,
        float eps
    ) {
        float total_weight = 0.0f;
        float mean_num_x = 0.0f;
        float mean_num_y = 0.0f;
        float second_num_00 = 0.0f;
        float second_num_01 = 0.0f;
        float second_num_11 = 0.0f;
        for (int m = 0; m < num_modes; ++m) {
            total_weight += fmaxf(raw_weights[m], 0.0f);
        }
        if (total_weight <= eps) {
            return 0.0f;
        }
        accumulate_component_moments_2d(
            raw_weights,
            num_modes,
            x_at_k,
            y_at_k,
            cov_00,
            cov_01,
            cov_11,
            variance_floor,
            eps,
            &mean_num_x,
            &mean_num_y,
            &second_num_00,
            &second_num_01,
            &second_num_11
        );
        return entropy_from_component_moment_sums_2d(
            total_weight,
            mean_num_x,
            mean_num_y,
            second_num_00,
            second_num_01,
            second_num_11,
            variance_floor,
            eps
        );
    }

    __device__ __forceinline__ float partition_separability_logdet_2d(
        const float* unnormalized_weights,
        int num_modes,
        const float* x_at_k,
        const float* y_at_k,
        float sigma00,
        float sigma01,
        float sigma11,
        float variance_floor,
        float eps
    ) {
        float weight_sum = 0.0f;
        for (int m = 0; m < num_modes; ++m) {
            weight_sum += fmaxf(unnormalized_weights[m], 0.0f);
        }
        if (weight_sum <= eps) {
            return 0.0f;
        }

        float state_cov_00 = sigma00;
        float state_cov_01 = sigma01;
        float state_cov_11 = sigma11;
        regularize_covariance_2x2(
            state_cov_00,
            state_cov_01,
            state_cov_11,
            variance_floor,
            &state_cov_00,
            &state_cov_01,
            &state_cov_11
        );

        float mean_x = 0.0f;
        float mean_y = 0.0f;
        for (int m = 0; m < num_modes; ++m) {
            float w = fmaxf(unnormalized_weights[m], 0.0f) / weight_sum;
            mean_x += w * x_at_k[m];
            mean_y += w * y_at_k[m];
        }

        float center_00 = 0.0f;
        float center_01 = 0.0f;
        float center_11 = 0.0f;
        for (int m = 0; m < num_modes; ++m) {
            float w = fmaxf(unnormalized_weights[m], 0.0f) / weight_sum;
            float dx = x_at_k[m] - mean_x;
            float dy = y_at_k[m] - mean_y;
            center_00 += w * dx * dx;
            center_01 += w * dx * dy;
            center_11 += w * dy * dy;
        }

        float total_cov_00 = state_cov_00 + center_00;
        float total_cov_01 = state_cov_01 + center_01;
        float total_cov_11 = state_cov_11 + center_11;
        regularize_covariance_2x2(
            total_cov_00,
            total_cov_01,
            total_cov_11,
            variance_floor,
            &total_cov_00,
            &total_cov_01,
            &total_cov_11
        );

        float det_state =
            fmaxf(state_cov_00 * state_cov_11 - state_cov_01 * state_cov_01, eps);
        float det_total =
            fmaxf(total_cov_00 * total_cov_11 - total_cov_01 * total_cov_01, eps);
        return logf(det_total) - logf(det_state);
    }

    __device__ __forceinline__ float partition_separability_trace_2d(
        const float* unnormalized_weights,
        int num_modes,
        const float* x_at_k,
        const float* y_at_k,
        float sigma00,
        float sigma01,
        float sigma11,
        float variance_floor,
        float eps
    ) {
        float weight_sum = 0.0f;
        for (int m = 0; m < num_modes; ++m) {
            weight_sum += fmaxf(unnormalized_weights[m], 0.0f);
        }
        if (weight_sum <= eps) {
            return 0.0f;
        }

        float state_cov_00 = sigma00;
        float state_cov_01 = sigma01;
        float state_cov_11 = sigma11;
        regularize_covariance_2x2(
            state_cov_00,
            state_cov_01,
            state_cov_11,
            variance_floor,
            &state_cov_00,
            &state_cov_01,
            &state_cov_11
        );

        float mean_x = 0.0f;
        float mean_y = 0.0f;
        for (int m = 0; m < num_modes; ++m) {
            float w = fmaxf(unnormalized_weights[m], 0.0f) / weight_sum;
            mean_x += w * x_at_k[m];
            mean_y += w * y_at_k[m];
        }

        float center_00 = 0.0f;
        float center_01 = 0.0f;
        float center_11 = 0.0f;
        for (int m = 0; m < num_modes; ++m) {
            float w = fmaxf(unnormalized_weights[m], 0.0f) / weight_sum;
            float dx = x_at_k[m] - mean_x;
            float dy = y_at_k[m] - mean_y;
            center_00 += w * dx * dx;
            center_01 += w * dx * dy;
            center_11 += w * dy * dy;
        }

        float inv_00 = 0.0f;
        float inv_01 = 0.0f;
        float inv_11 = 0.0f;
        invert_covariance_2x2(
            state_cov_00,
            state_cov_01,
            state_cov_11,
            variance_floor,
            eps,
            &inv_00,
            &inv_01,
            &inv_11
        );

        return
            inv_00 * center_00
            + 2.0f * inv_01 * center_01
            + inv_11 * center_11;
    }

    __device__ __forceinline__ float partition_spatial_metric_2d(
        int metric_variant,
        const float* unnormalized_weights,
        int num_modes,
        const float* x_at_k,
        const float* y_at_k,
        float sigma00,
        float sigma01,
        float sigma11,
        float variance_floor,
        float eps
    ) {
        if (metric_variant == 1) {
            return partition_separability_logdet_2d(
                unnormalized_weights,
                num_modes,
                x_at_k,
                y_at_k,
                sigma00,
                sigma01,
                sigma11,
                variance_floor,
                eps
            );
        }
        if (metric_variant == 2) {
            return partition_separability_trace_2d(
                unnormalized_weights,
                num_modes,
                x_at_k,
                y_at_k,
                sigma00,
                sigma01,
                sigma11,
                variance_floor,
                eps
            );
        }
        if (metric_variant == 3) {
            return partition_moment_entropy_2d(
                unnormalized_weights,
                num_modes,
                x_at_k,
                y_at_k,
                sigma00,
                sigma01,
                sigma11,
                variance_floor,
                eps
            );
        }
        return partition_det_proxy_2d(
            unnormalized_weights,
            num_modes,
            x_at_k,
            y_at_k,
            sigma00,
            sigma01,
            sigma11,
            variance_floor,
            eps
        );
    }

    __device__ __forceinline__ float partition_sigma_point_entropy_2d(
        const float* unnormalized_weights,
        int num_modes,
        const float* x_at_k,
        const float* y_at_k,
        float sigma00,
        float sigma01,
        float sigma11,
        float sigma_point_scale,
        float sigma_weight0,
        float sigma_weight_other,
        float variance_floor,
        float eps,
        float entropy_floor
    ) {
        const float LOG_2PI = 1.8378770664093453f;
        const float LOG_2PIE = 2.8378770664093453f;

        float weights[32];
        float active_weight_sum = 0.0f;
        for (int m = 0; m < num_modes; ++m) {
            float raw = fmaxf(unnormalized_weights[m], 0.0f);
            if (raw > eps) {
                active_weight_sum += raw;
            }
            weights[m] = 0.0f;
        }
        if (active_weight_sum <= eps) {
            return 0.0f;
        }
        for (int m = 0; m < num_modes; ++m) {
            float raw = fmaxf(unnormalized_weights[m], 0.0f);
            if (raw > eps) {
                weights[m] = raw / active_weight_sum;
            }
        }

        float cov00 = sigma00;
        float cov01 = sigma01;
        float cov11 = sigma11;
        regularize_covariance_2x2(
            cov00, cov01, cov11, variance_floor, &cov00, &cov01, &cov11
        );

        float det = fmaxf(cov00 * cov11 - cov01 * cov01, eps);
        float logdet = logf(det);
        float inv00 = 0.0f;
        float inv01 = 0.0f;
        float inv11 = 0.0f;
        invert_covariance_2x2(
            cov00, cov01, cov11, variance_floor, eps, &inv00, &inv01, &inv11
        );

        float scaled00 = sigma_point_scale * cov00;
        float scaled01 = sigma_point_scale * cov01;
        float scaled11 = sigma_point_scale * cov11;
        regularize_covariance_2x2(
            scaled00,
            scaled01,
            scaled11,
            variance_floor,
            &scaled00,
            &scaled01,
            &scaled11
        );

        float chol00 = sqrtf(fmaxf(scaled00, variance_floor));
        float chol10 = scaled01 / chol00;
        float chol11_sq = fmaxf(scaled11 - chol10 * chol10, variance_floor);
        float chol11 = sqrtf(chol11_sq);

        float log_norm_const = -LOG_2PI - 0.5f * logdet;
        float gaussian_entropy = LOG_2PIE + 0.5f * logdet;
        float mutual_information = 0.0f;

        for (int mode_idx = 0; mode_idx < num_modes; ++mode_idx) {
            float mode_weight = weights[mode_idx];
            if (mode_weight <= eps) {
                continue;
            }

            float mean_x = x_at_k[mode_idx];
            float mean_y = y_at_k[mode_idx];
            float sigma_px[5];
            float sigma_py[5];
            float sigma_pw[5];

            sigma_px[0] = mean_x;
            sigma_py[0] = mean_y;
            sigma_pw[0] = sigma_weight0;

            sigma_px[1] = mean_x + chol00;
            sigma_py[1] = mean_y + chol10;
            sigma_pw[1] = sigma_weight_other;

            sigma_px[2] = mean_x;
            sigma_py[2] = mean_y + chol11;
            sigma_pw[2] = sigma_weight_other;

            sigma_px[3] = mean_x - chol00;
            sigma_py[3] = mean_y - chol10;
            sigma_pw[3] = sigma_weight_other;

            sigma_px[4] = mean_x;
            sigma_py[4] = mean_y - chol11;
            sigma_pw[4] = sigma_weight_other;

            float mode_term = 0.0f;
            for (int sp_idx = 0; sp_idx < 5; ++sp_idx) {
                float px = sigma_px[sp_idx];
                float py = sigma_py[sp_idx];

                float dx0 = px - mean_x;
                float dy0 = py - mean_y;
                float own_qf =
                    inv00 * dx0 * dx0 + 2.0f * inv01 * dx0 * dy0 + inv11 * dy0 * dy0;
                float own_logpdf = log_norm_const - 0.5f * own_qf;

                float max_log = -3.402823466e38f;
                for (int mix_idx = 0; mix_idx < num_modes; ++mix_idx) {
                    float mix_weight = weights[mix_idx];
                    if (mix_weight <= eps) {
                        continue;
                    }
                    float dx = px - x_at_k[mix_idx];
                    float dy = py - y_at_k[mix_idx];
                    float qf =
                        inv00 * dx * dx + 2.0f * inv01 * dx * dy + inv11 * dy * dy;
                    float comp_log = logf(mix_weight) + log_norm_const - 0.5f * qf;
                    if (comp_log > max_log) {
                        max_log = comp_log;
                    }
                }

                float sum_exp = 0.0f;
                for (int mix_idx = 0; mix_idx < num_modes; ++mix_idx) {
                    float mix_weight = weights[mix_idx];
                    if (mix_weight <= eps) {
                        continue;
                    }
                    float dx = px - x_at_k[mix_idx];
                    float dy = py - y_at_k[mix_idx];
                    float qf =
                        inv00 * dx * dx + 2.0f * inv01 * dx * dy + inv11 * dy * dy;
                    float comp_log = logf(mix_weight) + log_norm_const - 0.5f * qf;
                    sum_exp += expf(comp_log - max_log);
                }
                float log_mixture = max_log + logf(fmaxf(sum_exp, eps));
                mode_term += sigma_pw[sp_idx] * (own_logpdf - log_mixture);
            }
            mutual_information += mode_weight * mode_term;
        }

        float entropy = gaussian_entropy + mutual_information;
        if (fabsf(entropy) <= entropy_floor) {
            return 0.0f;
        }
        return entropy;
    }
// ***************

    __device__ __forceinline__ bool chol_2x2_with_jitter(
        float cov_00,
        float cov_01,
        float cov_11,
        float min_variance,
        float* l00,
        float* l10,
        float* l11,
        float* logdet
    ) {
        float jitter = fmaxf(min_variance, 1.0e-9f);
        for (int attempt = 0; attempt < 8; ++attempt) {
            float a00 = cov_00 + jitter;
            float a11 = cov_11 + jitter;
            if (a00 <= 0.0f) {
                jitter = fmaxf(jitter * 10.0f, 1.0e-6f);
                continue;
            }

            float c00 = sqrtf(a00);
            float c10 = cov_01 / c00;
            float diag = a11 - c10 * c10;
            if (diag > 0.0f) {
                *l00 = c00;
                *l10 = c10;
                *l11 = sqrtf(diag);
                *logdet = 2.0f * (logf(*l00) + logf(*l11));
                return true;
            }
            jitter = fmaxf(jitter * 10.0f, 1.0e-6f);
        }

        float safe_00 = fmaxf(cov_00, min_variance);
        float safe_11 = fmaxf(cov_11, min_variance);
        *l00 = sqrtf(safe_00);
        *l10 = 0.0f;
        *l11 = sqrtf(safe_11);
        *logdet = 2.0f * (logf(*l00) + logf(*l11));
        return false;
    }

    __device__ __forceinline__ float gaussian_logpdf_2d(
        float x0,
        float x1,
        float mean0,
        float mean1,
        float l00,
        float l10,
        float l11,
        float log_norm_const
    ) {
        float diff0 = x0 - mean0;
        float diff1 = x1 - mean1;
        float z0 = diff0 / l00;
        float z1 = (diff1 - l10 * z0) / l11;
        float mahal = z0 * z0 + z1 * z1;
        return log_norm_const - 0.5f * mahal;
    }

    __device__ float sigma_point_partition_entropy_2d(
        const float* raw_weights,
        const float* means_x,
        const float* means_y,
        int num_modes,
        float cov_00,
        float cov_01,
        float cov_11,
        float sigma_point_alpha,
        float sigma_point_kappa,
        float eps,
        float kde_min_variance
    ) {
        float weights[MAX_LOCAL_MODES];
        float total_weight = 0.0f;
        for (int m = 0; m < num_modes; ++m) {
            float w = raw_weights[m];
            if (w > 0.0f) {
                total_weight += w;
            }
        }
        if (total_weight <= eps) {
            return 0.0f;
        }

        for (int m = 0; m < num_modes; ++m) {
            float w = raw_weights[m];
            weights[m] = (w > eps) ? (w / total_weight) : 0.0f;
        }

        float l00 = 0.0f;
        float l10 = 0.0f;
        float l11 = 0.0f;
        float logdet = 0.0f;
        chol_2x2_with_jitter(
            cov_00,
            cov_01,
            cov_11,
            kde_min_variance,
            &l00,
            &l10,
            &l11,
            &logdet
        );

        float log_norm_const = -0.5f * (2.0f * LOG_2PI_F + logdet);
        float gaussian_entropy = LOG_2PIE_F + 0.5f * logdet;

        float lam =
            sigma_point_alpha * sigma_point_alpha * (2.0f + sigma_point_kappa) - 2.0f;
        float scale = 2.0f + lam;
        if (scale <= eps) {
            return gaussian_entropy;
        }

        float sigma_weights[NUM_SIGMA_POINTS];
        sigma_weights[0] = lam / scale;
        for (int s = 1; s < NUM_SIGMA_POINTS; ++s) {
            sigma_weights[s] = 0.5f / scale;
        }

        float root_scale = sqrtf(scale);
        float scaled_l00 = root_scale * l00;
        float scaled_l10 = root_scale * l10;
        float scaled_l11 = root_scale * l11;
        float mutual_information = 0.0f;

        for (int m = 0; m < num_modes; ++m) {
            float mode_weight = weights[m];
            if (mode_weight <= eps) {
                continue;
            }

            float sigma_x[NUM_SIGMA_POINTS];
            float sigma_y[NUM_SIGMA_POINTS];
            float mean_x = means_x[m];
            float mean_y = means_y[m];

            sigma_x[0] = mean_x;
            sigma_y[0] = mean_y;
            sigma_x[1] = mean_x + scaled_l00;
            sigma_y[1] = mean_y + scaled_l10;
            sigma_x[2] = mean_x;
            sigma_y[2] = mean_y + scaled_l11;
            sigma_x[3] = mean_x - scaled_l00;
            sigma_y[3] = mean_y - scaled_l10;
            sigma_x[4] = mean_x;
            sigma_y[4] = mean_y - scaled_l11;

            float mode_term = 0.0f;
            for (int s = 0; s < NUM_SIGMA_POINTS; ++s) {
                float x0 = sigma_x[s];
                float x1 = sigma_y[s];
                float log_self = gaussian_logpdf_2d(
                    x0,
                    x1,
                    mean_x,
                    mean_y,
                    l00,
                    l10,
                    l11,
                    log_norm_const
                );

                float max_term = -3.402823466e+38F;
                for (int j = 0; j < num_modes; ++j) {
                    float comp_weight = weights[j];
                    if (comp_weight <= eps) {
                        continue;
                    }
                    float term = logf(fmaxf(comp_weight, eps)) + gaussian_logpdf_2d(
                        x0,
                        x1,
                        means_x[j],
                        means_y[j],
                        l00,
                        l10,
                        l11,
                        log_norm_const
                    );
                    if (term > max_term) {
                        max_term = term;
                    }
                }
                if (max_term <= -3.0e38f) {
                    continue;
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < num_modes; ++j) {
                    float comp_weight = weights[j];
                    if (comp_weight <= eps) {
                        continue;
                    }
                    float term = logf(fmaxf(comp_weight, eps)) + gaussian_logpdf_2d(
                        x0,
                        x1,
                        means_x[j],
                        means_y[j],
                        l00,
                        l10,
                        l11,
                        log_norm_const
                    );
                    sum_exp += expf(term - max_term);
                }

                float log_mixture = max_term + logf(fmaxf(sum_exp, eps));
                mode_term += sigma_weights[s] * (log_self - log_mixture);
            }

            mutual_information += mode_weight * mode_term;
        }

        float entropy = gaussian_entropy + mutual_information;
        if (fabsf(entropy) <= eps) {
            return 0.0f;
        }
        return entropy;
    }

    __global__ void compute_target_metrics_e2e_kde_spatial_sigma_point(
        const float* visibility,          // (N,T,M_total)
        const float* mode_priors,         // (M_total,)
        const int* mode_offsets,          // (A+1,)
        const float* target_states,       // (M_total,T,state_dim)
        const int* confusion_offsets,     // (A+1,)
        const float* confusion_packed,    // packed confusion tensors
        const int* target_risk_durations, // (A,)
        const float* i_times,             // (A,T)
        float* step_oce_out,              // (N*A*T)
        float* step_entropy_out,          // (N*A*T)
        float* step_state_out,            // (N*A*T)
        float* step_mi_out,               // (N*A*T)
        float* per_target_entropy_out,    // (N*A)
        float* per_target_oce_out,        // (N*A)
        float* per_target_state_out,      // (N*A)
        float* per_target_mi_out,         // (N*A)
        float* per_target_occ_out,        // (N*A)
        float* per_target_total_out,      // (N*A)
        int num_trajectories,
        int num_targets,
        int horizon,
        int total_modes,
        int target_state_dim,
        float lambda_oce,
        float lambda_vis,
        float lambda_state,
        float lambda_mi,
        float eps,
        float discount,
        float sigma_point_alpha,
        float sigma_point_kappa,
        float kde_min_variance,
        float obs_trace,
        float proc_cov_00,
        float proc_cov_01,
        float proc_cov_11,
        int growth_mode,
        float growth_exp_rate,
        int partition_mode
    ) {
        int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;
        int total_pairs = num_trajectories * num_targets;

        for(int idx = start_index; idx < total_pairs; idx += stride) {
            int n = idx / num_targets;    // ego trajectory index
            int a = idx % num_targets;    // target index
            int m0 = mode_offsets[a];     // start of modes for this target
            int m1 = mode_offsets[a + 1]; // end of modes for this target
            int Ma = m1 - m0;             // mode count
            int conf_start = confusion_offsets[a];
            int conf_end = confusion_offsets[a + 1];
            int conf_size = conf_end - conf_start;
            int horizon_limit = target_risk_durations[a];
            if (horizon_limit < 0) {
                horizon_limit = 0;
            } else if (horizon_limit > horizon) {
                horizon_limit = horizon;
            }

            if (Ma <= 0 || horizon_limit <= 0) {
                per_target_oce_out[idx] = 0.0f;
                per_target_state_out[idx] = 0.0f;
                per_target_mi_out[idx] = 0.0f;
                per_target_occ_out[idx] = 0.0f;
                per_target_total_out[idx] = 0.0f;
                per_target_entropy_out[idx] = 0.0f;
                continue;
            }

            float sum_occ = 0.0f;
            float sum_oce = 0.0f;
            float sum_entropy = 0.0f;
            float sum_state = 0.0f;
            float sum_mi = 0.0f;
            float disturbance_floor = fmaxf(obs_trace, eps);

            float surv_prefix[MAX_LOCAL_MODES];
            for (int m = 0; m < Ma; ++m) {
                surv_prefix[m] = 1.0f;
            }

            float tail_from_i[MAX_LOCAL_MODES];
            float w_seen[MAX_LOCAL_MODES];
            float work_weights[MAX_LOCAL_MODES];
            float outcome_weights[MAX_LOCAL_MODES];
            float means_kx[MAX_LOCAL_MODES];
            float means_ky[MAX_LOCAL_MODES];
            float marginal_weights[MAX_LOCAL_MODES];

            float disc_k = 1.0f;
            for (int k = 0; k < horizon_limit; ++k) {
                for (int m = 0; m < Ma; ++m) {
                    int g = m0 + m;
                    int v_idx = ((n * horizon + k) * total_modes) + g;
                    float occ_km = 1 - clamp_prob(visibility[v_idx], 0.0f, 1.0f);
                    sum_occ += mode_priors[g] * occ_km;
                    surv_prefix[m] *= occ_km;
                }

                if (k == 0) {
                    continue;
                }

                disc_k *= discount;

                for (int m = 0; m < Ma; ++m) {
                    int g = m0 + m;
                    int ts_idx = (g * horizon + k) * target_state_dim;
                    means_kx[m] = target_states[ts_idx + 0];
                    means_ky[m] = target_states[ts_idx + 1];
                }

                float step_entropy = 0.0f;   // weighted entropy sum H_terms_sum
                float step_state = 0.0f;     // seen partition mass
                float step_mi = 0.0f;        // seen partition MI mass
                float step_occ_mass = 0.0f;
                float aleatoric_weighted = 0.0f;
                for (int m = 0; m < Ma; ++m) {
                    marginal_weights[m] = 0.0f;
                }

                float alpha_unseen = 0.0f;
                for (int m = 0; m < Ma; ++m) {
                    int g = m0 + m;
                    float w = mode_priors[g] * surv_prefix[m];
                    work_weights[m] = w;
                    alpha_unseen += w;
                }

                if (alpha_unseen > eps) {
                    float factor = growth_factor(k, growth_mode, growth_exp_rate);
                    float part_entropy = sigma_point_partition_entropy_2d(
                        work_weights,
                        means_kx,
                        means_ky,
                        Ma,
                        factor * proc_cov_00,
                        factor * proc_cov_01,
                        factor * proc_cov_11,
                        sigma_point_alpha,
                        sigma_point_kappa,
                        eps,
                        kde_min_variance
                    );
                    step_entropy += alpha_unseen * part_entropy;
                    step_occ_mass += alpha_unseen;
                    aleatoric_weighted +=
                        alpha_unseen
                        * discrete_entropy_from_raw_weights(
                            work_weights,
                            Ma,
                            eps
                        );
                    for (int m = 0; m < Ma; ++m) {
                        marginal_weights[m] += work_weights[m];
                    }
                }

                for (int m = 0; m < Ma; ++m) {
                    int g = m0 + m;
                    int v_k_idx = ((n * horizon + k) * total_modes) + g;
                    tail_from_i[m] = 1.0f - clamp_prob(visibility[v_k_idx], 0.0f, 1.0f);
                }

                for (int i = k - 1; i >= 0; --i) {
                    if (i < k - 1) {
                        for (int m = 0; m < Ma; ++m) {
                            int g = m0 + m;
                            int v_next = ((n * horizon + (i + 1)) * total_modes) + g;
                            tail_from_i[m] *= 1.0f - clamp_prob(visibility[v_next], 0.0f, 1.0f);
                        }
                    }

                    float alpha_seen = 0.0f;
                    for (int m = 0; m < Ma; ++m) {
                        int g = m0 + m;
                        int v_i_idx = ((n * horizon + i) * total_modes) + g;
                        float w = mode_priors[g] * clamp_prob(visibility[v_i_idx], 0.0f, 1.0f) * tail_from_i[m];
                        w_seen[m] = w;
                        alpha_seen += w;
                    }
                    if (alpha_seen <= eps) {
                        continue;
                    }

                    step_state += alpha_seen;
                    step_mi += alpha_seen * i_times[a * horizon + i];

                    float factor = growth_factor(k - i, growth_mode, growth_exp_rate);
                    if (partition_mode == 0) {
                        for (int m = 0; m < Ma; ++m) {
                            work_weights[m] = w_seen[m];
                        }
                        float part_entropy = sigma_point_partition_entropy_2d(
                            work_weights,
                            means_kx,
                            means_ky,
                            Ma,
                            factor * proc_cov_00,
                            factor * proc_cov_01,
                            factor * proc_cov_11,
                            sigma_point_alpha,
                            sigma_point_kappa,
                            eps,
                            kde_min_variance
                        );
                        step_entropy += alpha_seen * part_entropy;
                        step_occ_mass += alpha_seen;
                        aleatoric_weighted +=
                            alpha_seen
                            * discrete_entropy_from_raw_weights(
                                work_weights,
                                Ma,
                                eps
                            );
                        for (int m = 0; m < Ma; ++m) {
                            marginal_weights[m] += work_weights[m];
                        }
                        continue;
                    }

                    if (conf_size <= 0) {
                        for (int outcome = 0; outcome < Ma; ++outcome) {
                            float outcome_mass = w_seen[outcome];
                            if (outcome_mass <= eps) {
                                continue;
                            }
                            for (int m = 0; m < Ma; ++m) {
                                outcome_weights[m] = 0.0f;
                            }
                            outcome_weights[outcome] = outcome_mass;
                            float part_entropy = sigma_point_partition_entropy_2d(
                                outcome_weights,
                                means_kx,
                                means_ky,
                                Ma,
                                factor * proc_cov_00,
                                factor * proc_cov_01,
                                factor * proc_cov_11,
                                sigma_point_alpha,
                                sigma_point_kappa,
                                eps,
                                kde_min_variance
                            );
                            step_entropy += outcome_mass * part_entropy;
                            step_occ_mass += outcome_mass;
                            aleatoric_weighted +=
                                outcome_mass
                                * discrete_entropy_from_raw_weights(
                                    outcome_weights,
                                    Ma,
                                    eps
                                );
                            for (int m = 0; m < Ma; ++m) {
                                marginal_weights[m] += outcome_weights[m];
                            }
                        }
                        continue;
                    }

                    int conf_time_base = conf_start + i * Ma * Ma;
                    for (int outcome = 0; outcome < Ma; ++outcome) {
                        float outcome_mass = 0.0f;
                        int conf_row_base = conf_time_base + outcome * Ma;
                        for (int m = 0; m < Ma; ++m) {
                            float outcome_weight =
                                confusion_packed[conf_row_base + m] * w_seen[m];
                            outcome_weights[m] = outcome_weight;
                            outcome_mass += outcome_weight;
                        }
                        if (outcome_mass <= eps) {
                            continue;
                        }
                        float part_entropy = sigma_point_partition_entropy_2d(
                            outcome_weights,
                            means_kx,
                            means_ky,
                            Ma,
                            factor * proc_cov_00,
                            factor * proc_cov_01,
                            factor * proc_cov_11,
                            sigma_point_alpha,
                            sigma_point_kappa,
                            eps,
                            kde_min_variance
                        );
                        step_entropy += outcome_mass * part_entropy;
                        step_occ_mass += outcome_mass;
                        aleatoric_weighted +=
                            outcome_mass
                            * discrete_entropy_from_raw_weights(
                                outcome_weights,
                                Ma,
                                eps
                            );
                        for (int m = 0; m < Ma; ++m) {
                            marginal_weights[m] += outcome_weights[m];
                        }
                    }
                }

                float step_epistemic = 0.0f;
                if (step_occ_mass > eps) {
                    float marginal_entropy = discrete_entropy_from_raw_weights(
                        marginal_weights,
                        Ma,
                        eps
                    );
                    float step_within = aleatoric_weighted / step_occ_mass;
                    step_epistemic = fmaxf(
                        marginal_entropy - step_within,
                        0.0f
                    );
                }
                float step_entropy_discounted = step_epistemic * disc_k;
                sum_entropy += step_entropy_discounted;
                float step_oce_discounted = (
                    (step_occ_mass > eps)
                        ? (step_entropy / step_occ_mass) * disc_k
                        : disturbance_floor * disc_k
                );
                float step_state_discounted = step_state * disc_k;
                float step_mi_discounted = step_mi * disc_k;
                sum_oce += step_oce_discounted;
                sum_state += step_state_discounted;
                sum_mi += step_mi_discounted;

                int step_idx = idx * horizon + k;
                step_oce_out[step_idx] = step_oce_discounted;
                step_entropy_out[step_idx] = step_entropy_discounted;
                step_state_out[step_idx] = step_state_discounted;
                step_mi_out[step_idx] = step_mi_discounted;
            }

            per_target_entropy_out[idx] = sum_entropy;
            per_target_oce_out[idx] = sum_oce;
            per_target_state_out[idx] = sum_state;
            per_target_mi_out[idx] = sum_mi;
            per_target_occ_out[idx] = sum_occ;
            per_target_total_out[idx] =
                lambda_oce * sum_oce + lambda_vis * sum_occ + lambda_state * sum_state - lambda_mi * sum_mi;
        }
    }


    __global__ void compute_target_metrics_e2e_kde_spatial(
        const float* visibility,          // (N,T,M_total)
        const float* mode_priors,         // (M_total,)
        const int* mode_offsets,          // (A+1,)
        const int* confusion_offsets,     // (A+1,)
        const float* confusion_packed,    // packed (T,Ma,Ma) per target
        const float* target_states,       // (M_total,T,state_dim)
        const int* target_risk_durations, // (A,)
        const float* i_times,             // (A,T)
        float* per_target_entropy_out,    // (N*A)
        float* per_target_oce_out,        // (N*A)
        float* per_target_state_out,      // (N*A)
        float* per_target_mi_out,         // (N*A)
        float* per_target_occ_out,        // (N*A)
        float* per_target_total_out,      // (N*A)
        float* step_oce_out,              // (N*A*T)
        float* step_entropy_out,          // (N*A*T)
        float* step_within_out,           // (N*A*T)
        float* step_state_out,            // (N*A*T)
        float* step_mi_out,               // (N*A*T)
        int num_trajectories,
        int num_targets,
        int horizon,
        int total_modes,
        int target_state_dim,
        int partition_mode,               // 0 approximate, 1 exact
        float lambda_oce,
        float lambda_vis,
        float lambda_state,
        float lambda_mi,
        float lambda_e,
        float eps,
        float discount,
        int growth_mode,
        float process_cov_00,
        float process_cov_01,
        float process_cov_11,
        float process_exp_rate,
        float disturbance_floor,
        float variance_floor,
        int metric_variant
    ) {

        int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;
        int total_pairs = num_trajectories * num_targets;

        for (int idx = start_index; idx < total_pairs; idx += stride) {
            int n = idx / num_targets;
            int a = idx % num_targets;
            int m0 = mode_offsets[a];
            int m1 = mode_offsets[a + 1];
            int Ma = m1 - m0;
            int risk_horizon = target_risk_durations[a];
            if (risk_horizon < 0) {
                risk_horizon = 0;
            }
            if (risk_horizon > horizon) {
                risk_horizon = horizon;
            }

            int step_base = idx * horizon;
            for (int t = 0; t < horizon; ++t) {
                step_oce_out[step_base + t] = 0.0f;
                step_entropy_out[step_base + t] = 0.0f;
                step_within_out[step_base + t] = 0.0f;
                step_state_out[step_base + t] = 0.0f;
                step_mi_out[step_base + t] = 0.0f;
            }

            if (Ma <= 0 || Ma > MAX_LOCAL_MODES || horizon <= 0 || risk_horizon <= 0) {
                per_target_oce_out[idx] = 0.0f;
                per_target_state_out[idx] = 0.0f;
                per_target_mi_out[idx] = 0.0f;
                per_target_occ_out[idx] = 0.0f;
                per_target_total_out[idx] = 0.0f;
                per_target_entropy_out[idx] = 0.0f;
                continue;
            }

            float x_at_k[MAX_LOCAL_MODES];
            float y_at_k[MAX_LOCAL_MODES];
            float surv_prefix[MAX_LOCAL_MODES];
            float tail_from_i[MAX_LOCAL_MODES];
            float seen_weights[MAX_LOCAL_MODES];
            float partition_weights[MAX_LOCAL_MODES];
            float marginal_weights[MAX_LOCAL_MODES];
            bool terminal_occlusion_metric =
                (metric_variant == 1 || metric_variant == 2);
            bool info_metric = (metric_variant == 3);

            for (int m = 0; m < Ma; ++m) {
                surv_prefix[m] = 1.0f;
            }

            float sum_occ = 0.0f;
            float sum_oce = 0.0f;
            float sum_entropy = 0.0f;
            float sum_state = 0.0f;
            float sum_mi = 0.0f;
            float discount_k = 1.0f;

            for (int k = 0; k < risk_horizon; ++k) {
                for (int m = 0; m < Ma; ++m) {
                    int g = m0 + m;
                    int v_idx = ((n * horizon + k) * total_modes) + g;
                    float occ_km = 1.0f - clamp_prob(visibility[v_idx], 0.0f, 1.0f);
                    sum_occ += mode_priors[g] * occ_km;
                    surv_prefix[m] *= occ_km;

                    int ts_idx = (g * horizon + k) * target_state_dim;
                    x_at_k[m] = target_states[ts_idx + 0];
                    y_at_k[m] = target_states[ts_idx + 1];
                }

                if (k == 0) {
                    continue;
                }

                discount_k *= discount;
                float weighted_metric_sum = 0.0f;
                float total_occ_mass = 0.0f;
                float aleatoric_weighted = 0.0f;
                float seen_partition_mass = 0.0f;
                float seen_partition_mi = 0.0f;
                for (int m = 0; m < Ma; ++m) {
                    marginal_weights[m] = 0.0f;
                }

                if (terminal_occlusion_metric) {
                    for (int m = 0; m < Ma; ++m) {
                        int g = m0 + m;
                        int v_k_idx = ((n * horizon + k) * total_modes) + g;
                        partition_weights[m] =
                            mode_priors[g]
                            * (1.0f - clamp_prob(visibility[v_k_idx], 0.0f, 1.0f));
                        total_occ_mass += partition_weights[m];
                    }
                    if (total_occ_mass > eps) {
                        float growth = growth_factor(k, growth_mode, process_exp_rate);
                        float sigma00 = growth * process_cov_00;
                        float sigma01 = growth * process_cov_01;
                        float sigma11 = growth * process_cov_11;
                        if (!info_metric) {
                            float metric_value = partition_spatial_metric_2d(
                                metric_variant,
                                partition_weights,
                                Ma,
                                x_at_k,
                                y_at_k,
                                sigma00,
                                sigma01,
                                sigma11,
                                variance_floor,
                                eps
                            );
                            weighted_metric_sum += total_occ_mass * metric_value;
                        }
                        aleatoric_weighted +=
                            total_occ_mass
                            * discrete_entropy_from_raw_weights(
                                partition_weights,
                                Ma,
                                eps
                            );
                        for (int m = 0; m < Ma; ++m) {
                            marginal_weights[m] += partition_weights[m];
                        }
                    }
                } else {
                    float unseen_mass = 0.0f;
                    for (int m = 0; m < Ma; ++m) {
                        int g = m0 + m;
                        partition_weights[m] = mode_priors[g] * surv_prefix[m];
                        unseen_mass += partition_weights[m];
                    }
                    if (unseen_mass > eps) {
                        float growth = growth_factor(k, growth_mode, process_exp_rate);
                        float sigma00 = growth * process_cov_00;
                        float sigma01 = growth * process_cov_01;
                        float sigma11 = growth * process_cov_11;
                        if (!info_metric) {
                            float metric_value = partition_spatial_metric_2d(
                                metric_variant,
                                partition_weights,
                                Ma,
                                x_at_k,
                                y_at_k,
                                sigma00,
                                sigma01,
                                sigma11,
                                variance_floor,
                                eps
                            );
                            weighted_metric_sum += unseen_mass * metric_value;
                        }
                        total_occ_mass += unseen_mass;
                        aleatoric_weighted +=
                            unseen_mass
                            * discrete_entropy_from_raw_weights(
                                partition_weights,
                                Ma,
                                eps
                            );
                        for (int m = 0; m < Ma; ++m) {
                            marginal_weights[m] += partition_weights[m];
                        }
                    }

                    for (int m = 0; m < Ma; ++m) {
                        int g = m0 + m;
                        int v_k_idx = ((n * horizon + k) * total_modes) + g;
                        tail_from_i[m] = 1.0f - clamp_prob(visibility[v_k_idx], 0.0f, 1.0f);
                    }

                    for (int i = k - 1; i >= 0; --i) {
                        if (i < k - 1) {
                            for (int m = 0; m < Ma; ++m) {
                                int g = m0 + m;
                                int v_next_idx = ((n * horizon + (i + 1)) * total_modes) + g;
                                float occ_next =
                                    1.0f - clamp_prob(visibility[v_next_idx], 0.0f, 1.0f);
                                tail_from_i[m] *= occ_next;
                            }
                        }

                        float alpha_seen = 0.0f;
                        for (int m = 0; m < Ma; ++m) {
                            int g = m0 + m;
                            int v_i_idx = ((n * horizon + i) * total_modes) + g;
                            float seen_weight =
                                mode_priors[g]
                                * clamp_prob(visibility[v_i_idx], 0.0f, 1.0f)
                                * tail_from_i[m];
                            seen_weights[m] = seen_weight;
                            alpha_seen += seen_weight;
                        }
                        if (alpha_seen <= eps) {
                            continue;
                        }

                        seen_partition_mass += alpha_seen;
                        seen_partition_mi += alpha_seen * i_times[a * horizon + i];
                        total_occ_mass += alpha_seen;

                        float growth =
                            growth_factor(k - i, growth_mode, process_exp_rate);
                        float sigma00 = growth * process_cov_00;
                        float sigma01 = growth * process_cov_01;
                        float sigma11 = growth * process_cov_11;

                        if (partition_mode == 0) {
                            if (!info_metric) {
                                float metric_value = partition_spatial_metric_2d(
                                    metric_variant,
                                    seen_weights,
                                    Ma,
                                    x_at_k,
                                    y_at_k,
                                    sigma00,
                                    sigma01,
                                    sigma11,
                                    variance_floor,
                                    eps
                                );
                                weighted_metric_sum += alpha_seen * metric_value;
                            }
                            aleatoric_weighted +=
                                alpha_seen
                                * discrete_entropy_from_raw_weights(
                                    seen_weights,
                                    Ma,
                                    eps
                                );
                            for (int m = 0; m < Ma; ++m) {
                                marginal_weights[m] += seen_weights[m];
                            }
                        } else {
                            int conf_step_base = confusion_offsets[a] + i * Ma * Ma;
                            for (int outcome_idx = 0; outcome_idx < Ma; ++outcome_idx) {
                                float outcome_mass = 0.0f;
                                int conf_row_base = conf_step_base + outcome_idx * Ma;
                                for (int m = 0; m < Ma; ++m) {
                                    float outcome_weight =
                                        confusion_packed[conf_row_base + m]
                                        * seen_weights[m];
                                    partition_weights[m] = outcome_weight;
                                    outcome_mass += outcome_weight;
                                }
                                if (outcome_mass <= eps) {
                                    continue;
                                }
                                float metric_value = partition_spatial_metric_2d(
                                        metric_variant,
                                        partition_weights,
                                        Ma,
                                        x_at_k,
                                        y_at_k,
                                        sigma00,
                                        sigma01,
                                        sigma11,
                                        variance_floor,
                                        eps
                                    );
                                if (!info_metric) {
                                    weighted_metric_sum += outcome_mass * metric_value;
                                }
                                aleatoric_weighted +=
                                    outcome_mass
                                    * discrete_entropy_from_raw_weights(
                                        partition_weights,
                                        Ma,
                                        eps
                                    );
                                for (int m = 0; m < Ma; ++m) {
                                    marginal_weights[m] += partition_weights[m];
                                }
                            }
                        }
                    }
                }

                float step_oce = 0.0f;
                float step_entropy = 0.0f;
                float step_within = 0.0f;
                float step_epistemic = 0.0f;
                if (total_occ_mass > eps) {
                    float marginal_entropy = discrete_entropy_from_raw_weights(
                        marginal_weights,
                        Ma,
                        eps
                    );
                    step_within = aleatoric_weighted / total_occ_mass;
                    step_epistemic = fmaxf(
                        marginal_entropy - step_within,
                        0.0f
                    );
                    if (info_metric) {
                        step_entropy = marginal_entropy;
                        step_oce = step_entropy - lambda_e * step_epistemic;
                    } else {
                        step_oce = weighted_metric_sum / total_occ_mass;
                        step_entropy = step_epistemic;
                    }
                } else if (metric_variant == 0) {
                    step_oce = fmaxf(step_oce, disturbance_floor);
                }
                float step_entropy_discounted = step_entropy * discount_k;
                float step_within_discounted = step_within * discount_k;
                float step_state = seen_partition_mass * discount_k;
                float step_mi = seen_partition_mi * discount_k;

                step_oce_out[step_base + k] = step_oce;
                step_entropy_out[step_base + k] = step_entropy_discounted;
                step_within_out[step_base + k] = step_within_discounted;
                step_state_out[step_base + k] = step_state;
                step_mi_out[step_base + k] = step_mi;

                sum_oce += step_oce * discount_k;
                sum_entropy += step_entropy_discounted;
                sum_state += step_state;
                sum_mi += step_mi;
            }

            per_target_entropy_out[idx] = sum_entropy;
            per_target_oce_out[idx] = sum_oce;
            per_target_state_out[idx] = sum_state;
            per_target_mi_out[idx] = sum_mi;
            per_target_occ_out[idx] = sum_occ;
            per_target_total_out[idx] =
                lambda_oce * sum_oce
                + lambda_vis * sum_occ
                + lambda_state * sum_state
                - lambda_mi * sum_mi;
        }
    }

    __global__ void compute_target_metrics_e2e_kde_sigma_point(
        const float* visibility,          // (N,T,M_total)
        const float* mode_priors,         // (M_total,)
        const int* mode_offsets,          // (A+1,)
        const int* target_risk_durations, // (A,)
        const int* confusion_offsets,     // (A+1,)
        const float* confusion_packed,    // packed (T,Ma,Ma) per target
        const float* target_states,       // (M_total,T,state_dim)
        const float* i_times,             // (A,T)
        float* per_target_entropy_out,    // (N*A)
        float* per_target_oce_out,        // (N*A)
        float* per_target_state_out,      // (N*A)
        float* per_target_mi_out,         // (N*A)
        float* per_target_occ_out,        // (N*A)
        float* per_target_total_out,      // (N*A)
        float* step_oce_out,              // (N*A*T)
        float* step_entropy_out,          // (N*A*T)
        float* step_within_out,           // (N*A*T)
        float* step_state_out,            // (N*A*T)
        float* step_mi_out,               // (N*A*T)
        int num_trajectories,
        int num_targets,
        int horizon,
        int total_modes,
        int target_state_dim,
        int partition_mode,               // 0 approximate, 1 exact
        float lambda_oce,
        float lambda_vis,
        float lambda_state,
        float lambda_mi,
        float lambda_e,
        float eps,
        float discount,
        int growth_mode,
        float process_cov_00,
        float process_cov_01,
        float process_cov_11,
        float process_exp_rate,
        float disturbance_floor,
        float variance_floor,
        float sigma_point_scale,
        float sigma_weight0,
        float sigma_weight_other,
        float entropy_floor
    ) {
        const int MAX_LOCAL_MODES = 32;
        (void)lambda_e;

        int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;
        int total_pairs = num_trajectories * num_targets;

        for (int idx = start_index; idx < total_pairs; idx += stride) {
            int n = idx / num_targets;
            int a = idx % num_targets;
            int m0 = mode_offsets[a];
            int m1 = mode_offsets[a + 1];
            int Ma = m1 - m0;
            int risk_horizon = target_risk_durations[a];
            if (risk_horizon < 0) {
                risk_horizon = 0;
            }
            if (risk_horizon > horizon) {
                risk_horizon = horizon;
            }

            int step_base = idx * horizon;
            for (int t = 0; t < horizon; ++t) {
                step_oce_out[step_base + t] = 0.0f;
                step_entropy_out[step_base + t] = 0.0f;
                step_within_out[step_base + t] = 0.0f;
                step_state_out[step_base + t] = 0.0f;
                step_mi_out[step_base + t] = 0.0f;
            }

            if (Ma <= 0 || Ma > MAX_LOCAL_MODES || horizon <= 0 || risk_horizon <= 0) {
                per_target_oce_out[idx] = 0.0f;
                per_target_state_out[idx] = 0.0f;
                per_target_mi_out[idx] = 0.0f;
                per_target_occ_out[idx] = 0.0f;
                per_target_total_out[idx] = 0.0f;
                per_target_entropy_out[idx] = 0.0f;
                continue;
            }

            float x_at_k[MAX_LOCAL_MODES];
            float y_at_k[MAX_LOCAL_MODES];
            float surv_prefix[MAX_LOCAL_MODES];
            float tail_from_i[MAX_LOCAL_MODES];
            float seen_weights[MAX_LOCAL_MODES];
            float partition_weights[MAX_LOCAL_MODES];
            float marginal_weights[MAX_LOCAL_MODES];

            for (int m = 0; m < Ma; ++m) {
                surv_prefix[m] = 1.0f;
            }

            float sum_occ = 0.0f;
            float sum_oce = 0.0f;
            float sum_entropy = 0.0f;
            float sum_state = 0.0f;
            float sum_mi = 0.0f;
            float discount_k = 1.0f;

            for (int k = 0; k < risk_horizon; ++k) {
                for (int m = 0; m < Ma; ++m) {
                    int g = m0 + m;
                    int v_idx = ((n * horizon + k) * total_modes) + g;
                    float occ_km = 1.0f - clamp_prob(visibility[v_idx], 0.0f, 1.0f);
                    sum_occ += mode_priors[g] * occ_km;
                    surv_prefix[m] *= occ_km;

                    int ts_idx = (g * horizon + k) * target_state_dim;
                    x_at_k[m] = target_states[ts_idx + 0];
                    y_at_k[m] = target_states[ts_idx + 1];
                }

                if (k == 0) {
                    continue;
                }

                discount_k *= discount;
                float weighted_entropy_sum = 0.0f;
                float total_occ_mass = 0.0f;
                float aleatoric_weighted = 0.0f;
                float seen_partition_mass = 0.0f;
                float seen_partition_mi = 0.0f;
                for (int m = 0; m < Ma; ++m) {
                    marginal_weights[m] = 0.0f;
                }

                float unseen_mass = 0.0f;
                for (int m = 0; m < Ma; ++m) {
                    int g = m0 + m;
                    partition_weights[m] = mode_priors[g] * surv_prefix[m];
                    unseen_mass += partition_weights[m];
                }
                if (unseen_mass > eps) {
                    float growth = growth_factor(k, growth_mode, process_exp_rate);
                    float sigma00 = growth * process_cov_00;
                    float sigma01 = growth * process_cov_01;
                    float sigma11 = growth * process_cov_11;
                    float entropy_val = partition_sigma_point_entropy_2d(
                        partition_weights,
                        Ma,
                        x_at_k,
                        y_at_k,
                        sigma00,
                        sigma01,
                        sigma11,
                        sigma_point_scale,
                        sigma_weight0,
                        sigma_weight_other,
                        variance_floor,
                        eps,
                        entropy_floor
                    );
                    weighted_entropy_sum += unseen_mass * entropy_val;
                    total_occ_mass += unseen_mass;
                    aleatoric_weighted +=
                        unseen_mass
                        * discrete_entropy_from_raw_weights(
                            partition_weights,
                            Ma,
                            eps
                        );
                    for (int m = 0; m < Ma; ++m) {
                        marginal_weights[m] += partition_weights[m];
                    }
                }

                for (int m = 0; m < Ma; ++m) {
                    int g = m0 + m;
                    int v_k_idx = ((n * horizon + k) * total_modes) + g;
                    tail_from_i[m] = 1.0f - clamp_prob(visibility[v_k_idx], 0.0f, 1.0f);
                }

                for (int i = k - 1; i >= 0; --i) {
                    if (i < k - 1) {
                        for (int m = 0; m < Ma; ++m) {
                            int g = m0 + m;
                            int v_next_idx = ((n * horizon + (i + 1)) * total_modes) + g;
                            float occ_next =
                                1.0f - clamp_prob(visibility[v_next_idx], 0.0f, 1.0f);
                            tail_from_i[m] *= occ_next;
                        }
                    }

                    float alpha_seen = 0.0f;
                    for (int m = 0; m < Ma; ++m) {
                        int g = m0 + m;
                        int v_i_idx = ((n * horizon + i) * total_modes) + g;
                        float seen_weight =
                            mode_priors[g]
                            * clamp_prob(visibility[v_i_idx], 0.0f, 1.0f)
                            * tail_from_i[m];
                        seen_weights[m] = seen_weight;
                        alpha_seen += seen_weight;
                    }
                    if (alpha_seen <= eps) {
                        continue;
                    }

                    seen_partition_mass += alpha_seen;
                    seen_partition_mi += alpha_seen * i_times[a * horizon + i];
                    total_occ_mass += alpha_seen;

                    float growth =
                        growth_factor(k - i, growth_mode, process_exp_rate);
                    float sigma00 = growth * process_cov_00;
                    float sigma01 = growth * process_cov_01;
                    float sigma11 = growth * process_cov_11;

                    if (partition_mode == 0) {
                        float entropy_val = partition_sigma_point_entropy_2d(
                            seen_weights,
                            Ma,
                            x_at_k,
                            y_at_k,
                            sigma00,
                            sigma01,
                            sigma11,
                            sigma_point_scale,
                            sigma_weight0,
                            sigma_weight_other,
                            variance_floor,
                            eps,
                            entropy_floor
                        );
                        weighted_entropy_sum += alpha_seen * entropy_val;
                        aleatoric_weighted +=
                            alpha_seen
                            * discrete_entropy_from_raw_weights(
                                seen_weights,
                                Ma,
                                eps
                            );
                        for (int m = 0; m < Ma; ++m) {
                            marginal_weights[m] += seen_weights[m];
                        }
                    } else {
                        int conf_step_base = confusion_offsets[a] + i * Ma * Ma;
                        for (int outcome_idx = 0; outcome_idx < Ma; ++outcome_idx) {
                            float outcome_mass = 0.0f;
                            int conf_row_base = conf_step_base + outcome_idx * Ma;
                            for (int m = 0; m < Ma; ++m) {
                                float outcome_weight =
                                    confusion_packed[conf_row_base + m]
                                    * seen_weights[m];
                                partition_weights[m] = outcome_weight;
                                outcome_mass += outcome_weight;
                            }
                            if (outcome_mass <= eps) {
                                continue;
                            }
                            float entropy_val = partition_sigma_point_entropy_2d(
                                partition_weights,
                                Ma,
                                x_at_k,
                                y_at_k,
                                sigma00,
                                sigma01,
                                sigma11,
                                sigma_point_scale,
                                sigma_weight0,
                                sigma_weight_other,
                                variance_floor,
                                eps,
                                entropy_floor
                            );
                            weighted_entropy_sum += outcome_mass * entropy_val;
                            aleatoric_weighted +=
                                outcome_mass
                                * discrete_entropy_from_raw_weights(
                                    partition_weights,
                                    Ma,
                                    eps
                                );
                            for (int m = 0; m < Ma; ++m) {
                                marginal_weights[m] += partition_weights[m];
                            }
                        }
                    }
                }

                float step_oce = 0.0f;
                float step_epistemic = 0.0f;
                float step_within = 0.0f;
                if (total_occ_mass > eps) {
                    step_oce = weighted_entropy_sum / total_occ_mass;
                    float marginal_entropy = discrete_entropy_from_raw_weights(
                        marginal_weights,
                        Ma,
                        eps
                    );
                    step_within = aleatoric_weighted / total_occ_mass;
                    step_epistemic = fmaxf(
                        marginal_entropy - step_within,
                        0.0f
                    );
                } else {
                    step_oce = fmaxf(step_oce, disturbance_floor);
                }
                float step_entropy = step_epistemic * discount_k;
                float step_within_discounted = step_within * discount_k;
                float step_state = seen_partition_mass * discount_k;
                float step_mi = seen_partition_mi * discount_k;

                step_oce_out[step_base + k] = step_oce;
                step_entropy_out[step_base + k] = step_entropy;
                step_within_out[step_base + k] = step_within_discounted;
                step_state_out[step_base + k] = step_state;
                step_mi_out[step_base + k] = step_mi;

                sum_oce += step_oce * discount_k;
                sum_entropy += step_entropy;
                sum_state += step_state;
                sum_mi += step_mi;
            }

            per_target_entropy_out[idx] = sum_entropy;
            per_target_oce_out[idx] = sum_oce;
            per_target_state_out[idx] = sum_state;
            per_target_mi_out[idx] = sum_mi;
            per_target_occ_out[idx] = sum_occ;
            per_target_total_out[idx] =
                lambda_oce * sum_oce
                + lambda_vis * sum_occ
                + lambda_state * sum_state
                - lambda_mi * sum_mi;
        }
    }


    __global__ void reduce_target_terms(
        const float* step_oce,
        const float* step_state,
        const float* step_mi,
        const float* per_target_occ,
        float* per_target_oce_out,
        float* per_target_state_out,
        float* per_target_mi_out,
        float* per_target_total_out,
        int num_targets_total,
        int horizon,
        float lambda_oce,
        float lambda_vis,
        float lambda_state,
        float lambda_mi
    ) {
        int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (int idx = start_index; idx < num_targets_total; idx += stride) {
            int base = idx * horizon;
            float s_oce = 0.0f;
            float s_state = 0.0f;
            float s_mi = 0.0f;

            for (int k = 0; k < horizon; ++k) {
                s_oce += step_oce[base + k];
                s_state += step_state[base + k];
                s_mi += step_mi[base + k];
            }

            float occ = per_target_occ[idx];
            per_target_oce_out[idx] = s_oce;
            per_target_state_out[idx] = s_state;
            per_target_mi_out[idx] = s_mi;
            per_target_total_out[idx] = lambda_oce * s_oce + lambda_vis * occ + lambda_state * s_state - lambda_mi * s_mi;
        }
    }

    __global__ void reduce_trajectory_totals(
        const float* per_target_total,
        float* total_entropies,
        int num_trajectories,
        int num_targets
    ) {
        int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n >= num_trajectories) {
            return;
        }

        int base = n * num_targets;
        float s = 0.0f;
        for (int a = 0; a < num_targets; ++a) {
            s += per_target_total[base + a];
        }
        total_entropies[n] = s;
    }
}
"""

_P4_CUDA_MODULE = None
_P4_COMPUTE_TARGET_E2E_KERNEL = None
_P4_COMPUTE_TARGET_E2E_KDE_SPATIAL_KERNEL = None
_P4_COMPUTE_TARGET_E2E_KDE_SIGMA_KERNEL = None
_P4_REDUCE_TARGET_KERNEL = None
_P4_REDUCE_TRAJ_KERNEL = None


def _get_oce_sigma_point_cuda_kernel():
    global _P4_CUDA_MODULE
    global _P4_COMPUTE_TARGET_E2E_KDE_SIGMA_KERNEL

    if not _PYCUDA_AVAILABLE:
        raise RuntimeError("PyCUDA is not available for P4 CUDA execution.")

    if _P4_COMPUTE_TARGET_E2E_KDE_SIGMA_KERNEL is not None:
        return _P4_COMPUTE_TARGET_E2E_KDE_SIGMA_KERNEL

    with _active_cuda_context():
        _P4_CUDA_MODULE = _CudaSourceModule(_P4_CUDA_SOURCE, no_extern_c=True)
        _P4_COMPUTE_TARGET_E2E_KDE_SIGMA_KERNEL = _P4_CUDA_MODULE.get_function(
            "compute_target_metrics_e2e_kde_sigma_point"
        )
    return _P4_COMPUTE_TARGET_E2E_KDE_SIGMA_KERNEL


def _get_oce_spatial_cuda_kernel():
    global _P4_CUDA_MODULE
    global _P4_COMPUTE_TARGET_E2E_KDE_SPATIAL_KERNEL

    if not _PYCUDA_AVAILABLE:
        raise RuntimeError("PyCUDA is not available for P4 CUDA execution.")

    if _P4_COMPUTE_TARGET_E2E_KDE_SPATIAL_KERNEL is not None:
        return _P4_COMPUTE_TARGET_E2E_KDE_SPATIAL_KERNEL

    with _active_cuda_context():
        _P4_CUDA_MODULE = _CudaSourceModule(_P4_CUDA_SOURCE, no_extern_c=True)
        _P4_COMPUTE_TARGET_E2E_KDE_SPATIAL_KERNEL = _P4_CUDA_MODULE.get_function(
            "compute_target_metrics_e2e_kde_spatial"
        )
    return _P4_COMPUTE_TARGET_E2E_KDE_SPATIAL_KERNEL


def _get_oce_reduce_traj_cuda_kernel():
    global _P4_CUDA_MODULE
    global _P4_REDUCE_TRAJ_KERNEL

    if not _PYCUDA_AVAILABLE:
        raise RuntimeError("PyCUDA is not available for P4 CUDA execution.")

    if _P4_REDUCE_TRAJ_KERNEL is not None:
        return _P4_REDUCE_TRAJ_KERNEL

    with _active_cuda_context():
        _P4_CUDA_MODULE = _CudaSourceModule(_P4_CUDA_SOURCE, no_extern_c=True)
        _P4_REDUCE_TRAJ_KERNEL = _P4_CUDA_MODULE.get_function(
            "reduce_trajectory_totals"
        )
    return _P4_REDUCE_TRAJ_KERNEL


_P2_OCC_VIS_CUDA_SOURCE = r"""
extern "C" {
    #include <math.h>

    #define MAX_FOOTPRINT_CELLS 1500
    #define MAX_BOUNDARY_CELLS 100

    __device__ __forceinline__ float clamp01f(float x) {
        if (x < 0.0f) return 0.0f;
        if (x > 1.0f) return 1.0f;
        return x;
    }

    __device__ __forceinline__ float wrap_to_pi_f(float angle) {
        const float PI_F = 3.14159265358979323846f;
        const float TWO_PI_F = 6.28318530717958647692f;
        while (angle <= -PI_F) angle += TWO_PI_F;
        while (angle > PI_F) angle -= TWO_PI_F;
        return angle;
    }

    __device__ __forceinline__ int abs_i(int x) {
        return x < 0 ? -x : x;
    }

    __device__ int gcd_i(int a, int b) {
        a = abs_i(a);
        b = abs_i(b);
        if (a == 0) return b;
        if (b == 0) return a;
        while (b != 0) {
            int t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    __device__ bool point_in_polygon4(float px, float py, const float* poly_x, const float* poly_y) {
        bool inside = false;
        int j = 3;
        for (int i = 0; i < 4; ++i) {
            float yi = poly_y[i];
            float yj = poly_y[j];
            float xi = poly_x[i];
            float xj = poly_x[j];

            if ((yi > py) != (yj > py)) {
                float denom = yj - yi;
                if (fabsf(denom) >= 1e-12f) {
                    float x_intersect = (xj - xi) * (py - yi) / denom + xi;
                    if (px <= x_intersect) {
                        inside = !inside;
                    }
                }
            }
            j = i;
        }
        return inside;
    }


    __device__ int rasterize_footprint_cells(
        float size_x,
        float size_y,
        float centre_x,
        float centre_y,
        float heading,
        float origin_x,
        float origin_y,
        float resolution,
        int width,
        int height,
        int* out_x,
        int* out_y,
        int max_cells
    ) {
        float half_x_cells = ceilf(size_x / (2.0f * resolution));
        float half_y_cells = ceilf(size_y / (2.0f * resolution));

        // Match planning.agent_footprint.rasterize_agent_footprint_cells():
        // build a rotated grid-space rectangle and keep integer grid points that
        // fall inside the polygon. skimage.draw.polygon uses integer pixel
        // coordinates rather than cell centres, and using the same convention
        // avoids a one-cell footprint offset relative to the CPU path.
        float local_x[4];
        float local_y[4];
        local_x[0] = -half_x_cells;          local_y[0] = -half_y_cells;
        local_x[1] = -half_x_cells;          local_y[1] = half_y_cells + 1.0f;
        local_x[2] = half_x_cells + 1.0f;    local_y[2] = half_y_cells + 1.0f;
        local_x[3] = half_x_cells + 1.0f;    local_y[3] = -half_y_cells;

        float c = cosf(heading);
        float s = sinf(heading);
        float grid_cx = (centre_x - origin_x) / resolution;
        float grid_cy = (centre_y - origin_y) / resolution;

        float poly_x[4];
        float poly_y[4];
        float min_x = 1.0e30f;
        float max_x = -1.0e30f;
        float min_y = 1.0e30f;
        float max_y = -1.0e30f;

        for (int i = 0; i < 4; ++i) {
            float tx = local_x[i] * c - local_y[i] * s + grid_cx;
            float ty = local_x[i] * s + local_y[i] * c + grid_cy;
            poly_x[i] = tx;
            poly_y[i] = ty;
            if (tx < min_x) min_x = tx;
            if (tx > max_x) max_x = tx;
            if (ty < min_y) min_y = ty;
            if (ty > max_y) max_y = ty;
        }

        int x0 = (int)floorf(min_x);
        int x1 = (int)ceilf(max_x);
        int y0 = (int)floorf(min_y);
        int y1 = (int)ceilf(max_y);

        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;
        if (x1 > width - 1) x1 = width - 1;
        if (y1 > height - 1) y1 = height - 1;
        if (x0 > x1 || y0 > y1) return 0;

        int count = 0;
        for (int yy = y0; yy <= y1; ++yy) {
            for (int xx = x0; xx <= x1; ++xx) {
                float px = (float)xx;
                float py = (float)yy;
                if (!point_in_polygon4(px, py, poly_x, poly_y)) {
                    continue;
                }
                if (count < max_cells) {
                    out_x[count] = xx;
                    out_y[count] = yy;
                    count += 1;
                    if (count >= max_cells) {
                        printf("Warning: reached max_cells limit in rasterization\n");
                        printf("Warning: need %d cells but only have capacity for %d\n", (int)((2*half_x_cells+1)*(2*half_y_cells+1)), max_cells);
                        break;
                    }
                }
            }
        }

        return count < max_cells ? count : max_cells;
    }

    __global__ void copy_base_grid(
        const float* base_grid,
        float* out_grid,
        int num_cells
    ) {
        int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (int i = start_index; i < num_cells; i += stride) {
            out_grid[i] = clamp01f(base_grid[i]);
        }
    }

    __global__ void accumulate_static_agents(
        float* static_grid,
        const float* static_states,
        const float* static_sizes,
        int num_static,
        int width,
        int height,
        float origin_x,
        float origin_y,
        float resolution
    ) {
        int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        // if (!start_index) {
        //     printf("**\n**\n**\n");
        //     printf("Static Grid: origin=(%.2f, %.2f) res=%.2f size=(%d, %d)\n", origin_x, origin_y, resolution, width, height);
        // }

        for (int idx = start_index; idx < num_static; idx += stride) {
            int cells_x[MAX_FOOTPRINT_CELLS];
            int cells_y[MAX_FOOTPRINT_CELLS];

            const float* state = static_states + idx * 3;
            const float* size = static_sizes + idx * 2;
            int count = rasterize_footprint_cells(
                size[0], size[1],
                state[0], state[1], state[2],
                origin_x, origin_y, resolution,
                width, height,
                cells_x, cells_y, MAX_FOOTPRINT_CELLS
            );

            for (int i = 0; i < count; ++i) {
                int cell = cells_y[i] * width + cells_x[i];
                static_grid[cell] = 1.0f;
            }
        }
    }

    __global__ void clamp_buffer(float* data, int num_values) {
        int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (int i = start_index; i < num_values; i += stride) {
            data[i] = clamp01f(data[i]);
        }
    }

    __global__ void broadcast_static_template(
        const float* static_grid,
        float* occupancy,
        int num_trajectories,
        int horizon,
        int num_cells
    ) {
        int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        int total = num_trajectories * horizon * num_cells;
        auto stride = blockDim.x * gridDim.x;

        for (int i = start_index; i < total; i += stride) {
            occupancy[i] = static_grid[i % num_cells];
        }
    }

    __global__ void zero_buffer(float* data, int num_values) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_values) return;
        data[idx] = 0.0f;
    }

    __global__ void accumulate_dynamic_occupancy_ownership(
        float* dynamic_sum,
        unsigned long long* owner_key,
        const float* target_states,
        const float* target_sizes,
        const int* mode_to_target,
        const float* mode_probabilities,
        int total_modes,
        int horizon,
        int width,
        int height,
        float origin_x,
        float origin_y,
        float resolution,
        float probability_threshold,
        float owner_min_contrib_eps,
        int* conflict_mask,
        int* conflict_counts_per_t,
        int* first_conflict_cell_per_t,
        int* first_conflict_owner_a_per_t,
        int* first_conflict_owner_b_per_t
    ) {
        int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;
        int total = total_modes * horizon;
        int num_cells = width * height;
        bool track_conflicts = (
            conflict_mask != NULL &&
            conflict_counts_per_t != NULL &&
            first_conflict_cell_per_t != NULL &&
            first_conflict_owner_a_per_t != NULL &&
            first_conflict_owner_b_per_t != NULL
        );

        for (int idx = start_index; idx < total; idx += stride) {
            int mode_idx = idx / horizon;
            int step_idx = idx % horizon;
            float prob = mode_probabilities[idx];
            if (prob <= probability_threshold) {
                continue;
            }

            int target_idx = mode_to_target[mode_idx];
            int cells_x[MAX_FOOTPRINT_CELLS];
            int cells_y[MAX_FOOTPRINT_CELLS];

            const float* size = target_sizes + target_idx * 2;
            const float* state = target_states + idx * 3;
            int count = rasterize_footprint_cells(
                size[0], size[1],
                state[0], state[1], state[2],
                origin_x, origin_y, resolution,
                width, height,
                cells_x, cells_y, MAX_FOOTPRINT_CELLS
            );

            int base = step_idx * num_cells;
            for (int i = 0; i < count; ++i) {
                int local_cell = cells_y[i] * width + cells_x[i];
                int cell = base + local_cell;

                atomicAdd(dynamic_sum + cell, prob);
                if (prob <= owner_min_contrib_eps) {
                    continue;
                }

                unsigned int contrib_bits = __float_as_uint(prob);
                unsigned int owner_rank = 0xffffffffu - (unsigned int)target_idx;
                unsigned long long new_key =
                    (((unsigned long long)contrib_bits) << 32) |
                    (unsigned long long)owner_rank;
                unsigned long long old_key = atomicMax(owner_key + cell, new_key);

                if (track_conflicts) {
                    unsigned int old_contrib_bits = (unsigned int)(old_key >> 32);
                    if (old_contrib_bits != 0u) {
                        unsigned int old_rank = (unsigned int)(old_key & 0xffffffffull);
                        int old_owner = (int)(0xffffffffu - old_rank);
                        if (old_owner != target_idx) {
                            if (atomicCAS(conflict_mask + cell, 0, 1) == 0) {
                                atomicAdd(conflict_counts_per_t + step_idx, 1);
                                if (atomicCAS(first_conflict_cell_per_t + step_idx, -1, local_cell) == -1) {
                                    first_conflict_owner_a_per_t[step_idx] = old_owner;
                                    first_conflict_owner_b_per_t[step_idx] = target_idx;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    __global__ void decode_ownership_from_keys(
        const unsigned long long* owner_key,
        int* ownership,
        int num_values,
        float owner_min_contrib_eps
    ) {
        int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;
        for (int idx = start_index; idx < num_values; idx += stride) {
            unsigned long long key = owner_key[idx];
            unsigned int contrib_bits = (unsigned int)(key >> 32);
            if (contrib_bits == 0u) {
                ownership[idx] = -1;
                continue;
            }
            float contrib = __uint_as_float(contrib_bits);
            if (contrib <= owner_min_contrib_eps) {
                ownership[idx] = -1;
                continue;
            }
            unsigned int owner_rank = (unsigned int)(key & 0xffffffffull);
            ownership[idx] = (int)(0xffffffffu - owner_rank);
        }
    }

    __device__ float trace_visibility_ray(
        const float* static_grid,
        const float* dynamic_sum,
        const int* ownership,
        int width,
        int height,
        int sx,
        int sy,
        int ex,
        int ey,
        int target_idx,
        float eps,
        int debug
    ) {
        int dx = abs_i(sx - ex);
        int step_x = sx < ex ? 1 : -1;
        int dy = -abs_i(sy - ey);
        int step_y = sy < ey ? 1 : -1;
        int error = dx + dy;

        if (sx == ex && sy == ey) {
            return 1.0f;
        }

        float observation = 1.0f;
        for (;;) {
            if (sx == ex && sy == ey) {
                break;
            }

            int e2 = 2 * error;
            if (e2 >= dy) {
                if (sx == ex) {
                    break;
                }
                error += dy;
                sx += step_x;
            }
            if (e2 <= dx) {
                if (sy == ey) {
                    break;
                }
                error += dx;
                sy += step_y;
            }

            if (sx == ex && sy == ey) {
                break;
            }

            if (sx < 0 || sx >= width || sy < 0 || sy >= height) {
                return 0.0f;
            }

            int cell = sy * width + sx;
            float non_target_occ = clamp01f(static_grid[cell]);

            if( debug ) {
                printf("Ray step: cell=(%d, %d) static_occ=%.3f, dyn_occ=%.3f\n", sx, sy, non_target_occ, dynamic_sum[cell]);
            }
            if (dynamic_sum[cell] > eps) {
                int owner_idx = ownership ? ownership[cell] : -1;
                if (owner_idx != target_idx) {
                    non_target_occ = clamp01f(non_target_occ + dynamic_sum[cell]);
                }
            }
            if (non_target_occ > eps) {
                observation *= (1.0f - non_target_occ);
                if (observation <= eps) {
                    return 0.0f;
                }
            }
            if( debug ) {
                printf("Updated observation: %.6f\n", observation);
            }
        }

        return observation;
    }

    __global__ void compute_visibility_boundary_rays(
        const float* static_grid,
        const float* dynamic_sum,
        const int* ownership,
        const float* ego_states,
        const float* target_states,
        const float* target_sizes,
        const int* mode_to_target,
        float* visibility,
        int num_trajectories,
        int horizon,
        int total_modes,
        int num_targets,
        int width,
        int height,
        float origin_x,
        float origin_y,
        float resolution,
        int ego_state_stride,
        float eps_visibility,
        int use_boundary_cells
    ) {
        int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;
        int total = num_trajectories * horizon * total_modes;

        if (!start_index) {
            printf("**\n");
            printf("Grid: origin=(%.2f, %.2f) res=%.2f size=(%d, %d)\n", origin_x, origin_y, resolution, width, height);
        }

        for (int idx = start_index; idx < total; idx += stride) {

            int mode_idx = idx % total_modes;
            int tmp = idx / total_modes;
            int step_idx = tmp % horizon;
            int traj_idx = tmp / horizon;

            int target_idx = mode_to_target[mode_idx];
            if (target_idx < 0 || target_idx >= num_targets) {
                visibility[idx] = 0.0f;
                continue;
            }

            const float* ego_state = ego_states + (traj_idx * horizon + step_idx) * ego_state_stride;
            int sx = (int)floorf((ego_state[0] - origin_x) / resolution);
            int sy = (int)floorf((ego_state[1] - origin_y) / resolution);
            if (sx < 0 || sx >= width || sy < 0 || sy >= height) {
                visibility[idx] = 0.0f;
                continue;
            }

            int cells_x[MAX_FOOTPRINT_CELLS];
            int cells_y[MAX_FOOTPRINT_CELLS];
            const float* size = target_sizes + target_idx * 2;
            const float* state = target_states + (mode_idx * horizon + step_idx) * 3;
            int foot_count = rasterize_footprint_cells(
                size[0], size[1],
                state[0], state[1], state[2],
                origin_x, origin_y, resolution,
                width, height,
                cells_x, cells_y, MAX_FOOTPRINT_CELLS
            );
            if (foot_count <= 0) {
                visibility[idx] = 0.0f;
                continue;
            }

            int boundary_x[MAX_BOUNDARY_CELLS];
            int boundary_y[MAX_BOUNDARY_CELLS];
            int boundary_count = 0;

            if (use_boundary_cells != 0) {
                for (int i = 0; i < foot_count; ++i) {
                    int x = cells_x[i];
                    int y = cells_y[i];
                    bool is_boundary = false;
                    const int nx[4] = {x - 1, x + 1, x, x};
                    const int ny[4] = {y, y, y - 1, y + 1};
                    for (int nb = 0; nb < 4; ++nb) {
                        int qx = nx[nb];
                        int qy = ny[nb];
                        if (qx < 0 || qx >= width || qy < 0 || qy >= height) {
                            is_boundary = true;
                            break;
                        }
                        bool found = false;
                        for (int j = 0; j < foot_count; ++j) {
                            if (cells_x[j] == qx && cells_y[j] == qy) {
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            is_boundary = true;
                            break;
                        }
                    }
                    if (!is_boundary) {
                        continue;
                    }
                    if (boundary_count < MAX_BOUNDARY_CELLS) {
                        boundary_x[boundary_count] = x;
                        boundary_y[boundary_count] = y;
                    }
                    boundary_count += 1;
                }
                if (boundary_count > MAX_BOUNDARY_CELLS) {
                    boundary_count = MAX_BOUNDARY_CELLS;
                }
                if (boundary_count <= 0) {
                    visibility[idx] = 0.0f;
                    continue;
                }
            }

            const float* static_slice = static_grid;
            const float* dynamic_slice = dynamic_sum + step_idx * width * height;
            const int* owner_slice = ownership + step_idx * width * height;

            float obs_sum = 0.0f;
            int rays_tested = 0;
            if (use_boundary_cells != 0) {
                for (int i = 0; i < boundary_count; ++i) {
                    int ex = boundary_x[i];
                    int ey = boundary_y[i];
                    if (ex == sx && ey == sy) {
                        obs_sum += 1.0f;
                        rays_tested += 1;
                        continue;
                    }
                    float obs = trace_visibility_ray(
                        static_slice,
                        dynamic_slice,
                        owner_slice,
                        width,
                        height,
                        sx,
                        sy,
                        ex,
                        ey,
                        target_idx,
                        eps_visibility,
                        0
                    );
                    obs_sum += obs;
                    rays_tested += 1;
                }
            } else {
                for (int i = 0; i < foot_count; ++i) {
                    int ex = cells_x[i];
                    int ey = cells_y[i];
                    if (ex == sx && ey == sy) {
                        obs_sum += 1.0f;
                        rays_tested += 1;
                        continue;
                    }
                    float obs = trace_visibility_ray(
                        static_slice,
                        dynamic_slice,
                        owner_slice,
                        width,
                        height,
                        sx,
                        sy,
                        ex,
                        ey,
                        target_idx,
                        eps_visibility,
                        0
                    );
                    obs_sum += obs;
                    rays_tested += 1;
                }
            }
            visibility[idx] = rays_tested > 0 ? (obs_sum / (float)rays_tested) : 0.0f;
            // printf("Visibility idx %d (mode %d, traj %d, step %d): obs_sum=%.4f rays=%d visibility=%.4f\n",
            //     idx, mode_idx, traj_idx, step_idx, obs_sum, rays_tested, visibility[idx] );
        }
    }

    __global__ void compute_visibility_queries(
        const float* static_grid,
        const float* dynamic_sum,
        const int* ownership,
        const int* query_steps,
        const float* query_states,
        const float* target_states,
        const float* target_sizes,
        const int* mode_to_target,
        float* visibility,
        int num_queries,
        int horizon,
        int total_modes,
        int num_targets,
        int width,
        int height,
        float origin_x,
        float origin_y,
        float resolution,
        float eps_visibility,
        int use_boundary_cells
    ) {
        int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;
        int total = num_queries * total_modes;

        for (int idx = start_index; idx < total; idx += stride) {
            int mode_idx = idx % total_modes;
            int query_idx = idx / total_modes;
            int step_idx = query_steps[query_idx];

            if (step_idx < 0 || step_idx >= horizon) {
                visibility[idx] = 0.0f;
                continue;
            }

            int target_idx = mode_to_target[mode_idx];
            if (target_idx < 0 || target_idx >= num_targets) {
                visibility[idx] = 0.0f;
                continue;
            }

            const float* ego_state = query_states + query_idx * 3;
            int sx = (int)floorf((ego_state[0] - origin_x) / resolution);
            int sy = (int)floorf((ego_state[1] - origin_y) / resolution);
            if (sx < 0 || sx >= width || sy < 0 || sy >= height) {
                visibility[idx] = 0.0f;
                continue;
            }

            int cells_x[MAX_FOOTPRINT_CELLS];
            int cells_y[MAX_FOOTPRINT_CELLS];
            const float* size = target_sizes + target_idx * 2;
            const float* state = target_states + (mode_idx * horizon + step_idx) * 3;
            int foot_count = rasterize_footprint_cells(
                size[0], size[1],
                state[0], state[1], state[2],
                origin_x, origin_y, resolution,
                width, height,
                cells_x, cells_y, MAX_FOOTPRINT_CELLS
            );
            if (foot_count <= 0) {
                visibility[idx] = 0.0f;
                continue;
            }

            int boundary_x[MAX_BOUNDARY_CELLS];
            int boundary_y[MAX_BOUNDARY_CELLS];
            int boundary_count = 0;

            if (use_boundary_cells != 0) {
                for (int i = 0; i < foot_count; ++i) {
                    int x = cells_x[i];
                    int y = cells_y[i];
                    bool is_boundary = false;
                    const int nx[4] = {x - 1, x + 1, x, x};
                    const int ny[4] = {y, y, y - 1, y + 1};
                    for (int nb = 0; nb < 4; ++nb) {
                        int qx = nx[nb];
                        int qy = ny[nb];
                        if (qx < 0 || qx >= width || qy < 0 || qy >= height) {
                            is_boundary = true;
                            break;
                        }
                        bool found = false;
                        for (int j = 0; j < foot_count; ++j) {
                            if (cells_x[j] == qx && cells_y[j] == qy) {
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            is_boundary = true;
                            break;
                        }
                    }
                    if (!is_boundary) {
                        continue;
                    }
                    if (boundary_count < MAX_BOUNDARY_CELLS) {
                        boundary_x[boundary_count] = x;
                        boundary_y[boundary_count] = y;
                    }
                    boundary_count += 1;
                }
                if (boundary_count > MAX_BOUNDARY_CELLS) {
                    boundary_count = MAX_BOUNDARY_CELLS;
                }
                if (boundary_count <= 0) {
                    visibility[idx] = 0.0f;
                    continue;
                }
            }

            const float* static_slice = static_grid;
            const float* dynamic_slice = dynamic_sum + step_idx * width * height;
            const int* owner_slice = ownership + step_idx * width * height;

            float obs_sum = 0.0f;
            int rays_tested = 0;
            if (use_boundary_cells != 0) {
                for (int i = 0; i < boundary_count; ++i) {
                    int ex = boundary_x[i];
                    int ey = boundary_y[i];
                    if (ex == sx && ey == sy) {
                        obs_sum += 1.0f;
                        rays_tested += 1;
                        continue;
                    }
                    float obs = trace_visibility_ray(
                        static_slice,
                        dynamic_slice,
                        owner_slice,
                        width,
                        height,
                        sx,
                        sy,
                        ex,
                        ey,
                        target_idx,
                        eps_visibility,
                        0
                    );
                    obs_sum += obs;
                    rays_tested += 1;
                }
            } else {
                for (int i = 0; i < foot_count; ++i) {
                    int ex = cells_x[i];
                    int ey = cells_y[i];
                    if (ex == sx && ey == sy) {
                        obs_sum += 1.0f;
                        rays_tested += 1;
                        continue;
                    }
                    float obs = trace_visibility_ray(
                        static_slice,
                        dynamic_slice,
                        owner_slice,
                        width,
                        height,
                        sx,
                        sy,
                        ex,
                        ey,
                        target_idx,
                        eps_visibility,
                        0
                    );
                    obs_sum += obs;
                    rays_tested += 1;
                }
            }
            visibility[idx] = rays_tested > 0 ? (obs_sum / (float)rays_tested) : 0.0f;
        }
    }

    __global__ void compute_sensor_gate_tensor(
        const float* ego_states,
        const float* target_states,
        const float* target_sizes,
        const int* mode_to_target,
        float* sensor_gate,
        int num_trajectories,
        int horizon,
        int total_modes,
        int num_targets,
        int width,
        int height,
        float origin_x,
        float origin_y,
        float resolution,
        int ego_state_stride,
        int ego_heading_index,
        float sensor_range,
        float sensor_fov
    ) {
        int start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;
        int total = num_trajectories * horizon * total_modes;
        bool full_fov = sensor_fov >= 6.19591884457987f;

        for (int idx = start_index; idx < total; idx += stride) {
            if (sensor_range <= 0.0f || sensor_fov <= 0.0f) {
                sensor_gate[idx] = 0.0f;
                continue;
            }

            int mode_idx = idx % total_modes;
            int tmp = idx / total_modes;
            int step_idx = tmp % horizon;
            int traj_idx = tmp / horizon;

            int target_idx = mode_to_target[mode_idx];
            if (target_idx < 0 || target_idx >= num_targets) {
                sensor_gate[idx] = 0.0f;
                continue;
            }

            const float* ego_state =
                ego_states + (traj_idx * horizon + step_idx) * ego_state_stride;
            float observer_x = ego_state[0];
            float observer_y = ego_state[1];
            float observer_heading =
                ego_heading_index >= 0 && ego_heading_index < ego_state_stride
                ? ego_state[ego_heading_index]
                : 0.0f;

            int cells_x[MAX_FOOTPRINT_CELLS];
            int cells_y[MAX_FOOTPRINT_CELLS];
            const float* size = target_sizes + target_idx * 2;
            const float* state = target_states + (mode_idx * horizon + step_idx) * 3;
            int foot_count = rasterize_footprint_cells(
                size[0], size[1],
                state[0], state[1], state[2],
                origin_x, origin_y, resolution,
                width, height,
                cells_x, cells_y, MAX_FOOTPRINT_CELLS
            );
            if (foot_count <= 0) {
                sensor_gate[idx] = 0.0f;
                continue;
            }

            int hit_count = 0;
            for (int i = 0; i < foot_count; ++i) {
                float world_x = origin_x + ((float)cells_x[i] + 0.5f) * resolution;
                float world_y = origin_y + ((float)cells_y[i] + 0.5f) * resolution;
                float dx = world_x - observer_x;
                float dy = world_y - observer_y;
                float distance = sqrtf(dx * dx + dy * dy);
                if (distance > sensor_range + 1.0e-6f) {
                    continue;
                }
                if (!full_fov) {
                    float bearing = atan2f(dy, dx);
                    float angle_delta = fabsf(wrap_to_pi_f(bearing - observer_heading));
                    if (angle_delta > (0.5f * sensor_fov + 1.0e-6f)) {
                        continue;
                    }
                }
                hit_count += 1;
            }

            sensor_gate[idx] =
                foot_count > 0 ? ((float)hit_count / (float)foot_count) : 0.0f;
        }
    }
}
"""

_P2_OCC_VIS_CUDA_MODULE = None
_P2_COPY_BASE_GRID_KERNEL = None
_P2_ACCUM_STATIC_AGENTS_KERNEL = None
_P2_CLAMP_BUFFER_KERNEL = None
_P2_BROADCAST_TEMPLATE_KERNEL = None
_P2_ZERO_BUFFER_KERNEL = None
_P2_ACCUM_DYNAMIC_KERNEL = None
_P2_DECODE_OWNERSHIP_KERNEL = None
_P2_COMPUTE_VISIBILITY_KERNEL = None
_P2_COMPUTE_QUERY_VISIBILITY_KERNEL = None
_P2_COMPUTE_SENSOR_GATE_KERNEL = None


def _get_occ_vis_cuda_kernels():
    global _P2_OCC_VIS_CUDA_MODULE
    global _P2_COPY_BASE_GRID_KERNEL
    global _P2_ACCUM_STATIC_AGENTS_KERNEL
    global _P2_CLAMP_BUFFER_KERNEL
    global _P2_BROADCAST_TEMPLATE_KERNEL
    global _P2_ZERO_BUFFER_KERNEL
    global _P2_ACCUM_DYNAMIC_KERNEL
    global _P2_DECODE_OWNERSHIP_KERNEL
    global _P2_COMPUTE_VISIBILITY_KERNEL
    global _P2_COMPUTE_QUERY_VISIBILITY_KERNEL
    global _P2_COMPUTE_SENSOR_GATE_KERNEL

    if not _PYCUDA_AVAILABLE:
        raise RuntimeError(
            "PyCUDA is not available for occupancy/visibility CUDA execution."
        )

    if (
        _P2_COPY_BASE_GRID_KERNEL is not None
        and _P2_ACCUM_STATIC_AGENTS_KERNEL is not None
        and _P2_CLAMP_BUFFER_KERNEL is not None
        and _P2_BROADCAST_TEMPLATE_KERNEL is not None
        and _P2_ZERO_BUFFER_KERNEL is not None
        and _P2_ACCUM_DYNAMIC_KERNEL is not None
        and _P2_DECODE_OWNERSHIP_KERNEL is not None
        and _P2_COMPUTE_VISIBILITY_KERNEL is not None
        and _P2_COMPUTE_QUERY_VISIBILITY_KERNEL is not None
        and _P2_COMPUTE_SENSOR_GATE_KERNEL is not None
    ):
        return (
            _P2_COPY_BASE_GRID_KERNEL,
            _P2_ACCUM_STATIC_AGENTS_KERNEL,
            _P2_CLAMP_BUFFER_KERNEL,
            _P2_BROADCAST_TEMPLATE_KERNEL,
            _P2_ZERO_BUFFER_KERNEL,
            _P2_ACCUM_DYNAMIC_KERNEL,
            _P2_DECODE_OWNERSHIP_KERNEL,
            _P2_COMPUTE_VISIBILITY_KERNEL,
            _P2_COMPUTE_QUERY_VISIBILITY_KERNEL,
            _P2_COMPUTE_SENSOR_GATE_KERNEL,
        )

    with _active_cuda_context():
        _P2_OCC_VIS_CUDA_MODULE = _CudaSourceModule(
            _P2_OCC_VIS_CUDA_SOURCE, no_extern_c=True
        )
        _P2_COPY_BASE_GRID_KERNEL = _P2_OCC_VIS_CUDA_MODULE.get_function(
            "copy_base_grid"
        )
        _P2_ACCUM_STATIC_AGENTS_KERNEL = _P2_OCC_VIS_CUDA_MODULE.get_function(
            "accumulate_static_agents"
        )
        _P2_CLAMP_BUFFER_KERNEL = _P2_OCC_VIS_CUDA_MODULE.get_function("clamp_buffer")
        _P2_BROADCAST_TEMPLATE_KERNEL = _P2_OCC_VIS_CUDA_MODULE.get_function(
            "broadcast_static_template"
        )
        _P2_ZERO_BUFFER_KERNEL = _P2_OCC_VIS_CUDA_MODULE.get_function("zero_buffer")
        _P2_ACCUM_DYNAMIC_KERNEL = _P2_OCC_VIS_CUDA_MODULE.get_function(
            "accumulate_dynamic_occupancy_ownership"
        )
        _P2_DECODE_OWNERSHIP_KERNEL = _P2_OCC_VIS_CUDA_MODULE.get_function(
            "decode_ownership_from_keys"
        )
        _P2_COMPUTE_VISIBILITY_KERNEL = _P2_OCC_VIS_CUDA_MODULE.get_function(
            "compute_visibility_boundary_rays"
        )
        _P2_COMPUTE_QUERY_VISIBILITY_KERNEL = _P2_OCC_VIS_CUDA_MODULE.get_function(
            "compute_visibility_queries"
        )
        _P2_COMPUTE_SENSOR_GATE_KERNEL = _P2_OCC_VIS_CUDA_MODULE.get_function(
            "compute_sensor_gate_tensor"
        )
    return (
        _P2_COPY_BASE_GRID_KERNEL,
        _P2_ACCUM_STATIC_AGENTS_KERNEL,
        _P2_CLAMP_BUFFER_KERNEL,
        _P2_BROADCAST_TEMPLATE_KERNEL,
        _P2_ZERO_BUFFER_KERNEL,
        _P2_ACCUM_DYNAMIC_KERNEL,
        _P2_DECODE_OWNERSHIP_KERNEL,
        _P2_COMPUTE_VISIBILITY_KERNEL,
        _P2_COMPUTE_QUERY_VISIBILITY_KERNEL,
        _P2_COMPUTE_SENSOR_GATE_KERNEL,
    )


@dataclass(frozen=True)
class CanonicalEntropyInputs:
    """Packed, contiguous canonical tensors for GPU OCE evaluation."""

    time_step: int
    prediction_length: int
    steps_per_prediction: int
    num_trajectories: int
    num_targets: int
    num_modes_total: int
    grid: np.ndarray
    origin: np.ndarray
    resolution: float
    trajectory_states: np.ndarray
    target_agent_ids: np.ndarray
    scored_target_indices: np.ndarray
    target_sizes: np.ndarray
    mode_offsets: np.ndarray
    mode_to_target: np.ndarray
    mode_priors_packed: np.ndarray
    mode_probabilities_packed: np.ndarray
    target_states_packed: np.ndarray
    static_agent_states: np.ndarray
    static_agent_sizes: np.ndarray
    confusion_offsets: np.ndarray
    confusion_packed: np.ndarray
    target_risk_durations: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )


@dataclass
class QuantizedVisibilityCache:
    """Per-scene visibility cache keyed by (step_idx, ego_cell_x, ego_cell_y)."""

    rows_by_key: dict
    hits: int = 0
    misses: int = 0


@dataclass
class OCESceneInputs:
    """Build-phase scene payload reusable across many rollout batches."""

    time_step: int
    prediction_length: int
    steps_per_prediction: int
    grid: np.ndarray
    origin: np.ndarray
    resolution: float
    num_targets: int
    num_modes_total: int
    target_agent_ids: np.ndarray
    scored_target_indices: np.ndarray
    target_sizes: np.ndarray
    mode_offsets: np.ndarray
    mode_to_target: np.ndarray
    mode_priors_packed: np.ndarray
    mode_probabilities_packed: np.ndarray
    target_states_packed: np.ndarray
    static_agent_states: np.ndarray
    static_agent_sizes: np.ndarray
    confusion_offsets: np.ndarray
    confusion_packed: np.ndarray
    visibility_cache: QuantizedVisibilityCache
    target_risk_durations: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )


def _coerce_scene_inputs(scene):
    if isinstance(scene, OCESceneInputs):
        return scene
    required = [f.name for f in fields(OCESceneInputs)]
    if all(hasattr(scene, name) for name in required):
        return OCESceneInputs(**{name: getattr(scene, name) for name in required})
    return scene


@dataclass(frozen=True)
class SceneOccupancyDeviceBuffers:
    """Device-side occupancy buffers reusable across visibility queries."""

    static_grid_d: object
    dynamic_sum_d: object
    ownership_d: object
    target_states_d: object
    target_sizes_d: object
    mode_to_target_d: object
    width: int
    height: int


@dataclass(frozen=True)
class VisibilityPrecomputeResult:
    """Precomputed visibility payload for packed OCE execution."""

    visibility_tensor: np.ndarray
    anchors_xy_packed: np.ndarray
    anchor_offsets: np.ndarray
    mode_to_target: np.ndarray
    footprint_cache_hit: bool
    map_cache_hit: bool


def _resolved_target_risk_durations(scene_or_canonical):
    """Resolve clipped per-target risk durations, defaulting to the full horizon."""

    num_targets = max(int(getattr(scene_or_canonical, "num_targets", 0)), 0)
    horizon = max(int(getattr(scene_or_canonical, "prediction_length", 0)), 0)
    if num_targets <= 0:
        return np.zeros((0,), dtype=np.int32)
    if horizon <= 0:
        return np.zeros((num_targets,), dtype=np.int32)

    raw = getattr(scene_or_canonical, "target_risk_durations", None)
    if raw is None:
        return np.full((num_targets,), horizon, dtype=np.int32)

    durations = np.asarray(raw, dtype=np.int32).reshape(-1)
    if durations.size == 0:
        return np.full((num_targets,), horizon, dtype=np.int32)
    if durations.size != num_targets:
        raise ValueError(
            "target_risk_durations length must match num_targets: "
            f"expected {num_targets}, got {durations.size}."
        )

    return np.ascontiguousarray(
        np.clip(durations, 0, horizon).astype(np.int32, copy=False)
    )


@dataclass(frozen=True)
class PartitionProbabilityResult:
    """Partition probability tensors derived from visibility."""

    unseen_mass: np.ndarray
    seen_mass: np.ndarray
    total_mass: np.ndarray
    occlusion_mass: np.ndarray
    target_agent_ids: np.ndarray
    max_mass_residual: float


@dataclass(frozen=True)
class OCEAccumulationResult:
    """Reduced OCE scores accumulated from partition probabilities."""

    total_entropies: np.ndarray
    best_traj_index: int
    per_target_total: np.ndarray
    per_target_oce: np.ndarray
    per_target_entropy: np.ndarray
    per_target_occ: np.ndarray
    per_target_state: np.ndarray
    per_target_mi: np.ndarray
    per_target_within: np.ndarray
    step_oce: np.ndarray
    step_entropy: np.ndarray
    step_within: np.ndarray
    step_state: np.ndarray
    step_mi: np.ndarray
    i_times_per_target: np.ndarray
    target_agent_ids: np.ndarray
    target_indices: np.ndarray
    entropy_space_norm: str = "exact_partition-info"
    lambda_e: float = 0.0


@dataclass(frozen=True)
class OCEDeviceAccumulationBuffers:
    """Device-resident OCE reductions used by MPPI integration."""

    per_target_entropy_d: object
    per_target_oce_d: object
    per_target_state_d: object
    per_target_mi_d: object
    per_target_occ_d: object
    per_target_total_d: object
    total_entropies_d: object
    i_times_per_target: np.ndarray
    step_oce_d: object | None = None
    step_entropy_d: object | None = None
    step_within_d: object | None = None
    step_state_d: object | None = None
    step_mi_d: object | None = None
    entropy_space_norm: str = "exact_partition-info"
    lambda_e: float = 0.0


@dataclass(frozen=True)
class OCEScoreResult:
    """Score-phase outputs for a reusable OCE scene."""

    canonical: CanonicalEntropyInputs
    accumulation: OCEAccumulationResult | None
    visibility_tensor: np.ndarray | None
    cache_hits: int
    cache_misses: int
    num_unique_queries: int
    device_accumulation: OCEDeviceAccumulationBuffers | None = None


@dataclass(frozen=True)
class P4PipelineResult:
    """End-to-end P4 standalone pipeline outputs."""

    visibility_d: object
    partitions: PartitionProbabilityResult
    accumulation: OCEAccumulationResult


def _safe_attr(obj, name, default):
    if obj is None:
        return default
    return getattr(obj, name, default)


def _empty_partition_probability_result(canonical):
    """Create a zero-filled partition payload for fast CUDA e2e pipeline mode."""

    n = int(canonical.num_trajectories)
    a = int(canonical.num_targets)
    t = int(canonical.prediction_length)
    return PartitionProbabilityResult(
        unseen_mass=np.zeros((n, a, t), dtype=np.float32),
        seen_mass=np.zeros((n, a, t, t), dtype=np.float32),
        total_mass=np.zeros((n, a, t), dtype=np.float32),
        occlusion_mass=np.zeros((n, a, t), dtype=np.float32),
        target_agent_ids=np.ascontiguousarray(
            canonical.target_agent_ids.astype(np.int32)
        ),
        max_mass_residual=float("nan"),
    )


def _resolved_scored_target_indices(canonical):
    if hasattr(canonical, "scored_target_indices"):
        return np.asarray(canonical.scored_target_indices, dtype=np.int32).reshape(-1)

    return np.arange(int(canonical.num_targets), dtype=np.int32)


def _filter_accumulation_to_scored_targets(canonical, accumulation):
    """Restrict returned scoring outputs to the configured scored targets only."""

    scored_target_indices = _resolved_scored_target_indices(canonical)
    num_trajectories = int(canonical.num_trajectories)
    horizon = int(canonical.prediction_length)

    if scored_target_indices.size == 0:
        return OCEAccumulationResult(
            total_entropies=np.zeros((num_trajectories,), dtype=np.float32),
            best_traj_index=0,
            per_target_total=np.zeros((num_trajectories, 0), dtype=np.float32),
            per_target_oce=np.zeros((num_trajectories, 0), dtype=np.float32),
            per_target_entropy=np.zeros((num_trajectories, 0), dtype=np.float32),
            per_target_occ=np.zeros((num_trajectories, 0), dtype=np.float32),
            per_target_state=np.zeros((num_trajectories, 0), dtype=np.float32),
            per_target_mi=np.zeros((num_trajectories, 0), dtype=np.float32),
            per_target_within=np.zeros((num_trajectories, 0), dtype=np.float32),
            step_oce=np.zeros((num_trajectories, 0, horizon), dtype=np.float32),
            step_entropy=np.zeros((num_trajectories, 0, horizon), dtype=np.float32),
            step_within=np.zeros((num_trajectories, 0, horizon), dtype=np.float32),
            step_state=np.zeros((num_trajectories, 0, horizon), dtype=np.float32),
            step_mi=np.zeros((num_trajectories, 0, horizon), dtype=np.float32),
            i_times_per_target=np.zeros((0, horizon), dtype=np.float32),
            target_agent_ids=np.zeros((0,), dtype=np.int32),
            target_indices=np.zeros((0,), dtype=np.int32),
            entropy_space_norm=str(
                getattr(accumulation, "entropy_space_norm", "exact_partition-info")
            ),
            lambda_e=float(getattr(accumulation, "lambda_e", 0.0)),
        )

    per_target_total = np.ascontiguousarray(
        np.asarray(accumulation.per_target_total, dtype=np.float32)[
            :, scored_target_indices
        ]
    )
    per_target_oce = np.ascontiguousarray(
        np.asarray(accumulation.per_target_oce, dtype=np.float32)[
            :, scored_target_indices
        ]
    )
    per_target_entropy = np.ascontiguousarray(
        np.asarray(accumulation.per_target_entropy, dtype=np.float32)[
            :, scored_target_indices
        ]
    )
    per_target_occ = np.ascontiguousarray(
        np.asarray(accumulation.per_target_occ, dtype=np.float32)[
            :, scored_target_indices
        ]
    )
    per_target_state = np.ascontiguousarray(
        np.asarray(accumulation.per_target_state, dtype=np.float32)[
            :, scored_target_indices
        ]
    )
    per_target_mi = np.ascontiguousarray(
        np.asarray(accumulation.per_target_mi, dtype=np.float32)[
            :, scored_target_indices
        ]
    )
    per_target_within = np.ascontiguousarray(
        np.asarray(accumulation.per_target_within, dtype=np.float32)[
            :, scored_target_indices
        ]
    )
    step_oce = np.ascontiguousarray(
        np.asarray(accumulation.step_oce, dtype=np.float32)[:, scored_target_indices, :]
    )
    step_entropy = np.ascontiguousarray(
        np.asarray(accumulation.step_entropy, dtype=np.float32)[
            :, scored_target_indices, :
        ]
    )
    step_state = np.ascontiguousarray(
        np.asarray(accumulation.step_state, dtype=np.float32)[
            :, scored_target_indices, :
        ]
    )
    step_within = np.ascontiguousarray(
        np.asarray(accumulation.step_within, dtype=np.float32)[
            :, scored_target_indices, :
        ]
    )
    step_mi = np.ascontiguousarray(
        np.asarray(accumulation.step_mi, dtype=np.float32)[:, scored_target_indices, :]
    )
    i_times_per_target = np.ascontiguousarray(
        np.asarray(accumulation.i_times_per_target, dtype=np.float32)[
            scored_target_indices, :
        ]
    )

    total_entropies = (
        np.ascontiguousarray(np.sum(per_target_total, axis=1, dtype=np.float32))
        if num_trajectories > 0
        else np.zeros((0,), dtype=np.float32)
    )
    best_traj_index = int(np.argmin(total_entropies)) if total_entropies.size > 0 else 0

    return OCEAccumulationResult(
        total_entropies=total_entropies,
        best_traj_index=best_traj_index,
        per_target_total=per_target_total,
        per_target_oce=per_target_oce,
        per_target_entropy=per_target_entropy,
        per_target_occ=per_target_occ,
        per_target_state=per_target_state,
        per_target_mi=per_target_mi,
        per_target_within=per_target_within,
        step_oce=step_oce,
        step_entropy=step_entropy,
        step_within=step_within,
        step_state=step_state,
        step_mi=step_mi,
        i_times_per_target=i_times_per_target,
        target_agent_ids=np.ascontiguousarray(
            np.asarray(canonical.target_agent_ids, dtype=np.int32)[
                scored_target_indices
            ]
        ),
        target_indices=np.ascontiguousarray(scored_target_indices.astype(np.int32)),
        entropy_space_norm=str(
            getattr(accumulation, "entropy_space_norm", "exact_partition-info")
        ),
        lambda_e=float(getattr(accumulation, "lambda_e", 0.0)),
    )


def _materialize_results_from_score_result(
    canonical,
    score_result,
    include_visibility=False,
):
    """Build CPU-compatible result records from a reusable-scene score phase."""

    n = int(canonical.num_trajectories)
    t = int(canonical.prediction_length)

    accum = score_result.accumulation
    if accum is None:
        raise ValueError("score_result does not include host-materialized accumulation")
    target_ids = np.asarray(accum.target_agent_ids, dtype=np.int32)
    target_indices = np.asarray(accum.target_indices, dtype=np.int32)
    a = int(target_indices.size)
    mode_offsets = np.asarray(canonical.mode_offsets, dtype=np.int32)
    per_target_total = np.asarray(accum.per_target_total, dtype=np.float32)
    per_target_entropy = np.asarray(accum.per_target_entropy, dtype=np.float32)
    per_target_oce = np.asarray(accum.per_target_oce, dtype=np.float32)
    per_target_occ = np.asarray(accum.per_target_occ, dtype=np.float32)
    per_target_state = np.asarray(accum.per_target_state, dtype=np.float32)
    per_target_mi = np.asarray(accum.per_target_mi, dtype=np.float32)
    per_target_within = np.asarray(accum.per_target_within, dtype=np.float32)
    i_times = np.asarray(accum.i_times_per_target, dtype=np.float32)
    step_oce = np.asarray(accum.step_oce, dtype=np.float32)
    step_entropy = np.asarray(accum.step_entropy, dtype=np.float32)
    step_within = np.asarray(accum.step_within, dtype=np.float32)
    step_state = np.asarray(accum.step_state, dtype=np.float32)
    step_mi = np.asarray(accum.step_mi, dtype=np.float32)
    entropy_space_norm = str(
        getattr(accum, "entropy_space_norm", "exact_partition-info")
    )
    lambda_e = float(getattr(accum, "lambda_e", 0.0))
    info_entropy_space = entropy_space_norm.endswith("partition-info")
    target_risk_durations = _resolved_target_risk_durations(canonical)
    visibility = (
        np.asarray(score_result.visibility_tensor, dtype=np.float32)
        if include_visibility
        else None
    )

    results = [[] for _ in range(n)]
    for traj_idx in range(n):
        for target_idx in range(a):
            original_target_idx = int(target_indices[target_idx])
            agent_id = int(target_ids[target_idx])
            m0 = int(mode_offsets[original_target_idx])
            m1 = int(mode_offsets[original_target_idx + 1])
            risk_duration = int(target_risk_durations[original_target_idx])
            step_len = max(min(risk_duration, t) - 1, 0)
            step_oce_k = (
                np.ascontiguousarray(step_oce[traj_idx, target_idx, 1 : 1 + step_len])
                if step_oce.shape == (n, a, t)
                else np.zeros((step_len,), dtype=np.float32)
            )
            step_entropy_k = (
                np.ascontiguousarray(
                    step_entropy[traj_idx, target_idx, 1 : 1 + step_len]
                )
                if step_entropy.shape == (n, a, t)
                else np.zeros((step_len,), dtype=np.float32)
            )
            step_state_k = (
                np.ascontiguousarray(step_state[traj_idx, target_idx, 1 : 1 + step_len])
                if step_state.shape == (n, a, t)
                else np.zeros((step_len,), dtype=np.float32)
            )
            step_within_k = (
                np.ascontiguousarray(
                    step_within[traj_idx, target_idx, 1 : 1 + step_len]
                )
                if step_within.shape == (n, a, t)
                else np.zeros((step_len,), dtype=np.float32)
            )
            step_mi_k = (
                np.ascontiguousarray(step_mi[traj_idx, target_idx, 1 : 1 + step_len])
                if step_mi.shape == (n, a, t)
                else np.zeros((step_len,), dtype=np.float32)
            )
            if info_entropy_space:
                step_epistemic_k = np.ascontiguousarray(
                    np.maximum(step_entropy_k - step_within_k, 0.0).astype(
                        np.float32, copy=False
                    )
                )
                scalar_entropy = float(per_target_entropy[traj_idx, target_idx])
                scalar_within = float(per_target_within[traj_idx, target_idx])
                scalar_epistemic = float(np.sum(step_epistemic_k, dtype=np.float64))
                scalar_raw_epistemic = float(
                    np.sum(step_entropy_k - step_within_k, dtype=np.float64)
                )
            else:
                step_epistemic_k = step_entropy_k
                scalar_entropy = float(per_target_entropy[traj_idx, target_idx])
                scalar_within = float(per_target_within[traj_idx, target_idx])
                scalar_epistemic = scalar_entropy
                scalar_raw_epistemic = scalar_entropy

            info = {
                "J_total": float(per_target_total[traj_idx, target_idx]),
                "J_OCE": float(per_target_oce[traj_idx, target_idx]),
                "J_entropy": scalar_entropy,
                "J_epistemic": scalar_epistemic,
                "J_within": scalar_within,
                "J_MI": float(per_target_mi[traj_idx, target_idx]),
                "J_occ": float(per_target_occ[traj_idx, target_idx]),
                "J_state": float(per_target_state[traj_idx, target_idx]),
                "J_state_score": (
                    float(per_target_oce[traj_idx, target_idx])
                    if info_entropy_space
                    else float("nan")
                ),
                "I_times": np.ascontiguousarray(i_times[target_idx, :risk_duration]),
                "J_OCE_results": step_oce_k,
                "J_entropy_results": step_entropy_k,
                "J_epistemic_results": step_epistemic_k,
                "J_within_results": step_within_k,
                "J_MI_results": step_mi_k,
                "J_state_results": step_state_k,
                "J_occ_results": np.zeros((risk_duration,), dtype=np.float32),
                "risk_duration": risk_duration,
                "lambda_e": lambda_e,
            }
            if info_entropy_space:
                info.update(
                    {
                        "H_state": scalar_entropy,
                        "A_state": scalar_within,
                        "E_state": scalar_epistemic,
                        "raw_E_state": scalar_raw_epistemic,
                        "H_state_results": step_entropy_k,
                        "A_state_results": step_within_k,
                        "E_state_results": step_epistemic_k,
                        "raw_E_state_results": np.ascontiguousarray(
                            (step_entropy_k - step_within_k).astype(
                                np.float32, copy=False
                            )
                        ),
                    }
                )
            if include_visibility and visibility is not None:
                info["visibility_matrix"] = np.ascontiguousarray(
                    visibility[traj_idx, :risk_duration, m0:m1]
                )

            results[traj_idx].append(
                {
                    "trajectory_index": traj_idx,
                    "agent_id": agent_id,
                    "score": float(per_target_total[traj_idx, target_idx]),
                    "info": info,
                    "debug_plot": None,
                }
            )
    return results


def _update_debug_episode_metrics(
    time_step,
    results,
    debug_episode_reset=False,
):
    """Maintain CPU-compatible debug histories for episode summary plotting."""

    if not hasattr(_cpu_eval, "__episode_oce_sums_by_trajectory"):
        return

    oce_hist = _cpu_eval.__episode_oce_sums_by_trajectory
    occ_hist = _cpu_eval.__episode_occ_sums_by_trajectory
    entropy_hist = _cpu_eval.__episode_entropy_sums_by_trajectory
    step_hist = _cpu_eval.__episode_step_by_trajectory

    if (
        debug_episode_reset
        or len(oce_hist) == 0
        or len(occ_hist) == 0
        or len(entropy_hist) == 0
        or len(step_hist) == 0
    ):
        oce_hist = [[] for _ in range(len(results))]
        occ_hist = [[] for _ in range(len(results))]
        entropy_hist = [[] for _ in range(len(results))]
        step_hist = [[] for _ in range(len(results))]
    else:
        target_len = max(
            len(results),
            len(oce_hist),
            len(occ_hist),
            len(entropy_hist),
            len(step_hist),
        )
        if len(oce_hist) < target_len:
            oce_hist.extend([[] for _ in range(target_len - len(oce_hist))])
        if len(occ_hist) < target_len:
            occ_hist.extend([[] for _ in range(target_len - len(occ_hist))])
        if len(entropy_hist) < target_len:
            entropy_hist.extend([[] for _ in range(target_len - len(entropy_hist))])
        if len(step_hist) < target_len:
            step_hist.extend([[] for _ in range(target_len - len(step_hist))])

    for traj_idx, trajectory_results in enumerate(results):
        step_oce_sum = _cpu_eval._sum_metric_over_trajectory_results(
            trajectory_results, "J_OCE_results"
        )
        step_occ_sum = _cpu_eval._sum_metric_over_trajectory_results(
            trajectory_results, "J_occ_results"
        )
        step_entropy_sum = _cpu_eval._sum_metric_over_trajectory_results(
            trajectory_results, "J_entropy_results"
        )
        oce_hist[traj_idx].append(step_oce_sum)
        occ_hist[traj_idx].append(step_occ_sum)
        entropy_hist[traj_idx].append(step_entropy_sum)
        step_hist[traj_idx].append(time_step)

    _cpu_eval.__episode_oce_sums_by_trajectory = oce_hist
    _cpu_eval.__episode_occ_sums_by_trajectory = occ_hist
    _cpu_eval.__episode_entropy_sums_by_trajectory = entropy_hist
    _cpu_eval.__episode_step_by_trajectory = step_hist


class CudaBufferCache:
    """Reusable device buffers for end-to-end CUDA execution."""

    def __init__(self):
        self._capacity = {}
        self._buffers = {}

    def reserve_data(self, name, required_capacity, zero_fill=False):
        with _active_cuda_context():
            current = self._capacity.get(name, 0)
            buf = self._buffers.get(name)

            if buf is None or current < required_capacity:
                if buf is not None:
                    try:
                        buf.free()
                    except Exception:
                        pass

                # make required_capacity a multiple of 32 bytes to reduce churn
                required_capacity = ((required_capacity + 31) // 32) * 32
                buf = _cuda.mem_alloc(required_capacity)
                self._buffers[name] = buf
                self._capacity[name] = required_capacity
            else:
                required_capacity = (
                    (required_capacity + 3) // 4 * 4
                )  # round up to multiple of 4 bytes for memset_d32

            if zero_fill:
                _cuda.memset_d32(buf, 0, required_capacity // 4)

            return buf

    def reserve_array(self, name, arr):
        buf = self.reserve_data(name, arr.nbytes, zero_fill=False)
        with _active_cuda_context():
            _cuda.memcpy_htod(buf, arr)
        return buf

    def get(self, name):
        return self._buffers.get(name)

    def release(self):
        with _active_cuda_context():
            for name, buf in list(self._buffers.items()):
                if buf is not None:
                    try:
                        buf.free()
                    except Exception:
                        print(f"Warning: failed to free CUDA buffer '{name}'")
                        pass
                self._buffers[name] = None
                self._capacity[name] = 0

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.release()
        except Exception:
            pass


global _cuda_buffer_cache
_cuda_buffer_cache = CudaBufferCache()


def _path_length(path):
    if hasattr(path, "x"):
        return len(path.x)
    return len(path)
    # arr = np.asarray(path)
    # if arr.ndim >= 2:
    #     return int(arr.shape[0])
    # raise ValueError("Unsupported trajectory type; expected object with .x/.y or array")


def _extract_trajectory_state(path, prediction_length, steps_per_prediction=1):
    out = np.zeros((prediction_length, EGO_STATE_DIM), dtype=np.float32)
    step_stride = max(int(steps_per_prediction), 1)

    if hasattr(path, "x") and hasattr(path, "y"):
        x = np.asarray(path.x, dtype=np.float32)
        y = np.asarray(path.y, dtype=np.float32)
        if x.shape[0] < prediction_length or y.shape[0] < prediction_length:
            raise ValueError("Trajectory shorter than prediction length")
        sample_idx = np.minimum(
            np.arange(prediction_length, dtype=np.int32) * step_stride,
            min(x.shape[0], y.shape[0]) - 1,
        )
        out[:, 0] = x[sample_idx]
        out[:, 1] = y[sample_idx]
        if hasattr(path, "yaw"):
            yaw = np.asarray(path.yaw, dtype=np.float32)
            if yaw.shape[0] > 0:
                yaw_idx = np.minimum(sample_idx, yaw.shape[0] - 1)
                out[:, 2] = yaw[yaw_idx]
        return out

    arr = np.asarray(path, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < prediction_length or arr.shape[1] < 2:
        raise ValueError("Unsupported trajectory array shape; expected (T, >=2)")
    sample_idx = np.minimum(
        np.arange(prediction_length, dtype=np.int32) * step_stride,
        arr.shape[0] - 1,
    )
    out[:, 0] = arr[sample_idx, 0]
    out[:, 1] = arr[sample_idx, 1]
    if arr.shape[1] >= 3:
        out[:, 2] = arr[sample_idx, 2]
    return out


def _normalize_modes(agent_modes):
    if isinstance(agent_modes, np.ndarray):
        if agent_modes.ndim == 3:
            return [agent_modes[idx] for idx in range(agent_modes.shape[0])]
        if agent_modes.ndim == 2:
            return [agent_modes]
    return list(agent_modes)


def _extract_target_state(mode, prediction_length):
    arr = np.asarray(mode, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < prediction_length or arr.shape[1] < 2:
        raise ValueError("Prediction mode must have shape (T, >=2)")

    out = np.zeros((prediction_length, TARGET_STATE_DIM), dtype=np.float32)
    out[:, 0] = arr[:prediction_length, 0]
    out[:, 1] = arr[:prediction_length, 1]
    if arr.shape[1] >= 3:
        # Match the CPU visibility path, which treats the last prediction column
        # as heading when rasterizing target footprints.
        out[:, 2] = arr[:prediction_length, -1]
    return out


def _normalize_initial_belief(initial_belief, num_modes):
    if num_modes <= 0:
        return np.zeros((0,), dtype=np.float32)

    if initial_belief is None:
        return np.full((num_modes,), 1.0 / num_modes, dtype=np.float32)

    arr = np.asarray(initial_belief, dtype=np.float32)
    if arr.ndim == 0:
        vec = np.full((num_modes,), float(arr), dtype=np.float32)
    else:
        if arr.ndim >= 2:
            arr = arr[0]
        vec = np.zeros((num_modes,), dtype=np.float32)
        count = min(num_modes, int(arr.shape[0]))
        if count > 0:
            vec[:count] = arr[:count]

    np.clip(vec, 0.0, np.finfo(np.float32).max, out=vec)
    total = float(np.sum(vec))
    if total <= 0.0:
        vec.fill(1.0 / num_modes)
    else:
        vec /= total
    return vec


def _expand_mode_probability_series(initial_belief, num_modes, prediction_length):
    """Expand per-mode beliefs into a dense (K, M) matrix using CPU semantics."""

    if num_modes <= 0 or prediction_length <= 0:
        return np.zeros(
            (max(prediction_length, 0), max(num_modes, 0)), dtype=np.float32
        )

    out = np.zeros((prediction_length, num_modes), dtype=np.float32)
    if initial_belief is None:
        out.fill(1.0)
        return out

    try:
        belief_array = np.asarray(initial_belief, dtype=np.float32)
    except (TypeError, ValueError):
        out.fill(1.0)
        return out

    for step_idx in range(prediction_length):
        if belief_array.ndim == 0:
            out[step_idx, :] = float(belief_array)
            continue

        if belief_array.ndim >= 2:
            if belief_array.shape[0] == 0:
                continue
            row = belief_array[min(step_idx, belief_array.shape[0] - 1)]
        else:
            row = belief_array

        row = np.asarray(row, dtype=np.float32).reshape(-1)
        count = min(num_modes, int(row.shape[0]))
        if count > 0:
            out[step_idx, :count] = row[:count]

    return out


def _extract_agent_pose(agent):
    """Extract (x, y, heading) from an agent-like object when available."""

    if agent is None:
        return None

    try:
        state, _ = agent.get_state()
    except AttributeError:
        return None

    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.size < 2:
        return None

    x = float(state[0])
    y = float(state[1])
    heading = 0.0
    if hasattr(agent, "get_speed_and_heading"):
        try:
            _, heading = agent.get_speed_and_heading()
            heading = float(heading)
        except Exception:
            heading = 0.0
    elif state.size >= 4:
        heading = float(np.arctan2(state[3], state[2]))

    return np.array([x, y, heading], dtype=np.float32)


def _is_static_agent(agent):
    """Return True when an agent reports static behavior."""

    if agent is None:
        return False

    is_static_fn = getattr(agent, "is_static", None)
    if not callable(is_static_fn):
        return False

    try:
        return bool(is_static_fn())
    except Exception:
        return False


def _compute_prediction_length(trajectories, predictions):
    if len(trajectories) == 0:
        return 0

    traj_min = min(len(path.x) for path in trajectories)
    pred_min = min(
        len(mode)
        for agent_modes in predictions.values()
        if agent_modes is not None
        for mode in agent_modes
    )

    # mode_lengths = []
    # for agent_modes in predictions.values():
    #     for mode in _normalize_modes(agent_modes):
    #         arr = np.asarray(mode)
    #         if arr.ndim < 2:
    #             continue
    #         mode_lengths.append(int(arr.shape[0]))

    # if len(mode_lengths) == 0:
    #     return 0

    return int(min(traj_min, pred_min))


def _compute_prediction_length_from_predictions(predictions):
    if not predictions:
        return 0

    mode_lengths = [
        len(mode)
        for agent_modes in predictions.values()
        if agent_modes is not None
        for mode in agent_modes
    ]
    if not mode_lengths:
        return 0
    return int(min(mode_lengths))


def _empty_oce_scene_inputs(
    time_step,
    steps_per_prediction,
    grid,
    origin,
    resolution,
):
    """Return an empty scene payload when no targets are available to score."""

    return OCESceneInputs(
        time_step=int(time_step),
        prediction_length=0,
        steps_per_prediction=int(steps_per_prediction),
        grid=grid,
        origin=origin,
        resolution=float(resolution),
        num_targets=0,
        num_modes_total=0,
        target_agent_ids=np.zeros((0,), dtype=np.int32),
        scored_target_indices=np.zeros((0,), dtype=np.int32),
        target_sizes=np.zeros((0, 2), dtype=np.float32),
        target_risk_durations=np.zeros((0,), dtype=np.int32),
        mode_offsets=np.zeros((1,), dtype=np.int32),
        mode_to_target=np.zeros((0,), dtype=np.int32),
        mode_priors_packed=np.zeros((0,), dtype=np.float32),
        mode_probabilities_packed=np.zeros((0, 0), dtype=np.float32),
        target_states_packed=np.zeros((0, 0, TARGET_STATE_DIM), dtype=np.float32),
        static_agent_states=np.zeros((0, TARGET_STATE_DIM), dtype=np.float32),
        static_agent_sizes=np.zeros((0, 2), dtype=np.float32),
        confusion_offsets=np.zeros((1,), dtype=np.int32),
        confusion_packed=np.zeros((0,), dtype=np.float32),
        visibility_cache=QuantizedVisibilityCache(rows_by_key={}),
    )


def build_oce_scene_inputs(
    time_step,
    grid,
    origin,
    resolution,
    agents,
    predictions,
    probabilities,
    prediction_interval,
    prediction_length=None,
    score_agent_ids: Optional[Iterable[int]] = None,
    dt=_cpu_eval.DEFAULT_TIME_TICK,
):
    """Build scene-static OCE payload reusable across many rollout batches."""

    steps_per_prediction = max(int(prediction_interval / dt), 1)
    grid_np = np.ascontiguousarray(np.asarray(grid, dtype=np.float32))
    origin_np = np.ascontiguousarray(np.asarray(origin, dtype=np.float32).reshape(2))
    resolved_scored_ids = set(
        _cpu_eval._resolve_scored_agent_ids(
            predictions=predictions,
            probabilities=probabilities,
            score_agent_ids=score_agent_ids,
        )
    )

    if not predictions:
        return _empty_oce_scene_inputs(
            time_step=time_step,
            steps_per_prediction=steps_per_prediction,
            grid=grid_np,
            origin=origin_np,
            resolution=resolution,
        )

    available_prediction_length = _compute_prediction_length_from_predictions(
        predictions
    )
    if prediction_length is None:
        prediction_length = available_prediction_length
    else:
        prediction_length = min(int(prediction_length), available_prediction_length)

    if prediction_length <= 0:
        return _empty_oce_scene_inputs(
            time_step=time_step,
            steps_per_prediction=steps_per_prediction,
            grid=grid_np,
            origin=origin_np,
            resolution=resolution,
        )

    target_agent_ids = []
    scored_target_indices = []
    target_sizes = []
    target_risk_durations = []
    mode_offsets = [0]
    mode_to_target = []
    mode_priors = []
    mode_probabilities = []
    packed_target_states = []
    confusion_offsets = [0]
    packed_confusion = []
    target_risk_durations = []
    static_agent_states = []
    static_agent_sizes = []

    if isinstance(agents, dict):
        for agent_id, agent in agents.items():
            agent_size = _cpu_eval._extract_agent_size(agent)
            if agent_size is None:
                continue
            has_prediction = (
                agent_id in predictions and predictions.get(agent_id) is not None
            )
            has_probability = (
                agent_id in probabilities and probabilities.get(agent_id) is not None
            )
            include_background = _is_static_agent(agent) or (
                not has_prediction or not has_probability
            )
            if not include_background:
                continue
            pose = _extract_agent_pose(agent)
            if pose is None:
                continue
            static_agent_states.append(pose)
            static_agent_sizes.append(np.asarray(agent_size, dtype=np.float32))

    for agent_id, agent_modes in predictions.items():
        agent = agents.get(agent_id) if isinstance(agents, dict) else None
        agent_size = _cpu_eval._extract_agent_size(agent)
        if agent_size is None:
            continue

        initial_belief = probabilities.get(agent_id)
        if initial_belief is None:
            continue

        mode_list = _normalize_modes(agent_modes)
        if len(mode_list) == 0:
            continue

        target_states = []
        for mode in mode_list:
            target_states.append(_extract_target_state(mode, prediction_length))

        num_modes = len(target_states)
        if num_modes == 0:
            continue

        target_agent_ids.append(int(agent_id))
        target_sizes.append(np.asarray(agent_size, dtype=np.float32))
        target_idx = len(target_agent_ids) - 1
        target_risk_durations.append(
            _cpu_eval._resolve_agent_risk_duration(
                agent,
                prediction_length=prediction_length,
                steps_per_prediction=steps_per_prediction,
            )
        )
        if agent_id in resolved_scored_ids:
            scored_target_indices.append(target_idx)
        packed_target_states.extend(target_states)
        mode_to_target.extend([target_idx] * num_modes)

        belief = _normalize_initial_belief(initial_belief, num_modes)
        mode_priors.extend(belief.tolist())
        belief_series = _expand_mode_probability_series(
            initial_belief,
            num_modes,
            prediction_length,
        )
        mode_probabilities.extend(
            [belief_series[:, mode_idx] for mode_idx in range(num_modes)]
        )
        mode_offsets.append(mode_offsets[-1] + num_modes)

        target_states_np = np.ascontiguousarray(np.stack(target_states, axis=0))
        confusion = _cpu_eval.calculate_confusion_matrix(
            target_states_np, prediction_length=prediction_length
        )
        confusion_flat = np.ascontiguousarray(confusion.ravel())
        packed_confusion.extend(confusion_flat.tolist())
        confusion_offsets.append(confusion_offsets[-1] + confusion_flat.size)

    num_targets = len(target_agent_ids)
    num_modes_total = int(mode_offsets[-1]) if mode_offsets else 0

    target_states_packed = (
        np.ascontiguousarray(
            np.stack(packed_target_states, axis=0).astype(np.float32, copy=False)
        )
        if num_modes_total > 0
        else np.zeros((0, prediction_length, TARGET_STATE_DIM), dtype=np.float32)
    )
    mode_probabilities_packed = (
        np.ascontiguousarray(
            np.stack(mode_probabilities, axis=0).astype(np.float32, copy=False)
        )
        if num_modes_total > 0
        else np.zeros((0, prediction_length), dtype=np.float32)
    )
    target_sizes_np = (
        np.ascontiguousarray(
            np.stack(target_sizes, axis=0).astype(np.float32, copy=False)
        )
        if num_targets > 0
        else np.zeros((0, 2), dtype=np.float32)
    )
    static_agent_states_np = (
        np.ascontiguousarray(
            np.stack(static_agent_states, axis=0).astype(np.float32, copy=False)
        )
        if static_agent_states
        else np.zeros((0, TARGET_STATE_DIM), dtype=np.float32)
    )
    static_agent_sizes_np = (
        np.ascontiguousarray(
            np.stack(static_agent_sizes, axis=0).astype(np.float32, copy=False)
        )
        if static_agent_sizes
        else np.zeros((0, 2), dtype=np.float32)
    )

    return OCESceneInputs(
        time_step=int(time_step),
        prediction_length=int(prediction_length),
        steps_per_prediction=steps_per_prediction,
        grid=grid_np,
        origin=origin_np,
        resolution=float(resolution),
        num_targets=num_targets,
        num_modes_total=num_modes_total,
        target_agent_ids=np.ascontiguousarray(
            np.asarray(target_agent_ids, dtype=np.int32).reshape(-1)
        ),
        scored_target_indices=np.ascontiguousarray(
            np.asarray(scored_target_indices, dtype=np.int32).reshape(-1)
        ),
        target_sizes=target_sizes_np,
        target_risk_durations=np.ascontiguousarray(
            np.asarray(target_risk_durations, dtype=np.int32).reshape(-1)
        ),
        mode_offsets=np.ascontiguousarray(np.asarray(mode_offsets, dtype=np.int32)),
        mode_to_target=np.ascontiguousarray(np.asarray(mode_to_target, dtype=np.int32)),
        mode_priors_packed=np.ascontiguousarray(
            np.asarray(mode_priors, dtype=np.float32).reshape(-1)
        ),
        mode_probabilities_packed=mode_probabilities_packed,
        target_states_packed=target_states_packed,
        static_agent_states=static_agent_states_np,
        static_agent_sizes=static_agent_sizes_np,
        confusion_offsets=np.ascontiguousarray(
            np.asarray(confusion_offsets, dtype=np.int32)
        ),
        confusion_packed=np.ascontiguousarray(
            np.asarray(packed_confusion, dtype=np.float32).reshape(-1)
        ),
        visibility_cache=QuantizedVisibilityCache(rows_by_key={}),
    )


def _pack_trajectory_states(trajectories, prediction_length, steps_per_prediction):
    """Materialize packed ego rollout states for OCE scoring."""

    num_trajectories = len(trajectories)
    if prediction_length <= 0 or num_trajectories == 0:
        return np.zeros((num_trajectories, 0, EGO_STATE_DIM), dtype=np.float32)

    trajectory_states = np.zeros(
        (num_trajectories, prediction_length, EGO_STATE_DIM), dtype=np.float32
    )
    for traj_idx, path in enumerate(trajectories):
        trajectory_states[traj_idx] = _extract_trajectory_state(
            path,
            prediction_length,
            steps_per_prediction=steps_per_prediction,
        )
    return trajectory_states


def pack_rollout_states(trajectories, prediction_length, steps_per_prediction):
    """Public wrapper for packing rollout states into `(N, T, 3)` arrays."""

    return _pack_trajectory_states(
        trajectories=trajectories,
        prediction_length=prediction_length,
        steps_per_prediction=steps_per_prediction,
    )


def build_canonical_entropy_inputs_from_scene(
    scene,
    num_trajectories=None,
    trajectory_states=None,
    trajectory_heading_index=None,
):
    """Combine a reusable OCE scene payload with ego trajectory states.

    Packed geometric trajectories use ``[x, y, heading]`` and therefore
    heading index 2.  Four-value MPPI rollouts use ``[x, y, speed, heading]``
    and therefore heading index 3.  ``trajectory_heading_index`` is available
    for callers with another explicit layout.
    """

    if not isinstance(scene, OCESceneInputs):
        raise TypeError("scene must be an OCESceneInputs instance")

    if trajectory_states is None:
        if num_trajectories is None:
            raise ValueError(
                "num_trajectories must be provided when trajectory_states is omitted"
            )
        traj_np = np.empty(
            (num_trajectories, int(scene.prediction_length), EGO_STATE_DIM),
            dtype=np.float32,
        )
    else:
        traj_np = np.ascontiguousarray(np.asarray(trajectory_states, dtype=np.float32))

    if traj_np.ndim != 3 or traj_np.shape[2] < EGO_STATE_DIM:
        raise ValueError("trajectory_states must have shape (N, T, >=3)")
    if scene.prediction_length > 0 and traj_np.shape[1] < scene.prediction_length:
        raise ValueError(
            "trajectory_states horizon is shorter than the scene prediction length"
        )

    if trajectory_states is not None:
        if trajectory_heading_index is None:
            trajectory_heading_index = 3 if traj_np.shape[2] >= 4 else 2
        trajectory_heading_index = int(trajectory_heading_index)
        if not 0 <= trajectory_heading_index < traj_np.shape[2]:
            raise ValueError("trajectory_heading_index is outside the state layout")
        canonical_states = np.empty(
            (traj_np.shape[0], scene.prediction_length, EGO_STATE_DIM),
            dtype=np.float32,
        )
        canonical_states[:, :, 0:2] = traj_np[:, : scene.prediction_length, 0:2]
        canonical_states[:, :, 2] = traj_np[
            :, : scene.prediction_length, trajectory_heading_index
        ]
        traj_np = np.ascontiguousarray(canonical_states)
    else:
        traj_np = traj_np[:, : scene.prediction_length, :EGO_STATE_DIM]
    num_trajectories = int(traj_np.shape[0])

    return CanonicalEntropyInputs(
        time_step=int(scene.time_step),
        prediction_length=int(scene.prediction_length),
        steps_per_prediction=int(scene.steps_per_prediction),
        num_trajectories=num_trajectories,
        num_targets=int(scene.num_targets),
        num_modes_total=int(scene.num_modes_total),
        grid=np.ascontiguousarray(scene.grid),
        origin=np.ascontiguousarray(scene.origin),
        resolution=float(scene.resolution),
        trajectory_states=traj_np,
        target_agent_ids=np.ascontiguousarray(scene.target_agent_ids),
        scored_target_indices=np.ascontiguousarray(scene.scored_target_indices),
        target_sizes=np.ascontiguousarray(scene.target_sizes),
        mode_offsets=np.ascontiguousarray(scene.mode_offsets),
        mode_to_target=np.ascontiguousarray(scene.mode_to_target),
        mode_priors_packed=np.ascontiguousarray(scene.mode_priors_packed),
        mode_probabilities_packed=np.ascontiguousarray(scene.mode_probabilities_packed),
        target_states_packed=np.ascontiguousarray(scene.target_states_packed),
        static_agent_states=np.ascontiguousarray(scene.static_agent_states),
        static_agent_sizes=np.ascontiguousarray(scene.static_agent_sizes),
        confusion_offsets=np.ascontiguousarray(scene.confusion_offsets),
        confusion_packed=np.ascontiguousarray(scene.confusion_packed),
        target_risk_durations=np.ascontiguousarray(
            _resolved_target_risk_durations(scene)
        ),
    )


def build_canonical_entropy_inputs(
    time_step,
    grid,
    origin,
    resolution,
    trajectories,
    agents,
    predictions,
    probabilities,
    prediction_interval,
    score_agent_ids: Optional[Iterable[int]] = None,
    dt=_cpu_eval.DEFAULT_TIME_TICK,
):
    """Build packed canonical arrays for standalone GPU OCE evaluation."""

    prediction_length = _compute_prediction_length(trajectories, predictions)
    scene = build_oce_scene_inputs(
        time_step=time_step,
        grid=grid,
        origin=origin,
        resolution=resolution,
        agents=agents,
        predictions=predictions,
        probabilities=probabilities,
        prediction_interval=prediction_interval,
        prediction_length=prediction_length,
        score_agent_ids=score_agent_ids,
        dt=dt,
    )

    if scene.prediction_length <= 0 or len(trajectories) == 0:
        return build_canonical_entropy_inputs_from_scene(
            scene=scene,
            num_trajectories=len(trajectories),
            trajectory_states=np.zeros(
                (len(trajectories), 0, EGO_STATE_DIM), dtype=np.float32
            ),
        )

    trajectory_states = _pack_trajectory_states(
        trajectories=trajectories,
        prediction_length=scene.prediction_length,
        steps_per_prediction=scene.steps_per_prediction,
    )

    return build_canonical_entropy_inputs_from_scene(
        scene=scene,
        num_trajectories=len(trajectories),
        trajectory_states=trajectory_states,
    )


def _warn_on_ownership_conflicts(
    conflict_counts_per_t,
    first_conflict_cell_per_t,
    width,
    first_conflict_owner_a_per_t=None,
    first_conflict_owner_b_per_t=None,
):
    """Emit a warning summarizing ownership conflicts detected during occupancy build."""

    counts = np.asarray(conflict_counts_per_t, dtype=np.int64).reshape(-1)
    first_cells = np.asarray(first_conflict_cell_per_t, dtype=np.int64).reshape(-1)
    owner_a = (
        np.asarray(first_conflict_owner_a_per_t, dtype=np.int64).reshape(-1)
        if first_conflict_owner_a_per_t is not None
        else None
    )
    owner_b = (
        np.asarray(first_conflict_owner_b_per_t, dtype=np.int64).reshape(-1)
        if first_conflict_owner_b_per_t is not None
        else None
    )
    total_conflicts = int(np.sum(counts))
    if total_conflicts <= 0:
        return

    active_steps = np.flatnonzero(counts > 0)
    sample_chunks = []
    for step_idx in active_steps[:5]:
        first_cell = (
            int(first_cells[step_idx]) if step_idx < first_cells.shape[0] else -1
        )
        step_count = int(counts[step_idx])
        owner_suffix = ""
        if owner_a is not None and owner_b is not None:
            oa = int(owner_a[step_idx]) if step_idx < owner_a.shape[0] else -1
            ob = int(owner_b[step_idx]) if step_idx < owner_b.shape[0] else -1
            if oa >= 0 and ob >= 0:
                owner_suffix = f" owners=({oa},{ob})"
        if first_cell >= 0 and width > 0:
            x = first_cell % int(width)
            y = first_cell // int(width)
            sample_chunks.append(
                f"t={int(step_idx)} count={step_count} first_cell=({x},{y}){owner_suffix}"
            )
        else:
            sample_chunks.append(f"t={int(step_idx)} count={step_count}{owner_suffix}")

    sample_suffix = "; ".join(sample_chunks) if sample_chunks else "no sample cells"
    warnings.warn(
        "Ownership conflicts detected while building GPU occupancy ownership map: "
        f"{total_conflicts} conflicted cells across {int(active_steps.size)} timesteps. "
        f"Samples: {sample_suffix}",
        RuntimeWarning,
        stacklevel=2,
    )


def _build_scene_occupancy_cuda(
    scene,
    cuda_cache,
    probability_threshold=0.0,
    debug=False,
):
    """Build reusable scene occupancy/ownership buffers on the device."""

    if not isinstance(scene, (CanonicalEntropyInputs, OCESceneInputs)):
        raise TypeError("scene must be CanonicalEntropyInputs or OCESceneInputs")

    (
        _,
        accum_static_kernel,
        clamp_kernel,
        _,
        _,
        accum_dynamic_kernel,
        decode_ownership_kernel,
        _,
        _,
        _,
    ) = _get_occ_vis_cuda_kernels()

    grid_h = np.ascontiguousarray(np.asarray(scene.grid, dtype=np.float32))
    target_states_h = np.ascontiguousarray(
        np.asarray(scene.target_states_packed, dtype=np.float32)
    )
    target_sizes_h = np.ascontiguousarray(
        np.asarray(scene.target_sizes, dtype=np.float32)
    )
    mode_to_target_h = np.ascontiguousarray(
        np.asarray(scene.mode_to_target, dtype=np.int32)
    )
    mode_probs_h = np.ascontiguousarray(
        np.asarray(scene.mode_probabilities_packed, dtype=np.float32)
    )
    static_states_h = np.ascontiguousarray(
        np.asarray(scene.static_agent_states, dtype=np.float32)
    )
    static_sizes_h = np.ascontiguousarray(
        np.asarray(scene.static_agent_sizes, dtype=np.float32)
    )

    horizon = int(scene.prediction_length)
    total_modes = int(scene.num_modes_total)
    num_targets = int(scene.num_targets)
    height, width = grid_h.shape
    num_cells = int(height * width)

    if mode_probs_h.shape != (total_modes, horizon):
        raise ValueError(
            f"mode_probabilities_packed shape {mode_probs_h.shape} does not match {(total_modes, horizon)}"
        )

    owner_min_contrib_eps = _cuda_owner_min_contrib_eps()

    grid_d = cuda_cache.reserve_array("grid", grid_h.reshape(-1))
    target_states_d = cuda_cache.reserve_array(
        "target_states", target_states_h.reshape(-1)
    )
    target_sizes_d = cuda_cache.reserve_array(
        "target_sizes", target_sizes_h.reshape(-1)
    )
    mode_to_target_d = cuda_cache.reserve_array(
        "mode_to_target", mode_to_target_h.reshape(-1)
    )
    mode_probs_d = cuda_cache.reserve_array("mode_probs", mode_probs_h.reshape(-1))

    static_states_d = None
    static_sizes_d = None
    if static_states_h.size > 0:
        static_states_d = cuda_cache.reserve_array(
            "static_states", static_states_h.reshape(-1)
        )
        static_sizes_d = cuda_cache.reserve_array(
            "static_sizes", static_sizes_h.reshape(-1)
        )

    ownership_d = cuda_cache.reserve_data(
        "ownership", np.dtype(np.int32).itemsize * horizon * num_cells, zero_fill=True
    )
    dynamic_sum_d = cuda_cache.reserve_data(
        "dynamic_sum",
        np.dtype(np.float32).itemsize * horizon * num_cells,
        zero_fill=True,
    )
    owner_key_d = cuda_cache.reserve_data(
        "owner_key", np.dtype(np.uint64).itemsize * horizon * num_cells, zero_fill=True
    )

    block_flat = 256
    grid_copy = ((num_cells + block_flat - 1) // block_flat, 1, 1)

    if static_states_d is not None and static_states_h.shape[0] > 0:
        block_static = min(block_flat, int(static_states_h.shape[0]))
        grid_static = (
            (int(static_states_h.shape[0] + block_static - 1) // block_static),
            1,
            1,
        )
        accum_static_kernel(
            grid_d,
            static_states_d,
            static_sizes_d,
            np.int32(static_states_h.shape[0]),
            np.int32(width),
            np.int32(height),
            np.float32(scene.origin[0]),
            np.float32(scene.origin[1]),
            np.float32(scene.resolution),
            block=(block_static, 1, 1),
            grid=grid_static,
        )
        clamp_kernel(
            grid_d,
            np.int32(num_cells),
            block=(block_flat, 1, 1),
            grid=grid_copy,
        )

    total_occ_values = horizon * num_cells
    grid_occ = ((total_occ_values + block_flat - 1) // block_flat, 1, 1)

    total_mode_steps = total_modes * horizon
    block_mode_steps = min(block_flat, max(total_mode_steps, 1))
    grid_mode_steps = (
        (total_mode_steps + block_mode_steps - 1) // block_mode_steps,
        1,
        1,
    )

    conflict_counts_h = None
    first_conflict_cell_h = None
    first_conflict_owner_a_h = None
    first_conflict_owner_b_h = None
    conflict_mask_arg = np.uintp(0)
    conflict_counts_arg = np.uintp(0)
    first_conflict_cell_arg = np.uintp(0)
    first_conflict_owner_a_arg = np.uintp(0)
    first_conflict_owner_b_arg = np.uintp(0)

    if debug and num_targets > 1:
        conflict_counts_h = np.zeros((horizon,), dtype=np.int32)
        first_conflict_cell_h = np.full((horizon,), -1, dtype=np.int32)
        first_conflict_owner_a_h = np.full((horizon,), -1, dtype=np.int32)
        first_conflict_owner_b_h = np.full((horizon,), -1, dtype=np.int32)
        conflict_counts_d = cuda_cache.reserve_array(
            "conflict_counts", conflict_counts_h
        )
        first_conflict_cell_d = cuda_cache.reserve_array(
            "first_conflict_cell", first_conflict_cell_h
        )
        first_conflict_owner_a_d = cuda_cache.reserve_array(
            "first_conflict_owner_a", first_conflict_owner_a_h
        )
        first_conflict_owner_b_d = cuda_cache.reserve_array(
            "first_conflict_owner_b", first_conflict_owner_b_h
        )
        conflict_mask_d = cuda_cache.reserve_data(
            "conflict_mask",
            np.dtype(np.int32).itemsize * horizon * num_cells,
            zero_fill=True,
        )
        conflict_mask_arg = conflict_mask_d
        conflict_counts_arg = conflict_counts_d
        first_conflict_cell_arg = first_conflict_cell_d
        first_conflict_owner_a_arg = first_conflict_owner_a_d
        first_conflict_owner_b_arg = first_conflict_owner_b_d

    accum_dynamic_kernel(
        dynamic_sum_d,
        owner_key_d,
        target_states_d,
        target_sizes_d,
        mode_to_target_d,
        mode_probs_d,
        np.int32(total_modes),
        np.int32(horizon),
        np.int32(width),
        np.int32(height),
        np.float32(scene.origin[0]),
        np.float32(scene.origin[1]),
        np.float32(scene.resolution),
        np.float32(probability_threshold),
        np.float32(owner_min_contrib_eps),
        conflict_mask_arg,
        conflict_counts_arg,
        first_conflict_cell_arg,
        first_conflict_owner_a_arg,
        first_conflict_owner_b_arg,
        block=(block_mode_steps, 1, 1),
        grid=grid_mode_steps,
    )

    decode_ownership_kernel(
        owner_key_d,
        ownership_d,
        np.int32(total_occ_values),
        np.float32(owner_min_contrib_eps),
        block=(block_flat, 1, 1),
        grid=grid_occ,
    )

    if conflict_counts_h is not None:
        _cuda.memcpy_dtoh(conflict_counts_h, conflict_counts_arg)
        _cuda.memcpy_dtoh(first_conflict_cell_h, first_conflict_cell_arg)
        _cuda.memcpy_dtoh(first_conflict_owner_a_h, first_conflict_owner_a_arg)
        _cuda.memcpy_dtoh(first_conflict_owner_b_h, first_conflict_owner_b_arg)
        _warn_on_ownership_conflicts(
            conflict_counts_per_t=conflict_counts_h,
            first_conflict_cell_per_t=first_conflict_cell_h,
            width=width,
            first_conflict_owner_a_per_t=first_conflict_owner_a_h,
            first_conflict_owner_b_per_t=first_conflict_owner_b_h,
        )

    return SceneOccupancyDeviceBuffers(
        static_grid_d=grid_d,
        dynamic_sum_d=dynamic_sum_d,
        ownership_d=ownership_d,
        target_states_d=target_states_d,
        target_sizes_d=target_sizes_d,
        mode_to_target_d=mode_to_target_d,
        width=width,
        height=height,
    )


@_numba_njit(cache=True)
def _quantize_visibility_cells_numba(states, origin_x, origin_y, resolution):
    """Return flattened quantized ego cells for rollout visibility queries."""

    num_rollouts, horizon, _ = states.shape
    total = num_rollouts * horizon
    cell_x = np.empty((total,), dtype=np.int32)
    cell_y = np.empty((total,), dtype=np.int32)

    idx = 0
    for rollout_idx in range(num_rollouts):
        for step_idx in range(horizon):
            x = states[rollout_idx, step_idx, 0]
            y = states[rollout_idx, step_idx, 1]
            cell_x[idx] = int(np.floor((x - origin_x) / resolution))
            cell_y[idx] = int(np.floor((y - origin_y) / resolution))
            idx += 1

    return cell_x, cell_y


@_numba_njit(cache=True)
def _build_query_states_from_unique_keys_numba(
    unique_keys, origin_x, origin_y, resolution
):
    """Build query states from quantized keys using cell-center coordinates."""

    query_states = np.zeros((unique_keys.shape[0], 3), dtype=np.float32)
    for idx in range(unique_keys.shape[0]):
        query_states[idx, 0] = (
            origin_x + (np.float32(unique_keys[idx, 1]) + 0.5) * resolution
        )
        query_states[idx, 1] = (
            origin_y + (np.float32(unique_keys[idx, 2]) + 0.5) * resolution
        )
    return query_states


def _deduplicate_quantized_visibility_keys(
    cell_x_flat,
    cell_y_flat,
    num_rollouts,
    horizon,
):
    """Deduplicate `(step_idx, cell_x, cell_y)` using a 1D packed-key fast path."""

    total = int(num_rollouts * horizon)
    if total == 0:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((num_rollouts, horizon), dtype=np.int64),
        )

    cell_x_flat = np.ascontiguousarray(
        np.asarray(cell_x_flat, dtype=np.int32).reshape(total)
    )
    cell_y_flat = np.ascontiguousarray(
        np.asarray(cell_y_flat, dtype=np.int32).reshape(total)
    )

    min_x = int(np.min(cell_x_flat))
    max_x = int(np.max(cell_x_flat))
    min_y = int(np.min(cell_y_flat))
    max_y = int(np.max(cell_y_flat))
    range_x = int(max_x - min_x + 1)
    range_y = int(max_y - min_y + 1)
    max_packed = (int(horizon) * range_x * range_y) - 1

    if max_packed <= np.iinfo(np.int64).max:
        step_idx = np.arange(total, dtype=np.int64) % int(horizon)
        packed = (
            step_idx * np.int64(range_x)
            + (cell_x_flat.astype(np.int64, copy=False) - np.int64(min_x))
        ) * np.int64(range_y) + (
            cell_y_flat.astype(np.int64, copy=False) - np.int64(min_y)
        )
        unique_packed, inverse = np.unique(packed, return_inverse=True)

        unique_keys = np.empty((unique_packed.shape[0], 3), dtype=np.int32)
        packed_work = unique_packed.copy()
        unique_keys[:, 2] = (
            np.mod(packed_work, np.int64(range_y)).astype(np.int32, copy=False) + min_y
        )
        packed_work //= np.int64(range_y)
        unique_keys[:, 1] = (
            np.mod(packed_work, np.int64(range_x)).astype(np.int32, copy=False) + min_x
        )
        packed_work //= np.int64(range_x)
        unique_keys[:, 0] = packed_work.astype(np.int32, copy=False)
    else:
        step_idx = np.arange(total, dtype=np.int32) % int(horizon)
        keys = np.empty((total, 3), dtype=np.int32)
        keys[:, 0] = step_idx
        keys[:, 1] = cell_x_flat
        keys[:, 2] = cell_y_flat
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

    return np.ascontiguousarray(unique_keys), inverse.reshape(num_rollouts, horizon)


def _build_quantized_visibility_queries(ego_states, origin, resolution):
    """Deduplicate rollout states by (step_idx, quantized ego cell)."""

    states = np.ascontiguousarray(np.asarray(ego_states, dtype=np.float32))
    if states.ndim != 3 or states.shape[2] < 2:
        raise ValueError("ego_states must have shape (N, T, >=2)")

    num_rollouts, horizon, _ = states.shape
    origin_x = float(origin[0])
    origin_y = float(origin[1])
    resolution_f = float(resolution)

    if _NUMBA_AVAILABLE:
        cell_x_flat, cell_y_flat = _quantize_visibility_cells_numba(
            states,
            origin_x,
            origin_y,
            resolution_f,
        )
    else:
        cell_x_flat = (
            np.floor((states[:, :, 0] - origin_x) / resolution_f)
            .astype(np.int32)
            .reshape(-1)
        )
        cell_y_flat = (
            np.floor((states[:, :, 1] - origin_y) / resolution_f)
            .astype(np.int32)
            .reshape(-1)
        )

    unique_keys, inverse = _deduplicate_quantized_visibility_keys(
        cell_x_flat=cell_x_flat,
        cell_y_flat=cell_y_flat,
        num_rollouts=num_rollouts,
        horizon=horizon,
    )

    if _NUMBA_AVAILABLE:
        query_states = _build_query_states_from_unique_keys_numba(
            unique_keys,
            np.float32(origin_x),
            np.float32(origin_y),
            np.float32(resolution_f),
        )
    else:
        query_states = np.zeros((unique_keys.shape[0], 3), dtype=np.float32)
        query_states[:, 0] = (
            origin_x + (unique_keys[:, 1].astype(np.float32) + 0.5) * resolution_f
        )
        query_states[:, 1] = (
            origin_y + (unique_keys[:, 2].astype(np.float32) + 0.5) * resolution_f
        )

    return unique_keys, inverse, query_states


def _launch_visibility_tensor_cuda(
    scene,
    ego_states_d,
    num_trajectories,
    cuda_cache,
    probability_threshold=0.0,
    eps=1e-6,
    use_boundary_cells=False,
    debug=False,
    ego_state_stride=EGO_STATE_DIM,
):
    """Launch the full `(trajectory, step, mode)` visibility kernel on device."""

    if not isinstance(scene, (CanonicalEntropyInputs, OCESceneInputs)):
        raise TypeError("scene must be CanonicalEntropyInputs or OCESceneInputs")

    num_trajectories = int(num_trajectories)
    ego_state_stride = int(ego_state_stride)
    if num_trajectories < 0:
        raise ValueError("num_trajectories must be non-negative")
    if ego_state_stride < 2:
        raise ValueError("ego_state_stride must be at least 2")

    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        visibility_kernel,
        _,
        _,
    ) = _get_occ_vis_cuda_kernels()

    horizon = int(scene.prediction_length)
    total_modes = int(scene.num_modes_total)
    num_targets = int(scene.num_targets)
    scene_buffers = _build_scene_occupancy_cuda(
        scene=scene,
        cuda_cache=cuda_cache,
        probability_threshold=probability_threshold,
        debug=debug,
    )
    visibility_d = cuda_cache.reserve_data(
        "visibility",
        np.dtype(np.float32).itemsize * num_trajectories * horizon * total_modes,
        zero_fill=True,
    )

    block_flat = 256
    total_visibility = num_trajectories * horizon * total_modes
    block_visibility = min(block_flat, max(total_visibility, 1))
    grid_visibility = (
        (total_visibility + block_visibility - 1) // block_visibility,
        1,
        1,
    )
    visibility_kernel(
        scene_buffers.static_grid_d,
        scene_buffers.dynamic_sum_d,
        scene_buffers.ownership_d,
        ego_states_d,
        scene_buffers.target_states_d,
        scene_buffers.target_sizes_d,
        scene_buffers.mode_to_target_d,
        visibility_d,
        np.int32(num_trajectories),
        np.int32(horizon),
        np.int32(total_modes),
        np.int32(num_targets),
        np.int32(scene_buffers.width),
        np.int32(scene_buffers.height),
        np.float32(scene.origin[0]),
        np.float32(scene.origin[1]),
        np.float32(scene.resolution),
        np.int32(ego_state_stride),
        np.float32(eps),
        np.int32(1 if use_boundary_cells else 0),
        block=(block_visibility, 1, 1),
        grid=grid_visibility,
    )
    return visibility_d


def _precompute_visibility_tensor_cuda(
    canonical,
    cuda_cache,
    probability_threshold=0.0,
    eps=1e-6,
    use_boundary_cells=False,
    debug=False,
):
    """Compute visibility via on-device occupancy construction and boundary-ray tracing."""

    if not isinstance(canonical, CanonicalEntropyInputs):
        raise TypeError("canonical must be a CanonicalEntropyInputs instance")

    ego_states_h = np.ascontiguousarray(
        np.asarray(canonical.trajectory_states, dtype=np.float32)
    )

    num_trajectories = int(canonical.num_trajectories)
    ego_states_d = cuda_cache.reserve_array("ego_states", ego_states_h.reshape(-1))
    return _launch_visibility_tensor_cuda(
        scene=canonical,
        ego_states_d=ego_states_d,
        num_trajectories=num_trajectories,
        cuda_cache=cuda_cache,
        probability_threshold=probability_threshold,
        eps=eps,
        use_boundary_cells=use_boundary_cells,
        debug=debug,
        ego_state_stride=EGO_STATE_DIM,
    )


def precompute_visibility_tensor_device(
    scene,
    ego_states_d,
    num_trajectories,
    cuda_cache=None,
    probability_threshold=0.0,
    eps=1e-6,
    use_boundary_cells=False,
    debug=False,
    ego_state_stride=EGO_STATE_DIM,
):
    """Precompute visibility directly from device-resident ego states."""

    if not isinstance(scene, (CanonicalEntropyInputs, OCESceneInputs)):
        raise TypeError("scene must be CanonicalEntropyInputs or OCESceneInputs")
    if cuda_cache is None:
        cuda_cache = _cuda_buffer_cache

    return _launch_visibility_tensor_cuda(
        scene=scene,
        ego_states_d=ego_states_d,
        num_trajectories=num_trajectories,
        cuda_cache=cuda_cache,
        probability_threshold=probability_threshold,
        eps=eps,
        use_boundary_cells=use_boundary_cells,
        debug=debug,
        ego_state_stride=ego_state_stride,
    )


def precompute_visibility_tensor(
    canonical,
    cuda_cache,
    use_bresenham=True,
    debug=False,
):
    """Precompute visibility tensor V[n, t, m] for packed trajectories/modes."""

    if not isinstance(canonical, CanonicalEntropyInputs):
        raise TypeError("canonical must be a CanonicalEntropyInputs instance")

    if canonical.num_modes_total <= 0 or canonical.prediction_length <= 0:
        return None

    visibility_d = None
    if use_bresenham:
        visibility_d = _precompute_visibility_tensor_cuda(
            canonical, cuda_cache, debug=debug
        )
    else:
        raise NotImplementedError("Non-Bresenham visibility not implemented")

    return visibility_d


def _compute_visibility_queries_cuda(
    scene,
    query_keys,
    query_states,
    cuda_cache,
    use_boundary_cells=False,
    debug=False,
):
    """Compute per-mode visibility rows for unique quantized rollout queries."""

    if not isinstance(scene, OCESceneInputs):
        raise TypeError("scene must be an OCESceneInputs instance")

    query_keys = np.ascontiguousarray(np.asarray(query_keys, dtype=np.int32))
    query_states = np.ascontiguousarray(np.asarray(query_states, dtype=np.float32))
    if query_keys.ndim != 2 or query_keys.shape[1] != 3:
        raise ValueError("query_keys must have shape (Q, 3)")
    if query_states.ndim != 2 or query_states.shape[1] < 2:
        raise ValueError("query_states must have shape (Q, >=2)")

    num_queries = int(query_keys.shape[0])
    total_modes = int(scene.num_modes_total)
    if num_queries <= 0 or total_modes <= 0:
        return np.zeros((0, total_modes), dtype=np.float32)

    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        query_visibility_kernel,
        _,
    ) = _get_occ_vis_cuda_kernels()

    scene_buffers = _build_scene_occupancy_cuda(
        scene=scene,
        cuda_cache=cuda_cache,
        debug=debug,
    )

    query_steps_h = np.ascontiguousarray(query_keys[:, 0].reshape(-1))
    query_states_h = np.ascontiguousarray(query_states[:, :3].reshape(-1))
    query_steps_d = cuda_cache.reserve_array("query_steps", query_steps_h)
    query_states_d = cuda_cache.reserve_array("query_states", query_states_h)
    visibility_queries_d = cuda_cache.reserve_data(
        "visibility_queries",
        np.dtype(np.float32).itemsize * num_queries * total_modes,
        zero_fill=True,
    )

    block_size = 256
    total_items = num_queries * total_modes
    block_items = min(block_size, max(total_items, 1))
    grid_items = ((total_items + block_items - 1) // block_items, 1, 1)

    query_visibility_kernel(
        scene_buffers.static_grid_d,
        scene_buffers.dynamic_sum_d,
        scene_buffers.ownership_d,
        query_steps_d,
        query_states_d,
        scene_buffers.target_states_d,
        scene_buffers.target_sizes_d,
        scene_buffers.mode_to_target_d,
        visibility_queries_d,
        np.int32(num_queries),
        np.int32(scene.prediction_length),
        np.int32(total_modes),
        np.int32(scene.num_targets),
        np.int32(scene_buffers.width),
        np.int32(scene_buffers.height),
        np.float32(scene.origin[0]),
        np.float32(scene.origin[1]),
        np.float32(scene.resolution),
        np.float32(1e-6),
        np.int32(1 if use_boundary_cells else 0),
        block=(block_items, 1, 1),
        grid=grid_items,
    )

    visibility_rows = np.empty((num_queries, total_modes), dtype=np.float32)
    _cuda.memcpy_dtoh(visibility_rows, visibility_queries_d)
    return np.ascontiguousarray(visibility_rows)


def _copy_visibility_tensor_from_device(canonical, visibility_d):
    """Materialize a device visibility tensor to host for diagnostics."""

    visibility_tensor = np.empty(
        (
            int(canonical.num_trajectories),
            int(canonical.prediction_length),
            int(canonical.num_modes_total),
        ),
        dtype=np.float32,
    )
    _cuda.memcpy_dtoh(visibility_tensor, visibility_d)
    return np.ascontiguousarray(visibility_tensor)


def _copy_visibility_like_tensor_from_device(
    *,
    num_rollouts,
    horizon,
    total_modes,
    tensor_d,
):
    """Materialize a `(N, T, M)` float tensor from device memory."""
    tensor = np.empty(
        (int(num_rollouts), int(horizon), int(total_modes)),
        dtype=np.float32,
    )
    _cuda.memcpy_dtoh(tensor, tensor_d)
    return np.ascontiguousarray(tensor)


def _compute_sensor_gate_tensor_cuda(
    scene,
    ego_states_d,
    *,
    num_trajectories,
    sensor_range,
    sensor_fov,
    cuda_cache,
    ego_state_stride=EGO_STATE_DIM,
    ego_heading_index=None,
):
    """Compute per-(trajectory, step, mode) sensor coverage fractions on CUDA.

    The default heading field is index 2 for packed paths and index 3 for the
    four-value MPPI state layout, preventing speed from being used as heading.
    """
    if not isinstance(scene, OCESceneInputs):
        raise TypeError("scene must be an OCESceneInputs instance")

    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        sensor_gate_kernel,
    ) = _get_occ_vis_cuda_kernels()

    num_trajectories = int(num_trajectories)
    ego_state_stride = int(ego_state_stride)
    if ego_heading_index is None:
        ego_heading_index = (
            3 if ego_state_stride >= 4 else (2 if ego_state_stride >= 3 else -1)
        )
    ego_heading_index = int(ego_heading_index)
    if ego_heading_index < -1 or ego_heading_index >= ego_state_stride:
        raise ValueError("ego_heading_index is outside the ego state layout")
    horizon = int(scene.prediction_length)
    total_modes = int(scene.num_modes_total)
    num_targets = int(scene.num_targets)
    if num_trajectories <= 0 or horizon <= 0 or total_modes <= 0 or num_targets <= 0:
        return None

    target_states_h = np.ascontiguousarray(
        np.asarray(scene.target_states_packed, dtype=np.float32).reshape(-1)
    )
    target_sizes_h = np.ascontiguousarray(
        np.asarray(scene.target_sizes, dtype=np.float32).reshape(-1)
    )
    mode_to_target_h = np.ascontiguousarray(
        np.asarray(scene.mode_to_target, dtype=np.int32).reshape(-1)
    )
    target_states_d = cuda_cache.reserve_array(
        "sensor_gate_target_states", target_states_h
    )
    target_sizes_d = cuda_cache.reserve_array(
        "sensor_gate_target_sizes", target_sizes_h
    )
    mode_to_target_d = cuda_cache.reserve_array(
        "sensor_gate_mode_to_target", mode_to_target_h
    )

    gate_d = cuda_cache.reserve_data(
        "sensor_gate",
        np.dtype(np.float32).itemsize * num_trajectories * horizon * total_modes,
        zero_fill=True,
    )
    grid_shape = np.asarray(scene.grid).shape
    total_items = num_trajectories * horizon * total_modes
    block_items = min(256, max(total_items, 1))
    grid_items = ((total_items + block_items - 1) // block_items, 1, 1)

    sensor_gate_kernel(
        ego_states_d,
        target_states_d,
        target_sizes_d,
        mode_to_target_d,
        gate_d,
        np.int32(num_trajectories),
        np.int32(horizon),
        np.int32(total_modes),
        np.int32(num_targets),
        np.int32(int(grid_shape[1])),
        np.int32(int(grid_shape[0])),
        np.float32(scene.origin[0]),
        np.float32(scene.origin[1]),
        np.float32(scene.resolution),
        np.int32(ego_state_stride),
        np.int32(ego_heading_index),
        np.float32(sensor_range),
        np.float32(sensor_fov),
        block=(block_items, 1, 1),
        grid=grid_items,
    )
    return gate_d


def _score_oce_scene_rollouts_device_impl(
    scene,
    rollout_states_d,
    num_rollouts,
    oce_config=None,
    eps=1e-9,
    entropy_space="kde",
    discount=1.0,
    cuda_cache=None,
    materialize_host=True,
    return_visibility_tensor=False,
    rollout_state_stride=4,
    debug=False,
):
    """Score rollouts fully on device without quantization or cache reuse."""

    scene = _coerce_scene_inputs(scene)
    if not isinstance(scene, OCESceneInputs):
        raise TypeError("scene must be an OCESceneInputs instance")

    num_rollouts = int(num_rollouts)
    if num_rollouts <= 0:
        raise ValueError("num_rollouts must be non-negative")
    if rollout_states_d is None:
        raise ValueError("rollout_states_d must be provided when scoring rollouts")

    requested_entropy_space = _resolve_kde_entropy_space_request(
        entropy_space=entropy_space,
        oce_config=oce_config,
    )
    normalized_entropy_space = _normalize_kde_entropy_space(requested_entropy_space)
    if normalized_entropy_space not in (
        "exact_partition-info",
        "approximate_partition-info",
        "exact_kde-spatial",
        "approximate_kde-spatial",
        "exact_kde-sigma-point",
        "approximate_kde-sigma-point",
        "exact_kde-separability-logdet",
        "approximate_kde-separability-logdet",
        "exact_kde-separability-trace",
        "approximate_kde-separability-trace",
    ):
        raise ValueError(
            f"score_oce_scene_rollouts_device does not support '{requested_entropy_space}'. "
            "Device OCE scoring currently supports the partition-info/kde, "
            "kde-spatial, "
            "kde-sigma-point, kde-separability-logdet, and "
            "kde-separability-trace families only."
        )

    canonical = build_canonical_entropy_inputs_from_scene(
        scene=scene,
        num_trajectories=num_rollouts,
    )
    if (
        num_rollouts <= 0
        or scene.prediction_length <= 0
        or scene.num_targets <= 0
        or scene.num_modes_total <= 0
    ):
        return OCEScoreResult(
            canonical=canonical,
            accumulation=(
                _empty_oce_accumulation_result(canonical) if materialize_host else None
            ),
            visibility_tensor=(
                np.zeros(
                    (
                        num_rollouts,
                        int(scene.prediction_length),
                        int(scene.num_modes_total),
                    ),
                    dtype=np.float32,
                )
                if return_visibility_tensor
                else None
            ),
            cache_hits=0,
            cache_misses=0,
            num_unique_queries=0,
            device_accumulation=None,
        )

    if cuda_cache is None:
        cuda_cache = _cuda_buffer_cache

    visibility_d = precompute_visibility_tensor_device(
        scene=scene,
        ego_states_d=rollout_states_d,
        num_trajectories=num_rollouts,
        cuda_cache=cuda_cache,
        ego_state_stride=rollout_state_stride,
        debug=debug,
    )
    device_accumulation = accumulate_oce_from_partitions_device(
        canonical=canonical,
        oce_config=oce_config,
        visibility_d=visibility_d,
        entropy_space=normalized_entropy_space,
        eps=eps,
        discount=discount,
        cuda_cache=cuda_cache,
    )
    accumulation = (
        materialize_oce_device_accumulation(canonical, device_accumulation)
        if materialize_host
        else None
    )

    visibility_tensor = None
    if return_visibility_tensor:
        visibility_tensor = _copy_visibility_tensor_from_device(
            canonical=canonical, visibility_d=visibility_d
        )
    return OCEScoreResult(
        canonical=canonical,
        accumulation=accumulation,
        visibility_tensor=visibility_tensor,
        cache_hits=0,
        cache_misses=0,
        num_unique_queries=int(num_rollouts * scene.prediction_length),
        device_accumulation=device_accumulation,
    )


def score_oce_scene_rollouts_device(
    scene,
    rollout_states_d,
    num_rollouts,
    oce_config=None,
    eps=1e-9,
    entropy_space="kde",
    discount=1.0,
    cuda_cache=None,
    materialize_host=True,
    return_visibility_tensor=False,
    rollout_state_stride=4,
    debug=False,
):
    """Score rollouts on device with the OCE CUDA context pushed."""

    with _active_cuda_context():
        return _score_oce_scene_rollouts_device_impl(
            scene=scene,
            rollout_states_d=rollout_states_d,
            num_rollouts=num_rollouts,
            oce_config=oce_config,
            eps=eps,
            entropy_space=entropy_space,
            discount=discount,
            cuda_cache=cuda_cache,
            materialize_host=materialize_host,
            return_visibility_tensor=return_visibility_tensor,
            rollout_state_stride=rollout_state_stride,
            debug=debug,
        )


def _score_oce_scene_rollouts_gpu_impl(
    scene,
    rollout_states,
    oce_config=None,
    eps=1e-9,
    entropy_space="kde",
    discount=1.0,
    cuda_cache=None,
    materialize_host=True,
    return_visibility_tensor=True,
    debug=False,
):
    """Score host rollouts by uploading them and delegating to the device scorer."""

    scene = _coerce_scene_inputs(scene)
    if not isinstance(scene, OCESceneInputs):
        raise TypeError("scene must be an OCESceneInputs instance")

    rollout_states = np.ascontiguousarray(np.asarray(rollout_states, dtype=np.float32))
    if rollout_states.ndim != 3 or rollout_states.shape[2] < EGO_STATE_DIM:
        raise ValueError("rollout_states must have shape (N, T, >=3)")

    canonical = build_canonical_entropy_inputs_from_scene(
        scene=scene,
        num_trajectories=rollout_states.shape[0],
        trajectory_states=rollout_states,
        trajectory_heading_index=3 if rollout_states.shape[2] >= 4 else 2,
    )

    if cuda_cache is None:
        cuda_cache = _cuda_buffer_cache

    rollout_states_d = None
    if (
        canonical.num_trajectories > 0
        and canonical.prediction_length > 0
        and canonical.num_targets > 0
        and canonical.num_modes_total > 0
    ):
        # Slice to the scoring horizon before upload so the device-side trajectory
        # indexing matches the packed `(num_rollouts, prediction_length, stride)` layout.
        rollout_states_upload = np.ascontiguousarray(
            rollout_states[:, : scene.prediction_length, :]
        )
        rollout_states_d = cuda_cache.reserve_array(
            "rollout_states_score_gpu", rollout_states_upload
        )
        rollout_state_stride = int(rollout_states_upload.shape[2])
    else:
        rollout_state_stride = int(rollout_states.shape[2])

    device_result = score_oce_scene_rollouts_device(
        scene=scene,
        rollout_states_d=rollout_states_d,
        num_rollouts=int(canonical.num_trajectories),
        oce_config=oce_config,
        eps=eps,
        entropy_space=entropy_space,
        discount=discount,
        cuda_cache=cuda_cache,
        materialize_host=materialize_host,
        return_visibility_tensor=return_visibility_tensor,
        rollout_state_stride=rollout_state_stride,
        debug=debug,
    )
    return OCEScoreResult(
        canonical=canonical,
        accumulation=device_result.accumulation,
        visibility_tensor=device_result.visibility_tensor,
        cache_hits=device_result.cache_hits,
        cache_misses=device_result.cache_misses,
        num_unique_queries=device_result.num_unique_queries,
        device_accumulation=device_result.device_accumulation,
    )


def score_oce_scene_rollouts_gpu(
    scene,
    rollout_states,
    oce_config=None,
    eps=1e-9,
    entropy_space="kde",
    discount=1.0,
    cuda_cache=None,
    materialize_host=True,
    return_visibility_tensor=True,
    debug=False,
):
    """Score host rollouts with the OCE CUDA context pushed."""

    with _active_cuda_context():
        return _score_oce_scene_rollouts_gpu_impl(
            scene=scene,
            rollout_states=rollout_states,
            oce_config=oce_config,
            eps=eps,
            entropy_space=entropy_space,
            discount=discount,
            cuda_cache=cuda_cache,
            materialize_host=materialize_host,
            return_visibility_tensor=return_visibility_tensor,
            debug=debug,
        )


def _score_visibility_timeline_gpu_impl(
    scene,
    ego_states,
    *,
    sensor_range,
    sensor_fov,
    cuda_cache=None,
    eps=1e-6,
    debug=False,
):
    """Compute the sandbox visibility tensor on CUDA for a single ego path."""
    if not isinstance(scene, OCESceneInputs):
        raise TypeError("scene must be an OCESceneInputs instance")
    if not backend_ready():
        raise RuntimeError(backend_reason_unavailable())

    horizon = int(scene.prediction_length)
    total_modes = int(scene.num_modes_total)
    if horizon <= 0 or scene.num_targets <= 0 or total_modes <= 0:
        return {
            "visibility_tensor": np.zeros(
                (max(horizon, 0), max(total_modes, 0)), dtype=np.float32
            ),
            "backend": BACKEND_NAME,
            "execution_path": "cuda_visibility_tensor",
        }

    states_np = np.ascontiguousarray(np.asarray(ego_states, dtype=np.float32))
    if states_np.ndim != 2 or states_np.shape[0] < horizon or states_np.shape[1] < 2:
        raise ValueError("ego_states must have shape (prediction_length, >=2).")

    if cuda_cache is None:
        cuda_cache = _cuda_buffer_cache

    rollout_states = np.ascontiguousarray(
        states_np[:horizon].reshape(1, horizon, states_np.shape[1]),
        dtype=np.float32,
    )
    rollout_states_d = cuda_cache.reserve_array(
        "rollout_states_visibility_gpu",
        rollout_states.reshape(-1),
    )
    rollout_state_stride = int(rollout_states.shape[2])

    visibility_d = precompute_visibility_tensor_device(
        scene=scene,
        ego_states_d=rollout_states_d,
        num_trajectories=1,
        cuda_cache=cuda_cache,
        eps=eps,
        debug=debug,
        ego_state_stride=rollout_state_stride,
    )
    sensor_gate_d = _compute_sensor_gate_tensor_cuda(
        scene,
        rollout_states_d,
        num_trajectories=1,
        sensor_range=float(sensor_range),
        sensor_fov=float(sensor_fov),
        cuda_cache=cuda_cache,
        ego_state_stride=rollout_state_stride,
        ego_heading_index=(
            3 if rollout_state_stride >= 4 else (2 if rollout_state_stride >= 3 else -1)
        ),
    )

    visibility_tensor = _copy_visibility_like_tensor_from_device(
        num_rollouts=1,
        horizon=horizon,
        total_modes=total_modes,
        tensor_d=visibility_d,
    )
    if sensor_gate_d is not None:
        sensor_gate = _copy_visibility_like_tensor_from_device(
            num_rollouts=1,
            horizon=horizon,
            total_modes=total_modes,
            tensor_d=sensor_gate_d,
        )
        visibility_tensor = np.clip(visibility_tensor * sensor_gate, 0.0, 1.0)
    else:
        visibility_tensor = np.clip(visibility_tensor, 0.0, 1.0)

    return {
        "visibility_tensor": np.ascontiguousarray(visibility_tensor[0]),
        "backend": BACKEND_NAME,
        "execution_path": "cuda_visibility_tensor",
    }


def score_visibility_timeline_gpu(
    scene,
    ego_states,
    *,
    sensor_range,
    sensor_fov,
    cuda_cache=None,
    eps=1e-6,
    debug=False,
):
    """Compute the sandbox visibility tensor with the OCE CUDA context pushed."""

    with _active_cuda_context():
        return _score_visibility_timeline_gpu_impl(
            scene=scene,
            ego_states=ego_states,
            sensor_range=sensor_range,
            sensor_fov=sensor_fov,
            cuda_cache=cuda_cache,
            eps=eps,
            debug=debug,
        )


def _confusion_matrices_for_target(canonical, target_idx):
    """Unpack per-target confusion tensor from packed storage."""

    start = int(canonical.confusion_offsets[target_idx])
    end = int(canonical.confusion_offsets[target_idx + 1])
    m0 = int(canonical.mode_offsets[target_idx])
    m1 = int(canonical.mode_offsets[target_idx + 1])
    num_modes = max(m1 - m0, 0)
    T = int(canonical.prediction_length)

    if num_modes <= 0 or T <= 0:
        return np.zeros((0, 0, 0), dtype=np.float64)

    expected = T * num_modes * num_modes
    size = end - start
    if size == 0:
        eye = np.eye(num_modes, dtype=np.float64)
        return np.repeat(eye[None, :, :], T, axis=0)

    if size != expected:
        raise ValueError(
            f"Packed confusion size mismatch for target {target_idx}: expected {expected}, got {size}"
        )

    block = np.asarray(canonical.confusion_packed[start:end], dtype=np.float64)
    return block.reshape(T, num_modes, num_modes)


def _prepare_i_times(canonical, eps):
    """Build per-target/time MI terms used in state/MI weighted reductions."""

    A_targets = int(canonical.num_targets)
    T = int(canonical.prediction_length)
    target_risk_durations = _resolved_target_risk_durations(canonical)
    i_times = np.zeros((A_targets, T), dtype=np.float32)
    if A_targets <= 0 or T <= 0:
        return np.ascontiguousarray(i_times)

    for target_idx in range(A_targets):
        risk_duration = int(target_risk_durations[target_idx])
        if risk_duration <= 0:
            continue
        m0 = int(canonical.mode_offsets[target_idx])
        m1 = int(canonical.mode_offsets[target_idx + 1])
        Ma = m1 - m0
        if Ma <= 0:
            continue

        priors = np.asarray(canonical.mode_priors_packed[m0:m1], dtype=np.float64)
        priors = np.clip(priors, 0.0, np.finfo(np.float64).max)
        s = float(np.sum(priors))
        if s <= eps:
            priors[:] = 1.0 / float(Ma)
        else:
            priors /= s

        conf = _confusion_matrices_for_target(canonical, target_idx)
        for t in range(min(T, conf.shape[0], risk_duration)):
            i_times[target_idx, t] = float(
                mode_information_at_time(
                    mode_priors=priors,
                    confusion_matrix=np.asarray(conf[t], dtype=np.float64),
                    eps=eps,
                )
            )

    return np.ascontiguousarray(i_times)


def _is_kde_entropy_space(entropy_space):
    return _parse_kde_entropy_space(str(entropy_space).strip().lower()) is not None


def _normalize_kde_entropy_space(entropy_space):
    parsed = _parse_kde_entropy_space(str(entropy_space).strip().lower())
    if parsed is None:
        raise ValueError(f"Unsupported KDE entropy_space '{entropy_space}'.")
    partition_mode, base_space = parsed
    return f"{partition_mode}_{base_space}"


def _resolve_kde_entropy_space_request(entropy_space, oce_config):
    """Resolve the effective KDE space, letting config override the generic default."""

    requested = None if entropy_space is None else str(entropy_space).strip()
    configured = None
    if oce_config is not None:
        configured = getattr(oce_config, "oce_entropy_space", None)
        if configured is not None:
            configured = str(configured).strip()
            if not configured:
                configured = None

    if requested is None or requested == "":
        return configured or "kde"
    if requested.lower() in ("kde", "partition-info") and configured is not None:
        return configured
    return requested


def _resolve_spatial_sigma_entropy_spaces(partition_mode):
    """Resolve the paired KDE comparison spaces for a requested partition mode."""

    if partition_mode is None:
        return partition_mode, "kde-spatial", "kde-sigma-point"

    mode = _normalize_spatial_sigma_partition_mode(partition_mode)
    return mode, f"{mode}_kde-spatial", f"{mode}_kde-sigma-point"


def _compare_kde_spatial_and_sigma_gpu_impl(
    scene,
    rollout_states,
    oce_config=None,
    eps=1e-9,
    cuda_cache=None,
    debug=False,
    partition_mode="exact",
):
    """Compute one GPU visibility tensor and reduce it with both spatial and sigma-point forms."""

    if not isinstance(scene, OCESceneInputs):
        raise TypeError("scene must be an OCESceneInputs instance")

    rollout_states = np.ascontiguousarray(np.asarray(rollout_states, dtype=np.float32))
    if rollout_states.ndim != 3 or rollout_states.shape[2] < EGO_STATE_DIM:
        raise ValueError("rollout_states must have shape (N, T, >=3)")

    canonical = build_canonical_entropy_inputs_from_scene(
        scene=scene,
        num_trajectories=rollout_states.shape[0],
        trajectory_states=rollout_states,
    )
    if cuda_cache is None:
        cuda_cache = _cuda_buffer_cache

    rollout_upload = np.ascontiguousarray(
        rollout_states[:, : scene.prediction_length, :]
    )
    rollout_states_d = cuda_cache.reserve_array(
        "rollout_states_compare_gpu", rollout_upload
    )
    visibility_d = precompute_visibility_tensor_device(
        scene=scene,
        ego_states_d=rollout_states_d,
        num_trajectories=int(canonical.num_trajectories),
        cuda_cache=cuda_cache,
        debug=debug,
        ego_state_stride=int(rollout_upload.shape[2]),
    )
    partition_mode, spatial_space, sigma_space = _resolve_spatial_sigma_entropy_spaces(
        partition_mode
    )

    spatial_device = accumulate_oce_from_partitions_device(
        canonical=canonical,
        visibility_d=visibility_d,
        oce_config=oce_config,
        eps=eps,
        entropy_space=spatial_space,
        discount=(
            float(_safe_attr(oce_config, "oce_discount", 1.0))
            if oce_config is not None
            else 1.0
        ),
        cuda_cache=cuda_cache,
    )
    spatial = materialize_oce_device_accumulation(canonical, spatial_device)
    sigma_device = accumulate_oce_from_partitions_device(
        canonical=canonical,
        visibility_d=visibility_d,
        oce_config=oce_config,
        eps=eps,
        entropy_space=sigma_space,
        discount=(
            float(_safe_attr(oce_config, "oce_discount", 1.0))
            if oce_config is not None
            else 1.0
        ),
        cuda_cache=cuda_cache,
    )
    sigma = materialize_oce_device_accumulation(canonical, sigma_device)
    visibility_tensor = _copy_visibility_tensor_from_device(
        canonical=canonical, visibility_d=visibility_d
    )
    comparison = {
        "partition_mode": partition_mode,
        "spatial_space": _normalize_kde_entropy_space(spatial_space),
        "sigma_space": _normalize_kde_entropy_space(sigma_space),
        "spatial": spatial,
        "sigma": sigma,
        "delta_total_entropies": np.ascontiguousarray(
            np.asarray(sigma.total_entropies, dtype=np.float32)
            - np.asarray(spatial.total_entropies, dtype=np.float32)
        ),
        "delta_per_target_oce": np.ascontiguousarray(
            np.asarray(sigma.per_target_oce, dtype=np.float32)
            - np.asarray(spatial.per_target_oce, dtype=np.float32)
        ),
        "delta_per_target_entropy": np.ascontiguousarray(
            np.asarray(sigma.per_target_entropy, dtype=np.float32)
            - np.asarray(spatial.per_target_entropy, dtype=np.float32)
        ),
    }
    comparison["canonical"] = canonical
    comparison["visibility_tensor"] = visibility_tensor
    return comparison


def compare_kde_spatial_and_sigma_gpu(
    scene,
    rollout_states,
    oce_config=None,
    eps=1e-9,
    cuda_cache=None,
    debug=False,
    partition_mode="exact",
):
    """Compare OCE CUDA reducers with the OCE CUDA context pushed."""

    with _active_cuda_context():
        return _compare_kde_spatial_and_sigma_gpu_impl(
            scene=scene,
            rollout_states=rollout_states,
            oce_config=oce_config,
            eps=eps,
            cuda_cache=cuda_cache,
            debug=debug,
            partition_mode=partition_mode,
        )


def _normalize_spatial_sigma_partition_mode(partition_mode):
    mode = str(partition_mode).strip().lower()
    if mode not in ("exact", "approximate"):
        raise ValueError("partition_mode must be one of {'exact', 'approximate'}.")
    return mode


def _summarize_comparison_accumulation(accumulation):
    total_scores = np.ascontiguousarray(
        np.asarray(accumulation.total_entropies, dtype=np.float32).reshape(-1)
    )
    total_oce = np.ascontiguousarray(
        np.sum(np.asarray(accumulation.per_target_oce, dtype=np.float32), axis=1)
    )
    total_entropy = np.ascontiguousarray(
        np.sum(np.asarray(accumulation.per_target_entropy, dtype=np.float32), axis=1)
    )
    return total_scores, total_oce, total_entropy


def _summarize_public_comparison_results(total_scores, results):
    total_scores = np.ascontiguousarray(
        np.asarray(total_scores, dtype=np.float32).reshape(-1)
    )
    num_trajectories = int(total_scores.size)
    total_oce = np.zeros((num_trajectories,), dtype=np.float32)
    total_entropy = np.zeros((num_trajectories,), dtype=np.float32)

    if results is None:
        return total_scores, total_oce, total_entropy

    for traj_idx in range(min(num_trajectories, len(results))):
        oce_sum = 0.0
        entropy_sum = 0.0
        for record in results[traj_idx]:
            info = record.get("info", {}) if isinstance(record, dict) else {}
            oce_sum += float(info.get("J_OCE", 0.0))
            entropy_value = info.get("J_entropy", None)
            if entropy_value is None:
                entropy_value = float(
                    np.sum(
                        np.asarray(info.get("J_entropy_results", []), dtype=np.float64),
                        dtype=np.float64,
                    )
                )
            entropy_sum += float(entropy_value)
        total_oce[traj_idx] = float(oce_sum)
        total_entropy[traj_idx] = float(entropy_sum)

    return total_scores, total_oce, total_entropy


def _build_spatial_sigma_comparison_summary(
    *,
    backend,
    partition_mode,
    spatial_total_scores,
    sigma_total_scores,
    spatial_total_oce,
    sigma_total_oce,
    spatial_total_entropy,
    sigma_total_entropy,
    num_targets,
    num_modes_total,
    prediction_length,
):
    spatial_total_scores = np.ascontiguousarray(
        np.asarray(spatial_total_scores, dtype=np.float32).reshape(-1)
    )
    sigma_total_scores = np.ascontiguousarray(
        np.asarray(sigma_total_scores, dtype=np.float32).reshape(-1)
    )
    spatial_total_oce = np.ascontiguousarray(
        np.asarray(spatial_total_oce, dtype=np.float32).reshape(-1)
    )
    sigma_total_oce = np.ascontiguousarray(
        np.asarray(sigma_total_oce, dtype=np.float32).reshape(-1)
    )
    spatial_total_entropy = np.ascontiguousarray(
        np.asarray(spatial_total_entropy, dtype=np.float32).reshape(-1)
    )
    sigma_total_entropy = np.ascontiguousarray(
        np.asarray(sigma_total_entropy, dtype=np.float32).reshape(-1)
    )

    return {
        "backend": str(backend),
        "partition_mode": str(partition_mode),
        "spatial_space": f"{partition_mode}_kde-spatial",
        "sigma_space": f"{partition_mode}_kde-sigma-point",
        "best_spatial_index": (
            int(np.argmin(spatial_total_scores)) if spatial_total_scores.size > 0 else 0
        ),
        "best_sigma_index": (
            int(np.argmin(sigma_total_scores)) if sigma_total_scores.size > 0 else 0
        ),
        "num_targets": int(num_targets),
        "num_modes_total": int(num_modes_total),
        "prediction_length": int(prediction_length),
        "spatial_total_scores": spatial_total_scores,
        "sigma_total_scores": sigma_total_scores,
        "delta_total_scores": np.ascontiguousarray(
            sigma_total_scores - spatial_total_scores
        ),
        "spatial_total_oce": spatial_total_oce,
        "sigma_total_oce": sigma_total_oce,
        "delta_total_oce": np.ascontiguousarray(sigma_total_oce - spatial_total_oce),
        "spatial_total_entropy": spatial_total_entropy,
        "sigma_total_entropy": sigma_total_entropy,
        "delta_total_entropy": np.ascontiguousarray(
            sigma_total_entropy - spatial_total_entropy
        ),
    }


def compare_kde_spatial_and_sigma_rollouts(
    time_step,
    grid,
    origin,
    resolution,
    trajectories,
    agents,
    predictions,
    probabilities,
    prediction_interval,
    score_agent_ids: Optional[Iterable[int]] = None,
    oce_config=None,
    dt=_cpu_eval.DEFAULT_TIME_TICK,
    partition_mode="exact",
    backend="auto",
    debug=False,
):
    """Compare spatial and sigma-point forms on live rollout candidates."""

    mode = _normalize_spatial_sigma_partition_mode(partition_mode)
    backend_name = str(backend).strip().lower()
    if backend_name not in ("auto", "gpu", "cpu"):
        raise ValueError("backend must be one of {'auto', 'gpu', 'cpu'}.")

    num_trajectories = len(trajectories)
    if num_trajectories <= 0 or not predictions:
        zeros = np.zeros((num_trajectories,), dtype=np.float32)
        resolved_backend = (
            "gpu" if backend_name in ("auto", "gpu") and P4_CUDA_AVAILABLE else "cpu"
        )
        return _build_spatial_sigma_comparison_summary(
            backend=resolved_backend,
            partition_mode=mode,
            spatial_total_scores=zeros,
            sigma_total_scores=zeros.copy(),
            spatial_total_oce=zeros.copy(),
            sigma_total_oce=zeros.copy(),
            spatial_total_entropy=zeros.copy(),
            sigma_total_entropy=zeros.copy(),
            num_targets=0,
            num_modes_total=0,
            prediction_length=0,
        )

    scene = build_oce_scene_inputs(
        time_step=time_step,
        grid=grid,
        origin=origin,
        resolution=resolution,
        agents=agents,
        predictions=predictions,
        probabilities=probabilities,
        prediction_interval=prediction_interval,
        prediction_length=_compute_prediction_length(trajectories, predictions),
        score_agent_ids=score_agent_ids,
        dt=dt,
    )
    if (
        int(scene.prediction_length) <= 0
        or int(scene.num_targets) <= 0
        or int(scene.num_modes_total) <= 0
    ):
        zeros = np.zeros((num_trajectories,), dtype=np.float32)
        resolved_backend = (
            "gpu" if backend_name in ("auto", "gpu") and P4_CUDA_AVAILABLE else "cpu"
        )
        return _build_spatial_sigma_comparison_summary(
            backend=resolved_backend,
            partition_mode=mode,
            spatial_total_scores=zeros,
            sigma_total_scores=zeros.copy(),
            spatial_total_oce=zeros.copy(),
            sigma_total_oce=zeros.copy(),
            spatial_total_entropy=zeros.copy(),
            sigma_total_entropy=zeros.copy(),
            num_targets=int(scene.num_targets),
            num_modes_total=int(scene.num_modes_total),
            prediction_length=int(scene.prediction_length),
        )

    if backend_name in ("auto", "gpu") and P4_CUDA_AVAILABLE:
        rollout_len = min(
            [len(traj.x) for traj in trajectories] + [int(scene.prediction_length)]
        )
        rollout_states = pack_rollout_states(
            trajectories=trajectories,
            prediction_length=rollout_len,
            steps_per_prediction=int(scene.steps_per_prediction),
        )
        comparison = compare_kde_spatial_and_sigma_gpu(
            scene=scene,
            rollout_states=rollout_states,
            oce_config=oce_config,
            debug=debug,
            partition_mode=mode,
        )
        spatial_scores, spatial_oce, spatial_entropy = (
            _summarize_comparison_accumulation(comparison["spatial"])
        )
        sigma_scores, sigma_oce, sigma_entropy = _summarize_comparison_accumulation(
            comparison["sigma"]
        )
        summary = _build_spatial_sigma_comparison_summary(
            backend="gpu",
            partition_mode=mode,
            spatial_total_scores=spatial_scores,
            sigma_total_scores=sigma_scores,
            spatial_total_oce=spatial_oce,
            sigma_total_oce=sigma_oce,
            spatial_total_entropy=spatial_entropy,
            sigma_total_entropy=sigma_entropy,
            num_targets=int(scene.num_targets),
            num_modes_total=int(scene.num_modes_total),
            prediction_length=int(scene.prediction_length),
        )
        summary["raw_comparison"] = comparison
        return summary

    if backend_name == "gpu":
        raise RuntimeError(
            "Spatial/sigma comparison requested with backend='gpu', but PyCUDA is unavailable."
        )

    spatial_cfg = (
        copy.deepcopy(oce_config) if oce_config is not None else _cpu_eval.OCEConfig()
    )
    sigma_cfg = (
        copy.deepcopy(oce_config) if oce_config is not None else _cpu_eval.OCEConfig()
    )
    spatial_cfg.oce_entropy_space = f"{mode}_kde-spatial"
    sigma_cfg.oce_entropy_space = f"{mode}_kde-sigma-point"

    spatial_best, spatial_scores, spatial_results = (
        _cpu_eval.evaluate_trajectories_by_entropy(
            time_step=time_step,
            grid=grid,
            origin=origin,
            resolution=resolution,
            trajectories=trajectories,
            agents=agents,
            predictions=predictions,
            probabilities=probabilities,
            prediction_interval=prediction_interval,
            score_agent_ids=score_agent_ids,
            oce_config=spatial_cfg,
            dt=dt,
            debug=debug,
        )
    )
    sigma_best, sigma_scores, sigma_results = (
        _cpu_eval.evaluate_trajectories_by_entropy(
            time_step=time_step,
            grid=grid,
            origin=origin,
            resolution=resolution,
            trajectories=trajectories,
            agents=agents,
            predictions=predictions,
            probabilities=probabilities,
            prediction_interval=prediction_interval,
            score_agent_ids=score_agent_ids,
            oce_config=sigma_cfg,
            dt=dt,
            debug=debug,
        )
    )

    spatial_scores, spatial_oce, spatial_entropy = _summarize_public_comparison_results(
        spatial_scores, spatial_results
    )
    sigma_scores, sigma_oce, sigma_entropy = _summarize_public_comparison_results(
        sigma_scores, sigma_results
    )
    summary = _build_spatial_sigma_comparison_summary(
        backend="cpu",
        partition_mode=mode,
        spatial_total_scores=spatial_scores,
        sigma_total_scores=sigma_scores,
        spatial_total_oce=spatial_oce,
        sigma_total_oce=sigma_oce,
        spatial_total_entropy=spatial_entropy,
        sigma_total_entropy=sigma_entropy,
        num_targets=int(scene.num_targets),
        num_modes_total=int(scene.num_modes_total),
        prediction_length=int(scene.prediction_length),
    )
    summary["best_spatial_index"] = int(spatial_best)
    summary["best_sigma_index"] = int(sigma_best)
    summary["spatial_results"] = spatial_results
    summary["sigma_results"] = sigma_results
    return summary


def _resolve_kde_spatial_cuda_params(oce_config, eps):
    """Resolve 2D covariance parameters for CUDA kde-spatial."""

    if getattr(oce_config, "measurement_projection", None) is not None:
        raise ValueError(
            "CUDA kde-spatial path currently does not support custom measurement_projection."
        )

    obs_cov = getattr(oce_config, "obs_covariance", None)
    if obs_cov is None:
        obs_cov_np = np.eye(2, dtype=np.float64)
    else:
        obs_cov_np = np.asarray(obs_cov, dtype=np.float64)
        if obs_cov_np.shape != (2, 2):
            raise ValueError(
                f"obs_covariance for CUDA kde-spatial must be shape (2,2), got {obs_cov_np.shape}"
            )
        obs_cov_np = 0.5 * (obs_cov_np + obs_cov_np.T)

    obs_cov_np = obs_cov_np + float(eps) * np.eye(2, dtype=np.float64)

    proc_cov = getattr(oce_config, "occlusion_process_covariance", None)
    if proc_cov is None:
        proc_cov_np = np.array(obs_cov_np, copy=True)
    else:
        proc_cov_np = np.asarray(proc_cov, dtype=np.float64)
        if proc_cov_np.shape != (2, 2):
            raise ValueError(
                "occlusion_process_covariance for CUDA kde-spatial must be shape (2,2)"
            )
        proc_cov_np = 0.5 * (proc_cov_np + proc_cov_np.T)

    growth_scale = float(getattr(oce_config, "occlusion_cov_scale", 1.0))
    if growth_scale < 0.0:
        raise ValueError("occlusion_cov_scale must be non-negative.")
    proc_cov_np *= growth_scale
    disturbance_floor = float(max(np.trace(obs_cov_np), eps))
    process_exp_rate = float(np.trace(proc_cov_np))
    variance_floor = float(getattr(oce_config, "kde_min_variance", 1e-6))

    growth_mode_name = (
        str(getattr(oce_config, "occlusion_cov_growth", "constant")).strip().lower()
    )
    growth_mode_lut = {
        "linear": 0,
        "sqrt": 1,
        "square": 2,
        "exp": 3,
        "constant": 4,
        "none": 5,
    }
    if growth_mode_name not in growth_mode_lut:
        raise ValueError(
            "occlusion_cov_growth for CUDA kde-spatial must be one of "
            "{'linear', 'sqrt', 'square', 'exp', 'constant', 'none'}."
        )
    growth_mode = growth_mode_lut[growth_mode_name]

    return (
        int(growth_mode),
        float(proc_cov_np[0, 0]),
        float(proc_cov_np[0, 1]),
        float(proc_cov_np[1, 1]),
        float(process_exp_rate),
        float(disturbance_floor),
        float(variance_floor),
    )


def _resolve_kde_sigma_cuda_params(oce_config, eps):
    """Resolve 2D covariance and sigma-point parameters for CUDA kde-sigma-point."""

    (
        growth_mode,
        process_cov_00,
        process_cov_01,
        process_cov_11,
        process_exp_rate,
        disturbance_floor,
        variance_floor,
    ) = _resolve_kde_spatial_cuda_params(oce_config=oce_config, eps=eps)

    sigma_point_alpha = float(getattr(oce_config, "sigma_point_alpha", 1.0))
    sigma_point_kappa = float(getattr(oce_config, "sigma_point_kappa", 0.0))
    if sigma_point_alpha <= 0.0:
        raise ValueError("sigma_point_alpha must be positive.")

    sigma_point_scale = sigma_point_alpha**2 * (2.0 + sigma_point_kappa)
    if sigma_point_scale <= 0.0:
        raise ValueError(
            "sigma_point_alpha and sigma_point_kappa produced a non-positive sigma-point scale."
        )

    lam = sigma_point_scale - 2.0
    sigma_weight0 = lam / sigma_point_scale
    sigma_weight_other = 0.5 / sigma_point_scale
    entropy_floor = float(getattr(oce_config, "entropy_floor", 1e-12))

    return (
        growth_mode,
        process_cov_00,
        process_cov_01,
        process_cov_11,
        process_exp_rate,
        disturbance_floor,
        variance_floor,
        float(sigma_point_scale),
        float(sigma_weight0),
        float(sigma_weight_other),
        float(entropy_floor),
    )


def _build_oce_accumulation_result(
    canonical,
    per_target_oce,
    per_target_entropy,
    per_target_state,
    per_target_mi,
    per_target_occ,
    per_target_total,
    total_entropies,
    i_times_per_target,
    per_target_within=None,
    step_oce=None,
    step_entropy=None,
    step_within=None,
    step_state=None,
    step_mi=None,
    entropy_space_norm="exact_partition-info",
    lambda_e=0.0,
):
    """Build a CPU-facing accumulation result from dense host arrays."""

    N = int(canonical.num_trajectories)
    T = int(canonical.prediction_length)
    A_targets = int(canonical.num_targets)
    best_traj_index = int(np.argmin(total_entropies)) if N > 0 else 0
    if step_oce is None:
        step_oce = np.zeros((N, A_targets, T), dtype=np.float32)
    if step_entropy is None:
        step_entropy = np.zeros((N, A_targets, T), dtype=np.float32)
    if step_within is None:
        step_within = np.zeros((N, A_targets, T), dtype=np.float32)
    if step_state is None:
        step_state = np.zeros((N, A_targets, T), dtype=np.float32)
    if step_mi is None:
        step_mi = np.zeros((N, A_targets, T), dtype=np.float32)
    if per_target_within is None:
        per_target_within = np.sum(
            np.asarray(step_within, dtype=np.float32),
            axis=2,
            dtype=np.float32,
        )

    return OCEAccumulationResult(
        total_entropies=np.ascontiguousarray(
            np.asarray(total_entropies, dtype=np.float32).reshape(N)
        ),
        best_traj_index=best_traj_index,
        per_target_total=np.ascontiguousarray(
            np.asarray(per_target_total, dtype=np.float32).reshape(N, A_targets)
        ),
        per_target_oce=np.ascontiguousarray(
            np.asarray(per_target_oce, dtype=np.float32).reshape(N, A_targets)
        ),
        per_target_entropy=np.ascontiguousarray(
            np.asarray(per_target_entropy, dtype=np.float32).reshape(N, A_targets)
        ),
        per_target_occ=np.ascontiguousarray(
            np.asarray(per_target_occ, dtype=np.float32).reshape(N, A_targets)
        ),
        per_target_state=np.ascontiguousarray(
            np.asarray(per_target_state, dtype=np.float32).reshape(N, A_targets)
        ),
        per_target_mi=np.ascontiguousarray(
            np.asarray(per_target_mi, dtype=np.float32).reshape(N, A_targets)
        ),
        per_target_within=np.ascontiguousarray(
            np.asarray(per_target_within, dtype=np.float32).reshape(N, A_targets)
        ),
        step_oce=np.ascontiguousarray(
            np.asarray(step_oce, dtype=np.float32).reshape(N, A_targets, T)
        ),
        step_entropy=np.ascontiguousarray(
            np.asarray(step_entropy, dtype=np.float32).reshape(N, A_targets, T)
        ),
        step_within=np.ascontiguousarray(
            np.asarray(step_within, dtype=np.float32).reshape(N, A_targets, T)
        ),
        step_state=np.ascontiguousarray(
            np.asarray(step_state, dtype=np.float32).reshape(N, A_targets, T)
        ),
        step_mi=np.ascontiguousarray(
            np.asarray(step_mi, dtype=np.float32).reshape(N, A_targets, T)
        ),
        i_times_per_target=np.ascontiguousarray(
            np.asarray(i_times_per_target, dtype=np.float32).reshape(A_targets, T)
        ),
        target_agent_ids=np.ascontiguousarray(
            canonical.target_agent_ids.astype(np.int32)
        ),
        target_indices=np.arange(A_targets, dtype=np.int32),
        entropy_space_norm=str(entropy_space_norm),
        lambda_e=float(lambda_e),
    )


def _empty_oce_accumulation_result(canonical):
    """Build an all-zero accumulation result for degenerate scenes."""

    N = int(canonical.num_trajectories)
    T = int(canonical.prediction_length)
    A_targets = int(canonical.num_targets)
    zeros_na = np.zeros((N, A_targets), dtype=np.float32)
    return _build_oce_accumulation_result(
        canonical=canonical,
        per_target_oce=zeros_na.copy(),
        per_target_entropy=zeros_na.copy(),
        per_target_state=zeros_na.copy(),
        per_target_mi=zeros_na.copy(),
        per_target_occ=zeros_na.copy(),
        per_target_total=zeros_na.copy(),
        total_entropies=np.zeros((N,), dtype=np.float32),
        i_times_per_target=np.zeros((A_targets, T), dtype=np.float32),
        per_target_within=zeros_na.copy(),
        step_oce=np.zeros((N, A_targets, T), dtype=np.float32),
        step_entropy=np.zeros((N, A_targets, T), dtype=np.float32),
        step_within=np.zeros((N, A_targets, T), dtype=np.float32),
        step_state=np.zeros((N, A_targets, T), dtype=np.float32),
        step_mi=np.zeros((N, A_targets, T), dtype=np.float32),
    )


def _run_cuda_end_to_end_kde_spatial_device(
    canonical,
    visibility_d,
    partition_mode,
    oce_config,
    lambda_oce,
    lambda_vis,
    lambda_state,
    lambda_mi,
    lambda_e,
    eps,
    discount,
    cuda_cache,
    metric_variant=0,
):
    """Run the log-det spatial or separability CUDA reduction and keep outputs on device."""

    compute_kde_kernel = _get_oce_spatial_cuda_kernel()
    reduce_traj_kernel = _get_oce_reduce_traj_cuda_kernel()

    N = int(canonical.num_trajectories)
    T = int(canonical.prediction_length)
    M_total = int(canonical.num_modes_total)

    A_targets = int(canonical.num_targets)

    if N <= 0 or A_targets <= 0 or T <= 0 or M_total <= 0:
        return OCEDeviceAccumulationBuffers(
            per_target_entropy_d=np.uintp(0),
            per_target_oce_d=np.uintp(0),
            per_target_state_d=np.uintp(0),
            per_target_mi_d=np.uintp(0),
            per_target_occ_d=np.uintp(0),
            per_target_total_d=np.uintp(0),
            total_entropies_d=np.uintp(0),
            i_times_per_target=np.zeros((A_targets, T), dtype=np.float32),
            step_oce_d=np.uintp(0),
            step_entropy_d=np.uintp(0),
            step_within_d=np.uintp(0),
            step_state_d=np.uintp(0),
            step_mi_d=np.uintp(0),
            entropy_space_norm=(
                "exact_partition-info"
                if int(metric_variant) == 3
                else "exact_kde-spatial"
            ),
            lambda_e=float(lambda_e),
        )

    target_state_dim = (
        int(canonical.target_states_packed.shape[2])
        if canonical.target_states_packed.ndim == 3
        else 0
    )
    if target_state_dim < 2:
        raise ValueError(
            "canonical.target_states_packed must have at least 2 state dimensions."
        )
    max_modes_per_target = (
        int(np.max(np.diff(canonical.mode_offsets)))
        if canonical.mode_offsets.size > 1
        else 0
    )
    if max_modes_per_target > 32:
        raise ValueError(
            "CUDA kde-spatial kernel currently supports at most 32 modes per target."
        )

    mode_priors_h = np.ascontiguousarray(
        np.asarray(canonical.mode_priors_packed, dtype=np.float32).reshape(-1)
    )
    mode_offsets_h = np.ascontiguousarray(
        np.asarray(canonical.mode_offsets, dtype=np.int32).reshape(-1)
    )
    confusion_offsets_h = np.ascontiguousarray(
        np.asarray(canonical.confusion_offsets, dtype=np.int32).reshape(-1)
    )
    confusion_packed_h = np.ascontiguousarray(
        np.asarray(canonical.confusion_packed, dtype=np.float32).reshape(-1)
    )
    target_risk_durations_h = _resolved_target_risk_durations(canonical)
    target_states_h = np.ascontiguousarray(
        np.asarray(canonical.target_states_packed, dtype=np.float32).reshape(-1)
    )
    i_times_h = _prepare_i_times(canonical=canonical, eps=eps)
    i_times_flat_h = np.ascontiguousarray(i_times_h.reshape(-1))
    max_modes_per_target = (
        int(np.max(np.diff(mode_offsets_h))) if mode_offsets_h.size >= 2 else 0
    )
    if max_modes_per_target > 32:
        raise ValueError(
            "CUDA kde-spatial currently supports at most 32 modes per target; "
            f"received {max_modes_per_target}."
        )

    (
        growth_mode,
        process_cov_00,
        process_cov_01,
        process_cov_11,
        process_exp_rate,
        disturbance_floor,
        variance_floor,
    ) = _resolve_kde_spatial_cuda_params(oce_config=oce_config, eps=eps)

    total_pairs = N * A_targets

    bytes_target = np.dtype(np.float32).itemsize * total_pairs
    bytes_traj = np.dtype(np.float32).itemsize * N
    bytes_steps = np.dtype(np.float32).itemsize * total_pairs * T

    mode_priors_d = cuda_cache.reserve_array("mode_priors_d", mode_priors_h)
    mode_offsets_d = cuda_cache.reserve_array("mode_offsets_d", mode_offsets_h)
    target_risk_durations_d = cuda_cache.reserve_array(
        "target_risk_durations_d", target_risk_durations_h
    )
    confusion_offsets_d = cuda_cache.reserve_array(
        "confusion_offsets_d", confusion_offsets_h
    )
    confusion_packed_d = cuda_cache.reserve_array(
        "confusion_packed_d", confusion_packed_h
    )
    target_states_d = cuda_cache.reserve_array("target_states_d", target_states_h)
    i_times_d = cuda_cache.reserve_array("i_times_d", i_times_flat_h)

    per_target_oce_d = cuda_cache.reserve_data(
        "per_target_oce_d", bytes_target, zero_fill=True
    )
    per_target_entropy_d = cuda_cache.reserve_data(
        "per_target_entropy_d", bytes_target, zero_fill=True
    )
    per_target_state_d = cuda_cache.reserve_data(
        "per_target_state_d", bytes_target, zero_fill=True
    )
    per_target_mi_d = cuda_cache.reserve_data(
        "per_target_mi_d", bytes_target, zero_fill=True
    )
    per_target_occ_d = cuda_cache.reserve_data(
        "per_target_occ_d", bytes_target, zero_fill=True
    )
    per_target_total_d = cuda_cache.reserve_data(
        "per_target_total_d", bytes_target, zero_fill=True
    )
    total_entropies_d = cuda_cache.reserve_data(
        "total_entropies_d", bytes_traj, zero_fill=True
    )
    step_oce_d = cuda_cache.reserve_data("step_oce_d", bytes_steps, zero_fill=True)
    step_entropy_d = cuda_cache.reserve_data(
        "step_entropy_d", bytes_steps, zero_fill=True
    )
    step_within_d = cuda_cache.reserve_data(
        "step_within_d", bytes_steps, zero_fill=True
    )
    step_state_d = cuda_cache.reserve_data("step_state_d", bytes_steps, zero_fill=True)
    step_mi_d = cuda_cache.reserve_data("step_mi_d", bytes_steps, zero_fill=True)

    block_size = 128
    grid_pairs = (int((total_pairs + block_size - 1) // block_size), 1, 1)
    compute_kde_kernel(
        visibility_d,
        mode_priors_d,
        mode_offsets_d,
        confusion_offsets_d,
        confusion_packed_d,
        target_states_d,
        target_risk_durations_d,
        i_times_d,
        per_target_entropy_d,
        per_target_oce_d,
        per_target_state_d,
        per_target_mi_d,
        per_target_occ_d,
        per_target_total_d,
        step_oce_d,
        step_entropy_d,
        step_within_d,
        step_state_d,
        step_mi_d,
        np.int32(N),
        np.int32(A_targets),
        np.int32(T),
        np.int32(M_total),
        np.int32(target_state_dim),
        np.int32(int(partition_mode)),
        np.float32(lambda_oce),
        np.float32(lambda_vis),
        np.float32(lambda_state),
        np.float32(lambda_mi),
        np.float32(lambda_e),
        np.float32(eps),
        np.float32(discount),
        np.int32(growth_mode),
        np.float32(process_cov_00),
        np.float32(process_cov_01),
        np.float32(process_cov_11),
        np.float32(process_exp_rate),
        np.float32(disturbance_floor),
        np.float32(variance_floor),
        np.int32(int(metric_variant)),
        block=(block_size, 1, 1),
        grid=grid_pairs,
    )

    grid_traj = (int((N + block_size - 1) // block_size), 1, 1)
    reduce_traj_kernel(
        per_target_total_d,
        total_entropies_d,
        np.int32(N),
        np.int32(A_targets),
        block=(block_size, 1, 1),
        grid=grid_traj,
    )

    return OCEDeviceAccumulationBuffers(
        per_target_entropy_d=per_target_entropy_d,
        per_target_oce_d=per_target_oce_d,
        per_target_state_d=per_target_state_d,
        per_target_mi_d=per_target_mi_d,
        per_target_occ_d=per_target_occ_d,
        per_target_total_d=per_target_total_d,
        total_entropies_d=total_entropies_d,
        i_times_per_target=np.ascontiguousarray(i_times_h),
        step_oce_d=step_oce_d,
        step_entropy_d=step_entropy_d,
        step_within_d=step_within_d,
        step_state_d=step_state_d,
        step_mi_d=step_mi_d,
        entropy_space_norm=(
            f"{'exact' if int(partition_mode) == 1 else 'approximate'}_"
            + (
                "partition-info"
                if int(metric_variant) == 3
                else (
                    "kde-separability-logdet"
                    if int(metric_variant) == 1
                    else (
                        "kde-separability-trace"
                        if int(metric_variant) == 2
                        else "kde-spatial"
                    )
                )
            )
        ),
        lambda_e=float(lambda_e),
    )


def _run_cuda_end_to_end_kde_sigma_device(
    canonical,
    visibility_d,
    partition_mode,
    oce_config,
    lambda_oce,
    lambda_vis,
    lambda_state,
    lambda_mi,
    lambda_e,
    eps,
    discount,
    cuda_cache,
):
    """Run end-to-end P4 CUDA execution for kde-sigma-point and keep reductions on device."""

    compute_kde_sigma_kernel = _get_oce_sigma_point_cuda_kernel()
    reduce_traj_kernel = _get_oce_reduce_traj_cuda_kernel()

    N = int(canonical.num_trajectories)
    T = int(canonical.prediction_length)
    M_total = int(canonical.num_modes_total)
    A_targets = int(canonical.num_targets)

    if N <= 0 or A_targets <= 0 or T <= 0 or M_total <= 0:
        return OCEDeviceAccumulationBuffers(
            per_target_entropy_d=np.uintp(0),
            per_target_oce_d=np.uintp(0),
            per_target_state_d=np.uintp(0),
            per_target_mi_d=np.uintp(0),
            per_target_occ_d=np.uintp(0),
            per_target_total_d=np.uintp(0),
            total_entropies_d=np.uintp(0),
            i_times_per_target=np.zeros((A_targets, T), dtype=np.float32),
            step_oce_d=np.uintp(0),
            step_entropy_d=np.uintp(0),
            step_within_d=np.uintp(0),
            step_state_d=np.uintp(0),
            step_mi_d=np.uintp(0),
            entropy_space_norm=(
                f"{'exact' if int(partition_mode) == 1 else 'approximate'}_"
                "kde-sigma-point"
            ),
            lambda_e=float(lambda_e),
        )

    target_state_dim = (
        int(canonical.target_states_packed.shape[2])
        if canonical.target_states_packed.ndim == 3
        else 0
    )
    if target_state_dim < 2:
        raise ValueError(
            "canonical.target_states_packed must have at least 2 state dimensions."
        )
    max_modes_per_target = (
        int(np.max(np.diff(canonical.mode_offsets)))
        if canonical.mode_offsets.size > 1
        else 0
    )
    if max_modes_per_target > 32:
        raise ValueError(
            "CUDA kde-sigma-point kernel currently supports at most 32 modes per target."
        )

    mode_priors_h = np.ascontiguousarray(
        np.asarray(canonical.mode_priors_packed, dtype=np.float32).reshape(-1)
    )
    mode_offsets_h = np.ascontiguousarray(
        np.asarray(canonical.mode_offsets, dtype=np.int32).reshape(-1)
    )
    target_risk_durations_h = np.ascontiguousarray(
        np.asarray(canonical.target_risk_durations, dtype=np.int32).reshape(-1)
    )
    confusion_offsets_h = np.ascontiguousarray(
        np.asarray(canonical.confusion_offsets, dtype=np.int32).reshape(-1)
    )
    confusion_packed_h = np.ascontiguousarray(
        np.asarray(canonical.confusion_packed, dtype=np.float32).reshape(-1)
    )
    target_states_h = np.ascontiguousarray(
        np.asarray(canonical.target_states_packed, dtype=np.float32).reshape(-1)
    )
    i_times_h = _prepare_i_times(canonical=canonical, eps=eps)
    i_times_flat_h = np.ascontiguousarray(i_times_h.reshape(-1))

    (
        growth_mode,
        process_cov_00,
        process_cov_01,
        process_cov_11,
        process_exp_rate,
        disturbance_floor,
        variance_floor,
        sigma_point_scale,
        sigma_weight0,
        sigma_weight_other,
        entropy_floor,
    ) = _resolve_kde_sigma_cuda_params(oce_config=oce_config, eps=eps)

    total_pairs = N * A_targets
    bytes_target = np.dtype(np.float32).itemsize * total_pairs
    bytes_traj = np.dtype(np.float32).itemsize * N
    bytes_steps = np.dtype(np.float32).itemsize * total_pairs * T

    mode_priors_d = cuda_cache.reserve_array("mode_priors_d", mode_priors_h)
    mode_offsets_d = cuda_cache.reserve_array("mode_offsets_d", mode_offsets_h)
    target_risk_durations_d = cuda_cache.reserve_array(
        "target_risk_durations_d", target_risk_durations_h
    )
    confusion_offsets_d = cuda_cache.reserve_array(
        "confusion_offsets_d", confusion_offsets_h
    )
    confusion_packed_d = cuda_cache.reserve_array(
        "confusion_packed_d", confusion_packed_h
    )
    target_states_d = cuda_cache.reserve_array("target_states_d", target_states_h)
    i_times_d = cuda_cache.reserve_array("i_times_d", i_times_flat_h)

    per_target_oce_d = cuda_cache.reserve_data(
        "per_target_oce_d", bytes_target, zero_fill=True
    )
    per_target_entropy_d = cuda_cache.reserve_data(
        "per_target_entropy_d", bytes_target, zero_fill=True
    )
    per_target_state_d = cuda_cache.reserve_data(
        "per_target_state_d", bytes_target, zero_fill=True
    )
    per_target_mi_d = cuda_cache.reserve_data(
        "per_target_mi_d", bytes_target, zero_fill=True
    )
    per_target_occ_d = cuda_cache.reserve_data(
        "per_target_occ_d", bytes_target, zero_fill=True
    )
    per_target_total_d = cuda_cache.reserve_data(
        "per_target_total_d", bytes_target, zero_fill=True
    )
    total_entropies_d = cuda_cache.reserve_data(
        "total_entropies_d", bytes_traj, zero_fill=True
    )
    step_oce_d = cuda_cache.reserve_data("step_oce_d", bytes_steps, zero_fill=True)
    step_entropy_d = cuda_cache.reserve_data(
        "step_entropy_d", bytes_steps, zero_fill=True
    )
    step_within_d = cuda_cache.reserve_data(
        "step_within_d", bytes_steps, zero_fill=True
    )
    step_state_d = cuda_cache.reserve_data("step_state_d", bytes_steps, zero_fill=True)
    step_mi_d = cuda_cache.reserve_data("step_mi_d", bytes_steps, zero_fill=True)

    block_size = 128
    grid_pairs = (int((total_pairs + block_size - 1) // block_size), 1, 1)
    compute_kde_sigma_kernel(
        visibility_d,
        mode_priors_d,
        mode_offsets_d,
        target_risk_durations_d,
        confusion_offsets_d,
        confusion_packed_d,
        target_states_d,
        i_times_d,
        per_target_entropy_d,
        per_target_oce_d,
        per_target_state_d,
        per_target_mi_d,
        per_target_occ_d,
        per_target_total_d,
        step_oce_d,
        step_entropy_d,
        step_within_d,
        step_state_d,
        step_mi_d,
        np.int32(N),
        np.int32(A_targets),
        np.int32(T),
        np.int32(M_total),
        np.int32(target_state_dim),
        np.int32(int(partition_mode)),
        np.float32(lambda_oce),
        np.float32(lambda_vis),
        np.float32(lambda_state),
        np.float32(lambda_mi),
        np.float32(lambda_e),
        np.float32(eps),
        np.float32(discount),
        np.int32(growth_mode),
        np.float32(process_cov_00),
        np.float32(process_cov_01),
        np.float32(process_cov_11),
        np.float32(process_exp_rate),
        np.float32(disturbance_floor),
        np.float32(variance_floor),
        np.float32(sigma_point_scale),
        np.float32(sigma_weight0),
        np.float32(sigma_weight_other),
        np.float32(entropy_floor),
        block=(block_size, 1, 1),
        grid=grid_pairs,
    )

    grid_traj = (int((N + block_size - 1) // block_size), 1, 1)
    reduce_traj_kernel(
        per_target_total_d,
        total_entropies_d,
        np.int32(N),
        np.int32(A_targets),
        block=(block_size, 1, 1),
        grid=grid_traj,
    )

    return OCEDeviceAccumulationBuffers(
        per_target_entropy_d=per_target_entropy_d,
        per_target_oce_d=per_target_oce_d,
        per_target_state_d=per_target_state_d,
        per_target_mi_d=per_target_mi_d,
        per_target_occ_d=per_target_occ_d,
        per_target_total_d=per_target_total_d,
        total_entropies_d=total_entropies_d,
        i_times_per_target=np.ascontiguousarray(i_times_h),
        step_oce_d=step_oce_d,
        step_entropy_d=step_entropy_d,
        step_within_d=step_within_d,
        step_state_d=step_state_d,
        step_mi_d=step_mi_d,
        entropy_space_norm=(
            f"{'exact' if int(partition_mode) == 1 else 'approximate'}_"
            "kde-sigma-point"
        ),
        lambda_e=float(lambda_e),
    )


def _materialize_cuda_end_to_end_kde_spatial(
    canonical,
    device_buffers,
):
    """Copy device-resident kde-spatial reductions back to host arrays."""

    N = int(canonical.num_trajectories)
    A_targets = int(canonical.num_targets)
    T = int(canonical.prediction_length)
    M_total = int(canonical.num_modes_total)

    if N <= 0 or A_targets <= 0 or T <= 0 or M_total <= 0:
        zeros_na = np.zeros((N, A_targets), dtype=np.float32)
        return (
            zeros_na.copy(),
            zeros_na.copy(),
            zeros_na.copy(),
            zeros_na.copy(),
            zeros_na.copy(),
            zeros_na.copy(),
            np.zeros((N,), dtype=np.float32),
            np.zeros((N, A_targets, T), dtype=np.float32),
            np.zeros((N, A_targets, T), dtype=np.float32),
            np.zeros((N, A_targets, T), dtype=np.float32),
            np.zeros((N, A_targets, T), dtype=np.float32),
            np.zeros((N, A_targets, T), dtype=np.float32),
            np.ascontiguousarray(device_buffers.i_times_per_target.astype(np.float32)),
        )

    total_pairs = N * A_targets
    total_steps = total_pairs * T
    per_target_entropy_h = np.empty((total_pairs,), dtype=np.float32)
    per_target_oce_h = np.empty((total_pairs,), dtype=np.float32)
    per_target_state_h = np.empty((total_pairs,), dtype=np.float32)
    per_target_mi_h = np.empty((total_pairs,), dtype=np.float32)
    per_target_occ_h = np.empty((total_pairs,), dtype=np.float32)
    per_target_total_h = np.empty((total_pairs,), dtype=np.float32)
    total_entropies_h = np.empty((N,), dtype=np.float32)
    step_oce_h = np.zeros((total_steps,), dtype=np.float32)
    step_entropy_h = np.zeros((total_steps,), dtype=np.float32)
    step_within_h = np.zeros((total_steps,), dtype=np.float32)
    step_state_h = np.zeros((total_steps,), dtype=np.float32)
    step_mi_h = np.zeros((total_steps,), dtype=np.float32)
    _cuda.memcpy_dtoh(per_target_oce_h, device_buffers.per_target_oce_d)
    _cuda.memcpy_dtoh(per_target_entropy_h, device_buffers.per_target_entropy_d)
    _cuda.memcpy_dtoh(per_target_state_h, device_buffers.per_target_state_d)
    _cuda.memcpy_dtoh(per_target_mi_h, device_buffers.per_target_mi_d)
    _cuda.memcpy_dtoh(per_target_occ_h, device_buffers.per_target_occ_d)
    _cuda.memcpy_dtoh(per_target_total_h, device_buffers.per_target_total_d)
    _cuda.memcpy_dtoh(total_entropies_h, device_buffers.total_entropies_d)
    if device_buffers.step_oce_d is not None and device_buffers.step_oce_d != np.uintp(
        0
    ):
        _cuda.memcpy_dtoh(step_oce_h, device_buffers.step_oce_d)
    if (
        device_buffers.step_entropy_d is not None
        and device_buffers.step_entropy_d != np.uintp(0)
    ):
        _cuda.memcpy_dtoh(step_entropy_h, device_buffers.step_entropy_d)
    if (
        device_buffers.step_within_d is not None
        and device_buffers.step_within_d != np.uintp(0)
    ):
        _cuda.memcpy_dtoh(step_within_h, device_buffers.step_within_d)
    if (
        device_buffers.step_state_d is not None
        and device_buffers.step_state_d != np.uintp(0)
    ):
        _cuda.memcpy_dtoh(step_state_h, device_buffers.step_state_d)
    if device_buffers.step_mi_d is not None and device_buffers.step_mi_d != np.uintp(0):
        _cuda.memcpy_dtoh(step_mi_h, device_buffers.step_mi_d)

    return (
        np.ascontiguousarray(per_target_oce_h.reshape(N, A_targets)),
        np.ascontiguousarray(per_target_entropy_h.reshape(N, A_targets)),
        np.ascontiguousarray(per_target_state_h.reshape(N, A_targets)),
        np.ascontiguousarray(per_target_mi_h.reshape(N, A_targets)),
        np.ascontiguousarray(per_target_occ_h.reshape(N, A_targets)),
        np.ascontiguousarray(per_target_total_h.reshape(N, A_targets)),
        np.ascontiguousarray(total_entropies_h),
        np.ascontiguousarray(step_oce_h.reshape(N, A_targets, T)),
        np.ascontiguousarray(step_entropy_h.reshape(N, A_targets, T)),
        np.ascontiguousarray(step_within_h.reshape(N, A_targets, T)),
        np.ascontiguousarray(step_state_h.reshape(N, A_targets, T)),
        np.ascontiguousarray(step_mi_h.reshape(N, A_targets, T)),
        np.ascontiguousarray(device_buffers.i_times_per_target.astype(np.float32)),
    )


def accumulate_oce_from_partitions_device(
    canonical,
    visibility_d,
    oce_config=None,
    eps=1e-9,
    entropy_space="kde",
    discount=1.0,
    cuda_cache=None,
):
    """Accumulate OCE contributions on device for consumers that manage host transfer."""

    if not isinstance(canonical, CanonicalEntropyInputs):
        raise TypeError("canonical must be a CanonicalEntropyInputs instance")

    if not _PYCUDA_AVAILABLE:
        raise RuntimeError(
            "PyCUDA is required for CUDA execution but is not available. Please install PyCUDA."
        )

    if oce_config is None:
        oce_config = _cpu_eval.OCEConfig(
            lambda_oce=1.0,
            lambda_vis=0.0,
            lambda_state=0.0,
            lambda_mi=0.0,
        )

    entropy_space_norm = _normalize_kde_entropy_space(entropy_space)
    if entropy_space_norm in (
        "exact_partition-info",
        "approximate_partition-info",
    ):
        if cuda_cache is None:
            cuda_cache = _cuda_buffer_cache
        return _run_cuda_end_to_end_kde_spatial_device(
            canonical=canonical,
            visibility_d=visibility_d,
            partition_mode=1 if entropy_space_norm.startswith("exact_") else 0,
            oce_config=oce_config,
            lambda_oce=float(_safe_attr(oce_config, "lambda_oce", 1.0)),
            lambda_vis=float(_safe_attr(oce_config, "lambda_vis", 0.0)),
            lambda_state=float(_safe_attr(oce_config, "lambda_state", 0.0)),
            lambda_mi=float(_safe_attr(oce_config, "lambda_mi", 0.0)),
            lambda_e=float(_safe_attr(oce_config, "lambda_e", 0.0)),
            eps=float(eps),
            discount=float(discount),
            cuda_cache=cuda_cache,
            metric_variant=3,
        )
    if entropy_space_norm in (
        "exact_kde-spatial",
        "approximate_kde-spatial",
    ):
        if cuda_cache is None:
            cuda_cache = _cuda_buffer_cache
        return _run_cuda_end_to_end_kde_spatial_device(
            canonical=canonical,
            visibility_d=visibility_d,
            partition_mode=1 if entropy_space_norm.startswith("exact_") else 0,
            oce_config=oce_config,
            lambda_oce=float(_safe_attr(oce_config, "lambda_oce", 1.0)),
            lambda_vis=float(_safe_attr(oce_config, "lambda_vis", 0.0)),
            lambda_state=float(_safe_attr(oce_config, "lambda_state", 0.0)),
            lambda_mi=float(_safe_attr(oce_config, "lambda_mi", 0.0)),
            lambda_e=float(_safe_attr(oce_config, "lambda_e", 0.0)),
            eps=float(eps),
            discount=float(discount),
            cuda_cache=cuda_cache,
        )
    if entropy_space_norm in (
        "exact_kde-separability-logdet",
        "approximate_kde-separability-logdet",
    ):
        if cuda_cache is None:
            cuda_cache = _cuda_buffer_cache
        return _run_cuda_end_to_end_kde_spatial_device(
            canonical=canonical,
            visibility_d=visibility_d,
            partition_mode=1 if entropy_space_norm.startswith("exact_") else 0,
            oce_config=oce_config,
            lambda_oce=float(_safe_attr(oce_config, "lambda_oce", 1.0)),
            lambda_vis=float(_safe_attr(oce_config, "lambda_vis", 0.0)),
            lambda_state=float(_safe_attr(oce_config, "lambda_state", 0.0)),
            lambda_mi=float(_safe_attr(oce_config, "lambda_mi", 0.0)),
            lambda_e=float(_safe_attr(oce_config, "lambda_e", 0.0)),
            eps=float(eps),
            discount=float(discount),
            cuda_cache=cuda_cache,
            metric_variant=1,
        )
    if entropy_space_norm in (
        "exact_kde-separability-trace",
        "approximate_kde-separability-trace",
    ):
        if cuda_cache is None:
            cuda_cache = _cuda_buffer_cache
        return _run_cuda_end_to_end_kde_spatial_device(
            canonical=canonical,
            visibility_d=visibility_d,
            partition_mode=1 if entropy_space_norm.startswith("exact_") else 0,
            oce_config=oce_config,
            lambda_oce=float(_safe_attr(oce_config, "lambda_oce", 1.0)),
            lambda_vis=float(_safe_attr(oce_config, "lambda_vis", 0.0)),
            lambda_state=float(_safe_attr(oce_config, "lambda_state", 0.0)),
            lambda_mi=float(_safe_attr(oce_config, "lambda_mi", 0.0)),
            lambda_e=float(_safe_attr(oce_config, "lambda_e", 0.0)),
            eps=float(eps),
            discount=float(discount),
            cuda_cache=cuda_cache,
            metric_variant=2,
        )
    if entropy_space_norm in (
        "exact_kde-sigma-point",
        "approximate_kde-sigma-point",
    ):
        if cuda_cache is None:
            cuda_cache = _cuda_buffer_cache
        return _run_cuda_end_to_end_kde_sigma_device(
            canonical=canonical,
            visibility_d=visibility_d,
            partition_mode=1 if entropy_space_norm.startswith("exact_") else 0,
            oce_config=oce_config,
            lambda_oce=float(_safe_attr(oce_config, "lambda_oce", 1.0)),
            lambda_vis=float(_safe_attr(oce_config, "lambda_vis", 0.0)),
            lambda_state=float(_safe_attr(oce_config, "lambda_state", 0.0)),
            lambda_mi=float(_safe_attr(oce_config, "lambda_mi", 0.0)),
            lambda_e=float(_safe_attr(oce_config, "lambda_e", 0.0)),
            eps=float(eps),
            discount=float(discount),
            cuda_cache=cuda_cache,
        )

    raise ValueError(
        f"Unsupported entropy_space '{entropy_space}' for CUDA accumulation. "
        "CUDA accumulation currently supports the partition-info/kde, "
        "kde-spatial, "
        "kde-sigma-point, kde-separability-logdet, and "
        "kde-separability-trace families only."
    )


def _materialize_oce_device_accumulation_impl(
    canonical,
    device_buffers,
):
    """Convert device-side accumulation buffers into the standalone CPU-facing result type."""

    if not isinstance(canonical, CanonicalEntropyInputs):
        raise TypeError("canonical must be a CanonicalEntropyInputs instance")
    if device_buffers is None:
        return _empty_oce_accumulation_result(canonical)

    (
        per_target_oce,
        per_target_entropy,
        per_target_state,
        per_target_mi,
        per_target_occ,
        per_target_total,
        total_entropies,
        step_oce,
        step_entropy,
        step_within,
        step_state,
        step_mi,
        i_times_per_target,
    ) = _materialize_cuda_end_to_end_kde_spatial(
        canonical=canonical,
        device_buffers=device_buffers,
    )

    unfiltered = _build_oce_accumulation_result(
        canonical=canonical,
        per_target_oce=per_target_oce,
        per_target_entropy=per_target_entropy,
        per_target_state=per_target_state,
        per_target_mi=per_target_mi,
        per_target_occ=per_target_occ,
        per_target_total=per_target_total,
        total_entropies=total_entropies,
        i_times_per_target=i_times_per_target,
        per_target_within=np.sum(step_within, axis=2, dtype=np.float32),
        step_oce=step_oce,
        step_entropy=step_entropy,
        step_within=step_within,
        step_state=step_state,
        step_mi=step_mi,
        entropy_space_norm=str(
            getattr(device_buffers, "entropy_space_norm", "exact_partition-info")
        ),
        lambda_e=float(getattr(device_buffers, "lambda_e", 0.0)),
    )
    return _filter_accumulation_to_scored_targets(canonical, unfiltered)


def materialize_oce_device_accumulation(
    canonical,
    device_buffers,
):
    """Convert device-side accumulation buffers with the OCE CUDA context pushed."""

    with _active_cuda_context():
        return _materialize_oce_device_accumulation_impl(
            canonical=canonical,
            device_buffers=device_buffers,
        )


def evaluate_trajectories_by_entropy_gpu(
    time_step,
    grid,
    origin,
    resolution,
    trajectories,
    agents,
    predictions,
    probabilities,
    prediction_interval,
    score_agent_ids: Optional[Iterable[int]] = None,
    oce_config=None,
    dt=_cpu_eval.DEFAULT_TIME_TICK,
    debug=False,
    debug_output_dir="test_data/entropy_eval_debug",
    debug_visibility_stride=5,
    debug_episode_reset=False,
    debug_record_episode_metrics=True,
):
    """Drop-in GPU entrypoint with the same interface as CPU entropy evaluation.

    P5 status:
    - Evaluates trajectories through the standalone visibility->partition->OCE pipeline.
    - Returns CPU-compatible `(best_traj_index, total_entropies, results)` materialization.
    """
    num_trajectories = len(trajectories)
    empty_results = [[] for _ in range(num_trajectories)]

    if not predictions:
        # print("No predictions available to evaluate.")
        return 0, [], None

    scored_agent_ids = _cpu_eval._resolve_scored_agent_ids(
        predictions=predictions,
        probabilities=probabilities,
        score_agent_ids=score_agent_ids,
    )
    if len(scored_agent_ids) == 0:
        return 0, np.zeros(num_trajectories, dtype=np.float32), empty_results

    if not _PYCUDA_AVAILABLE:
        raise RuntimeError(
            "evaluate_trajectories_by_entropy_gpu was called, but PyCUDA is not available. "
            "Please ensure that PyCUDA is installed and a compatible NVIDIA GPU is present."
        )

    prediction_length = _compute_prediction_length(trajectories, predictions)
    scene = build_oce_scene_inputs(
        time_step=time_step,
        grid=grid,
        origin=origin,
        resolution=resolution,
        agents=agents,
        predictions=predictions,
        probabilities=probabilities,
        prediction_interval=prediction_interval,
        prediction_length=prediction_length,
        score_agent_ids=scored_agent_ids,
        dt=dt,
    )

    if len(trajectories) <= 0:
        return 0, np.zeros((0,), dtype=np.float32), []

    if (
        scene.num_targets <= 0
        or scene.num_modes_total <= 0
        or scene.prediction_length <= 0
    ):
        return 0, np.zeros(num_trajectories, dtype=np.float32), empty_results

    if oce_config is None:
        oce_config = _cpu_eval.OCEConfig()

    entropy_space = _normalize_kde_entropy_space(
        _safe_attr(oce_config, "oce_entropy_space", "kde")
    )
    if not _is_kde_entropy_space(entropy_space):
        raise ValueError(
            f"Unsupported entropy_space '{entropy_space}' in oce_config. "
            "Supported spaces are the KDE family, including exact_/approximate_ prefixes."
        )

    discount = float(_safe_attr(oce_config, "oce_discount", 1.0))
    trajectory_states = _pack_trajectory_states(
        trajectories=trajectories,
        prediction_length=scene.prediction_length,
        steps_per_prediction=scene.steps_per_prediction,
    )

    global _cuda_buffer_cache
    score_result = score_oce_scene_rollouts_gpu(
        scene=scene,
        rollout_states=trajectory_states,
        oce_config=oce_config,
        eps=float(_safe_attr(oce_config, "eps", 1e-9)),
        entropy_space=entropy_space,
        discount=discount,
        cuda_cache=_cuda_buffer_cache,
        debug=bool(debug),
    )

    total_entropies = np.ascontiguousarray(
        np.asarray(score_result.accumulation.total_entropies, dtype=np.float32)
    )
    best_traj_index = int(score_result.accumulation.best_traj_index)
    results = _materialize_results_from_score_result(
        canonical=score_result.canonical,
        score_result=score_result,
        include_visibility=bool(debug),
    )

    if debug and debug_record_episode_metrics:
        _update_debug_episode_metrics(
            time_step=time_step,
            results=results,
            debug_episode_reset=debug_episode_reset,
        )

    if hasattr(_cpu_eval, "__entropy_eval_count"):
        _cpu_eval.__entropy_eval_count += 1

    _ = (
        debug_output_dir,
        debug_visibility_stride,
    )  # reserved for future debug plotting parity
    return best_traj_index, total_entropies, results


# evaluate_trajectories_by_entropy = evaluate_trajectories_by_entropy_gpu


def backend_available() -> bool:
    """Return True when the CUDA backend dependencies are importable."""
    return bool(_PYCUDA_AVAILABLE)


def backend_ready() -> bool:
    """Return True when the CUDA backend is ready to be selected."""
    return bool(_PYCUDA_AVAILABLE)


def backend_reason_unavailable() -> str:
    """Return a human-readable reason when the CUDA backend cannot be used."""
    if _PYCUDA_AVAILABLE:
        return ""
    return "PyCUDA is unavailable or no compatible CUDA runtime was found."


def __getattr__(name):
    """Fall back to the CPU reference module for API surface not overridden here."""
    if name.startswith("_selector_impl_"):
        raise AttributeError(name)
    return getattr(_cpu_eval, name)


def __dir__():
    """Expose the union of GPU overrides and the CPU compatibility surface."""
    return sorted(set(globals()) | set(dir(_cpu_eval)))
