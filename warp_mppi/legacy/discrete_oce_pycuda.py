"""PyCUDA backend for discrete-grid OCE scoring.

This module intentionally keeps the public surface small.  It accepts already
packed discrete OCE inputs, computes visibility/occlusion and exact entropy
scores on CUDA when PyCUDA is available, and otherwise raises a RuntimeError so
callers can fall back to the CPU oracle.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule

    PYCUDA_AVAILABLE = True
except Exception:  # pragma: no cover - depends on local CUDA install
    cuda = None
    SourceModule = None
    PYCUDA_AVAILABLE = False


TOLERANCE = 1.0e-10
DISCRETE_OCE_SCORING_MODES = {
    "entropy",
    "oc_entropy",
    "entropy_plus_information",
    "oc_entropy_plus_information",
    "information_only",
}

_MODULE_CONTEXT = None
_owns_module_context_ref = False
_module_context_pushed = False


@dataclass
class DiscreteOCEResult:
    best_trajectory: int
    scores: np.ndarray
    device_scores: Any | None = None
    step_entropy: np.ndarray | None = None
    step_probability: np.ndarray | None = None
    step_e_state: np.ndarray | None = None
    step_a_state: np.ndarray | None = None
    step_oc_entropy: np.ndarray | None = None
    step_total_entropy: np.ndarray | None = None
    step_total_oc_entropy: np.ndarray | None = None
    step_spatial_separation: np.ndarray | None = None
    per_agent_step_entropy: np.ndarray | None = None
    per_agent_step_probability: np.ndarray | None = None
    per_agent_step_e_state: np.ndarray | None = None
    per_agent_step_a_state: np.ndarray | None = None
    per_agent_step_oc_entropy: np.ndarray | None = None
    per_agent_step_total_entropy: np.ndarray | None = None
    per_agent_step_spatial_separation: np.ndarray | None = None
    step_belief_sums: np.ndarray | None = None
    # BUGBUG - removed
    # per_agent_score_components: np.ndarray | None = None
    # score_components: np.ndarray | None = None
    visibility_tensor: np.ndarray | None = None
    execution_path: str = "cuda_discrete_exact"
    metadata: dict | None = None


def normalize_discrete_oce_scoring_mode(scoring_mode: str | None) -> str:
    scoring_mode = str(scoring_mode or "information_only").strip().lower()
    if scoring_mode not in DISCRETE_OCE_SCORING_MODES:
        allowed = ", ".join(sorted(DISCRETE_OCE_SCORING_MODES))
        raise ValueError(
            f"Unsupported discrete OCE scoring mode: {scoring_mode!r}. "
            f"Expected one of: {allowed}."
        )
    return scoring_mode


def score_discrete_oce_components(
    *,
    scoring_mode: str,
    step_entropy: np.ndarray,
    step_oc_entropy: np.ndarray,
    step_e_state: np.ndarray,
    gamma: float = 0.95,
    alpha: float = 1.0,
) -> np.ndarray:
    scoring_mode = normalize_discrete_oce_scoring_mode(scoring_mode)
    step_entropy = np.asarray(step_entropy, dtype=np.float32)
    step_oc_entropy = np.asarray(step_oc_entropy, dtype=np.float32)
    step_e_state = np.asarray(step_e_state, dtype=np.float32)
    if step_entropy.ndim != 2:
        raise ValueError("step_entropy must have shape (num_paths, horizon)")
    horizon = step_entropy.shape[1]
    weights = (float(gamma) ** np.arange(horizon, dtype=np.float32))[np.newaxis, :]

    if scoring_mode == "entropy":
        components = step_entropy
    elif scoring_mode == "oc_entropy":
        components = step_oc_entropy
    elif scoring_mode == "entropy_plus_information":
        components = step_entropy - float(alpha) * step_e_state
    elif scoring_mode == "oc_entropy_plus_information":
        components = step_oc_entropy - float(alpha) * step_e_state
    elif scoring_mode == "information_only":
        components = -step_e_state
    else:  # pragma: no cover - normalize_discrete_oce_scoring_mode covers this.
        raise AssertionError(f"Unhandled scoring mode: {scoring_mode}")

    return np.sum(weights * components, axis=1).astype(np.float32, copy=False)


def _establish_module_context():
    global _MODULE_CONTEXT, _owns_module_context_ref, _module_context_pushed
    if _MODULE_CONTEXT is not None:
        return _MODULE_CONTEXT
    if not PYCUDA_AVAILABLE:
        raise RuntimeError("PyCUDA is not available for discrete OCE CUDA execution.")

    cuda.init()
    device = None
    try:
        current_context = cuda.Context.get_current()
        if current_context is not None:
            device = cuda.Context.get_device()
    except cuda.LogicError:
        pass
    if device is None:
        device = cuda.Device(0)

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
    if context is None:
        context = _establish_module_context()
        _pop_module_context_after_compile()
    context.push()
    try:
        yield
    finally:
        context.pop()


CUDA_SOURCE = r"""
extern "C" {
#include <math.h>

__device__ __forceinline__ float clamp_prob(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

__device__ __forceinline__ int world_to_col(
    float x,
    float origin_x,
    float resolution
) {
    return (int)floorf((x - origin_x) / resolution);
}

__device__ __forceinline__ int world_to_row(
    float y,
    float origin_y,
    float resolution
) {
    return (int)floorf((y - origin_y) / resolution);
}

__device__ int blocked_line(
    const unsigned char* static_grid,
    int rows,
    int cols,
    float origin_x,
    float origin_y,
    float resolution,
    float x0,
    float y0,
    float x1,
    float y1
) {
    int c0 = world_to_col(x0, origin_x, resolution);
    int r0 = world_to_row(y0, origin_y, resolution);
    int c1 = world_to_col(x1, origin_x, resolution);
    int r1 = world_to_row(y1, origin_y, resolution);
    if (r0 < 0 || r0 >= rows || c0 < 0 || c0 >= cols) return 1;
    if (r1 < 0 || r1 >= rows || c1 < 0 || c1 >= cols) return 1;

    int dc = abs(c1 - c0);
    int dr = abs(r1 - r0);
    int sc = c0 < c1 ? 1 : -1;
    int sr = r0 < r1 ? 1 : -1;
    int err = dc - dr;
    int c = c0;
    int r = r0;

    for (int iter = 0; iter < rows + cols + 4; ++iter) {
        if (r < 0 || r >= rows || c < 0 || c >= cols) return 1;
        if (static_grid[r * cols + c] != 0) return 1;
        if (c == c1 && r == r1) break;
        int e2 = 2 * err;
        if (e2 > -dr) {
            err -= dr;
            c += sc;
        }
        if (e2 < dc) {
            err += dc;
            r += sr;
        }
    }
    return 0;
}

__device__ __forceinline__ long long occlusion_offset(
    int path,
    int agent,
    int step,
    int state,
    int num_agents,
    int horizon_plus_one,
    int num_states
) {
    return (((long long)path * num_agents + agent) * horizon_plus_one + step) * num_states + state;
}

__device__ int blocked_line_with_occupancy(
    const unsigned char* static_grid,
    const float* occupancy_grids,
    const unsigned long long* owner_mask_grids,
    int occupancy_steps,
    float occupancy_threshold,
    unsigned long long target_owner_bit,
    int rows,
    int cols,
    float origin_x,
    float origin_y,
    float resolution,
    int step,
    float x0,
    float y0,
    float x1,
    float y1
) {
    int c0 = world_to_col(x0, origin_x, resolution);
    int r0 = world_to_row(y0, origin_y, resolution);
    int c1 = world_to_col(x1, origin_x, resolution);
    int r1 = world_to_row(y1, origin_y, resolution);
    if (r0 < 0 || r0 >= rows || c0 < 0 || c0 >= cols) return 1;
    if (r1 < 0 || r1 >= rows || c1 < 0 || c1 >= cols) return 1;

    int grid_step = step;
    if (grid_step < 0) grid_step = 0;
    if (grid_step >= occupancy_steps) grid_step = occupancy_steps - 1;

    int dc = abs(c1 - c0);
    int dr = abs(r1 - r0);
    int sc = c0 < c1 ? 1 : -1;
    int sr = r0 < r1 ? 1 : -1;
    int err = dc - dr;
    int c = c0;
    int r = r0;

    for (int iter = 0; iter < rows + cols + 4; ++iter) {
        if (r < 0 || r >= rows || c < 0 || c >= cols) return 1;
        int grid_idx = r * cols + c;
        if (static_grid != NULL && static_grid[grid_idx] != 0) return 1;
        if (occupancy_grids != NULL && occupancy_steps > 0) {
            float probability = occupancy_grids[(grid_step * rows + r) * cols + c];
            if (probability >= occupancy_threshold) {
                unsigned long long owners = 0ULL;
                if (owner_mask_grids != NULL) {
                    owners = owner_mask_grids[(grid_step * rows + r) * cols + c];
                }
                if (!(owners != 0ULL && target_owner_bit != 0ULL && (owners & ~target_owner_bit) == 0ULL)) {
                    return 1;
                }
            }
        }
        if (c == c1 && r == r1) break;
        int e2 = 2 * err;
        if (e2 > -dr) {
            err -= dr;
            c += sc;
        }
        if (e2 < dc) {
            err += dc;
            r += sr;
        }
    }
    return 0;
}

__global__ void compute_discrete_occlusion(
    const float* paths,
    const float* state_centers,
    const unsigned char* static_grid,
    const float* occupancy_grids,
    const unsigned long long* owner_mask_grids,
    const unsigned long long* agent_owner_bits,
    int num_paths,
    int num_agents,
    int horizon_plus_one,
    int num_states,
    int rows,
    int cols,
    float origin_x,
    float origin_y,
    float resolution,
    float scan_range,
    float occupancy_threshold,
    int occupancy_steps,
    unsigned char* occlusion
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_paths * num_agents * horizon_plus_one * num_states;
    if (idx >= total) return;

    int state = idx % num_states;
    int tmp = idx / num_states;
    int step = tmp % horizon_plus_one;
    tmp /= horizon_plus_one;
    int agent = tmp % num_agents;
    int path = tmp / num_agents;

    float ox = paths[(path * horizon_plus_one + step) * 2 + 0];
    float oy = paths[(path * horizon_plus_one + step) * 2 + 1];
    float sx = state_centers[state * 2 + 0];
    float sy = state_centers[state * 2 + 1];
    float dx = sx - ox;
    float dy = sy - oy;
    float dist2 = dx * dx + dy * dy;
    unsigned char occ = 0;
    if (dist2 > scan_range * scan_range) {
        occ = 1;
    } else if (
        occupancy_grids != NULL
        ? blocked_line_with_occupancy(
            static_grid,
            occupancy_grids,
            owner_mask_grids,
            occupancy_steps,
            occupancy_threshold,
            agent_owner_bits != NULL ? agent_owner_bits[agent] : 0ULL,
            rows,
            cols,
            origin_x,
            origin_y,
            resolution,
            step,
            ox,
            oy,
            sx,
            sy
        )
        : blocked_line(
            static_grid, rows, cols, origin_x, origin_y, resolution, ox, oy, sx, sy
        )
    ) {
        occ = 1;
    }
    occlusion[idx] = occ;
}

__global__ void compute_discrete_occlusion_from_rollouts(
    const float* rollout_states,
    const float* x_init,
    const float* state_centers,
    const unsigned char* static_grid,
    const float* occupancy_grids,
    const unsigned long long* owner_mask_grids,
    const unsigned long long* agent_owner_bits,
    int rollout_state_stride,
    int num_paths,
    int num_agents,
    int horizon_plus_one,
    int num_states,
    int rows,
    int cols,
    float origin_x,
    float origin_y,
    float resolution,
    float scan_range,
    float occupancy_threshold,
    int occupancy_steps,
    unsigned char* occlusion
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_paths * num_agents * horizon_plus_one * num_states;
    if (idx >= total) return;

    int state = idx % num_states;
    int tmp = idx / num_states;
    int step = tmp % horizon_plus_one;
    tmp /= horizon_plus_one;
    int agent = tmp % num_agents;
    int path = tmp / num_agents;

    float ox;
    float oy;
    if (step == 0) {
        ox = x_init[0];
        oy = x_init[1];
    } else {
        const float* rollout_state =
            rollout_states + (((long long)path * (horizon_plus_one - 1) + (step - 1)) * rollout_state_stride);
        ox = rollout_state[0];
        oy = rollout_state[1];
    }

    float sx = state_centers[state * 2 + 0];
    float sy = state_centers[state * 2 + 1];
    float dx = sx - ox;
    float dy = sy - oy;
    float dist2 = dx * dx + dy * dy;
    unsigned char occ = 0;
    if (dist2 > scan_range * scan_range) {
        occ = 1;
    } else if (
        occupancy_grids != NULL
        ? blocked_line_with_occupancy(
            static_grid,
            occupancy_grids,
            owner_mask_grids,
            occupancy_steps,
            occupancy_threshold,
            agent_owner_bits != NULL ? agent_owner_bits[agent] : 0ULL,
            rows,
            cols,
            origin_x,
            origin_y,
            resolution,
            step,
            ox,
            oy,
            sx,
            sy
        )
        : blocked_line(
            static_grid, rows, cols, origin_x, origin_y, resolution, ox, oy, sx, sy
        )
    ) {
        occ = 1;
    }
    occlusion[idx] = occ;
}

__device__ void dense_transition_step(
    const float* P,
    const float* src,
    float* dst,
    int num_states,
    int agent_idx
) {
    int tid = threadIdx.x;
    int block_threads = blockDim.x;
    const float* P_agent = P + ((long long)agent_idx) * num_states * num_states;
    for (int j = tid; j < num_states; j += block_threads) {
        float val = 0.0f;
        for (int i = 0; i < num_states; ++i) {
            val += src[i] * P_agent[i * num_states + j];
        }
        dst[j] = val;
    }
}

__device__ void csr_transition_step(
    const float* data,
    const int* indices,
    const int* indptr,
    const float* src,
    float* dst,
    int num_states,
    int agent_idx
) {
    int tid = threadIdx.x;
    int block_threads = blockDim.x;
    const int* indptr_agent = indptr + ((long long)agent_idx) * (num_states + 1);

    for (int j = tid; j < num_states; j += block_threads) {
        dst[j] = 0.0f;
    }
    __syncthreads();

    for (int row = tid; row < num_states; row += block_threads) {
        float mass = src[row];
        if (mass <= 1.0e-20f) {
            continue;
        }
        int start = indptr_agent[row];
        int end = indptr_agent[row + 1];
        for (int e = start; e < end; ++e) {
            atomicAdd(dst + indices[e], mass * data[e]);
        }
    }
}

__device__ void csr_transition_single_state_step(
    const float* data,
    const int* indices,
    const int* indptr,
    int state,
    float mass,
    float* dst,
    int num_states,
    int agent_idx
) {
    int tid = threadIdx.x;
    int block_threads = blockDim.x;
    const int* indptr_agent = indptr + ((long long)agent_idx) * (num_states + 1);

    for (int j = tid; j < num_states; j += block_threads) {
        dst[j] = 0.0f;
    }
    __syncthreads();

    int start = indptr_agent[state];
    int end = indptr_agent[state + 1];
    for (int e = start + tid; e < end; e += block_threads) {
        atomicAdd(dst + indices[e], mass * data[e]);
    }
}

__device__ float block_sum(float* scratch, float value) {
    int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ float partition_entropy(float* belief, int num_states, float* scratch) {
    int tid = threadIdx.x;
    float local_prob = 0.0f;
    for (int i = tid; i < num_states; i += blockDim.x) {
        local_prob += fmaxf(belief[i], 0.0f);
    }
    float prob = block_sum(scratch, local_prob);
    if (prob <= 1.0e-10f) {
        return 0.0f;
    }

    float local_entropy = 0.0f;
    for (int i = tid; i < num_states; i += blockDim.x) {
        float q = fmaxf(belief[i], 0.0f) / prob;
        if (q > 1.0e-10f) {
            local_entropy += -q * logf(q);
        }
    }
    float entropy = block_sum(scratch, local_entropy);
    return prob * entropy;
}

__device__ float partition_entropy_and_prob(
    float* belief,
    int num_states,
    float* scratch,
    float* prob_out
) {
    int tid = threadIdx.x;
    float local_prob = 0.0f;
    for (int i = tid; i < num_states; i += blockDim.x) {
        local_prob += fmaxf(belief[i], 0.0f);
    }
    float prob = block_sum(scratch, local_prob);
    *prob_out = prob;
    if (prob <= 1.0e-10f) {
        return 0.0f;
    }

    float local_entropy = 0.0f;
    for (int i = tid; i < num_states; i += blockDim.x) {
        float q = fmaxf(belief[i], 0.0f) / prob;
        if (q > 1.0e-10f) {
            local_entropy += -q * logf(q);
        }
    }
    float entropy = block_sum(scratch, local_entropy);
    return prob * entropy;
}

__device__ float partition_spatial_separation(
    float* belief,
    const float* state_coords,
    int num_states,
    float prob,
    float* scratch
) {
    if (state_coords == NULL || prob <= 1.0e-10f) {
        return 0.0f;
    }

    int tid = threadIdx.x;
    float local_weight = 0.0f;
    float local_x = 0.0f;
    float local_y = 0.0f;
    for (int i = tid; i < num_states; i += blockDim.x) {
        float q = fmaxf(belief[i], 0.0f) / prob;
        if (q > 1.0e-10f) {
            local_weight += q;
            local_x += q * state_coords[i * 2 + 0];
            local_y += q * state_coords[i * 2 + 1];
        }
    }
    float weight_total = block_sum(scratch, local_weight);
    if (weight_total <= 1.0e-10f) {
        return 0.0f;
    }
    float mean_x = block_sum(scratch, local_x) / weight_total;
    float mean_y = block_sum(scratch, local_y) / weight_total;

    float local_xx = 0.0f;
    float local_xy = 0.0f;
    float local_yy = 0.0f;
    for (int i = tid; i < num_states; i += blockDim.x) {
        float q = fmaxf(belief[i], 0.0f) / prob;
        if (q > 1.0e-10f) {
            float dx = state_coords[i * 2 + 0] - mean_x;
            float dy = state_coords[i * 2 + 1] - mean_y;
            float w = q / weight_total;
            local_xx += w * dx * dx;
            local_xy += w * dx * dy;
            local_yy += w * dy * dy;
        }
    }
    float cov_xx = block_sum(scratch, local_xx);
    float cov_xy = block_sum(scratch, local_xy);
    float cov_yy = block_sum(scratch, local_yy);

    float half_trace = 0.5f * (cov_xx + cov_yy);
    float half_delta = 0.5f * (cov_xx - cov_yy);
    float radius = sqrtf(fmaxf(0.0f, half_delta * half_delta + cov_xy * cov_xy));
    float lambda_hi = half_trace + radius;
    float lambda_lo = half_trace - radius;
    float regularized_trace = fmaxf(lambda_hi, 1.0e-6f) + fmaxf(lambda_lo, 1.0e-6f);
    return prob * regularized_trace;
}

__device__ float accumulate_partition_outputs(
    float* work,
    const float* state_coords,
    int path,
    int agent,
    int k,
    int num_agents,
    int horizon,
    int num_states,
    float* step_entropy,
    float* step_probability,
    float* step_spatial_separation,
    float* belief_sums,
    float* scratch
) {
    float prob = 0.0f;
    float contribution = partition_entropy_and_prob(work, num_states, scratch, &prob);
    if (prob <= 1.0e-10f) {
        return 0.0f;
    }

    long long detail_idx =
        (((long long)path * num_agents + agent) * horizon + (k - 1));
    if (threadIdx.x == 0) {
        atomicAdd(step_entropy + detail_idx, contribution);
        atomicAdd(step_probability + detail_idx, prob);
    }
    float separation = partition_spatial_separation(
        work,
        state_coords,
        num_states,
        prob,
        scratch
    );
    if (threadIdx.x == 0 && step_spatial_separation != NULL) {
        atomicAdd(step_spatial_separation + detail_idx, separation);
    }
    for (int i = threadIdx.x; i < num_states; i += blockDim.x) {
        float value = fmaxf(work[i], 0.0f);
        if (value > 0.0f) {
            atomicAdd(belief_sums + detail_idx * num_states + i, value);
        }
    }
    return contribution;
}

__global__ void finalize_discrete_entropy_details(
    const float* step_entropy,
    const float* step_probability,
    const float* belief_sums,
    int num_paths,
    int num_agents,
    int horizon,
    int num_states,
    float* step_a_state,
    float* step_e_state
) {
    int task = blockIdx.x;
    int tid = threadIdx.x;
    int total = num_paths * num_agents * horizon;
    if (task >= total) return;

    extern __shared__ float scratch[];
    float prob = step_probability[task];
    if (prob <= 1.0e-10f) {
        if (tid == 0) {
            step_a_state[task] = 0.0f;
            step_e_state[task] = 0.0f;
        }
        return;
    }

    const float* belief = belief_sums + ((long long)task) * num_states;
    float local_entropy = 0.0f;
    for (int i = tid; i < num_states; i += blockDim.x) {
        float q = fmaxf(belief[i], 0.0f) / prob;
        if (q > 1.0e-10f) {
            local_entropy += -q * logf(q);
        }
    }
    float mixed_state_entropy = block_sum(scratch, local_entropy);
    if (tid == 0) {
        float a_state = step_entropy[task] / prob;
        float e_state = mixed_state_entropy - a_state;
        if (e_state < 0.0f) {
            e_state = 0.0f;
        }
        step_a_state[task] = a_state;
        step_e_state[task] = e_state;
    }
}

__global__ void discrete_exact_entropy_scores(
    const float* transitions,
    const float* beliefs,
    const unsigned char* occlusion,
    float* workspace,
    int num_paths,
    int num_agents,
    int horizon,
    int num_states,
    float* path_scores
) {
    int task = blockIdx.x;
    int path = task / num_agents;
    int agent = task % num_agents;
    int tid = threadIdx.x;
    extern __shared__ float shared[];
    long long workspace_stride = ((long long)(horizon + 1) + 2LL) * num_states;
    float* prefix = workspace + ((long long)task) * workspace_stride;
    float* work = prefix + (horizon + 1) * num_states;
    float* next = work + num_states;
    float* scratch = shared;

    const float* belief0 = beliefs + agent * num_states;
    for (int i = tid; i < num_states; i += blockDim.x) {
        prefix[i] = belief0[i];
    }
    __syncthreads();

    for (int step = 1; step <= horizon; ++step) {
        float* prev = prefix + (step - 1) * num_states;
        float* out = prefix + step * num_states;
        dense_transition_step(transitions, prev, out, num_states, agent);
        __syncthreads();
    }

    float cumulative = 0.0f;
    for (int k = 1; k <= horizon; ++k) {
        // Unseen partition: transition then mask at every step 1..k.
        for (int i = tid; i < num_states; i += blockDim.x) {
            work[i] = belief0[i];
        }
        __syncthreads();
        for (int t = 1; t <= k; ++t) {
            dense_transition_step(transitions, work, next, num_states, agent);
            __syncthreads();
            for (int j = tid; j < num_states; j += blockDim.x) {
                int occ_idx = occlusion_offset(path, agent, t, j, num_agents, horizon + 1, num_states);
                next[j] *= (float)occlusion[occ_idx];
                work[j] = next[j];
            }
            __syncthreads();
        }
        cumulative += partition_entropy(work, num_states, scratch);
        __syncthreads();

        // Seen-at-state partitions for p=1..k-1, matching _exact_entropy.
        for (int p = 1; p < k; ++p) {
            for (int s = 0; s < num_states; ++s) {
                int occ_idx = occlusion_offset(path, agent, p, s, num_agents, horizon + 1, num_states);
                if (occlusion[occ_idx] != 0) {
                    continue;
                }
                float mass = prefix[p * num_states + s];
                if (mass <= 1.0e-10f) {
                    continue;
                }
                for (int i = tid; i < num_states; i += blockDim.x) {
                    work[i] = (i == s) ? mass : 0.0f;
                }
                __syncthreads();
                for (int t = p + 1; t <= k; ++t) {
                    dense_transition_step(transitions, work, next, num_states, agent);
                    __syncthreads();
                    for (int j = tid; j < num_states; j += blockDim.x) {
                        int future_occ_idx =
                            occlusion_offset(path, agent, t, j, num_agents, horizon + 1, num_states);
                        next[j] *= (float)occlusion[future_occ_idx];
                        work[j] = next[j];
                    }
                    __syncthreads();
                }
                cumulative += partition_entropy(work, num_states, scratch);
                __syncthreads();
            }
        }
    }

    if (tid == 0) {
        atomicAdd(path_scores + path, cumulative);
    }
}

__global__ void discrete_exact_entropy_partition_scores(
    const float* transitions,
    const float* prefix_beliefs,
    const unsigned char* occlusion,
    const float* state_coords,
    int num_paths,
    int num_agents,
    int horizon,
    int num_states,
    float* step_entropy,
    float* step_probability,
    float* step_spatial_separation,
    float* belief_sums,
    float* path_scores
) {
    int task = blockIdx.x;
    int tid = threadIdx.x;
    int unseen_tasks = num_paths * num_agents * horizon;
    extern __shared__ float shared[];
    float* work = shared;
    float* next = work + num_states;
    float* scratch = next + num_states;

    int path = 0;
    int agent = 0;
    int k = 0;
    int partition_step = 0;
    int partition_state = -1;

    if (task < unseen_tasks) {
        int rem = task;
        k = (rem % horizon) + 1;
        rem /= horizon;
        agent = rem % num_agents;
        path = rem / num_agents;
    } else {
        int rem = task - unseen_tasks;
        partition_state = rem % num_states;
        rem /= num_states;
        partition_step = (rem % horizon) + 1;
        rem /= horizon;
        k = (rem % horizon) + 1;
        rem /= horizon;
        agent = rem % num_agents;
        path = rem / num_agents;

        if (partition_step >= k) {
            return;
        }
        int occ_idx =
            occlusion_offset(path, agent, partition_step, partition_state, num_agents, horizon + 1, num_states);
        if (occlusion[occ_idx] != 0) {
            return;
        }
    }

    const float* prefix_agent = prefix_beliefs + agent * (horizon + 1) * num_states;
    if (partition_step == 0) {
        const float* belief0 = prefix_agent;
        for (int i = tid; i < num_states; i += blockDim.x) {
            work[i] = belief0[i];
        }
        __syncthreads();
        for (int t = 1; t <= k; ++t) {
            dense_transition_step(transitions, work, next, num_states, agent);
            __syncthreads();
            for (int j = tid; j < num_states; j += blockDim.x) {
                int occ_idx = occlusion_offset(path, agent, t, j, num_agents, horizon + 1, num_states);
                next[j] *= (float)occlusion[occ_idx];
                work[j] = next[j];
            }
            __syncthreads();
        }
    } else {
        float mass = prefix_agent[partition_step * num_states + partition_state];
        if (mass <= 1.0e-10f) {
            return;
        }
        for (int i = tid; i < num_states; i += blockDim.x) {
            work[i] = (i == partition_state) ? mass : 0.0f;
        }
        __syncthreads();
        for (int t = partition_step + 1; t <= k; ++t) {
            dense_transition_step(transitions, work, next, num_states, agent);
            __syncthreads();
            for (int j = tid; j < num_states; j += blockDim.x) {
                int occ_idx = occlusion_offset(path, agent, t, j, num_agents, horizon + 1, num_states);
                next[j] *= (float)occlusion[occ_idx];
                work[j] = next[j];
            }
            __syncthreads();
        }
    }

    float contribution = accumulate_partition_outputs(
        work,
        state_coords,
        path,
        agent,
        k,
        num_agents,
        horizon,
        num_states,
        step_entropy,
        step_probability,
        step_spatial_separation,
        belief_sums,
        scratch
    );
    if (tid == 0 && contribution > 0.0f) {
        atomicAdd(path_scores + path, contribution);
    }
}

__global__ void discrete_exact_entropy_partition_scores_csr(
    const float* transition_data,
    const int* transition_indices,
    const int* transition_indptr,
    const float* prefix_beliefs,
    const unsigned char* occlusion,
    const float* state_coords,
    int num_paths,
    int num_agents,
    int horizon,
    int num_states,
    float* step_entropy,
    float* step_probability,
    float* step_spatial_separation,
    float* belief_sums,
    float* path_scores
) {
    int task = blockIdx.x;
    int tid = threadIdx.x;
    int unseen_tasks = num_paths * num_agents * horizon;
    extern __shared__ float shared[];
    float* work = shared;
    float* next = work + num_states;
    float* scratch = next + num_states;

    int path = 0;
    int agent = 0;
    int k = 0;
    int partition_step = 0;
    int partition_state = -1;

    if (task < unseen_tasks) {
        int rem = task;
        k = (rem % horizon) + 1;
        rem /= horizon;
        agent = rem % num_agents;
        path = rem / num_agents;
    } else {
        int rem = task - unseen_tasks;
        partition_state = rem % num_states;
        rem /= num_states;
        partition_step = (rem % horizon) + 1;
        rem /= horizon;
        k = (rem % horizon) + 1;
        rem /= horizon;
        agent = rem % num_agents;
        path = rem / num_agents;

        if (partition_step >= k) {
            return;
        }
        int occ_idx =
            occlusion_offset(path, agent, partition_step, partition_state, num_agents, horizon + 1, num_states);
        if (occlusion[occ_idx] != 0) {
            return;
        }
    }

    const float* prefix_agent = prefix_beliefs + agent * (horizon + 1) * num_states;
    if (partition_step == 0) {
        const float* belief0 = prefix_agent;
        for (int i = tid; i < num_states; i += blockDim.x) {
            work[i] = belief0[i];
        }
        __syncthreads();
        for (int t = 1; t <= k; ++t) {
            csr_transition_step(
                transition_data,
                transition_indices,
                transition_indptr,
                work,
                next,
                num_states,
                agent
            );
            __syncthreads();
            for (int j = tid; j < num_states; j += blockDim.x) {
                int occ_idx = occlusion_offset(path, agent, t, j, num_agents, horizon + 1, num_states);
                next[j] *= (float)occlusion[occ_idx];
                work[j] = next[j];
            }
            __syncthreads();
        }
    } else {
        float mass = prefix_agent[partition_step * num_states + partition_state];
        if (mass <= 1.0e-10f) {
            return;
        }
        csr_transition_single_state_step(
            transition_data,
            transition_indices,
            transition_indptr,
            partition_state,
            mass,
            work,
            num_states,
            agent
        );
        __syncthreads();
        for (int j = tid; j < num_states; j += blockDim.x) {
            int occ_idx = occlusion_offset(path, agent, partition_step + 1, j, num_agents, horizon + 1, num_states);
            work[j] *= (float)occlusion[occ_idx];
        }
        __syncthreads();
        for (int t = partition_step + 2; t <= k; ++t) {
            csr_transition_step(
                transition_data,
                transition_indices,
                transition_indptr,
                work,
                next,
                num_states,
                agent
            );
            __syncthreads();
            for (int j = tid; j < num_states; j += blockDim.x) {
                int occ_idx = occlusion_offset(path, agent, t, j, num_agents, horizon + 1, num_states);
                next[j] *= (float)occlusion[occ_idx];
                work[j] = next[j];
            }
            __syncthreads();
        }
    }

    float contribution = accumulate_partition_outputs(
        work,
        state_coords,
        path,
        agent,
        k,
        num_agents,
        horizon,
        num_states,
        step_entropy,
        step_probability,
        step_spatial_separation,
        belief_sums,
        scratch
    );
    if (tid == 0 && contribution > 0.0f) {
        atomicAdd(path_scores + path, contribution);
    }
}

__global__ void discrete_exact_entropy_active_partition_scores_csr(
    const float* transition_data,
    const int* transition_indices,
    const int* transition_indptr,
    const float* prefix_beliefs,
    const unsigned char* occlusion,
    const float* state_coords,
    const int* partition_paths,
    const int* partition_agents,
    const int* partition_ks,
    const int* partition_steps,
    const int* partition_states,
    int num_active_partitions,
    int num_paths,
    int num_agents,
    int horizon,
    int num_states,
    float* step_entropy,
    float* step_probability,
    float* step_spatial_separation,
    float* belief_sums,
    float* path_scores
) {
    int task = blockIdx.x;
    int tid = threadIdx.x;
    int unseen_tasks = num_paths * num_agents * horizon;
    extern __shared__ float shared[];
    float* work = shared;
    float* next = work + num_states;
    float* scratch = next + num_states;

    int path = 0;
    int agent = 0;
    int k = 0;
    int partition_step = 0;
    int partition_state = -1;

    if (task < unseen_tasks) {
        int rem = task;
        k = (rem % horizon) + 1;
        rem /= horizon;
        agent = rem % num_agents;
        path = rem / num_agents;
    } else {
        int active_idx = task - unseen_tasks;
        if (active_idx >= num_active_partitions) {
            return;
        }
        path = partition_paths[active_idx];
        agent = partition_agents[active_idx];
        k = partition_ks[active_idx];
        partition_step = partition_steps[active_idx];
        partition_state = partition_states[active_idx];
    }

    const float* prefix_agent = prefix_beliefs + agent * (horizon + 1) * num_states;
    if (partition_step == 0) {
        const float* belief0 = prefix_agent;
        for (int i = tid; i < num_states; i += blockDim.x) {
            work[i] = belief0[i];
        }
        __syncthreads();
        for (int t = 1; t <= k; ++t) {
            csr_transition_step(
                transition_data,
                transition_indices,
                transition_indptr,
                work,
                next,
                num_states,
                agent
            );
            __syncthreads();
            for (int j = tid; j < num_states; j += blockDim.x) {
                int occ_idx = occlusion_offset(path, agent, t, j, num_agents, horizon + 1, num_states);
                next[j] *= (float)occlusion[occ_idx];
                work[j] = next[j];
            }
            __syncthreads();
        }
    } else {
        float mass = prefix_agent[partition_step * num_states + partition_state];
        if (mass <= 1.0e-10f) {
            return;
        }
        csr_transition_single_state_step(
            transition_data,
            transition_indices,
            transition_indptr,
            partition_state,
            mass,
            work,
            num_states,
            agent
        );
        __syncthreads();
        for (int j = tid; j < num_states; j += blockDim.x) {
            int occ_idx = occlusion_offset(path, agent, partition_step + 1, j, num_agents, horizon + 1, num_states);
            work[j] *= (float)occlusion[occ_idx];
        }
        __syncthreads();
        for (int t = partition_step + 2; t <= k; ++t) {
            csr_transition_step(
                transition_data,
                transition_indices,
                transition_indptr,
                work,
                next,
                num_states,
                agent
            );
            __syncthreads();
            for (int j = tid; j < num_states; j += blockDim.x) {
                int occ_idx = occlusion_offset(path, agent, t, j, num_agents, horizon + 1, num_states);
                next[j] *= (float)occlusion[occ_idx];
                work[j] = next[j];
            }
            __syncthreads();
        }
    }

    float contribution = accumulate_partition_outputs(
        work,
        state_coords,
        path,
        agent,
        k,
        num_agents,
        horizon,
        num_states,
        step_entropy,
        step_probability,
        step_spatial_separation,
        belief_sums,
        scratch
    );
    if (tid == 0 && contribution > 0.0f) {
        atomicAdd(path_scores + path, contribution);
    }
}

__global__ void discrete_approximate_entropy_partition_scores(
    const float* transitions,
    const float* prefix_beliefs,
    const unsigned char* occlusion,
    const float* state_coords,
    int num_paths,
    int num_agents,
    int horizon,
    int num_states,
    float* step_entropy,
    float* step_probability,
    float* step_spatial_separation,
    float* belief_sums,
    float* path_scores
) {
    int task = blockIdx.x;
    int tid = threadIdx.x;
    int unseen_tasks = num_paths * num_agents * horizon;
    extern __shared__ float shared[];
    float* work = shared;
    float* next = work + num_states;
    float* scratch = next + num_states;

    int path = 0;
    int agent = 0;
    int k = 0;
    int partition_step = 0;

    if (task < unseen_tasks) {
        int rem = task;
        k = (rem % horizon) + 1;
        rem /= horizon;
        agent = rem % num_agents;
        path = rem / num_agents;
    } else {
        int rem = task - unseen_tasks;
        partition_step = (rem % horizon) + 1;
        rem /= horizon;
        k = (rem % horizon) + 1;
        rem /= horizon;
        agent = rem % num_agents;
        path = rem / num_agents;
        if (partition_step >= k) {
            return;
        }
    }

    const float* prefix_agent = prefix_beliefs + agent * (horizon + 1) * num_states;
    if (partition_step == 0) {
        const float* belief0 = prefix_agent;
        for (int i = tid; i < num_states; i += blockDim.x) {
            work[i] = belief0[i];
        }
        __syncthreads();
        for (int t = 1; t <= k; ++t) {
            dense_transition_step(transitions, work, next, num_states, agent);
            __syncthreads();
            for (int j = tid; j < num_states; j += blockDim.x) {
                int occ_idx = occlusion_offset(path, agent, t, j, num_agents, horizon + 1, num_states);
                next[j] *= (float)occlusion[occ_idx];
                work[j] = next[j];
            }
            __syncthreads();
        }
    } else {
        for (int i = tid; i < num_states; i += blockDim.x) {
            int occ_idx = occlusion_offset(path, agent, partition_step, i, num_agents, horizon + 1, num_states);
            work[i] = (occlusion[occ_idx] == 0) ? prefix_agent[partition_step * num_states + i] : 0.0f;
        }
        __syncthreads();
        for (int t = partition_step + 1; t <= k; ++t) {
            dense_transition_step(transitions, work, next, num_states, agent);
            __syncthreads();
            for (int j = tid; j < num_states; j += blockDim.x) {
                int occ_idx = occlusion_offset(path, agent, t, j, num_agents, horizon + 1, num_states);
                next[j] *= (float)occlusion[occ_idx];
                work[j] = next[j];
            }
            __syncthreads();
        }
    }

    float contribution = accumulate_partition_outputs(
        work,
        state_coords,
        path,
        agent,
        k,
        num_agents,
        horizon,
        num_states,
        step_entropy,
        step_probability,
        step_spatial_separation,
        belief_sums,
        scratch
    );
    if (tid == 0 && contribution > 0.0f) {
        atomicAdd(path_scores + path, contribution);
    }
}

__global__ void discrete_approximate_entropy_partition_scores_csr(
    const float* transition_data,
    const int* transition_indices,
    const int* transition_indptr,
    const float* prefix_beliefs,
    const unsigned char* occlusion,
    const float* state_coords,
    int num_paths,
    int num_agents,
    int horizon,
    int num_states,
    float* step_entropy,
    float* step_probability,
    float* step_spatial_separation,
    float* belief_sums,
    float* path_scores
) {
    int task = blockIdx.x;
    int tid = threadIdx.x;
    int unseen_tasks = num_paths * num_agents * horizon;
    extern __shared__ float shared[];
    float* work = shared;
    float* next = work + num_states;
    float* scratch = next + num_states;

    int path = 0;
    int agent = 0;
    int k = 0;
    int partition_step = 0;

    if (task < unseen_tasks) {
        int rem = task;
        k = (rem % horizon) + 1;
        rem /= horizon;
        agent = rem % num_agents;
        path = rem / num_agents;
    } else {
        int rem = task - unseen_tasks;
        partition_step = (rem % horizon) + 1;
        rem /= horizon;
        k = (rem % horizon) + 1;
        rem /= horizon;
        agent = rem % num_agents;
        path = rem / num_agents;
        if (partition_step >= k) {
            return;
        }
    }

    const float* prefix_agent = prefix_beliefs + agent * (horizon + 1) * num_states;
    if (partition_step == 0) {
        const float* belief0 = prefix_agent;
        for (int i = tid; i < num_states; i += blockDim.x) {
            work[i] = belief0[i];
        }
        __syncthreads();
        for (int t = 1; t <= k; ++t) {
            csr_transition_step(
                transition_data,
                transition_indices,
                transition_indptr,
                work,
                next,
                num_states,
                agent
            );
            __syncthreads();
            for (int j = tid; j < num_states; j += blockDim.x) {
                int occ_idx = occlusion_offset(path, agent, t, j, num_agents, horizon + 1, num_states);
                next[j] *= (float)occlusion[occ_idx];
                work[j] = next[j];
            }
            __syncthreads();
        }
    } else {
        for (int i = tid; i < num_states; i += blockDim.x) {
            int occ_idx = occlusion_offset(path, agent, partition_step, i, num_agents, horizon + 1, num_states);
            work[i] = (occlusion[occ_idx] == 0) ? prefix_agent[partition_step * num_states + i] : 0.0f;
        }
        __syncthreads();
        for (int t = partition_step + 1; t <= k; ++t) {
            csr_transition_step(
                transition_data,
                transition_indices,
                transition_indptr,
                work,
                next,
                num_states,
                agent
            );
            __syncthreads();
            for (int j = tid; j < num_states; j += blockDim.x) {
                int occ_idx = occlusion_offset(path, agent, t, j, num_agents, horizon + 1, num_states);
                next[j] *= (float)occlusion[occ_idx];
                work[j] = next[j];
            }
            __syncthreads();
        }
    }

    float contribution = accumulate_partition_outputs(
        work,
        state_coords,
        path,
        agent,
        k,
        num_agents,
        horizon,
        num_states,
        step_entropy,
        step_probability,
        step_spatial_separation,
        belief_sums,
        scratch
    );
    if (tid == 0 && contribution > 0.0f) {
        atomicAdd(path_scores + path, contribution);
    }
}

__global__ void discrete_exact_entropy_active_partition_scores_csr_score_only(
    const float* transition_data,
    const int* transition_indices,
    const int* transition_indptr,
    const float* prefix_beliefs,
    const unsigned char* occlusion,
    const int* partition_paths,
    const int* partition_agents,
    const int* partition_ks,
    const int* partition_steps,
    const int* partition_states,
    int num_active_partitions,
    int num_paths,
    int num_agents,
    int horizon,
    int num_states,
    float* path_scores
) {
    int task = blockIdx.x;
    int tid = threadIdx.x;
    int unseen_tasks = num_paths * num_agents * horizon;
    extern __shared__ float shared[];
    float* work = shared;
    float* next = work + num_states;
    float* scratch = next + num_states;

    int path = 0;
    int agent = 0;
    int k = 0;
    int partition_step = 0;
    int partition_state = -1;

    if (task < unseen_tasks) {
        int rem = task;
        k = (rem % horizon) + 1;
        rem /= horizon;
        agent = rem % num_agents;
        path = rem / num_agents;
    } else {
        int active_idx = task - unseen_tasks;
        if (active_idx >= num_active_partitions) {
            return;
        }
        path = partition_paths[active_idx];
        agent = partition_agents[active_idx];
        k = partition_ks[active_idx];
        partition_step = partition_steps[active_idx];
        partition_state = partition_states[active_idx];
    }

    const float* prefix_agent = prefix_beliefs + agent * (horizon + 1) * num_states;
    if (partition_step == 0) {
        const float* belief0 = prefix_agent;
        for (int i = tid; i < num_states; i += blockDim.x) {
            work[i] = belief0[i];
        }
        __syncthreads();
        for (int t = 1; t <= k; ++t) {
            csr_transition_step(
                transition_data,
                transition_indices,
                transition_indptr,
                work,
                next,
                num_states,
                agent
            );
            __syncthreads();
            for (int j = tid; j < num_states; j += blockDim.x) {
                int occ_idx = occlusion_offset(path, agent, t, j, num_agents, horizon + 1, num_states);
                next[j] *= (float)occlusion[occ_idx];
                work[j] = next[j];
            }
            __syncthreads();
        }
    } else {
        float mass = prefix_agent[partition_step * num_states + partition_state];
        if (mass <= 1.0e-10f) {
            return;
        }
        csr_transition_single_state_step(
            transition_data,
            transition_indices,
            transition_indptr,
            partition_state,
            mass,
            work,
            num_states,
            agent
        );
        __syncthreads();
        for (int j = tid; j < num_states; j += blockDim.x) {
            int occ_idx = occlusion_offset(path, agent, partition_step + 1, j, num_agents, horizon + 1, num_states);
            work[j] *= (float)occlusion[occ_idx];
        }
        __syncthreads();
        for (int t = partition_step + 2; t <= k; ++t) {
            csr_transition_step(
                transition_data,
                transition_indices,
                transition_indptr,
                work,
                next,
                num_states,
                agent
            );
            __syncthreads();
            for (int j = tid; j < num_states; j += blockDim.x) {
                int occ_idx = occlusion_offset(path, agent, t, j, num_agents, horizon + 1, num_states);
                next[j] *= (float)occlusion[occ_idx];
                work[j] = next[j];
            }
            __syncthreads();
        }
    }

    float contribution = partition_entropy(work, num_states, scratch);
    if (tid == 0 && contribution > 0.0f) {
        atomicAdd(path_scores + path, contribution);
    }
}

} // extern "C"
"""


@lru_cache(maxsize=1)
def _compiled_module():
    if not PYCUDA_AVAILABLE:
        raise RuntimeError("PyCUDA is not available for discrete OCE CUDA execution.")
    context = _establish_module_context()
    with _active_cuda_context(context):
        module = SourceModule(CUDA_SOURCE, options=["--use_fast_math"])
    _pop_module_context_after_compile()
    return module


def _alloc_and_copy(array):
    array = np.ascontiguousarray(array)
    buf = cuda.mem_alloc(array.nbytes)
    cuda.memcpy_htod(buf, array)
    return buf


def _device_attribute(device, attribute, default=None):
    try:
        return int(device.get_attribute(attribute))
    except Exception:
        return default


def _configure_dynamic_shared_memory(
    function, required_bytes: int, *, label: str
) -> None:
    """Opt into larger dynamic shared memory, or raise before cuLaunchKernel does."""
    required_bytes = int(required_bytes)
    if required_bytes <= 0:
        return

    device = cuda.Context.get_device()
    default_limit = _device_attribute(
        device,
        cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK,
        48 * 1024,
    )
    optin_limit = _device_attribute(
        device,
        cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        default_limit,
    )
    max_limit = max(default_limit, optin_limit)

    if required_bytes <= default_limit:
        return
    if required_bytes > max_limit:
        raise RuntimeError(
            f"{label} requires {required_bytes} bytes of dynamic shared memory per block, "
            f"but device {device.name()} supports at most {max_limit} bytes "
            f"(default {default_limit}, opt-in {optin_limit}). "
            "Use a coarser SDD grid, fewer states, or the CPU discrete OCE backend."
        )

    try:
        function.set_attribute(
            cuda.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES,
            required_bytes,
        )
    except Exception as exc:
        raise RuntimeError(
            f"{label} requires {required_bytes} bytes of dynamic shared memory per block, "
            f"which exceeds the default device limit of {default_limit} bytes. "
            f"The device reports an opt-in limit of {optin_limit} bytes, but PyCUDA/CUDA "
            f"could not enable it: {exc}"
        ) from exc


def _build_active_partition_index(
    occlusion: np.ndarray,
    prefix_beliefs: np.ndarray,
    *,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return compact seen-partition tasks that can contribute nonzero entropy."""
    if occlusion.ndim == 3:
        num_paths, _, num_states = occlusion.shape
        agent_specific_occlusion = False
    elif occlusion.ndim == 4:
        num_paths, num_occlusion_agents, _, num_states = occlusion.shape
        agent_specific_occlusion = True
    else:
        raise ValueError(
            "occlusion must have shape (paths, steps, states) or (paths, agents, steps, states)"
        )
    num_agents = prefix_beliefs.shape[0]
    if agent_specific_occlusion and num_occlusion_agents != num_agents:
        raise ValueError("agent-specific occlusion does not match prefix_beliefs")
    horizon = int(horizon)

    path_chunks = []
    agent_chunks = []
    k_chunks = []
    step_chunks = []
    state_chunks = []

    for path_idx in range(num_paths):
        for agent_idx in range(num_agents):
            path_occlusion = (
                occlusion[path_idx, agent_idx]
                if agent_specific_occlusion
                else occlusion[path_idx]
            )
            agent_prefix = prefix_beliefs[agent_idx]
            for partition_step in range(1, horizon):
                active_states = np.flatnonzero(
                    (path_occlusion[partition_step] == 0)
                    & (agent_prefix[partition_step] > TOLERANCE)
                ).astype(np.int32, copy=False)
                if active_states.size == 0:
                    continue

                suffix_ks = np.arange(
                    partition_step + 1,
                    horizon + 1,
                    dtype=np.int32,
                )
                repeat = int(suffix_ks.size)
                states = np.tile(active_states, repeat)
                count = int(states.size)

                path_chunks.append(np.full(count, path_idx, dtype=np.int32))
                agent_chunks.append(np.full(count, agent_idx, dtype=np.int32))
                k_chunks.append(np.repeat(suffix_ks, active_states.size))
                step_chunks.append(np.full(count, partition_step, dtype=np.int32))
                state_chunks.append(states)

    if not state_chunks:
        empty = np.zeros((0,), dtype=np.int32)
        return empty, empty, empty, empty, empty

    return (
        np.concatenate(path_chunks).astype(np.int32, copy=False),
        np.concatenate(agent_chunks).astype(np.int32, copy=False),
        np.concatenate(k_chunks).astype(np.int32, copy=False),
        np.concatenate(step_chunks).astype(np.int32, copy=False),
        np.concatenate(state_chunks).astype(np.int32, copy=False),
    )


def _evaluate_discrete_oce_gpu_impl(
    *,
    paths: np.ndarray | None = None,
    rollout_states_d=None,
    num_rollouts: int | None = None,
    x_init: np.ndarray | None = None,
    rollout_state_stride: int = 4,
    state_centers: np.ndarray,
    state_coords: np.ndarray | None = None,
    static_grid: np.ndarray,
    grid_origin: tuple[float, float],
    grid_resolution: float,
    transition_matrices: np.ndarray | None = None,
    transition_data: np.ndarray | None = None,
    transition_indices: np.ndarray | None = None,
    transition_indptr: np.ndarray | None = None,
    prefix_beliefs: np.ndarray | None = None,
    beliefs: np.ndarray,
    horizon: int,
    scan_range: float,
    return_visibility: bool = False,
    return_belief_sums: bool = False,
    occupancy_probability_grids: np.ndarray | None = None,
    occupancy_owner_mask_grids: np.ndarray | None = None,
    agent_owner_bits: np.ndarray | None = None,
    occupancy_threshold: float = 0.25,
    return_device_scores: bool = False,
    compact_active_partitions: bool = True,
    entropy_method: str = "discrete_exact_entropy",
    scoring_mode: str = "information_only",
) -> DiscreteOCEResult:
    if not PYCUDA_AVAILABLE:
        raise RuntimeError("PyCUDA is not available for discrete OCE CUDA execution.")

    if paths is not None and rollout_states_d is not None:
        raise ValueError("provide either paths or rollout_states_d, not both")
    if paths is None and rollout_states_d is None:
        raise ValueError("paths or rollout_states_d is required")
    if paths is not None:
        paths = np.ascontiguousarray(paths, dtype=np.float32)
    else:
        if num_rollouts is None:
            raise ValueError("num_rollouts is required with rollout_states_d")
        if x_init is None:
            raise ValueError("x_init is required with rollout_states_d")
        x_init = np.ascontiguousarray(np.asarray(x_init, dtype=np.float32).reshape(-1))
        if x_init.size < 2:
            raise ValueError("x_init must contain at least x and y")
        rollout_state_stride = int(rollout_state_stride)
        if rollout_state_stride < 2:
            raise ValueError("rollout_state_stride must be at least 2")
    entropy_method = str(entropy_method or "discrete_exact_entropy").lower()
    scoring_mode = normalize_discrete_oce_scoring_mode(scoring_mode)
    if entropy_method not in {
        "discrete_exact_entropy",
        "exact",
        "approximate_entropy",
        "approximate",
    }:
        raise ValueError(
            "entropy_method must be one of {'discrete_exact_entropy', 'exact', "
            "'approximate_entropy', 'approximate'}"
        )
    use_approximate_entropy = entropy_method in {"approximate_entropy", "approximate"}

    state_centers = np.ascontiguousarray(state_centers, dtype=np.float32)
    if state_coords is not None:
        state_coords = np.ascontiguousarray(np.asarray(state_coords, dtype=np.float32))
    static_grid = np.ascontiguousarray(static_grid.astype(np.uint8, copy=False))
    beliefs = np.ascontiguousarray(beliefs, dtype=np.float32)
    use_occupancy = occupancy_probability_grids is not None
    if use_occupancy:
        occupancy_probability_grids = np.ascontiguousarray(
            np.asarray(occupancy_probability_grids, dtype=np.float32)
        )
        if occupancy_probability_grids.ndim != 3:
            raise ValueError(
                "occupancy_probability_grids must have shape (steps, rows, cols)"
            )
        if occupancy_probability_grids.shape[1:] != static_grid.shape:
            raise ValueError("occupancy grid shape must match static_grid")
        if occupancy_owner_mask_grids is None:
            occupancy_owner_mask_grids = np.zeros(
                occupancy_probability_grids.shape,
                dtype=np.uint64,
            )
        else:
            occupancy_owner_mask_grids = np.ascontiguousarray(
                np.asarray(occupancy_owner_mask_grids, dtype=np.uint64)
            )
            if occupancy_owner_mask_grids.shape != occupancy_probability_grids.shape:
                raise ValueError(
                    "occupancy_owner_mask_grids must match occupancy_probability_grids"
                )
    else:
        occupancy_probability_grids = None
        occupancy_owner_mask_grids = None
    use_csr = transition_data is not None
    if use_csr:
        if transition_indices is None or transition_indptr is None:
            raise ValueError("transition_indices and transition_indptr are required")
        transition_data = np.ascontiguousarray(transition_data, dtype=np.float32)
        transition_indices = np.ascontiguousarray(transition_indices, dtype=np.int32)
        transition_indptr = np.ascontiguousarray(transition_indptr, dtype=np.int32)
    else:
        if transition_matrices is None:
            raise ValueError(
                "transition_matrices or CSR transition_data/indices/indptr are required"
            )
        transition_matrices = np.ascontiguousarray(
            transition_matrices, dtype=np.float32
        )

    if paths is not None and (paths.ndim != 3 or paths.shape[2] != 2):
        raise ValueError("paths must have shape (num_paths, horizon + 1, 2)")
    if state_centers.ndim != 2 or state_centers.shape[1] != 2:
        raise ValueError("state_centers must have shape (num_states, 2)")
    if state_coords is not None and (
        state_coords.ndim != 2 or state_coords.shape[1] != 2
    ):
        raise ValueError("state_coords must have shape (num_states, 2)")
    if not use_csr and transition_matrices.ndim != 3:
        raise ValueError("transition_matrices must have shape (num_agents, N, N)")
    if beliefs.ndim != 2:
        raise ValueError("beliefs must have shape (num_agents, N)")

    if paths is not None:
        num_paths = int(paths.shape[0])
        horizon_plus_one = int(paths.shape[1])
        horizon = min(int(horizon), horizon_plus_one - 1)
    else:
        num_paths = int(num_rollouts)
        horizon = int(horizon)
        horizon_plus_one = horizon + 1
    num_states = int(state_centers.shape[0])
    if state_coords is not None and int(state_coords.shape[0]) != num_states:
        raise ValueError("state_coords dimensions do not match state_centers")
    num_agents = int(beliefs.shape[0])
    rows, cols = static_grid.shape

    # BUGBUG - score_only mode is only available for exact_entropy at this time
    score_only = bool(
        not use_approximate_entropy
        and return_device_scores
        and use_csr
        and compact_active_partitions
        and not return_visibility
        and not return_belief_sums
    )

    if agent_owner_bits is None:
        agent_owner_bits = np.zeros((num_agents,), dtype=np.uint64)
    else:
        agent_owner_bits = np.ascontiguousarray(
            np.asarray(agent_owner_bits, dtype=np.uint64)
        )
        if agent_owner_bits.shape != (num_agents,):
            raise ValueError("agent_owner_bits must have shape (num_agents,)")
    num_active_partitions = None
    score_block_count = 0

    if num_paths == 0 or num_agents == 0 or horizon <= 0:
        return DiscreteOCEResult(
            best_trajectory=0,
            scores=np.zeros((num_paths,), dtype=np.float32),
            step_entropy=np.zeros((num_paths, 0), dtype=np.float32),
            step_probability=np.zeros((num_paths, 0), dtype=np.float32),
            step_e_state=np.zeros((num_paths, 0), dtype=np.float32),
            step_a_state=np.zeros((num_paths, 0), dtype=np.float32),
            step_oc_entropy=np.zeros((num_paths, 0), dtype=np.float32),
            step_total_entropy=np.zeros((num_paths, 0), dtype=np.float32),
            step_total_oc_entropy=np.zeros((num_paths, 0), dtype=np.float32),
            step_spatial_separation=np.zeros((num_paths, 0), dtype=np.float32),
            per_agent_step_entropy=np.zeros(
                (num_paths, num_agents, 0), dtype=np.float32
            ),
            per_agent_step_probability=np.zeros(
                (num_paths, num_agents, 0), dtype=np.float32
            ),
            per_agent_step_e_state=np.zeros(
                (num_paths, num_agents, 0), dtype=np.float32
            ),
            per_agent_step_a_state=np.zeros(
                (num_paths, num_agents, 0), dtype=np.float32
            ),
            per_agent_step_oc_entropy=np.zeros(
                (num_paths, num_agents, 0), dtype=np.float32
            ),
            per_agent_step_total_entropy=np.zeros(
                (num_paths, num_agents, 0), dtype=np.float32
            ),
            per_agent_step_spatial_separation=np.zeros(
                (num_paths, num_agents, 0), dtype=np.float32
            ),
            step_belief_sums=(
                np.zeros((num_paths, num_agents, 0, num_states), dtype=np.float32)
                if return_belief_sums
                else None
            ),
            # per_agent_score_components=np.zeros(
            #     (num_paths, num_agents, 0, 7), dtype=np.float32
            # ),
            # score_components=np.zeros((num_paths, 0, 7), dtype=np.float32),
            visibility_tensor=None,
            execution_path=(
                "cuda_discrete_approximate_empty"
                if use_approximate_entropy
                else "cuda_discrete_exact_empty"
            ),
            metadata={
                "score_blocks": 0,
                "active_partitions": 0,
                "entropy_method": entropy_method,
            },
        )
    if num_states != beliefs.shape[1]:
        raise ValueError("belief dimensions do not match state_centers")
    if use_csr:
        if transition_indptr.shape != (num_agents, num_states + 1):
            raise ValueError("transition_indptr must have shape (num_agents, N + 1)")
    elif transition_matrices.shape != (num_agents, num_states, num_states):
        raise ValueError("transition dimensions do not match state_centers")

    module = _compiled_module()
    compute_occ = module.get_function(
        "compute_discrete_occlusion"
        if paths is not None
        else "compute_discrete_occlusion_from_rollouts"
    )
    if use_approximate_entropy:
        score_kernel_name = (
            "discrete_approximate_entropy_partition_scores_csr"
            if use_csr
            else "discrete_approximate_entropy_partition_scores"
        )
    elif score_only:
        score_kernel_name = (
            "discrete_exact_entropy_active_partition_scores_csr_score_only"
        )
    elif use_csr and compact_active_partitions:
        score_kernel_name = "discrete_exact_entropy_active_partition_scores_csr"
    elif use_csr:
        score_kernel_name = "discrete_exact_entropy_partition_scores_csr"
    else:
        score_kernel_name = "discrete_exact_entropy_partition_scores"
    score_kernel = module.get_function(score_kernel_name)
    finalize_kernel = module.get_function("finalize_discrete_entropy_details")

    context = _establish_module_context()
    with _active_cuda_context(context):
        if prefix_beliefs is None:
            if use_csr:
                raise ValueError("prefix_beliefs is required for CSR transitions")
            prefix_beliefs = np.empty(
                (num_agents, horizon + 1, num_states),
                dtype=np.float32,
            )
            prefix_beliefs[:, 0, :] = beliefs
            for agent_idx in range(num_agents):
                for step in range(1, horizon + 1):
                    prefix_beliefs[agent_idx, step] = (
                        prefix_beliefs[agent_idx, step - 1]
                        @ transition_matrices[agent_idx]
                    )
        else:
            prefix_beliefs = np.ascontiguousarray(prefix_beliefs, dtype=np.float32)
            if prefix_beliefs.shape != (num_agents, horizon + 1, num_states):
                raise ValueError(
                    "prefix_beliefs must have shape (num_agents, horizon + 1, N)"
                )

        if paths is not None:
            paths_d = _alloc_and_copy(paths[:, : horizon + 1, :])
            x_init_d = np.intp(0)
        else:
            paths_d = None
            x_init_d = _alloc_and_copy(x_init[:4] if x_init.size >= 4 else x_init[:2])
        centers_d = _alloc_and_copy(state_centers)
        state_coords_d = (
            _alloc_and_copy(state_coords) if state_coords is not None else np.intp(0)
        )
        static_d = _alloc_and_copy(static_grid)
        agent_owner_bits_d = _alloc_and_copy(agent_owner_bits)
        if use_occupancy:
            occupancy_d = _alloc_and_copy(occupancy_probability_grids)
            owner_masks_d = _alloc_and_copy(occupancy_owner_mask_grids)
            occupancy_steps = int(occupancy_probability_grids.shape[0])
        else:
            occupancy_d = np.intp(0)
            owner_masks_d = np.intp(0)
            occupancy_steps = 0
        prefix_d = _alloc_and_copy(prefix_beliefs)
        if use_csr:
            transition_data_d = _alloc_and_copy(transition_data)
            transition_indices_d = _alloc_and_copy(transition_indices)
            transition_indptr_d = _alloc_and_copy(transition_indptr)
        else:
            transitions_d = _alloc_and_copy(transition_matrices)

        occ_size = num_paths * num_agents * (horizon + 1) * num_states
        occlusion = np.empty((occ_size,), dtype=np.uint8)
        occlusion_d = cuda.mem_alloc(occlusion.nbytes)
        scores = np.zeros((num_paths,), dtype=np.float32)
        scores_d = _alloc_and_copy(scores)
        detail_shape = (num_paths, num_agents, horizon)
        detail_size = num_paths * num_agents * horizon
        if score_only:
            step_entropy = None
            step_probability = None
            step_oc_entropy = None
            step_a_state = None
            step_e_state = None
            step_spatial_separation = None
            belief_sums = None
            step_entropy_d = np.intp(0)
            step_probability_d = np.intp(0)
            step_a_state_d = np.intp(0)
            step_e_state_d = np.intp(0)
            step_spatial_separation_d = np.intp(0)
            belief_sums_d = np.intp(0)
        else:
            step_entropy = np.zeros(detail_shape, dtype=np.float32)
            step_probability = np.zeros(detail_shape, dtype=np.float32)
            step_oc_entropy = np.zeros(detail_shape, dtype=np.float32)
            step_a_state = np.zeros(detail_shape, dtype=np.float32)
            step_e_state = np.zeros(detail_shape, dtype=np.float32)
            step_spatial_separation = np.zeros(detail_shape, dtype=np.float32)
            belief_sums = np.zeros((detail_size, num_states), dtype=np.float32)
            step_entropy_d = _alloc_and_copy(step_entropy.ravel())
            step_probability_d = _alloc_and_copy(step_probability.ravel())
            step_a_state_d = _alloc_and_copy(step_a_state.ravel())
            step_e_state_d = _alloc_and_copy(step_e_state.ravel())
            step_spatial_separation_d = _alloc_and_copy(step_spatial_separation.ravel())
            belief_sums_d = _alloc_and_copy(belief_sums.ravel())

        block = 128
        grid = ((occ_size + block - 1) // block, 1, 1)
        if paths is not None:
            compute_occ(
                paths_d,
                centers_d,
                static_d,
                occupancy_d,
                owner_masks_d,
                agent_owner_bits_d,
                np.int32(num_paths),
                np.int32(num_agents),
                np.int32(horizon + 1),
                np.int32(num_states),
                np.int32(rows),
                np.int32(cols),
                np.float32(grid_origin[0]),
                np.float32(grid_origin[1]),
                np.float32(grid_resolution),
                np.float32(scan_range),
                np.float32(occupancy_threshold),
                np.int32(occupancy_steps),
                occlusion_d,
                block=(block, 1, 1),
                grid=grid,
            )
        else:
            compute_occ(
                rollout_states_d,
                x_init_d,
                centers_d,
                static_d,
                occupancy_d,
                owner_masks_d,
                agent_owner_bits_d,
                np.int32(rollout_state_stride),
                np.int32(num_paths),
                np.int32(num_agents),
                np.int32(horizon + 1),
                np.int32(num_states),
                np.int32(rows),
                np.int32(cols),
                np.float32(grid_origin[0]),
                np.float32(grid_origin[1]),
                np.float32(grid_resolution),
                np.float32(scan_range),
                np.float32(occupancy_threshold),
                np.int32(occupancy_steps),
                occlusion_d,
                block=(block, 1, 1),
                grid=grid,
            )

        score_shared = int((2 * num_states + block) * np.dtype(np.float32).itemsize)
        _configure_dynamic_shared_memory(
            score_kernel,
            score_shared,
            label="discrete OCE entropy score kernel",
        )
        if use_approximate_entropy:
            # Approximate entropy aggregates all visible states at each partition
            # step, so it has horizon*horizon partition tasks, not per-state tasks.
            score_grid = (
                num_paths * num_agents * horizon
                + num_paths * num_agents * horizon * horizon,
                1,
                1,
            )
            score_block_count = int(score_grid[0])
            if use_csr:
                score_kernel(
                    transition_data_d,
                    transition_indices_d,
                    transition_indptr_d,
                    prefix_d,
                    occlusion_d,
                    state_coords_d,
                    np.int32(num_paths),
                    np.int32(num_agents),
                    np.int32(horizon),
                    np.int32(num_states),
                    step_entropy_d,
                    step_probability_d,
                    step_spatial_separation_d,
                    belief_sums_d,
                    scores_d,
                    block=(block, 1, 1),
                    grid=score_grid,
                    shared=score_shared,
                )
            else:
                score_kernel(
                    transitions_d,
                    prefix_d,
                    occlusion_d,
                    state_coords_d,
                    np.int32(num_paths),
                    np.int32(num_agents),
                    np.int32(horizon),
                    np.int32(num_states),
                    step_entropy_d,
                    step_probability_d,
                    step_spatial_separation_d,
                    belief_sums_d,
                    scores_d,
                    block=(block, 1, 1),
                    grid=score_grid,
                    shared=score_shared,
                )
        else:
            # Exact entropy either compacts active seen-state partitions for CSR
            # or launches the full per-state partition grid.
            if use_csr and compact_active_partitions:
                cuda.memcpy_dtoh(occlusion, occlusion_d)
                (
                    partition_paths,
                    partition_agents,
                    partition_ks,
                    partition_steps,
                    partition_states,
                ) = _build_active_partition_index(
                    occlusion.reshape(
                        num_paths,
                        num_agents,
                        horizon + 1,
                        num_states,
                    ),
                    prefix_beliefs,
                    horizon=horizon,
                )
                num_active_partitions = int(partition_states.size)
                dummy_partitions = np.zeros((1,), dtype=np.int32)
                partition_paths_d = _alloc_and_copy(
                    partition_paths if num_active_partitions else dummy_partitions
                )
                partition_agents_d = _alloc_and_copy(
                    partition_agents if num_active_partitions else dummy_partitions
                )
                partition_ks_d = _alloc_and_copy(
                    partition_ks if num_active_partitions else dummy_partitions
                )
                partition_steps_d = _alloc_and_copy(
                    partition_steps if num_active_partitions else dummy_partitions
                )
                partition_states_d = _alloc_and_copy(
                    partition_states if num_active_partitions else dummy_partitions
                )
                score_grid = (
                    num_paths * num_agents * horizon + num_active_partitions,
                    1,
                    1,
                )
                score_block_count = int(score_grid[0])
                if score_only:
                    score_kernel(
                        transition_data_d,
                        transition_indices_d,
                        transition_indptr_d,
                        prefix_d,
                        occlusion_d,
                        partition_paths_d,
                        partition_agents_d,
                        partition_ks_d,
                        partition_steps_d,
                        partition_states_d,
                        np.int32(num_active_partitions),
                        np.int32(num_paths),
                        np.int32(num_agents),
                        np.int32(horizon),
                        np.int32(num_states),
                        scores_d,
                        block=(block, 1, 1),
                        grid=score_grid,
                        shared=score_shared,
                    )
                else:
                    score_kernel(
                        transition_data_d,
                        transition_indices_d,
                        transition_indptr_d,
                        prefix_d,
                        occlusion_d,
                        state_coords_d,
                        partition_paths_d,
                        partition_agents_d,
                        partition_ks_d,
                        partition_steps_d,
                        partition_states_d,
                        np.int32(num_active_partitions),
                        np.int32(num_paths),
                        np.int32(num_agents),
                        np.int32(horizon),
                        np.int32(num_states),
                        step_entropy_d,
                        step_probability_d,
                        step_spatial_separation_d,
                        belief_sums_d,
                        scores_d,
                        block=(block, 1, 1),
                        grid=score_grid,
                        shared=score_shared,
                    )
            else:
                score_grid = (
                    num_paths * num_agents * horizon
                    + num_paths * num_agents * horizon * horizon * num_states,
                    1,
                    1,
                )
                score_block_count = int(score_grid[0])
                if use_csr:
                    score_kernel(
                        transition_data_d,
                        transition_indices_d,
                        transition_indptr_d,
                        prefix_d,
                        occlusion_d,
                        state_coords_d,
                        np.int32(num_paths),
                        np.int32(num_agents),
                        np.int32(horizon),
                        np.int32(num_states),
                        step_entropy_d,
                        step_probability_d,
                        step_spatial_separation_d,
                        belief_sums_d,
                        scores_d,
                        block=(block, 1, 1),
                        grid=score_grid,
                        shared=score_shared,
                    )
                else:
                    score_kernel(
                        transitions_d,
                        prefix_d,
                        occlusion_d,
                        state_coords_d,
                        np.int32(num_paths),
                        np.int32(num_agents),
                        np.int32(horizon),
                        np.int32(num_states),
                        step_entropy_d,
                        step_probability_d,
                        step_spatial_separation_d,
                        belief_sums_d,
                        scores_d,
                        block=(block, 1, 1),
                        grid=score_grid,
                        shared=score_shared,
                    )

        cuda.memcpy_dtoh(scores, scores_d)
        step_belief_sums = None
        # score_components = None
        step_total_entropy = None
        public_step_entropy = None
        public_step_probability = None
        public_step_oc_entropy = None
        public_step_a_state = None
        public_step_e_state = None
        public_step_spatial_separation = None
        public_step_total_entropy = None
        if not score_only:
            finalize_shared = int(block * np.dtype(np.float32).itemsize)
            _configure_dynamic_shared_memory(
                finalize_kernel,
                finalize_shared,
                label="discrete OCE partition detail finalization kernel",
            )
            finalize_kernel(
                step_entropy_d,
                step_probability_d,
                belief_sums_d,
                np.int32(num_paths),
                np.int32(num_agents),
                np.int32(horizon),
                np.int32(num_states),
                step_a_state_d,
                step_e_state_d,
                block=(block, 1, 1),
                grid=(detail_size, 1, 1),
                shared=finalize_shared,
            )

            cuda.memcpy_dtoh(step_entropy.ravel(), step_entropy_d)
            cuda.memcpy_dtoh(step_probability.ravel(), step_probability_d)
            cuda.memcpy_dtoh(step_a_state.ravel(), step_a_state_d)
            cuda.memcpy_dtoh(step_e_state.ravel(), step_e_state_d)
            cuda.memcpy_dtoh(
                step_spatial_separation.ravel(),
                step_spatial_separation_d,
            )
            step_oc_entropy = np.divide(
                step_entropy,
                step_probability,
                out=np.zeros_like(step_entropy),
                where=step_probability > TOLERANCE,
            )
            step_total_entropy = np.cumsum(step_entropy, axis=2)
            step_total_oc_entropy = np.cumsum(step_oc_entropy, axis=2)
            if return_belief_sums:
                cuda.memcpy_dtoh(belief_sums.ravel(), belief_sums_d)
                step_belief_sums = belief_sums.reshape(
                    num_paths,
                    num_agents,
                    horizon,
                    num_states,
                )
            # score_components = np.stack(
            #     [
            #         step_entropy,
            #         step_total_entropy,
            #         step_oc_entropy,
            #         step_total_oc_entropy,
            #         step_spatial_separation,
            #         step_probability,
            #         step_e_state,
            #         step_a_state,
            #     ],
            #     axis=-1,
            # )
            public_step_entropy = np.sum(step_entropy, axis=1)
            public_step_probability = np.sum(step_probability, axis=1)
            public_step_a_state = np.sum(step_a_state, axis=1)
            public_step_e_state = np.sum(step_e_state, axis=1)
            public_step_spatial_separation = np.sum(
                step_spatial_separation,
                axis=1,
            )
            public_step_oc_entropy = np.divide(
                public_step_entropy,
                public_step_probability,
                out=np.zeros_like(public_step_entropy),
                where=public_step_probability > TOLERANCE,
            )
            public_step_total_entropy = np.cumsum(public_step_entropy, axis=1)
            public_step_total_oc_entropy = np.cumsum(public_step_oc_entropy, axis=1)
            # public_score_components = np.stack(
            #     [
            #         public_step_entropy,
            #         public_step_total_entropy,
            #         public_step_oc_entropy,
            #         public_step_total_oc_entropy,
            #         public_step_spatial_separation,
            #         public_step_probability,
            #         public_step_e_state,
            #         public_step_a_state,
            #     ],
            #     axis=-1,
            # )
        else:
            # public_score_components = None
            pass

        visibility_tensor = None
        if return_visibility:
            if not use_csr or not compact_active_partitions or use_approximate_entropy:
                cuda.memcpy_dtoh(occlusion, occlusion_d)
            visibility_tensor = 1.0 - occlusion.reshape(
                num_paths, num_agents, horizon + 1, num_states
            ).astype(np.float32)
            if not use_occupancy:
                visibility_tensor = visibility_tensor[:, 0, :, :]

    if score_only:
        best = int(np.argmin(scores)) if scores.size else 0
    else:
        scores = score_discrete_oce_components(
            scoring_mode=scoring_mode,
            step_entropy=public_step_entropy,
            step_oc_entropy=public_step_oc_entropy,
            step_e_state=public_step_e_state,
        )
        best = int(np.argmin(scores)) if scores.size else 0

    return DiscreteOCEResult(
        best_trajectory=best,
        scores=scores,
        device_scores=scores_d if return_device_scores else None,
        step_entropy=public_step_entropy,
        step_probability=public_step_probability,
        step_e_state=public_step_e_state,
        step_a_state=public_step_a_state,
        step_oc_entropy=public_step_oc_entropy,
        step_total_entropy=public_step_total_entropy,
        step_total_oc_entropy=public_step_total_oc_entropy,
        step_spatial_separation=public_step_spatial_separation,
        per_agent_step_entropy=step_entropy,
        per_agent_step_probability=step_probability,
        per_agent_step_e_state=step_e_state,
        per_agent_step_a_state=step_a_state,
        per_agent_step_oc_entropy=step_oc_entropy,
        per_agent_step_total_entropy=step_total_entropy,
        per_agent_step_spatial_separation=step_spatial_separation,
        step_belief_sums=step_belief_sums,
        # BUGBUG - score components are not used upsteam - temporarily removed
        # per_agent_score_components=score_components,
        # score_components=public_score_components,
        visibility_tensor=visibility_tensor,
        execution_path=(
            "cuda_discrete_exact_csr_score_only"
            if score_only and use_csr
            else (
                (
                    "cuda_discrete_approximate_csr"
                    if use_csr
                    else "cuda_discrete_approximate_dense"
                )
                if use_approximate_entropy
                else (
                    "cuda_discrete_exact_csr"
                    if use_csr
                    else "cuda_discrete_exact_dense"
                )
            )
        ),
        metadata={
            "active_partitions": num_active_partitions,
            "score_blocks": score_block_count,
            "score_only": score_only,
            "entropy_method": entropy_method,
            "scoring_mode": scoring_mode,
        },
    )


def evaluate_discrete_oce_gpu(
    *,
    paths: np.ndarray,
    state_centers: np.ndarray,
    state_coords: np.ndarray | None = None,
    static_grid: np.ndarray,
    grid_origin: tuple[float, float],
    grid_resolution: float,
    transition_matrices: np.ndarray | None = None,
    transition_data: np.ndarray | None = None,
    transition_indices: np.ndarray | None = None,
    transition_indptr: np.ndarray | None = None,
    prefix_beliefs: np.ndarray | None = None,
    beliefs: np.ndarray,
    horizon: int,
    scan_range: float,
    return_visibility: bool = False,
    return_belief_sums: bool = False,
    occupancy_probability_grids: np.ndarray | None = None,
    occupancy_owner_mask_grids: np.ndarray | None = None,
    agent_owner_bits: np.ndarray | None = None,
    occupancy_threshold: float = 0.25,
    entropy_method: str = "discrete_exact_entropy",
    scoring_mode: str = "information_only",
) -> DiscreteOCEResult:
    return _evaluate_discrete_oce_gpu_impl(
        paths=paths,
        state_centers=state_centers,
        state_coords=state_coords,
        static_grid=static_grid,
        grid_origin=grid_origin,
        grid_resolution=grid_resolution,
        transition_matrices=transition_matrices,
        transition_data=transition_data,
        transition_indices=transition_indices,
        transition_indptr=transition_indptr,
        prefix_beliefs=prefix_beliefs,
        beliefs=beliefs,
        horizon=horizon,
        scan_range=scan_range,
        return_visibility=return_visibility,
        return_belief_sums=return_belief_sums,
        occupancy_probability_grids=occupancy_probability_grids,
        occupancy_owner_mask_grids=occupancy_owner_mask_grids,
        agent_owner_bits=agent_owner_bits,
        occupancy_threshold=occupancy_threshold,
        entropy_method=entropy_method,
        scoring_mode=scoring_mode,
    )


def evaluate_discrete_oce_rollouts_gpu(
    *,
    rollout_states_d,
    num_rollouts: int,
    x_init: np.ndarray,
    rollout_state_stride: int = 4,
    state_centers: np.ndarray,
    state_coords: np.ndarray | None = None,
    static_grid: np.ndarray,
    grid_origin: tuple[float, float],
    grid_resolution: float,
    transition_matrices: np.ndarray | None = None,
    transition_data: np.ndarray | None = None,
    transition_indices: np.ndarray | None = None,
    transition_indptr: np.ndarray | None = None,
    prefix_beliefs: np.ndarray | None = None,
    beliefs: np.ndarray,
    horizon: int,
    scan_range: float,
    return_visibility: bool = False,
    return_belief_sums: bool = False,
    occupancy_probability_grids: np.ndarray | None = None,
    occupancy_owner_mask_grids: np.ndarray | None = None,
    agent_owner_bits: np.ndarray | None = None,
    occupancy_threshold: float = 0.25,
    return_device_scores: bool = True,
    entropy_method: str = "discrete_exact_entropy",
    scoring_mode: str = "information_only",
) -> DiscreteOCEResult:
    return _evaluate_discrete_oce_gpu_impl(
        rollout_states_d=rollout_states_d,
        num_rollouts=num_rollouts,
        x_init=x_init,
        rollout_state_stride=rollout_state_stride,
        state_centers=state_centers,
        state_coords=state_coords,
        static_grid=static_grid,
        grid_origin=grid_origin,
        grid_resolution=grid_resolution,
        transition_matrices=transition_matrices,
        transition_data=transition_data,
        transition_indices=transition_indices,
        transition_indptr=transition_indptr,
        prefix_beliefs=prefix_beliefs,
        beliefs=beliefs,
        horizon=horizon,
        scan_range=scan_range,
        return_visibility=return_visibility,
        return_belief_sums=return_belief_sums,
        occupancy_probability_grids=occupancy_probability_grids,
        occupancy_owner_mask_grids=occupancy_owner_mask_grids,
        agent_owner_bits=agent_owner_bits,
        occupancy_threshold=occupancy_threshold,
        return_device_scores=return_device_scores,
        compact_active_partitions=True,
        entropy_method=entropy_method,
        scoring_mode=scoring_mode,
    )
