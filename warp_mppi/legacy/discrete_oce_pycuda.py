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

_MODULE_CONTEXT = None
_owns_module_context_ref = False
_module_context_pushed = False


@dataclass
class DiscreteOCEResult:
    best_trajectory: int
    scores: np.ndarray
    step_entropy: np.ndarray | None = None
    step_probability: np.ndarray | None = None
    step_e_state: np.ndarray | None = None
    step_a_state: np.ndarray | None = None
    step_oc_entropy: np.ndarray | None = None
    step_state_entropy: np.ndarray | None = None
    step_belief_sums: np.ndarray | None = None
    score_components: np.ndarray | None = None
    visibility_tensor: np.ndarray | None = None
    execution_path: str = "cuda_discrete_exact"
    metadata: dict | None = None


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

__global__ void compute_discrete_occlusion(
    const float* paths,
    const float* state_centers,
    const unsigned char* static_grid,
    int num_paths,
    int horizon_plus_one,
    int num_states,
    int rows,
    int cols,
    float origin_x,
    float origin_y,
    float resolution,
    float scan_range,
    unsigned char* occlusion
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_paths * horizon_plus_one * num_states;
    if (idx >= total) return;

    int state = idx % num_states;
    int tmp = idx / num_states;
    int step = tmp % horizon_plus_one;
    int path = tmp / horizon_plus_one;

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
    } else if (blocked_line(
        static_grid, rows, cols, origin_x, origin_y, resolution, ox, oy, sx, sy
    )) {
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

__device__ float accumulate_partition_outputs(
    float* work,
    int path,
    int agent,
    int k,
    int num_agents,
    int horizon,
    int num_states,
    float* step_entropy,
    float* step_probability,
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
    for (int i = threadIdx.x; i < num_states; i += blockDim.x) {
        float value = fmaxf(work[i], 0.0f);
        if (value > 0.0f) {
            atomicAdd(belief_sums + detail_idx * num_states + i, value);
        }
    }
    return contribution;
}

__global__ void finalize_discrete_exact_entropy_details(
    const float* step_entropy,
    const float* step_probability,
    const float* belief_sums,
    int num_paths,
    int num_agents,
    int horizon,
    int num_states,
    float* step_oc_entropy,
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
            step_oc_entropy[task] = 0.0f;
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
    float oc_entropy = block_sum(scratch, local_entropy);
    if (tid == 0) {
        float a_state = step_entropy[task] / prob;
        float e_state = oc_entropy - a_state;
        if (e_state < 0.0f) {
            e_state = 0.0f;
        }
        step_oc_entropy[task] = oc_entropy;
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
                int occ_idx = (path * (horizon + 1) + t) * num_states + j;
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
                int occ_idx = (path * (horizon + 1) + p) * num_states + s;
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
                            (path * (horizon + 1) + t) * num_states + j;
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
    int num_paths,
    int num_agents,
    int horizon,
    int num_states,
    float* step_entropy,
    float* step_probability,
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
            (path * (horizon + 1) + partition_step) * num_states + partition_state;
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
                int occ_idx = (path * (horizon + 1) + t) * num_states + j;
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
                int occ_idx = (path * (horizon + 1) + t) * num_states + j;
                next[j] *= (float)occlusion[occ_idx];
                work[j] = next[j];
            }
            __syncthreads();
        }
    }

    float contribution = accumulate_partition_outputs(
        work,
        path,
        agent,
        k,
        num_agents,
        horizon,
        num_states,
        step_entropy,
        step_probability,
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
    int num_paths,
    int num_agents,
    int horizon,
    int num_states,
    float* step_entropy,
    float* step_probability,
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
            (path * (horizon + 1) + partition_step) * num_states + partition_state;
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
                int occ_idx = (path * (horizon + 1) + t) * num_states + j;
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
            int occ_idx = (path * (horizon + 1) + partition_step + 1) * num_states + j;
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
                int occ_idx = (path * (horizon + 1) + t) * num_states + j;
                next[j] *= (float)occlusion[occ_idx];
                work[j] = next[j];
            }
            __syncthreads();
        }
    }

    float contribution = accumulate_partition_outputs(
        work,
        path,
        agent,
        k,
        num_agents,
        horizon,
        num_states,
        step_entropy,
        step_probability,
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
                int occ_idx = (path * (horizon + 1) + t) * num_states + j;
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
            int occ_idx = (path * (horizon + 1) + partition_step + 1) * num_states + j;
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
                int occ_idx = (path * (horizon + 1) + t) * num_states + j;
                next[j] *= (float)occlusion[occ_idx];
                work[j] = next[j];
            }
            __syncthreads();
        }
    }

    float contribution = accumulate_partition_outputs(
        work,
        path,
        agent,
        k,
        num_agents,
        horizon,
        num_states,
        step_entropy,
        step_probability,
        belief_sums,
        scratch
    );
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
    num_paths, _, num_states = occlusion.shape
    num_agents = prefix_beliefs.shape[0]
    horizon = int(horizon)

    path_chunks = []
    agent_chunks = []
    k_chunks = []
    step_chunks = []
    state_chunks = []

    for path_idx in range(num_paths):
        path_occlusion = occlusion[path_idx]
        for agent_idx in range(num_agents):
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


def evaluate_discrete_oce_gpu(
    *,
    paths: np.ndarray,
    state_centers: np.ndarray,
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
) -> DiscreteOCEResult:
    if not PYCUDA_AVAILABLE:
        raise RuntimeError("PyCUDA is not available for discrete OCE CUDA execution.")

    paths = np.ascontiguousarray(paths, dtype=np.float32)
    state_centers = np.ascontiguousarray(state_centers, dtype=np.float32)
    static_grid = np.ascontiguousarray(static_grid.astype(np.uint8, copy=False))
    beliefs = np.ascontiguousarray(beliefs, dtype=np.float32)
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

    if paths.ndim != 3 or paths.shape[2] != 2:
        raise ValueError("paths must have shape (num_paths, horizon + 1, 2)")
    if state_centers.ndim != 2 or state_centers.shape[1] != 2:
        raise ValueError("state_centers must have shape (num_states, 2)")
    if not use_csr and transition_matrices.ndim != 3:
        raise ValueError("transition_matrices must have shape (num_agents, N, N)")
    if beliefs.ndim != 2:
        raise ValueError("beliefs must have shape (num_agents, N)")

    num_paths = int(paths.shape[0])
    horizon_plus_one = int(paths.shape[1])
    horizon = min(int(horizon), horizon_plus_one - 1)
    num_states = int(state_centers.shape[0])
    num_agents = int(beliefs.shape[0])
    rows, cols = static_grid.shape
    num_active_partitions = None
    score_block_count = 0

    if num_paths == 0 or num_agents == 0 or horizon <= 0:
        return DiscreteOCEResult(
            best_trajectory=0,
            scores=np.zeros((num_paths,), dtype=np.float32),
            step_entropy=np.zeros((num_paths, num_agents, 0), dtype=np.float32),
            step_probability=np.zeros((num_paths, num_agents, 0), dtype=np.float32),
            step_e_state=np.zeros((num_paths, num_agents, 0), dtype=np.float32),
            step_a_state=np.zeros((num_paths, num_agents, 0), dtype=np.float32),
            step_oc_entropy=np.zeros((num_paths, num_agents, 0), dtype=np.float32),
            step_state_entropy=np.zeros((num_paths, num_agents, 0), dtype=np.float32),
            step_belief_sums=(
                np.zeros((num_paths, num_agents, 0, num_states), dtype=np.float32)
                if return_belief_sums
                else None
            ),
            score_components=np.zeros((num_paths, num_agents, 0, 4), dtype=np.float32),
            visibility_tensor=None,
            execution_path="cuda_discrete_exact_empty",
            metadata={"score_blocks": 0, "active_partitions": 0},
        )
    if num_states != beliefs.shape[1]:
        raise ValueError("belief dimensions do not match state_centers")
    if use_csr:
        if transition_indptr.shape != (num_agents, num_states + 1):
            raise ValueError("transition_indptr must have shape (num_agents, N + 1)")
    elif transition_matrices.shape != (num_agents, num_states, num_states):
        raise ValueError("transition dimensions do not match state_centers")

    module = _compiled_module()
    compute_occ = module.get_function("compute_discrete_occlusion")
    if use_csr:
        score_kernel = module.get_function(
            "discrete_exact_entropy_active_partition_scores_csr"
        )
    else:
        score_kernel = module.get_function("discrete_exact_entropy_partition_scores")
    finalize_kernel = module.get_function("finalize_discrete_exact_entropy_details")

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

        paths_d = _alloc_and_copy(paths[:, : horizon + 1, :])
        centers_d = _alloc_and_copy(state_centers)
        static_d = _alloc_and_copy(static_grid)
        prefix_d = _alloc_and_copy(prefix_beliefs)
        if use_csr:
            transition_data_d = _alloc_and_copy(transition_data)
            transition_indices_d = _alloc_and_copy(transition_indices)
            transition_indptr_d = _alloc_and_copy(transition_indptr)
        else:
            transitions_d = _alloc_and_copy(transition_matrices)

        occ_size = num_paths * (horizon + 1) * num_states
        occlusion = np.empty((occ_size,), dtype=np.uint8)
        occlusion_d = cuda.mem_alloc(occlusion.nbytes)
        scores = np.zeros((num_paths,), dtype=np.float32)
        scores_d = _alloc_and_copy(scores)
        detail_shape = (num_paths, num_agents, horizon)
        detail_size = num_paths * num_agents * horizon
        step_entropy = np.zeros(detail_shape, dtype=np.float32)
        step_probability = np.zeros(detail_shape, dtype=np.float32)
        step_oc_entropy = np.zeros(detail_shape, dtype=np.float32)
        step_a_state = np.zeros(detail_shape, dtype=np.float32)
        step_e_state = np.zeros(detail_shape, dtype=np.float32)
        belief_sums = np.zeros((detail_size, num_states), dtype=np.float32)
        step_entropy_d = _alloc_and_copy(step_entropy.ravel())
        step_probability_d = _alloc_and_copy(step_probability.ravel())
        step_oc_entropy_d = _alloc_and_copy(step_oc_entropy.ravel())
        step_a_state_d = _alloc_and_copy(step_a_state.ravel())
        step_e_state_d = _alloc_and_copy(step_e_state.ravel())
        belief_sums_d = _alloc_and_copy(belief_sums.ravel())

        block = 128
        grid = ((occ_size + block - 1) // block, 1, 1)
        compute_occ(
            paths_d,
            centers_d,
            static_d,
            np.int32(num_paths),
            np.int32(horizon + 1),
            np.int32(num_states),
            np.int32(rows),
            np.int32(cols),
            np.float32(grid_origin[0]),
            np.float32(grid_origin[1]),
            np.float32(grid_resolution),
            np.float32(scan_range),
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
        if use_csr:
            cuda.memcpy_dtoh(occlusion, occlusion_d)
            (
                partition_paths,
                partition_agents,
                partition_ks,
                partition_steps,
                partition_states,
            ) = _build_active_partition_index(
                occlusion.reshape(num_paths, horizon + 1, num_states),
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
                step_entropy_d,
                step_probability_d,
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
            score_kernel(
                transitions_d,
                prefix_d,
                occlusion_d,
                np.int32(num_paths),
                np.int32(num_agents),
                np.int32(horizon),
                np.int32(num_states),
                step_entropy_d,
                step_probability_d,
                belief_sums_d,
                scores_d,
                block=(block, 1, 1),
                grid=score_grid,
                shared=score_shared,
            )

        finalize_shared = int(block * np.dtype(np.float32).itemsize)
        _configure_dynamic_shared_memory(
            finalize_kernel,
            finalize_shared,
            label="discrete OCE entropy detail finalization kernel",
        )
        finalize_kernel(
            step_entropy_d,
            step_probability_d,
            belief_sums_d,
            np.int32(num_paths),
            np.int32(num_agents),
            np.int32(horizon),
            np.int32(num_states),
            step_oc_entropy_d,
            step_a_state_d,
            step_e_state_d,
            block=(block, 1, 1),
            grid=(detail_size, 1, 1),
            shared=finalize_shared,
        )

        cuda.memcpy_dtoh(scores, scores_d)
        cuda.memcpy_dtoh(step_entropy.ravel(), step_entropy_d)
        cuda.memcpy_dtoh(step_probability.ravel(), step_probability_d)
        cuda.memcpy_dtoh(
            step_oc_entropy.ravel(),
            step_oc_entropy_d,
        )
        cuda.memcpy_dtoh(step_a_state.ravel(), step_a_state_d)
        cuda.memcpy_dtoh(step_e_state.ravel(), step_e_state_d)
        step_belief_sums = None
        if return_belief_sums:
            cuda.memcpy_dtoh(belief_sums.ravel(), belief_sums_d)
            step_belief_sums = belief_sums.reshape(
                num_paths,
                num_agents,
                horizon,
                num_states,
            )
        visibility_tensor = None
        if return_visibility:
            if not use_csr:
                cuda.memcpy_dtoh(occlusion, occlusion_d)
            visibility_tensor = 1.0 - occlusion.reshape(
                num_paths, horizon + 1, num_states
            ).astype(np.float32)

    best = int(np.argmin(scores)) if scores.size else 0
    score_components = np.stack(
        [step_entropy, step_probability, step_e_state, step_a_state],
        axis=-1,
    )
    return DiscreteOCEResult(
        best_trajectory=best,
        scores=scores,
        step_entropy=step_entropy,
        step_probability=step_probability,
        step_e_state=step_e_state,
        step_a_state=step_a_state,
        step_oc_entropy=step_oc_entropy,
        step_state_entropy=step_oc_entropy,
        step_belief_sums=step_belief_sums,
        score_components=score_components,
        visibility_tensor=visibility_tensor,
        execution_path=(
            "cuda_discrete_exact_csr" if use_csr else "cuda_discrete_exact_dense"
        ),
        metadata={
            "active_partitions": num_active_partitions,
            "score_blocks": score_block_count,
        },
    )
