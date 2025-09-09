import sys
from typing import Any
import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit as _pycuda_autoinit  # ensures a context exists
from pycuda.compiler import SourceModule
from pycuda import characterize


_cuda: Any = cuda  # alias to satisfy static analyzers

BLOCK_SIZE = 32

# CUDA source (kernels) -------------------------------------------------------
_MPPI_CUDA_SOURCE = """
    #ifndef M_PI
    #define M_PI (3.14159265358979323846264338327950288)
    #endif

    #include <cuda_runtime.h>
    #include <curand.h>
    #include <curand_kernel.h>
    #include <cmath>
    #include <cfloat>

    #define BLOCK_SIZE 32

    enum VisibilityMethod {
        OURS = 0,
        HIGGINS = 1,
        ANDERSEN = 2,
        NO_VISIBILITY = 3, // Renamed from NONE for clarity, maps to "Nominal"
        INFO_GAIN_LIKE = 4, // Renamed from OTHER, maps to "Infogain" / "Ours" if costmap is infogain
    };

    struct Costmap_Params {
        int height;
        int width;
        float origin_x;
        float origin_y;
        float resolution;
    };

    struct Optimization_Params {
        int samples;
        float M;
        float dt;
        int num_controls;
        int num_obstacles;
        float x_init[4];
        float x_goal[4];
        float u_limits[2];
        float u_dist_limits[2];
        float Q[4];
        float Qf[4];
        float R[2];
        int method;
        float c_lambda;
        float scan_range;
        float vehicle_length;
    float steering_rate_weight; // added optional penalty weight
    };

    struct Object {
        float x;
        float y;
        float radius;
    };

    struct Obstacle {
        Object loc;
        float min_x;
        float min_y;
        float distance;
    };

    struct State {
        float x;
        float y;
        float v;
        float theta;
    };

    // union State {
    //     float4 xyvt; // x, y, v, theta
    //     struct {
    //         float x;
    //         float y;
    //         float v;
    //         float theta;
    //     };
    // };

    struct Control {
        float a;
        float delta;
    };

    //
    // Based on a comment from the following link on checking for zero:
    //
    // https://forums.developer.nvidia.com/t/on-tackling-float-point-precision-issues-in-cuda/79060
    //
    __device__
    inline bool is_zero(float f){
        return f >= -FLT_EPSILON && f <= FLT_EPSILON;
    }

    __device__
    inline bool is_equal(float f1, float f2){
        return fabs(f1 - f2) < FLT_EPSILON;
    }

    __device__ int
    epsilon_round(float value) {
        const float epsilon = 2e-6;

        float rounded_value = roundf(value);
        if( fabs(value - rounded_value) < epsilon ) {
            return static_cast<int>(rounded_value);
        } else {
            return static_cast<int>(value);
        }
    }

    __device__
    float obstacle_cost(const Obstacle *obstacles, int num_obstacles, float px, float py, float radius) {
      for (int i = 0; i < num_obstacles; i++) {
        auto obstacle = &obstacles[i];
        float dx = obstacle->loc.x - px;
        float dy = obstacle->loc.y - py;
        float d_2 = dx * dx + dy * dy;
        float min_dist = obstacle->loc.radius + radius;

        if (d_2 < min_dist * min_dist) {
          return 10000000.0;
        }
      }
      return 0.0;
    }


    __device__
    float higgins_cost(const float M, const Obstacle *obstacles, int num_obstacles, float px, float py, float scan_range) {

      float cost = 0.0;

      float r_fov = scan_range;
      float r_fov_2 = r_fov*r_fov;

      // ( "Checking higgins! px: %f, py: %f, scan_range: %f\\n", px, py, scan_range);

      for (int i = 0; i < num_obstacles; i++) {
        auto obstacle = &obstacles[i];
        float dx = obstacle->loc.x - px;
        float dy = obstacle->loc.y - py;
        float d_2 = dx * dx + dy * dy;
        float d = sqrtf(d_2);

        float inner = obstacle->loc.radius / d * (r_fov_2 - d_2);
        auto inner_exp = expf(inner);
        float score;

        // printf( "obstacle->loc.x: %f, obstacle->loc.y: %f, obstacle->loc.radius: %f\\n", obstacle->loc.x, obstacle->loc.y, obstacle->loc.radius );
        // printf( "px: %f, py: %f, scan_range: %f\\n", px, py, scan_range );
        // printf( "d_2: %f, d: %f\\n", d_2, d );
        // printf( "inner: %f, inner_exp: %f\\n", inner, inner_exp );

        if( isinf(inner_exp) || isnan(inner_exp) ) {
          score = inner;
        } else {
          score = logf(1 + inner_exp);
        }
        cost += M * score * score;
      }

      return cost;
    }

    __device__
    float
    andersen_cost( const float M, const Obstacle *obstacles, int num_obstacles, float px, float py, float vx, float vy) {
      float cost = 0.0;
      float v = sqrtf(vx * vx + vy * vy);

      for (int i = 0; i < num_obstacles; i++) {
        auto obstacle = &obstacles[i];
        float dx = obstacle->min_x - px;
        float dy = obstacle->min_y - py;

        // check if the obstacle is in front of the vehicle
        auto dot = dx * vx + dy * vy;
        if (dot > 0) {
          float d = sqrtf(dx * dx + dy * dy);
          // Andersen is a reward
          cost -= M * acosf(dot / (d * v));
          break;   // only consider the closest obstacle
        }
      }

      return cost;
    }


    __device__
    float our_cost(const float M, const float *costmap, int height, int width, float origin_x, float origin_y, float resolution, const float px, const float py,
                   const int step) {
      float cost = 0.0;

      auto map_x = epsilon_round((px - origin_x) / resolution);
      auto map_y = epsilon_round((py - origin_y) / resolution);

      if (map_x < 0 || map_x >= width || map_y < 0 || map_y >= height) {
        return 10000000.0;
      }

      cost = -M * (costmap[map_y * width + map_x]);

      return cost;
    }


    // Basic step function -- apply the control to advance one step
    __device__
    void euler(const State *state, const Control *control, float vehicle_length, State *result) {
        result->x     = state->v * cosf(state->theta);
        result->y     = state->v * sinf(state->theta);
        result->theta = state->v * tanf(control->delta) / vehicle_length;
        result->v     = control->a;

        // printf( "tan(control->delta): %f\\n", tan(control->delta) );
        // printf( "state->x: %f, state->y: %f, state->v: %f, state->theta: %f\\n", state->x, state->y, state->v, state->theta );
        // printf( "control->a: %f, control->delta: %f\\n", control->a, control->delta );
        // printf( "vehicle_length: %f\\n", vehicle_length );
        // printf( "result->x: %f, result->y: %f, result->v: %f, result->theta: %f\\n", result->x, result->y, result->v, result->theta );

    }

    inline __device__
    void update_state(const State *state, const State *update, float dt, State *result) {
      result->x     = state->x     + update->x * dt;
      result->y     = state->y     + update->y * dt;
      result->v     = state->v     + update->v * dt;
      result->theta = state->theta + update->theta * dt;
    }

    //
    // Also define the Runge-Kutta variant as it is (apparently) a much
    // better approximation of the first order derivative
    //  https://en.wikipedia.org/wiki/Runge-Kutta_methods
    __device__
    void runge_kutta_step(const State *state, const Control *control, float vehicle_length, float dt, State *result) {
      State k1, k2, k3, k4;
      State tmp_state;

      euler(state, control, vehicle_length, &k1);
      update_state(state, &k1, dt / 2.0, &tmp_state);
      euler(&tmp_state, control, vehicle_length, &k2);
      update_state(state, &k2, dt / 2.0, &tmp_state);
      euler(&tmp_state, control, vehicle_length, &k3);
      update_state(state, &k3, dt, &tmp_state);
      euler(&tmp_state, control, vehicle_length, &k4);

      result->x = (k1.x + 2.0 * (k2.x + k3.x) + k4.x) / 6.0;
      result->y = (k1.y + 2.0 * (k2.y + k3.y) + k4.y) / 6.0;
      result->v = (k1.v + 2.0 * (k2.v + k3.v) + k4.v) / 6.0;
      result->theta = (k1.theta + 2.0 * (k2.theta + k3.theta) + k4.theta) / 6.0;
    }


    __device__
    void generate_controls(
            curandState *globalState,
            int index,
            const Control *u_nom,
            const int num_controls,
            const float *u_limits,
            const float *u_dist_limits,
            Control *u_dist
    ) {
      curandState localState = globalState[index];
            for (int i = 0; i < num_controls; i++) {
                // Gaussian (normal) noise, std dev = u_dist_limits[*]
                float a_noise = curand_normal(&localState) * u_dist_limits[0];
                float delta_noise = curand_normal(&localState) * u_dist_limits[1];
                float a_candidate = u_nom[i].a + a_noise;
                float delta_candidate = u_nom[i].delta + delta_noise;
                // Clamp to limits
                if (a_candidate >  u_limits[0]) a_candidate =  u_limits[0];
                if (a_candidate < -u_limits[0]) a_candidate = -u_limits[0];
                if (delta_candidate >  u_limits[1]) delta_candidate =  u_limits[1];
                if (delta_candidate < -u_limits[1]) delta_candidate = -u_limits[1];
                // Store disturbance (difference from nominal)
                u_dist[i].a = a_candidate - u_nom[i].a;
                u_dist[i].delta = delta_candidate - u_nom[i].delta;
            }
      globalState[index] = localState;
    }


    // External functions -- each is wrapped with extern "C" to prevent name mangling
    // because pycuda doesn't support C++ name mangling
    extern "C" __global__
    void setup_kernel(curandState *state, unsigned long seed) {
      int id = threadIdx.x + blockIdx.x * blockDim.x;
      curand_init(seed, id, 0, &state[id]);
    }


    extern "C" __global__
    void perform_rollout(
            curandState *globalState,
            const float *costmap,
            const Costmap_Params *costmap_args,
            const State *x_nom,   // nominal states, num_controls + 1 x state_size
            const Control *u_nom,   // nominal controls, num_controls x control_size
            const Obstacle *obstacle_data,
            const Optimization_Params *optimization_args,
            Control *u_dists,
            float *u_weights
    ) {
        int start_sample_index = blockIdx.x * blockDim.x + threadIdx.x;
        int samples = optimization_args->samples;

        for (int sample_index = start_sample_index; sample_index < samples; sample_index += blockDim.x * gridDim.x) {

            int num_controls = optimization_args->num_controls;
            int num_obstacles = optimization_args->num_obstacles;
            // float M = optimization_args->M;
            float dt = optimization_args->dt;
            const float *u_limits = optimization_args->u_limits;
            const float *u_dist_limits = optimization_args->u_dist_limits;
            const float *Q = optimization_args->Q;
            const float *Qf = optimization_args->Qf;
            const float *R = optimization_args->R;
            VisibilityMethod method = (VisibilityMethod) optimization_args->method;

            float score = 0.0;
            // float prev_score = 0.0;

            float state_err = 0.0;
            float final_state_err = 0.0;
            float control_err = 0.0;
            float obstacle_err = 0.0;
            float visibility_err = 0.0;

            // rollout the trajectory -- assume we are placing the result in the larger u_dist/u_weight arrays
            const State *x_init_state = reinterpret_cast<const State *>(optimization_args->x_init);
            const State *x_goal_state = reinterpret_cast<const State *>(optimization_args->x_goal);
            Control *u_dist_controls = reinterpret_cast<Control *>(&u_dists[sample_index * num_controls]);

            generate_controls(globalState, sample_index, u_nom, num_controls, u_limits, u_dist_limits, u_dist_controls);

            State current_state = {x_init_state->x, x_init_state->y, x_init_state->v, x_init_state->theta};
            State state_step = {0, 0, 0, 0};

            for (int i = 1; i <= num_controls; i++) {
                // generate the next state
                Control c = {u_nom[i - 1].a + u_dist_controls[i - 1].a, u_nom[i - 1].delta + u_dist_controls[i - 1].delta};
                // runge_kutta_step returns derivative, we multiply by dt here
                runge_kutta_step(&current_state, &c, optimization_args->vehicle_length, dt, &state_step);
                update_state(&current_state, &state_step, dt, &current_state);

                // penalize error in trajectory
                auto theta_diff = x_nom[i].theta - current_state.theta;
                theta_diff = fmod(theta_diff + M_PI, 2 * M_PI) - M_PI;

                state_err = (x_nom[i].x - current_state.x)         * Q[0] * (x_nom[i].x - current_state.x) +
                            (x_nom[i].y - current_state.y)         * Q[1] * (x_nom[i].y - current_state.y) +
                            (x_nom[i].v - current_state.v)         * Q[2] * (x_nom[i].v - current_state.v) +
                            (theta_diff)                           * Q[3] * (theta_diff); // Q is for running state cost

                // penalize final state error
                final_state_err = 0.0;
                if( i == num_controls ) {
                    // last control
                    auto final_theta_diff = x_goal_state->theta - current_state.theta; // Should be x_goal_state->theta? or x_nom[num_controls].theta
                    final_theta_diff = fmod(final_theta_diff + M_PI, 2 * M_PI) - M_PI;
                    final_state_err = (x_goal_state->x - current_state.x) * Qf[0] * (x_goal_state->x - current_state.x) +
                                      (x_goal_state->y - current_state.y) * Qf[1] * (x_goal_state->y - current_state.y) +
                                      (x_goal_state->v - current_state.v) * Qf[2] * (x_goal_state->v - current_state.v) +
                                      (final_theta_diff)                  * Qf[3] * (final_theta_diff); // Qf is for final state cost
                }

                // penalize control action
                control_err = (c.a - u_nom[i - 1].a)         * R[0] * (c.a - u_nom[i - 1].a) +
                              (c.delta - u_nom[i - 1].delta) * R[1] * (c.delta - u_nom[i - 1].delta);

                // optional steering rate penalty (difference between successive applied steering commands)
                if (optimization_args->steering_rate_weight > 0.0f && i > 1) {
                    float prev_delta = u_nom[i - 2].delta + u_dist_controls[i - 2].delta; // previous applied delta
                    float rate = c.delta - prev_delta; // instantaneous change (already per-step)
                    control_err += optimization_args->steering_rate_weight * rate * rate;
                }

                // penalize obstacles
                obstacle_err = obstacle_cost(obstacle_data, num_obstacles, current_state.x, current_state.y, optimization_args->vehicle_length / 2.0);

                // penalize visibility
                visibility_err = 0;

                if (method == OURS) {
                    // Note: 'M' for visibility costs is optimization_args->M.
                    // The 'costmap' related parameters (height, width, origin_x, etc.) are in costmap_args.
                    visibility_err = our_cost(optimization_args->M, costmap, costmap_args->height, costmap_args->width, costmap_args->origin_x, costmap_args->origin_y, costmap_args->resolution, current_state.x, current_state.y, i);
                } else if (method == HIGGINS) {
                    visibility_err = higgins_cost(optimization_args->M, obstacle_data, num_obstacles, current_state.x, current_state.y, optimization_args->scan_range);
                } else if (method == ANDERSEN) {
                    // Velocity for Andersen cost is based on nominal trajectory difference
                    visibility_err = andersen_cost(optimization_args->M, obstacle_data, num_obstacles, current_state.x, current_state.y,
                                                   (x_nom[i].x - x_nom[i-1].x), (x_nom[i].y - x_nom[i-1].y));
                }
                // NO_VISIBILITY and INFO_GAIN_LIKE (if it implies using 'our_cost' already handled by OURS) might not need explicit handling here if OURS covers INFO_GAIN_LIKE

                score += state_err + final_state_err + control_err + obstacle_err + visibility_err;

                if( isnan(score) ){
                    // printf( "score overflow -- prev score: %f, state: %f, final: %f, control: %f, obstacle: %f, visibility: %f\\n", prev_score, state_err, final_state_err, control_err, obstacle_err, visibility_err );
                    score = FLT_MAX / (samples > 0 ? samples : 1); // Assign a large penalty
                    break;
                }
                // prev_score = score;
            }
            u_weights[sample_index] = score;
        }
    }

    extern "C" __global__
    void min_weight(
        int samples,
        float *u_weights,
        float *u_weight_min
    ) {
        // assert( BLOCK_SIZE == blockDim.x ); // This assert cannot be used in CUDA C
        extern __shared__ float shared_min[]; // Dynamically sized shared memory
        int tid = threadIdx.x;

        shared_min[tid] = FLT_MAX;
        for (int i = tid; i < samples; i += blockDim.x) { // Grid-stride loop
            if (u_weights[i] < shared_min[tid]) {
                shared_min[tid] = u_weights[i];
            }
        }

        __syncthreads();

        // Reduction in shared memory
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (shared_min[tid + s] < shared_min[tid]) {
                     shared_min[tid] = shared_min[tid + s];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            // Original: atomicMin(u_weight_min, shared_min[0]);
            // Since this kernel is launched with grid=(1,1,1), only one block executes it.
            // Thus, only thread 0 of this single block will write to u_weight_min.
            // A direct write is safe and avoids issues with atomicMin for floats.
            *u_weight_min = shared_min[0];
        }
    }

    extern "C" __global__
    void calculate_weights(
            int samples,
            float *u_weights,
            float *u_weight_min,
            float c_lambda,
            float *u_weight_total
    ) {
      int start_sample_index = blockIdx.x * blockDim.x + threadIdx.x;
      float u_weight_min_float = *u_weight_min; // Dereference pointer
            // Guard against pathological small temperature
            if (c_lambda < 1e-6f) {
                    c_lambda = 1e-6f;
            }

      for (int sample_index = start_sample_index; sample_index < samples; sample_index += blockDim.x * gridDim.x) {
                // Convert cost to a non-negative difference
                float diff = (u_weights[sample_index] - u_weight_min_float) / c_lambda; // diff >= 0 ideally
                if (diff < 0.0f) diff = 0.0f; // Numerical safety
                // Clip diff to avoid exp underflow -> all zeros (keeps relative shape for large gaps)
                if (diff > 60.0f) diff = 60.0f; // exp(-60) ~ 8.8e-27
                float weight = expf(-diff);

        if( isnan(weight) ){
            weight = 0.0;
        }
        u_weights[sample_index] = weight;
        atomicAdd(u_weight_total, weight);
      }
    }


    extern "C" __global__
    void calculate_mppi_control(
            int samples,
            const Control *u_nom,
            Control *u_dist, // Array of Control structs
            int num_controls,
            const float *u_weights,
            const float *u_weight_total,
            Control *u_mppi
    ) {
      int sample_idx = blockIdx.x * blockDim.x + threadIdx.x; // Iterate over samples
      float u_weight_total_float = *u_weight_total; // Dereference

      if (sample_idx < samples) {
        float weight_normalized;
        if (!is_zero(u_weight_total_float)) {
            weight_normalized = u_weights[sample_idx] / u_weight_total_float;
        } else {
            weight_normalized = 1.0f/samples; // If total weight is zero, normalize to equal distribution
        }

        for (int ctrl_idx = 0; ctrl_idx < num_controls; ++ctrl_idx) {
            int dist_flat_idx = sample_idx * num_controls + ctrl_idx;
            // Atomically add weighted disturbances to the nominal control
            // This needs to be done carefully if u_mppi is shared output for all controls
            // The current u_mppi is an array of Controls, one per timestep.
            // We add the weighted disturbance for *this sample* to *each* u_mppi[ctrl_idx]
            atomicAdd(&(u_mppi[ctrl_idx].a), u_dist[dist_flat_idx].a * weight_normalized);
            atomicAdd(&(u_mppi[ctrl_idx].delta), u_dist[dist_flat_idx].delta * weight_normalized);
        }
      }
    }
"""

try:
    _COMPILED_MPPI_MODULE = SourceModule(_MPPI_CUDA_SOURCE, no_extern_c=True)
    MPPI_MODULE_CONTEXT = getattr(_pycuda_autoinit, "context", None)
    if MPPI_MODULE_CONTEXT is None:
        raise RuntimeError("Autoinit did not supply a CUDA context")
except cuda.CompileError as e:
    print("CUDA Compilation Error:", file=sys.stderr)
    print(e.stderr, file=sys.stderr)
    raise
except Exception as e:
    print(f"Error compiling MPPI CUDA module: {e}", file=sys.stderr)
    raise


class MPPI:
    visibility_methods = {
        "Ours": 0,  # CUDA: OURS (can be used for information gain if costmap is such)
        "Ours-Wide": 0,  # Alias for Ours (wider roads in planning)
        "Right": 0,  # Alias for Ours
        "Left": 0,  # Alias for Ours
        "Higgins": 1,  # CUDA: HIGGINS
        "Andersen": 2,  # CUDA: ANDERSEN
        "Nominal": 3,  # CUDA: NO_VISIBILITY (i.e., no explicit visibility cost term)
        "Infogain": 0,  # CUDA: OURS (assuming 'our_cost' handles infogain maps)
        "Dynamic": 0,  # Alias for Ours
        # If Infogain is a distinct CUDA method INFO_GAIN_LIKE (4), map to 4.
        "Ignore": 3,  # CUDA: NO_VISIBILITY (treat as no visibility cost)
    }

    # Define optimization_dtype as a class attribute
    optimization_dtype = np.dtype(
        [
            ("samples", np.int32),
            ("M", np.float32),
            ("dt", np.float32),
            ("num_controls", np.int32),
            ("num_obstacles", np.int32),
            ("x_init", np.float32, 4),
            ("x_goal", np.float32, 4),
            ("u_limits", np.float32, 2),
            ("u_dist_limits", np.float32, 2),
            ("Q", np.float32, 4),
            ("Qf", np.float32, 4),
            ("R", np.float32, 2),
            ("method", np.int32),
            ("c_lambda", np.float32),
            ("scan_range", np.float32),
            ("vehicle_length", np.float32),
            ("steering_rate_weight", np.float32),
        ]
    )

    costmap_dtype = np.dtype(
        [
            ("height", np.int32),
            ("width", np.int32),
            ("origin_x", np.float32),
            ("origin_y", np.float32),
            ("resolution", np.float32),
        ]
    )

    def __init__(
        self,
        vehicle,
        samples,
        seed,
        u_limits,
        u_dist_limits,
        M,
        Q,
        Qf,
        R,
        method,
        c_lambda,
        scan_range,
        debug=False,
        steering_rate_weight=0.0,
    ):
        """Initialize MPPI controller.

        Parameters mirror optimization / cost settings. Gaussian noise sampling
        is used for disturbances with std dev = u_dist_limits (clamped to u_limits).
        """
        if MPPI_MODULE_CONTEXT is None:
            raise RuntimeError("CUDA context unavailable for MPPI initialization")

        self.mppi_context = MPPI_MODULE_CONTEXT
        self.vehicle = vehicle
        self.samples = np.int32(samples)
        self.debug = debug

        # Host-side struct (single element array for ease of memcpy)
        self.optimization_args = np.zeros(1, dtype=MPPI.optimization_dtype)
        self.optimization_args["samples"] = self.samples
        self.optimization_args["M"] = np.float32(M)
        self.optimization_args["u_limits"] = np.array(u_limits, dtype=np.float32)
        self.optimization_args["u_dist_limits"] = np.array(
            u_dist_limits, dtype=np.float32
        )
        self.optimization_args["Q"] = np.array(Q, dtype=np.float32)
        self.optimization_args["Qf"] = np.array(Qf, dtype=np.float32)
        self.optimization_args["R"] = np.array(R, dtype=np.float32)

        if method not in MPPI.visibility_methods:
            raise ValueError(
                f"Unknown visibility method: {method}. Choices: {list(MPPI.visibility_methods.keys())}"
            )
        self.optimization_args["method"] = np.int32(MPPI.visibility_methods[method])
        self.optimization_args["c_lambda"] = np.float32(c_lambda)
        self.optimization_args["scan_range"] = np.float32(scan_range)
        self.optimization_args["vehicle_length"] = (
            np.float32(vehicle.L) if vehicle else np.float32(1.0)
        )
        self.optimization_args["steering_rate_weight"] = np.float32(
            steering_rate_weight
        )

        # Allocate GPU buffers inside context
        try:
            self.mppi_context.push()

            self.optimization_args_gpu = _cuda.mem_alloc(self.optimization_args.nbytes)  # type: ignore[attr-defined]
            _cuda.memcpy_htod(self.optimization_args_gpu, self.optimization_args)  # type: ignore[attr-defined]

            self.costmap_args = np.zeros(1, dtype=MPPI.costmap_dtype)
            self.costmap_args_gpu = _cuda.mem_alloc(self.costmap_args.nbytes)  # type: ignore[attr-defined]

            # RNG states
            block_cfg = (BLOCK_SIZE, 1, 1)
            grid_dim_x = max(1, int((self.samples + block_cfg[0] - 1) / block_cfg[0]))
            grid_cfg = (grid_dim_x, 1)
            curand_state_size = characterize.sizeof(
                "curandState", "#include <curand_kernel.h>"
            )
            self.globalState_gpu = _cuda.mem_alloc(  # type: ignore[attr-defined]
                block_cfg[0] * grid_cfg[0] * curand_state_size
            )
            setup_kernel_func = _COMPILED_MPPI_MODULE.get_function("setup_kernel")
            setup_kernel_func(
                self.globalState_gpu,
                np.uint32(
                    seed if seed is not None else np.random.randint(0, 2**32 - 1)
                ),
                block=block_cfg,
                grid=grid_cfg,
            )
        finally:
            try:
                self.mppi_context.pop()
            except Exception:
                pass

        # Diagnostics placeholders
        self.last_ess = None
        self.last_cost_min = None
        self.last_cost_max = None
        self.last_cost_mean = None

        if self.debug:
            print(
                f"[MPPI] Initialized samples={self.samples} c_lambda={c_lambda} u_limits={u_limits} "
                f"u_dist_limits={u_dist_limits} method={method} L={vehicle_length} steer_rate_w={steering_rate_weight}"
            )

    def set_steering_limit(self, max_steer_rad: float):
        self.optimization_args["u_limits"][0, 1] = np.float32(max_steer_rad)
        _cuda.memcpy_htod(self.optimization_args_gpu, self.optimization_args)  # type: ignore[attr-defined]
        if self.debug:
            print(
                f"[MPPI] Steering limit set to {max_steer_rad:.3f} rad ({np.degrees(max_steer_rad):.1f} deg)"
            )

    def sync_steering_limit_from_carla(self, actor):
        try:
            physics = actor.get_physics_control()
            max_deg = max(w.max_steer_angle for w in physics.wheels)
            self.set_steering_limit(np.deg2rad(max_deg))
        except Exception as e:
            if self.debug:
                print(f"[MPPI] Steering limit sync failed: {e}")

    def find_control(
        self, costmap, origin, resolution, x_init, x_goal, x_nom, u_nom, actors, dt
    ):
        if self.mppi_context is None:
            raise RuntimeError("MPPI has no CUDA context")

        # Host-side outputs
        u_mppi_host = np.zeros_like(u_nom, dtype=np.float32)
        u_dist_host_reshaped = np.zeros(
            (self.samples, u_nom.shape[0], u_nom.shape[1]), dtype=np.float32
        )
        u_weights_host = np.zeros(self.samples, dtype=np.float32)

        costmap_gpu = None
        actors_gpu = np.intp(0)
        u_nom_gpu = None
        x_nom_gpu = None
        u_mppi_gpu = None
        u_weight_gpu = None
        u_dist_gpu = None
        u_weight_min_gpu = None
        u_weight_total_gpu = None

        try:
            self.mppi_context.push()
            try:
                # Prepare costmap
                costmap_host = costmap.astype(np.float32)
                height, width = costmap_host.shape
                costmap_gpu = _cuda.mem_alloc(costmap_host.nbytes)  # type: ignore[attr-defined]
                _cuda.memcpy_htod(costmap_gpu, costmap_host)  # type: ignore[attr-defined]
                self.costmap_args["height"] = height
                self.costmap_args["width"] = width
                self.costmap_args["origin_x"] = origin[0]
                self.costmap_args["origin_y"] = origin[1]
                self.costmap_args["resolution"] = resolution
                _cuda.memcpy_htod(self.costmap_args_gpu, self.costmap_args)  # type: ignore[attr-defined]

                # Obstacles / actors
                num_actors = np.int32(len(actors))
                if num_actors > 0:
                    actors_host = np.array(actors, dtype=np.float32)
                    actors_gpu = _cuda.mem_alloc(actors_host.nbytes)  # type: ignore[attr-defined]
                    _cuda.memcpy_htod(actors_gpu, actors_host)  # type: ignore[attr-defined]

                # Nominal control & state trajectories
                u_nom_host = np.array(u_nom, dtype=np.float32)
                num_controls_timesteps, num_control_elements = u_nom_host.shape
                u_nom_gpu = _cuda.mem_alloc(u_nom_host.nbytes)  # type: ignore[attr-defined]
                _cuda.memcpy_htod(u_nom_gpu, u_nom_host)  # type: ignore[attr-defined]

                x_nom_host = np.array(x_nom, dtype=np.float32)
                x_nom_gpu = _cuda.mem_alloc(x_nom_host.nbytes)  # type: ignore[attr-defined]
                _cuda.memcpy_htod(x_nom_gpu, x_nom_host)  # type: ignore[attr-defined]

                # Disturbance & weights buffers
                u_weight_gpu = _cuda.mem_alloc(  # type: ignore[attr-defined]
                    int(self.samples * np.dtype(np.float32).itemsize)
                )
                u_dist_gpu = _cuda.mem_alloc(int(self.samples * u_nom_host.nbytes))  # type: ignore[attr-defined]

                # Update optimization args for this solve
                self.optimization_args["dt"] = np.float32(dt)
                self.optimization_args["num_controls"] = np.int32(
                    num_controls_timesteps
                )
                self.optimization_args["num_obstacles"] = num_actors
                self.optimization_args["x_init"] = np.array(x_init, dtype=np.float32)
                self.optimization_args["x_goal"] = np.array(x_goal, dtype=np.float32)
                _cuda.memcpy_htod(self.optimization_args_gpu, self.optimization_args)  # type: ignore[attr-defined]

                # Kernel handles
                perform_rollout_func = _COMPILED_MPPI_MODULE.get_function(
                    "perform_rollout"
                )
                min_weight_func = _COMPILED_MPPI_MODULE.get_function("min_weight")
                calculate_weights_func = _COMPILED_MPPI_MODULE.get_function(
                    "calculate_weights"
                )
                calculate_mppi_control_func = _COMPILED_MPPI_MODULE.get_function(
                    "calculate_mppi_control"
                )

                block_1d = (BLOCK_SIZE, 1, 1)
                grid_1d_x = max(1, int((self.samples + block_1d[0] - 1) / block_1d[0]))
                grid_1d = (grid_1d_x, 1)

                # Rollouts (costs written to u_weight_gpu)
                perform_rollout_func(
                    self.globalState_gpu,
                    costmap_gpu,
                    self.costmap_args_gpu,
                    x_nom_gpu,
                    u_nom_gpu,
                    actors_gpu,
                    self.optimization_args_gpu,
                    u_dist_gpu,
                    u_weight_gpu,
                    block=block_1d,
                    grid=grid_1d,
                )

                # Capture raw costs before weight transform
                raw_costs_host = np.zeros(self.samples, dtype=np.float32)
                _cuda.memcpy_dtoh(raw_costs_host, u_weight_gpu)  # type: ignore[attr-defined]

                # Find min cost
                u_weight_min_gpu = _cuda.mem_alloc(np.dtype(np.float32).itemsize)  # type: ignore[attr-defined]
                large_float_val = np.array([np.finfo(np.float32).max], dtype=np.float32)
                _cuda.memcpy_htod(u_weight_min_gpu, large_float_val)  # type: ignore[attr-defined]
                min_weight_func(
                    self.samples,
                    u_weight_gpu,
                    u_weight_min_gpu,
                    block=(BLOCK_SIZE, 1, 1),
                    grid=(1, 1, 1),
                    shared=BLOCK_SIZE * np.dtype(np.float32).itemsize,
                )

                # Convert to weights
                u_weight_total_gpu = _cuda.mem_alloc(np.dtype(np.float32).itemsize)  # type: ignore[attr-defined]
                _cuda.memset_d8(u_weight_total_gpu, 0, np.dtype(np.float32).itemsize)  # type: ignore[attr-defined]
                calculate_weights_func(
                    self.samples,
                    u_weight_gpu,
                    u_weight_min_gpu,
                    self.optimization_args["c_lambda"][0],
                    u_weight_total_gpu,
                    block=block_1d,
                    grid=grid_1d,
                )

                # Accumulate weighted disturbances into nominal
                u_mppi_gpu = _cuda.mem_alloc(u_nom_host.nbytes)  # type: ignore[attr-defined]
                _cuda.memcpy_dtod(u_mppi_gpu, u_nom_gpu, u_nom_host.nbytes)  # type: ignore[attr-defined]
                calculate_mppi_control_func(
                    self.samples,
                    u_nom_gpu,
                    u_dist_gpu,
                    np.int32(num_controls_timesteps),
                    u_weight_gpu,
                    u_weight_total_gpu,
                    u_mppi_gpu,
                    block=block_1d,
                    grid=grid_1d,
                )

                # Copy back
                _cuda.memcpy_dtoh(u_mppi_host, u_mppi_gpu)  # type: ignore[attr-defined]
                u_dist_raw_host = np.zeros(
                    self.samples * num_controls_timesteps * num_control_elements,
                    dtype=np.float32,
                )
                _cuda.memcpy_dtoh(u_dist_raw_host, u_dist_gpu)  # type: ignore[attr-defined]
                u_dist_host_reshaped = u_dist_raw_host.reshape(
                    (self.samples, num_controls_timesteps, num_control_elements)
                )
                _cuda.memcpy_dtoh(u_weights_host, u_weight_gpu)  # type: ignore[attr-defined]

                # Diagnostics
                weight_sum = float(np.sum(u_weights_host) + 1e-12)
                ess = (weight_sum**2) / (float(np.sum(u_weights_host**2)) + 1e-12)
                self.last_ess = ess
                if raw_costs_host.size:
                    self.last_cost_min = float(np.min(raw_costs_host))
                    self.last_cost_max = float(np.max(raw_costs_host))
                    self.last_cost_mean = float(np.mean(raw_costs_host))
                if self.debug:
                    pct = (ess / self.samples) * 100.0
                    print(
                        f"[MPPI] dt={dt:.3f} cost(min/mean/max)=({self.last_cost_min:.2f}/{self.last_cost_mean:.2f}/{self.last_cost_max:.2f}) ESS={ess:.1f}/{self.samples} ({pct:.1f}%)"
                    )

                # Clamp final control outputs within limits
                u_mppi_host[:, 0] = np.clip(
                    u_mppi_host[:, 0],
                    -self.optimization_args["u_limits"][0, 0],
                    self.optimization_args["u_limits"][0, 0],
                )
                u_mppi_host[:, 1] = np.clip(
                    u_mppi_host[:, 1],
                    -self.optimization_args["u_limits"][0, 1],
                    self.optimization_args["u_limits"][0, 1],
                )

            finally:
                # Free GPU temporaries
                for buf in [
                    costmap_gpu,
                    actors_gpu if not isinstance(actors_gpu, int) else None,
                    u_nom_gpu,
                    x_nom_gpu,
                    u_mppi_gpu,
                    u_weight_gpu,
                    u_dist_gpu,
                    u_weight_min_gpu,
                    u_weight_total_gpu,
                ]:
                    try:
                        if buf:
                            buf.free()
                    except Exception:
                        pass
        finally:
            try:
                self.mppi_context.pop()
            except Exception:
                pass

        return u_mppi_host, u_dist_host_reshaped, u_weights_host
