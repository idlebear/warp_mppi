"""
Warp-based MPPI implementation.

This module contains the NVIDIA Warp implementation of the MPPI controller,
migrated from the original PyCUDA version.
"""

import warp as wp
import numpy as np
from typing import Tuple, List

# Initialize Warp with quiet mode to suppress status messages
wp.config.quiet = True
wp.init()

# Visibility method constants
OURS = 0
HIGGINS = 1
ANDERSEN = 2
NO_VISIBILITY = 3
INFO_GAIN_LIKE = 4

COLLISION_PENALTY = 1e7


# Warp data structures
@wp.struct
class State:
    """Vehicle state representation."""

    x: wp.float32  # Position x
    y: wp.float32  # Position y
    v: wp.float32  # Velocity
    theta: wp.float32  # Heading angle


@wp.struct
class Control:
    """Vehicle control input."""

    a: wp.float32  # Acceleration
    delta: wp.float32  # Steering angle


@wp.struct
class Obstacle:
    """Circular obstacle representation."""

    x: wp.float32  # Center x
    y: wp.float32  # Center y
    radius: wp.float32  # Obstacle radius
    min_x: wp.float32  # Minimum distance point x
    min_y: wp.float32  # Minimum distance point y
    distance: wp.float32  # Distance to vehicle


@wp.struct
class OptimizationParams:
    """MPPI optimization parameters."""

    samples: wp.int32
    M: wp.float32
    dt: wp.float32
    num_controls: wp.int32
    num_obstacles: wp.int32
    vehicle_length: wp.float32
    c_lambda: wp.float32
    scan_range: wp.float32
    steering_rate_weight: wp.float32
    method: wp.int32


@wp.struct
class CostWeights:
    """Cost function weight matrices."""

    Q: wp.vec4  # State tracking weights [x, y, v, theta]
    Qf: wp.vec4  # Final state weights
    R: wp.vec2  # Control effort weights [a, delta]


@wp.struct
class Limits:
    """Control limits and noise parameters."""

    u_limits: wp.vec2  # Maximum control values [a_max, delta_max]
    u_dist_limits: wp.vec2  # Noise standard deviations


@wp.func
def euler_step(state: State, control: Control, vehicle_length: wp.float32) -> State:
    """Single Euler integration step for bicycle model dynamics."""
    result = State()
    result.x = state.v * wp.cos(state.theta)
    result.y = state.v * wp.sin(state.theta)
    result.v = control.a
    result.theta = state.v * wp.tan(control.delta) / vehicle_length
    return result


@wp.func
def update_state(state: State, derivative: State, dt: wp.float32) -> State:
    """Update state using derivative and timestep."""
    result = State()
    result.x = state.x + derivative.x * dt
    result.y = state.y + derivative.y * dt
    result.v = state.v + derivative.v * dt
    result.theta = state.theta + derivative.theta * dt
    return result


@wp.func
def runge_kutta_step(
    state: State, control: Control, vehicle_length: wp.float32, dt: wp.float32
) -> State:
    """4th-order Runge-Kutta integration step."""
    k1 = euler_step(state, control, vehicle_length)

    tmp_state = update_state(state, k1, dt * 0.5)
    k2 = euler_step(tmp_state, control, vehicle_length)

    tmp_state = update_state(state, k2, dt * 0.5)
    k3 = euler_step(tmp_state, control, vehicle_length)

    tmp_state = update_state(state, k3, dt)
    k4 = euler_step(tmp_state, control, vehicle_length)

    # Combine derivatives
    result = State()
    result.x = (k1.x + 2.0 * (k2.x + k3.x) + k4.x) / 6.0
    result.y = (k1.y + 2.0 * (k2.y + k3.y) + k4.y) / 6.0
    result.v = (k1.v + 2.0 * (k2.v + k3.v) + k4.v) / 6.0
    result.theta = (k1.theta + 2.0 * (k2.theta + k3.theta) + k4.theta) / 6.0

    return result


# Cost function implementations
@wp.func
def obstacle_cost(
    obstacles: wp.array(dtype=Obstacle),
    num_obstacles: wp.int32,
    px: wp.float32,
    py: wp.float32,
    vehicle_radius: wp.float32,
) -> wp.float32:
    """Compute collision cost with circular obstacles."""
    for i in range(num_obstacles):
        obstacle = obstacles[i]
        dx = obstacle.x - px
        dy = obstacle.y - py
        d_sq = dx * dx + dy * dy
        min_dist = obstacle.radius + vehicle_radius

        if d_sq < min_dist * min_dist:
            return COLLISION_PENALTY  # Large penalty for collision

    return 0.0


@wp.func
def higgins_cost(
    obstacles: wp.array(dtype=Obstacle),
    num_obstacles: wp.int32,
    M: wp.float32,
    px: wp.float32,
    py: wp.float32,
    scan_range: wp.float32,
) -> wp.float32:
    """Higgins visibility cost function."""
    cost = float(0.0)  # Dynamic variable for loop mutation
    r_fov_sq = scan_range * scan_range

    for i in range(num_obstacles):
        obstacle = obstacles[i]
        dx = obstacle.x - px
        dy = obstacle.y - py
        d_sq = dx * dx + dy * dy
        d = wp.sqrt(d_sq)

        inner = obstacle.radius / d * (r_fov_sq - d_sq)

        # Avoid overflow in exponential
        if inner > 60.0:
            score = inner
        else:
            score = wp.log(1.0 + wp.exp(inner))

        cost += M * score * score

    return cost


@wp.func
def andersen_cost(
    obstacles: wp.array(dtype=Obstacle),
    num_obstacles: wp.int32,
    M: wp.float32,
    px: wp.float32,
    py: wp.float32,
    vx: wp.float32,
    vy: wp.float32,
) -> wp.float32:
    """Andersen visibility cost function."""
    cost = float(0.0)  # Dynamic variable for loop mutation
    v = wp.sqrt(vx * vx + vy * vy)

    for i in range(num_obstacles):
        obstacle = obstacles[i]
        dx = obstacle.x - px
        dy = obstacle.y - py

        # Check if the obstacle is in front of the vehicle
        dot = dx * vx + dy * vy
        if dot > 0.0:
            d = wp.sqrt(dx * dx + dy * dy)
            # Andersen is a reward (negative cost)
            cost -= M * wp.acos(wp.clamp(dot / (d * v + 1e-8), -1.0, 1.0))
            break  # Only consider the closest obstacle

    return cost


@wp.func
def our_cost(
    costmap: wp.array2d(dtype=wp.float32),
    height: wp.int32,
    width: wp.int32,
    origin_x: wp.float32,
    origin_y: wp.float32,
    resolution: wp.float32,
    M: wp.float32,
    px: wp.float32,
    py: wp.float32,
) -> wp.float32:
    """Our visibility cost based on costmap."""
    # Convert world coordinates to map coordinates
    map_x = wp.int32(wp.round((px - origin_x) / resolution))
    map_y = wp.int32(wp.round((py - origin_y) / resolution))

    # Bounds checking
    if map_x < 0 or map_x >= width or map_y < 0 or map_y >= height:
        return COLLISION_PENALTY  # Large penalty for out of bounds

    # Sample costmap value
    cost_value = costmap[map_y, map_x]
    return -M * cost_value  # Negative because higher costmap values = lower cost


@wp.func
def wrap_angle(angle: wp.float32) -> wp.float32:
    """Wrap angle to [-pi, pi]."""
    while angle > wp.pi:
        angle -= 2.0 * wp.pi
    while angle < -wp.pi:
        angle += 2.0 * wp.pi
    return angle


# Random number generation utilities
@wp.func
def pcg_hash(input: wp.uint32) -> wp.uint32:
    """PCG hash function for random number generation."""
    state = input * wp.uint32(747796405) + wp.uint32(2891336453)
    word = ((state >> ((state >> wp.uint32(28)) + wp.uint32(4))) ^ state) * wp.uint32(
        277803737
    )
    return (word >> wp.uint32(22)) ^ word


@wp.func
def random_uniform(rng_state: wp.uint32) -> wp.vec2:
    """Generate uniform random number in [0, 1) and return (value, new_state)."""
    new_state = pcg_hash(rng_state)
    value = wp.float32(new_state) / wp.float32(0xFFFFFFFF)
    return wp.vec2(value, wp.float32(new_state))


@wp.func
def random_normal(rng_state: wp.uint32, mean: wp.float32, std: wp.float32) -> wp.vec2:
    """Generate normal random number using Box-Muller and return (value, new_state)."""
    # Get first uniform sample
    u1_result = random_uniform(rng_state)
    u1 = u1_result[0]
    state1 = wp.uint32(u1_result[1])

    # Get second uniform sample
    u2_result = random_uniform(state1)
    u2 = u2_result[0]
    final_state = wp.uint32(u2_result[1])

    # Ensure u1 > 0 to avoid log(0)
    u1 = wp.max(u1, 1e-8)

    # Box-Muller transform
    z0 = wp.sqrt(-2.0 * wp.log(u1)) * wp.cos(2.0 * wp.float32(wp.pi) * u2)
    value = mean + std * z0
    return wp.vec2(value, wp.float32(final_state))


# Main MPPI kernel - optimized version using flattened arrays
@wp.kernel
def mppi_rollout_kernel_fast(
    # Random states
    rand_states: wp.array(dtype=wp.uint32),
    # Trajectories - flattened arrays for better performance
    x_nominal_flat: wp.array(dtype=wp.float32),  # [num_states * 4]
    u_nominal_flat: wp.array(dtype=wp.float32),  # [num_controls * 2]
    # Obstacles and costmap
    obstacles_flat: wp.array(dtype=wp.float32),  # [num_obstacles * 3] (x, y, radius)
    costmap: wp.array2d(dtype=wp.float32),
    # Parameters
    params: OptimizationParams,
    weights: CostWeights,
    limits: Limits,
    # Outputs
    u_disturbances_flat: wp.array2d(dtype=wp.float32),  # [samples, num_controls * 2]
    costs: wp.array(dtype=wp.float32),  # [samples]
    # Costmap parameters
    costmap_height: wp.int32,
    costmap_width: wp.int32,
    costmap_origin_x: wp.float32,
    costmap_origin_y: wp.float32,
    costmap_resolution: wp.float32,
    # State parameters
    x_init: wp.vec4,
    x_goal: wp.vec4,
):
    """Optimized MPPI rollout kernel using flattened arrays."""
    sample_idx = wp.tid()

    if sample_idx >= params.samples:
        return

    # Initialize random state
    rng_state = rand_states[sample_idx]

    # Generate control disturbances for this sample
    for ctrl_idx in range(params.num_controls):
        # Generate acceleration noise and update RNG state
        a_result = random_normal(rng_state, 0.0, limits.u_dist_limits[0] / 4.0)
        a_noise = a_result[0]
        rng_state = wp.uint32(a_result[1])

        # Generate steering noise and update RNG state
        delta_result = random_normal(rng_state, 0.0, limits.u_dist_limits[1] / 4.0)
        delta_noise = delta_result[0]
        rng_state = wp.uint32(delta_result[1])

        # Clamp noise to disturbance limits first
        a_noise = wp.clamp(a_noise, -limits.u_dist_limits[0], limits.u_dist_limits[0])
        delta_noise = wp.clamp(
            delta_noise, -limits.u_dist_limits[1], limits.u_dist_limits[1]
        )

        # Access nominal control directly from flattened array
        u_nom_a = u_nominal_flat[ctrl_idx * 2 + 0]
        u_nom_delta = u_nominal_flat[ctrl_idx * 2 + 1]

        # Apply disturbance and then clamp to control limits
        a_disturbed = wp.clamp(
            u_nom_a + a_noise, -limits.u_limits[0], limits.u_limits[0]
        )
        delta_disturbed = wp.clamp(
            u_nom_delta + delta_noise, -limits.u_limits[1], limits.u_limits[1]
        )

        # Store disturbance (difference from nominal) in flattened format
        u_disturbances_flat[sample_idx, ctrl_idx * 2 + 0] = a_disturbed - u_nom_a
        u_disturbances_flat[sample_idx, ctrl_idx * 2 + 1] = (
            delta_disturbed - u_nom_delta
        )

    # Forward simulate trajectory and compute cost
    total_cost = float(0.0)

    # Initial state
    current_x = x_init[0]
    current_y = x_init[1]
    current_v = x_init[2]
    current_theta = x_init[3]

    for step in range(params.num_controls):
        # Apply disturbed control
        control_a = (
            u_nominal_flat[step * 2 + 0] + u_disturbances_flat[sample_idx, step * 2 + 0]
        )
        control_delta = (
            u_nominal_flat[step * 2 + 1] + u_disturbances_flat[sample_idx, step * 2 + 1]
        )

        # Integrate dynamics - simplified Euler for speed
        dt = params.dt
        current_x += current_v * wp.cos(current_theta) * dt
        current_y += current_v * wp.sin(current_theta) * dt
        current_v += control_a * dt
        current_theta += current_v * wp.tan(control_delta) / params.vehicle_length * dt

        # Access nominal state from flattened array
        nom_x = x_nominal_flat[(step + 1) * 4 + 0]
        nom_y = x_nominal_flat[(step + 1) * 4 + 1]
        nom_v = x_nominal_flat[(step + 1) * 4 + 2]
        nom_theta = x_nominal_flat[(step + 1) * 4 + 3]

        # Compute state tracking cost
        error_x = nom_x - current_x
        error_y = nom_y - current_y
        error_v = nom_v - current_v
        error_theta = nom_theta - current_theta

        # Wrap angle error
        while error_theta > wp.pi:
            error_theta -= 2.0 * wp.pi
        while error_theta < -wp.pi:
            error_theta += 2.0 * wp.pi

        state_cost = (
            error_x * weights.Q[0] * error_x
            + error_y * weights.Q[1] * error_y
            + error_v * weights.Q[2] * error_v
            + error_theta * weights.Q[3] * error_theta
        )

        # Compute control cost
        dist_a = u_disturbances_flat[sample_idx, step * 2 + 0]
        dist_delta = u_disturbances_flat[sample_idx, step * 2 + 1]
        control_cost = (
            dist_a * weights.R[0] * dist_a + dist_delta * weights.R[1] * dist_delta
        )

        # Obstacle cost - simplified
        vehicle_radius = params.vehicle_length / 2.0
        obstacle_cost_val = float(0.0)  # Declare as dynamic variable
        for i in range(params.num_obstacles):
            obs_x = obstacles_flat[i * 3 + 0]
            obs_y = obstacles_flat[i * 3 + 1]
            obs_radius = obstacles_flat[i * 3 + 2]

            dx = obs_x - current_x
            dy = obs_y - current_y
            d_sq = dx * dx + dy * dy
            min_dist = obs_radius + vehicle_radius

            if d_sq < min_dist * min_dist:
                obstacle_cost_val = COLLISION_PENALTY
                break

        total_cost += state_cost + control_cost + obstacle_cost_val

        # Early termination for collision
        if total_cost >= COLLISION_PENALTY:
            break

    # Final state cost
    final_error_x = x_goal[0] - current_x
    final_error_y = x_goal[1] - current_y
    final_error_v = x_goal[2] - current_v
    final_error_theta = x_goal[3] - current_theta

    # Wrap final angle error
    while final_error_theta > wp.pi:
        final_error_theta -= 2.0 * wp.pi
    while final_error_theta < -wp.pi:
        final_error_theta += 2.0 * wp.pi

    final_cost = (
        final_error_x * weights.Qf[0] * final_error_x
        + final_error_y * weights.Qf[1] * final_error_y
        + final_error_v * weights.Qf[2] * final_error_v
        + final_error_theta * weights.Qf[3] * final_error_theta
    )

    total_cost += final_cost

    # Handle NaN/inf
    if wp.isnan(total_cost) or wp.isinf(total_cost):
        total_cost = COLLISION_PENALTY

    costs[sample_idx] = total_cost

    # Update random state
    rand_states[sample_idx] = rng_state


# Weight computation kernels
@wp.kernel
def find_min_cost_kernel(
    costs: wp.array(dtype=wp.float32), min_cost: wp.array(dtype=wp.float32)
):
    """Find minimum cost across all samples."""
    tid = wp.tid()

    # Simple reduction for now - can be optimized with shared memory
    if tid == 0:
        min_val = costs[0]
        for i in range(len(costs)):
            if costs[i] < min_val:
                min_val = costs[i]
        min_cost[0] = min_val


@wp.kernel
def compute_weights_kernel(
    costs: wp.array(dtype=wp.float32),
    min_cost: wp.array(dtype=wp.float32),
    c_lambda: wp.float32,
    weights: wp.array(dtype=wp.float32),
    total_weight: wp.array(dtype=wp.float32),
):
    """Compute importance weights from costs."""
    tid = wp.tid()

    if tid >= len(costs):
        return

    # Guard against pathological small temperature
    lambda_safe = wp.max(c_lambda, 1e-6)

    # Convert cost to weight
    # If the cost includes a large collision penalty, zero it out
    if costs[tid] >= COLLISION_PENALTY:
        weight = 0.0
    else:
        diff = (costs[tid] - min_cost[0]) / lambda_safe
        diff = wp.clamp(diff, 0.0, 60.0)  # Avoid underflow
        weight = wp.exp(-diff)

    if wp.isnan(weight):
        weight = 0.0

    weights[tid] = weight

    # Atomic add to total weight
    wp.atomic_add(total_weight, 0, weight)


@wp.kernel
def compute_mppi_update_kernel_fast(
    u_nominal_flat: wp.array(dtype=wp.float32),
    u_disturbances_flat: wp.array2d(dtype=wp.float32),
    weights: wp.array(dtype=wp.float32),
    total_weight: wp.array(dtype=wp.float32),
    u_mppi_flat: wp.array(dtype=wp.float32),
):
    """Compute MPPI control update using flattened arrays."""
    ctrl_element_idx = wp.tid()

    num_controls = len(u_nominal_flat)
    if ctrl_element_idx >= num_controls:
        return

    # Initialize with nominal control
    u_mppi_flat[ctrl_element_idx] = u_nominal_flat[ctrl_element_idx]

    # Weighted sum of disturbances
    total_w = total_weight[0]
    if total_w > 1e-12:
        weighted_disturbance = float(0.0)  # Declare as dynamic variable

        for sample_idx in range(len(weights)):
            w_norm = weights[sample_idx] / total_w
            weighted_disturbance += (
                u_disturbances_flat[sample_idx, ctrl_element_idx] * w_norm
            )

        # Add weighted disturbance to nominal
        u_mppi_flat[ctrl_element_idx] = (
            u_nominal_flat[ctrl_element_idx] + weighted_disturbance
        )


class WarpMPPI:
    """
    Warp-based MPPI controller implementation.

    This class provides the same interface as the PyCUDA version but uses
    NVIDIA Warp for GPU acceleration.
    """

    visibility_methods = {
        "Ours": OURS,
        "Ours-Wide": OURS,
        "Right": OURS,
        "Left": OURS,
        "Higgins": HIGGINS,
        "Andersen": ANDERSEN,
        "Nominal": NO_VISIBILITY,
        "Infogain": OURS,
        "Dynamic": OURS,
        "Ignore": NO_VISIBILITY,
    }

    def __init__(
        self,
        vehicle=None,  # For compatibility
        samples: int = 1000,
        seed: int = None,
        u_limits: List[float] = [3.0, 0.5],
        u_dist_limits: List[float] = [0.5, 0.1],
        Q: List[float] = [1.0, 1.0, 0.1, 0.1],
        Qf: List[float] = [10.0, 10.0, 1.0, 1.0],
        R: List[float] = [0.1, 0.1],
        M: float = 1.0,
        c_lambda: float = 1.0,
        scan_range: float = 10.0,
        method: str = "Ours",
        debug: bool = False,
        steering_rate_weight: float = 0.0,
    ):
        """Initialize Warp-based MPPI controller."""

        self.samples = samples
        self.debug = debug
        self.device = "cuda:0"

        # Create parameter structures
        self.params = OptimizationParams()
        self.params.samples = samples
        self.params.M = M
        self.params.vehicle_length = vehicle.L if vehicle else 1.0
        self.params.c_lambda = c_lambda
        self.params.scan_range = scan_range
        self.params.steering_rate_weight = steering_rate_weight

        if method not in self.visibility_methods:
            raise ValueError(f"Unknown method: {method}")
        self.params.method = self.visibility_methods[method]

        self.weights = CostWeights()
        self.weights.Q = wp.vec4(Q[0], Q[1], Q[2], Q[3])
        self.weights.Qf = wp.vec4(Qf[0], Qf[1], Qf[2], Qf[3])
        self.weights.R = wp.vec2(R[0], R[1])

        self.limits = Limits()
        self.limits.u_limits = wp.vec2(u_limits[0], u_limits[1])
        self.limits.u_dist_limits = wp.vec2(u_dist_limits[0], u_dist_limits[1])

        # Initialize random states
        if seed is None:
            seed = np.random.randint(0, 2**31)

        rng_states = np.random.RandomState(seed).randint(
            0, 2**31, size=samples, dtype=np.uint32
        )
        self.rand_states = wp.array(rng_states, dtype=wp.uint32, device=self.device)

        # Diagnostics
        self.last_ess = None
        self.last_cost_min = None
        self.last_cost_max = None
        self.last_cost_mean = None

        if debug:
            print(f"[WarpMPPI] Initialized with {samples} samples, method={method}")

    def _pcg_hash(self, input_val: int) -> int:
        """CPU version of PCG hash for advancing RNG states."""
        state = (input_val * 747796405 + 2891336453) & 0xFFFFFFFF
        word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737 & 0xFFFFFFFF
        return ((word >> 22) ^ word) & 0xFFFFFFFF

    def set_steering_limit(self, max_steer_rad: float):
        """Set steering limit."""
        self.limits.u_limits = wp.vec2(self.limits.u_limits[0], max_steer_rad)
        if self.debug:
            print(f"[WarpMPPI] Steering limit set to {max_steer_rad:.3f} rad")

    def find_control(
        self,
        costmap: np.ndarray,
        origin: Tuple[float, float],
        resolution: float,
        x_init: List[float],
        x_goal: List[float],
        x_nom: np.ndarray,
        u_nom: np.ndarray,
        actors: List[List[float]],
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find optimal control using Warp-accelerated MPPI.

        Args:
            costmap: Visibility/occupancy map
            origin: Map origin (x, y)
            resolution: Map resolution
            x_init: Initial state [x, y, v, theta]
            x_goal: Goal state [x, y, v, theta]
            x_nom: Nominal state trajectory
            u_nom: Nominal control sequence
            actors: List of obstacles [[x, y, radius], ...]
            dt: Time step

        Returns:
            Tuple of (optimal_controls, control_samples, weights)
        """

        # Update parameters for this solve
        self.params.dt = dt
        self.params.num_controls = len(u_nom)
        self.params.num_obstacles = len(actors)

        # Convert inputs to Warp arrays
        costmap_wp = wp.array(
            costmap.astype(np.float32), dtype=wp.float32, device=self.device
        )

        # Convert nominal trajectory directly from numpy arrays
        x_nom_flat = x_nom.astype(np.float32).flatten()
        x_nominal_wp = wp.array(x_nom_flat, dtype=wp.float32, device=self.device)

        # Convert nominal controls directly from numpy arrays
        u_nom_flat = u_nom.astype(np.float32).flatten()
        u_nominal_wp = wp.array(u_nom_flat, dtype=wp.float32, device=self.device)

        # Convert obstacles to flattened format
        obstacles_flat = np.zeros(len(actors) * 3, dtype=np.float32)
        for i, actor in enumerate(actors):
            obstacles_flat[i * 3 + 0] = actor[0]  # x
            obstacles_flat[i * 3 + 1] = actor[1]  # y
            obstacles_flat[i * 3 + 2] = actor[2]  # radius
        obstacles_wp = wp.array(obstacles_flat, dtype=wp.float32, device=self.device)

        # Allocate output arrays - use flattened format
        u_disturbances_wp = wp.zeros(
            (self.samples, len(u_nom) * 2), dtype=wp.float32, device=self.device
        )
        costs_wp = wp.zeros(self.samples, dtype=wp.float32, device=self.device)

        # Launch optimized rollout kernel
        wp.launch(
            mppi_rollout_kernel_fast,
            dim=self.samples,
            inputs=[
                self.rand_states,
                x_nominal_wp,
                u_nominal_wp,
                obstacles_wp,
                costmap_wp,
                self.params,
                self.weights,
                self.limits,
                u_disturbances_wp,
                costs_wp,
                # Costmap parameters
                costmap.shape[0],  # height
                costmap.shape[1],  # width
                origin[0],  # origin_x
                origin[1],  # origin_y
                resolution,
                # State parameters
                wp.vec4(x_init[0], x_init[1], x_init[2], x_init[3]),
                wp.vec4(x_goal[0], x_goal[1], x_goal[2], x_goal[3]),
            ],
            device=self.device,
        )

        # Find minimum cost
        min_cost_wp = wp.zeros(1, dtype=wp.float32, device=self.device)
        wp.launch(
            find_min_cost_kernel,
            dim=1,
            inputs=[costs_wp, min_cost_wp],
            device=self.device,
        )

        min_cost = min_cost_wp.numpy()[0]
        if np.isnan(min_cost) or min_cost >= COLLISION_PENALTY:
            # All trajectories are invalid
            if self.debug:
                print("[WarpMPPI] All trajectories invalid, returning zeros")
            u_optimal = np.zeros((len(u_nom), 2), dtype=np.float32)
            u_samples = np.zeros((self.samples, len(u_nom), 2), dtype=np.float32)
            u_weights = np.zeros(self.samples, dtype=np.float32)
            self.last_ess = 0.0
            self.last_cost_min = float("nan")
            self.last_cost_max = float("nan")
            self.last_cost_mean = float("nan")
            return u_optimal, u_samples, u_weights

        # Compute weights
        weights_wp = wp.zeros(self.samples, dtype=wp.float32, device=self.device)
        total_weight_wp = wp.zeros(1, dtype=wp.float32, device=self.device)

        wp.launch(
            compute_weights_kernel,
            dim=self.samples,
            inputs=[
                costs_wp,
                min_cost_wp,
                self.params.c_lambda,
                weights_wp,
                total_weight_wp,
            ],
            device=self.device,
        )

        # Compute MPPI update
        u_mppi_wp = wp.zeros(len(u_nom) * 2, dtype=wp.float32, device=self.device)
        wp.launch(
            compute_mppi_update_kernel_fast,
            dim=len(u_nom) * 2,  # Each control element separately
            inputs=[
                u_nominal_wp,
                u_disturbances_wp,
                weights_wp,
                total_weight_wp,
                u_mppi_wp,
            ],
            device=self.device,
        )

        # Copy results back to host - much simpler now!
        costs_host = costs_wp.numpy()
        weights_host = weights_wp.numpy()
        u_mppi_host = u_mppi_wp.numpy()
        u_disturbances_host = u_disturbances_wp.numpy()

        # Convert flattened arrays back to expected format
        u_optimal = u_mppi_host.reshape((len(u_nom), 2)).astype(np.float32)
        u_samples = u_disturbances_host.reshape((self.samples, len(u_nom), 2)).astype(
            np.float32
        )

        # Compute diagnostics
        if self.debug:
            if weights_host.sum() > 1e-12:
                self.last_ess = (weights_host.sum() ** 2) / (
                    np.sum(weights_host**2) + 1e-12
                )
            else:
                self.last_ess = 0.0

            self.last_cost_min = float(np.min(costs_host))
            self.last_cost_max = float(np.max(costs_host))
            self.last_cost_mean = float(np.mean(costs_host))

            pct = (self.last_ess / self.samples) * 100.0
            print(
                f"[WarpMPPI] dt={dt:.3f} cost(min/mean/max)="
                f"({self.last_cost_min:.2f}/{self.last_cost_mean:.2f}/{self.last_cost_max:.2f}) "
                f"ESS={self.last_ess:.1f}/{self.samples} ({pct:.1f}%)"
            )

        # Apply final control limits
        u_optimal = np.clip(
            u_optimal,
            [-self.limits.u_limits[0], -self.limits.u_limits[1]],
            [self.limits.u_limits[0], self.limits.u_limits[1]],
        )

        return u_optimal, u_samples, weights_host


# Factory function for backward compatibility
def create_mppi_controller(**kwargs) -> WarpMPPI:
    """Create a Warp-based MPPI controller with the same interface."""
    return WarpMPPI(**kwargs)


# Alias for compatibility
MPPI = WarpMPPI


if __name__ == "__main__":
    # Basic functionality test
    print("Testing Warp MPPI implementation...")

    # Create a simple controller
    controller = WarpMPPI(
        samples=100,  # Smaller for testing
        vehicle_length=2.5,
        u_limits=[3.0, 0.5],
        u_dist_limits=[0.5, 0.1],
        Q=[1.0, 1.0, 0.1, 0.1],
        Qf=[10.0, 10.0, 1.0, 1.0],
        R=[0.1, 0.1],
        debug=True,
    )

    print("✓ Warp MPPI controller created successfully!")

    # Test with dummy data
    costmap = np.random.rand(50, 50).astype(np.float32)
    x_init = [0.0, 0.0, 1.0, 0.0]
    x_goal = [10.0, 0.0, 1.0, 0.0]
    x_nom = np.array([[i, 0.0, 1.0, 0.0] for i in range(11)])
    u_nom = np.array([[0.0, 0.0] for _ in range(10)])
    actors = [[5.0, 2.0, 1.0]]

    try:
        u_opt, u_samples, weights = controller.find_control(
            costmap=costmap,
            origin=(0.0, 0.0),
            resolution=0.1,
            x_init=x_init,
            x_goal=x_goal,
            x_nom=x_nom,
            u_nom=u_nom,
            actors=actors,
            dt=0.1,
        )

        print(f"✓ Control computed successfully!")
        print(f"  Optimal control shape: {u_opt.shape}")
        print(f"  Samples shape: {u_samples.shape}")
        print(f"  Weights shape: {weights.shape}")
        print(f"  ESS: {controller.last_ess:.2f}")

    except Exception as e:
        print(f"✗ Error during control computation: {e}")
        import traceback

        traceback.print_exc()

    print("Warp MPPI implementation test complete.")
