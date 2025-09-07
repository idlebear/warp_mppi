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
        dx = obstacle.min_x - px
        dy = obstacle.min_y - py

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


@wp.struct
class RngState:
    """Wrapper for RNG state to enable pass-by-reference semantics."""

    state: wp.uint32


@wp.func
def random_uniform(rng: RngState) -> wp.float32:
    """Generate uniform random number in [0, 1) and advance state."""
    rng.state = pcg_hash(rng.state)
    return wp.float32(rng.state) / wp.float32(0xFFFFFFFF)


@wp.func
def random_normal(rng: RngState, mean: wp.float32, std: wp.float32) -> wp.float32:
    """Generate normal random number using Box-Muller transform and advance state."""
    # Get first uniform sample
    u1 = random_uniform(rng)

    # Get second uniform sample
    u2 = random_uniform(rng)

    # Ensure u1 > 0 to avoid log(0)
    u1 = wp.max(u1, 1e-8)

    # Box-Muller transform
    z0 = wp.sqrt(-2.0 * wp.log(u1)) * wp.cos(2.0 * wp.float32(wp.pi) * u2)
    return mean + std * z0


# Main MPPI kernel
@wp.kernel
def mppi_rollout_kernel(
    # Random states
    rand_states: wp.array(dtype=wp.uint32),
    # Trajectories
    x_nominal: wp.array(dtype=State),
    u_nominal: wp.array(dtype=Control),
    # Obstacles and costmap
    obstacles: wp.array(dtype=Obstacle),
    costmap: wp.array2d(dtype=wp.float32),
    # Parameters
    params: OptimizationParams,
    weights: CostWeights,
    limits: Limits,
    # Outputs
    u_disturbances: wp.array2d(dtype=Control),  # [samples, num_controls]
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
    """Main MPPI rollout kernel - computes costs for sampled trajectories."""
    sample_idx = wp.tid()

    if sample_idx >= params.samples:
        return

    # Initialize random state
    rng = RngState()
    rng.state = rand_states[sample_idx]

    # Generate control disturbances for this sample
    for ctrl_idx in range(params.num_controls):
        # Generate acceleration noise and update RNG state
        a_noise = random_normal(rng, 0.0, limits.u_dist_limits[0] / 4.0)

        # Generate steering noise and update RNG state
        delta_noise = random_normal(rng, 0.0, limits.u_dist_limits[1] / 4.0)

        # Clamp noise to disturbance limits first
        a_noise = wp.clamp(a_noise, -limits.u_dist_limits[0], limits.u_dist_limits[0])
        delta_noise = wp.clamp(
            delta_noise, -limits.u_dist_limits[1], limits.u_dist_limits[1]
        )

        # Apply disturbance and then clamp to control limits
        a_disturbed = wp.clamp(
            u_nominal[ctrl_idx].a + a_noise, -limits.u_limits[0], limits.u_limits[0]
        )
        delta_disturbed = wp.clamp(
            u_nominal[ctrl_idx].delta + delta_noise,
            -limits.u_limits[1],
            limits.u_limits[1],
        )

        wp.printf(
            "Sample %d, Control %d: a_nom=%.3f, a_dist=%.3f, a_dist_limit=%.3f, delta_nom=%.3f, delta_dist=%.3f, delta_dist_limit=%.3f\n",
            sample_idx,
            ctrl_idx,
            u_nominal[ctrl_idx].a,
            a_disturbed - u_nominal[ctrl_idx].a,
            limits.u_dist_limits[0],
            u_nominal[ctrl_idx].delta,
            delta_disturbed - u_nominal[ctrl_idx].delta,
            limits.u_dist_limits[1],
        )

        # Store disturbance (difference from nominal)
        disturbance = Control()
        disturbance.a = a_disturbed - u_nominal[ctrl_idx].a
        disturbance.delta = delta_disturbed - u_nominal[ctrl_idx].delta
        u_disturbances[sample_idx, ctrl_idx] = disturbance

    # Forward simulate trajectory and compute cost
    total_cost = float(0.0)  # Dynamic variable for loop mutation

    # Initial state
    current_state = State()
    current_state.x = x_init[0]
    current_state.y = x_init[1]
    current_state.v = x_init[2]
    current_state.theta = x_init[3]

    for step in range(params.num_controls):
        # Apply disturbed control
        control = Control()
        control.a = u_nominal[step].a + u_disturbances[sample_idx, step].a
        control.delta = u_nominal[step].delta + u_disturbances[sample_idx, step].delta

        # Integrate dynamics
        derivative = runge_kutta_step(
            current_state, control, params.vehicle_length, params.dt
        )
        current_state = update_state(current_state, derivative, params.dt)

        # Compute state tracking cost
        state_error = State()
        state_error.x = x_nominal[step + 1].x - current_state.x
        state_error.y = x_nominal[step + 1].y - current_state.y
        state_error.v = x_nominal[step + 1].v - current_state.v
        state_error.theta = wrap_angle(x_nominal[step + 1].theta - current_state.theta)

        state_cost = (
            state_error.x * weights.Q[0] * state_error.x
            + state_error.y * weights.Q[1] * state_error.y
            + state_error.v * weights.Q[2] * state_error.v
            + state_error.theta * weights.Q[3] * state_error.theta
        )

        # Compute control cost
        control_cost = (
            u_disturbances[sample_idx, step].a
            * weights.R[0]
            * u_disturbances[sample_idx, step].a
            + u_disturbances[sample_idx, step].delta
            * weights.R[1]
            * u_disturbances[sample_idx, step].delta
        )

        # Steering rate penalty
        steering_rate_cost = 0.0
        if params.steering_rate_weight > 0.0 and step > 0:
            prev_delta = (
                u_nominal[step - 1].delta + u_disturbances[sample_idx, step - 1].delta
            )
            curr_delta = u_nominal[step].delta + u_disturbances[sample_idx, step].delta
            rate = curr_delta - prev_delta
            steering_rate_cost = params.steering_rate_weight * rate * rate

        # Obstacle cost
        vehicle_radius = params.vehicle_length / 2.0
        obstacle_cost_val = obstacle_cost(
            obstacles,
            params.num_obstacles,
            current_state.x,
            current_state.y,
            vehicle_radius,
        )

        # Visibility cost
        visibility_cost_val = 0.0
        if params.method == OURS:
            visibility_cost_val = our_cost(
                costmap,
                costmap_height,
                costmap_width,
                costmap_origin_x,
                costmap_origin_y,
                costmap_resolution,
                params.M,
                current_state.x,
                current_state.y,
            )
        elif params.method == HIGGINS:
            visibility_cost_val = higgins_cost(
                obstacles,
                params.num_obstacles,
                params.M,
                current_state.x,
                current_state.y,
                params.scan_range,
            )
        elif params.method == ANDERSEN:
            # Velocity for Andersen (from nominal trajectory difference)
            if step > 0:
                vx = x_nominal[step + 1].x - x_nominal[step].x
                vy = x_nominal[step + 1].y - x_nominal[step].y
                visibility_cost_val = andersen_cost(
                    obstacles,
                    params.num_obstacles,
                    params.M,
                    current_state.x,
                    current_state.y,
                    vx,
                    vy,
                )

        total_cost += (
            state_cost
            + control_cost
            + steering_rate_cost
            + obstacle_cost_val
            + visibility_cost_val
        )

        # Early termination for collision
        if total_cost >= COLLISION_PENALTY:
            break

    # Final state cost
    final_error = State()
    final_error.x = x_goal[0] - current_state.x
    final_error.y = x_goal[1] - current_state.y
    final_error.v = x_goal[2] - current_state.v
    final_error.theta = wrap_angle(x_goal[3] - current_state.theta)

    final_cost = (
        final_error.x * weights.Qf[0] * final_error.x
        + final_error.y * weights.Qf[1] * final_error.y
        + final_error.v * weights.Qf[2] * final_error.v
        + final_error.theta * weights.Qf[3] * final_error.theta
    )

    total_cost += final_cost

    # Handle NaN/inf
    if wp.isnan(total_cost) or wp.isinf(total_cost):
        total_cost = COLLISION_PENALTY

    costs[sample_idx] = total_cost

    # Update random state
    rand_states[sample_idx] = rng.state


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
def compute_mppi_update_kernel(
    u_nominal: wp.array(dtype=Control),
    u_disturbances: wp.array2d(dtype=Control),
    weights: wp.array(dtype=wp.float32),
    total_weight: wp.array(dtype=wp.float32),
    u_mppi: wp.array(dtype=Control),
):
    """Compute MPPI control update."""
    ctrl_idx = wp.tid()

    if ctrl_idx >= len(u_nominal):
        return

    # Initialize with nominal control
    u_mppi[ctrl_idx] = u_nominal[ctrl_idx]

    # Weighted sum of disturbances
    total_w = total_weight[0]
    if total_w > 1e-12:
        weighted_a = float(0.0)  # Declare as dynamic variable
        weighted_delta = float(0.0)  # Declare as dynamic variable

        for sample_idx in range(len(weights)):
            w_norm = weights[sample_idx] / total_w
            weighted_a += u_disturbances[sample_idx, ctrl_idx].a * w_norm
            weighted_delta += u_disturbances[sample_idx, ctrl_idx].delta * w_norm

        # Add weighted disturbance to nominal
        updated_control = Control()
        updated_control.a = u_nominal[ctrl_idx].a + weighted_a
        updated_control.delta = u_nominal[ctrl_idx].delta + weighted_delta
        u_mppi[ctrl_idx] = updated_control


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
        vehicle_length: float = 2.5,
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
        self.params.vehicle_length = vehicle_length
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

        # Convert nominal trajectory
        x_nom_states = []
        for x in x_nom:
            state = State()
            state.x = x[0]
            state.y = x[1]
            state.v = x[2]
            state.theta = x[3]
            x_nom_states.append(state)
        x_nominal_wp = wp.array(x_nom_states, dtype=State, device=self.device)

        # Convert nominal controls
        u_nom_controls = []
        for u in u_nom:
            control = Control()
            control.a = u[0]
            control.delta = u[1]
            u_nom_controls.append(control)
        u_nominal_wp = wp.array(u_nom_controls, dtype=Control, device=self.device)

        # Convert obstacles
        obstacles_list = []
        for actor in actors:
            obstacle = Obstacle()
            obstacle.x = actor[0]
            obstacle.y = actor[1]
            obstacle.radius = actor[2]
            obstacle.min_x = actor[0]
            obstacle.min_y = actor[1]
            obstacle.distance = 0.0
            obstacles_list.append(obstacle)
        obstacles_wp = wp.array(obstacles_list, dtype=Obstacle, device=self.device)

        # Allocate output arrays
        u_disturbances_wp = wp.zeros(
            (self.samples, len(u_nom)), dtype=Control, device=self.device
        )
        costs_wp = wp.zeros(self.samples, dtype=wp.float32, device=self.device)

        # Launch main rollout kernel
        wp.launch(
            mppi_rollout_kernel,
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
        u_mppi_wp = wp.zeros(len(u_nom), dtype=Control, device=self.device)
        wp.launch(
            compute_mppi_update_kernel,
            dim=len(u_nom),
            inputs=[
                u_nominal_wp,
                u_disturbances_wp,
                weights_wp,
                total_weight_wp,
                u_mppi_wp,
            ],
            device=self.device,
        )

        # Copy results back to host
        costs_host = costs_wp.numpy()
        weights_host = weights_wp.numpy()
        u_mppi_host = u_mppi_wp.numpy()
        u_disturbances_host = u_disturbances_wp.numpy()

        # Convert output format to match original API
        u_optimal = np.array(
            [
                [u_mppi_host[i]["a"], u_mppi_host[i]["delta"]]
                for i in range(len(u_mppi_host))
            ],
            dtype=np.float32,
        )
        u_samples = np.array(
            [
                [
                    [u_disturbances_host[s, c]["a"], u_disturbances_host[s, c]["delta"]]
                    for c in range(u_disturbances_host.shape[1])
                ]
                for s in range(u_disturbances_host.shape[0])
            ],
            dtype=np.float32,
        )

        # Compute diagnostics
        if weights_host.sum() > 1e-12:
            self.last_ess = (weights_host.sum() ** 2) / (
                np.sum(weights_host**2) + 1e-12
            )
        else:
            self.last_ess = 0.0

        self.last_cost_min = float(np.min(costs_host))
        self.last_cost_max = float(np.max(costs_host))
        self.last_cost_mean = float(np.mean(costs_host))

        if self.debug:
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
