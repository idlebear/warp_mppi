import atexit  # For ensuring context cleanup on exit
import contextlib
import functools
import numpy as np
from time import perf_counter

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda

# import pycuda.autoinit  # Removed to allow flexible context management
from pycuda.compiler import SourceModule
from pycuda import characterize

import sys
from typing import Any

try:
    from .trajectory_eval_pycuda import score_oce_scene_rollouts_device
except Exception as _oce_import_error:  # pragma: no cover - optional OCE path
    score_oce_scene_rollouts_device = None

try:
    from .discrete_oce_pycuda import evaluate_discrete_oce_rollouts_gpu
except Exception as _discrete_oce_import_error:  # pragma: no cover - optional OCE path
    evaluate_discrete_oce_rollouts_gpu = None


_cuda: Any = cuda  # alias to satisfy static analyzers

# --- CUDA Context Management ---
_MODULE_CONTEXT = None  # Stores the context active during SourceModule compilation
_owns_module_context_ref = False  # True if this module must detach _MODULE_CONTEXT
_module_context_pushed = (
    False  # True while this module has _MODULE_CONTEXT on the stack
)


def _establish_module_context():
    """
    Ensures a CUDA context is available for SourceModule compilation and sets
    _MODULE_CONTEXT. Called once when the module is loaded.

    Retains the primary context on the current device, or device 0 if no
    context is current, and pushes it only long enough to compile the kernels.
    """
    global _MODULE_CONTEXT, _owns_module_context_ref, _module_context_pushed
    # This function should only effectively run once at module import.
    if _MODULE_CONTEXT is not None:
        return

    cuda.init()  # Ensure CUDA driver is initialized before querying contexts

    # A current context handle is not retained ownership. Use it only to choose
    # the device, then retain our own primary-context reference on that device.
    device = None
    try:
        current_context = cuda.Context.get_current()
        if current_context is not None:
            device = cuda.Context.get_device()
    except cuda.LogicError:
        pass

    if device is None:
        device = cuda.Device(0)  # Default to device 0
    try:
        _MODULE_CONTEXT = device.retain_primary_context()
        _MODULE_CONTEXT.push()
        _owns_module_context_ref = True
    except AttributeError:
        # Older PyCUDA fallback. make_context() creates and pushes a user context.
        _MODULE_CONTEXT = device.make_context()
        _owns_module_context_ref = True

    _module_context_pushed = True


def _pop_module_context_after_compile():
    """Undo the temporary import-time push performed for kernel compilation."""
    global _module_context_pushed
    if _module_context_pushed and _MODULE_CONTEXT is not None:
        _MODULE_CONTEXT.pop()
        _module_context_pushed = False


@contextlib.contextmanager
def _active_cuda_context(context):
    """Temporarily make a retained MPPI context current on this thread."""
    context.push()
    try:
        yield
    finally:
        context.pop()


def _cleanup_context_atexit():
    """
    Registered with atexit. Releases the CUDA context reference owned by this
    module, if this module retained or created one at import time.
    """
    global _MODULE_CONTEXT, _owns_module_context_ref, _module_context_pushed
    if _owns_module_context_ref and _MODULE_CONTEXT is not None:
        try:
            if _module_context_pushed:
                _MODULE_CONTEXT.pop()
                _module_context_pushed = False

            _MODULE_CONTEXT.detach()  # Release/destroy the retained context

            _MODULE_CONTEXT = None
            _owns_module_context_ref = False
        except cuda.Error:
            # Suppress errors during atexit cleanup (e.g., if context was already destroyed)
            pass
        except Exception:
            # Suppress any other unexpected errors during atexit cleanup
            pass


atexit.register(_cleanup_context_atexit)
# --- End CUDA Context Management ---


BLOCK_SIZE = 32
LARGE_COLLISION_COST = np.float32(1.0e7)


def _actor_collision_geometry_from_extent(extent, buffer=0.0):
    extent_arr = np.asarray(extent, dtype=np.float32).reshape(-1)
    if extent_arr.size <= 0:
        return np.zeros((3,), dtype=np.float32)

    half_length = float(abs(extent_arr[0]))
    half_width = float(abs(extent_arr[1])) if extent_arr.size > 1 else half_length
    margin = max(0.0, float(buffer))
    half_length += margin
    half_width += margin

    radius = max(min(half_length, half_width), 1.0e-3)
    if half_length >= half_width:
        offset_x = max(half_length - radius, 0.0)
        offset_y = 0.0
    else:
        offset_x = 0.0
        offset_y = max(half_width - radius, 0.0)

    return np.asarray([radius, offset_x, offset_y], dtype=np.float32)


def _legacy_extent_from_collision_geometry(collision_geometry):
    radius, offset_x, offset_y = np.asarray(
        collision_geometry, dtype=np.float32
    ).reshape(3)
    return np.asarray(
        [radius + abs(offset_x), radius + abs(offset_y)], dtype=np.float32
    )


def _obstacle_polygon_points(obstacle):
    if isinstance(obstacle, dict):
        for key in ("polygon", "points", "vertices"):
            if key in obstacle:
                return obstacle[key]
        return None
    if hasattr(obstacle, "points"):
        return obstacle.points
    return None


def _pack_polygon_obstacles(polygons):
    if not polygons:
        return np.zeros((0,), dtype=np.float32), np.zeros((1,), dtype=np.int32)

    vertices = []
    offsets = [0]
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
            continue
        points = np.ascontiguousarray(points[:, :2], dtype=np.float32)
        if points.shape[0] > 1 and np.allclose(points[0], points[-1]):
            points = points[:-1]
        if points.shape[0] < 3:
            continue
        vertices.append(points)
        offsets.append(offsets[-1] + int(points.shape[0]))

    if not vertices:
        return np.zeros((0,), dtype=np.float32), np.zeros((1,), dtype=np.int32)

    return (
        np.ascontiguousarray(
            np.concatenate(vertices, axis=0).reshape(-1), dtype=np.float32
        ),
        np.ascontiguousarray(offsets, dtype=np.int32),
    )


def _prepare_obstacle_batches(obstacles, horizon):
    static_states = []
    static_extents = []
    dynamic_actors = []
    static_polygons = []

    for obstacle in obstacles if obstacles is not None else []:
        polygon_points = _obstacle_polygon_points(obstacle)
        if polygon_points is not None:
            blocking = (
                obstacle.get("blocking", True)
                if isinstance(obstacle, dict)
                else getattr(obstacle, "blocking", True)
            )
            if bool(blocking):
                static_polygons.append(polygon_points)
            continue

        if isinstance(obstacle, dict):
            states = np.asarray(obstacle.get("states", []), dtype=np.float32)
            extent = obstacle.get("extent", (0.0, 0.0))
            collision_buffer = obstacle.get("collision_buffer", 0.0)
            is_static = bool(obstacle.get("static", False))
            obstacle_id = obstacle.get("id")
        else:
            obstacle_arr = np.asarray(obstacle, dtype=np.float32).reshape(-1)
            if obstacle_arr.size >= 9:
                x, y, theta, radius, offset_x, offset_y = obstacle_arr[:6]
                collision_geometry = np.asarray(
                    [radius, offset_x, offset_y], dtype=np.float32
                )
                static_states.append(np.asarray([x, y, 0.0, theta], dtype=np.float32))
                static_extents.append(
                    _legacy_extent_from_collision_geometry(collision_geometry)
                )
            elif obstacle_arr.size >= 6:
                x, y, radius, _min_x, _min_y, _distance = obstacle_arr[:6]
                static_states.append(np.asarray([x, y, 0.0, 0.0], dtype=np.float32))
                static_extents.append(np.asarray([radius, radius], dtype=np.float32))
            continue

        if states.ndim == 1:
            states = states.reshape(1, -1)
        if states.ndim != 2 or states.shape[0] <= 0 or states.shape[1] < 2:
            continue

        collision_geometry = _actor_collision_geometry_from_extent(
            extent, buffer=collision_buffer
        )
        if is_static or states.shape[0] <= 1:
            state = np.zeros((4,), dtype=np.float32)
            state_cols = min(4, states.shape[1])
            state[:state_cols] = states[0, :state_cols]
            static_states.append(state)
            static_extents.append(
                _legacy_extent_from_collision_geometry(collision_geometry)
            )
            continue

        padded = np.zeros((horizon, 4), dtype=np.float32)
        valid_steps = min(horizon, states.shape[0])
        state_cols = min(4, states.shape[1])
        padded[:valid_steps, :state_cols] = states[:valid_steps, :state_cols]
        if valid_steps < horizon:
            padded[valid_steps:, :] = padded[valid_steps - 1, :]
        dynamic_actors.append(
            {
                "id": obstacle_id,
                "states": padded,
                "collision_geometry": np.asarray(collision_geometry, dtype=np.float32),
                "extent": np.asarray(extent, dtype=np.float32).reshape(-1),
                "collision_buffer": float(collision_buffer),
            }
        )

    static_states_host = (
        np.ascontiguousarray(np.stack(static_states, axis=0).astype(np.float32))
        if static_states
        else np.zeros((0, 4), dtype=np.float32)
    )
    static_extents_host = (
        np.ascontiguousarray(np.stack(static_extents, axis=0).astype(np.float32))
        if static_extents
        else np.zeros((0, 2), dtype=np.float32)
    )
    polygon_vertices_host, polygon_offsets_host = _pack_polygon_obstacles(
        static_polygons
    )
    return (
        static_extents_host,
        static_states_host,
        dynamic_actors,
        polygon_vertices_host,
        polygon_offsets_host,
    )


def _pack_dynamic_actor_arrays(dynamic_actors):
    if not dynamic_actors:
        return np.zeros((0, 0, 4), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    actor_states = np.ascontiguousarray(
        np.stack(
            [np.asarray(actor["states"], dtype=np.float32) for actor in dynamic_actors],
            axis=0,
        ).astype(np.float32, copy=False)
    )
    actor_collision_geometry = np.ascontiguousarray(
        np.stack(
            [
                np.asarray(actor["collision_geometry"], dtype=np.float32)
                for actor in dynamic_actors
            ],
            axis=0,
        ).astype(np.float32, copy=False)
    )
    return actor_states, actor_collision_geometry


def _prepare_occupancy_grid_stack(occupancy_grids):
    if occupancy_grids is None or occupancy_grids is False:
        return None
    if isinstance(occupancy_grids, dict):
        grids = occupancy_grids.get(
            "probability_grids",
            occupancy_grids.get("grids", occupancy_grids.get("occupancy")),
        )
        origin = occupancy_grids.get("origin", (0.0, 0.0))
        resolution = occupancy_grids.get("resolution", 1.0)
        hard_threshold = occupancy_grids.get("planning_hard_threshold", 0.65)
        soft_weight = occupancy_grids.get("mppi_occupancy_weight")
    else:
        grids = getattr(occupancy_grids, "probability_grids", None)
        origin = getattr(occupancy_grids, "origin", (0.0, 0.0))
        resolution = getattr(occupancy_grids, "resolution", 1.0)
        hard_threshold = getattr(occupancy_grids, "planning_hard_threshold", 0.65)
        soft_weight = getattr(occupancy_grids, "mppi_occupancy_weight", None)

    if grids is None:
        return None
    grids = np.asarray(grids, dtype=np.float32)
    if grids.ndim == 2:
        grids = grids[np.newaxis, ...]
    if grids.ndim != 3 or grids.shape[0] <= 0:
        raise ValueError(
            "occupancy probability grids must have shape (steps, rows, cols)"
        )
    grids = np.ascontiguousarray(np.clip(grids, 0.0, 1.0), dtype=np.float32)
    return {
        "grids": grids,
        "origin": (float(origin[0]), float(origin[1])),
        "resolution": float(resolution),
        "hard_threshold": float(hard_threshold),
        "soft_weight": None if soft_weight is None else float(soft_weight),
    }


def _prepare_rollout_obstacle_batches(obstacles, horizon, occupancy_payload=None):
    if occupancy_payload is not None:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            [],
            np.zeros((0,), dtype=np.float32),
            np.zeros((1,), dtype=np.int32),
        )
    return _prepare_obstacle_batches(obstacles, horizon)


def occupancy_grid_cost_for_state(
    state,
    probability_grids,
    *,
    origin,
    resolution,
    step=0,
    ego_collision_geometry=(0.0, 0.0, 0.0),
    soft_weight=1.0,
    collision_cost=float(LARGE_COLLISION_COST),
    hard_threshold=0.65,
):
    grids = np.asarray(probability_grids, dtype=np.float32)
    if grids.ndim == 2:
        grids = grids[np.newaxis, ...]
    step = int(np.clip(step, 0, grids.shape[0] - 1))
    grid = grids[step]
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    radius, offset_x, offset_y = np.asarray(
        ego_collision_geometry, dtype=np.float32
    ).reshape(3)
    centers = _circle_centers_from_pose(
        state[0],
        state[1],
        state[3] if state.size > 3 else 0.0,
        offset_x,
        offset_y,
    )
    max_probability = 0.0
    rows, cols = grid.shape
    for center in centers:
        col = int(np.floor((float(center[0]) - float(origin[0])) / float(resolution)))
        row = int(np.floor((float(center[1]) - float(origin[1])) / float(resolution)))
        if row < 0 or row >= rows or col < 0 or col >= cols:
            return float(collision_cost)
        max_probability = max(max_probability, float(grid[row, col]))
    cost = float(soft_weight) * max_probability
    if max_probability >= float(hard_threshold):
        cost += float(collision_cost)
    return cost


def _resolve_oce_payload(oce_data):
    if oce_data is None or oce_data is False:
        return None, None, "kde", 1.0e-9, 1.0, False, False, None
    if isinstance(oce_data, dict):
        if not bool(oce_data.get("enabled", True)):
            return None, None, "kde", 1.0e-9, 1.0, False, False, None
        scene = oce_data.get("scene")
        config = oce_data.get("oce_config", oce_data.get("config"))
        scorer = oce_data.get("scorer", oce_data.get("score_func"))
        entropy_space = oce_data.get(
            "entropy_space",
            (
                getattr(config, "oce_entropy_space", "kde")
                if config is not None
                else "kde"
            ),
        )
        eps = float(oce_data.get("eps", getattr(config, "eps", 1.0e-9)))
        discount = float(oce_data.get("discount", getattr(config, "oce_discount", 1.0)))
        materialize_host = bool(oce_data.get("materialize_host", False))
        return_visibility_tensor = bool(oce_data.get("return_visibility_tensor", False))
        return (
            scene,
            config,
            entropy_space,
            eps,
            discount,
            materialize_host,
            return_visibility_tensor,
            scorer,
        )
    return oce_data, None, "kde", 1.0e-9, 1.0, False, False, None


def _resolve_discrete_oce_payload(oce_data):
    if not isinstance(oce_data, dict) or not bool(oce_data.get("enabled", True)):
        return None
    oce_type = str(
        oce_data.get("type", oce_data.get("kind", oce_data.get("mode", "")))
    ).lower()
    if oce_type == "discrete" or "tracker" in oce_data:
        return oce_data
    return None


def _compact_csr_state_space(
    *,
    state_centers,
    beliefs,
    prefix_beliefs,
    transition_data,
    transition_indices,
    transition_indptr,
    max_states=None,
    probability_floor=0.0,
):
    """Reduce discrete OCE to high-probability states before rollout scoring.

    The full transition grid can be thousands of states.  MPPI rollout scoring
    only needs a relative OCE cost, so we keep the states that are plausible
    over the evaluated horizon and renormalize the induced transition rows.
    """
    max_states = None if max_states is None else int(max_states)
    probability_floor = float(probability_floor or 0.0)
    num_states = int(np.asarray(state_centers).shape[0])
    if max_states is not None and max_states <= 0:
        max_states = None
    if (max_states is None or max_states >= num_states) and probability_floor <= 0.0:
        return {
            "state_centers": state_centers,
            "beliefs": beliefs,
            "prefix_beliefs": prefix_beliefs,
            "transition_data": transition_data,
            "transition_indices": transition_indices,
            "transition_indptr": transition_indptr,
            "state_indices": np.arange(num_states, dtype=np.int32),
            "reduced": False,
        }

    prefix = np.asarray(prefix_beliefs, dtype=np.float32)
    scores = np.max(prefix, axis=(0, 1))
    if probability_floor > 0.0:
        selected = np.flatnonzero(scores >= probability_floor)
    else:
        selected = np.arange(num_states, dtype=np.int64)
    if selected.size == 0:
        selected = np.asarray([int(np.argmax(scores))], dtype=np.int64)
    if max_states is not None and selected.size > max_states:
        selected = np.argsort(scores)[-max_states:]
    selected = np.asarray(np.unique(selected), dtype=np.int64)
    selected.sort()

    old_to_new = np.full((num_states,), -1, dtype=np.int32)
    old_to_new[selected] = np.arange(selected.size, dtype=np.int32)
    num_agents = int(np.asarray(beliefs).shape[0])

    data_chunks = []
    index_chunks = []
    new_indptr = np.empty((num_agents, selected.size + 1), dtype=np.int32)
    offset = 0
    transition_data = np.asarray(transition_data, dtype=np.float32)
    transition_indices = np.asarray(transition_indices, dtype=np.int32)
    transition_indptr = np.asarray(transition_indptr, dtype=np.int32)

    for agent_idx in range(num_agents):
        new_indptr[agent_idx, 0] = offset
        for new_row, old_row in enumerate(selected):
            start = int(transition_indptr[agent_idx, old_row])
            end = int(transition_indptr[agent_idx, old_row + 1])
            row_cols = transition_indices[start:end]
            row_data = transition_data[start:end]
            keep_mask = old_to_new[row_cols] >= 0
            cols = old_to_new[row_cols[keep_mask]].astype(np.int32, copy=False)
            vals = row_data[keep_mask].astype(np.float32, copy=True)
            row_sum = float(vals.sum())
            if row_sum > 1.0e-12:
                vals /= row_sum
            else:
                cols = np.asarray([new_row], dtype=np.int32)
                vals = np.asarray([1.0], dtype=np.float32)
            data_chunks.append(vals)
            index_chunks.append(cols)
            offset += int(vals.size)
            new_indptr[agent_idx, new_row + 1] = offset

    if data_chunks:
        new_data = np.concatenate(data_chunks).astype(np.float32, copy=False)
        new_indices = np.concatenate(index_chunks).astype(np.int32, copy=False)
    else:
        new_data = np.zeros((0,), dtype=np.float32)
        new_indices = np.zeros((0,), dtype=np.int32)

    new_beliefs = np.ascontiguousarray(
        np.asarray(beliefs)[:, selected], dtype=np.float32
    )
    belief_sums = new_beliefs.sum(axis=1, keepdims=True)
    np.divide(new_beliefs, belief_sums, out=new_beliefs, where=belief_sums > 1.0e-12)

    new_prefix = np.ascontiguousarray(prefix[:, :, selected], dtype=np.float32)
    prefix_sums = new_prefix.sum(axis=2, keepdims=True)
    np.divide(new_prefix, prefix_sums, out=new_prefix, where=prefix_sums > 1.0e-12)

    return {
        "state_centers": np.ascontiguousarray(
            np.asarray(state_centers, dtype=np.float32)[selected]
        ),
        "beliefs": new_beliefs,
        "prefix_beliefs": new_prefix,
        "transition_data": new_data,
        "transition_indices": new_indices,
        "transition_indptr": new_indptr,
        "state_indices": selected.astype(np.int32, copy=False),
        "reduced": selected.size != num_states,
    }


def _prepare_discrete_oce_rollout_inputs(oce_payload, horizon):
    tracker = oce_payload.get("tracker")
    if tracker is None:
        raise ValueError("discrete oce_data requires a tracker")
    if not getattr(tracker, "agent_hmms", None):
        return None

    eval_horizon = max(1, int(oce_payload.get("horizon", horizon)))
    agent_ids, beliefs = tracker.agent_belief_matrix()
    if len(agent_ids) == 0:
        return None

    (
        transition_data,
        transition_indices,
        transition_indptr,
        prefix_beliefs,
    ) = tracker.agent_mixed_transition_csr(agent_ids, eval_horizon)
    state_centers = np.asarray(tracker.state_centers_sim, dtype=np.float32)
    max_states = oce_payload.get("max_states", oce_payload.get("state_limit", 384))
    if max_states is not None and int(max_states) <= 0:
        max_states = None
        probability_floor = 0.0
    else:
        probability_floor = float(oce_payload.get("state_probability_floor", 1.0e-4))
    compacted = _compact_csr_state_space(
        state_centers=state_centers,
        beliefs=np.asarray(beliefs, dtype=np.float32),
        prefix_beliefs=np.asarray(prefix_beliefs, dtype=np.float32),
        transition_data=np.asarray(transition_data, dtype=np.float32),
        transition_indices=np.asarray(transition_indices, dtype=np.int32),
        transition_indptr=np.asarray(transition_indptr, dtype=np.int32),
        max_states=max_states,
        probability_floor=probability_floor,
    )

    occupancy_horizon = oce_payload.get("occupancy_horizon")
    agent_owner_bits = np.zeros((len(agent_ids),), dtype=np.uint64)
    if occupancy_horizon is not None:
        for idx, agent_id in enumerate(agent_ids):
            bit_index = getattr(occupancy_horizon, "agent_bit_indices", {}).get(
                agent_id
            )
            if bit_index is not None:
                agent_owner_bits[idx] = np.uint64(1) << np.uint64(bit_index)
        static_grid = np.zeros(
            occupancy_horizon.probability_grids.shape[1:],
            dtype=np.uint8,
        )
        grid_origin = occupancy_horizon.origin
        grid_resolution = occupancy_horizon.resolution
        occupancy_probability_grids = occupancy_horizon.probability_grids
        occupancy_owner_mask_grids = occupancy_horizon.owner_mask_grids
        occupancy_threshold = getattr(occupancy_horizon, "visibility_threshold", 0.25)
    else:
        static_grid = tracker.static_occupancy_grid
        grid_origin = (
            float(tracker.bounds["min_x"]),
            float(tracker.bounds["min_y"]),
        )
        grid_resolution = tracker.cell_size
        occupancy_probability_grids = None
        occupancy_owner_mask_grids = None
        occupancy_threshold = float(oce_payload.get("occupancy_threshold", 0.25))

    return {
        "agent_ids": agent_ids,
        "beliefs": compacted["beliefs"],
        "state_centers": compacted["state_centers"],
        "static_grid": np.asarray(static_grid, dtype=np.uint8),
        "grid_origin": (float(grid_origin[0]), float(grid_origin[1])),
        "grid_resolution": float(grid_resolution),
        "transition_data": compacted["transition_data"],
        "transition_indices": compacted["transition_indices"],
        "transition_indptr": compacted["transition_indptr"],
        "prefix_beliefs": compacted["prefix_beliefs"],
        "horizon": eval_horizon,
        "scan_range": float(oce_payload.get("scan_range", np.inf)),
        "return_visibility": bool(oce_payload.get("return_visibility_tensor", False)),
        "occupancy_probability_grids": occupancy_probability_grids,
        "occupancy_owner_mask_grids": occupancy_owner_mask_grids,
        "agent_owner_bits": agent_owner_bits,
        "occupancy_threshold": float(occupancy_threshold),
        "state_indices": compacted["state_indices"],
        "state_reduction": {
            "enabled": bool(compacted["reduced"]),
            "original_states": int(state_centers.shape[0]),
            "selected_states": int(compacted["state_centers"].shape[0]),
            "max_states": None if max_states is None else int(max_states),
            "probability_floor": float(probability_floor),
        },
    }


def _circle_centers_from_pose(x, y, theta, offset_x, offset_y):
    local = np.asarray(
        [[-offset_x, -offset_y], [0.0, 0.0], [offset_x, offset_y]],
        dtype=np.float32,
    )
    c = np.cos(theta)
    s = np.sin(theta)
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    return local @ rotation.T + np.asarray([x, y], dtype=np.float32)


def _dynamic_rollout_clearance_summary(
    rollout_states,
    actor_states,
    actor_collision_geometry,
    ego_collision_geometry,
):
    if rollout_states is None or actor_states is None or actor_states.size == 0:
        return {"num_dynamic_actors": 0, "colliding_samples": 0, "min_clearance": None}

    rollout_states = np.asarray(rollout_states, dtype=np.float32)
    actor_states = np.asarray(actor_states, dtype=np.float32)
    actor_collision_geometry = np.asarray(actor_collision_geometry, dtype=np.float32)
    ego_radius, ego_offset_x, ego_offset_y = np.asarray(
        ego_collision_geometry, dtype=np.float32
    ).reshape(3)

    sample_min = np.full(rollout_states.shape[0], np.inf, dtype=np.float32)
    for sample_idx in range(rollout_states.shape[0]):
        for step_idx in range(rollout_states.shape[1]):
            ego = rollout_states[sample_idx, step_idx]
            ego_centers = _circle_centers_from_pose(
                ego[0], ego[1], ego[3], ego_offset_x, ego_offset_y
            )
            for actor_idx in range(actor_states.shape[0]):
                actor_radius, actor_offset_x, actor_offset_y = actor_collision_geometry[
                    actor_idx
                ]
                actor = actor_states[actor_idx, step_idx]
                actor_centers = _circle_centers_from_pose(
                    actor[0], actor[1], actor[3], actor_offset_x, actor_offset_y
                )
                distances = np.linalg.norm(
                    ego_centers[:, np.newaxis, :] - actor_centers[np.newaxis, :, :],
                    axis=2,
                )
                clearance = np.min(distances) - (ego_radius + actor_radius)
                if clearance < sample_min[sample_idx]:
                    sample_min[sample_idx] = clearance

    finite = np.isfinite(sample_min)
    if not np.any(finite):
        return {
            "num_dynamic_actors": int(actor_states.shape[0]),
            "colliding_samples": 0,
            "min_clearance": None,
        }

    return {
        "num_dynamic_actors": int(actor_states.shape[0]),
        "colliding_samples": int(np.count_nonzero(sample_min[finite] < 0.0)),
        "min_clearance": float(np.min(sample_min[finite])),
        "mean_min_clearance": float(np.mean(sample_min[finite])),
    }


def _summarize_dynamic_actors(dynamic_actors, ego_collision_geometry):
    ego_radius, ego_offset_x, ego_offset_y = np.asarray(
        ego_collision_geometry, dtype=np.float32
    ).reshape(3)
    summary = []
    for idx, actor in enumerate(dynamic_actors):
        states = np.asarray(actor.get("states", []), dtype=np.float32)
        geometry = np.asarray(
            actor.get("collision_geometry", np.zeros(3, dtype=np.float32)),
            dtype=np.float32,
        ).reshape(3)
        item = {
            "index": int(idx),
            "id": actor.get("id"),
            "steps": int(states.shape[0]) if states.ndim >= 2 else 0,
            "radius": float(geometry[0]),
            "offset_x": float(geometry[1]),
            "offset_y": float(geometry[2]),
            "ego_radius": float(ego_radius),
            "ego_offset_x": float(ego_offset_x),
            "ego_offset_y": float(ego_offset_y),
            "combined_circle_radius": float(geometry[0] + ego_radius),
            "extent": (
                np.asarray(actor.get("extent"), dtype=float).reshape(-1).tolist()
                if actor.get("extent") is not None
                else None
            ),
            "collision_buffer": float(actor.get("collision_buffer", 0.0)),
        }
        if states.ndim == 2 and states.shape[0] > 0:
            item["start"] = states[0, : min(4, states.shape[1])].astype(float).tolist()
            item["end"] = states[-1, : min(4, states.shape[1])].astype(float).tolist()
        summary.append(item)
    return summary


def _summarize_weights_and_costs(
    weights,
    total_costs,
    dynamic_costs,
    samples,
    dynamic_collision_cost,
):
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    total_costs = np.asarray(total_costs, dtype=np.float32).reshape(-1)
    dynamic_costs = np.asarray(dynamic_costs, dtype=np.float32).reshape(-1)
    weight_sum = float(np.sum(weights))
    weight_sq_sum = float(np.sum(weights * weights))
    ess = (weight_sum * weight_sum) / (weight_sq_sum + 1.0e-12)
    mean_weight = float(np.mean(weights)) if weights.size else 0.0
    dynamic_collision_threshold = 0.5 * float(dynamic_collision_cost)
    dynamic_nonzero = int(np.count_nonzero(dynamic_costs > 0.0))
    dynamic_hard = int(np.count_nonzero(dynamic_costs >= dynamic_collision_threshold))

    return {
        "weights": {
            "sum": weight_sum,
            "min": float(np.min(weights)) if weights.size else 0.0,
            "max": float(np.max(weights)) if weights.size else 0.0,
            "mean": mean_weight,
            "std": float(np.std(weights)) if weights.size else 0.0,
            "nonzero": int(np.count_nonzero(weights > 0.0)),
            "ess": float(ess),
            "ess_pct": float(ess / max(float(samples), 1.0) * 100.0),
            "nearly_equal": bool(
                weights.size > 0
                and np.allclose(weights, mean_weight, rtol=1.0e-3, atol=1.0e-8)
            ),
        },
        "total_costs": {
            "min": float(np.min(total_costs)) if total_costs.size else 0.0,
            "max": float(np.max(total_costs)) if total_costs.size else 0.0,
            "mean": float(np.mean(total_costs)) if total_costs.size else 0.0,
            "std": float(np.std(total_costs)) if total_costs.size else 0.0,
            "range": (
                float(np.max(total_costs) - np.min(total_costs))
                if total_costs.size
                else 0.0
            ),
        },
        "dynamic_costs": {
            "min": float(np.min(dynamic_costs)) if dynamic_costs.size else 0.0,
            "max": float(np.max(dynamic_costs)) if dynamic_costs.size else 0.0,
            "mean": float(np.mean(dynamic_costs)) if dynamic_costs.size else 0.0,
            "std": float(np.std(dynamic_costs)) if dynamic_costs.size else 0.0,
            "nonzero": dynamic_nonzero,
            "hard_collision_like": dynamic_hard,
            "all_zero": bool(dynamic_costs.size > 0 and dynamic_nonzero == 0),
            "all_hard_collision_like": bool(
                dynamic_costs.size > 0 and dynamic_hard == dynamic_costs.size
            ),
        },
    }


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
        int obstacle_steps;
        int num_polygon_obstacles;
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
        float vehicle_width;
        float steering_rate_weight; // added optional penalty weight
        float static_collision_cost;
        float static_hard_clearance_margin;
        float static_clearance_margin;
        float static_clearance_weight;
        int dynamics_type; // 0=Ackermann, 1=holonomic
        float holonomic_max_heading_change;
        float holonomic_max_curvature;
    };

    struct Obstacle {
        float dx;
        float dy;
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

    __device__
    inline float clamp_unit(float value) {
        const float eps = 1e-5f;
        if (value > 1.0f - eps) {
            return 1.0f - eps;
        }
        if (value < -1.0f + eps) {
            return -1.0f + eps;
        }
        return value;
    }

    __device__
    inline float unsquash_control(float value, float limit) {
        if (limit <= FLT_EPSILON) {
            return 0.0f;
        }
        float unit_value = clamp_unit(value / limit);
        return 0.5f * logf((1.0f + unit_value) / (1.0f - unit_value));
    }

    __device__
    inline float squash_control(float z_value, float limit) {
        if (limit <= FLT_EPSILON) {
            return 0.0f;
        }
        return limit * tanhf(z_value);
    }

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
    float obstacle_cost(
        const Obstacle *obstacles,
        const State *obstacle_states,
        int num_obstacles,
        int obstacle_steps,
        int step_idx, // current timestep index
        float px, float py, float p_theta, // ego vehicle pose
        float vehicle_length, float vehicle_width // ego vehicle dimensions
    ) {
      if (num_obstacles == 0 || obstacle_steps <= 0) {
        return 0.0f;
      }

      int clamped_step = step_idx;
      if (clamped_step < 0) {
        clamped_step = 0;
      }
      if (clamped_step >= obstacle_steps) {
        clamped_step = obstacle_steps - 1;
      }

      // Ego vehicle three-circle model
      float ego_radius = vehicle_width / 2.0f;
      float ego_offset = fmaxf(vehicle_length - vehicle_width, 0.0f) / 2.0f;
      float ego_cos_theta = cosf(p_theta);
      float ego_sin_theta = sinf(p_theta);
      const int ego_circles = (ego_offset <= 0.0f) ? 1 : 3;

      for (int i = 0; i < num_obstacles; i++) {
        const State obstacle_state = obstacle_states[i * obstacle_steps + clamped_step];
        const float obs_extent_x = obstacles[i].dx;
        const float obs_extent_y = obstacles[i].dy;
        const float obs_radius = fminf(obs_extent_x, obs_extent_y);
        const float obs_offset = fmaxf(obs_extent_x - obs_radius, 0.0f);

        const float obs_cos_theta = cosf(obstacle_state.theta);
        const float obs_sin_theta = sinf(obstacle_state.theta);
        const int obs_circles = (obs_offset <= 0.0f) ? 1 : 3;

        // 3x3 circle checks
        for (int ego_circle_idx = 0; ego_circle_idx < ego_circles; ++ego_circle_idx) {
          float ego_dist_along = 0.0f;
          if (ego_circle_idx == 1) ego_dist_along = ego_offset;
          else if (ego_circle_idx == 2) ego_dist_along = -ego_offset;

          const float ego_cx = px + ego_dist_along * ego_cos_theta;
          const float ego_cy = py + ego_dist_along * ego_sin_theta;

          for (int obs_circle_idx = 0; obs_circle_idx < obs_circles; ++obs_circle_idx) {
            float obs_dist_along = 0.0f;
            if (obs_circle_idx == 1) obs_dist_along = obs_offset;
            else if (obs_circle_idx == 2) obs_dist_along = -obs_offset;

            const float obs_cx = obstacle_state.x + obs_dist_along * obs_cos_theta;
            const float obs_cy = obstacle_state.y + obs_dist_along * obs_sin_theta;

            const float dx = obs_cx - ego_cx;
            const float dy = obs_cy - ego_cy;
            const float min_dist = obs_radius + ego_radius;

            if ((dx * dx + dy * dy) < (min_dist * min_dist)) {
              return 10000000.0f;
            }
          }
        }
      }

      return 0.0f;
    }

    __device__ __forceinline__
    float point_segment_distance_sq(
        float px,
        float py,
        float ax,
        float ay,
        float bx,
        float by
    ) {
        const float abx = bx - ax;
        const float aby = by - ay;
        const float apx = px - ax;
        const float apy = py - ay;
        const float ab_len_sq = abx * abx + aby * aby;
        float t = 0.0f;
        if (ab_len_sq > FLT_EPSILON) {
            t = fminf(fmaxf((apx * abx + apy * aby) / ab_len_sq, 0.0f), 1.0f);
        }
        const float cx = ax + t * abx;
        const float cy = ay + t * aby;
        const float dx = px - cx;
        const float dy = py - cy;
        return dx * dx + dy * dy;
    }

    __device__ __forceinline__
    bool point_in_polygon(
        float px,
        float py,
        const float *polygon_vertices,
        int start,
        int end
    ) {
        bool inside = false;
        int previous = end - 1;
        for (int current = start; current < end; ++current) {
            const float xi = polygon_vertices[current * 2 + 0];
            const float yi = polygon_vertices[current * 2 + 1];
            const float xj = polygon_vertices[previous * 2 + 0];
            const float yj = polygon_vertices[previous * 2 + 1];
            const bool crosses = ((yi > py) != (yj > py));
            if (crosses) {
                const float x_intersect = (xj - xi) * (py - yi) / (yj - yi + FLT_EPSILON) + xi;
                if (px < x_intersect) {
                    inside = !inside;
                }
            }
            previous = current;
        }
        return inside;
    }

    __device__ __forceinline__
    bool circle_intersects_polygon(
        float cx,
        float cy,
        float radius,
        const float *polygon_vertices,
        int start,
        int end
    ) {
        if (end - start < 3) {
            return false;
        }
        if (point_in_polygon(cx, cy, polygon_vertices, start, end)) {
            return true;
        }

        const float radius_sq = radius * radius;
        int previous = end - 1;
        for (int current = start; current < end; ++current) {
            const float ax = polygon_vertices[previous * 2 + 0];
            const float ay = polygon_vertices[previous * 2 + 1];
            const float bx = polygon_vertices[current * 2 + 0];
            const float by = polygon_vertices[current * 2 + 1];
            if (point_segment_distance_sq(cx, cy, ax, ay, bx, by) <= radius_sq) {
                return true;
            }
            previous = current;
        }
        return false;
    }

    __device__ __forceinline__
    float circle_polygon_clearance(
        float cx,
        float cy,
        float radius,
        const float *polygon_vertices,
        int start,
        int end
    ) {
        if (end - start < 3) {
            return FLT_MAX;
        }

        float min_dist_sq = FLT_MAX;
        int previous = end - 1;
        for (int current = start; current < end; ++current) {
            const float ax = polygon_vertices[previous * 2 + 0];
            const float ay = polygon_vertices[previous * 2 + 1];
            const float bx = polygon_vertices[current * 2 + 0];
            const float by = polygon_vertices[current * 2 + 1];
            min_dist_sq = fminf(
                min_dist_sq,
                point_segment_distance_sq(cx, cy, ax, ay, bx, by)
            );
            previous = current;
        }

        if (point_in_polygon(cx, cy, polygon_vertices, start, end)) {
            return -radius;
        }
        return sqrtf(min_dist_sq) - radius;
    }

    __device__
    float polygon_obstacle_cost(
        const float *polygon_vertices,
        const int *polygon_offsets,
        int num_polygon_obstacles,
        float px,
        float py,
        float p_theta,
        float vehicle_length,
        float vehicle_width,
        float collision_penalty,
        float hard_clearance_margin,
        float clearance_margin,
        float clearance_weight
    ) {
        if (num_polygon_obstacles <= 0 || polygon_vertices == NULL || polygon_offsets == NULL) {
            return 0.0f;
        }

        float cost = 0.0f;
        const float ego_radius = vehicle_width / 2.0f;
        const float ego_offset = fmaxf(vehicle_length - vehicle_width, 0.0f) / 2.0f;
        const float ego_cos_theta = cosf(p_theta);
        const float ego_sin_theta = sinf(p_theta);
        const int ego_circles = (ego_offset <= 0.0f) ? 1 : 3;

        for (int polygon_idx = 0; polygon_idx < num_polygon_obstacles; ++polygon_idx) {
            const int start = polygon_offsets[polygon_idx];
            const int end = polygon_offsets[polygon_idx + 1];
            for (int ego_circle_idx = 0; ego_circle_idx < ego_circles; ++ego_circle_idx) {
                float ego_dist_along = 0.0f;
                if (ego_circle_idx == 1) ego_dist_along = ego_offset;
                else if (ego_circle_idx == 2) ego_dist_along = -ego_offset;

                const float ego_cx = px + ego_dist_along * ego_cos_theta;
                const float ego_cy = py + ego_dist_along * ego_sin_theta;
                float clearance = circle_polygon_clearance(
                    ego_cx,
                    ego_cy,
                    ego_radius,
                    polygon_vertices,
                    start,
                    end
                );
                if (clearance < 0.0f) {
                    return collision_penalty;
                }
                const float active_clearance_margin = fmaxf(clearance_margin, hard_clearance_margin);
                if (active_clearance_margin > 0.0f && clearance_weight > 0.0f && clearance < active_clearance_margin) {
                    float deficit = active_clearance_margin - clearance;
                    cost += clearance_weight * deficit * deficit;
                }
            }
        }
        return cost;
    }

    __device__ __forceinline__
    void circle_center_from_pose(
        float x,
        float y,
        float theta,
        float offset_x,
        float offset_y,
        int circle_idx,
        float *cx,
        float *cy
    ) {
        float local_x = 0.0f;
        float local_y = 0.0f;
        if (circle_idx == 0) {
            local_x = -offset_x;
            local_y = -offset_y;
        } else if (circle_idx == 2) {
            local_x = offset_x;
            local_y = offset_y;
        }

        const float cos_theta = cosf(theta);
        const float sin_theta = sinf(theta);
        *cx = x + local_x * cos_theta - local_y * sin_theta;
        *cy = y + local_x * sin_theta + local_y * cos_theta;
    }

    __device__ __forceinline__
    bool three_circle_collision(
        float ax,
        float ay,
        float atheta,
        float aradius,
        float aoffset_x,
        float aoffset_y,
        float bx,
        float by,
        float btheta,
        float bradius,
        float boffset_x,
        float boffset_y
    ) {
        float min_dist = aradius + bradius;
        float min_dist_sq = min_dist * min_dist;

        float a_cx[3];
        float a_cy[3];
        float b_cx[3];
        float b_cy[3];

        for (int i = 0; i < 3; ++i) {
            circle_center_from_pose(ax, ay, atheta, aoffset_x, aoffset_y, i, &a_cx[i], &a_cy[i]);
            circle_center_from_pose(bx, by, btheta, boffset_x, boffset_y, i, &b_cx[i], &b_cy[i]);
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                float dx = a_cx[i] - b_cx[j];
                float dy = a_cy[i] - b_cy[j];
                if ((dx * dx + dy * dy) < min_dist_sq) {
                    return true;
                }
            }
        }

        return false;
    }

    __device__ __forceinline__
    float three_circle_min_clearance(
        float ax,
        float ay,
        float atheta,
        float aradius,
        float aoffset_x,
        float aoffset_y,
        float bx,
        float by,
        float btheta,
        float bradius,
        float boffset_x,
        float boffset_y
    ) {
        float min_clearance = FLT_MAX;
        float min_dist = aradius + bradius;

        float a_cx[3];
        float a_cy[3];
        float b_cx[3];
        float b_cy[3];

        for (int i = 0; i < 3; ++i) {
            circle_center_from_pose(ax, ay, atheta, aoffset_x, aoffset_y, i, &a_cx[i], &a_cy[i]);
            circle_center_from_pose(bx, by, btheta, boffset_x, boffset_y, i, &b_cx[i], &b_cy[i]);
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                float dx = a_cx[i] - b_cx[j];
                float dy = a_cy[i] - b_cy[j];
                float clearance = sqrtf(dx * dx + dy * dy) - min_dist;
                if (clearance < min_clearance) {
                    min_clearance = clearance;
                }
            }
        }

        return min_clearance;
    }


    __device__
    float higgins_cost(
        const float M,
        const Obstacle *obstacles,
        const State *obstacle_states,
        int num_obstacles,
        int obstacle_steps,
        int step_idx,
        float px,
        float py,
        float scan_range
    ) {

      float cost = 0.0f;

      if (num_obstacles == 0 || obstacle_steps <= 0) {
        return cost;
      }

      int clamped_step = step_idx;
      if (clamped_step < 0) {
        clamped_step = 0;
      }
      if (clamped_step >= obstacle_steps) {
        clamped_step = obstacle_steps - 1;
      }

      const float r_fov = scan_range;
      const float r_fov_2 = r_fov * r_fov;

      for (int i = 0; i < num_obstacles; i++) {
        const State obstacle_state = obstacle_states[i * obstacle_steps + clamped_step];
        const float dx = obstacle_state.x - px;
        const float dy = obstacle_state.y - py;
        const float d_2 = dx * dx + dy * dy;
        const float d = sqrtf(d_2);

        const float radius = obstacles[i].dy;
        float inner = 0.0f;
        if (d > FLT_EPSILON) {
          inner = radius / d * (r_fov_2 - d_2);
        }
        const float inner_exp = expf(inner);
        float score;

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
    andersen_cost(
        const float M,
        const Obstacle *obstacles,
        const State *obstacle_states,
        int num_obstacles,
        int obstacle_steps,
        int step_idx,
        float px,
        float py,
        float vx,
        float vy
    ) {
      float cost = 0.0f;
      float v = sqrtf(vx * vx + vy * vy);

      if (num_obstacles == 0 || obstacle_steps <= 0 || v <= FLT_EPSILON) {
        return cost;
      }

      int clamped_step = step_idx;
      if (clamped_step < 0) {
        clamped_step = 0;
      }
      if (clamped_step >= obstacle_steps) {
        clamped_step = obstacle_steps - 1;
      }

      for (int i = 0; i < num_obstacles; i++) {
        const State obstacle_state = obstacle_states[i * obstacle_steps + clamped_step];
        float dx = obstacle_state.x - px;
        float dy = obstacle_state.y - py;

        auto dot = dx * vx + dy * vy;
        if (dot > 0.0f) {
          float d = sqrtf(dx * dx + dy * dy);
          if (d > FLT_EPSILON) {
            cost -= M * acosf(fminf(fmaxf(dot / (d * v), -1.0f), 1.0f));
          }
          break;
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

    inline __device__
    float wrap_angle(float angle) {
      return fmodf(angle + 3.0f * M_PI, 2.0f * M_PI) - M_PI;
    }

    __device__
    void holonomic_step(
            const State *state,
            const Control *control,
            float dt,
            float max_speed,
            float max_heading_change,
            float max_curvature,
            State *result
    ) {
      float vx = control->a;
      float vy = control->delta;
      float speed = hypotf(vx, vy);
      if (speed <= FLT_EPSILON || dt <= 0.0f) {
        *result = *state;
        result->v = 0.0f;
        return;
      }
      speed = fminf(speed, fmaxf(max_speed, 0.0f));
      float requested_heading = atan2f(vy, vx);
      float allowed_change = fminf(
          fmaxf(max_heading_change, 0.0f),
          fmaxf(max_curvature, 0.0f) * speed * dt
      );
      float heading_error = wrap_angle(requested_heading - state->theta);
      float accepted_change = fminf(fmaxf(heading_error, -allowed_change), allowed_change);
      float accepted_heading = wrap_angle(state->theta + accepted_change);
      result->x = state->x + speed * cosf(accepted_heading) * dt;
      result->y = state->y + speed * sinf(accepted_heading) * dt;
      result->v = speed;
      result->theta = accepted_heading;
    }


    __device__
    void generate_controls(
            curandState *globalState,
            int index,
            const Control *u_nom,
            const int num_controls,
            const float *u_limits,
            const float *u_dist_limits,
            int dynamics_type,
            Control *u_dist
    ) {
      curandState localState = globalState[index];
            for (int i = 0; i < num_controls; i++) {
                // Sample in unconstrained squashed-control coordinates:
                //   u = limit * tanh(z)
                // This avoids the boundary bias caused by clipping sampled controls
                // and then averaging the clipped disturbances in control space.
                float a_z_nom = unsquash_control(u_nom[i].a, u_limits[0]);
                float delta_z_nom = unsquash_control(u_nom[i].delta, u_limits[1]);
                float a_z_noise = curand_normal(&localState) * (u_dist_limits[0] / fmaxf(u_limits[0], FLT_EPSILON));
                float delta_z_noise = curand_normal(&localState) * (u_dist_limits[1] / fmaxf(u_limits[1], FLT_EPSILON));
                float a_candidate = squash_control(a_z_nom + a_z_noise, u_limits[0]);
                float delta_candidate = squash_control(delta_z_nom + delta_z_noise, u_limits[1]);
                if (dynamics_type == 1) {
                    float magnitude = hypotf(a_candidate, delta_candidate);
                    float max_speed = fmaxf(u_limits[0], 0.0f);
                    if (magnitude > max_speed && magnitude > FLT_EPSILON) {
                        float scale = max_speed / magnitude;
                        a_candidate *= scale;
                        delta_candidate *= scale;
                    }
                }
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
            const Obstacle *obstacles,
            const State *obstacle_states,
            const float *polygon_vertices,
            const int *polygon_offsets,
            const Optimization_Params *optimization_args,
            Control *u_dists,
            State *rollout_states,
            float *u_weights,
            float *component_costs
    ) {
        int start_sample_index = blockIdx.x * blockDim.x + threadIdx.x;
        int samples = optimization_args->samples;

        for (int sample_index = start_sample_index; sample_index < samples; sample_index += blockDim.x * gridDim.x) {

            int num_controls = optimization_args->num_controls;
            int num_obstacles = optimization_args->num_obstacles;
            int obstacle_steps = optimization_args->obstacle_steps;
            int num_polygon_obstacles = optimization_args->num_polygon_obstacles;
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
            float sample_state_cost = 0.0;
            float sample_control_cost = 0.0;
            float sample_static_cost = 0.0;
            float sample_visibility_cost = 0.0;

            // rollout the trajectory -- assume we are placing the result in the larger u_dist/u_weight arrays
            const State *x_init_state = reinterpret_cast<const State *>(optimization_args->x_init);
            const State *x_goal_state = reinterpret_cast<const State *>(optimization_args->x_goal);
            Control *u_dist_controls = reinterpret_cast<Control *>(&u_dists[sample_index * num_controls]);

            generate_controls(globalState, sample_index, u_nom, num_controls, u_limits, u_dist_limits, optimization_args->dynamics_type, u_dist_controls);

            State current_state = {x_init_state->x, x_init_state->y, x_init_state->v, x_init_state->theta};
            State state_step = {0, 0, 0, 0};

            for (int i = 1; i <= num_controls; i++) {
                // generate the next state
                Control c = {u_nom[i - 1].a + u_dist_controls[i - 1].a, u_nom[i - 1].delta + u_dist_controls[i - 1].delta};
                if (optimization_args->dynamics_type == 1) {
                    holonomic_step(
                        &current_state,
                        &c,
                        dt,
                        u_limits[0],
                        optimization_args->holonomic_max_heading_change,
                        optimization_args->holonomic_max_curvature,
                        &current_state
                    );
                } else {
                    // runge_kutta_step returns derivative, we multiply by dt here
                    runge_kutta_step(&current_state, &c, optimization_args->vehicle_length, dt, &state_step);
                    update_state(&current_state, &state_step, dt, &current_state);
                }
                if (rollout_states != NULL) {
                    rollout_states[sample_index * num_controls + (i - 1)] = current_state;
                }

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
                if (optimization_args->dynamics_type == 0 && optimization_args->steering_rate_weight > 0.0f && i > 1) {
                    float prev_delta = u_nom[i - 2].delta + u_dist_controls[i - 2].delta; // previous applied delta
                    float rate = c.delta - prev_delta; // instantaneous change (already per-step)
                    control_err += optimization_args->steering_rate_weight * rate * rate;
                }

                // penalize obstacles
                obstacle_err = obstacle_cost(
                    obstacles,
                    obstacle_states,
                    num_obstacles,
                    obstacle_steps,
                    i - 1,
                    current_state.x,
                    current_state.y,
                    current_state.theta,
                    optimization_args->vehicle_length,
                    optimization_args->vehicle_width
                );
                obstacle_err += polygon_obstacle_cost(
                    polygon_vertices,
                    polygon_offsets,
                    num_polygon_obstacles,
                    current_state.x,
                    current_state.y,
                    current_state.theta,
                    optimization_args->vehicle_length,
                    optimization_args->vehicle_width,
                    optimization_args->static_collision_cost,
                    optimization_args->static_hard_clearance_margin,
                    optimization_args->static_clearance_margin,
                    optimization_args->static_clearance_weight
                );

                // penalize visibility
                visibility_err = 0;

                if (method == OURS) {
                    // Note: 'M' for visibility costs is optimization_args->M.
                    // The 'costmap' related parameters (height, width, origin_x, etc.) are in costmap_args.
                    visibility_err = our_cost(optimization_args->M, costmap, costmap_args->height, costmap_args->width, costmap_args->origin_x, costmap_args->origin_y, costmap_args->resolution, current_state.x, current_state.y, i);
                } else if (method == HIGGINS) {
                    visibility_err = higgins_cost(
                        optimization_args->M,
                        obstacles,
                        obstacle_states,
                        num_obstacles,
                        obstacle_steps,
                        i - 1,
                        current_state.x,
                        current_state.y,
                        optimization_args->scan_range
                    );
                } else if (method == ANDERSEN) {
                    // Velocity for Andersen cost is based on nominal trajectory difference
                    visibility_err = andersen_cost(
                        optimization_args->M,
                        obstacles,
                        obstacle_states,
                        num_obstacles,
                        obstacle_steps,
                        i - 1,
                        current_state.x,
                        current_state.y,
                        (x_nom[i].x - x_nom[i-1].x),
                        (x_nom[i].y - x_nom[i-1].y)
                    );
                }
                // NO_VISIBILITY and INFO_GAIN_LIKE (if it implies using 'our_cost' already handled by OURS) might not need explicit handling here if OURS covers INFO_GAIN_LIKE

                score += state_err + final_state_err + control_err + obstacle_err + visibility_err;
                sample_state_cost += state_err + final_state_err;
                sample_control_cost += control_err;
                sample_static_cost += obstacle_err;
                sample_visibility_cost += visibility_err;

                if( isnan(score) ){
                    // printf( "score overflow -- prev score: %f, state: %f, final: %f, control: %f, obstacle: %f, visibility: %f\\n", prev_score, state_err, final_state_err, control_err, obstacle_err, visibility_err );
                    score = FLT_MAX / (samples > 0 ? samples : 1); // Assign a large penalty
                    break;
                }
                // prev_score = score;
            }
            u_weights[sample_index] = score;
            if (component_costs != NULL) {
                int component_base = sample_index * 5;
                component_costs[component_base + 0] = sample_state_cost;
                component_costs[component_base + 1] = sample_control_cost;
                component_costs[component_base + 2] = sample_static_cost;
                component_costs[component_base + 3] = sample_visibility_cost;
                component_costs[component_base + 4] = 0.0f;
            }
        }
    }

    extern "C" __global__
    void add_dynamic_collision_costs(
            const State *rollout_states,
            const float *actor_states,
            const float *actor_collision_geometry,
            int samples,
            int num_controls,
            int num_dynamic_actors,
            float ego_circle_radius,
            float ego_circle_offset_x,
            float ego_circle_offset_y,
            float collision_penalty,
            float clearance_margin,
            float clearance_weight,
            float *dynamic_costs,
            float *sample_costs
    ) {
        int start_sample_index = blockIdx.x * blockDim.x + threadIdx.x;

        for (int sample_index = start_sample_index; sample_index < samples; sample_index += blockDim.x * gridDim.x) {
            bool collided = false;
            float dynamic_cost = 0.0f;
            float min_clearance = FLT_MAX;

            for (int actor_idx = 0; actor_idx < num_dynamic_actors; ++actor_idx) {
                int actor_base = actor_idx * num_controls * 4;
                int geometry_base = actor_idx * 3;
                float actor_circle_radius = actor_collision_geometry[geometry_base + 0];
                float actor_circle_offset_x = actor_collision_geometry[geometry_base + 1];
                float actor_circle_offset_y = actor_collision_geometry[geometry_base + 2];

                for (int step_idx = 0; step_idx < num_controls; ++step_idx) {
                    const State ego_state = rollout_states[sample_index * num_controls + step_idx];
                    int actor_offset = actor_base + step_idx * 4;
                    float clearance = three_circle_min_clearance(
                        ego_state.x,
                        ego_state.y,
                        ego_state.theta,
                        ego_circle_radius,
                        ego_circle_offset_x,
                        ego_circle_offset_y,
                        actor_states[actor_offset + 0],
                        actor_states[actor_offset + 1],
                        actor_states[actor_offset + 3],
                        actor_circle_radius,
                        actor_circle_offset_x,
                        actor_circle_offset_y
                    );
                    if (clearance < min_clearance) {
                        min_clearance = clearance;
                    }
                    if (clearance < 0.0f) {
                        collided = true;
                        // printf( "Sample %d, Actor %d, Step %d: Collision detected ( %f / %f)  -- Ego state: (x: %f, y: %f, theta: %f)  -- Actor state: (x: %f, y: %f, theta: %f)\\n",
                        //         sample_index, actor_idx, step_idx, clearance, min_clearance, ego_state.x, ego_state.y, ego_state.theta,
                        //         actor_states[actor_offset + 0], actor_states[actor_offset + 1], actor_states[actor_offset + 3]  );
                    } else if (clearance_margin > 0.0f && clearance_weight > 0.0f && clearance < clearance_margin) {
                        float deficit = clearance_margin - clearance;
                        dynamic_cost += clearance_weight * deficit * deficit;
                        // printf( "Sample %d, Actor %d, Step %d: Clearance penalty (clearance = %f, deficit = %f, incremental cost = %f) -- Ego state: (x: %f, y: %f, theta: %f)  -- Actor state: (x: %f, y: %f, theta: %f)\\n",
                        //         sample_index, actor_idx, step_idx, clearance, deficit, clearance_weight * deficit * deficit, ego_state.x, ego_state.y, ego_state.theta,
                        //         actor_states[actor_offset + 0], actor_states[actor_offset + 1], actor_states[actor_offset + 3] );
                    }
                }
            }

            if (collided) {
                dynamic_cost += collision_penalty;
            }

            if( dynamic_cost > 0.0f ){
                // printf( "***\\n Sample %d: dynamic cost = %f (min clearance = %f, collided = %d) -- clearance margin: %f, clearance weight: %f, collision penalty: %f\\n",
                //         sample_index, dynamic_cost, min_clearance, collided, clearance_margin, clearance_weight, collision_penalty );
            }
            sample_costs[sample_index] += dynamic_cost;
            if (dynamic_costs != NULL) {
                dynamic_costs[sample_index] = dynamic_cost;
            }
        }
    }

    extern "C" __global__
    void add_occupancy_grid_costs(
            const State *rollout_states,
            const float *occupancy_grids,
            int samples,
            int num_controls,
            int occupancy_steps,
            int rows,
            int cols,
            float origin_x,
            float origin_y,
            float resolution,
            float ego_circle_radius,
            float ego_circle_offset_x,
            float ego_circle_offset_y,
            float collision_penalty,
            float soft_weight,
            float hard_threshold,
            float *occupancy_costs,
            float *sample_costs
    ) {
        int start_sample_index = blockIdx.x * blockDim.x + threadIdx.x;

        for (int sample_index = start_sample_index; sample_index < samples; sample_index += blockDim.x * gridDim.x) {
            float occupancy_cost = 0.0f;
            bool collided = false;

            for (int step_idx = 0; step_idx < num_controls; ++step_idx) {
                int grid_step = step_idx;
                if (grid_step < 0) {
                    grid_step = 0;
                }
                if (grid_step >= occupancy_steps) {
                    grid_step = occupancy_steps - 1;
                }
                const State ego_state = rollout_states[sample_index * num_controls + step_idx];
                float max_probability = 0.0f;
                for (int circle_idx = 0; circle_idx < 3; ++circle_idx) {
                    float cx;
                    float cy;
                    circle_center_from_pose(
                        ego_state.x,
                        ego_state.y,
                        ego_state.theta,
                        ego_circle_offset_x,
                        ego_circle_offset_y,
                        circle_idx,
                        &cx,
                        &cy
                    );
                    int col = epsilon_round((cx - origin_x) / resolution);
                    int row = epsilon_round((cy - origin_y) / resolution);
                    if (row < 0 || row >= rows || col < 0 || col >= cols) {
                        collided = true;
                        max_probability = 1.0f;
                        continue;
                    }
                    float probability = occupancy_grids[(grid_step * rows + row) * cols + col];
                    if (probability > max_probability) {
                        max_probability = probability;
                    }
                }
                occupancy_cost += soft_weight * max_probability;
                if (max_probability >= hard_threshold) {
                    collided = true;
                }
            }

            if (collided) {
                occupancy_cost += collision_penalty;
            }
            sample_costs[sample_index] += occupancy_cost;
            if (occupancy_costs != NULL) {
                occupancy_costs[sample_index] = occupancy_cost;
            }
        }
    }

    extern "C" __global__
    void add_sample_costs(
            float *sample_costs,
            const float *extra_costs,
            int samples
    ) {
        int start_sample_index = blockIdx.x * blockDim.x + threadIdx.x;
        for (int sample_index = start_sample_index; sample_index < samples; sample_index += blockDim.x * gridDim.x) {
            sample_costs[sample_index] += extra_costs[sample_index];
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
    void filter_dynamic_collision_weights(
            int samples,
            float *u_weights,
            const float *dynamic_costs,
            float collision_threshold,
            float *u_weight_total
    ) {
      int start_sample_index = blockIdx.x * blockDim.x + threadIdx.x;

      for (int sample_index = start_sample_index; sample_index < samples; sample_index += blockDim.x * gridDim.x) {
        float weight = u_weights[sample_index];
        if (dynamic_costs[sample_index] >= collision_threshold) {
            weight = 0.0f;
            u_weights[sample_index] = 0.0f;
        }
        if (weight > 0.0f && !isnan(weight)) {
            atomicAdd(u_weight_total, weight);
        }
      }
    }

    extern "C" __global__
    void filter_static_collision_weights(
            int samples,
            float *u_weights,
            const float *component_costs,
            float collision_threshold,
            float *u_weight_total
    ) {
      int start_sample_index = blockIdx.x * blockDim.x + threadIdx.x;

      for (int sample_index = start_sample_index; sample_index < samples; sample_index += blockDim.x * gridDim.x) {
        float weight = u_weights[sample_index];
        float static_cost = component_costs[sample_index * 5 + 2];
        if (static_cost >= collision_threshold) {
            weight = 0.0f;
            u_weights[sample_index] = 0.0f;
        }
        if (weight > 0.0f && !isnan(weight)) {
            atomicAdd(u_weight_total, weight);
        }
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
            const Optimization_Params *optimization_args,
            Control *u_mppi
    ) {
      int sample_idx = blockIdx.x * blockDim.x + threadIdx.x; // Iterate over samples
      float u_weight_total_float = *u_weight_total; // Dereference
      const float *u_limits = optimization_args->u_limits;

      if (sample_idx < samples) {
        float weight_normalized;
        if (!is_zero(u_weight_total_float)) {
            weight_normalized = u_weights[sample_idx] / u_weight_total_float;
        } else {
            return;
        }

        for (int ctrl_idx = 0; ctrl_idx < num_controls; ++ctrl_idx) {
            int dist_flat_idx = sample_idx * num_controls + ctrl_idx;
            float a_nom = u_nom[ctrl_idx].a;
            float delta_nom = u_nom[ctrl_idx].delta;
            float a_candidate = a_nom + u_dist[dist_flat_idx].a;
            float delta_candidate = delta_nom + u_dist[dist_flat_idx].delta;

            if (optimization_args->dynamics_type == 1) {
                atomicAdd(&(u_mppi[ctrl_idx].a), (a_candidate - a_nom) * weight_normalized);
                atomicAdd(&(u_mppi[ctrl_idx].delta), (delta_candidate - delta_nom) * weight_normalized);
            } else {
                float a_nom_z = unsquash_control(a_nom, u_limits[0]);
                float delta_nom_z = unsquash_control(delta_nom, u_limits[1]);
                float a_candidate_z = unsquash_control(a_candidate, u_limits[0]);
                float delta_candidate_z = unsquash_control(delta_candidate, u_limits[1]);

                // Ackermann controls are accumulated in unconstrained space.
                atomicAdd(&(u_mppi[ctrl_idx].a), (a_candidate_z - a_nom_z) * weight_normalized);
                atomicAdd(&(u_mppi[ctrl_idx].delta), (delta_candidate_z - delta_nom_z) * weight_normalized);
            }
        }
      }
    }
"""

try:
    _establish_module_context()  # Ensure context is ready for SourceModule compilation
    try:
        _COMPILED_MODULE = SourceModule(_MPPI_CUDA_SOURCE, no_extern_c=True)
    finally:
        _pop_module_context_after_compile()
    if _MODULE_CONTEXT is None:
        raise RuntimeError("Failed to create a CUDA context")
except cuda.CompileError as e:
    print("CUDA Compilation Error:", file=sys.stderr)
    print(e.stderr, file=sys.stderr)
    raise
except Exception as e:
    print(f"Error compiling MPPI CUDA module: {e}", file=sys.stderr)
    raise


class CudaBufferCache:
    """Reusable device buffers for end-to-end CUDA execution."""

    def __init__(self):
        self._capacity = {}
        self._buffers = {}

    def reserve_data(self, name, required_capacity, zero_fill=False):
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
        _cuda.memcpy_htod(buf, arr)
        return buf

    def get(self, name):
        return self._buffers.get(name)

    def release(self):
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


class MPPI:

    visibility_methods = {
        "Ours": 0,  # CUDA: OURS (can be used for information gain if costmap is such)
        "Ours-Wide": 0,  # Alias for Ours (wider roads in planning)
        "Right": 0,  # Alias for Ours
        "Left": 0,  # Alias for Ours
        "Higgins": 1,  # CUDA: HIGGINS
        "Andersen": 2,  # CUDA: ANDERSEN
        "Nominal": 3,  # CUDA: NO_VISIBILITY (i.e., no explicit visibility cost term)
        "OCE": 3,  # Rollout visibility disabled; OCE is applied in a second stage.
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
            ("obstacle_steps", np.int32),
            ("num_polygon_obstacles", np.int32),
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
            ("vehicle_width", np.float32),
            ("steering_rate_weight", np.float32),
            ("static_collision_cost", np.float32),
            ("static_hard_clearance_margin", np.float32),
            ("static_clearance_margin", np.float32),
            ("static_clearance_weight", np.float32),
            ("dynamics_type", np.int32),
            ("holonomic_max_heading_change", np.float32),
            ("holonomic_max_curvature", np.float32),
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
        vehicle_length: float,
        vehicle_width: float,
        samples: int,
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
        dynamic_clearance_margin=0.5,
        dynamic_clearance_weight=100.0,
        dynamic_collision_cost=float(LARGE_COLLISION_COST),
        static_clearance_margin=0.0,
        static_clearance_weight=0.0,
        static_collision_cost=float(LARGE_COLLISION_COST),
        static_hard_clearance_margin=0.0,
        mppi_occupancy_weight=None,
        dynamics_type="ackermann",
        holonomic_max_heading_change=0.0,
        holonomic_max_curvature=0.0,
    ):
        """Initialize MPPI controller.

        Parameters mirror optimization / cost settings. Gaussian noise sampling
        is used for disturbances with std dev = u_dist_limits (clamped to u_limits).
        """
        if _MODULE_CONTEXT is None:
            raise RuntimeError("CUDA context unavailable for MPPI initialization")

        self.mppi_context = _MODULE_CONTEXT
        self.samples = np.int32(samples)
        self.debug = debug
        self.vehicle_length = float(vehicle_length)
        self.vehicle_width = float(vehicle_width)
        self.dynamics_type = str(dynamics_type).strip().lower()
        if self.dynamics_type not in {"ackermann", "holonomic"}:
            raise ValueError("dynamics_type must be 'ackermann' or 'holonomic'")
        self.dynamic_clearance_margin = float(dynamic_clearance_margin)
        self.dynamic_clearance_weight = float(dynamic_clearance_weight)
        self.dynamic_collision_cost = float(dynamic_collision_cost)
        self.static_clearance_margin = float(static_clearance_margin)
        self.static_clearance_weight = float(static_clearance_weight)
        self.static_collision_cost = float(static_collision_cost)
        self.static_hard_clearance_margin = float(static_hard_clearance_margin)
        self.mppi_occupancy_weight = (
            float(dynamic_clearance_weight)
            if mppi_occupancy_weight is None
            else float(mppi_occupancy_weight)
        )
        self.dynamic_clearance_diagnostics = bool(debug)
        self.ego_collision_geometry = _actor_collision_geometry_from_extent(
            (self.vehicle_length / 2.0, self.vehicle_width / 2.0)
        )

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
        self.optimization_args["vehicle_length"] = np.float32(vehicle_length)
        self.optimization_args["vehicle_width"] = np.float32(vehicle_width)
        self.optimization_args["steering_rate_weight"] = np.float32(
            steering_rate_weight
        )
        self.optimization_args["static_collision_cost"] = np.float32(
            self.static_collision_cost
        )
        self.optimization_args["static_hard_clearance_margin"] = np.float32(
            self.static_hard_clearance_margin
        )
        self.optimization_args["static_clearance_margin"] = np.float32(
            self.static_clearance_margin
        )
        self.optimization_args["static_clearance_weight"] = np.float32(
            self.static_clearance_weight
        )
        self.optimization_args["dynamics_type"] = np.int32(
            1 if self.dynamics_type == "holonomic" else 0
        )
        self.optimization_args["holonomic_max_heading_change"] = np.float32(
            holonomic_max_heading_change
        )
        self.optimization_args["holonomic_max_curvature"] = np.float32(
            holonomic_max_curvature
        )

        # Allocate GPU buffers inside context
        with _active_cuda_context(self.mppi_context):
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
            setup_kernel_func = _COMPILED_MODULE.get_function("setup_kernel")
            setup_kernel_func(
                self.globalState_gpu,
                np.uint32(
                    seed if seed is not None else np.random.randint(0, 2**32 - 1)
                ),
                block=block_cfg,
                grid=grid_cfg,
            )

        # Diagnostics placeholders
        self.last_ess = None
        self.last_cost_min = None
        self.last_cost_max = None
        self.last_cost_mean = None
        self.last_total_costs = None
        self.last_oce_costs = None
        self.last_rollout_costs = None
        self.last_cost_components = None
        self.last_dynamic_costs = None
        self.last_dynamic_clearance_summary = None
        self.last_dynamic_actor_debug = None
        self.last_weight_summary = None
        self.last_static_filter_all_invalid = False
        self.last_dynamic_filter_all_invalid = False
        self.last_oce_result = None
        self.last_rollout_states = None
        self.last_num_polygon_obstacles = 0
        self.last_timing = None

        if self.debug:
            print(
                f"[MPPI] Initialized samples={self.samples} c_lambda={c_lambda} u_limits={u_limits} "
                f"u_dist_limits={u_dist_limits} method={method} L={vehicle_length} "
                f"steer_rate_w={steering_rate_weight} dynamic_clearance_margin={dynamic_clearance_margin} "
                f"dynamic_clearance_weight={dynamic_clearance_weight} "
                f"static_hard_clearance_margin={static_hard_clearance_margin} "
                f"static_clearance_margin={static_clearance_margin} "
                f"static_clearance_weight={static_clearance_weight}"
            )

    def set_steering_limit(self, max_steer_rad: float):
        self.optimization_args["u_limits"][0, 1] = np.float32(max_steer_rad)
        with _active_cuda_context(self.mppi_context):
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
        self,
        costmap,
        origin,
        resolution,
        x_init,
        x_goal,
        x_nom,
        u_nom,
        obstacles,
        dt,
        oce_data=None,
        occupancy_grids=None,
    ):
        if self.mppi_context is None:
            raise RuntimeError("MPPI has no CUDA context")

        if obstacles is None:
            obstacles = []

        timing_start = perf_counter()
        timing_last = timing_start
        timing = {}

        def record_timing(name, sync_cuda=False):
            nonlocal timing_last
            if sync_cuda and self.debug:
                _cuda.Context.synchronize()
            now = perf_counter()
            timing[name] = timing.get(name, 0.0) + (now - timing_last)
            timing_last = now

        # Host-side outputs
        u_mppi_host = np.zeros_like(u_nom, dtype=np.float32)
        u_dist_host_reshaped = np.zeros(
            (self.samples, u_nom.shape[0], u_nom.shape[1]), dtype=np.float32
        )
        u_weights_host = np.zeros(self.samples, dtype=np.float32)
        record_timing("host_output_init")

        costmap_gpu = None
        obstacles_gpu = np.intp(0)
        obstacle_states_gpu = np.intp(0)
        polygon_vertices_gpu = np.intp(0)
        polygon_offsets_gpu = np.intp(0)
        u_nom_gpu = None
        x_nom_gpu = None
        u_mppi_gpu = None
        u_weight_gpu = None
        u_dist_gpu = None
        u_weight_min_gpu = None
        u_weight_total_gpu = None
        pre_static_weight_gpu = None
        pre_dynamic_weight_gpu = None
        rollout_states_gpu = None
        combined_costs_gpu = None
        component_costs_gpu = None
        dynamic_costs_gpu = None
        dynamic_actor_states_gpu = None
        dynamic_actor_geometry_gpu = None
        occupancy_grids_gpu = None
        occupancy_costs_gpu = None
        oce_cost_gpu = None
        owned_oce_cost_gpu = None

        with _active_cuda_context(self.mppi_context):
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
                record_timing("costmap_upload")

                # Nominal control & state trajectories
                u_nom_host = np.array(u_nom, dtype=np.float32)
                num_controls_timesteps, num_control_elements = u_nom_host.shape
                occupancy_payload = _prepare_occupancy_grid_stack(occupancy_grids)
                discrete_oce_payload = _resolve_discrete_oce_payload(oce_data)

                (
                    obstacle_extents_host,
                    obstacle_states_host,
                    dynamic_actors,
                    polygon_vertices_host,
                    polygon_offsets_host,
                ) = _prepare_rollout_obstacle_batches(
                    obstacles, num_controls_timesteps, occupancy_payload
                )
                if discrete_oce_payload is None:
                    (
                        scene_payload,
                        oce_config,
                        entropy_space,
                        oce_eps,
                        oce_discount,
                        oce_materialize_host,
                        oce_return_visibility_tensor,
                        oce_scorer,
                    ) = _resolve_oce_payload(oce_data)
                else:
                    scene_payload = None
                    oce_config = None
                    entropy_space = "discrete"
                    oce_eps = 1.0e-9
                    oce_discount = 1.0
                    oce_materialize_host = False
                    oce_return_visibility_tensor = bool(
                        discrete_oce_payload.get("return_visibility_tensor", False)
                    )
                    oce_scorer = None
                self.last_dynamic_actor_debug = _summarize_dynamic_actors(
                    dynamic_actors,
                    self.ego_collision_geometry,
                )
                record_timing("obstacle_prepare")

                obstacle_steps = np.int32(1 if obstacle_states_host.shape[0] else 0)
                num_obstacles = np.int32(int(obstacle_states_host.shape[0]))
                if num_obstacles > 0:
                    obstacles_gpu = _cuda.mem_alloc(obstacle_extents_host.nbytes)  # type: ignore[attr-defined]
                    _cuda.memcpy_htod(obstacles_gpu, obstacle_extents_host)  # type: ignore[attr-defined]

                    obstacle_states_gpu = _cuda.mem_alloc(obstacle_states_host.nbytes)  # type: ignore[attr-defined]
                    _cuda.memcpy_htod(obstacle_states_gpu, obstacle_states_host)  # type: ignore[attr-defined]
                record_timing("static_obstacle_upload")

                num_polygon_obstacles = np.int32(
                    max(0, polygon_offsets_host.shape[0] - 1)
                )
                if num_polygon_obstacles > 0:
                    polygon_vertices_gpu = _cuda.mem_alloc(polygon_vertices_host.nbytes)  # type: ignore[attr-defined]
                    _cuda.memcpy_htod(polygon_vertices_gpu, polygon_vertices_host)  # type: ignore[attr-defined]
                    polygon_offsets_gpu = _cuda.mem_alloc(polygon_offsets_host.nbytes)  # type: ignore[attr-defined]
                    _cuda.memcpy_htod(polygon_offsets_gpu, polygon_offsets_host)  # type: ignore[attr-defined]
                record_timing("polygon_upload")
                self.last_num_polygon_obstacles = int(num_polygon_obstacles)
                if self.debug and occupancy_payload is not None:
                    print(
                        "[MPPI] using occupancy grid obstacle costs; "
                        "legacy obstacle geometry disabled"
                    )
                if self.debug and self.last_num_polygon_obstacles:
                    print(
                        "[MPPI] static polygon obstacles="
                        f"{self.last_num_polygon_obstacles} vertices="
                        f"{polygon_vertices_host.shape[0] // 2}"
                    )

                u_nom_gpu = _cuda.mem_alloc(u_nom_host.nbytes)  # type: ignore[attr-defined]
                _cuda.memcpy_htod(u_nom_gpu, u_nom_host)  # type: ignore[attr-defined]

                x_nom_host = np.array(x_nom, dtype=np.float32)
                x_nom_gpu = _cuda.mem_alloc(x_nom_host.nbytes)  # type: ignore[attr-defined]
                _cuda.memcpy_htod(x_nom_gpu, x_nom_host)  # type: ignore[attr-defined]
                record_timing("nominal_upload")

                # Disturbance & weights buffers
                u_weight_gpu = _cuda.mem_alloc(  # type: ignore[attr-defined]
                    int(self.samples * np.dtype(np.float32).itemsize)
                )
                u_dist_gpu = _cuda.mem_alloc(int(self.samples * u_nom_host.nbytes))  # type: ignore[attr-defined]
                need_rollout_trace = (
                    bool(dynamic_actors)
                    or scene_payload is not None
                    or discrete_oce_payload is not None
                    or occupancy_payload is not None
                    or bool(self.debug)
                )
                if need_rollout_trace:
                    rollout_state_count = (
                        int(self.samples) * int(num_controls_timesteps) * 4
                    )
                    rollout_states_gpu = _cuda.mem_alloc(
                        int(rollout_state_count * np.dtype(np.float32).itemsize)
                    )  # type: ignore[attr-defined]
                component_costs_gpu = _cuda.mem_alloc(
                    int(self.samples) * 5 * np.dtype(np.float32).itemsize
                )  # type: ignore[attr-defined]
                record_timing("rollout_buffer_alloc")

                # Update optimization args for this solve
                self.optimization_args["dt"] = np.float32(dt)
                self.optimization_args["num_controls"] = np.int32(
                    num_controls_timesteps
                )
                self.optimization_args["num_obstacles"] = num_obstacles
                self.optimization_args["obstacle_steps"] = obstacle_steps
                self.optimization_args["num_polygon_obstacles"] = num_polygon_obstacles
                self.optimization_args["x_init"] = np.array(x_init, dtype=np.float32)
                self.optimization_args["x_goal"] = np.array(x_goal, dtype=np.float32)
                _cuda.memcpy_htod(self.optimization_args_gpu, self.optimization_args)  # type: ignore[attr-defined]
                record_timing("optimization_args_upload")

                # Kernel handles
                perform_rollout_func = _COMPILED_MODULE.get_function("perform_rollout")
                add_dynamic_collision_costs_func = _COMPILED_MODULE.get_function(
                    "add_dynamic_collision_costs"
                )
                add_occupancy_grid_costs_func = _COMPILED_MODULE.get_function(
                    "add_occupancy_grid_costs"
                )
                add_sample_costs_func = _COMPILED_MODULE.get_function(
                    "add_sample_costs"
                )
                min_weight_func = _COMPILED_MODULE.get_function("min_weight")
                calculate_weights_func = _COMPILED_MODULE.get_function(
                    "calculate_weights"
                )
                filter_static_collision_weights_func = _COMPILED_MODULE.get_function(
                    "filter_static_collision_weights"
                )
                filter_dynamic_collision_weights_func = _COMPILED_MODULE.get_function(
                    "filter_dynamic_collision_weights"
                )
                calculate_mppi_control_func = _COMPILED_MODULE.get_function(
                    "calculate_mppi_control"
                )
                record_timing("kernel_lookup")

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
                    obstacles_gpu,
                    obstacle_states_gpu,
                    polygon_vertices_gpu,
                    polygon_offsets_gpu,
                    self.optimization_args_gpu,
                    u_dist_gpu,
                    (
                        rollout_states_gpu
                        if rollout_states_gpu is not None
                        else np.intp(0)
                    ),
                    u_weight_gpu,
                    component_costs_gpu,
                    block=block_1d,
                    grid=grid_1d,
                )
                record_timing("rollout_kernel", sync_cuda=True)

                sample_cost_bytes = int(self.samples * np.dtype(np.float32).itemsize)

                if dynamic_actors:
                    dynamic_actor_states_host, dynamic_actor_geometry_host = (
                        _pack_dynamic_actor_arrays(dynamic_actors)
                    )
                    dynamic_actor_states_gpu = _cuda.mem_alloc(
                        dynamic_actor_states_host.nbytes
                    )  # type: ignore[attr-defined]
                    dynamic_actor_geometry_gpu = _cuda.mem_alloc(
                        dynamic_actor_geometry_host.nbytes
                    )  # type: ignore[attr-defined]
                    dynamic_costs_gpu = _cuda.mem_alloc(sample_cost_bytes)  # type: ignore[attr-defined]
                    _cuda.memset_d8(dynamic_costs_gpu, 0, sample_cost_bytes)  # type: ignore[attr-defined]
                    _cuda.memcpy_htod(
                        dynamic_actor_states_gpu, dynamic_actor_states_host
                    )  # type: ignore[attr-defined]
                    _cuda.memcpy_htod(
                        dynamic_actor_geometry_gpu, dynamic_actor_geometry_host
                    )  # type: ignore[attr-defined]
                    record_timing("dynamic_actor_upload")
                    add_dynamic_collision_costs_func(
                        rollout_states_gpu,
                        dynamic_actor_states_gpu,
                        dynamic_actor_geometry_gpu,
                        self.samples,
                        np.int32(num_controls_timesteps),
                        np.int32(dynamic_actor_geometry_host.shape[0]),
                        np.float32(self.ego_collision_geometry[0]),
                        np.float32(self.ego_collision_geometry[1]),
                        np.float32(self.ego_collision_geometry[2]),
                        np.float32(self.dynamic_collision_cost),
                        np.float32(self.dynamic_clearance_margin),
                        np.float32(self.dynamic_clearance_weight),
                        dynamic_costs_gpu,
                        u_weight_gpu,
                        block=block_1d,
                        grid=grid_1d,
                    )
                    record_timing("dynamic_cost_kernel", sync_cuda=True)

                if occupancy_payload is not None:
                    occupancy_host = occupancy_payload["grids"]
                    occupancy_grids_gpu = _cuda.mem_alloc(occupancy_host.nbytes)  # type: ignore[attr-defined]
                    occupancy_costs_gpu = _cuda.mem_alloc(sample_cost_bytes)  # type: ignore[attr-defined]
                    _cuda.memset_d8(occupancy_costs_gpu, 0, sample_cost_bytes)  # type: ignore[attr-defined]
                    _cuda.memcpy_htod(occupancy_grids_gpu, occupancy_host)  # type: ignore[attr-defined]
                    record_timing("occupancy_grid_upload")
                    occupancy_weight = (
                        self.mppi_occupancy_weight
                        if occupancy_payload["soft_weight"] is None
                        else occupancy_payload["soft_weight"]
                    )
                    add_occupancy_grid_costs_func(
                        rollout_states_gpu,
                        occupancy_grids_gpu,
                        self.samples,
                        np.int32(num_controls_timesteps),
                        np.int32(occupancy_host.shape[0]),
                        np.int32(occupancy_host.shape[1]),
                        np.int32(occupancy_host.shape[2]),
                        np.float32(occupancy_payload["origin"][0]),
                        np.float32(occupancy_payload["origin"][1]),
                        np.float32(occupancy_payload["resolution"]),
                        np.float32(self.ego_collision_geometry[0]),
                        np.float32(self.ego_collision_geometry[1]),
                        np.float32(self.ego_collision_geometry[2]),
                        np.float32(self.dynamic_collision_cost),
                        np.float32(occupancy_weight),
                        np.float32(occupancy_payload["hard_threshold"]),
                        occupancy_costs_gpu,
                        u_weight_gpu,
                        block=block_1d,
                        grid=grid_1d,
                    )
                    record_timing("occupancy_cost_kernel", sync_cuda=True)

                # BUGBUG - Disable OCE as the scoring is expensive over 1000's of rollouts - for
                #          now, the diversity comes from the route planner.  MPPI can focus on
                #          a local collision free path.
                #
                # self.last_oce_result = None
                # if discrete_oce_payload is not None:
                #     if evaluate_discrete_oce_rollouts_gpu is None:
                #         raise RuntimeError(
                #             "Discrete OCE weighting was requested, but the PyCUDA "
                #             "discrete OCE evaluator could not be imported: "
                #             f"{_discrete_oce_import_error}"
                #         )
                #     discrete_inputs = _prepare_discrete_oce_rollout_inputs(
                #         discrete_oce_payload,
                #         num_controls_timesteps,
                #     )
                #     if discrete_inputs is not None:
                #         discrete_horizon = min(
                #             int(discrete_inputs["horizon"]),
                #             int(num_controls_timesteps),
                #         )
                #         oce_result = evaluate_discrete_oce_rollouts_gpu(
                #             rollout_states_d=rollout_states_gpu,
                #             num_rollouts=int(self.samples),
                #             x_init=np.asarray(x_init, dtype=np.float32),
                #             rollout_state_stride=4,
                #             state_centers=discrete_inputs["state_centers"],
                #             static_grid=discrete_inputs["static_grid"],
                #             grid_origin=discrete_inputs["grid_origin"],
                #             grid_resolution=discrete_inputs["grid_resolution"],
                #             transition_data=discrete_inputs["transition_data"],
                #             transition_indices=discrete_inputs["transition_indices"],
                #             transition_indptr=discrete_inputs["transition_indptr"],
                #             prefix_beliefs=discrete_inputs["prefix_beliefs"][
                #                 :, : discrete_horizon + 1, :
                #             ],
                #             beliefs=discrete_inputs["beliefs"],
                #             horizon=discrete_horizon,
                #             scan_range=discrete_inputs["scan_range"],
                #             return_visibility=discrete_inputs["return_visibility"],
                #             occupancy_probability_grids=discrete_inputs[
                #                 "occupancy_probability_grids"
                #             ],
                #             occupancy_owner_mask_grids=discrete_inputs[
                #                 "occupancy_owner_mask_grids"
                #             ],
                #             agent_owner_bits=discrete_inputs["agent_owner_bits"],
                #             occupancy_threshold=discrete_inputs["occupancy_threshold"],
                #             return_device_scores=True,
                #         )
                #         if getattr(oce_result, "metadata", None) is not None:
                #             oce_result.metadata["state_reduction"] = discrete_inputs[
                #                 "state_reduction"
                #             ]
                #         self.last_oce_result = oce_result
                #         oce_cost_gpu = getattr(oce_result, "device_scores", None)
                #         if oce_cost_gpu is None:
                #             raise RuntimeError(
                #                 "Discrete OCE rollout scoring did not return a device score buffer."
                #             )
                #         owned_oce_cost_gpu = oce_cost_gpu
                #         add_sample_costs_func(
                #             u_weight_gpu,
                #             oce_cost_gpu,
                #             self.samples,
                #             block=block_1d,
                #             grid=grid_1d,
                #         )
                #         if self.debug:
                #             print(
                #                 "[MPPI] OCE entropy_space=discrete "
                #                 f"device evaluations={int(self.samples)} "
                #                 f"agents={len(discrete_inputs['agent_ids'])} "
                #                 f"states={discrete_inputs['state_reduction']['selected_states']}/"
                #                 f"{discrete_inputs['state_reduction']['original_states']}"
                #             )
                #     record_timing("oce_scoring", sync_cuda=True)
                # elif scene_payload is not None:
                #     if oce_scorer is None:
                #         oce_scorer = score_oce_scene_rollouts_device
                #     if oce_scorer is None:
                #         raise RuntimeError(
                #             "OCE weighting was requested, but the PyCUDA OCE evaluator "
                #             f"could not be imported: {_oce_import_error}"
                #         )
                #     if not callable(oce_scorer):
                #         raise TypeError("oce_data scorer/score_func must be callable.")
                #     oce_result = oce_scorer(
                #         scene=scene_payload,
                #         rollout_states_d=rollout_states_gpu,
                #         num_rollouts=int(self.samples),
                #         oce_config=oce_config,
                #         eps=oce_eps,
                #         entropy_space=entropy_space,
                #         discount=oce_discount,
                #         cuda_cache=_cuda_buffer_cache,
                #         materialize_host=oce_materialize_host,
                #         return_visibility_tensor=oce_return_visibility_tensor,
                #         rollout_state_stride=4,
                #         debug=bool(self.debug),
                #     )
                #     self.last_oce_result = oce_result
                #     if (
                #         hasattr(oce_result, "device_accumulation")
                #         and oce_result.device_accumulation is not None
                #     ):
                #         oce_cost_gpu = oce_result.device_accumulation.total_entropies_d
                #     else:
                #         host_costs = (
                #             oce_result
                #             if isinstance(oce_result, np.ndarray)
                #             else getattr(oce_result, "host_scores", None)
                #         )
                #         if (
                #             host_costs is None
                #             and hasattr(oce_result, "accumulation")
                #             and oce_result.accumulation is not None
                #         ):
                #             host_costs = getattr(
                #                 oce_result.accumulation, "total_entropies", None
                #             )
                #         if host_costs is not None:
                #             host_costs = np.ascontiguousarray(
                #                 np.asarray(host_costs, dtype=np.float32).reshape(-1)
                #             )
                #             if host_costs.shape[0] != int(self.samples):
                #                 raise ValueError(
                #                     "OCE scorer host costs must have one value per sample."
                #                 )
                #             owned_oce_cost_gpu = _cuda.mem_alloc(host_costs.nbytes)  # type: ignore[attr-defined]
                #             _cuda.memcpy_htod(owned_oce_cost_gpu, host_costs)  # type: ignore[attr-defined]
                #             oce_cost_gpu = owned_oce_cost_gpu
                #     if oce_cost_gpu is not None:
                #         add_sample_costs_func(
                #             u_weight_gpu,
                #             oce_cost_gpu,
                #             self.samples,
                #             block=block_1d,
                #             grid=grid_1d,
                #         )
                #     if self.debug:
                #         oce_queries = getattr(oce_result, "num_unique_queries", 0)
                #         print(
                #             f"[MPPI] OCE entropy_space={entropy_space} device evaluations={oce_queries}"
                #         )
                #     record_timing("oce_scoring", sync_cuda=True)

                combined_costs_gpu = _cuda.mem_alloc(sample_cost_bytes)  # type: ignore[attr-defined]
                _cuda.memcpy_dtod(combined_costs_gpu, u_weight_gpu, sample_cost_bytes)  # type: ignore[attr-defined]

                # Capture raw costs before weight transform
                raw_costs_host = np.zeros(self.samples, dtype=np.float32)
                _cuda.memcpy_dtoh(raw_costs_host, combined_costs_gpu)  # type: ignore[attr-defined]
                record_timing("raw_cost_snapshot")

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
                record_timing("min_weight_kernel", sync_cuda=True)

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
                record_timing("calculate_weights_kernel", sync_cuda=True)
                self.last_static_filter_all_invalid = False
                self.last_dynamic_filter_all_invalid = False
                if num_polygon_obstacles > 0:
                    pre_static_weight_gpu = _cuda.mem_alloc(sample_cost_bytes)  # type: ignore[attr-defined]
                    _cuda.memcpy_dtod(
                        pre_static_weight_gpu, u_weight_gpu, sample_cost_bytes
                    )  # type: ignore[attr-defined]
                    _cuda.memset_d8(
                        u_weight_total_gpu, 0, np.dtype(np.float32).itemsize
                    )  # type: ignore[attr-defined]
                    filter_static_collision_weights_func(
                        self.samples,
                        u_weight_gpu,
                        component_costs_gpu,
                        np.float32(0.5 * self.static_collision_cost),
                        u_weight_total_gpu,
                        block=block_1d,
                        grid=grid_1d,
                    )
                    record_timing("static_weight_filter_kernel", sync_cuda=True)
                    static_weight_total = np.zeros(1, dtype=np.float32)
                    _cuda.memcpy_dtoh(static_weight_total, u_weight_total_gpu)  # type: ignore[attr-defined]
                    record_timing("static_weight_total_download")
                    if float(static_weight_total[0]) <= 0.0:
                        pre_static_weights_host = np.zeros(
                            self.samples, dtype=np.float32
                        )
                        _cuda.memcpy_dtoh(
                            pre_static_weights_host, pre_static_weight_gpu
                        )  # type: ignore[attr-defined]
                        restored_total = np.array(
                            [np.sum(pre_static_weights_host, dtype=np.float32)],
                            dtype=np.float32,
                        )
                        if float(restored_total[0]) > 0.0:
                            _cuda.memcpy_dtod(
                                u_weight_gpu,
                                pre_static_weight_gpu,
                                sample_cost_bytes,
                            )  # type: ignore[attr-defined]
                            _cuda.memcpy_htod(
                                u_weight_total_gpu, restored_total
                            )  # type: ignore[attr-defined]
                            self.last_static_filter_all_invalid = True
                        record_timing("static_weight_filter_restore")
                if dynamic_costs_gpu is not None:
                    pre_dynamic_weight_gpu = _cuda.mem_alloc(sample_cost_bytes)  # type: ignore[attr-defined]
                    _cuda.memcpy_dtod(
                        pre_dynamic_weight_gpu, u_weight_gpu, sample_cost_bytes
                    )  # type: ignore[attr-defined]
                    pre_dynamic_weight_total = np.zeros(1, dtype=np.float32)
                    _cuda.memcpy_dtoh(pre_dynamic_weight_total, u_weight_total_gpu)  # type: ignore[attr-defined]
                    _cuda.memset_d8(
                        u_weight_total_gpu, 0, np.dtype(np.float32).itemsize
                    )  # type: ignore[attr-defined]
                    filter_dynamic_collision_weights_func(
                        self.samples,
                        u_weight_gpu,
                        dynamic_costs_gpu,
                        np.float32(0.5 * self.dynamic_collision_cost),
                        u_weight_total_gpu,
                        block=block_1d,
                        grid=grid_1d,
                    )
                    record_timing("dynamic_weight_filter_kernel", sync_cuda=True)
                    dynamic_weight_total = np.zeros(1, dtype=np.float32)
                    _cuda.memcpy_dtoh(dynamic_weight_total, u_weight_total_gpu)  # type: ignore[attr-defined]
                    record_timing("dynamic_weight_total_download")
                    if (
                        float(dynamic_weight_total[0]) <= 0.0
                        and float(pre_dynamic_weight_total[0]) > 0.0
                    ):
                        _cuda.memcpy_dtod(
                            u_weight_gpu,
                            pre_dynamic_weight_gpu,
                            sample_cost_bytes,
                        )  # type: ignore[attr-defined]
                        _cuda.memcpy_htod(
                            u_weight_total_gpu,
                            pre_dynamic_weight_total,
                        )  # type: ignore[attr-defined]
                        self.last_dynamic_filter_all_invalid = True
                        record_timing("dynamic_weight_filter_restore")

                # Accumulate weighted disturbances in unconstrained control space.
                # This avoids biasing saturated steering commands back toward zero.
                u_limits_host = self.optimization_args["u_limits"][0]
                if self.dynamics_type == "holonomic":
                    u_nom_z_host = u_nom_host.copy()
                else:
                    unit_u_nom = np.zeros_like(u_nom_host)
                    unit_u_nom[:, 0] = np.clip(
                        u_nom_host[:, 0]
                        / max(float(u_limits_host[0]), np.finfo(np.float32).eps),
                        -1.0 + 1e-5,
                        1.0 - 1e-5,
                    )
                    unit_u_nom[:, 1] = np.clip(
                        u_nom_host[:, 1]
                        / max(float(u_limits_host[1]), np.finfo(np.float32).eps),
                        -1.0 + 1e-5,
                        1.0 - 1e-5,
                    )
                    u_nom_z_host = np.arctanh(unit_u_nom).astype(np.float32)

                u_mppi_gpu = _cuda.mem_alloc(u_nom_host.nbytes)  # type: ignore[attr-defined]
                _cuda.memcpy_htod(u_mppi_gpu, u_nom_z_host)  # type: ignore[attr-defined]
                record_timing("control_prep_upload")
                calculate_mppi_control_func(
                    self.samples,
                    u_nom_gpu,
                    u_dist_gpu,
                    np.int32(num_controls_timesteps),
                    u_weight_gpu,
                    u_weight_total_gpu,
                    self.optimization_args_gpu,
                    u_mppi_gpu,
                    block=block_1d,
                    grid=grid_1d,
                )
                record_timing("control_kernel", sync_cuda=True)

                # Copy back
                _cuda.memcpy_dtoh(u_mppi_host, u_mppi_gpu)  # type: ignore[attr-defined]
                if self.dynamics_type == "holonomic":
                    magnitudes = np.linalg.norm(u_mppi_host[:, :2], axis=1)
                    over_limit = magnitudes > float(u_limits_host[0])
                    u_mppi_host[over_limit, :2] *= (
                        float(u_limits_host[0]) / magnitudes[over_limit]
                    )[:, np.newaxis]
                else:
                    u_mppi_host[:, 0] = u_limits_host[0] * np.tanh(
                        u_mppi_host[:, 0]
                    )
                    u_mppi_host[:, 1] = u_limits_host[1] * np.tanh(
                        u_mppi_host[:, 1]
                    )
                record_timing("control_download")

                u_dist_raw_host = np.zeros(
                    self.samples * num_controls_timesteps * num_control_elements,
                    dtype=np.float32,
                )
                _cuda.memcpy_dtoh(u_dist_raw_host, u_dist_gpu)  # type: ignore[attr-defined]
                u_dist_host_reshaped = u_dist_raw_host.reshape(
                    (self.samples, num_controls_timesteps, num_control_elements)
                )
                record_timing("disturbance_download")
                _cuda.memcpy_dtoh(u_weights_host, u_weight_gpu)  # type: ignore[attr-defined]
                record_timing("weights_download")
                if rollout_states_gpu is not None and self.debug:
                    rollout_states_host = np.zeros(
                        (self.samples, num_controls_timesteps, 4), dtype=np.float32
                    )
                    _cuda.memcpy_dtoh(rollout_states_host, rollout_states_gpu)  # type: ignore[attr-defined]
                    self.last_rollout_states = rollout_states_host
                else:
                    self.last_rollout_states = None
                record_timing("rollout_states_download")
                combined_costs_host = np.zeros(self.samples, dtype=np.float32)
                _cuda.memcpy_dtoh(combined_costs_host, combined_costs_gpu)  # type: ignore[attr-defined]
                oce_costs_host = np.zeros(self.samples, dtype=np.float32)
                if oce_cost_gpu is not None:
                    _cuda.memcpy_dtoh(oce_costs_host, oce_cost_gpu)  # type: ignore[attr-defined]
                component_costs_host = np.zeros(
                    (int(self.samples), 5), dtype=np.float32
                )
                if component_costs_gpu is not None:
                    _cuda.memcpy_dtoh(component_costs_host, component_costs_gpu)  # type: ignore[attr-defined]
                dynamic_costs_host = np.zeros(self.samples, dtype=np.float32)
                if dynamic_costs_gpu is not None:
                    _cuda.memcpy_dtoh(dynamic_costs_host, dynamic_costs_gpu)  # type: ignore[attr-defined]
                    component_costs_host[:, 4] = dynamic_costs_host
                occupancy_costs_host = np.zeros(self.samples, dtype=np.float32)
                if occupancy_costs_gpu is not None:
                    _cuda.memcpy_dtoh(occupancy_costs_host, occupancy_costs_gpu)  # type: ignore[attr-defined]
                record_timing("cost_components_download")

                # Diagnostics
                weight_sum = float(np.sum(u_weights_host) + 1e-12)
                ess = (weight_sum**2) / (float(np.sum(u_weights_host**2)) + 1e-12)
                self.last_ess = ess
                self.last_total_costs = combined_costs_host
                self.last_oce_costs = oce_costs_host
                self.last_rollout_costs = combined_costs_host - oce_costs_host
                self.last_dynamic_costs = dynamic_costs_host
                self.last_cost_components = {
                    "state": component_costs_host[:, 0],
                    "control": component_costs_host[:, 1],
                    "static_obstacle": component_costs_host[:, 2],
                    "visibility": component_costs_host[:, 3],
                    "dynamic_obstacle": component_costs_host[:, 4],
                    "oce": oce_costs_host,
                    "occupancy": occupancy_costs_host,
                }
                self.last_dynamic_clearance_summary = None
                # if (
                #     self.dynamic_clearance_diagnostics
                #     and dynamic_actors
                #     and self.last_rollout_states is not None
                # ):
                #     self.last_dynamic_clearance_summary = (
                #         _dynamic_rollout_clearance_summary(
                #             self.last_rollout_states,
                #             dynamic_actor_states_host,
                #             dynamic_actor_geometry_host,
                #             self.ego_collision_geometry,
                #         )
                #     )
                # self.last_weight_summary = _summarize_weights_and_costs(
                #     weights=u_weights_host,
                #     total_costs=combined_costs_host,
                #     dynamic_costs=dynamic_costs_host,
                #     samples=int(self.samples),
                #     dynamic_collision_cost=self.dynamic_collision_cost,
                # )
                # self.last_weight_summary["static_filter_all_invalid"] = bool(
                #     self.last_static_filter_all_invalid
                # )
                # self.last_weight_summary["dynamic_filter_all_invalid"] = bool(
                #     self.last_dynamic_filter_all_invalid
                # )
                # if combined_costs_host.size:
                #     self.last_cost_min = float(np.min(combined_costs_host))
                #     self.last_cost_max = float(np.max(combined_costs_host))
                #     self.last_cost_mean = float(np.mean(combined_costs_host))
                # if self.debug:
                #     pct = (ess / self.samples) * 100.0
                #     print(
                #         f"[MPPI] dt={dt:.3f} cost(min/mean/max)=({self.last_cost_min:.2f}/{self.last_cost_mean:.2f}/{self.last_cost_max:.2f}) ESS={ess:.1f}/{self.samples} ({pct:.1f}%)"
                #     )
                #     component_means = {
                #         key: float(np.mean(value)) if np.size(value) else 0.0
                #         for key, value in self.last_cost_components.items()
                #     }
                #     print(f"[MPPI] cost components mean={component_means}")
                #     print(f"[MPPI] dynamic actors={self.last_dynamic_actor_debug}")
                #     print(f"[MPPI] weight summary={self.last_weight_summary}")
                #     if self.last_dynamic_clearance_summary is not None:
                #         print(
                #             "[MPPI] dynamic clearance "
                #             f"{self.last_dynamic_clearance_summary}"
                #         )
                # record_timing("diagnostics")

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
                    obstacles_gpu if not isinstance(obstacles_gpu, int) else None,
                    (
                        obstacle_states_gpu
                        if not isinstance(obstacle_states_gpu, int)
                        else None
                    ),
                    (
                        polygon_vertices_gpu
                        if not isinstance(polygon_vertices_gpu, int)
                        else None
                    ),
                    (
                        polygon_offsets_gpu
                        if not isinstance(polygon_offsets_gpu, int)
                        else None
                    ),
                    u_nom_gpu,
                    x_nom_gpu,
                    u_mppi_gpu,
                    u_weight_gpu,
                    u_dist_gpu,
                    rollout_states_gpu,
                    combined_costs_gpu,
                    component_costs_gpu,
                    pre_static_weight_gpu,
                    pre_dynamic_weight_gpu,
                    dynamic_costs_gpu,
                    dynamic_actor_states_gpu,
                    dynamic_actor_geometry_gpu,
                    occupancy_grids_gpu,
                    occupancy_costs_gpu,
                    owned_oce_cost_gpu,
                    u_weight_min_gpu,
                    u_weight_total_gpu,
                ]:
                    try:
                        if buf:
                            buf.free()
                    except Exception:
                        pass
                record_timing("cleanup")

        timing["total"] = perf_counter() - timing_start
        self.last_timing = dict(timing)
        if self.debug or timing["total"] > 1.0:
            timing_parts = " ".join(
                f"{name}={elapsed * 1000.0:.2f}ms"
                for name, elapsed in timing.items()
                if name != "total"
            )
            print(
                "[MPPI timing] "
                f"total={timing['total'] * 1000.0:.2f}ms "
                f"sync_cuda={bool(self.debug)} "
                f"samples={int(self.samples)} horizon={int(u_nom.shape[0])} "
                f"obstacles={len(obstacles)} "
                f"{timing_parts}"
            )
            if self.last_dynamic_actor_debug:
                print(f"[MPPI dynamic actors] {self.last_dynamic_actor_debug}")
            if self.last_weight_summary:
                print(f"[MPPI weight summary] {self.last_weight_summary}")
            if self.last_dynamic_clearance_summary:
                print(f"[MPPI dynamic clearance] {self.last_dynamic_clearance_summary}")
        return u_mppi_host, u_dist_host_reshaped, u_weights_host
