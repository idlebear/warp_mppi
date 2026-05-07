# OCE MPPI Integration Plan

## Scope

Only the PyCUDA legacy implementation is being updated now:

- `warp_mppi/legacy/mppi_pycuda.py`
- `warp_mppi/legacy/trajectory_eval_pycuda.py`

Do not update `warp_mppi/mppi_warp.py` in this pass. The Warp backend is out of
date and will be reconciled separately.

## Current Merge Step

Completed in `mppi_pycuda.py`:

1. Keep the destination file's retained CUDA context management.
2. Preserve the existing MPPI public API.
3. Split obstacles into:
   - static obstacles, evaluated during the rollout kernel
   - dynamic actors, evaluated after rollout materialization
4. Materialize rollout states on device only when dynamic actors are present.
5. Apply dynamic actor collision costs with a CUDA post-pass using three-circle
   collision geometry.
6. Add optional OCE weighting as a second-stage sample-cost term.
7. Keep `oce_data=None` as the default path so regular MPPI and MPPI+OCE can be
   compared by changing only the call payload.

The dynamic obstacle input accepts:

```python
{
    "states": np.ndarray,   # shape (T, 4), x/y/v/theta
    "extent": (half_length, half_width),
    "collision_buffer": 0.0,
    "static": False,
}
```

Legacy static rows are still accepted for compatibility.

## OCE Weighting Integration

The reusable OCE scene packing and entropy-kernel logic from the
information-gain `trajectory_eval_gpu.py` is vendored into
`legacy/trajectory_eval_pycuda.py`. Its planning dependencies live under
`legacy/planning/` and use package-local imports.

`MPPI.find_control(..., oce_data=None)` accepts either an `OCESceneInputs` scene
or a dict:

```python
{
    "enabled": True,
    "scene": oce_scene,
    "oce_config": oce_config,
    "entropy_space": "kde",
    "discount": 1.0,
    "eps": 1e-9,
    "scorer": optional_custom_scorer,
}
```

When enabled, MPPI materializes rollout states on device, calls
`score_oce_scene_rollouts_device`, and adds the returned per-sample OCE entropy
costs before weight normalization. The OCE implementation is selected through
`entropy_space` or `oce_config.oce_entropy_space`; supported CUDA paths include
the partition-info/kde, kde-spatial, kde-sigma-point,
kde-separability-logdet, and kde-separability-trace families.

For non-CUDA or experimental OCE methods, pass `scorer`/`score_func` in
`oce_data`. The scorer receives the same keyword arguments as
`score_oce_scene_rollouts_device` and may return either an `OCEScoreResult`
with device accumulation buffers or a host `np.ndarray` of per-sample costs.

Remaining work:

1. Add tests that run without CUDA by validating:
   - input packing
   - device-call argument construction
   - cost-addition ordering with mocked device buffers
2. Add CUDA integration tests separately for machines with GPU access.
3. Add a built-in CPU fallback scorer for methods that are not CUDA-backed,
   such as `kde-mc`, if those need to run in the MPPI loop without a custom
   scorer.

## Coordinate Contract

OCE rollout states must be in the same world frame used by MPPI. Any SDD
scene-to-simulator y-axis conversion must happen before MPPI/OCE scoring.

## Non-goals For This Pass

- Porting OCE kernels to Warp.
- Changing the Warp backend API.
- Replacing the PyCUDA context-management code in `mppi_pycuda.py`.
