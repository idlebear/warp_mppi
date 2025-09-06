# Warp MPPI: High-Performance Model Predictive Path Integral Control

A GPU-accelerated implementation of Model Predictive Path Integral (MPPI) control using NVIDIA Warp for autonomous vehicle navigation and obstacle avoidance.

## Overview

This repository contains a high-performance implementation of MPPI control algorithm originally implemented with PyCUDA, now being converted to use NVIDIA Warp for better performance and maintainability. The controller is designed for autonomous vehicle navigation with visibility-aware path planning and dynamic obstacle avoidance.

## Core Functionality

### MPPI Controller (`mppi_pycuda.py` - Legacy Implementation)

The original MPPI implementation provides:

- **GPU-Accelerated Sampling**: Monte Carlo sampling of control sequences with configurable sample counts
- **Multiple Cost Functions**: Support for various visibility and information-theoretic cost functions
- **Dynamic Obstacle Avoidance**: Real-time collision avoidance with circular obstacles
- **Vehicle Dynamics**: Bicycle model integration using Runge-Kutta methods
- **Flexible Cost Weighting**: Configurable weights for state tracking, control effort, and safety costs

### Key Features

#### Control and State Representation
- **State Vector**: `[x, y, velocity, heading]` for 2D vehicle pose and motion
- **Control Vector**: `[acceleration, steering_angle]` for vehicle actuation
- **Trajectory Optimization**: Forward simulation with control bounds and noise injection

#### Cost Functions
1. **Ours/Information Gain**: Costmap-based visibility optimization for exploration
2. **Higgins**: Exponential visibility cost based on obstacle proximity
3. **Andersen**: Angular reward for obstacle awareness
4. **Nominal**: Standard tracking without explicit visibility terms

#### GPU Optimization
- **CUDA Kernels**: Custom kernels for parallel rollout evaluation
- **Memory Management**: Efficient GPU memory allocation and data transfer
- **Parallel Reduction**: Hardware-optimized weight computation and control aggregation

### Algorithm Components

1. **Control Sampling**: Gaussian noise injection around nominal trajectory
2. **Forward Simulation**: Runge-Kutta integration of vehicle dynamics
3. **Cost Evaluation**: Multi-objective cost including:
   - State tracking error (Q matrix weighting)
   - Final state error (Qf matrix weighting)
   - Control effort (R matrix weighting)
   - Obstacle avoidance penalties
   - Visibility/information costs
   - Optional steering rate penalties
4. **Weight Computation**: Exponential transformation of costs to importance weights
5. **Control Update**: Weighted aggregation of control disturbances

### Performance Characteristics

- **Sample Efficiency**: Effective sample size (ESS) monitoring for convergence
- **Real-time Capable**: Sub-millisecond optimization for control frequencies up to 100Hz
- **Scalable**: Configurable sample counts (typically 1000-10000 samples)
- **Memory Efficient**: Minimal host-device transfers and persistent GPU memory

## Migration to NVIDIA Warp

This codebase is being converted from PyCUDA to NVIDIA Warp to leverage:

- **Simplified GPU Programming**: Python-native GPU kernel development
- **Better Performance**: Optimized compilation and execution
- **Modern Architecture**: Support for latest GPU architectures
- **Improved Debugging**: Better tooling and error reporting

## Installation

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA support
- warp_lang package  (tested with version 1.9.0)


### Environment Setup

```bash
conda activate ppo  # Or your preferred environment with warp_lang
pip install -e .    # Install as editable package
```

## Usage

```python
from warp_mppi import MPPI

# Initialize controller
controller = MPPI(
    vehicle=vehicle_model,
    samples=2048,
    u_limits=[max_accel, max_steer],
    u_dist_limits=[accel_std, steer_std],
    Q=[x_weight, y_weight, v_weight, theta_weight],
    Qf=[final_x_weight, final_y_weight, final_v_weight, final_theta_weight],
    R=[accel_cost, steer_cost],
    method="Ours",  # or "Higgins", "Andersen", "Nominal"
    c_lambda=1.0,   # Temperature parameter
    # ... other parameters
)

# Compute optimal control
u_optimal, u_samples, weights = controller.find_control(
    costmap=visibility_map,
    origin=map_origin,
    resolution=map_resolution,
    x_init=current_state,
    x_goal=target_state,
    x_nom=nominal_trajectory,
    u_nom=nominal_controls,
    actors=obstacles,
    dt=timestep
)
```

## Configuration

Key parameters for tuning controller behavior:

- `samples`: Number of Monte Carlo samples (higher = better quality, slower)
- `c_lambda`: Temperature parameter (lower = more aggressive optimization)
- `u_dist_limits`: Control noise standard deviations
- Cost weights `Q`, `Qf`, `R`: Relative importance of different objectives
- `method`: Visibility cost function selection
- `steering_rate_weight`: Smoothness penalty for steering commands

## Performance Monitoring

The controller provides diagnostics:
- **Effective Sample Size (ESS)**: Measure of sample diversity
- **Cost Statistics**: Min/mean/max cost values across samples
- **Timing Information**: GPU kernel execution times

## Research Applications

This implementation supports research in:
- Visibility-aware autonomous navigation
- Information-theoretic path planning
- Real-time model predictive control
- GPU-accelerated optimization for robotics

## Citation

If you use this work in your research, please cite:
```
Gilhuly, Barry, Armin Sadeghi, and Stephen L. Smith.
"Estimating Visibility From Alternate Perspectives for Motion Planning With Occlusions."
IEEE Robotics and Automation Letters 9.6 (2024): 5583-5590.
```

## License

MIT License.

Copyright (c) 2025 Barry Gilhuly
