#!/usr/bin/env python3
"""
Benchmark script to compare PyCUDA vs Warp MPPI performance.
"""

import time
import numpy as np
from typing import Optional

# Try to import both implementations
try:
    from warp_mppi.legacy.mppi_pycuda import MPPI as PyCudaMPPI

    PYCUDA_AVAILABLE = True
    print("✓ PyCUDA implementation available")
except ImportError as e:
    print(f"✗ PyCUDA implementation not available: {e}")
    PYCUDA_AVAILABLE = False

try:
    from warp_mppi.mppi_warp import WarpMPPI

    WARP_AVAILABLE = True
    print("✓ Warp implementation available")
except ImportError as e:
    print(f"✗ Warp implementation not available: {e}")
    WARP_AVAILABLE = False


class DummyVehicle:
    """Dummy vehicle class for testing."""

    def __init__(self):
        self.L = 2.5  # wheelbase length


def create_test_data(horizon=10, samples=1000):
    """Create test data for benchmarking."""
    # Simple test case
    costmap = np.random.rand(50, 50).astype(np.float32)
    origin = (0.0, 0.0)
    resolution = 0.1
    x_init = [0.0, 0.0, 1.0, 0.0]  # x, y, v, theta
    x_goal = [10.0, 0.0, 1.0, 0.0]

    # Linear nominal trajectory
    x_nom = np.array([[i, 0.0, 1.0, 0.0] for i in range(horizon + 1)])
    u_nom = np.array([[0.0, 0.0] for _ in range(horizon)])

    # Some obstacles
    actors = [[5.0, 2.0, 1.0], [7.0, -2.0, 1.0]]
    dt = 0.1

    return costmap, origin, resolution, x_init, x_goal, x_nom, u_nom, actors, dt


def benchmark_controller(controller, test_data, num_runs=5, warmup_runs=1):
    """Benchmark a controller implementation."""
    costmap, origin, resolution, x_init, x_goal, x_nom, u_nom, actors, dt = test_data

    print(f"Running {warmup_runs} warmup iterations...")

    # Warmup runs
    for _ in range(warmup_runs):
        try:
            controller.find_control(
                costmap=costmap,
                origin=origin,
                resolution=resolution,
                x_init=x_init,
                x_goal=x_goal,
                x_nom=x_nom,
                u_nom=u_nom,
                actors=actors,
                dt=dt,
            )
        except Exception as e:
            print(f"Warmup failed: {e}")
            return None, None, None

    print(f"Running {num_runs} timed iterations...")

    # Timed runs
    times = []
    for i in range(num_runs):
        start_time = time.perf_counter()
        try:
            u_opt, u_samples, weights = controller.find_control(
                costmap=costmap,
                origin=origin,
                resolution=resolution,
                x_init=x_init,
                x_goal=x_goal,
                x_nom=x_nom,
                u_nom=u_nom,
                actors=actors,
                dt=dt,
            )
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            print(f"  Run {i+1}: {times[-1]*1000:.2f} ms")

        except Exception as e:
            print(f"Run {i+1} failed: {e}")
            return None, None, None

    return times, u_opt, (u_samples, weights)


def main():
    """Main benchmark function."""
    if not (PYCUDA_AVAILABLE or WARP_AVAILABLE):
        print("No implementations available for testing!")
        return

    # Configuration
    samples = 1000
    horizon = 10
    num_runs = 5

    vehicle = DummyVehicle()
    test_data = create_test_data(horizon=horizon, samples=samples)

    print(f"\n{'='*60}")
    print(f"MPPI Performance Benchmark")
    print(f"{'='*60}")
    print(f"Samples: {samples}")
    print(f"Horizon: {horizon}")
    print(f"Runs: {num_runs}")
    print(f"{'='*60}")

    results = {}

    # Test PyCUDA implementation
    if PYCUDA_AVAILABLE:
        print(f"\n🔥 Testing PyCUDA Implementation")
        print("-" * 40)
        try:
            pycuda_controller = PyCudaMPPI(
                vehicle=vehicle,
                samples=samples,
                seed=42,
                u_limits=[3.0, 0.5],
                u_dist_limits=[0.5, 0.1],
                M=1.0,
                Q=[1.0, 1.0, 0.1, 0.1],
                Qf=[10.0, 10.0, 1.0, 1.0],
                R=[0.1, 0.1],
                method="Nominal",
                c_lambda=1.0,
                scan_range=10.0,
                debug=False,
            )

            times, u_opt, extra = benchmark_controller(
                pycuda_controller, test_data, num_runs
            )
            if times:
                results["PyCUDA"] = {
                    "times": times,
                    "mean": np.mean(times),
                    "std": np.std(times),
                    "min": np.min(times),
                    "max": np.max(times),
                }
                print(f"✓ PyCUDA completed successfully")
            else:
                print("✗ PyCUDA benchmark failed")

        except Exception as e:
            print(f"✗ PyCUDA initialization failed: {e}")

    # Test Warp implementation
    if WARP_AVAILABLE:
        print(f"\n🚀 Testing Warp Implementation")
        print("-" * 40)
        try:
            warp_controller = WarpMPPI(
                vehicle=vehicle,
                samples=samples,
                seed=42,
                u_limits=[3.0, 0.5],
                u_dist_limits=[0.5, 0.1],
                M=1.0,
                Q=[1.0, 1.0, 0.1, 0.1],
                Qf=[10.0, 10.0, 1.0, 1.0],
                R=[0.1, 0.1],
                method="Nominal",
                c_lambda=1.0,
                scan_range=10.0,
                debug=False,
            )

            times, u_opt, extra = benchmark_controller(
                warp_controller, test_data, num_runs
            )
            if times:
                results["Warp"] = {
                    "times": times,
                    "mean": np.mean(times),
                    "std": np.std(times),
                    "min": np.min(times),
                    "max": np.max(times),
                }
                print(f"✓ Warp completed successfully")
            else:
                print("✗ Warp benchmark failed")

        except Exception as e:
            print(f"✗ Warp initialization failed: {e}")
            import traceback

            traceback.print_exc()

    # Results summary
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")

    if results:
        for impl_name, data in results.items():
            print(f"\n{impl_name}:")
            print(f"  Mean: {data['mean']*1000:.2f} ± {data['std']*1000:.2f} ms")
            print(f"  Range: [{data['min']*1000:.2f}, {data['max']*1000:.2f}] ms")
            print(f"  Frequency: ~{1.0/data['mean']:.1f} Hz")

        if len(results) == 2:
            pycuda_mean = results["PyCUDA"]["mean"]
            warp_mean = results["Warp"]["mean"]
            speedup = warp_mean / pycuda_mean
            print(f"\n🏁 Performance Comparison:")
            print(f"   PyCUDA: {pycuda_mean*1000:.2f} ms")
            print(f"   Warp:   {warp_mean*1000:.2f} ms")
            if speedup > 1:
                print(f"   PyCUDA is {speedup:.2f}x FASTER than Warp")
            else:
                print(f"   Warp is {1/speedup:.2f}x FASTER than PyCUDA")
    else:
        print("No successful benchmarks to report.")


if __name__ == "__main__":
    main()
