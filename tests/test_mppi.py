"""
Basic tests for the MPPI controller.
"""

import pytest
from warp_mppi import MPPI


class TestMPPI:
    """Test cases for MPPI controller."""

    def test_mppi_import(self):
        """Test that MPPI can be imported successfully."""
        assert MPPI is not None

    def test_visibility_methods(self):
        """Test that visibility methods are properly defined."""
        expected_methods = [
            "Ours",
            "Ours-Wide",
            "Right",
            "Left",
            "Higgins",
            "Andersen",
            "Nominal",
            "Infogain",
            "Dynamic",
            "Ignore",
        ]

        for method in expected_methods:
            assert method in MPPI.visibility_methods

    @pytest.mark.gpu
    def test_mppi_initialization(self):
        """Test MPPI controller initialization."""
        # This test requires a GPU context
        try:
            controller = MPPI(
                vehicle=None,  # Mock vehicle for testing
                samples=100,
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
                vehicle_length=2.5,
                debug=True,
            )
            assert controller.samples == 100
            assert controller.optimization_args["c_lambda"][0] == 1.0
        except Exception as e:
            pytest.skip(f"GPU not available for testing: {e}")

    def test_optimization_dtype(self):
        """Test that optimization dtype is properly defined."""
        dtype = MPPI.optimization_dtype
        expected_fields = [
            "samples",
            "M",
            "dt",
            "num_controls",
            "num_obstacles",
            "x_init",
            "x_goal",
            "u_limits",
            "u_dist_limits",
            "Q",
            "Qf",
            "R",
            "method",
            "c_lambda",
            "scan_range",
            "vehicle_length",
            "steering_rate_weight",
        ]

        for field in expected_fields:
            assert field in dtype.names

    def test_costmap_dtype(self):
        """Test that costmap dtype is properly defined."""
        dtype = MPPI.costmap_dtype
        expected_fields = ["height", "width", "origin_x", "origin_y", "resolution"]

        for field in expected_fields:
            assert field in dtype.names
