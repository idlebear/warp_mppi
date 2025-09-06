#!/usr/bin/env python3
"""
Simple test script to verify the warp_mppi package installation.
"""


def test_import():
    """Test basic import functionality."""
    try:
        from warp_mppi import MPPI

        print("✓ MPPI imported successfully!")

        # Test visibility methods
        methods = list(MPPI.visibility_methods.keys())
        print(f"✓ Available visibility methods: {methods}")

        # Test data types
        print(f"✓ Optimization dtype has {len(MPPI.optimization_dtype.names)} fields")
        print(f"✓ Costmap dtype has {len(MPPI.costmap_dtype.names)} fields")

        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_warp():
    """Test Warp import."""
    try:
        import warp

        print(f"✓ Warp {warp.__version__} imported successfully!")
        return True
    except ImportError as e:
        print(f"✗ Warp import failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing warp_mppi package installation...")
    print("=" * 50)

    success = True
    success &= test_warp()
    success &= test_import()

    print("=" * 50)
    if success:
        print("🎉 All tests passed! Package is ready for development.")
    else:
        print("❌ Some tests failed. Check the installation.")
