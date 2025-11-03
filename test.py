#!/usr/bin/env python3
"""
Quick test runner for rotary-insight project.
For comprehensive testing, use: python run_tests.py
"""

import sys

try:
    # Try to run with pytest
    import pytest

    print("Running tests with pytest...")
    sys.exit(pytest.main(["tests/", "-v"]))
except ImportError:
    # Fall back to manual tests
    print("pytest not installed, running manual tests...")
    print("(Install pytest for comprehensive testing: pip install pytest)")
    print()

    from tests.test_pu_dataset import run_manual_tests

    try:
        run_manual_tests()
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Tests failed with: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
