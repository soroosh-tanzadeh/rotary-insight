#!/usr/bin/env python3
"""
Comprehensive test runner for rotary-insight project.
Can run tests using pytest or manual test runners.
"""

import sys
import os
import argparse


def run_pytest():
    """Run tests using pytest."""
    try:
        import pytest

        # Run pytest with configuration
        args = [
            "tests/",
            "-v",
            "--tb=short",
        ]

        return pytest.main(args)
    except ImportError:
        print("ERROR: pytest not installed. Install it with: pip install pytest")
        return 1


def run_manual_tests():
    """Run manual test runners."""
    print("=" * 80)
    print("Running Manual Tests")
    print("=" * 80)

    try:
        from tests.test_pu_dataset import run_manual_tests as run_pu_tests

        run_pu_tests()
        return 0
    except Exception as e:
        print(f"ERROR: Failed to run manual tests: {e}")
        import traceback

        traceback.print_exc()
        return 1


def run_specific_test(test_name):
    """Run a specific test."""
    try:
        import pytest

        args = [f"tests/{test_name}", "-v"]
        return pytest.main(args)
    except ImportError:
        print("ERROR: pytest not installed")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Run tests for rotary-insight")
    parser.add_argument(
        "--mode",
        choices=["pytest", "manual", "all"],
        default="pytest",
        help="Test mode to use (default: pytest)",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Specific test file to run (e.g., test_pu_dataset.py)",
    )
    parser.add_argument(
        "--markers",
        type=str,
        help="Pytest markers to filter tests (e.g., 'not slow')",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting (requires pytest-cov)",
    )

    args = parser.parse_args()

    if args.test:
        return run_specific_test(args.test)

    if args.mode == "pytest" or args.mode == "all":
        print("\n" + "=" * 80)
        print("Running Pytest Suite")
        print("=" * 80 + "\n")

        try:
            import pytest

            pytest_args = ["tests/", "-v"]

            if args.markers:
                pytest_args.extend(["-m", args.markers])

            if args.coverage:
                pytest_args.extend(
                    [
                        "--cov=datasets",
                        "--cov=models",
                        "--cov=preprocessings",
                        "--cov=utils",
                        "--cov-report=html",
                        "--cov-report=term",
                    ]
                )

            result = pytest.main(pytest_args)

            if result != 0:
                return result
        except ImportError:
            print("WARNING: pytest not installed, skipping pytest tests")
            print("Install with: pip install pytest pytest-cov")

    if args.mode == "manual" or args.mode == "all":
        print("\n" + "=" * 80)
        print("Running Manual Tests")
        print("=" * 80 + "\n")

        result = run_manual_tests()
        if result != 0:
            return result

    print("\n" + "=" * 80)
    print("All Tests Completed Successfully!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
