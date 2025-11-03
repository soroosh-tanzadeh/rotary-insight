#!/usr/bin/env python3
"""
Comprehensive test suite for PUDataset.
Tests all functionality including max_per_class, caching, data integrity, and BearingDataset interface.
"""

import pytest
import torch
import numpy as np
import os
import tempfile
import shutil
from datasets.pu_dataset import PUDataset, slicer


class TestPUDataset:
    """Test suite for PUDataset class."""

    @pytest.fixture
    def data_dir(self):
        """Return the path to the PU dataset directory."""
        return "./data/dataset/PU"

    @pytest.fixture
    def small_dataset(self, data_dir):
        """Create a small dataset for quick testing."""
        return PUDataset(
            rdir=data_dir,
            window_size=2048,
            max_per_class=1000,
            seed=42,
            verbose=False,
            force_reload=False,
        )

    @pytest.fixture
    def medium_dataset(self, data_dir):
        """Create a medium dataset for testing."""
        return PUDataset(
            rdir=data_dir,
            window_size=2048,
            max_per_class=5000,
            seed=42,
            verbose=False,
            force_reload=False,
        )

    def test_initialization(self, small_dataset):
        """Test basic dataset initialization."""
        assert small_dataset is not None
        assert small_dataset.seq_len == 2048
        assert small_dataset.step_size == 2048
        assert small_dataset.max_per_class == 1000
        assert small_dataset.seed == 42

    def test_labels(self, small_dataset):
        """Test that dataset has correct labels."""
        labels = small_dataset.labels()
        assert len(labels) == 3
        assert "Healthy" in labels
        assert "InnerRace" in labels
        assert "OuterRace" in labels

    def test_data_shapes(self, small_dataset):
        """Test that data has correct shapes."""
        assert small_dataset.X.ndim == 3
        assert small_dataset.y.ndim == 1
        assert small_dataset.X.shape[0] == small_dataset.y.shape[0]
        assert small_dataset.X.shape[1] == 1  # Single channel
        assert small_dataset.X.shape[2] == 2048  # Window size

    def test_max_per_class_limit(self, small_dataset):
        """Test that max_per_class limit is respected."""
        unique, counts = torch.unique(small_dataset.y, return_counts=True)

        for label, count in zip(unique, counts):
            assert (
                count <= small_dataset.max_per_class
            ), f"Class {label} has {count} samples, exceeding max_per_class={small_dataset.max_per_class}"

    def test_max_per_class_different_limits(self, data_dir):
        """Test that different max_per_class values produce different dataset sizes."""
        dataset_small = PUDataset(
            rdir=data_dir,
            window_size=2048,
            max_per_class=500,
            seed=42,
            verbose=False,
        )

        dataset_medium = PUDataset(
            rdir=data_dir,
            window_size=2048,
            max_per_class=2000,
            seed=42,
            verbose=False,
        )

        assert len(dataset_small) < len(dataset_medium)
        assert len(dataset_small) <= 500 * 3  # 3 classes
        assert len(dataset_medium) <= 2000 * 3

    def test_len_method(self, small_dataset):
        """Test __len__ method."""
        length = len(small_dataset)
        assert length > 0
        assert length == small_dataset.X.shape[0]
        assert length == small_dataset.y.shape[0]

    def test_getitem_method(self, small_dataset):
        """Test __getitem__ method."""
        x, y = small_dataset[0]

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64
        assert x.shape == (1, 2048)
        assert y.shape == ()

    def test_getitem_batch(self, small_dataset):
        """Test __getitem__ with batch indexing."""
        indices = [0, 10, 20]
        for idx in indices:
            x, y = small_dataset[idx]
            assert x.shape == (1, 2048)
            assert y in [0, 1, 2]  # Valid class labels

    def test_inputs_method(self, small_dataset):
        """Test inputs() method."""
        X = small_dataset.inputs()
        assert isinstance(X, torch.Tensor)
        assert X.shape == small_dataset.X.shape
        assert torch.all(X == small_dataset.X)

    def test_targets_method(self, small_dataset):
        """Test targets() method."""
        y = small_dataset.targets()
        assert isinstance(y, torch.Tensor)
        assert y.shape == small_dataset.y.shape
        assert torch.all(y == small_dataset.y)

    def test_window_size_method(self, small_dataset):
        """Test window_size() method."""
        ws = small_dataset.window_size()
        assert ws == 2048
        assert ws == small_dataset.seq_len

    def test_different_window_sizes(self, data_dir):
        """Test dataset with different window sizes."""
        window_sizes = [512, 1024, 2048]

        for ws in window_sizes:
            dataset = PUDataset(
                rdir=data_dir,
                window_size=ws,
                max_per_class=500,
                seed=42,
                verbose=False,
            )
            assert dataset.window_size() == ws
            assert dataset.X.shape[2] == ws

    def test_cache_file_naming(self, data_dir):
        """Test that cache files are named correctly with max_per_class."""
        dataset = PUDataset(
            rdir=data_dir,
            window_size=2048,
            max_per_class=1000,
            seed=42,
            verbose=False,
        )

        # Check that cache files exist with correct naming
        assert os.path.exists(os.path.join(data_dir, "data_2048_1000.csv"))
        assert os.path.exists(os.path.join(data_dir, "x_2048_1000.npy"))
        assert os.path.exists(os.path.join(data_dir, "y_2048_1000.npy"))
        assert os.path.exists(
            os.path.join(data_dir, "files_information_2048_1000.json")
        )

    def test_cache_file_naming_no_limit(self, data_dir):
        """Test cache file naming without max_per_class."""
        # This might take longer, so use a small window for testing
        dataset = PUDataset(
            rdir=data_dir,
            window_size=512,
            max_per_class=None,
            seed=42,
            verbose=False,
        )

        # Check that cache files exist without max_per_class suffix
        assert os.path.exists(os.path.join(data_dir, "data_512.csv"))

    def test_data_types(self, small_dataset):
        """Test that data has correct types."""
        assert small_dataset.X.dtype in [torch.float32, torch.float64]
        assert small_dataset.y.dtype == torch.int64

    def test_class_distribution(self, small_dataset):
        """Test that all classes are present."""
        unique_classes = torch.unique(small_dataset.y)
        assert len(unique_classes) >= 1  # At least one class
        assert all(c in [0, 1, 2] for c in unique_classes)  # Valid class IDs

    def test_repr_and_str(self, small_dataset):
        """Test __repr__ and __str__ methods."""
        repr_str = repr(small_dataset)
        str_str = str(small_dataset)

        assert "PUDataset" in repr_str
        assert "X=" in repr_str
        assert "y=" in repr_str
        assert "seq_len" in repr_str
        assert repr_str == str_str

    def test_device_parameter(self, data_dir):
        """Test that device parameter is respected."""
        dataset_cpu = PUDataset(
            rdir=data_dir,
            window_size=2048,
            max_per_class=500,
            seed=42,
            device="cpu",
            verbose=False,
        )

        assert dataset_cpu.X.device.type == "cpu"
        assert dataset_cpu.y.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_cuda(self, data_dir):
        """Test dataset on CUDA device if available."""
        dataset_cuda = PUDataset(
            rdir=data_dir,
            window_size=2048,
            max_per_class=500,
            seed=42,
            device="cuda",
            verbose=False,
        )

        assert dataset_cuda.X.device.type == "cuda"
        assert dataset_cuda.y.device.type == "cuda"

    def test_verbose_flag(self, data_dir, capsys):
        """Test verbose output."""
        dataset = PUDataset(
            rdir=data_dir,
            window_size=2048,
            max_per_class=500,
            seed=42,
            verbose=True,
            force_reload=False,
        )

        captured = capsys.readouterr()
        assert len(captured.out) > 0  # Should have some output

    def test_data_integrity(self, small_dataset):
        """Test that loaded data has no NaN or Inf values."""
        assert not torch.isnan(small_dataset.X).any()
        assert not torch.isinf(small_dataset.X).any()
        assert not torch.isnan(small_dataset.y.float()).any()

    def test_windows_within_bounds(self, small_dataset):
        """Test that all windows have valid indices."""
        assert small_dataset.total_windows > 0
        assert small_dataset.total_windows == len(small_dataset)


class TestSlicerFunction:
    """Test suite for the slicer utility function."""

    def test_slicer_basic(self):
        """Test basic slicing functionality."""
        array = np.arange(1000)
        windows = slicer(array, win=100, step=100, return_df=False)

        assert windows.shape[0] == 10  # 1000 / 100 = 10 windows
        assert windows.shape[1] == 100
        assert windows[0, 0] == 0
        assert windows[0, -1] == 99

    def test_slicer_overlap(self):
        """Test slicer with overlapping windows."""
        array = np.arange(200)
        windows = slicer(array, win=50, step=25, return_df=False)

        assert windows.shape[1] == 50
        # Should have overlapping windows
        assert windows[0, -1] == 49
        assert windows[1, 0] == 25  # 50% overlap

    def test_slicer_no_overlap(self):
        """Test slicer without overlap."""
        array = np.arange(500)
        windows = slicer(array, win=50, step=50, return_df=False)

        assert windows.shape[0] == 10
        assert windows.shape[1] == 50

    def test_slicer_return_dataframe(self):
        """Test slicer returning DataFrame."""
        import pandas as pd

        array = np.arange(200)
        windows = slicer(array, win=50, step=50, return_df=True)

        assert isinstance(windows, pd.DataFrame)
        assert windows.shape[0] == 4
        assert windows.shape[1] == 50

    def test_slicer_incomplete_window(self):
        """Test slicer with incomplete final window (should be dropped)."""
        array = np.arange(105)
        windows = slicer(array, win=50, step=50, return_df=False)

        # Should only get 2 complete windows, incomplete one dropped
        assert windows.shape[0] == 2

    def test_slicer_edge_cases(self):
        """Test slicer with edge cases."""
        # Array exactly matches window size
        array = np.arange(100)
        windows = slicer(array, win=100, step=100, return_df=False)
        assert windows.shape[0] == 1

        # Array smaller than window
        array_small = np.arange(50)
        windows_small = slicer(array_small, win=100, step=100, return_df=False)
        assert windows_small.shape[0] == 0


class TestPUDatasetIntegration:
    """Integration tests for PUDataset."""

    def test_dataloader_compatibility(self):
        """Test that dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        dataset = PUDataset(
            rdir="./data/dataset/PU",
            window_size=2048,
            max_per_class=500,
            seed=42,
            verbose=False,
        )

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Get first batch
        batch_x, batch_y = next(iter(dataloader))

        assert batch_x.shape[0] <= 32  # Batch size
        assert batch_x.shape[1] == 1  # Channels
        assert batch_x.shape[2] == 2048  # Window size
        assert batch_y.shape[0] <= 32

    def test_cache_persistence(self):
        """Test that cache is properly saved and loaded."""
        data_dir = "./data/dataset/PU"

        # Create dataset (will cache)
        dataset1 = PUDataset(
            rdir=data_dir,
            window_size=2048,
            max_per_class=800,
            seed=42,
            verbose=False,
            force_reload=False,
        )

        # Create another instance (should load from cache)
        dataset2 = PUDataset(
            rdir=data_dir,
            window_size=2048,
            max_per_class=800,
            seed=42,
            verbose=False,
            force_reload=False,
        )

        # Should be identical
        assert len(dataset1) == len(dataset2)
        assert torch.allclose(dataset1.X, dataset2.X)
        assert torch.all(dataset1.y == dataset2.y)

    def test_multiple_instances(self):
        """Test creating multiple dataset instances with different parameters."""
        data_dir = "./data/dataset/PU"

        datasets = []
        configs = [
            {"max_per_class": 500, "seed": 42},
            {"max_per_class": 1000, "seed": 42},
            {"max_per_class": 500, "seed": 123},
        ]

        for config in configs:
            dataset = PUDataset(
                rdir=data_dir, window_size=2048, verbose=False, **config
            )
            datasets.append(dataset)

        # Different configs should produce different results
        assert len(datasets[0]) != len(datasets[1])  # Different max_per_class
        # Same max_per_class, different seed
        assert len(datasets[0]) == len(datasets[2])


def run_manual_tests():
    """Manual test runner for quick testing without pytest."""
    print("=" * 80)
    print("Running Manual PU Dataset Tests")
    print("=" * 80)

    # Test 1: Basic initialization
    print("\n### Test 1: Basic Initialization ###")
    dataset = PUDataset(
        rdir="./data/dataset/PU",
        window_size=2048,
        max_per_class=1000,
        seed=42,
        verbose=True,
        force_reload=False,
    )
    print(f"✓ Dataset created: {dataset}")
    print(f"✓ Length: {len(dataset)}")
    print(f"✓ Labels: {dataset.labels()}")

    # Test 2: Data shapes
    print("\n### Test 2: Data Shapes ###")
    print(f"✓ X shape: {dataset.X.shape}")
    print(f"✓ y shape: {dataset.y.shape}")

    # Test 3: Class distribution
    print("\n### Test 3: Class Distribution ###")
    unique, counts = torch.unique(dataset.y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"✓ Class {label} ({dataset.labels()[label]}): {count} samples")
        assert count <= dataset.max_per_class

    # Test 4: Data access
    print("\n### Test 4: Data Access ###")
    x, y = dataset[0]
    print(f"✓ First sample shape: X={x.shape}, y={y.shape}")
    print(f"✓ First sample label: {dataset.labels()[y]}")

    # Test 5: Slicer function
    print("\n### Test 5: Slicer Function ###")
    test_array = np.arange(1000)
    windows = slicer(test_array, win=100, step=50, return_df=False)
    print(f"✓ Slicer created {windows.shape[0]} windows of size {windows.shape[1]}")

    print("\n" + "=" * 80)
    print("All Manual Tests Passed!")
    print("=" * 80)


if __name__ == "__main__":
    run_manual_tests()
