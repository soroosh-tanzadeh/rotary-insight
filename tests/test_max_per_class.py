#!/usr/bin/env python3
"""
Test script to demonstrate max_per_class functionality in PUDataset.
This shows how max_per_class reduces memory usage by only loading necessary files.
"""

from datasets.pu_dataset import PUDataset


def test_max_per_class():
    print("=" * 80)
    print("Testing PUDataset with max_per_class parameter")
    print("=" * 80)

    # Test 1: Load dataset with max_per_class limit
    print("\n### Test 1: Loading with max_per_class=10000 ###")
    dataset_limited = PUDataset(
        rdir="./data/dataset/PU",
        window_size=2048,
        max_per_class=10000,
        seed=42,
        verbose=True,
        force_reload=False,
    )

    print(f"\nDataset with max_per_class=10000:")
    print(f"  Total windows: {len(dataset_limited)}")
    print(f"  X shape: {dataset_limited.X.shape}")
    print(f"  y shape: {dataset_limited.y.shape}")
    print(f"  Labels: {dataset_limited.labels()}")

    # Check class distribution
    import torch

    unique, counts = torch.unique(dataset_limited.y, return_counts=True)
    print(f"\nClass distribution:")
    for label, count in zip(unique, counts):
        print(f"  Class {label} ({dataset_limited.labels()[label]}): {count} samples")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
