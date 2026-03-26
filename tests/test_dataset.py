"""Tests for VoxelNPZDataset loading and __getitem__."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dataset import VoxelNPZDataset


@pytest.fixture
def temp_npz(tmp_path: Path) -> Path:
    """Create a minimal NPZ with voxels (N, R, R, R) and optional paths."""
    n, r = 3, 32
    voxels = (np.random.rand(n, r, r, r) > 0.7).astype(np.uint8)
    paths = np.array([f"mesh_{i}.obj" for i in range(n)], dtype=object)
    style = np.array([0, 1, 2], dtype=np.int64)
    out = tmp_path / "data.npz"
    np.savez_compressed(out, voxels=voxels, paths=paths, style=style)
    return out


def test_dataset_len_and_shape(temp_npz: Path) -> None:
    """Dataset length and __getitem__ shape match NPZ."""
    ds = VoxelNPZDataset(str(temp_npz), resolution=32, augment=False)
    assert len(ds) == 3
    x = ds[0]
    assert x.shape == (1, 32, 32, 32)
    assert x.dtype == torch.float32
    assert "style" in ds.extra_arrays
    assert ds.extra_arrays["style"].tolist() == [0, 1, 2]


def test_dataset_with_augment(temp_npz: Path) -> None:
    """Dataset with augment=True runs without error (stochastic)."""
    ds = VoxelNPZDataset(str(temp_npz), resolution=32, augment=True, seed=42)
    _ = ds[0]
    _ = ds[1]
    # No shape/resolution mismatch
    assert ds[0].shape == (1, 32, 32, 32)


def test_dataset_wrong_resolution_raises(temp_npz: Path) -> None:
    """Passing resolution that doesn't match data raises."""
    with pytest.raises(ValueError, match="Expected resolution 64"):
        VoxelNPZDataset(str(temp_npz), resolution=64, augment=False)


def test_dataset_missing_voxels_key(tmp_path: Path) -> None:
    """NPZ without 'voxels' key raises."""
    np.savez_compressed(tmp_path / "bad.npz", data=np.zeros((1, 8, 8, 8)))
    with pytest.raises(KeyError, match="voxels"):
        VoxelNPZDataset(str(tmp_path / "bad.npz"), resolution=8, augment=False)
