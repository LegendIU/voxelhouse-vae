"""Tests for optional conditioning utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from conditioning import build_shape_condition_ids, infer_condition_vocab_sizes, sample_condition_ids_from_dataset
from dataset import VoxelNPZDataset


def test_build_shape_condition_ids_returns_binned_fields() -> None:
    voxels = torch.zeros(2, 1, 8, 8, 8)
    voxels[0, 0, :4, :4, :4] = 1.0
    voxels[1, 0, 2:8, 2:8, 1:7] = 1.0

    cond = build_shape_condition_ids(voxels, num_bins=6)

    assert cond.shape == (2, 4)
    assert cond.dtype == torch.long
    assert int(cond.min()) >= 0
    assert int(cond.max()) < 6


def test_npz_field_conditioning_schema_and_sampling(tmp_path: Path) -> None:
    voxels = np.zeros((3, 8, 8, 8), dtype=np.uint8)
    voxels[:, 1:3, 1:3, 1:3] = 1
    cls = np.array([0, 1, 2], dtype=np.int64)
    style = np.array([1, 0, 1], dtype=np.int64)
    out = tmp_path / "data.npz"
    np.savez_compressed(out, voxels=voxels, cls=cls, style=style)

    ds = VoxelNPZDataset(str(out), resolution=8, augment=False)
    names, vocab_sizes = infer_condition_vocab_sizes(ds, mode="npz_fields", fields=["cls", "style"])
    sampled = sample_condition_ids_from_dataset(
        ds,
        mode="npz_fields",
        n_samples=2,
        fields=names,
        seed=7,
    )

    assert names == ["cls", "style"]
    assert vocab_sizes == [3, 2]
    assert sampled is not None
    assert sampled.shape == (2, 2)


def test_house_attribute_conditioning_schema_and_sampling(tmp_path: Path) -> None:
    voxels = np.zeros((4, 8, 8, 8), dtype=np.uint8)
    voxels[0, 2:6, 2:6, :3] = 1
    voxels[1, 1:7, 1:7, :5] = 1
    voxels[2, 2:6, 1:7, :2] = 1
    voxels[3, 1:7, 2:6, :6] = 1
    out = tmp_path / "data_house.npz"
    np.savez_compressed(out, voxels=voxels)

    ds = VoxelNPZDataset(str(out), resolution=8, augment=False)
    names, vocab_sizes = infer_condition_vocab_sizes(ds, mode="house_attributes", num_bins=6)
    sampled = sample_condition_ids_from_dataset(
        ds,
        mode="house_attributes",
        n_samples=3,
        num_bins=6,
        seed=11,
    )

    assert names == [
        "stories_bin",
        "footprint_bin",
        "aspect_ratio_bin",
        "roof_type",
        "symmetry_flag",
        "compactness_flag",
    ]
    assert vocab_sizes == [4, 6, 6, 3, 2, 2]
    assert sampled is not None
    assert sampled.shape == (3, 6)
