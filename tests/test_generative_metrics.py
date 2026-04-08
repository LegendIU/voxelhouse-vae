"""Tests for unconditional generation metrics and diversity summaries."""
from __future__ import annotations

import numpy as np

from generative_metrics import pairwise_diversity_stats, summarize_voxel_samples


def test_pairwise_diversity_detects_different_shapes() -> None:
    voxels = np.zeros((2, 4, 4, 4), dtype=np.uint8)
    voxels[0, :2, :2, :2] = 1
    voxels[1, 2:, 2:, 2:] = 1

    stats = pairwise_diversity_stats(voxels, max_pairs=4, seed=3)

    assert stats["pairwise_hamming"] > 0.0
    assert 0.0 <= stats["pairwise_iou"] <= 1.0
    assert 0.0 <= stats["pairwise_iou_diversity"] <= 1.0


def test_summarize_voxel_samples_reports_unique_ratio() -> None:
    voxels = np.zeros((3, 4, 4, 4), dtype=np.uint8)
    voxels[0, :2, :2, :2] = 1
    voxels[1, :2, :2, :2] = 1
    voxels[2, 2:, 2:, 2:] = 1

    summary = summarize_voxel_samples(voxels, max_pairs=4, seed=1)

    assert summary["n_samples"] == 3
    assert 0.0 < summary["unique_ratio"] < 1.0
    assert 0.0 <= summary["valid_ratio"] <= 1.0
