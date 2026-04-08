from __future__ import annotations

from typing import Iterable

import numpy as np

from eval_3d_recon import connected_components_stats_single


def _sample_pairs(n_items: int, max_pairs: int, seed: int) -> Iterable[tuple[int, int]]:
    if n_items < 2 or max_pairs <= 0:
        return []
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, int]] = set()
    limit = min(max_pairs, n_items * (n_items - 1) // 2)
    pairs: list[tuple[int, int]] = []
    while len(pairs) < limit:
        i = int(rng.integers(0, n_items))
        j = int(rng.integers(0, n_items - 1))
        if j >= i:
            j += 1
        a, b = sorted((i, j))
        pair = (a, b)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def pairwise_diversity_stats(voxels: np.ndarray, max_pairs: int = 256, seed: int = 42) -> dict[str, float]:
    if voxels.ndim != 4:
        raise ValueError(f"Expected voxels with shape [N,R,R,R], got {voxels.shape}")
    pairs = list(_sample_pairs(voxels.shape[0], max_pairs=max_pairs, seed=seed))
    if not pairs:
        return {"pairwise_hamming": 0.0, "pairwise_iou": 1.0, "pairwise_iou_diversity": 0.0}

    hamming_values = []
    iou_values = []
    for i, j in pairs:
        a = voxels[i] > 0
        b = voxels[j] > 0
        hamming_values.append(float(np.not_equal(a, b).mean()))

        inter = float(np.logical_and(a, b).sum())
        union = float(np.logical_or(a, b).sum())
        iou_values.append(inter / max(union, 1.0))

    mean_iou = float(np.mean(iou_values))
    return {
        "pairwise_hamming": float(np.mean(hamming_values)),
        "pairwise_iou": mean_iou,
        "pairwise_iou_diversity": 1.0 - mean_iou,
    }


def reference_nearest_iou(
    generated: np.ndarray,
    reference: np.ndarray,
    max_reference: int = 64,
    seed: int = 42,
) -> dict[str, float]:
    if generated.ndim != 4 or reference.ndim != 4:
        raise ValueError(
            f"Expected generated/reference with shape [N,R,R,R], got {generated.shape} and {reference.shape}"
        )
    if len(reference) == 0 or len(generated) == 0:
        return {"reference_nn_iou_mean": 0.0, "reference_nn_iou_std": 0.0}

    rng = np.random.default_rng(seed)
    if len(reference) > max_reference:
        ref_idx = rng.choice(len(reference), size=max_reference, replace=False)
        reference = reference[ref_idx]

    reference = reference > 0
    best_scores = []
    for sample in generated > 0:
        inter = np.logical_and(reference, sample[None, ...]).sum(axis=(1, 2, 3)).astype(np.float32)
        union = np.logical_or(reference, sample[None, ...]).sum(axis=(1, 2, 3)).astype(np.float32)
        best_scores.append(float((inter / np.maximum(union, 1.0)).max()))

    return {
        "reference_nn_iou_mean": float(np.mean(best_scores)),
        "reference_nn_iou_std": float(np.std(best_scores)),
    }


def summarize_voxel_samples(
    voxels: np.ndarray,
    reference_voxels: np.ndarray | None = None,
    max_pairs: int = 256,
    max_reference: int = 64,
    seed: int = 42,
) -> dict[str, float]:
    if voxels.ndim != 4:
        raise ValueError(f"Expected voxels with shape [N,R,R,R], got {voxels.shape}")
    if len(voxels) == 0:
        raise ValueError("voxels must contain at least one sample")

    occ = voxels.astype(np.uint8)
    occupancy = occ.mean(axis=(1, 2, 3))
    valid = np.logical_and(occupancy > 1e-5, occupancy < 1.0 - 1e-5)

    unique = {occ[i].tobytes() for i in range(occ.shape[0])}
    num_components = []
    lcc_ratios = []
    for sample in occ:
        cc, lcc = connected_components_stats_single(sample)
        num_components.append(float(cc))
        lcc_ratios.append(float(lcc))

    out = {
        "n_samples": int(occ.shape[0]),
        "valid_ratio": float(valid.mean()),
        "occupancy_mean": float(occupancy.mean()),
        "occupancy_std": float(occupancy.std()),
        "unique_ratio": float(len(unique) / max(len(occ), 1)),
        "num_components": float(np.mean(num_components)),
        "largest_component_ratio": float(np.mean(lcc_ratios)),
    }
    out.update(pairwise_diversity_stats(occ, max_pairs=max_pairs, seed=seed))

    if reference_voxels is not None:
        out.update(reference_nearest_iou(occ, reference_voxels, max_reference=max_reference, seed=seed))

    return out
