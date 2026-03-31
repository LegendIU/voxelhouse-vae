from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from dataset import VoxelNPZDataset


SHAPE_STAT_FIELD_NAMES = (
    "occupancy_bin",
    "height_bin",
    "footprint_bin",
    "vertical_center_bin",
)


def parse_condition_fields(raw_fields: str | Sequence[str] | None) -> list[str]:
    if raw_fields is None:
        return []
    if isinstance(raw_fields, str):
        return [field.strip() for field in raw_fields.split(",") if field.strip()]
    return [str(field).strip() for field in raw_fields if str(field).strip()]


def _require_num_bins(num_bins: int) -> int:
    if num_bins <= 1:
        raise ValueError(f"num_bins must be > 1, got {num_bins}")
    return int(num_bins)


def _bin_values(values: torch.Tensor, num_bins: int) -> torch.Tensor:
    num_bins = _require_num_bins(num_bins)
    clipped = values.clamp(0.0, 1.0 - 1e-8)
    return torch.floor(clipped * num_bins).long().clamp(0, num_bins - 1)


def build_shape_condition_ids(voxels: torch.Tensor, num_bins: int = 8) -> torch.Tensor:
    if voxels.ndim == 5:
        voxels = voxels[:, 0]
    if voxels.ndim != 4:
        raise ValueError(f"Expected voxels with shape [B,R,R,R] or [B,1,R,R,R], got {tuple(voxels.shape)}")

    num_bins = _require_num_bins(num_bins)
    occ = voxels > 0.5
    bsz, _, _, resolution = occ.shape

    occupancy_ratio = occ.float().mean(dim=(1, 2, 3))
    footprint_ratio = occ.any(dim=3).float().mean(dim=(1, 2))
    height_ratio = occ.any(dim=(1, 2)).float().mean(dim=1)

    z_coords = torch.arange(resolution, device=voxels.device, dtype=torch.float32).view(1, 1, 1, resolution)
    occ_mass = occ.float().sum(dim=(1, 2, 3)).clamp_min(1.0)
    vertical_center = (occ.float() * z_coords).sum(dim=(1, 2, 3)) / occ_mass
    vertical_center = vertical_center / max(float(resolution - 1), 1.0)

    return torch.stack(
        [
            _bin_values(occupancy_ratio, num_bins),
            _bin_values(height_ratio, num_bins),
            _bin_values(footprint_ratio, num_bins),
            _bin_values(vertical_center, num_bins),
        ],
        dim=1,
    )


def infer_condition_vocab_sizes(
    dataset: VoxelNPZDataset,
    mode: str,
    fields: Sequence[str] | None = None,
    num_bins: int = 8,
) -> tuple[list[str], list[int]]:
    if mode == "none":
        return [], []
    if mode == "shape_stats":
        _require_num_bins(num_bins)
        return list(SHAPE_STAT_FIELD_NAMES), [int(num_bins)] * len(SHAPE_STAT_FIELD_NAMES)
    if mode == "npz_fields":
        field_names = parse_condition_fields(fields)
        if not field_names:
            raise ValueError("For condition_mode=npz_fields, provide at least one --condition_fields entry")
        vocab_sizes = []
        for field in field_names:
            if field not in dataset.extra_arrays:
                raise KeyError(f"Condition field '{field}' is not present in dataset extras")
            values = np.asarray(dataset.extra_arrays[field])
            if values.ndim != 1:
                raise ValueError(f"Condition field '{field}' must be 1D, got shape {values.shape}")
            if values.shape[0] != len(dataset):
                raise ValueError(
                    f"Condition field '{field}' has length {values.shape[0]}, expected {len(dataset)}"
                )
            min_value = int(values.min())
            if min_value < 0:
                raise ValueError(f"Condition field '{field}' must be non-negative integer ids, got min={min_value}")
            vocab_sizes.append(int(values.max()) + 1)
        return field_names, vocab_sizes
    raise ValueError(f"Unsupported condition mode: {mode}")


def gather_condition_ids(
    dataset: VoxelNPZDataset,
    batch_indices: torch.Tensor,
    batch_voxels: torch.Tensor,
    mode: str,
    fields: Sequence[str] | None = None,
    num_bins: int = 8,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode == "shape_stats":
        return build_shape_condition_ids(batch_voxels, num_bins=num_bins)
    if mode == "npz_fields":
        if batch_indices.ndim != 1:
            raise ValueError(f"Expected batch_indices with shape [B], got {tuple(batch_indices.shape)}")
        device = batch_voxels.device
        ids = []
        for field in parse_condition_fields(fields):
            values = np.asarray(dataset.extra_arrays[field])[batch_indices.cpu().numpy()]
            ids.append(torch.as_tensor(values, device=device, dtype=torch.long))
        return torch.stack(ids, dim=1)
    raise ValueError(f"Unsupported condition mode: {mode}")


def sample_condition_ids_from_dataset(
    dataset: VoxelNPZDataset,
    mode: str,
    n_samples: int,
    fields: Sequence[str] | None = None,
    num_bins: int = 8,
    seed: int = 42,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(dataset), size=n_samples)
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    voxels = torch.from_numpy(dataset.voxels[indices].astype(np.float32, copy=False))[:, None, ...]
    return gather_condition_ids(dataset, index_tensor, voxels, mode=mode, fields=fields, num_bins=num_bins)
