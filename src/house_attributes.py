from __future__ import annotations

from dataclasses import dataclass

import torch


HOUSE_ATTRIBUTE_FIELD_NAMES = (
    "stories_bin",
    "footprint_bin",
    "aspect_ratio_bin",
    "roof_type",
    "symmetry_flag",
    "compactness_flag",
)


@dataclass(frozen=True)
class HouseAttributeSpec:
    ratio_bins: int = 8
    story_bins: int = 4

    def vocab_sizes(self) -> list[int]:
        return [self.story_bins, self.ratio_bins, self.ratio_bins, 3, 2, 2]


def _require_voxels(voxels: torch.Tensor) -> torch.Tensor:
    if voxels.ndim == 5:
        voxels = voxels[:, 0]
    if voxels.ndim != 4:
        raise ValueError(f"Expected voxels with shape [B,R,R,R] or [B,1,R,R,R], got {tuple(voxels.shape)}")
    return voxels


def _bin_ratio(values: torch.Tensor, num_bins: int) -> torch.Tensor:
    if num_bins <= 1:
        raise ValueError(f"num_bins must be > 1, got {num_bins}")
    clipped = values.clamp(0.0, 1.0 - 1e-8)
    return torch.floor(clipped * num_bins).long().clamp(0, num_bins - 1)


def extract_house_attribute_ids(
    voxels: torch.Tensor,
    ratio_bins: int = 8,
    story_bins: int = 4,
) -> torch.Tensor:
    voxels = _require_voxels(voxels)
    if story_bins <= 1:
        raise ValueError(f"story_bins must be > 1, got {story_bins}")

    occ = voxels > 0.5
    _, resolution, _, _ = occ.shape
    occ_f = occ.float()

    # Floors (z-axis occupancy profile): mapped to low-rise / mid / high categories.
    z_profile = occ.any(dim=(1, 2)).float()  # [B, Z]
    active_levels = z_profile.sum(dim=1)
    story_ratio = active_levels / max(float(resolution), 1.0)
    stories_bin = _bin_ratio(story_ratio, story_bins)

    # Footprint occupancy on ground plane.
    footprint = occ.any(dim=3).float().mean(dim=(1, 2))
    footprint_bin = _bin_ratio(footprint, ratio_bins)

    # Aspect ratio in x/y footprint bounding box.
    footprint_mask = occ.any(dim=3)
    x_extent = footprint_mask.any(dim=2).sum(dim=1).float().clamp_min(1.0)
    y_extent = footprint_mask.any(dim=1).sum(dim=1).float().clamp_min(1.0)
    aspect = torch.minimum(x_extent, y_extent) / torch.maximum(x_extent, y_extent)
    aspect_ratio_bin = _bin_ratio(aspect, ratio_bins)

    # Roof type proxy from top profile:
    # 0 = flat-like, 1 = sloped, 2 = peaked.
    top_idx = torch.argmax(torch.flip(z_profile, dims=[1]), dim=1)
    top_z = (resolution - 1 - top_idx).long()
    gather_idx = top_z.view(-1, 1, 1, 1).expand(-1, occ_f.shape[1], occ_f.shape[2], 1)
    roof_slice = torch.gather(occ_f, dim=3, index=gather_idx).squeeze(-1)
    roof_coverage = roof_slice.mean(dim=(1, 2))
    roof_height_center = (z_profile * torch.arange(resolution, device=voxels.device).float()).sum(dim=1) / active_levels.clamp_min(1.0)
    roof_sharpness = top_z.float() - roof_height_center
    roof_type = torch.where(
        roof_coverage > 0.55,
        torch.zeros_like(roof_coverage, dtype=torch.long),
        torch.where(roof_sharpness > 2.0, torch.full_like(roof_coverage, 2, dtype=torch.long), torch.ones_like(roof_coverage, dtype=torch.long)),
    )

    # Symmetry proxy over x/y mirrors.
    mirror_x = torch.flip(occ_f, dims=[1])
    mirror_y = torch.flip(occ_f, dims=[2])
    same_x = 1.0 - torch.abs(occ_f - mirror_x).mean(dim=(1, 2, 3))
    same_y = 1.0 - torch.abs(occ_f - mirror_y).mean(dim=(1, 2, 3))
    symmetry_flag = ((same_x + same_y) * 0.5 > 0.82).long()

    # Compactness proxy: occupancy relative to tight AABB volume.
    z_extent = occ.any(dim=(1, 2)).sum(dim=1).float().clamp_min(1.0)
    occupied = occ_f.sum(dim=(1, 2, 3)).clamp_min(1.0)
    aabb_volume = (x_extent * y_extent * z_extent).clamp_min(1.0)
    compactness = occupied / aabb_volume
    compactness_flag = (compactness > 0.26).long()

    return torch.stack(
        [stories_bin, footprint_bin, aspect_ratio_bin, roof_type, symmetry_flag, compactness_flag],
        dim=1,
    )

