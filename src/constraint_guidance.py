from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eval_3d_recon import connected_components_stats_single


@dataclass(frozen=True)
class ConstraintSpec:
    min_connectedness: float = 0.85
    max_unsupported_mass: float = 0.08
    max_component_count: int = 2
    min_symmetry: float = 0.45
    min_plausibility: float = 0.55
    require_compact: bool = False


def connectedness_score(sample: np.ndarray) -> float:
    _, lcc_ratio = connected_components_stats_single(sample.astype(np.uint8))
    return float(lcc_ratio)


def unsupported_mass_ratio(sample: np.ndarray) -> float:
    # Axis convention (shared with house_attributes.extract_house_attribute_ids):
    # voxel grid is [X, Y, Z]; Z is the vertical/up axis, Z=0 is the ground plane.
    # A voxel is "supported" if it sits on the ground or directly above another occupied voxel.
    occ = sample.astype(bool)
    if occ.sum() == 0:
        return 1.0
    supported = np.zeros_like(occ, dtype=bool)
    supported[:, :, 0] = occ[:, :, 0]
    supported[:, :, 1:] = occ[:, :, 1:] & occ[:, :, :-1]
    unsupported = occ & ~supported
    return float(unsupported.sum() / max(occ.sum(), 1))


def component_count(sample: np.ndarray) -> int:
    cc, _ = connected_components_stats_single(sample.astype(np.uint8))
    return int(cc)


def symmetry_proxy(sample: np.ndarray) -> float:
    occ = sample.astype(np.float32)
    if occ.size == 0:
        return 0.0
    sx = 1.0 - float(np.abs(occ - occ[::-1, :, :]).mean())
    sy = 1.0 - float(np.abs(occ - occ[:, ::-1, :]).mean())
    return float(max(0.0, min(1.0, 0.5 * (sx + sy))))


def compactness_proxy(sample: np.ndarray) -> float:
    occ = sample.astype(bool)
    if occ.sum() == 0:
        return 0.0
    xs = np.where(occ.any(axis=(1, 2)))[0]
    ys = np.where(occ.any(axis=(0, 2)))[0]
    zs = np.where(occ.any(axis=(0, 1)))[0]
    if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
        return 0.0
    vol = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1) * (zs.max() - zs.min() + 1)
    return float(occ.sum() / max(vol, 1))


def _plausibility_from_parts(
    connected: float,
    unsupported: float,
    comps: int,
    symmetry: float,
    compact: float,
) -> float:
    component_term = float(np.exp(-max(comps - 1, 0)))
    score = (
        0.35 * connected
        + 0.25 * (1.0 - unsupported)
        + 0.15 * component_term
        + 0.15 * symmetry
        + 0.10 * compact
    )
    return float(max(0.0, min(1.0, score)))


def plausibility_score(sample: np.ndarray) -> float:
    occ_u8 = sample.astype(np.uint8)
    comps, lcc_ratio = connected_components_stats_single(occ_u8)
    return _plausibility_from_parts(
        connected=float(lcc_ratio),
        unsupported=unsupported_mass_ratio(occ_u8),
        comps=int(comps),
        symmetry=symmetry_proxy(occ_u8),
        compact=compactness_proxy(occ_u8),
    )


def _batch_unsupported_mass(occ: np.ndarray) -> np.ndarray:
    # Vectorized over the leading batch dim. occ is bool[N, X, Y, Z]; Z=0 is ground.
    supported = np.zeros_like(occ)
    supported[:, :, :, 0] = occ[:, :, :, 0]
    supported[:, :, :, 1:] = occ[:, :, :, 1:] & occ[:, :, :, :-1]
    unsupported = occ & ~supported
    occupied = occ.sum(axis=(1, 2, 3)).astype(np.float64)
    unsupported_count = unsupported.sum(axis=(1, 2, 3)).astype(np.float64)
    out = np.where(occupied > 0, unsupported_count / np.maximum(occupied, 1.0), 1.0)
    return out


def _batch_symmetry(occ: np.ndarray) -> np.ndarray:
    occ_f = occ.astype(np.float32)
    sx = 1.0 - np.abs(occ_f - occ_f[:, ::-1, :, :]).mean(axis=(1, 2, 3))
    sy = 1.0 - np.abs(occ_f - occ_f[:, :, ::-1, :]).mean(axis=(1, 2, 3))
    out = 0.5 * (sx + sy)
    return np.clip(out, 0.0, 1.0)


def _batch_compactness(occ: np.ndarray) -> np.ndarray:
    n = occ.shape[0]
    out = np.zeros(n, dtype=np.float64)
    any_x = occ.any(axis=(2, 3))
    any_y = occ.any(axis=(1, 3))
    any_z = occ.any(axis=(1, 2))
    occupied = occ.sum(axis=(1, 2, 3)).astype(np.float64)
    for i in range(n):
        if occupied[i] == 0:
            continue
        xs = np.flatnonzero(any_x[i])
        ys = np.flatnonzero(any_y[i])
        zs = np.flatnonzero(any_z[i])
        if xs.size == 0 or ys.size == 0 or zs.size == 0:
            continue
        vol = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1) * (zs.max() - zs.min() + 1)
        out[i] = occupied[i] / max(int(vol), 1)
    return out


def score_samples(samples: np.ndarray) -> list[dict[str, float]]:
    occ_u8 = samples.astype(np.uint8)
    occ_bool = occ_u8.astype(bool)
    n = occ_u8.shape[0]

    unsupported = _batch_unsupported_mass(occ_bool)
    symmetry = _batch_symmetry(occ_bool)
    compact = _batch_compactness(occ_bool)

    out: list[dict[str, float]] = []
    for i in range(n):
        comps, lcc_ratio = connected_components_stats_single(occ_u8[i])
        plausibility = _plausibility_from_parts(
            connected=float(lcc_ratio),
            unsupported=float(unsupported[i]),
            comps=int(comps),
            symmetry=float(symmetry[i]),
            compact=float(compact[i]),
        )
        out.append(
            {
                "connectedness": float(lcc_ratio),
                "unsupported_mass": float(unsupported[i]),
                "component_count": float(comps),
                "symmetry_proxy": float(symmetry[i]),
                "compactness_proxy": float(compact[i]),
                "plausibility_score": plausibility,
                "energy": float(1.0 - plausibility),
            }
        )
    return out


def _passes_constraints(metrics: dict[str, float], spec: ConstraintSpec) -> bool:
    if metrics["connectedness"] < spec.min_connectedness:
        return False
    if metrics["unsupported_mass"] > spec.max_unsupported_mass:
        return False
    if metrics["component_count"] > float(spec.max_component_count):
        return False
    if metrics["symmetry_proxy"] < spec.min_symmetry:
        return False
    if metrics["plausibility_score"] < spec.min_plausibility:
        return False
    if spec.require_compact and metrics["compactness_proxy"] < 0.24:
        return False
    return True


def rerank_and_filter(
    samples: np.ndarray,
    spec: ConstraintSpec,
    n_select: int,
) -> tuple[np.ndarray, list[dict[str, float]], list[int]]:
    if samples.ndim != 4:
        raise ValueError(f"Expected samples shape [N,R,R,R], got {samples.shape}")
    if n_select <= 0:
        raise ValueError(f"n_select must be > 0, got {n_select}")

    metrics = score_samples(samples)
    ranked = sorted(
        list(enumerate(metrics)),
        key=lambda item: (
            not _passes_constraints(item[1], spec),
            item[1]["energy"],
        ),
    )
    kept_idx = [idx for idx, _ in ranked[: min(n_select, len(ranked))]]
    kept_metrics = [metrics[idx] for idx in kept_idx]
    return samples[kept_idx], kept_metrics, kept_idx

