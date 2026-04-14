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


def plausibility_score(sample: np.ndarray) -> float:
    connected = connectedness_score(sample)
    unsupported = unsupported_mass_ratio(sample)
    comps = component_count(sample)
    symmetry = symmetry_proxy(sample)
    compact = compactness_proxy(sample)
    component_term = float(np.exp(-max(comps - 1, 0)))
    score = (
        0.35 * connected
        + 0.25 * (1.0 - unsupported)
        + 0.15 * component_term
        + 0.15 * symmetry
        + 0.10 * compact
    )
    return float(max(0.0, min(1.0, score)))


def score_samples(samples: np.ndarray) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for sample in samples.astype(np.uint8):
        connected = connectedness_score(sample)
        unsupported = unsupported_mass_ratio(sample)
        comps = component_count(sample)
        symmetry = symmetry_proxy(sample)
        compact = compactness_proxy(sample)
        plausibility = plausibility_score(sample)
        energy = float(1.0 - plausibility)
        out.append(
            {
                "connectedness": connected,
                "unsupported_mass": unsupported,
                "component_count": float(comps),
                "symmetry_proxy": symmetry,
                "compactness_proxy": compact,
                "plausibility_score": plausibility,
                "energy": energy,
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

