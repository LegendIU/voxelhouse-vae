from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
from torch.utils.data import get_worker_info


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    info = get_worker_info()
    if info is None:
        return
    base = getattr(info.dataset, "base", info.dataset)
    if hasattr(base, "rng"):
        base.rng = np.random.default_rng(worker_seed)


def make_grad_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device="cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def estimate_pos_weight(voxels: np.ndarray, max_value: float = 1e4) -> float:
    if voxels.size == 0:
        raise ValueError("Training voxels array is empty")
    pos = float((voxels > 0).sum())
    neg = float(voxels.size - pos)
    if pos <= 0:
        raise ValueError("Training voxels contain no occupied cells")
    return min(neg / pos, float(max_value))


@torch.no_grad()
def save_recon_grid(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    batch: torch.Tensor,
    out_path: str,
    threshold: float = 0.5,
    max_items: int = 8,
) -> bool:
    """Render a 2-row XY-projection grid (GT vs reconstruction) and save it.

    forward_fn(x) must return reconstruction logits with the same spatial shape as x.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    x = batch[:max_items]
    logits = forward_fn(x)
    probs = torch.sigmoid(logits)

    def proj(v: torch.Tensor) -> np.ndarray:
        v = (v > threshold).float()
        xy = v.max(dim=4).values.squeeze(1)
        return xy.cpu().numpy()

    gt = proj(x)
    rc = proj(probs)
    bsz = gt.shape[0]

    fig, axes = plt.subplots(2, bsz, figsize=(bsz * 2, 4))
    if bsz == 1:
        axes = np.array([[axes[0]], [axes[1]]], dtype=object)

    for i in range(bsz):
        axes[0, i].imshow(gt[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(rc[i], cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_title("GT")
    axes[1, 0].set_title("Recon")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
