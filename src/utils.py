from __future__ import annotations

import os
import json
from typing import Tuple

import numpy as np
import torch

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def proj_images_from_voxels(vox: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = (vox > 0.5).astype(np.uint8)
    xy = (v.max(axis=2) * 255).astype(np.uint8)
    xz = (v.max(axis=1) * 255).astype(np.uint8)
    yz = (v.max(axis=0) * 255).astype(np.uint8)
    return xy, xz, yz


def choose_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] CUDA device '{device_arg}' requested but CUDA is unavailable, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_arg)


def compute_iou(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    p = (torch.sigmoid(pred_logits) > threshold).float()
    t = (target > 0.5).float()
    inter = (p * t).sum(dim=(1, 2, 3, 4))
    union = ((p + t) > 0).float().sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
    return float((inter / union).mean().item())


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits)
    t = (target > 0.5).float()
    num = 2.0 * (p * t).sum(dim=(1, 2, 3, 4)) + eps
    den = (p + t).sum(dim=(1, 2, 3, 4)) + eps
    return (1.0 - (num / den)).mean()
