
from __future__ import annotations
import os, json
import numpy as np
from typing import Tuple

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
