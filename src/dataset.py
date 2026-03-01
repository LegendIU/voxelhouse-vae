from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

class VoxelNPZDataset(Dataset):
    def __init__(self, npz_path: str, resolution: int | None = None, augment: bool = False, seed: int = 42):
        data = np.load(npz_path, allow_pickle=True)
        self.voxels = data["voxels"]  # uint8 [N,R,R,R]
        self.paths = data.get("paths", None)
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        if resolution is not None:
            assert self.voxels.shape[1] == resolution, f"Expected resolution {resolution}, got {self.voxels.shape[1]}"

    def __len__(self):
        return int(self.voxels.shape[0])

    def _augment(self, v: np.ndarray) -> np.ndarray:
        # v: [R,R,R] in (x,y,z). We apply random rotation around z (vertical) and flips in x/y.
        k = int(self.rng.integers(0, 4))
        if k:
            v = np.rot90(v, k=k, axes=(0, 1))  # rotate in x-y plane
        if bool(self.rng.integers(0, 2)):
            v = np.flip(v, axis=0)
        if bool(self.rng.integers(0, 2)):
            v = np.flip(v, axis=1)
        return v.copy()

    def __getitem__(self, idx):
        v = self.voxels[idx]
        if self.augment:
            v = self._augment(v)
        x = torch.from_numpy(v.astype(np.float32))[None, ...]  # [1,R,R,R]
        return x
