from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

class VoxelNPZDataset(Dataset):
    def __init__(self, npz_path: str, resolution: int | None = None, augment: bool = False, seed: int = 42):
        self.voxels, self.paths, self.extra_arrays = self._load_arrays(npz_path)
        self.augment = augment
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        if self.voxels.ndim != 4:
            raise ValueError(f"Expected voxels with shape [N,R,R,R], got {self.voxels.shape}")
        if resolution is not None:
            if self.voxels.shape[1] != resolution:
                raise ValueError(f"Expected resolution {resolution}, got {self.voxels.shape[1]}")

    @staticmethod
    def _load_arrays(npz_path: str):
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                if "voxels" not in data:
                    raise KeyError(f"'voxels' key is missing in {npz_path}")
                voxels = np.asarray(data["voxels"])
                paths = np.asarray(data["paths"]) if "paths" in data.files else None
                extras = {
                    key: np.asarray(data[key])
                    for key in data.files
                    if key not in {"voxels", "paths"}
                }
                return voxels, paths, extras
        except ValueError as exc:
            # Backward compatibility with old archives that store object arrays.
            if "allow_pickle=False" not in str(exc):
                raise
            with np.load(npz_path, allow_pickle=True) as data:
                if "voxels" not in data:
                    raise KeyError(f"'voxels' key is missing in {npz_path}")
                voxels = np.asarray(data["voxels"])
                paths = np.asarray(data["paths"]) if "paths" in data.files else None
                extras = {
                    key: np.asarray(data[key])
                    for key in data.files
                    if key not in {"voxels", "paths"}
                }
                return voxels, paths, extras

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
        x = torch.from_numpy(v.astype(np.float32, copy=False))[None, ...]  # [1,R,R,R]
        return x
