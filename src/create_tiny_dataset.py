from __future__ import annotations

import argparse
import json
import os

import numpy as np


def make_sample(resolution: int, rng: np.random.Generator) -> np.ndarray:
    vox = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    base_w = int(rng.integers(max(3, resolution // 6), max(4, resolution // 3)))
    base_d = int(rng.integers(max(3, resolution // 6), max(4, resolution // 3)))
    height = int(rng.integers(max(2, resolution // 8), max(3, resolution // 2)))
    x0 = int(rng.integers(1, max(2, resolution - base_w - 1)))
    y0 = int(rng.integers(1, max(2, resolution - base_d - 1)))
    vox[x0 : x0 + base_w, y0 : y0 + base_d, :height] = 1
    if rng.random() < 0.6 and height + 2 < resolution:
        roof_h = int(rng.integers(1, 3))
        vox[x0 + 1 : x0 + base_w - 1, y0 + 1 : y0 + base_d - 1, height : height + roof_h] = 1
    return vox


def build_split(n: int, resolution: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    vox = np.stack([make_sample(resolution, rng) for _ in range(n)], axis=0)
    paths = np.asarray([f"synthetic_{i:04d}.obj" for i in range(n)])
    return {"voxels": vox, "paths": paths}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--n_train", type=int, default=24)
    parser.add_argument("--n_val", type=int, default=8)
    parser.add_argument("--n_test", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    splits = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    for i, (name, n_items) in enumerate(splits.items()):
        arrs = build_split(n_items, args.resolution, seed=args.seed + i)
        np.savez_compressed(os.path.join(args.out_dir, f"{name}.npz"), **arrs)

    meta = {
        "source": "synthetic_tiny_smoke_dataset",
        "resolution": int(args.resolution),
        "splits": splits,
        "seed": int(args.seed),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Created tiny dataset at {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()

