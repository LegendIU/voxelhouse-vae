from __future__ import annotations
import argparse, os, random
import numpy as np
from tqdm import tqdm
import trimesh
from scipy.ndimage import binary_fill_holes, label
from utils import ensure_dir, save_json

def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.bounding_box.centroid)
    s = 1.0 / float(np.max(mesh.extents) + 1e-9)
    mesh.apply_scale(s)
    return mesh

def voxelize_surface(mesh: trimesh.Trimesh, resolution: int) -> np.ndarray:
    R = resolution
    pitch = 1.0 / R
    vg = trimesh.voxel.creation.voxelize(mesh, pitch=pitch, method="subdivide")
    occ = np.zeros((R, R, R), dtype=np.uint8)
    if vg.points is None or len(vg.points) == 0:
        return occ
    pts = vg.points
    # map points in [-0.5,0.5]^3 to indices
    u = (pts + 0.5) * R
    idx = np.floor(u).astype(np.int32)
    idx = np.clip(idx, 0, R - 1)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    return occ

def keep_largest_cc(occ: np.ndarray) -> np.ndarray:
    # Keep largest connected component in 3D (6-connectivity)
    structure = np.zeros((3,3,3), dtype=np.uint8)
    structure[1,1,0]=structure[1,1,2]=1
    structure[1,0,1]=structure[1,2,1]=1
    structure[0,1,1]=structure[2,1,1]=1
    lab, n = label(occ > 0, structure=structure)
    if n <= 1:
        return occ
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = counts.argmax()
    return (lab == keep).astype(np.uint8)

def list_meshes(mesh_dir: str):
    exts = (".obj", ".ply", ".glb", ".gltf", ".stl")
    out = []
    for root, _, files in os.walk(mesh_dir):
        for fn in files:
            if fn.lower().endswith(exts):
                out.append(os.path.join(root, fn))
    out.sort()
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--max_models", type=int, default=0, help="0 = use all meshes found")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--fill_holes", action="store_true", help="binary_fill_holes for solid occupancy")
    p.add_argument("--keep_lcc", action="store_true", help="keep largest connected component")
    args = p.parse_args()

    ensure_dir(args.out_dir)
    paths = list_meshes(args.mesh_dir)
    if not paths:
        raise SystemExit(f"No meshes found in {args.mesh_dir}")

    random.seed(args.seed)
    random.shuffle(paths)
    if args.max_models and args.max_models > 0:
        paths = paths[:args.max_models]

    n = len(paths)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    splits = {
        "train": paths[:n_train],
        "val": paths[n_train:n_train+n_val],
        "test": paths[n_train+n_val:],
    }

    save_json({
        "mesh_dir": os.path.abspath(args.mesh_dir),
        "out_dir": os.path.abspath(args.out_dir),
        "resolution": args.resolution,
        "max_models": args.max_models,
        "seed": args.seed,
        "sizes": {k: len(v) for k,v in splits.items()},
        "note": "surface occupancy via trimesh.voxelize(method=subdivide), optional 3D fill_holes + largest-CC cleanup"
    }, os.path.join(args.out_dir, "meta.json"))

    for split, split_paths in splits.items():
        voxels = np.zeros((len(split_paths), args.resolution, args.resolution, args.resolution), dtype=np.uint8)
        keep = []
        for path in tqdm(split_paths, desc=f"Voxelize {split}"):
            try:
                mesh = trimesh.load(path, force="mesh")
                if isinstance(mesh, trimesh.Scene):
                    geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                    if not geoms:
                        continue
                    mesh = trimesh.util.concatenate(geoms)
                if mesh is None or mesh.is_empty:
                    continue
                mesh = normalize_mesh(mesh)
                occ = voxelize_surface(mesh, args.resolution)
                if occ.sum() == 0:
                    continue
                if args.fill_holes:
                    occ = binary_fill_holes(occ > 0).astype(np.uint8)
                if args.keep_lcc:
                    occ = keep_largest_cc(occ)
                if occ.sum() == 0:
                    continue
                voxels[len(keep)] = occ
                keep.append(path)
            except Exception:
                continue
        voxels = voxels[:len(keep)]
        np.savez_compressed(os.path.join(args.out_dir, f"{split}.npz"),
                            voxels=voxels,
                            paths=np.array(keep, dtype=object))
        print(f"[{split}] kept {len(keep)}/{len(split_paths)}")

if __name__ == "__main__":
    main()
