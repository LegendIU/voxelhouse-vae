import os
import argparse
import numpy as np
import torch
from PIL import Image

from model_3d import VAE3D

# --- marching cubes backend (prefer skimage, fallback to trimesh) ---
def mc_vertices_faces(occ: np.ndarray):
    occ = occ.astype(np.float32)
    try:
        from skimage import measure
        verts, faces, _, _ = measure.marching_cubes(occ, level=0.5)
        return verts, faces
    except Exception:
        # fallback
        from trimesh.voxel.ops import matrix_to_marching_cubes
        mesh = matrix_to_marching_cubes(occ, pitch=1.0)
        return np.asarray(mesh.vertices), np.asarray(mesh.faces)

def write_obj(path: str, verts: np.ndarray, faces: np.ndarray):
    with open(path, "w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # OBJ is 1-indexed
        for tri in faces:
            a, b, c = tri + 1
            f.write(f"f {a} {b} {c}\n")

def export_mesh_from_occ(occ: np.ndarray, out_path: str):
    # skip empty / full
    if occ.mean() < 1e-5 or occ.mean() > 1 - 1e-5:
        return False
    verts, faces = mc_vertices_faces(occ)
    # center mesh for nicer viewing
    res = occ.shape[0]
    verts = verts - np.array([res / 2, res / 2, res / 2], dtype=np.float32)
    write_obj(out_path, verts, faces)
    return True

def render_projections(occ: np.ndarray) -> Image.Image:
    # occ: (D,H,W) bool/0-1
    # XY: max over Z, XZ: max over Y, YZ: max over X
    xy = occ.max(axis=0)
    xz = occ.max(axis=1)
    yz = occ.max(axis=2)

    def to_img(a):
        a = (a.astype(np.uint8) * 255)
        return Image.fromarray(a, mode="L").convert("RGB")

    im_xy = to_img(xy)
    im_xz = to_img(xz)
    im_yz = to_img(yz)

    w, h = im_xy.size
    out = Image.new("RGB", (w, h * 3))
    out.paste(im_xy, (0, 0))
    out.paste(im_xz, (0, h))
    out.paste(im_yz, (0, h * 2))
    return out

def save_grid(images, out_path: str, cols: int = 8):
    if len(images) == 0:
        return
    cols = min(cols, len(images))
    rows = (len(images) + cols - 1) // cols
    tw, th = images[0].size
    grid = Image.new("RGB", (cols * tw, rows * th))
    for i, im in enumerate(images):
        r = i // cols
        c = i % cols
        grid.paste(im, (c * tw, r * th))
    grid.save(out_path)

def load_npz_voxels(data_root: str, split: str) -> torch.Tensor:
    path = os.path.join(data_root, f"{split}.npz")
    npz = np.load(path)
    # try common keys
    key = None
    for k in ["voxels", "x", "data", "arr_0"]:
        if k in npz:
            key = k
            break
    if key is None:
        key = npz.files[0]
    v = npz[key]  # (N,D,H,W) or (N,1,D,H,W)
    if v.ndim == 4:
        v = v[:, None, ...]
    v = v.astype(np.float32)
    return torch.from_numpy(v)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--n_meshes", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--device", type=str, default="cuda")

    # NEW sampling modes
    ap.add_argument("--sample_mode", type=str, default="prior",
                    choices=["prior", "posterior", "posterior_noise", "interpolate"])
    ap.add_argument("--data_root", type=str, default=None, help="required for posterior/interpolate")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--posterior_sigma", type=float, default=0.20)
    ap.add_argument("--interp_steps", type=int, default=12)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    proj_dir = os.path.join(args.out_dir, "projections")
    mesh_dir = os.path.join(args.out_dir, "meshes")
    os.makedirs(proj_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})

    resolution = int(cfg.get("resolution", 64))
    latent_dim = int(cfg.get("latent_dim", 128))
    base_ch = int(cfg.get("base_ch", 56))

    model = VAE3D(resolution=resolution, latent_dim=latent_dim, base_ch=base_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # --- sample z ---
    if args.sample_mode == "prior":
        z = torch.randn(args.n_samples, latent_dim, device=device)

    else:
        if args.data_root is None:
            raise SystemExit("For sample_mode != prior, pass --data_root (e.g. data\\houses3k_vox64)")

        vox = load_npz_voxels(args.data_root, args.split)  # (N,1,D,H,W)
        N = vox.shape[0]
        need = max(args.n_samples, 2)
        idx = torch.randint(0, N, (need,))
        x = vox[idx].to(device)

        mu, logvar = model.encode(x)

        if args.sample_mode == "posterior":
            z = mu[:args.n_samples]

        elif args.sample_mode == "posterior_noise":
            eps = torch.randn_like(mu)
            z = (mu + args.posterior_sigma * eps)[:args.n_samples]

        elif args.sample_mode == "interpolate":
            z1 = mu[0]
            z2 = mu[1]
            ts = torch.linspace(0, 1, steps=args.interp_steps, device=device)
            z = torch.stack([(1 - t) * z1 + t * z2 for t in ts], dim=0)

    # decode -> logits -> occupancy
    logits = model.decode(z)  # (B,1,D,H,W)
    probs = torch.sigmoid(logits)
    occ = (probs[:, 0] > args.threshold).cpu().numpy().astype(np.uint8)  # (B,D,H,W)

    # projections grid
    imgs = [render_projections(occ[i]) for i in range(occ.shape[0])]
    save_grid(imgs, os.path.join(proj_dir, "samples_grid.png"), cols=8)

    # meshes
    saved = 0
    for i in range(occ.shape[0]):
        if saved >= args.n_meshes:
            break
        out_path = os.path.join(mesh_dir, f"sample_{i:03d}.obj")
        ok = export_mesh_from_occ(occ[i], out_path)
        if ok:
            saved += 1

    print(f"Saved grid + {saved} meshes to {os.path.abspath(args.out_dir)} (mode={args.sample_mode})")

if __name__ == "__main__":
    main()