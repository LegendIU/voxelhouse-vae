from __future__ import annotations

import argparse
import json
import os
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import VoxelNPZDataset
from model_3d import VAE3D, kl_divergence
from utils import choose_device, compute_iou, dice_loss_from_logits


def _load_checkpoint(ckpt_path: str) -> dict:
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Backward compatibility with older PyTorch versions.
        return torch.load(ckpt_path, map_location="cpu")


def _neighbors_6(z: int, y: int, x: int, d: int, h: int, w: int):
    if z > 0:
        yield z - 1, y, x
    if z + 1 < d:
        yield z + 1, y, x
    if y > 0:
        yield z, y - 1, x
    if y + 1 < h:
        yield z, y + 1, x
    if x > 0:
        yield z, y, x - 1
    if x + 1 < w:
        yield z, y, x + 1


def connected_components_stats_single(mask: np.ndarray) -> tuple[int, float]:
    """
    Returns:
        (num_components, largest_component_ratio)
    largest_component_ratio = largest_component_voxels / total_occupied_voxels
    """
    occ = mask.astype(np.bool_)
    total_occ = int(occ.sum())
    if total_occ == 0:
        return 0, 0.0

    visited = np.zeros_like(occ, dtype=np.bool_)
    d, h, w = occ.shape
    num_components = 0
    largest = 0

    occ_coords = np.argwhere(occ)
    for start in occ_coords:
        z0, y0, x0 = int(start[0]), int(start[1]), int(start[2])
        if visited[z0, y0, x0]:
            continue

        num_components += 1
        q = deque([(z0, y0, x0)])
        visited[z0, y0, x0] = True
        size = 0

        while q:
            z, y, x = q.popleft()
            size += 1
            for nz, ny, nx in _neighbors_6(z, y, x, d, h, w):
                if occ[nz, ny, nx] and not visited[nz, ny, nx]:
                    visited[nz, ny, nx] = True
                    q.append((nz, ny, nx))

        if size > largest:
            largest = size

    return num_components, float(largest) / float(total_occ)


def voxel_precision_recall_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    eps: float = 1e-8,
) -> tuple[float, float]:
    pred = (torch.sigmoid(logits) > threshold).float()
    tgt = (target > 0.5).float()

    tp = float((pred * tgt).sum().item())
    fp = float((pred * (1.0 - tgt)).sum().item())
    fn = float(((1.0 - pred) * tgt).sum().item())

    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    return precision, recall


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be > 0")
    if not (0.0 < args.threshold < 1.0):
        raise SystemExit("--threshold must be in (0, 1)")
    if args.num_workers < 0:
        raise SystemExit("--num_workers must be >= 0")

    device = choose_device(args.device)
    use_amp = bool(args.amp) and (device.type == "cuda")
    pin_memory = bool(args.pin_memory) or (device.type == "cuda")

    ckpt = _load_checkpoint(args.ckpt)
    if "model" not in ckpt:
        raise SystemExit(f"Checkpoint does not contain 'model' weights: {args.ckpt}")

    cfg = ckpt.get("config", {})
    pos_weight = float(ckpt.get("pos_weight", 1.0))
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    resolution = int(cfg.get("resolution", 64))
    latent_dim = int(cfg.get("latent_dim", 128))
    base_ch = int(cfg.get("base_ch", 48))
    kl_weight = float(cfg.get("kl_weight", 5e-4))
    dice_weight = float(cfg.get("dice_weight", 0.0))

    ds = VoxelNPZDataset(
        os.path.join(args.data_root, f"{args.split}.npz"),
        resolution=resolution,
        augment=False,
    )
    if len(ds) == 0:
        raise SystemExit(f"Split '{args.split}' is empty at {args.data_root}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(args.num_workers > 0),
    )

    model = VAE3D(resolution=resolution, latent_dim=latent_dim, base_ch=base_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_dice = 0.0
    total_kl = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0

    total_components = 0.0
    total_lcc_ratio = 0.0

    n = 0

    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, mu, logvar = model(x)
            recon = bce(logits, x)
            dice = dice_loss_from_logits(logits, x)
            kl = kl_divergence(mu, logvar)
            loss = recon + dice_weight * dice + kl_weight * kl

        iou = compute_iou(logits, x, threshold=args.threshold)
        precision, recall = voxel_precision_recall_from_logits(logits, x, threshold=args.threshold)

        pred_mask = (torch.sigmoid(logits)[:, 0] > args.threshold).cpu().numpy().astype(np.uint8)

        bs = x.shape[0]
        batch_components = 0.0
        batch_lcc_ratio = 0.0
        for i in range(bs):
            num_cc, lcc_ratio = connected_components_stats_single(pred_mask[i])
            batch_components += float(num_cc)
            batch_lcc_ratio += float(lcc_ratio)

        total_loss += float(loss.item()) * bs
        total_recon += float(recon.item()) * bs
        total_dice += float(dice.item()) * bs
        total_kl += float(kl.item()) * bs
        total_iou += float(iou) * bs
        total_precision += float(precision) * bs
        total_recall += float(recall) * bs
        total_components += batch_components
        total_lcc_ratio += batch_lcc_ratio
        n += bs

    out = {
        "split": args.split,
        "n": n,
        "loss": total_loss / max(n, 1),
        "recon_bce": total_recon / max(n, 1),
        "recon_dice": total_dice / max(n, 1),
        "kl": total_kl / max(n, 1),
        "iou": total_iou / max(n, 1),
        "voxel_precision": total_precision / max(n, 1),
        "voxel_recall": total_recall / max(n, 1),
        "num_components": total_components / max(n, 1),
        "largest_component_ratio": total_lcc_ratio / max(n, 1),
        "pos_weight": pos_weight,
        "threshold": args.threshold,
        "resolution": resolution,
        "latent_dim": latent_dim,
        "base_ch": base_ch,
        "kl_weight": kl_weight,
        "dice_weight": dice_weight,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()