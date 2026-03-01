from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from dataset import VoxelNPZDataset
from model_3d import VAE3D, kl_divergence
from utils import choose_device, compute_iou, dice_loss_from_logits


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

    ckpt = torch.load(args.ckpt, map_location="cpu")
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

    ds = VoxelNPZDataset(os.path.join(args.data_root, f"{args.split}.npz"), resolution=resolution, augment=False)
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

    total_loss = total_recon = total_dice = total_kl = total_iou = 0.0
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
        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_recon += float(recon.item()) * bs
        total_dice += float(dice.item()) * bs
        total_kl += float(kl.item()) * bs
        total_iou += float(iou) * bs
        n += bs

    out = {
        "split": args.split,
        "n": n,
        "loss": total_loss / max(n, 1),
        "recon_bce": total_recon / max(n, 1),
        "recon_dice": total_dice / max(n, 1),
        "kl": total_kl / max(n, 1),
        "iou": total_iou / max(n, 1),
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
