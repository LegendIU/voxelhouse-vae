from __future__ import annotations

import argparse
import csv
import os
import random
from datetime import UTC, datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, get_worker_info

from dataset import VoxelNPZDataset
from model_3d import VAE3D, kl_divergence
from utils import ensure_dir, save_json


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
    dice = 1.0 - (num / den)
    return dice.mean()


@torch.no_grad()
def save_recon_grid(model: torch.nn.Module, batch: torch.Tensor, out_path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    model.eval()
    x = batch[:8]
    logits, _, _ = model(x)
    probs = torch.sigmoid(logits)

    def proj(v: torch.Tensor) -> np.ndarray:
        v = (v > 0.5).float()
        xy = v.max(dim=4).values.squeeze(1)
        return xy.cpu().numpy()

    gt = proj(x)
    rc = proj(probs)
    bsz = gt.shape[0]
    fig, axes = plt.subplots(2, bsz, figsize=(bsz * 2, 4))
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


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng = np.random.default_rng(worker_seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] CUDA device '{device_arg}' requested but CUDA is unavailable, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_arg)


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise SystemExit("--epochs must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be > 0")
    if args.lr <= 0:
        raise SystemExit("--lr must be > 0")
    if args.weight_decay < 0:
        raise SystemExit("--weight_decay must be >= 0")
    if args.resolution not in (32, 64):
        raise SystemExit("--resolution must be 32 or 64")
    if args.num_workers < 0:
        raise SystemExit("--num_workers must be >= 0")
    if args.kl_warmup_epochs < 0:
        raise SystemExit("--kl_warmup_epochs must be >= 0")
    if args.grad_clip < 0:
        raise SystemExit("--grad_clip must be >= 0")


def estimate_pos_weight(voxels: np.ndarray) -> float:
    if voxels.size == 0:
        raise ValueError("Training voxels array is empty")
    pos = float((voxels > 0).sum())
    neg = float(voxels.size - pos)
    if pos <= 0:
        raise ValueError("Training voxels contain no occupied cells")
    # Clamp the ratio to avoid exploding gradients on degenerate datasets.
    return min(neg / pos, 1e4)


def make_grad_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device="cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--base_ch", type=int, default=48)
    parser.add_argument("--kl_weight", type=float, default=5e-4)
    parser.add_argument("--kl_warmup_epochs", type=int, default=30, help="linearly warm up kl_weight over first N epochs")
    parser.add_argument("--dice_weight", type=float, default=0.5, help="extra dice loss weight on top of BCE")
    parser.add_argument("--overfit_n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--amp", action="store_true", help="mixed precision on CUDA")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()
    validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    use_amp = bool(args.amp) and (device.type == "cuda")
    pin_memory = bool(args.pin_memory) or (device.type == "cuda")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"run_{timestamp}")
    ensure_dir(run_dir)
    save_json(vars(args), os.path.join(run_dir, "config.json"))

    train_ds = VoxelNPZDataset(os.path.join(args.data_root, "train.npz"), resolution=args.resolution, augment=True, seed=args.seed)
    val_ds = VoxelNPZDataset(os.path.join(args.data_root, "val.npz"), resolution=args.resolution, augment=False, seed=args.seed)

    if args.overfit_n and args.overfit_n > 0:
        train_ds.voxels = train_ds.voxels[: args.overfit_n]
        if train_ds.paths is not None:
            train_ds.paths = train_ds.paths[: args.overfit_n]
        cap = min(len(val_ds), args.overfit_n)
        val_ds.voxels = val_ds.voxels[:cap]
        if val_ds.paths is not None:
            val_ds.paths = val_ds.paths[:cap]

    if len(train_ds) == 0:
        raise SystemExit("Training split is empty after filtering")
    if len(val_ds) == 0:
        raise SystemExit("Validation split is empty after filtering")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    worker_init_fn = seed_worker if args.num_workers > 0 else None
    persistent_workers = bool(args.num_workers > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
    )

    preview_batch = next(iter(val_loader), None)
    if preview_batch is not None:
        preview_batch = preview_batch.to(device, non_blocking=True)

    model = VAE3D(resolution=args.resolution, latent_dim=args.latent_dim, base_ch=args.base_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    pos_weight_value = estimate_pos_weight(train_ds.voxels)
    pos_count = float((train_ds.voxels > 0).sum())
    neg_count = float(train_ds.voxels.size - pos_count)
    pos_weight = torch.tensor([pos_weight_value], device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Estimated pos_weight={float(pos_weight.item()):.2f} (pos={pos_count:.0f}, neg={neg_count:.0f})")

    scaler = make_grad_scaler(use_amp)

    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "kl_w", "train_loss", "train_bce", "train_dice", "train_kl", "val_loss", "val_iou"])

    best_val = float("inf")
    history: list[tuple[int, float, float, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = train_bce = train_dice = train_kl = 0.0
        n_batches = 0

        if args.kl_warmup_epochs > 0:
            kl_w = args.kl_weight * min(1.0, epoch / float(args.kl_warmup_epochs))
        else:
            kl_w = args.kl_weight

        for x in train_loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits, mu, logvar = model(x)
                recon_bce = bce(logits, x)
                recon_dice = dice_loss_from_logits(logits, x)
                kl = kl_divergence(mu, logvar)
                loss = recon_bce + args.dice_weight * recon_dice + kl_w * kl

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            train_loss += float(loss.item())
            train_bce += float(recon_bce.item())
            train_dice += float(recon_dice.item())
            train_kl += float(kl.item())
            n_batches += 1

        sched.step()

        train_loss /= max(n_batches, 1)
        train_bce /= max(n_batches, 1)
        train_dice /= max(n_batches, 1)
        train_kl /= max(n_batches, 1)
        lr_cur = float(opt.param_groups[0]["lr"])

        model.eval()
        val_loss = val_iou = 0.0
        val_batches = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits, mu, logvar = model(x)
                    recon_bce = bce(logits, x)
                    recon_dice = dice_loss_from_logits(logits, x)
                    kl = kl_divergence(mu, logvar)
                    loss = recon_bce + args.dice_weight * recon_dice + kl_w * kl
                val_loss += float(loss.item())
                val_iou += compute_iou(logits, x)
                val_batches += 1

        val_loss /= max(val_batches, 1)
        val_iou /= max(val_batches, 1)

        print(
            f"Epoch {epoch:03d} | lr {lr_cur:.2e} | kl_w {kl_w:.2e} | "
            f"train {train_loss:.4f} (bce {train_bce:.4f}, dice {train_dice:.4f}, kl {train_kl:.4f}) | "
            f"val {val_loss:.4f} | IoU {val_iou:.3f}"
        )

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, lr_cur, kl_w, train_loss, train_bce, train_dice, train_kl, val_loss, val_iou])

        history.append((epoch, train_loss, val_loss, val_iou))

        if preview_batch is not None:
            saved = save_recon_grid(model, preview_batch, os.path.join(run_dir, f"epoch_{epoch:03d}_recon.png"))
            if not saved and epoch == 1:
                print("[WARN] matplotlib is not installed; recon preview images are disabled")

        ckpt = {
            "model": model.state_dict(),
            "config": vars(args),
            "pos_weight": float(pos_weight.item()),
            "epoch": epoch,
            "best_val": best_val,
        }
        torch.save(ckpt, os.path.join(run_dir, "last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            ckpt["best_val"] = best_val
            torch.save(ckpt, os.path.join(run_dir, "best.pt"))

    epochs = [h[0] for h in history]
    train_curve = [h[1] for h in history]
    val_curve = [h[2] for h in history]
    iou_curve = [h[3] for h in history]

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.plot(epochs, train_curve, label="train_loss")
        plt.plot(epochs, val_curve, label="val_loss")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=150)
        plt.close(fig)

        fig = plt.figure()
        plt.plot(epochs, iou_curve, label="val_iou")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("IoU")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "iou_curve.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        print("[WARN] matplotlib is not installed; training curves were not saved")

    print("Done. Run directory:", os.path.abspath(run_dir))


if __name__ == "__main__":
    main()
