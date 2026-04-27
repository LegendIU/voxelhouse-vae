from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader

from dataset import VoxelNPZDataset
from mlops import MlflowLogger
from training_utils import (
    estimate_pos_weight,
    make_grad_scaler,
    save_recon_grid as _save_recon_grid_with_forward,
    seed_worker,
    set_global_seed,
)
from utils import choose_device, compute_iou, dice_loss_from_logits, ensure_dir, save_json
from vqvae_3d import VQVAE3D


@torch.no_grad()
def save_recon_grid(
    model: torch.nn.Module,
    batch: torch.Tensor,
    out_path: str,
    threshold: float = 0.5,
) -> bool:
    model.eval()

    def forward(x: torch.Tensor) -> torch.Tensor:
        return model(x).logits

    return _save_recon_grid_with_forward(forward, batch, out_path, threshold=threshold)


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
    if args.grad_clip < 0:
        raise SystemExit("--grad_clip must be >= 0")
    if args.dice_weight < 0:
        raise SystemExit("--dice_weight must be >= 0")
    if args.commitment_cost < 0:
        raise SystemExit("--commitment_cost must be >= 0")
    if args.vq_loss_weight < 0:
        raise SystemExit("--vq_loss_weight must be >= 0")
    if args.codebook_size <= 1:
        raise SystemExit("--codebook_size must be > 1")
    if args.embedding_dim <= 0:
        raise SystemExit("--embedding_dim must be > 0")
    if args.save_every <= 0:
        raise SystemExit("--save_every must be > 0")
    if args.early_stopping_patience < 0:
        raise SystemExit("--early_stopping_patience must be >= 0")
    if args.min_delta < 0:
        raise SystemExit("--min_delta must be >= 0")


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
    epoch: int,
    pos_weight: float,
    best_val_loss: float,
    best_val_iou: float,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": vars(args),
        "pos_weight": float(pos_weight),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_iou": float(best_val_iou),
        "token_grid_shape": tuple(int(v) for v in model.token_grid_shape),
    }
    torch.save(ckpt, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--base_ch", type=int, default=48)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--vq_loss_weight", type=float, default=1.0)

    parser.add_argument("--dice_weight", type=float, default=0.5)
    parser.add_argument("--overfit_n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Mixed precision on CUDA")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=25)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--recon_threshold", type=float, default=0.5)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow_experiment", type=str, default="voxelhouse-vae")
    parser.add_argument("--mlflow_tracking_uri", type=str, default="")

    args = parser.parse_args()
    validate_args(args)

    set_global_seed(args.seed)

    device = choose_device(args.device)
    use_amp = bool(args.amp) and (device.type == "cuda")
    pin_memory = bool(args.pin_memory) or (device.type == "cuda")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"run_{timestamp}")
    ensure_dir(run_dir)
    save_json(vars(args), os.path.join(run_dir, "config.json"))
    mlf = MlflowLogger.create(
        enabled=bool(args.mlflow),
        experiment_name=args.mlflow_experiment,
        run_name=f"train_3d_vqvae_{timestamp}",
        tracking_uri=(args.mlflow_tracking_uri or None),
        tags={"script": "train_3d_vqvae", "model": "vqvae3d"},
    )
    mlf.log_params(vars(args))

    train_ds = VoxelNPZDataset(
        os.path.join(args.data_root, "train.npz"),
        resolution=args.resolution,
        augment=True,
        seed=args.seed,
    )
    val_ds = VoxelNPZDataset(
        os.path.join(args.data_root, "val.npz"),
        resolution=args.resolution,
        augment=False,
        seed=args.seed,
    )

    if args.overfit_n and args.overfit_n > 0:
        train_ds.voxels = train_ds.voxels[: args.overfit_n]
        if train_ds.paths is not None:
            train_ds.paths = train_ds.paths[: args.overfit_n]
        for key, value in train_ds.extra_arrays.items():
            train_ds.extra_arrays[key] = value[: args.overfit_n]

        cap = min(len(val_ds), args.overfit_n)
        val_ds.voxels = val_ds.voxels[:cap]
        if val_ds.paths is not None:
            val_ds.paths = val_ds.paths[:cap]
        for key, value in val_ds.extra_arrays.items():
            val_ds.extra_arrays[key] = value[:cap]

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

    model = VQVAE3D(
        resolution=args.resolution,
        base_ch=args.base_ch,
        embedding_dim=args.embedding_dim,
        codebook_size=args.codebook_size,
        commitment_cost=args.commitment_cost,
    ).to(device)

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
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "train_bce",
                "train_dice",
                "train_vq",
                "train_codebook",
                "train_commitment",
                "train_perplexity",
                "val_loss",
                "val_bce",
                "val_dice",
                "val_vq",
                "val_codebook",
                "val_commitment",
                "val_perplexity",
                "val_iou",
                "best_val_iou_so_far",
            ]
        )

    best_val_loss = float("inf")
    best_val_iou = -float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = train_bce = train_dice = train_vq = 0.0
        train_codebook = train_commitment = train_perplexity = 0.0
        n_train_batches = 0

        for x in train_loader:
            x = x.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                recon_bce = bce(out.logits, x)
                recon_dice = dice_loss_from_logits(out.logits, x)
                loss = recon_bce + args.dice_weight * recon_dice + args.vq_loss_weight * out.vq_loss

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
            train_vq += float(out.vq_loss.item())
            train_codebook += float(out.codebook_loss.item())
            train_commitment += float(out.commitment_loss.item())
            train_perplexity += float(out.perplexity.item())
            n_train_batches += 1

        sched.step()

        train_loss /= max(n_train_batches, 1)
        train_bce /= max(n_train_batches, 1)
        train_dice /= max(n_train_batches, 1)
        train_vq /= max(n_train_batches, 1)
        train_codebook /= max(n_train_batches, 1)
        train_commitment /= max(n_train_batches, 1)
        train_perplexity /= max(n_train_batches, 1)
        lr_cur = float(opt.param_groups[0]["lr"])

        model.eval()
        val_loss = val_bce = val_dice = val_vq = 0.0
        val_codebook = val_commitment = val_perplexity = val_iou = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    out = model(x)
                    recon_bce = bce(out.logits, x)
                    recon_dice = dice_loss_from_logits(out.logits, x)
                    loss = recon_bce + args.dice_weight * recon_dice + args.vq_loss_weight * out.vq_loss

                val_loss += float(loss.item())
                val_bce += float(recon_bce.item())
                val_dice += float(recon_dice.item())
                val_vq += float(out.vq_loss.item())
                val_codebook += float(out.codebook_loss.item())
                val_commitment += float(out.commitment_loss.item())
                val_perplexity += float(out.perplexity.item())
                val_iou += compute_iou(out.logits, x, threshold=args.recon_threshold)
                n_val_batches += 1

        val_loss /= max(n_val_batches, 1)
        val_bce /= max(n_val_batches, 1)
        val_dice /= max(n_val_batches, 1)
        val_vq /= max(n_val_batches, 1)
        val_codebook /= max(n_val_batches, 1)
        val_commitment /= max(n_val_batches, 1)
        val_perplexity /= max(n_val_batches, 1)
        val_iou /= max(n_val_batches, 1)

        improved_iou = val_iou > (best_val_iou + args.min_delta)
        if improved_iou:
            best_val_iou = val_iou
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(
            f"Epoch {epoch:03d} | lr {lr_cur:.2e} | "
            f"train {train_loss:.4f} (bce {train_bce:.4f}, dice {train_dice:.4f}, vq {train_vq:.4f}) | "
            f"val {val_loss:.4f} (bce {val_bce:.4f}, dice {val_dice:.4f}, vq {val_vq:.4f}) | "
            f"ppl {val_perplexity:.2f} | IoU {val_iou:.4f} | best IoU {best_val_iou:.4f}"
        )

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    lr_cur,
                    train_loss,
                    train_bce,
                    train_dice,
                    train_vq,
                    train_codebook,
                    train_commitment,
                    train_perplexity,
                    val_loss,
                    val_bce,
                    val_dice,
                    val_vq,
                    val_codebook,
                    val_commitment,
                    val_perplexity,
                    val_iou,
                    best_val_iou,
                ]
            )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "train_perplexity": train_perplexity,
                "val_perplexity": val_perplexity,
            }
        )
        mlf.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "best_val_iou": best_val_iou,
                "val_perplexity": val_perplexity,
            },
            step=epoch,
        )

        save_checkpoint(
            os.path.join(run_dir, "last.pt"),
            model=model,
            optimizer=opt,
            scheduler=sched,
            args=args,
            epoch=epoch,
            pos_weight=float(pos_weight.item()),
            best_val_loss=best_val_loss,
            best_val_iou=best_val_iou,
        )

        if improved_iou:
            save_checkpoint(
                os.path.join(run_dir, "best.pt"),
                model=model,
                optimizer=opt,
                scheduler=sched,
                args=args,
                epoch=epoch,
                pos_weight=float(pos_weight.item()),
                best_val_loss=best_val_loss,
                best_val_iou=best_val_iou,
            )

        if epoch % args.save_every == 0 or epoch == 1 or epoch == args.epochs or improved_iou:
            if preview_batch is not None:
                saved = save_recon_grid(
                    model,
                    preview_batch,
                    os.path.join(run_dir, f"epoch_{epoch:03d}_recon.png"),
                    threshold=args.recon_threshold,
                )
                if not saved and epoch == 1:
                    print("[WARN] matplotlib is not installed; recon preview images are disabled")

            save_checkpoint(
                os.path.join(run_dir, f"epoch_{epoch:03d}.pt"),
                model=model,
                optimizer=opt,
                scheduler=sched,
                args=args,
                epoch=epoch,
                pos_weight=float(pos_weight.item()),
                best_val_loss=best_val_loss,
                best_val_iou=best_val_iou,
            )

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch:03d}: "
                f"val_iou has not improved by at least {args.min_delta:.1e} for "
                f"{args.early_stopping_patience} epochs."
            )
            break

    epochs = [int(h["epoch"]) for h in history]
    train_curve = [h["train_loss"] for h in history]
    val_curve = [h["val_loss"] for h in history]
    iou_curve = [h["val_iou"] for h in history]
    train_ppl_curve = [h["train_perplexity"] for h in history]
    val_ppl_curve = [h["val_perplexity"] for h in history]

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.plot(epochs, train_curve, label="train_loss")
        plt.plot(epochs, val_curve, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=150)
        plt.close(fig)

        fig = plt.figure()
        plt.plot(epochs, train_ppl_curve, label="train_perplexity")
        plt.plot(epochs, val_ppl_curve, label="val_perplexity")
        plt.xlabel("Epoch")
        plt.ylabel("Perplexity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "perplexity_curve.png"), dpi=150)
        plt.close(fig)

        fig = plt.figure()
        plt.plot(epochs, iou_curve, label="val_iou")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "iou_curve.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        print("[WARN] matplotlib is not installed; training curves were not saved")

    summary = {
        "best_val_loss": float(best_val_loss),
        "best_val_iou": float(best_val_iou),
        "epochs_completed": int(len(history)),
        "stopped_early": bool(len(history) < args.epochs),
    }
    save_json(summary, os.path.join(run_dir, "training_summary.json"))
    mlf.log_artifact(os.path.join(run_dir, "config.json"), artifact_path="config")
    mlf.log_artifact(os.path.join(run_dir, "metrics.csv"), artifact_path="metrics")
    mlf.log_artifact(os.path.join(run_dir, "training_summary.json"), artifact_path="metrics")
    mlf.close()

    print("Done. Run directory:", os.path.abspath(run_dir))


if __name__ == "__main__":
    main()
