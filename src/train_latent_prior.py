from __future__ import annotations

import argparse
import csv
import os
import random
from datetime import UTC, datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from conditioning import gather_condition_ids, infer_condition_vocab_sizes, parse_condition_fields
from dataset import VoxelNPZDataset
from latent_transformer import LatentTokenTransformer
from model_loading import load_vqvae_model
from utils import choose_device, ensure_dir, save_json


class IndexedDataset(Dataset):
    def __init__(self, base: VoxelNPZDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.base[idx], int(idx)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    info = get_worker_info()
    if info is not None:
        base = getattr(info.dataset, "base", None)
        if base is not None and hasattr(base, "rng"):
            base.rng = np.random.default_rng(worker_seed)


def make_grad_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device="cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise SystemExit("--epochs must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be > 0")
    if args.lr <= 0:
        raise SystemExit("--lr must be > 0")
    if args.weight_decay < 0:
        raise SystemExit("--weight_decay must be >= 0")
    if args.d_model <= 0:
        raise SystemExit("--d_model must be > 0")
    if args.nhead <= 0:
        raise SystemExit("--nhead must be > 0")
    if args.num_layers <= 0:
        raise SystemExit("--num_layers must be > 0")
    if args.dropout < 0:
        raise SystemExit("--dropout must be >= 0")
    if args.ff_mult <= 0:
        raise SystemExit("--ff_mult must be > 0")
    if args.num_workers < 0:
        raise SystemExit("--num_workers must be >= 0")
    if args.grad_clip < 0:
        raise SystemExit("--grad_clip must be >= 0")
    if args.save_every <= 0:
        raise SystemExit("--save_every must be > 0")
    if args.early_stopping_patience < 0:
        raise SystemExit("--early_stopping_patience must be >= 0")
    if args.min_delta < 0:
        raise SystemExit("--min_delta must be >= 0")
    if args.condition_mode not in {"none", "shape_stats", "house_attributes", "npz_fields"}:
        raise SystemExit("--condition_mode must be one of: none, shape_stats, house_attributes, npz_fields")
    if args.condition_mode == "npz_fields" and not parse_condition_fields(args.condition_fields):
        raise SystemExit("For --condition_mode npz_fields, pass --condition_fields")
    if args.condition_mode in {"shape_stats", "house_attributes"} and args.condition_bins <= 1:
        raise SystemExit("--condition_bins must be > 1 for shape_stats/house_attributes conditioning")


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
    epoch: int,
    best_val_loss: float,
    best_val_perplexity: float,
    best_val_accuracy: float,
    token_grid_shape: tuple[int, int, int],
    condition_fields: list[str],
    condition_vocab_sizes: list[int],
    codebook_size: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": vars(args),
            "epoch": int(epoch),
            "best_val_loss": float(best_val_loss),
            "best_val_perplexity": float(best_val_perplexity),
            "best_val_accuracy": float(best_val_accuracy),
            "token_grid_shape": tuple(int(v) for v in token_grid_shape),
            "condition_fields": condition_fields,
            "condition_vocab_sizes": [int(v) for v in condition_vocab_sizes],
            "codebook_size": int(codebook_size),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ff_mult", type=int, default=4)

    parser.add_argument(
        "--condition_mode",
        type=str,
        default="none",
        choices=["none", "shape_stats", "house_attributes", "npz_fields"],
    )
    parser.add_argument("--condition_fields", type=str, default="")
    parser.add_argument("--condition_bins", type=int, default=8)

    parser.add_argument("--overfit_n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=1e-4)

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

    vqvae, _, vq_cfg = load_vqvae_model(args.vqvae_ckpt, device=device)
    for param in vqvae.parameters():
        param.requires_grad_(False)
    resolution = int(vq_cfg.get("resolution", 64))

    train_ds = VoxelNPZDataset(
        os.path.join(args.data_root, "train.npz"),
        resolution=resolution,
        augment=False,
        seed=args.seed,
    )
    val_ds = VoxelNPZDataset(
        os.path.join(args.data_root, "val.npz"),
        resolution=resolution,
        augment=False,
        seed=args.seed,
    )

    if args.overfit_n and args.overfit_n > 0:
        cap = args.overfit_n
        train_ds.voxels = train_ds.voxels[:cap]
        if train_ds.paths is not None:
            train_ds.paths = train_ds.paths[:cap]
        for key, value in train_ds.extra_arrays.items():
            train_ds.extra_arrays[key] = value[:cap]

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

    condition_fields, condition_vocab_sizes = infer_condition_vocab_sizes(
        train_ds,
        mode=args.condition_mode,
        fields=parse_condition_fields(args.condition_fields),
        num_bins=args.condition_bins,
    )

    prior = LatentTokenTransformer(
        codebook_size=int(vq_cfg.get("codebook_size", 512)),
        token_grid_shape=tuple(int(v) for v in vqvae.token_grid_shape),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        ff_mult=args.ff_mult,
        condition_vocab_sizes=condition_vocab_sizes,
    ).to(device)

    opt = torch.optim.AdamW(prior.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = make_grad_scaler(use_amp)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    worker_init_fn = seed_worker if args.num_workers > 0 else None
    persistent_workers = bool(args.num_workers > 0)

    train_loader = DataLoader(
        IndexedDataset(train_ds),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        IndexedDataset(val_ds),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"run_{timestamp}")
    ensure_dir(run_dir)
    config_to_save = vars(args).copy()
    config_to_save["codebook_size"] = int(vq_cfg.get("codebook_size", 512))
    config_to_save["token_grid_shape"] = list(vqvae.token_grid_shape)
    config_to_save["condition_fields"] = condition_fields
    config_to_save["condition_vocab_sizes"] = condition_vocab_sizes
    save_json(config_to_save, os.path.join(run_dir, "config.json"))

    metrics_path = os.path.join(run_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "train_perplexity",
                "train_token_accuracy",
                "val_loss",
                "val_perplexity",
                "val_token_accuracy",
                "best_val_loss_so_far",
                "best_val_perplexity_so_far",
            ]
        )

    best_val_loss = float("inf")
    best_val_perplexity = float("inf")
    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        prior.train()
        train_loss = train_perplexity = train_accuracy = 0.0
        n_train_batches = 0

        for x, idx in train_loader:
            x = x.to(device, non_blocking=True)
            idx = idx.to(device=device, dtype=torch.long)

            with torch.no_grad():
                token_ids = vqvae.flatten_token_grid(vqvae.encode_tokens(x))
                condition_ids = gather_condition_ids(
                    train_ds,
                    idx,
                    x,
                    mode=args.condition_mode,
                    fields=condition_fields,
                    num_bins=args.condition_bins,
                )

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss, aux = prior.compute_loss(token_ids, condition_ids=condition_ids)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(prior.parameters(), args.grad_clip)

            scaler.step(opt)
            scaler.update()

            train_loss += float(loss.item())
            train_perplexity += float(aux["perplexity"])
            train_accuracy += float(aux["token_accuracy"])
            n_train_batches += 1

        sched.step()

        train_loss /= max(n_train_batches, 1)
        train_perplexity /= max(n_train_batches, 1)
        train_accuracy /= max(n_train_batches, 1)
        lr_cur = float(opt.param_groups[0]["lr"])

        prior.eval()
        val_loss = val_perplexity = val_accuracy = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for x, idx in val_loader:
                x = x.to(device, non_blocking=True)
                idx = idx.to(device=device, dtype=torch.long)

                token_ids = vqvae.flatten_token_grid(vqvae.encode_tokens(x))
                condition_ids = gather_condition_ids(
                    val_ds,
                    idx,
                    x,
                    mode=args.condition_mode,
                    fields=condition_fields,
                    num_bins=args.condition_bins,
                )

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    loss, aux = prior.compute_loss(token_ids, condition_ids=condition_ids)

                val_loss += float(loss.item())
                val_perplexity += float(aux["perplexity"])
                val_accuracy += float(aux["token_accuracy"])
                n_val_batches += 1

        val_loss /= max(n_val_batches, 1)
        val_perplexity /= max(n_val_batches, 1)
        val_accuracy /= max(n_val_batches, 1)

        improved = val_loss < (best_val_loss - args.min_delta)
        if improved:
            best_val_loss = val_loss
            best_val_perplexity = val_perplexity
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            best_val_accuracy = max(best_val_accuracy, val_accuracy)
            best_val_perplexity = min(best_val_perplexity, val_perplexity)

        print(
            f"Epoch {epoch:03d} | lr {lr_cur:.2e} | "
            f"train loss {train_loss:.4f}, ppl {train_perplexity:.2f}, acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f}, ppl {val_perplexity:.2f}, acc {val_accuracy:.4f} | "
            f"best val loss {best_val_loss:.4f}, best ppl {best_val_perplexity:.2f}"
        )

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    lr_cur,
                    train_loss,
                    train_perplexity,
                    train_accuracy,
                    val_loss,
                    val_perplexity,
                    val_accuracy,
                    best_val_loss,
                    best_val_perplexity,
                ]
            )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_perplexity": train_perplexity,
                "train_token_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_perplexity": val_perplexity,
                "val_token_accuracy": val_accuracy,
            }
        )

        save_checkpoint(
            os.path.join(run_dir, "last.pt"),
            model=prior,
            optimizer=opt,
            scheduler=sched,
            args=args,
            epoch=epoch,
            best_val_loss=best_val_loss,
            best_val_perplexity=best_val_perplexity,
            best_val_accuracy=best_val_accuracy,
            token_grid_shape=tuple(int(v) for v in vqvae.token_grid_shape),
            condition_fields=condition_fields,
            condition_vocab_sizes=condition_vocab_sizes,
            codebook_size=int(vq_cfg.get("codebook_size", 512)),
        )

        if improved:
            save_checkpoint(
                os.path.join(run_dir, "best.pt"),
                model=prior,
                optimizer=opt,
                scheduler=sched,
                args=args,
                epoch=epoch,
                best_val_loss=best_val_loss,
                best_val_perplexity=best_val_perplexity,
                best_val_accuracy=best_val_accuracy,
                token_grid_shape=tuple(int(v) for v in vqvae.token_grid_shape),
                condition_fields=condition_fields,
                condition_vocab_sizes=condition_vocab_sizes,
                codebook_size=int(vq_cfg.get("codebook_size", 512)),
            )
            save_checkpoint(
                os.path.join(run_dir, "best_by_val_ppl.pt"),
                model=prior,
                optimizer=opt,
                scheduler=sched,
                args=args,
                epoch=epoch,
                best_val_loss=best_val_loss,
                best_val_perplexity=best_val_perplexity,
                best_val_accuracy=best_val_accuracy,
                token_grid_shape=tuple(int(v) for v in vqvae.token_grid_shape),
                condition_fields=condition_fields,
                condition_vocab_sizes=condition_vocab_sizes,
                codebook_size=int(vq_cfg.get("codebook_size", 512)),
            )

        if epoch % args.save_every == 0 or epoch == 1 or epoch == args.epochs or improved:
            save_checkpoint(
                os.path.join(run_dir, f"epoch_{epoch:03d}.pt"),
                model=prior,
                optimizer=opt,
                scheduler=sched,
                args=args,
                epoch=epoch,
                best_val_loss=best_val_loss,
                best_val_perplexity=best_val_perplexity,
                best_val_accuracy=best_val_accuracy,
                token_grid_shape=tuple(int(v) for v in vqvae.token_grid_shape),
                condition_fields=condition_fields,
                condition_vocab_sizes=condition_vocab_sizes,
                codebook_size=int(vq_cfg.get("codebook_size", 512)),
            )

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch:03d}: "
                f"val_loss has not improved by at least {args.min_delta:.1e} for "
                f"{args.early_stopping_patience} epochs."
            )
            break

    epochs = [int(h["epoch"]) for h in history]
    train_loss_curve = [h["train_loss"] for h in history]
    val_loss_curve = [h["val_loss"] for h in history]
    train_ppl_curve = [h["train_perplexity"] for h in history]
    val_ppl_curve = [h["val_perplexity"] for h in history]
    train_acc_curve = [h["train_token_accuracy"] for h in history]
    val_acc_curve = [h["val_token_accuracy"] for h in history]

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.plot(epochs, train_loss_curve, label="train_loss")
        plt.plot(epochs, val_loss_curve, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
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
        plt.plot(epochs, train_acc_curve, label="train_token_accuracy")
        plt.plot(epochs, val_acc_curve, label="val_token_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "accuracy_curve.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        print("[WARN] matplotlib is not installed; training curves were not saved")

    summary = {
        "best_val_loss": float(best_val_loss),
        "best_val_perplexity": float(best_val_perplexity),
        "best_val_accuracy": float(best_val_accuracy),
        "epochs_completed": int(len(history)),
        "stopped_early": bool(len(history) < args.epochs),
        "condition_mode": str(args.condition_mode),
        "condition_fields": condition_fields,
    }
    save_json(summary, os.path.join(run_dir, "training_summary.json"))

    print("Done. Run directory:", os.path.abspath(run_dir))


if __name__ == "__main__":
    main()
