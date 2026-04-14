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
from logging_utils import append_jsonl, build_run_manifest, save_manifest
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
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise SystemExit("epochs, batch_size, lr must be > 0")
    if args.weight_decay < 0 or args.dropout < 0 or args.grad_clip < 0:
        raise SystemExit("weight_decay, dropout, grad_clip must be >= 0")
    if args.d_model <= 0 or args.nhead <= 0 or args.num_layers <= 0 or args.ff_mult <= 0:
        raise SystemExit("Transformer dimensions must be > 0")
    if args.num_workers < 0 or args.early_stopping_patience < 0 or args.min_delta < 0:
        raise SystemExit("num_workers, early_stopping_patience, min_delta must be >= 0")
    if args.save_every <= 0:
        raise SystemExit("save_every must be > 0")
    if args.condition_mode == "npz_fields" and not parse_condition_fields(args.condition_fields):
        raise SystemExit("For npz_fields conditioning, pass --condition_fields")


def save_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    args: argparse.Namespace,
    epoch: int,
    best_val_loss: float,
    best_val_accuracy: float,
    best_val_perplexity: float,
    token_grid_shape: tuple[int, int, int],
    condition_fields: list[str],
    condition_vocab_sizes: list[int],
    codebook_size: int,
    run_manifest: dict,
    checkpoint_role: str,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": vars(args),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_accuracy": float(best_val_accuracy),
        "best_val_perplexity": float(best_val_perplexity),
        "token_grid_shape": tuple(int(v) for v in token_grid_shape),
        "condition_fields": condition_fields,
        "condition_vocab_sizes": [int(v) for v in condition_vocab_sizes],
        "codebook_size": int(codebook_size),
        "checkpoint_role": checkpoint_role,
        "artifact_family": "latent_prior",
        "run_manifest": run_manifest,
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--condition_mode", type=str, default="none", choices=["none", "shape_stats", "house_attributes", "npz_fields"])
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
    for p in vqvae.parameters():
        p.requires_grad_(False)
    resolution = int(vq_cfg.get("resolution", 64))

    train_ds = VoxelNPZDataset(os.path.join(args.data_root, "train.npz"), resolution=resolution, augment=False, seed=args.seed)
    val_ds = VoxelNPZDataset(os.path.join(args.data_root, "val.npz"), resolution=resolution, augment=False, seed=args.seed)
    if args.overfit_n > 0:
        for ds in (train_ds, val_ds):
            cap = min(len(ds), args.overfit_n)
            ds.voxels = ds.voxels[:cap]
            if ds.paths is not None:
                ds.paths = ds.paths[:cap]
            for k, v in ds.extra_arrays.items():
                ds.extra_arrays[k] = v[:cap]
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("Training or validation split is empty after filtering")

    condition_fields, condition_vocab_sizes = infer_condition_vocab_sizes(train_ds, mode=args.condition_mode, fields=parse_condition_fields(args.condition_fields), num_bins=args.condition_bins)
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

    generator = torch.Generator(); generator.manual_seed(args.seed)
    worker_init_fn = seed_worker if args.num_workers > 0 else None
    persistent_workers = bool(args.num_workers > 0)
    train_loader = DataLoader(IndexedDataset(train_ds), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_fn, persistent_workers=persistent_workers, generator=generator)
    val_loader = DataLoader(IndexedDataset(val_ds), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory, worker_init_fn=worker_init_fn, persistent_workers=persistent_workers)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_stub = args.run_name.strip() or "latent_prior"
    run_dir = os.path.join(args.out_dir, f"{run_stub}_{timestamp}")
    ensure_dir(run_dir)

    config_to_save = vars(args).copy()
    config_to_save["codebook_size"] = int(vq_cfg.get("codebook_size", 512))
    config_to_save["token_grid_shape"] = list(vqvae.token_grid_shape)
    config_to_save["condition_fields"] = condition_fields
    config_to_save["condition_vocab_sizes"] = condition_vocab_sizes
    save_json(config_to_save, os.path.join(run_dir, "config.json"))

    run_manifest = build_run_manifest(stage="train_latent_prior", config=config_to_save, extra={"data_root": os.path.abspath(args.data_root), "vqvae_ckpt": os.path.abspath(args.vqvae_ckpt), "run_dir": os.path.abspath(run_dir), "artifact_family": "latent_prior"})
    save_manifest(run_dir, run_manifest)

    metrics_csv = os.path.join(run_dir, "metrics.csv")
    metrics_jsonl = os.path.join(run_dir, "metrics.jsonl")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_perplexity", "train_token_accuracy", "val_loss", "val_perplexity", "val_token_accuracy", "best_val_loss_so_far", "best_val_ppl_so_far"])

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_val_perplexity = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        prior.train()
        train_loss = train_ppl = train_acc = 0.0
        n_train = 0
        for x, idx in train_loader:
            x = x.to(device, non_blocking=True)
            idx = idx.to(device=device, dtype=torch.long)
            with torch.no_grad():
                token_ids = vqvae.flatten_token_grid(vqvae.encode_tokens(x))
                condition_ids = gather_condition_ids(train_ds, idx, x, mode=args.condition_mode, fields=condition_fields, num_bins=args.condition_bins)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss, aux = prior.compute_loss(token_ids, condition_ids=condition_ids)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(prior.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update()
            train_loss += float(loss.item()); train_ppl += float(aux["perplexity"]); train_acc += float(aux["token_accuracy"]); n_train += 1
        sched.step()
        train_loss /= max(n_train, 1); train_ppl /= max(n_train, 1); train_acc /= max(n_train, 1)
        lr_cur = float(opt.param_groups[0]["lr"])

        prior.eval()
        val_loss = val_ppl = val_acc = 0.0
        n_val = 0
        with torch.no_grad():
            for x, idx in val_loader:
                x = x.to(device, non_blocking=True)
                idx = idx.to(device=device, dtype=torch.long)
                token_ids = vqvae.flatten_token_grid(vqvae.encode_tokens(x))
                condition_ids = gather_condition_ids(val_ds, idx, x, mode=args.condition_mode, fields=condition_fields, num_bins=args.condition_bins)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    loss, aux = prior.compute_loss(token_ids, condition_ids=condition_ids)
                val_loss += float(loss.item()); val_ppl += float(aux["perplexity"]); val_acc += float(aux["token_accuracy"]); n_val += 1
        val_loss /= max(n_val, 1); val_ppl /= max(n_val, 1); val_acc /= max(n_val, 1)

        improved_loss = val_loss < (best_val_loss - args.min_delta)
        improved_ppl = val_ppl < best_val_perplexity
        if improved_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            best_val_accuracy = max(best_val_accuracy, val_acc)
        best_val_perplexity = min(best_val_perplexity, val_ppl)

        row = {"epoch": epoch, "lr": lr_cur, "train_loss": train_loss, "train_perplexity": train_ppl, "train_token_accuracy": train_acc, "val_loss": val_loss, "val_perplexity": val_ppl, "val_token_accuracy": val_acc, "best_val_loss": best_val_loss, "best_val_perplexity": best_val_perplexity}
        append_jsonl(metrics_jsonl, row)
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, lr_cur, train_loss, train_ppl, train_acc, val_loss, val_ppl, val_acc, best_val_loss, best_val_perplexity])

        save_checkpoint(os.path.join(run_dir, "last.pt"), model=prior, optimizer=opt, scheduler=sched, args=args, epoch=epoch, best_val_loss=best_val_loss, best_val_accuracy=best_val_accuracy, best_val_perplexity=best_val_perplexity, token_grid_shape=tuple(int(v) for v in vqvae.token_grid_shape), condition_fields=condition_fields, condition_vocab_sizes=condition_vocab_sizes, codebook_size=int(vq_cfg.get("codebook_size", 512)), run_manifest=run_manifest, checkpoint_role="last")
        if improved_loss:
            save_checkpoint(os.path.join(run_dir, "best.pt"), model=prior, optimizer=opt, scheduler=sched, args=args, epoch=epoch, best_val_loss=best_val_loss, best_val_accuracy=best_val_accuracy, best_val_perplexity=best_val_perplexity, token_grid_shape=tuple(int(v) for v in vqvae.token_grid_shape), condition_fields=condition_fields, condition_vocab_sizes=condition_vocab_sizes, codebook_size=int(vq_cfg.get("codebook_size", 512)), run_manifest=run_manifest, checkpoint_role="best_by_val_loss")
            save_checkpoint(os.path.join(run_dir, "best_by_val_loss.pt"), model=prior, optimizer=opt, scheduler=sched, args=args, epoch=epoch, best_val_loss=best_val_loss, best_val_accuracy=best_val_accuracy, best_val_perplexity=best_val_perplexity, token_grid_shape=tuple(int(v) for v in vqvae.token_grid_shape), condition_fields=condition_fields, condition_vocab_sizes=condition_vocab_sizes, codebook_size=int(vq_cfg.get("codebook_size", 512)), run_manifest=run_manifest, checkpoint_role="best_by_val_loss")
        if improved_ppl:
            save_checkpoint(os.path.join(run_dir, "best_by_val_ppl.pt"), model=prior, optimizer=opt, scheduler=sched, args=args, epoch=epoch, best_val_loss=best_val_loss, best_val_accuracy=best_val_accuracy, best_val_perplexity=best_val_perplexity, token_grid_shape=tuple(int(v) for v in vqvae.token_grid_shape), condition_fields=condition_fields, condition_vocab_sizes=condition_vocab_sizes, codebook_size=int(vq_cfg.get("codebook_size", 512)), run_manifest=run_manifest, checkpoint_role="best_by_val_ppl")
        if epoch % args.save_every == 0 or epoch == 1 or epoch == args.epochs:
            save_checkpoint(os.path.join(run_dir, f"epoch_{epoch:03d}.pt"), model=prior, optimizer=opt, scheduler=sched, args=args, epoch=epoch, best_val_loss=best_val_loss, best_val_accuracy=best_val_accuracy, best_val_perplexity=best_val_perplexity, token_grid_shape=tuple(int(v) for v in vqvae.token_grid_shape), condition_fields=condition_fields, condition_vocab_sizes=condition_vocab_sizes, codebook_size=int(vq_cfg.get("codebook_size", 512)), run_manifest=run_manifest, checkpoint_role=f"epoch_{epoch:03d}")
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            break

    summary = {"best_val_loss": float(best_val_loss), "best_val_accuracy": float(best_val_accuracy), "best_val_perplexity": float(best_val_perplexity), "run_dir": os.path.abspath(run_dir), "artifact_aliases": {"last": "last.pt", "best_by_val_loss": "best_by_val_loss.pt", "best_by_val_ppl": "best_by_val_ppl.pt"}}
    save_json(summary, os.path.join(run_dir, "training_summary.json"))
    print("Done. Run directory:", os.path.abspath(run_dir))


if __name__ == "__main__":
    main()
