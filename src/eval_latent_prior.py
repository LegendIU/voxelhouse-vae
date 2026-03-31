from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from conditioning import gather_condition_ids
from dataset import VoxelNPZDataset
from model_loading import load_latent_prior, load_vqvae_model
from utils import choose_device


class IndexedDataset(Dataset):
    def __init__(self, base: VoxelNPZDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.base[idx], int(idx)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_ckpt", required=True)
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be > 0")
    if args.num_workers < 0:
        raise SystemExit("--num_workers must be >= 0")

    device = choose_device(args.device)
    use_amp = bool(args.amp) and (device.type == "cuda")
    pin_memory = bool(args.pin_memory) or (device.type == "cuda")

    vqvae, _, vq_cfg = load_vqvae_model(args.vqvae_ckpt, device=device)
    prior, prior_ckpt, prior_cfg = load_latent_prior(args.prior_ckpt, device=device)

    condition_mode = str(prior_cfg.get("condition_mode", "none"))
    condition_fields = prior_ckpt.get("condition_fields", prior_cfg.get("condition_fields", []))
    condition_bins = int(prior_cfg.get("condition_bins", 8))

    ds = VoxelNPZDataset(
        os.path.join(args.data_root, f"{args.split}.npz"),
        resolution=int(vq_cfg.get("resolution", 64)),
        augment=False,
    )
    if len(ds) == 0:
        raise SystemExit(f"Split '{args.split}' is empty at {args.data_root}")

    loader = DataLoader(
        IndexedDataset(ds),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(args.num_workers > 0),
    )

    total_loss = 0.0
    total_acc = 0.0
    total_ppl = 0.0
    n_batches = 0
    token_count = 0

    for x, idx in loader:
        x = x.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long)

        token_ids = vqvae.flatten_token_grid(vqvae.encode_tokens(x))
        condition_ids = gather_condition_ids(
            ds,
            idx,
            x,
            mode=condition_mode,
            fields=condition_fields,
            num_bins=condition_bins,
        )

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            loss, aux = prior.compute_loss(token_ids, condition_ids=condition_ids)

        total_loss += float(loss.item())
        total_acc += float(aux["token_accuracy"])
        total_ppl += float(aux["perplexity"])
        n_batches += 1
        token_count += int(np.prod(token_ids.shape))

    out = {
        "split": args.split,
        "n_samples": len(ds),
        "n_batches": n_batches,
        "n_tokens": token_count,
        "loss": total_loss / max(n_batches, 1),
        "perplexity": total_ppl / max(n_batches, 1),
        "token_accuracy": total_acc / max(n_batches, 1),
        "condition_mode": condition_mode,
        "condition_fields": condition_fields,
        "token_grid_shape": list(vqvae.token_grid_shape),
        "codebook_size": int(vq_cfg.get("codebook_size", 512)),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
