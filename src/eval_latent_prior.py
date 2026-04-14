from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
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
    parser.add_argument("--out_dir", default="")
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
    codebook_size = int(vq_cfg.get("codebook_size", 512))

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
    total_tokens = 0
    n_batches = 0
    sequence_nll_sums: list[float] = []
    used_token_ids: set[int] = set()
    unique_tokens_per_sequence: list[float] = []

    for x, idx in loader:
        x = x.to(device, non_blocking=True)
        idx = idx.to(device=device, dtype=torch.long)

        token_ids = vqvae.flatten_token_grid(vqvae.encode_tokens(x))
        used_token_ids.update(int(v) for v in torch.unique(token_ids).cpu().tolist())
        unique_tokens_per_sequence.extend(
            [float(torch.unique(seq).numel()) for seq in token_ids]
        )

        condition_ids = gather_condition_ids(
            ds,
            idx,
            x,
            mode=condition_mode,
            fields=condition_fields,
            num_bins=condition_bins,
        )

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = prior(token_ids, condition_ids=condition_ids)
            loss = F.cross_entropy(logits.transpose(1, 2), token_ids.long())
            pred = torch.argmax(logits, dim=-1)
            token_acc = float((pred == token_ids).float().mean().item())
            perplexity = float(torch.exp(loss.detach()).item())

        token_nll = F.cross_entropy(
            logits.transpose(1, 2),
            token_ids.long(),
            reduction="none",
        )
        sequence_nll_sums.extend(token_nll.sum(dim=1).detach().cpu().tolist())

        total_loss += float(loss.item())
        total_acc += token_acc
        total_ppl += perplexity
        n_batches += 1
        total_tokens += int(token_ids.numel())

    out = {
        "split": args.split,
        "n_samples": len(ds),
        "n_batches": n_batches,
        "n_tokens": total_tokens,
        "loss": total_loss / max(n_batches, 1),
        "perplexity": total_ppl / max(n_batches, 1),
        "token_accuracy": total_acc / max(n_batches, 1),
        "sequence_nll_mean": float(np.mean(sequence_nll_sums)) if sequence_nll_sums else 0.0,
        "sequence_nll_std": float(np.std(sequence_nll_sums)) if sequence_nll_sums else 0.0,
        "distinct_token_ratio": float(len(used_token_ids) / max(codebook_size, 1)),
        "mean_unique_tokens_per_sequence": float(np.mean(unique_tokens_per_sequence)) if unique_tokens_per_sequence else 0.0,
        "condition_mode": condition_mode,
        "condition_fields": condition_fields,
        "token_grid_shape": list(vqvae.token_grid_shape),
        "codebook_size": codebook_size,
    }
    print(json.dumps(out, indent=2))

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, f"latent_prior_eval_{args.split}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        with open(os.path.join(args.out_dir, f"latent_prior_eval_{args.split}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(out.keys()))
            writer.writeheader()
            writer.writerow(out)


if __name__ == "__main__":
    main()
