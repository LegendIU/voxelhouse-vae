from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from conditional_sampling import ConditionalPriorSampler, HOUSE_CONDITION_PRESETS
from constraint_guidance import ConstraintSpec
from generative_metrics import summarize_voxel_samples
from model_loading import load_latent_prior, load_vqvae_model
from utils import choose_device


def _run_case(
    sampler: ConditionalPriorSampler,
    name: str,
    n_samples: int,
    threshold: float,
    device: torch.device,
    data_root: str | None,
    split: str,
    seed: int,
    condition: dict[str, int] | None,
    guidance: ConstraintSpec | None,
    guidance_candidates: int,
    out_dir: str,
) -> dict[str, float | str]:
    result = sampler.sample(
        n_samples=n_samples,
        threshold=threshold,
        device=device,
        condition=condition,
        data_root=data_root,
        split=split,
        seed=seed,
        temperature=1.0,
        top_k=32,
        top_p=0.9,
        guidance_spec=guidance,
        guidance_candidates=guidance_candidates,
    )
    vox = result.voxels.numpy()
    metrics = summarize_voxel_samples(vox, seed=seed)
    metrics["case"] = name
    out_path = os.path.join(out_dir, f"{name}.npz")
    os.makedirs(out_dir, exist_ok=True)
    payload = {"voxels": vox, "tokens": result.token_ids.numpy()}
    if result.condition_ids is not None:
        payload["condition_ids"] = result.condition_ids.numpy()
    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    np.savez_compressed(out_path, **payload)
    return metrics


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_ckpt", required=True)
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--guidance_candidates", type=int, default=96)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = choose_device(args.device)
    vqvae, _, _ = load_vqvae_model(args.vqvae_ckpt, device=device)
    prior, prior_ckpt, prior_cfg = load_latent_prior(args.prior_ckpt, device=device)
    sampler = ConditionalPriorSampler(prior=prior, vqvae=vqvae, prior_cfg=prior_cfg, prior_ckpt=prior_ckpt)

    baseline = _run_case(
        sampler=sampler,
        name="unconditional",
        n_samples=args.n_samples,
        threshold=args.threshold,
        device=device,
        data_root=args.data_root,
        split=args.split,
        seed=args.seed,
        condition=None,
        guidance=None,
        guidance_candidates=0,
        out_dir=args.out_dir,
    )
    two_story = _run_case(
        sampler=sampler,
        name="two_story_compact",
        n_samples=args.n_samples,
        threshold=args.threshold,
        device=device,
        data_root=args.data_root,
        split=args.split,
        seed=args.seed + 1,
        condition=HOUSE_CONDITION_PRESETS["two_story_compact"],
        guidance=None,
        guidance_candidates=0,
        out_dir=args.out_dir,
    )
    wide_low = _run_case(
        sampler=sampler,
        name="wide_lowrise_sloped",
        n_samples=args.n_samples,
        threshold=args.threshold,
        device=device,
        data_root=args.data_root,
        split=args.split,
        seed=args.seed + 2,
        condition=HOUSE_CONDITION_PRESETS["wide_lowrise_sloped"],
        guidance=None,
        guidance_candidates=0,
        out_dir=args.out_dir,
    )
    connected_guided = _run_case(
        sampler=sampler,
        name="connected_plausible_guided",
        n_samples=args.n_samples,
        threshold=args.threshold,
        device=device,
        data_root=args.data_root,
        split=args.split,
        seed=args.seed + 3,
        condition=HOUSE_CONDITION_PRESETS["two_story_compact"],
        guidance=ConstraintSpec(),
        guidance_candidates=args.guidance_candidates,
        out_dir=args.out_dir,
    )

    table = [baseline, two_story, wide_low, connected_guided]
    with open(os.path.join(args.out_dir, "demo_table.json"), "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()

