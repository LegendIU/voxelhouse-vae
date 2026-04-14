from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch

from conditional_sampling import ConditionalPriorSampler, HOUSE_CONDITION_PRESETS
from constraint_guidance import ConstraintSpec
from generative_metrics import summarize_voxel_samples
from infer_3d import export_mesh_from_occ, render_projections, save_grid
from logging_utils import build_run_manifest, save_manifest
from mlops import MlflowLogger
from model_loading import load_latent_prior, load_vae_model, load_vqvae_model
from utils import choose_device


def parse_modes(raw: str) -> list[str]:
    modes = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    valid = {"greedy", "temperature", "topk", "topp"}
    bad = [mode for mode in modes if mode not in valid]
    if bad:
        raise ValueError(f"Unsupported prior modes: {bad}")
    return modes


def parse_condition_values(raw: str) -> list[int]:
    values = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    return [int(v) for v in values]


def save_sample_artifacts(occ: np.ndarray, out_dir: str, prefix: str, grid_cols: int, export_meshes: bool, n_meshes: int) -> int:
    images = [render_projections(occ[i]) for i in range(occ.shape[0])]
    save_grid(images, os.path.join(out_dir, f"{prefix}_grid.png"), cols=grid_cols)
    np.savez_compressed(os.path.join(out_dir, f"{prefix}_samples.npz"), voxels=occ)
    saved_meshes = 0
    if export_meshes:
        mesh_dir = os.path.join(out_dir, f"{prefix}_meshes")
        os.makedirs(mesh_dir, exist_ok=True)
        for i in range(occ.shape[0]):
            if saved_meshes >= n_meshes:
                break
            if export_mesh_from_occ(occ[i], os.path.join(mesh_dir, f"sample_{i:03d}.obj")):
                saved_meshes += 1
    return saved_meshes


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--reference_split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--grid_cols", type=int, default=8)
    parser.add_argument("--export_meshes", action="store_true")
    parser.add_argument("--n_meshes", type=int, default=16)
    parser.add_argument("--max_pairs", type=int, default=256)
    parser.add_argument("--max_reference", type=int, default=64)
    parser.add_argument("--vae_ckpt", default=None)
    parser.add_argument("--vqvae_ckpt", default=None)
    parser.add_argument("--prior_ckpt", default=None)
    parser.add_argument("--prior_modes", type=str, default="greedy,temperature,topk,topp")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--condition_values", type=str, default="")
    parser.add_argument("--condition_json", type=str, default="")
    parser.add_argument("--condition_preset", type=str, default="")
    parser.add_argument("--guidance_candidates", type=int, default=0)
    parser.add_argument("--min_connectedness", type=float, default=0.85)
    parser.add_argument("--max_unsupported_mass", type=float, default=0.08)
    parser.add_argument("--max_component_count", type=int, default=2)
    parser.add_argument("--min_symmetry", type=float, default=0.45)
    parser.add_argument("--min_plausibility", type=float, default=0.55)
    parser.add_argument("--require_compact", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow_experiment", type=str, default="voxelhouse-vae")
    parser.add_argument("--mlflow_tracking_uri", type=str, default="")
    args = parser.parse_args()

    if args.vae_ckpt is None and (args.vqvae_ckpt is None or args.prior_ckpt is None):
        raise SystemExit("Provide at least --vae_ckpt or both --vqvae_ckpt and --prior_ckpt")
    os.makedirs(args.out_dir, exist_ok=True)
    mlf = MlflowLogger.create(
        enabled=bool(args.mlflow),
        experiment_name=args.mlflow_experiment,
        run_name="eval_generative_models",
        tracking_uri=(args.mlflow_tracking_uri or None),
        tags={"script": "eval_generative_models", "task": "benchmark"},
    )
    mlf.log_params(vars(args))
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    reference_voxels = None
    if args.data_root is not None:
        from dataset import VoxelNPZDataset
        ds = VoxelNPZDataset(os.path.join(args.data_root, f"{args.reference_split}.npz"), augment=False)
        if len(ds) > 0:
            reference_voxels = ds.voxels.astype(np.uint8)

    rows: list[dict[str, float | int | str]] = []

    if args.vae_ckpt is not None:
        vae, _, vae_cfg = load_vae_model(args.vae_ckpt, device=device)
        latent_dim = int(vae_cfg.get("latent_dim", 128))
        z = torch.randn(args.n_samples, latent_dim, device=device)
        logits = vae.decode(z)
        occ = (torch.sigmoid(logits)[:, 0] > args.threshold).cpu().numpy().astype(np.uint8)
        saved_meshes = save_sample_artifacts(occ, args.out_dir, "vae_gaussian", args.grid_cols, args.export_meshes, args.n_meshes)
        metrics = summarize_voxel_samples(occ, reference_voxels=reference_voxels, max_pairs=args.max_pairs, max_reference=args.max_reference, seed=args.seed)
        metrics.update({"model": "vae_gaussian", "regime": "unconditional", "sampling_mode": "gaussian", "saved_meshes": int(saved_meshes), "resolution": int(vae_cfg.get("resolution", 64)), "artifact_prefix": "vae_gaussian"})
        rows.append(metrics)

    if args.vqvae_ckpt is not None and args.prior_ckpt is not None:
        vqvae, _, vq_cfg = load_vqvae_model(args.vqvae_ckpt, device=device)
        prior, prior_ckpt, prior_cfg = load_latent_prior(args.prior_ckpt, device=device)
        sampler = ConditionalPriorSampler(prior=prior, vqvae=vqvae, prior_cfg=prior_cfg, prior_ckpt=prior_ckpt)
        prior_modes = parse_modes(args.prior_modes)
        condition_dict = None
        if args.condition_preset:
            condition_dict = HOUSE_CONDITION_PRESETS[args.condition_preset]
        if args.condition_json:
            condition_dict = json.loads(args.condition_json)
        explicit_condition_ids = None
        if args.condition_values:
            values = parse_condition_values(args.condition_values)
            explicit_condition_ids = torch.as_tensor(values, dtype=torch.long).unsqueeze(0).repeat(max(args.n_samples, args.guidance_candidates), 1)
        guidance_spec = None
        if args.guidance_candidates > 0:
            guidance_spec = ConstraintSpec(min_connectedness=float(args.min_connectedness), max_unsupported_mass=float(args.max_unsupported_mass), max_component_count=int(args.max_component_count), min_symmetry=float(args.min_symmetry), min_plausibility=float(args.min_plausibility), require_compact=bool(args.require_compact))
        decode_settings = {
            "greedy": dict(greedy=True, temperature=1.0, top_k=0, top_p=1.0),
            "temperature": dict(greedy=False, temperature=args.temperature, top_k=0, top_p=1.0),
            "topk": dict(greedy=False, temperature=args.temperature, top_k=args.top_k, top_p=1.0),
            "topp": dict(greedy=False, temperature=args.temperature, top_k=0, top_p=args.top_p),
        }
        for mode in prior_modes:
            base = decode_settings[mode]
            sampled = sampler.sample(n_samples=args.n_samples, device=device, threshold=args.threshold, data_root=args.data_root, split=args.reference_split, seed=args.seed, condition=None, condition_values=None, guidance_spec=None, guidance_candidates=0, repetition_penalty=args.repetition_penalty, **base)
            occ = sampled.voxels.numpy().astype(np.uint8)
            prefix = f"vqvae_transformer_{mode}_unconditional"
            saved_meshes = save_sample_artifacts(occ, args.out_dir, prefix, args.grid_cols, args.export_meshes, args.n_meshes)
            metrics = summarize_voxel_samples(occ, reference_voxels=reference_voxels, max_pairs=args.max_pairs, max_reference=args.max_reference, seed=args.seed)
            metrics.update({"model": "vqvae_transformer", "regime": "unconditional", "sampling_mode": mode, "saved_meshes": int(saved_meshes), "resolution": int(vq_cfg.get("resolution", 64)), "token_grid_shape": "x".join(str(v) for v in vqvae.token_grid_shape), "codebook_size": int(vq_cfg.get("codebook_size", 512)), "repetition_penalty": float(args.repetition_penalty), "artifact_prefix": prefix})
            rows.append(metrics)
            if condition_dict is not None or explicit_condition_ids is not None:
                sampled_cond = sampler.sample(n_samples=args.n_samples, device=device, threshold=args.threshold, data_root=args.data_root, split=args.reference_split, seed=args.seed, condition=condition_dict, condition_values=explicit_condition_ids, guidance_spec=None, guidance_candidates=0, repetition_penalty=args.repetition_penalty, **base)
                occ_c = sampled_cond.voxels.numpy().astype(np.uint8)
                prefix_c = f"vqvae_transformer_{mode}_conditional"
                save_sample_artifacts(occ_c, args.out_dir, prefix_c, args.grid_cols, args.export_meshes, args.n_meshes)
                metrics_c = summarize_voxel_samples(occ_c, reference_voxels=reference_voxels, max_pairs=args.max_pairs, max_reference=args.max_reference, seed=args.seed)
                metrics_c.update({"model": "vqvae_transformer", "regime": "conditional", "sampling_mode": mode, "resolution": int(vq_cfg.get("resolution", 64)), "token_grid_shape": "x".join(str(v) for v in vqvae.token_grid_shape), "codebook_size": int(vq_cfg.get("codebook_size", 512)), "repetition_penalty": float(args.repetition_penalty), "artifact_prefix": prefix_c})
                rows.append(metrics_c)
                if guidance_spec is not None:
                    sampled_guided = sampler.sample(n_samples=args.n_samples, device=device, threshold=args.threshold, data_root=args.data_root, split=args.reference_split, seed=args.seed, condition=condition_dict, condition_values=explicit_condition_ids, guidance_spec=guidance_spec, guidance_candidates=args.guidance_candidates, repetition_penalty=args.repetition_penalty, **base)
                    occ_g = sampled_guided.voxels.numpy().astype(np.uint8)
                    prefix_g = f"vqvae_transformer_{mode}_guided"
                    save_sample_artifacts(occ_g, args.out_dir, prefix_g, args.grid_cols, args.export_meshes, args.n_meshes)
                    metrics_g = summarize_voxel_samples(occ_g, reference_voxels=reference_voxels, max_pairs=args.max_pairs, max_reference=args.max_reference, seed=args.seed)
                    metrics_g.update({"model": "vqvae_transformer", "regime": "constraint_guided", "sampling_mode": mode, "resolution": int(vq_cfg.get("resolution", 64)), "token_grid_shape": "x".join(str(v) for v in vqvae.token_grid_shape), "codebook_size": int(vq_cfg.get("codebook_size", 512)), "repetition_penalty": float(args.repetition_penalty), "artifact_prefix": prefix_g})
                    rows.append(metrics_g)

    with open(os.path.join(args.out_dir, "benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    benchmark_json = os.path.join(args.out_dir, "benchmark.json")
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        benchmark_csv = os.path.join(args.out_dir, "benchmark.csv")
        with open(benchmark_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        summary_fields = [
            "model",
            "regime",
            "sampling_mode",
            "valid_ratio",
            "unique_ratio",
            "occupancy_mean",
            "occupancy_std",
            "connectedness",
            "unsupported_mass",
            "component_count",
            "symmetry_proxy",
            "plausibility_score",
            "pairwise_iou_diversity",
            "reference_nn_iou_mean",
            "artifact_prefix",
        ]
        summary_fields = [field for field in summary_fields if any(field in row for row in rows)]
        with open(os.path.join(args.out_dir, "comparison_summary.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in summary_fields})
    else:
        benchmark_csv = os.path.join(args.out_dir, "benchmark.csv")
    manifest = build_run_manifest(stage="eval_generative_models", config=vars(args), extra={"artifact_family": "generative_benchmark", "out_dir": os.path.abspath(args.out_dir)})
    save_manifest(args.out_dir, manifest)
    mlf.log_artifact(benchmark_json, artifact_path="benchmark")
    if os.path.exists(benchmark_csv):
        mlf.log_artifact(benchmark_csv, artifact_path="benchmark")
    comparison_csv = os.path.join(args.out_dir, "comparison_summary.csv")
    if os.path.exists(comparison_csv):
        mlf.log_artifact(comparison_csv, artifact_path="benchmark")
    mlf.close()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
