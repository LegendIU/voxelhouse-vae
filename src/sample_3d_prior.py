from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from conditional_sampling import ConditionalPriorSampler, HOUSE_CONDITION_PRESETS
from constraint_guidance import ConstraintSpec
from infer_3d import export_mesh_from_occ, render_projections, save_grid
from model_loading import load_latent_prior, load_vqvae_model
from utils import choose_device


def parse_condition_values(raw: str) -> list[int]:
    values = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    return [int(v) for v in values]


def parse_decode_modes(raw: str) -> list[str]:
    if not raw.strip():
        return []
    modes = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    valid = {"greedy", "temperature", "topk", "topp"}
    invalid = [mode for mode in modes if mode not in valid]
    if invalid:
        raise SystemExit(f"Unsupported decode modes: {invalid}. Valid modes: {sorted(valid)}")
    deduped: list[str] = []
    for mode in modes:
        if mode not in deduped:
            deduped.append(mode)
    return deduped


def decode_kwargs_for_mode(
    mode: str,
    *,
    greedy: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> dict[str, float | int | bool]:
    if mode == "greedy":
        return {"greedy": True, "temperature": 1.0, "top_k": 0, "top_p": 1.0}
    if mode == "temperature":
        return {"greedy": False, "temperature": temperature, "top_k": 0, "top_p": 1.0}
    if mode == "topk":
        return {"greedy": False, "temperature": temperature, "top_k": top_k, "top_p": 1.0}
    if mode == "topp":
        return {"greedy": False, "temperature": temperature, "top_k": 0, "top_p": top_p}
    return {
        "greedy": bool(greedy),
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
    }


def save_outputs(
    occ: np.ndarray,
    token_sequence: torch.Tensor,
    sampled_condition_ids: torch.Tensor | None,
    sampled_metrics: list[dict[str, float]],
    out_dir: str,
    *,
    export_projections: bool,
    export_meshes: bool,
    save_individual_projections: bool,
    grid_cols: int,
    n_meshes: int,
) -> int:
    os.makedirs(out_dir, exist_ok=True)
    proj_dir = os.path.join(out_dir, "projections")
    mesh_dir = os.path.join(out_dir, "meshes")
    os.makedirs(proj_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)

    condition_array = np.empty((0, 0), dtype=np.int32)
    if sampled_condition_ids is not None:
        condition_array = sampled_condition_ids.cpu().numpy().astype(np.int32)

    np.savez_compressed(
        os.path.join(out_dir, "generated_samples.npz"),
        voxels=occ,
        tokens=token_sequence.cpu().numpy().astype(np.int32),
        condition_ids=condition_array,
    )
    if sampled_metrics:
        with open(os.path.join(out_dir, "constraint_scores.json"), "w", encoding="utf-8") as f:
            json.dump(sampled_metrics, f, indent=2, ensure_ascii=False)

    if export_projections:
        images = [render_projections(occ[i]) for i in range(occ.shape[0])]
        save_grid(images, os.path.join(proj_dir, "samples_grid.png"), cols=grid_cols)
        if save_individual_projections:
            for i, image in enumerate(images):
                image.save(os.path.join(proj_dir, f"sample_{i:03d}.png"))

    saved_meshes = 0
    if export_meshes:
        for i in range(occ.shape[0]):
            if saved_meshes >= n_meshes:
                break
            out_path = os.path.join(mesh_dir, f"sample_{i:03d}.obj")
            if export_mesh_from_occ(occ[i], out_path):
                saved_meshes += 1
    return saved_meshes


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior_ckpt", required=True)
    parser.add_argument("--vqvae_ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--n_meshes", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--decode_modes",
        type=str,
        default="",
        help="Optional comma-separated sweep: greedy,temperature,topk,topp",
    )

    parser.add_argument("--condition_values", type=str, default="")
    parser.add_argument("--condition_json", type=str, default="")
    parser.add_argument("--condition_preset", type=str, default="")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--guidance_candidates", type=int, default=0)
    parser.add_argument("--min_connectedness", type=float, default=0.85)
    parser.add_argument("--max_unsupported_mass", type=float, default=0.08)
    parser.add_argument("--max_component_count", type=int, default=2)
    parser.add_argument("--min_symmetry", type=float, default=0.45)
    parser.add_argument("--min_plausibility", type=float, default=0.55)
    parser.add_argument("--require_compact", action="store_true")

    parser.add_argument("--export_projections", action="store_true")
    parser.add_argument("--export_meshes", action="store_true")
    parser.add_argument("--save_individual_projections", action="store_true")
    parser.add_argument("--grid_cols", type=int, default=8)
    args = parser.parse_args()

    if args.n_samples <= 0:
        raise SystemExit("--n_samples must be > 0")
    if args.n_meshes < 0:
        raise SystemExit("--n_meshes must be >= 0")
    if not (0.0 < args.threshold < 1.0):
        raise SystemExit("--threshold must be in (0, 1)")
    if args.temperature <= 0 and not args.greedy:
        raise SystemExit("--temperature must be > 0 for stochastic decoding")
    if args.top_k < 0:
        raise SystemExit("--top_k must be >= 0")
    if not (0.0 < args.top_p <= 1.0):
        raise SystemExit("--top_p must be in (0, 1]")
    if args.repetition_penalty < 1.0:
        raise SystemExit("--repetition_penalty must be >= 1.0")

    if args.guidance_candidates < 0:
        raise SystemExit("--guidance_candidates must be >= 0")
    if args.guidance_candidates and args.guidance_candidates < args.n_samples:
        raise SystemExit("--guidance_candidates must be >= --n_samples")

    os.makedirs(args.out_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    vqvae, _, vq_cfg = load_vqvae_model(args.vqvae_ckpt, device=device)
    prior, prior_ckpt, prior_cfg = load_latent_prior(args.prior_ckpt, device=device)

    sampler = ConditionalPriorSampler(prior=prior, vqvae=vqvae, prior_cfg=prior_cfg, prior_ckpt=prior_ckpt)
    condition_dict = None
    if args.condition_preset:
        if args.condition_preset not in HOUSE_CONDITION_PRESETS:
            valid = ",".join(sorted(HOUSE_CONDITION_PRESETS))
            raise SystemExit(f"Unknown --condition_preset='{args.condition_preset}'. Available presets: {valid}")
        condition_dict = HOUSE_CONDITION_PRESETS[args.condition_preset]
    if args.condition_json:
        condition_dict = json.loads(args.condition_json)

    explicit_condition_ids = None
    if args.condition_values:
        values = parse_condition_values(args.condition_values)
        if len(values) != prior.num_condition_tokens:
            raise SystemExit(f"--condition_values must contain {prior.num_condition_tokens} ids, got {len(values)}")
        explicit_condition_ids = torch.as_tensor(values, dtype=torch.long).unsqueeze(0).repeat(
            max(args.n_samples, args.guidance_candidates),
            1,
        )

    guidance_spec = None
    if args.guidance_candidates > 0:
        guidance_spec = ConstraintSpec(
            min_connectedness=float(args.min_connectedness),
            max_unsupported_mass=float(args.max_unsupported_mass),
            max_component_count=int(args.max_component_count),
            min_symmetry=float(args.min_symmetry),
            min_plausibility=float(args.min_plausibility),
            require_compact=bool(args.require_compact),
        )

    decode_modes = parse_decode_modes(args.decode_modes)
    if not decode_modes:
        decode_modes = ["single"]

    sweep_summary: list[dict[str, object]] = []
    for mode in decode_modes:
        mode_kwargs = decode_kwargs_for_mode(
            mode,
            greedy=args.greedy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        mode_seed = int(args.seed + len(sweep_summary))
        mode_out_dir = args.out_dir if mode == "single" else os.path.join(args.out_dir, mode)

        sampled = sampler.sample(
            n_samples=args.n_samples,
            threshold=args.threshold,
            greedy=bool(mode_kwargs["greedy"]),
            temperature=float(mode_kwargs["temperature"]),
            top_k=int(mode_kwargs["top_k"]),
            top_p=float(mode_kwargs["top_p"]),
            repetition_penalty=float(args.repetition_penalty),
            device=device,
            condition=condition_dict,
            condition_values=explicit_condition_ids,
            data_root=args.data_root,
            split=args.split,
            seed=mode_seed,
            guidance_spec=guidance_spec,
            guidance_candidates=args.guidance_candidates,
        )
        token_sequence = sampled.token_ids
        occ = sampled.voxels.numpy().astype(np.uint8)

        saved_meshes = save_outputs(
            occ,
            token_sequence,
            sampled.condition_ids,
            sampled.metrics,
            mode_out_dir,
            export_projections=bool(args.export_projections),
            export_meshes=bool(args.export_meshes),
            save_individual_projections=bool(args.save_individual_projections),
            grid_cols=int(args.grid_cols),
            n_meshes=int(args.n_meshes),
        )

        meta = {
            "sampling_mode": mode,
            "n_generated": int(occ.shape[0]),
            "n_meshes_saved": int(saved_meshes),
            "threshold": float(args.threshold),
            "seed": int(mode_seed),
            "greedy": bool(mode_kwargs["greedy"]),
            "temperature": float(mode_kwargs["temperature"]),
            "top_k": int(mode_kwargs["top_k"]),
            "top_p": float(mode_kwargs["top_p"]),
            "repetition_penalty": float(args.repetition_penalty),
            "condition_mode": str(prior_cfg.get("condition_mode", "none")),
            "condition_fields": prior_ckpt.get("condition_fields", prior_cfg.get("condition_fields", [])),
            "condition_values": None if sampled.condition_ids is None else sampled.condition_ids[0].tolist(),
            "condition_preset": args.condition_preset or None,
            "condition_json": condition_dict,
            "constraint_guidance_enabled": bool(guidance_spec is not None),
            "guidance_candidates": int(args.guidance_candidates),
            "guidance_spec": None if guidance_spec is None else guidance_spec.__dict__,
            "resolution": int(vq_cfg.get("resolution", 64)),
            "codebook_size": int(vq_cfg.get("codebook_size", 512)),
            "token_grid_shape": list(vqvae.token_grid_shape),
            "data_root": args.data_root,
            "split": args.split,
        }
        with open(os.path.join(mode_out_dir, "sampling_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        sweep_summary.append(
            {
                "sampling_mode": mode,
                "out_dir": os.path.abspath(mode_out_dir),
                "n_generated": int(occ.shape[0]),
                "n_meshes_saved": int(saved_meshes),
                "greedy": bool(mode_kwargs["greedy"]),
                "temperature": float(mode_kwargs["temperature"]),
                "top_k": int(mode_kwargs["top_k"]),
                "top_p": float(mode_kwargs["top_p"]),
                "repetition_penalty": float(args.repetition_penalty),
            }
        )
        print(
            f"Saved prior samples to {os.path.abspath(mode_out_dir)} "
            f"(mode={mode}, generated={occ.shape[0]}, meshes={saved_meshes})"
        )

    if len(sweep_summary) > 1:
        with open(os.path.join(args.out_dir, "sampling_sweep_summary.json"), "w", encoding="utf-8") as f:
            json.dump(sweep_summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
