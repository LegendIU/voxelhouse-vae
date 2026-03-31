from __future__ import annotations

import torch

from latent_transformer import LatentTokenTransformer
from model_3d import VAE3D
from vqvae_3d import VQVAE3D


def load_vae_model(ckpt_path: str, device: torch.device) -> tuple[VAE3D, dict, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = VAE3D(
        resolution=int(cfg.get("resolution", 64)),
        latent_dim=int(cfg.get("latent_dim", 128)),
        base_ch=int(cfg.get("base_ch", 48)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt, cfg


def load_vqvae_model(ckpt_path: str, device: torch.device) -> tuple[VQVAE3D, dict, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = VQVAE3D(
        resolution=int(cfg.get("resolution", 64)),
        base_ch=int(cfg.get("base_ch", 48)),
        embedding_dim=int(cfg.get("embedding_dim", 128)),
        codebook_size=int(cfg.get("codebook_size", 512)),
        commitment_cost=float(cfg.get("commitment_cost", 0.25)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt, cfg


def load_latent_prior(ckpt_path: str, device: torch.device) -> tuple[LatentTokenTransformer, dict, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    token_grid_shape = tuple(int(v) for v in ckpt.get("token_grid_shape", cfg.get("token_grid_shape", (4, 4, 4))))
    condition_vocab_sizes = [int(v) for v in ckpt.get("condition_vocab_sizes", cfg.get("condition_vocab_sizes", []))]
    codebook_size = int(ckpt.get("codebook_size", cfg.get("codebook_size", 512)))

    model = LatentTokenTransformer(
        codebook_size=codebook_size,
        token_grid_shape=token_grid_shape,
        d_model=int(cfg.get("d_model", 256)),
        nhead=int(cfg.get("nhead", 8)),
        num_layers=int(cfg.get("num_layers", 8)),
        dropout=float(cfg.get("dropout", 0.1)),
        ff_mult=int(cfg.get("ff_mult", 4)),
        condition_vocab_sizes=condition_vocab_sizes,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt, cfg
