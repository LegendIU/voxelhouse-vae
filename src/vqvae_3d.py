from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_3d import ResBlock3D


class VQForwardOutput(NamedTuple):
    logits: torch.Tensor
    indices: torch.Tensor
    quantized: torch.Tensor
    vq_loss: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    perplexity: torch.Tensor


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        if codebook_size <= 1:
            raise ValueError(f"codebook_size must be > 1, got {codebook_size}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}")
        if commitment_cost < 0:
            raise ValueError(f"commitment_cost must be >= 0, got {commitment_cost}")

        self.codebook_size = int(codebook_size)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.embedding = nn.Embedding(self.codebook_size, self.embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.embedding.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dtype != torch.long:
            indices = indices.long()
        flat = indices.reshape(-1)
        quantized = self.embedding(flat)
        quantized = quantized.view(*indices.shape, self.embedding_dim)
        dims = list(range(quantized.ndim))
        return quantized.permute(0, dims[-1], *dims[1:-1]).contiguous()

    def forward(
        self,
        z_e: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if z_e.ndim != 5:
            raise ValueError(f"Expected z_e with shape [B,C,D,H,W], got {tuple(z_e.shape)}")

        bsz, channels, depth, height, width = z_e.shape
        z_flat = z_e.permute(0, 2, 3, 4, 1).reshape(-1, channels)
        codebook = self.embedding.weight

        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
            - 2.0 * z_flat @ codebook.t()
        )
        indices = torch.argmin(distances, dim=1)
        z_q = self.lookup(indices.view(bsz, depth, height, width))

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_q.detach(), z_e)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: decoder sees discrete vectors, encoder still gets gradients.
        z_q_st = z_e + (z_q - z_e).detach()

        one_hot = F.one_hot(indices, num_classes=self.codebook_size).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        return z_q_st, indices.view(bsz, depth, height, width), vq_loss, codebook_loss, commitment_loss, perplexity


class VQVAE3D(nn.Module):
    def __init__(
        self,
        resolution: int = 64,
        base_ch: int = 48,
        embedding_dim: int = 128,
        codebook_size: int = 512,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        if resolution not in (32, 64):
            raise ValueError(f"resolution must be 32 or 64, got {resolution}")
        if base_ch % 8 != 0:
            raise ValueError(f"base_ch must be divisible by 8 for GroupNorm, got {base_ch}")

        self.resolution = int(resolution)
        self.base_ch = int(base_ch)
        self.embedding_dim = int(embedding_dim)
        self.codebook_size = int(codebook_size)

        self.enc_in = nn.Conv3d(1, base_ch, 3, 1, 1)
        self.encoder = nn.Sequential(
            ResBlock3D(base_ch),
            nn.Conv3d(base_ch, base_ch * 2, 4, 2, 1),
            ResBlock3D(base_ch * 2),
            nn.Conv3d(base_ch * 2, base_ch * 4, 4, 2, 1),
            ResBlock3D(base_ch * 4),
            nn.Conv3d(base_ch * 4, base_ch * 6, 4, 2, 1),
            ResBlock3D(base_ch * 6),
            nn.GroupNorm(8, base_ch * 6),
            nn.SiLU(inplace=True),
        )
        self.pre_quant = nn.Conv3d(base_ch * 6, embedding_dim, 1)

        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )

        self.post_quant = nn.Conv3d(embedding_dim, base_ch * 6, 1)
        self.decoder = nn.Sequential(
            ResBlock3D(base_ch * 6),
            nn.ConvTranspose3d(base_ch * 6, base_ch * 4, 4, 2, 1),
            ResBlock3D(base_ch * 4),
            nn.ConvTranspose3d(base_ch * 4, base_ch * 2, 4, 2, 1),
            ResBlock3D(base_ch * 2),
            nn.ConvTranspose3d(base_ch * 2, base_ch, 4, 2, 1),
            ResBlock3D(base_ch),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(base_ch, 1, 3, 1, 1),
        )

        feat_res = resolution // 8
        self.token_grid_shape = (feat_res, feat_res, feat_res)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.enc_in(x))
        return self.pre_quant(h)

    def quantize_features(
        self,
        z_e: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.quantizer(z_e)

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encode_features(x)
        _, indices, _, _, _, _ = self.quantize_features(z_e)
        return indices

    def flatten_token_grid(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 4:
            raise ValueError(f"Expected token grid with shape [B,D,H,W], got {tuple(indices.shape)}")
        return indices.reshape(indices.shape[0], -1)

    def unflatten_token_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 2:
            raise ValueError(f"Expected token sequence with shape [B,L], got {tuple(sequence.shape)}")
        expected = self.token_grid_shape[0] * self.token_grid_shape[1] * self.token_grid_shape[2]
        if sequence.shape[1] != expected:
            raise ValueError(f"Expected sequence length {expected}, got {sequence.shape[1]}")
        return sequence.view(sequence.shape[0], *self.token_grid_shape)

    def decode_quantized(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant(z_q))

    def decode_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        z_q = self.quantizer.lookup(indices)
        return self.decode_quantized(z_q)

    def decode_token_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.decode_tokens(self.unflatten_token_sequence(sequence))

    def forward(self, x: torch.Tensor) -> VQForwardOutput:
        z_e = self.encode_features(x)
        z_q, indices, vq_loss, codebook_loss, commitment_loss, perplexity = self.quantize_features(z_e)
        logits = self.decode_quantized(z_q)
        return VQForwardOutput(
            logits=logits,
            indices=indices,
            quantized=z_q,
            vq_loss=vq_loss,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
        )
