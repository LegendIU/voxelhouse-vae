"""Tests for VQVAE3D tokenization and decode roundtrips."""
from __future__ import annotations

import pytest
import torch

from vqvae_3d import VQVAE3D


def test_vqvae3d_forward_shapes() -> None:
    model = VQVAE3D(resolution=32, base_ch=48, embedding_dim=32, codebook_size=64)
    model.eval()
    x = torch.randn(2, 1, 32, 32, 32)
    with torch.no_grad():
        out = model(x)

    assert out.logits.shape == (2, 1, 32, 32, 32)
    assert out.indices.shape == (2, 4, 4, 4)
    assert out.quantized.shape == (2, 32, 4, 4, 4)
    assert out.vq_loss.dim() == 0
    assert out.codebook_loss.dim() == 0
    assert out.commitment_loss.dim() == 0
    assert out.perplexity.dim() == 0


def test_vqvae3d_token_sequence_roundtrip() -> None:
    model = VQVAE3D(resolution=32, base_ch=48, embedding_dim=32, codebook_size=64)
    model.eval()
    x = torch.randn(2, 1, 32, 32, 32)
    with torch.no_grad():
        indices = model.encode_tokens(x)
        sequence = model.flatten_token_grid(indices)
        logits_from_tokens = model.decode_tokens(indices)
        logits_from_sequence = model.decode_token_sequence(sequence)

    assert sequence.shape == (2, 64)
    assert logits_from_tokens.shape == (2, 1, 32, 32, 32)
    assert logits_from_sequence.shape == (2, 1, 32, 32, 32)


def test_vqvae3d_invalid_args_raise() -> None:
    with pytest.raises(ValueError, match="resolution must be 32 or 64"):
        VQVAE3D(resolution=16, base_ch=48, embedding_dim=32, codebook_size=64)
    with pytest.raises(ValueError, match="base_ch must be divisible by 8"):
        VQVAE3D(resolution=32, base_ch=7, embedding_dim=32, codebook_size=64)
