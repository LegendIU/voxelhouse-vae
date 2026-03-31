"""Tests for autoregressive latent-token Transformer prior."""
from __future__ import annotations

import torch

from latent_transformer import LatentTokenTransformer, filter_sampling_logits, sample_from_logits


def test_latent_transformer_forward_shapes_without_conditions() -> None:
    model = LatentTokenTransformer(
        codebook_size=32,
        token_grid_shape=(2, 2, 2),
        d_model=32,
        nhead=4,
        num_layers=2,
        dropout=0.0,
    )
    tokens = torch.randint(0, 32, (3, 8))
    logits = model(tokens)
    assert logits.shape == (3, 8, 32)


def test_latent_transformer_forward_shapes_with_conditions() -> None:
    model = LatentTokenTransformer(
        codebook_size=32,
        token_grid_shape=(2, 2, 2),
        d_model=32,
        nhead=4,
        num_layers=2,
        dropout=0.0,
        condition_vocab_sizes=[4, 5],
    )
    tokens = torch.randint(0, 32, (2, 8))
    conditions = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    logits = model(tokens, condition_ids=conditions)
    assert logits.shape == (2, 8, 32)


def test_latent_transformer_sampling_returns_valid_token_ids() -> None:
    model = LatentTokenTransformer(
        codebook_size=16,
        token_grid_shape=(2, 2, 2),
        d_model=32,
        nhead=4,
        num_layers=2,
        dropout=0.0,
    )
    samples = model.sample(n_samples=4, greedy=True)
    grid = model.sample_token_grid(n_samples=2, greedy=True)

    assert samples.shape == (4, 8)
    assert grid.shape == (2, 2, 2, 2)
    assert samples.min().item() >= 0
    assert samples.max().item() < 16


def test_sampling_helpers_respect_top_k_and_greedy() -> None:
    logits = torch.tensor([[0.1, 0.2, 0.3, 10.0]], dtype=torch.float32)
    filtered = filter_sampling_logits(logits, top_k=2, top_p=1.0)
    greedy = sample_from_logits(logits, greedy=True)

    assert torch.isneginf(filtered[0, 0])
    assert torch.isneginf(filtered[0, 1])
    assert greedy.item() == 3
