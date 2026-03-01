"""Tests for VAE3D forward pass and output shapes."""
from __future__ import annotations

import pytest
import torch

from model_3d import VAE3D


@pytest.mark.parametrize("resolution", [32, 64])
def test_vae3d_forward_shapes(resolution: int) -> None:
    """Forward returns (logits, mu, logvar) with correct shapes."""
    latent_dim = 32
    base_ch = 48
    model = VAE3D(resolution=resolution, latent_dim=latent_dim, base_ch=base_ch)
    model.eval()
    batch_size = 2
    x = torch.randn(batch_size, 1, resolution, resolution, resolution)
    with torch.no_grad():
        logits, mu, logvar = model(x)
    assert logits.shape == (batch_size, 1, resolution, resolution, resolution)
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)


def test_vae3d_encode_decode_shapes() -> None:
    """Encode then decode preserves batch and spatial size."""
    model = VAE3D(resolution=64, latent_dim=128, base_ch=48)
    model.eval()
    x = torch.randn(3, 1, 64, 64, 64)
    with torch.no_grad():
        mu, logvar = model.encode(x)
        z = model.reparameterize(mu, logvar)
        logits = model.decode(z)
    assert mu.shape == (3, 128)
    assert logvar.shape == (3, 128)
    assert logits.shape == (3, 1, 64, 64, 64)


def test_vae3d_invalid_resolution_raises() -> None:
    """Only resolution 32 or 64 is allowed."""
    with pytest.raises(ValueError, match="resolution must be 32 or 64"):
        VAE3D(resolution=16, latent_dim=32, base_ch=48)
    with pytest.raises(ValueError, match="resolution must be 32 or 64"):
        VAE3D(resolution=128, latent_dim=32, base_ch=48)


def test_vae3d_base_ch_must_be_divisible_by_8() -> None:
    """base_ch must be divisible by 8 for GroupNorm."""
    with pytest.raises(ValueError, match="base_ch must be divisible by 8"):
        VAE3D(resolution=32, latent_dim=32, base_ch=7)
