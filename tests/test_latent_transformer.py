from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from latent_transformer import LatentTokenTransformer


def test_compute_loss_returns_expected_keys() -> None:
    model = LatentTokenTransformer(
        codebook_size=16,
        token_grid_shape=(2, 2, 2),
        d_model=32,
        nhead=4,
        num_layers=2,
        dropout=0.0,
    )
    token_ids = torch.randint(0, 16, (3, 8))
    loss, aux = model.compute_loss(token_ids)
    assert loss.ndim == 0
    assert "token_accuracy" in aux
    assert "perplexity" in aux
    assert aux["perplexity"] > 0


def test_conditioned_model_sampling_shape() -> None:
    model = LatentTokenTransformer(
        codebook_size=32,
        token_grid_shape=(2, 2, 2),
        d_model=32,
        nhead=4,
        num_layers=2,
        dropout=0.0,
        condition_vocab_sizes=[4, 5],
    )
    cond = torch.tensor([[1, 2], [0, 3]], dtype=torch.long)
    samples = model.sample(n_samples=2, condition_ids=cond, greedy=True)
    assert samples.shape == (2, 8)
    assert samples.dtype == torch.long
