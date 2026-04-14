from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from latent_transformer import filter_sampling_logits, sample_from_logits


def test_top_k_filter_keeps_only_k_entries() -> None:
    logits = torch.tensor([[1.0, 0.5, 0.1, -0.2]])
    filtered = filter_sampling_logits(logits, top_k=2, top_p=1.0)
    kept = torch.isfinite(filtered).sum(dim=-1).item()
    assert kept == 2


def test_top_p_filter_keeps_non_empty_row() -> None:
    logits = torch.tensor([[5.0, 4.0, 1.0, 0.0]])
    filtered = filter_sampling_logits(logits, top_k=0, top_p=0.7)
    kept = torch.isfinite(filtered).sum(dim=-1).item()
    assert kept >= 1


def test_greedy_sampling_matches_argmax() -> None:
    logits = torch.tensor([[0.1, 0.2, 0.9]])
    token = sample_from_logits(logits, greedy=True)
    assert token.item() == 2
