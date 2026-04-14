from __future__ import annotations

import numpy as np

from constraint_guidance import ConstraintSpec, rerank_and_filter, score_samples


def test_score_samples_reports_plausibility_fields() -> None:
    vox = np.zeros((2, 8, 8, 8), dtype=np.uint8)
    vox[0, 2:6, 2:6, :3] = 1
    vox[1, 1:7, 1:7, 0] = 1
    metrics = score_samples(vox)
    assert len(metrics) == 2
    assert "connectedness" in metrics[0]
    assert "unsupported_mass" in metrics[0]
    assert "plausibility_score" in metrics[0]
    assert 0.0 <= metrics[0]["plausibility_score"] <= 1.0


def test_rerank_and_filter_prefers_valid_shapes() -> None:
    good = np.zeros((8, 8, 8), dtype=np.uint8)
    good[2:6, 2:6, :4] = 1
    bad = np.zeros((8, 8, 8), dtype=np.uint8)
    bad[0, 0, 7] = 1
    bad[7, 7, 7] = 1
    samples = np.stack([bad, good], axis=0)
    selected, metrics, keep_idx = rerank_and_filter(samples, ConstraintSpec(), n_select=1)
    assert selected.shape == (1, 8, 8, 8)
    assert metrics[0]["plausibility_score"] >= 0.0
    assert keep_idx == [1]
    assert int(selected[0, 3, 3, 1]) == 1
