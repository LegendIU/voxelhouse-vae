from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from conditioning import build_shape_condition_ids, parse_condition_fields
from house_attributes import extract_house_attribute_ids


def test_parse_condition_fields_handles_csv() -> None:
    fields = parse_condition_fields("stories_bin, roof_type , compactness_flag")
    assert fields == ["stories_bin", "roof_type", "compactness_flag"]


def test_build_shape_condition_ids_shape() -> None:
    vox = torch.zeros(2, 1, 8, 8, 8)
    vox[:, :, :4, :4, :3] = 1.0
    cond = build_shape_condition_ids(vox, num_bins=8)
    assert cond.shape == (2, 4)
    assert cond.dtype == torch.long
    assert torch.all(cond >= 0)
    assert torch.all(cond < 8)


def test_extract_house_attribute_ids_shape() -> None:
    vox = torch.zeros(2, 1, 8, 8, 8)
    vox[:, :, 2:6, 2:6, :4] = 1.0
    attrs = extract_house_attribute_ids(vox, ratio_bins=8, story_bins=4)
    assert attrs.shape == (2, 6)
    assert attrs.dtype == torch.long
