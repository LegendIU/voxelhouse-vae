from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from conditioning import sample_condition_ids_from_dataset
from constraint_guidance import ConstraintSpec, rerank_and_filter
from dataset import VoxelNPZDataset
from house_attributes import HOUSE_ATTRIBUTE_FIELD_NAMES, HouseAttributeSpec


HOUSE_CONDITION_PRESETS: dict[str, dict[str, int]] = {
    "two_story_compact": {
        "stories_bin": 2,
        "footprint_bin": 4,
        "aspect_ratio_bin": 6,
        "roof_type": 0,
        "symmetry_flag": 1,
        "compactness_flag": 1,
    },
    "wide_lowrise_sloped": {
        "stories_bin": 1,
        "footprint_bin": 5,
        "aspect_ratio_bin": 2,
        "roof_type": 1,
        "symmetry_flag": 0,
        "compactness_flag": 0,
    },
}


@dataclass
class GuidedSamplingResult:
    voxels: torch.Tensor
    token_ids: torch.Tensor
    condition_ids: torch.Tensor | None
    metrics: list[dict[str, float]]


def parse_condition_dict(raw: dict[str, Any], vocab_sizes: list[int]) -> torch.Tensor:
    values: list[int] = []
    for idx, field in enumerate(HOUSE_ATTRIBUTE_FIELD_NAMES):
        if field not in raw:
            raise KeyError(f"Missing condition field '{field}'")
        value = int(raw[field])
        if value < 0 or value >= int(vocab_sizes[idx]):
            raise ValueError(f"Condition '{field}' must be in [0, {int(vocab_sizes[idx]) - 1}], got {value}")
        values.append(value)
    return torch.as_tensor(values, dtype=torch.long)


class ConditionalPriorSampler:
    def __init__(self, prior, vqvae, prior_cfg: dict, prior_ckpt: dict):
        self.prior = prior
        self.vqvae = vqvae
        self.prior_cfg = prior_cfg
        self.prior_ckpt = prior_ckpt
        self.condition_mode = str(prior_cfg.get("condition_mode", "none"))
        self.condition_fields = prior_ckpt.get("condition_fields", prior_cfg.get("condition_fields", []))
        self.condition_bins = int(prior_cfg.get("condition_bins", 8))
        self.condition_vocab_sizes = [int(v) for v in prior_ckpt.get("condition_vocab_sizes", [])]

    def _resolve_condition_ids(
        self,
        n_samples: int,
        condition: dict[str, Any] | None,
        condition_values: torch.Tensor | None,
        data_root: str | None,
        split: str,
        seed: int,
    ) -> torch.Tensor | None:
        if self.prior.num_condition_tokens == 0:
            return None
        if condition_values is not None:
            if condition_values.ndim == 1:
                condition_values = condition_values.unsqueeze(0).repeat(n_samples, 1)
            if condition_values.shape != (n_samples, self.prior.num_condition_tokens):
                raise ValueError(
                    f"condition_values must have shape {(n_samples, self.prior.num_condition_tokens)}, "
                    f"got {tuple(condition_values.shape)}"
                )
            return condition_values.long()

        if condition is not None:
            if self.condition_mode != "house_attributes":
                raise ValueError(
                    "Dictionary-based condition is only supported for condition_mode='house_attributes'. "
                    "Use condition_values tensor for other modes."
                )
            if not self.condition_vocab_sizes:
                spec = HouseAttributeSpec(ratio_bins=self.condition_bins)
                self.condition_vocab_sizes = spec.vocab_sizes()
            one = parse_condition_dict(condition, vocab_sizes=self.condition_vocab_sizes)
            return one.unsqueeze(0).repeat(n_samples, 1)

        if data_root is None:
            raise ValueError("Conditioned prior requires condition, condition_values or data_root for empirical sampling.")
        ds = VoxelNPZDataset(
            f"{data_root}/{split}.npz",
            resolution=int(getattr(self.vqvae, "resolution", self.prior_cfg.get("resolution", 64))),
            augment=False,
        )
        return sample_condition_ids_from_dataset(
            ds,
            mode=self.condition_mode,
            n_samples=n_samples,
            fields=self.condition_fields,
            num_bins=self.condition_bins,
            seed=seed,
        )

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        threshold: float = 0.55,
        greedy: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        device: torch.device | None = None,
        condition: dict[str, Any] | None = None,
        condition_values: torch.Tensor | None = None,
        data_root: str | None = None,
        split: str = "train",
        seed: int = 42,
        guidance_spec: ConstraintSpec | None = None,
        guidance_candidates: int = 0,
    ) -> GuidedSamplingResult:
        if device is None:
            device = next(self.prior.parameters()).device
        condition_ids = self._resolve_condition_ids(
            n_samples=max(n_samples, guidance_candidates),
            condition=condition,
            condition_values=condition_values,
            data_root=data_root,
            split=split,
            seed=seed,
        )
        n_draw = max(n_samples, guidance_candidates)
        token_ids = self.prior.sample(
            n_samples=n_draw,
            condition_ids=None if condition_ids is None else condition_ids[:n_draw].to(device),
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )
        logits = self.vqvae.decode_token_sequence(token_ids)
        occ = (torch.sigmoid(logits)[:, 0] > threshold).cpu().to(torch.uint8)
        metrics: list[dict[str, float]] = []

        if guidance_spec is not None:
            selected, metrics, keep_idx = rerank_and_filter(occ.numpy(), spec=guidance_spec, n_select=n_samples)
            keep = torch.as_tensor(selected, dtype=torch.uint8)
            if keep.shape[0] < n_samples:
                # Fallback: if filters are very strict, pad with best available unguided samples.
                pad = occ[: (n_samples - keep.shape[0])]
                keep = torch.cat([keep, pad], dim=0)
            occ = keep
            keep_idx_t = torch.as_tensor(keep_idx[: occ.shape[0]], dtype=torch.long, device=token_ids.device)
            token_ids = token_ids.index_select(0, keep_idx_t).cpu()
            if condition_ids is not None:
                condition_ids = condition_ids.index_select(0, keep_idx_t.cpu())
        else:
            occ = occ[:n_samples]
            token_ids = token_ids[:n_samples].cpu()
            if condition_ids is not None:
                condition_ids = condition_ids[:n_samples]

        return GuidedSamplingResult(voxels=occ, token_ids=token_ids, condition_ids=condition_ids, metrics=metrics)

