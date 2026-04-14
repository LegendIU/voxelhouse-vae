# Reproducible Experiment Protocol for the Latent Prior

## Goal

Provide a clean, defensible experimental protocol for comparing unconditional, conditional,
and constraint-guided generation under the latent-token Transformer prior.

## Training protocol

1. Train VQ-VAE once and freeze it.
2. Train the latent prior with a saved config and fixed random seed.
3. Track both validation loss and validation perplexity.
4. Save multiple checkpoint aliases:
   - `last.pt`
   - `best_by_val_loss.pt`
   - `best_by_val_ppl.pt`

## Evaluation protocol

Evaluate on the held-out test split only.
Do not tune decoding hyperparameters on the test split.

For the prior itself, report:
- validation/test cross-entropy
- perplexity
- token accuracy
- sequence-level NLL mean/std
- distinct token ratio

For decoded voxel outputs, report:
- occupancy statistics
- connectedness
- unsupported mass
- component count
- symmetry proxy
- plausibility score
- pairwise diversity
- reference nearest-neighbor IoU

## Sampling protocol

For every qualitative comparison, keep the following fixed unless the experiment explicitly changes it:
- seed
- number of samples
- threshold
- codebook / prior checkpoints
- candidate count for guidance

Recommended decoding settings:
- Greedy: deterministic baseline
- Temperature: `temperature=1.0`
- Top-k: `temperature=1.0, top_k=32`
- Top-p: `temperature=1.0, top_p=0.9`

## Reporting rule

Each table row must map to one exact artifact prefix and one exact checkpoint alias.
This avoids ambiguity when presenting results during the defense.
