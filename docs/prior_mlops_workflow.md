# Prior MLOps Workflow

This document describes the reproducible workflow for the latent-token Transformer prior.

## 1. Train the VQ-VAE backbone

The prior assumes a trained VQ-VAE checkpoint with a fixed codebook and token grid shape.
The checkpoint must expose:
- `codebook_size`
- `token_grid_shape`
- `resolution`

## 2. Train the latent prior from a saved config

Windows PowerShell:

```powershell
.\scripts\run_prior_train.ps1 -ConfigPath configs/prior/train_prior_base.json
```

The training run writes a self-contained run directory with:
- `config.json`
- `run_manifest.json`
- `metrics.csv`
- `metrics.jsonl`
- `last.pt`
- `best.pt`
- `best_by_val_loss.pt`
- `best_by_val_ppl.pt`
- `training_summary.json`

## 3. Evaluate the prior

```powershell
.\scripts\run_prior_eval.ps1 -ConfigPath configs/prior/eval_prior_base.json
```

The evaluation stage exports:
- `metrics.json`
- `metrics.csv`
- `run_manifest.json`

Primary metrics:
- token cross-entropy
- perplexity
- token accuracy
- sequence NLL mean/std
- distinct token ratio

## 4. Sample from the prior

```powershell
.\scripts\run_prior_sample.ps1 -ConfigPath configs/prior/sample_prior_base.json
```

Recommended decoding sweep:
- greedy
- temperature
- top-k
- top-p

## 5. Benchmark against Gaussian VAE prior

Use `src/eval_generative_models.py` to compare:
- VAE Gaussian prior
- learned VQ-VAE + Transformer prior
- optional conditional prior
- optional constraint-guided prior

Artifacts:
- `benchmark.json`
- `benchmark.csv`
- `comparison_summary.csv`
- sample grids and meshes

## 6. Reproducibility contract

Every reported result should include:
- exact config file
- run directory path
- checkpoint alias used (`best_by_val_loss` or `best_by_val_ppl`)
- seed
- decoding parameters
- git commit SHA from `run_manifest.json`
