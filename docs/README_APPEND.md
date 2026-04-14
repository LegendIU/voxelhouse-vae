## Prior MLOps Additions

This repository includes a lightweight MLOps layer for the latent-token Transformer prior:
- config-driven train/eval/sample scripts
- smoke CI tests for prior modules
- run manifests with git commit and config hash
- JSONL/CSV metric logging
- standardized checkpoint aliases (`last`, `best_by_val_loss`, `best_by_val_ppl`)
- reproducible benchmark exports for qualitative and quantitative comparison

### Example workflow

```powershell
.\scripts\run_prior_train.ps1 -ConfigPath configs/prior/train_prior_base.json
.\scripts\run_prior_eval.ps1 -ConfigPath configs/prior/eval_prior_base.json
.\scripts\run_prior_sample.ps1 -ConfigPath configs/prior/sample_prior_base.json
```

### Key artifacts

- `run_manifest.json`
- `metrics.csv`
- `metrics.jsonl`
- `training_summary.json`
- `best_by_val_loss.pt`
- `best_by_val_ppl.pt`
- `comparison_summary.csv`
