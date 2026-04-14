# Benchmark Sheet

This sheet is a single, auditable table for comparing:

- unconditional generation
- attribute-conditioned generation
- constraint-guided generation

Use `benchmark_sheet_template.csv` as the canonical format for reports, papers, and demos.

## How to fill

1. Run `src/eval_generative_models.py` for each target setup.
2. Copy one row per run into `benchmark_sheet_template.csv`.
3. Keep `condition_payload` and guidance columns explicit for reproducibility.
4. Add a short, factual note in `notes` for notable trade-offs.

## Required comparison blocks

For each condition case (for example `two_story_compact`):

- one `unconditional` row
- one `conditional` row
- one `constraint_guided` row

This ensures fair apples-to-apples comparison under the same split and sampling mode.

## Recommended interpretation

- **Control quality**: focus on `attribute_match_rate`
- **Validity**: focus on `connectedness`, `unsupported_mass`, `component_count`, `plausibility_score`
- **Diversity**: focus on `unique_ratio`, `pairwise_hamming`, `pairwise_iou_diversity`
- **Realism alignment**: focus on `reference_nn_iou_mean`

The expected trade-off is that stronger guidance improves validity while reducing diversity.
