# voxelhouse-vae

3D VAE pipeline for house-shape reconstruction and sampling from voxel occupancy grids.

## Requirements

- Python 3.10+
- Optional: CUDA-capable GPU for faster training/inference
- Optional: Blender for FBX -> OBJ conversion

Install dependencies:

```bash
pip install -r requirements.txt
```

## Docker Quick Start

The repository includes a full Docker setup for CPU and GPU workflows:

- `Dockerfile.cpu` and `docker-compose.yml` for CPU
- `Dockerfile.gpu` and `docker-compose --profile gpu ...` for CUDA hosts
- ready services for interactive dev shell, smoke pipeline and MLflow UI

### 1) Build images

CPU:

```bash
docker compose build dev-cpu
```

GPU (requires NVIDIA Container Toolkit):

```bash
docker compose --profile gpu build dev-gpu
```

### 2) Open dev shell

CPU:

```bash
docker compose run --rm dev-cpu
```

GPU:

```bash
docker compose --profile gpu run --rm dev-gpu
```

Inside the container the project is mounted at `/workspace`, and `PYTHONPATH` is already set to `/workspace/src`.

### 3) Run smoke pipeline

CPU:

```bash
docker compose run --rm smoke-cpu
```

GPU:

```bash
docker compose --profile gpu run --rm smoke-gpu
```

### 4) Run MLflow UI

```bash
mkdir -p mlruns
docker compose up mlflow
```

Then open [http://localhost:5000](http://localhost:5000).

### Optional shortcuts

You can also use:

- `bash scripts/docker_run.sh <action>`
- `make docker-*` targets in `Makefile`

## Project Layout

- `src/build_voxel_dataset.py`: voxelizes meshes and creates `train/val/test` `.npz` files
- `src/train_3d_vae.py`: trains 3D VAE and writes checkpoints/metrics
- `src/train_3d_vqvae.py`: trains 3D VQ-VAE and exposes discrete latent tokens
- `src/latent_transformer.py`: autoregressive Transformer prior over latent tokens
- `src/train_latent_prior.py`: trains the Transformer prior on VQ-VAE token sequences
- `src/eval_latent_prior.py`: evaluates token-level likelihood / perplexity / accuracy
- `src/sample_3d_prior.py`: samples latent tokens, decodes them to voxels, saves grids/meshes
- `src/eval_generative_models.py`: compares Gaussian VAE prior vs Transformer prior and reports diversity metrics
- `src/eval_3d_recon.py`: evaluates reconstruction metrics on a split
- `src/infer_3d.py`: samples and exports projection grids + OBJ meshes
- `src/convert_fbx_to_obj.py`: batch FBX -> OBJ conversion via Blender CLI

## 1) Prepare Meshes

If your source data is FBX:

```bash
python src/convert_fbx_to_obj.py \
  --in_dir data/houses3k_fbx \
  --out_dir data/houses3k_obj \
  --blender_path blender
```

## 2) Build Voxel Dataset

```bash
python src/build_voxel_dataset.py \
  --mesh_dir data/houses3k_obj \
  --out_dir data/houses3k_vox64 \
  --resolution 64 \
  --fill_holes --keep_lcc
```

This produces:

- `train.npz`
- `val.npz`
- `test.npz`
- `meta.json`

## 3) Train

```bash
python src/train_3d_vae.py \
  --data_root data/houses3k_vox64 \
  --out_dir outputs/vae3d \
  --device cuda \
  --epochs 200 \
  --batch_size 6 \
  --amp
```

Artifacts are written into `outputs/vae3d/run_YYYYMMDD_HHMMSS/`:

- `best.pt`, `last.pt`
- `metrics.csv`
- `loss_curve.png`, `iou_curve.png`
- reconstruction previews `epoch_*_recon.png`

## 4) Train VQ-VAE

```bash
python src/train_3d_vqvae.py \
  --data_root data/houses3k_vox64 \
  --out_dir outputs/vqvae3d \
  --device cuda \
  --epochs 200 \
  --batch_size 6 \
  --embedding_dim 128 \
  --codebook_size 512
```

This produces a discrete latent grid (`R/8 x R/8 x R/8`) that can be modeled autoregressively.

## 5) Train Latent Transformer Prior

```bash
python src/train_latent_prior.py \
  --vqvae_ckpt outputs/vqvae3d/run_YYYYMMDD_HHMMSS/best.pt \
  --data_root data/houses3k_vox64 \
  --out_dir outputs/latent_prior \
  --device cuda \
  --epochs 100 \
  --batch_size 8 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 8
```

Optional conditioning is supported through:

- `--condition_mode none`
- `--condition_mode shape_stats` for automatically binned house-shape attributes
- `--condition_mode house_attributes` for controllable house factors:
  `stories_bin`, `footprint_bin`, `aspect_ratio_bin`, `roof_type`, `symmetry_flag`, `compactness_flag`
- `--condition_mode npz_fields --condition_fields class_id,style_id` for integer fields stored in the `.npz`

## 6) Evaluate Prior

```bash
python src/eval_latent_prior.py \
  --prior_ckpt outputs/latent_prior/run_YYYYMMDD_HHMMSS/best.pt \
  --vqvae_ckpt outputs/vqvae3d/run_YYYYMMDD_HHMMSS/best.pt \
  --data_root data/houses3k_vox64 \
  --split test \
  --device cuda
```

The script reports token-level cross-entropy, perplexity and token accuracy.

## 7) Evaluate Reconstruction

```bash
python src/eval_3d_recon.py \
  --ckpt outputs/vae3d/run_YYYYMMDD_HHMMSS/best.pt \
  --data_root data/houses3k_vox64 \
  --split test \
  --device cuda
```

## 8) Sample and Export Meshes

VAE Gaussian prior:

```bash
python src/infer_3d.py \
  --ckpt outputs/vae3d/run_YYYYMMDD_HHMMSS/best.pt \
  --out_dir outputs/samples \
  --sample_mode prior \
  --n_samples 64 \
  --n_meshes 32 \
  --device cuda
```

For posterior/interpolation modes, also pass `--data_root` and `--split`.

Transformer prior on VQ-VAE tokens:

```bash
python src/sample_3d_prior.py \
  --prior_ckpt outputs/latent_prior/run_YYYYMMDD_HHMMSS/best.pt \
  --vqvae_ckpt outputs/vqvae3d/run_YYYYMMDD_HHMMSS/best.pt \
  --out_dir outputs/prior_samples \
  --n_samples 64 \
  --temperature 1.0 \
  --top_k 32 \
  --export_projections \
  --export_meshes
```

Sampling modes are controlled by `--greedy`, `--temperature`, `--top_k` and `--top_p`.

Constraint-guided sampling with reranking/filtering is supported via:

- `--guidance_candidates` (draw more candidates before selecting top plausible samples)
- hard/soft validity knobs: `--min_connectedness`, `--max_unsupported_mass`,
  `--max_component_count`, `--min_symmetry`, `--min_plausibility`, `--require_compact`
- house-condition presets (`--condition_preset two_story_compact` or `wide_lowrise_sloped`)
  and custom JSON (`--condition_json '{"stories_bin":2,...}'`)

## 9) Benchmark Gaussian Prior vs Transformer Prior

```bash
python src/eval_generative_models.py \
  --out_dir outputs/generative_benchmark \
  --data_root data/houses3k_vox64 \
  --reference_split test \
  --vae_ckpt outputs/vae3d/run_YYYYMMDD_HHMMSS/best.pt \
  --vqvae_ckpt outputs/vqvae3d/run_YYYYMMDD_HHMMSS/best.pt \
  --prior_ckpt outputs/latent_prior/run_YYYYMMDD_HHMMSS/best.pt \
  --n_samples 64 \
  --prior_modes greedy,temperature,topk,topp \
  --temperature 1.0 \
  --top_k 32 \
  --top_p 0.9
```

Artifacts include:

- `benchmark.json` and `benchmark.csv`
- projection grids for each model / decoding mode
- diversity metrics: `unique_ratio`, pairwise Hamming / IoU diversity
- plausibility metrics: validity, connected components, largest-component ratio
- validity-aware metrics: `connectedness`, `unsupported_mass`, `component_count`,
  `symmetry_proxy`, `plausibility_score`
- optional nearest-reference IoU against a held-out split

If condition + guidance arguments are provided, the benchmark table includes regimes:
`unconditional`, `conditional`, `constraint_guided`.

## 10) Conditional + Validity-Aware Demo Scenarios

```bash
python src/demo_conditional_generation.py \
  --prior_ckpt outputs/latent_prior/run_YYYYMMDD_HHMMSS/best.pt \
  --vqvae_ckpt outputs/vqvae3d/run_YYYYMMDD_HHMMSS/best.pt \
  --out_dir outputs/conditional_demo \
  --data_root data/houses3k_vox64 \
  --split test \
  --n_samples 32 \
  --guidance_candidates 96
```

This script produces scenario bundles for:

- `unconditional`
- `two_story_compact` (two-story compact house)
- `wide_lowrise_sloped` (wide low-rise house with sloped roof)
- `connected_plausible_guided` (only connected plausible structures)

## 11) Evaluation Sheets

To keep reporting reproducible and fair, use:

- `benchmark_sheet_template.csv`: canonical per-run comparison table for
  unconditional vs conditional vs constraint-guided setups
- `benchmark_sheet.md`: guidance on how to fill and interpret the sheet
- `failure_gallery.md`: structured failure analysis log with visual evidence and hypotheses

## Running scripts

Run all commands from the **project root** so that `src` modules (e.g. `dataset`, `model_3d`, `utils`) resolve correctly:

```bash
cd /path/to/voxelhouse-vae
python src/train_3d_vae.py ...
```

## Development

Run tests and lint in an isolated environment (venv or Docker).

**Using venv (from project root):**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
PYTHONPATH=src pytest tests/ -v
ruff check src/
```

CI runs tests and ruff on every push/PR (see [.github/workflows/ci.yml](.github/workflows/ci.yml)).

## MLOps v1

The repository includes a lightweight MLOps baseline:

- optional MLflow tracking in training/evaluation scripts
- reproducible smoke pipeline (`scripts/mlops_smoke_pipeline.sh`)
- quality gate checks (`src/check_quality_gate.py`)

### MLflow tracking

Supported scripts expose:

- `--mlflow`
- `--mlflow_experiment`
- `--mlflow_tracking_uri`

Example:

```bash
python src/train_latent_prior.py \
  --vqvae_ckpt outputs/vqvae3d/run_YYYYMMDD_HHMMSS/best.pt \
  --data_root data/houses3k_vox64 \
  --out_dir outputs/latent_prior \
  --mlflow \
  --mlflow_experiment voxelhouse-vae
```

### Quality gate

After benchmarking:

```bash
python src/check_quality_gate.py \
  --benchmark_json outputs/generative_benchmark/benchmark.json \
  --gate_config mlops/quality_gates.json
```

The command exits with non-zero status if minimum quality constraints are violated.

Gate profiles:

- `mlops/quality_gates.json`: stricter thresholds for real experiments
- `mlops/quality_gates_smoke.json`: relaxed thresholds for tiny smoke runs

### Smoke pipeline

For fast end-to-end validation in CI or locally:

```bash
bash scripts/mlops_smoke_pipeline.sh .
```

It creates a tiny synthetic dataset, runs short VQ-VAE + prior training, runs generative benchmark, then validates quality gates.
The smoke script uses `mlops/quality_gates_smoke.json` by default.
It also logs runs to MLflow experiment `voxelhouse-vae-smoke` using the shared tracking DB (`mlruns/mlflow.db`).