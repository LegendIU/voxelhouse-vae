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
- optional nearest-reference IoU against a held-out split

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

## Notes

- Missing optional mesh-export dependencies no longer break sampling; the scripts warn and continue.
- Report guidance for the learned prior is summarized in [transformer_prior_report.md](transformer_prior_report.md).
