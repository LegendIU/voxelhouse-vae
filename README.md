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

## 4) Evaluate

```bash
python src/eval_3d_recon.py \
  --ckpt outputs/vae3d/run_YYYYMMDD_HHMMSS/best.pt \
  --data_root data/houses3k_vox64 \
  --split test \
  --device cuda
```

## 5) Sample and Export Meshes

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
