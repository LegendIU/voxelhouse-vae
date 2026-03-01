# Stage 2 Baseline Report (House3D)

## What I built
A controlled baseline to reconstruct and sample 3D house shapes from a real 3D house-mesh dataset.

## Dataset
- Source: Houses3K (3,000 textured 3D house models), Peralta et al., ECCV Workshops 2020.
- Raw format: FBX, converted to OBJ via Blender CLI.
- Preprocess: normalize meshes to unit cube; voxelize to fixed-resolution occupancy grids (`R=32`) using `trimesh`.

## Model
- Representation: voxel occupancy grid (`1 x R x R x R`, `R=32`).
- Architecture: compact 3D ConvVAE (3 downsample blocks + latent projection, mirrored decoder).
- Loss: `BCEWithLogitsLoss` with `pos_weight` (for sparsity) + `beta * KL` (`beta=1e-3`).
- Training: Adam, `lr=2e-4`, batch size 8, 15 epochs (subset for CPU).

## Evaluation
- Quantitative: reconstruction BCE + IoU on held-out test split.
- Qualitative: recon projection grids + unconditional samples, plus exported OBJ meshes.

## Sanity check
Overfit on `N=16` samples (expected IoU approaches ~1.0 on that tiny subset).

## Results
Paste `eval_3d_recon.py` JSON output and attach qualitative images plus a few OBJ files.
