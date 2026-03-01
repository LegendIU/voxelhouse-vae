# Stage 2 Baseline Report (House3D)

## What I built
A controlled baseline to **reconstruct and sample 3D house shapes** from a real 3D house-mesh dataset.

## Dataset
- Source: Houses3K (3,000 textured 3D house models), Peralta et al., ECCV Workshops 2020.
- Raw format: FBX, converted to OBJ via Blender CLI.
- Preprocess: normalize meshes to unit cube; voxelize to fixed resolution occupancy grids (R=32) using trimesh voxelization.

## Model
- Representation: voxel occupancy grid (1×R×R×R), R=32.
- Architecture: small 3D ConvVAE (3 downsample conv blocks + linear latent; mirrored transpose-conv decoder).
- Loss: BCEWithLogitsLoss with pos_weight (sparsity) + β·KL (β=1e-3).
- Training: Adam, lr=2e-4, batch size 8, 15 epochs (subset for CPU).

## Evaluation
- Quantitative: reconstruction BCE + IoU on held-out test split.
- Qualitative: recon projection grids + unconditional samples, plus exported OBJ meshes.

## Sanity check
Overfit on N=16 samples (expect IoU → ~1.0 on that tiny set).

## Results
Paste `eval_3d_recon.py` JSON output + attach qualitative images and a few OBJ files.

