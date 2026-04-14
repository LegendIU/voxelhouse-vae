#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-.}"
WORK_DIR="${ROOT_DIR}/out_smoke_mlops"
DATA_DIR="${WORK_DIR}/tiny_data"
VQ_OUT="${WORK_DIR}/vqvae"
PRIOR_OUT="${WORK_DIR}/prior"
BENCH_OUT="${WORK_DIR}/benchmark"
MLFLOW_TRACKING_URI="sqlite:////workspace/mlruns/mlflow.db"
MLFLOW_EXPERIMENT="voxelhouse-vae-smoke"

mkdir -p "${WORK_DIR}"
mkdir -p "${ROOT_DIR}/mlruns/artifacts"

PYTHONPATH="${ROOT_DIR}/src" python3 "${ROOT_DIR}/src/create_tiny_dataset.py" \
  --out_dir "${DATA_DIR}" \
  --resolution 32 \
  --n_train 18 \
  --n_val 6 \
  --n_test 6 \
  --seed 7

PYTHONPATH="${ROOT_DIR}/src" python3 "${ROOT_DIR}/src/train_3d_vqvae.py" \
  --data_root "${DATA_DIR}" \
  --out_dir "${VQ_OUT}" \
  --device cpu \
  --epochs 1 \
  --batch_size 2 \
  --resolution 32 \
  --base_ch 16 \
  --embedding_dim 32 \
  --codebook_size 64 \
  --num_workers 0 \
  --overfit_n 8 \
  --save_every 1 \
  --early_stopping_patience 0 \
  --mlflow \
  --mlflow_experiment "${MLFLOW_EXPERIMENT}" \
  --mlflow_tracking_uri "${MLFLOW_TRACKING_URI}"

VQ_BEST="$(ls -1 "${VQ_OUT}"/run_*/best.pt | head -n 1)"

PYTHONPATH="${ROOT_DIR}/src" python3 "${ROOT_DIR}/src/train_latent_prior.py" \
  --vqvae_ckpt "${VQ_BEST}" \
  --data_root "${DATA_DIR}" \
  --out_dir "${PRIOR_OUT}" \
  --device cpu \
  --epochs 1 \
  --batch_size 2 \
  --d_model 64 \
  --nhead 4 \
  --num_layers 2 \
  --dropout 0.0 \
  --ff_mult 2 \
  --condition_mode house_attributes \
  --condition_bins 6 \
  --num_workers 0 \
  --overfit_n 8 \
  --save_every 1 \
  --early_stopping_patience 0 \
  --mlflow \
  --mlflow_experiment "${MLFLOW_EXPERIMENT}" \
  --mlflow_tracking_uri "${MLFLOW_TRACKING_URI}"

PRIOR_BEST="$(ls -1 "${PRIOR_OUT}"/run_*/best.pt | head -n 1)"

PYTHONPATH="${ROOT_DIR}/src" python3 "${ROOT_DIR}/src/eval_generative_models.py" \
  --out_dir "${BENCH_OUT}" \
  --data_root "${DATA_DIR}" \
  --reference_split test \
  --vqvae_ckpt "${VQ_BEST}" \
  --prior_ckpt "${PRIOR_BEST}" \
  --n_samples 8 \
  --threshold 0.5 \
  --prior_modes topk \
  --temperature 1.0 \
  --top_k 16 \
  --condition_preset two_story_compact \
  --guidance_candidates 12 \
  --device cpu \
  --mlflow \
  --mlflow_experiment "${MLFLOW_EXPERIMENT}" \
  --mlflow_tracking_uri "${MLFLOW_TRACKING_URI}"

PYTHONPATH="${ROOT_DIR}/src" python3 "${ROOT_DIR}/src/check_quality_gate.py" \
  --benchmark_json "${BENCH_OUT}/benchmark.json" \
  --gate_config "${ROOT_DIR}/mlops/quality_gates_smoke.json"

echo "Smoke MLOps pipeline completed."
