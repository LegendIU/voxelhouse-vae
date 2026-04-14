#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-help}"

case "${ACTION}" in
  build-cpu)
    docker compose build dev-cpu
    ;;
  build-gpu)
    docker compose --profile gpu build dev-gpu
    ;;
  dev-cpu)
    docker compose run --rm dev-cpu
    ;;
  dev-gpu)
    docker compose --profile gpu run --rm dev-gpu
    ;;
  smoke-cpu)
    docker compose run --rm smoke-cpu
    ;;
  smoke-gpu)
    docker compose --profile gpu run --rm smoke-gpu
    ;;
  mlflow)
    mkdir -p mlruns
    docker compose up mlflow
    ;;
  *)
    cat <<'EOF'
Usage:
  bash scripts/docker_run.sh build-cpu
  bash scripts/docker_run.sh build-gpu
  bash scripts/docker_run.sh dev-cpu
  bash scripts/docker_run.sh dev-gpu
  bash scripts/docker_run.sh smoke-cpu
  bash scripts/docker_run.sh smoke-gpu
  bash scripts/docker_run.sh mlflow
EOF
    ;;
esac
