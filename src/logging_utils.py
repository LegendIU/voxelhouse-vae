from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from typing import Any

import torch


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def get_git_commit_sha(root: str | None = None) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_run_manifest(
    *,
    stage: str,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
    root: str | None = None,
) -> dict[str, Any]:
    manifest = {
        "stage": stage,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit_sha": get_git_commit_sha(root=root),
        "config_hash": config_hash(config),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "config": config,
    }
    if extra:
        manifest.update(extra)
    return manifest


def save_manifest(out_dir: str, manifest: dict[str, Any], filename: str = "run_manifest.json") -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return out_path
