from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_quality_gate_passes_for_good_metrics(tmp_path: Path) -> None:
    benchmark = [
        {
            "regime": "constraint_guided",
            "connectedness": 0.9,
            "unsupported_mass": 0.03,
            "component_count": 1.2,
            "plausibility_score": 0.75,
        }
    ]
    gates = {
        "default_regime": "constraint_guided",
        "fallback_regime": "unconditional",
        "thresholds": {
            "connectedness": {"min": 0.8},
            "unsupported_mass": {"max": 0.12},
            "component_count": {"max": 2.5},
            "plausibility_score": {"min": 0.55},
        },
    }
    bench_path = tmp_path / "benchmark.json"
    gate_path = tmp_path / "gate.json"
    bench_path.write_text(json.dumps(benchmark), encoding="utf-8")
    gate_path.write_text(json.dumps(gates), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "src/check_quality_gate.py",
            "--benchmark_json",
            str(bench_path),
            "--gate_config",
            str(gate_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

