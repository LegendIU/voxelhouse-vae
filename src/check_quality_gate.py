from __future__ import annotations

import argparse
import json
import os
import sys


def _load_json(path: str) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick_rows(rows: list[dict], preferred: str, fallback: str | None) -> list[dict]:
    selected = [row for row in rows if str(row.get("regime", "")) == preferred]
    if selected:
        return selected
    if fallback:
        selected = [row for row in rows if str(row.get("regime", "")) == fallback]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_json", required=True)
    parser.add_argument("--gate_config", default="mlops/quality_gates.json")
    parser.add_argument("--regime", default="")
    args = parser.parse_args()

    if not os.path.exists(args.benchmark_json):
        raise SystemExit(f"benchmark file not found: {args.benchmark_json}")
    if not os.path.exists(args.gate_config):
        raise SystemExit(f"gate config not found: {args.gate_config}")

    rows = _load_json(args.benchmark_json)
    if not isinstance(rows, list) or not rows:
        raise SystemExit("benchmark_json must contain a non-empty list")
    cfg = _load_json(args.gate_config)
    if not isinstance(cfg, dict):
        raise SystemExit("gate config must be a JSON object")

    preferred = args.regime or str(cfg.get("default_regime", "constraint_guided"))
    fallback = str(cfg.get("fallback_regime", "")) or None
    target_rows = _pick_rows(rows, preferred=preferred, fallback=fallback)
    if not target_rows:
        raise SystemExit(f"No rows found for preferred regime='{preferred}' or fallback='{fallback}'")

    means: dict[str, float] = {}
    for key in ["connectedness", "unsupported_mass", "component_count", "plausibility_score"]:
        values = [float(row[key]) for row in target_rows if key in row]
        if values:
            means[key] = float(sum(values) / len(values))

    thresholds = cfg.get("thresholds", {})
    violations: list[str] = []
    for metric, rule in thresholds.items():
        if metric not in means:
            violations.append(f"missing metric '{metric}'")
            continue
        value = means[metric]
        if "min" in rule and value < float(rule["min"]):
            violations.append(f"{metric}={value:.4f} < min {float(rule['min']):.4f}")
        if "max" in rule and value > float(rule["max"]):
            violations.append(f"{metric}={value:.4f} > max {float(rule['max']):.4f}")

    print(json.dumps({"regime_checked": preferred, "means": means, "violations": violations}, indent=2))
    if violations:
        sys.exit(1)


if __name__ == "__main__":
    main()

