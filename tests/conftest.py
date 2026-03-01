"""Pytest configuration: ensure src is on PYTHONPATH when running from repo root."""
from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
