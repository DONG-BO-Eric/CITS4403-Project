"""Pytest configuration: make src/ (model) and utils/ (helpers) importable."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed when running tests

ROOT = Path(__file__).resolve().parent.parent
for folder in ("utils", "src"):
    path = str(ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)
