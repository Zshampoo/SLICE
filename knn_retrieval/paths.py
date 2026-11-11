"""Shared path helpers for KNN scripts."""
from __future__ import annotations
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data"))
KNN_ROOT = Path(os.environ.get("KNN_ROOT", DATA_ROOT / "knn"))
