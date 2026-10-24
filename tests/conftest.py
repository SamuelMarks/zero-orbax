"""Pytest configuration and path setup for the zero-orbax test suite."""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SIBLING_PATHS = (
    os.path.join(_REPO_ROOT, "src"),
    os.path.join(_REPO_ROOT, "..", "ml-switcheroo-ir", "src"),
    os.path.join(_REPO_ROOT, "..", "ml-ecosystem-snapshots", "src"),
    os.path.join(_REPO_ROOT, "..", "ml-switcheroo-compiler", "src"),
    os.path.join(_REPO_ROOT, "..", "zero-jax", "src"),
)

for _path in reversed(_SIBLING_PATHS):
    if _path not in sys.path:
        sys.path.insert(0, _path)
