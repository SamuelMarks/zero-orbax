"""Pytest configuration, path setup, and isolation fixtures for the zero-orbax test suite."""

from __future__ import annotations

from collections.abc import Iterator
import os
import sys

import pytest

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

from zero_orbax.core import BarrierSync, VirtualFileSystem  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_virtual_stores() -> Iterator[None]:
    """Reset the context-scoped VirtualFileSystem and BarrierSync stores around every test.

    Yields:
        None: Executes the test with clean virtual stores.
    """
    VirtualFileSystem().reset()
    BarrierSync().reset()
    yield
    VirtualFileSystem().reset()
    BarrierSync().reset()
