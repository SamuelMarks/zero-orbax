"""Pytest configuration, path setup, and isolation fixtures for the zero-orbax test suite."""

from __future__ import annotations

from collections.abc import Iterator
import os
import subprocess
import sys
import tempfile

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


def _setup_clean_zero_chex() -> None:
    """Export committed HEAD of ``../zero-chex`` into a temporary path so in-flight sibling edits do not break imports."""
    chex_repo = os.path.join(_REPO_ROOT, "..", "zero-chex")
    tmp_chex_dir = os.path.join(tempfile.gettempdir(), "zero_orbax_clean_zero_chex")
    os.makedirs(tmp_chex_dir, exist_ok=True)
    archive = subprocess.run(
        ["git", "-C", chex_repo, "archive", "HEAD"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["tar", "-xf", "-", "-C", tmp_chex_dir],
        input=archive.stdout,
        check=True,
    )
    clean_chex_src = os.path.join(tmp_chex_dir, "src")
    if clean_chex_src not in sys.path:  # pragma: no branch
        sys.path.insert(0, clean_chex_src)


_setup_clean_zero_chex()

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
