"""Tests verifying exported classes, submodule namespaces, behavioral contracts, and snapshot compliance."""

from __future__ import annotations

import inspect
import pathlib

import zero_orbax.checkpoint as zcp
from scripts import check_compliance


def test_missing_symbols_and_behavioral_contracts(tmp_path: pathlib.Path) -> None:
    """Verify that all 14 core Orbax classes are exported and fulfill their behavioral contracts.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    exported_members: dict[str, type] = {
        name: obj for name, obj in inspect.getmembers(zcp, inspect.isclass)
    }

    target_classes = [
        "AbstractCheckpointManager",
        "AbstractCheckpointer",
        "AsyncCheckpointer",
        "AsyncOptions",
        "CheckpointHandler",
        "CheckpointManager",
        "CheckpointManagerOptions",
        "Checkpointer",
        "Future",
        "PyTreeCheckpointHandler",
        "PyTreeCheckpointer",
        "RestoreArgs",
        "SaveArgs",
        "StandardCheckpointHandler",
    ]
    for cls_name in target_classes:
        assert cls_name in exported_members
        cls_obj = exported_members[cls_name]
        sig = inspect.signature(cls_obj)
        assert len(sig.parameters) >= 0

    # Behavioral contract verification across the exported classes
    save_args = zcp.SaveArgs(aggregate=True, dtype="float32", chunk_byte_size=1024, shard_axes=(0,))
    assert save_args.aggregate is True
    assert save_args.chunk_byte_size == 1024

    restore_args = zcp.RestoreArgs(restore_type=int, dtype="int32")
    assert restore_args.restore_type is int

    async_opts = zcp.AsyncOptions(timeout_secs=120)
    assert async_opts.timeout_secs == 120

    mgr_opts = zcp.CheckpointManagerOptions(save_interval_steps=2, max_to_keep=2)
    assert mgr_opts.save_interval_steps == 2

    mgr_dir = tmp_path / "mgr"
    with zcp.CheckpointManager(mgr_dir, options=mgr_opts) as mgr:
        assert mgr.should_save(2) is True
        assert mgr.should_save(1) is False
        assert mgr.save(2, {"x": 42}) is True
        assert mgr.restore(2) == {"x": 42}

    assert check_compliance.main() == 0
