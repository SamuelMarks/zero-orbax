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

    expected_init_params: dict[str, tuple[str, ...]] = {
        "AbstractCheckpointManager": (),
        "AbstractCheckpointer": (),
        "AsyncCheckpointer": ("_handler",),
        "AsyncOptions": ("timeout_secs",),
        "CheckpointHandler": (),
        "CheckpointManager": ("directory", "checkpointers", "options"),
        "CheckpointManagerOptions": ("save_interval_steps", "max_to_keep"),
        "Checkpointer": ("handler",),
        "Future": ("result",),
        "PyTreeCheckpointHandler": ("aggregate_filename",),
        "PyTreeCheckpointer": ("primary_host",),
        "RestoreArgs": ("restore_type",),
        "SaveArgs": ("aggregate",),
        "StandardCheckpointHandler": ("save_concurrent_gb",),
    }
    for cls_name, expected_params in expected_init_params.items():
        assert cls_name in exported_members
        cls_obj = exported_members[cls_name]
        sig = inspect.signature(cls_obj)
        for param_name in expected_params:
            assert param_name in sig.parameters

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
