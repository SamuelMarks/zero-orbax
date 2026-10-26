"""Tests verifying exported classes, submodule namespaces, behavioral contracts, and snapshot compliance."""

from __future__ import annotations

import inspect
import pathlib
from typing import NamedTuple

import pytest
from ml_switcheroo_ir import PartitionSpec
from zero_flax.core import FrozenDict
import zero_flax.nnx as nnx
from zero_jax import numpy as jnp

import zero_orbax
import zero_orbax.checkpoint as zcp
import zero_orbax.export as zexp
from scripts import check_compliance
from zero_orbax.core import SerializedArray
from zero_orbax.types import PyTree


class _SamplePair(NamedTuple):
    """Sample NamedTuple for testing construct_restore_args."""

    alpha: int
    beta: float


def test_missing_symbols_and_behavioral_contracts(tmp_path: pathlib.Path) -> None:
    """Verify that all core Orbax classes are exported and fulfill their behavioral contracts.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    exported_members: dict[str, type] = {
        name: obj for name, obj in inspect.getmembers(zcp, inspect.isclass)
    }

    expected_init_params: dict[str, tuple[str, ...]] = {
        "AbstractCheckpointManager": (),
        "AbstractCheckpointer": (),
        "AsyncCheckpointer": ("handler", "timeout_secs", "_handler"),
        "AsyncOptions": ("timeout_secs", "async_origin"),
        "CheckpointHandler": (),
        "CheckpointManager": ("directory", "checkpointers", "options"),
        "CheckpointManagerOptions": ("save_interval_steps", "max_to_keep", "should_save_fn"),
        "Checkpointer": ("handler", "primary_host", "temporary_path_cls"),
        "Future": ("result",),
        "PyTreeCheckpointHandler": ("aggregate_filename", "primary_host"),
        "PyTreeCheckpointer": ("primary_host",),
        "RandomKeyCheckpointHandler": ("key_name",),
        "RestoreArgs": ("restore_type", "dtype", "strict"),
        "SaveArgs": ("aggregate", "dtype", "chunk_byte_size", "ocdbt_target_data_file_size"),
        "SingleReplicaArrayRestoreArgs": ("restore_type", "single_replica_sharding"),
        "StandardCheckpointHandler": ("save_concurrent_gb",),
    }
    for cls_name, expected_params in expected_init_params.items():
        assert cls_name in exported_members
        cls_obj = exported_members[cls_name]
        sig = inspect.signature(cls_obj)
        for param_name in expected_params:
            assert param_name in sig.parameters

    save_args = zcp.SaveArgs(
        aggregate=True,
        dtype="float32",
        chunk_byte_size=1024,
        ocdbt_target_data_file_size=4096,
        shard_axes=(0,),
    )
    assert save_args.aggregate is True
    assert save_args.chunk_byte_size == 1024
    assert save_args.ocdbt_target_data_file_size == 4096

    restore_args = zcp.RestoreArgs(restore_type=int, dtype="int32", strict=False)
    assert restore_args.restore_type is int
    assert restore_args.strict is False

    sr_args = zcp.SingleReplicaArrayRestoreArgs(
        restore_type="jax",
        global_shape=(4, 4),
        shape=(2, 4),
        dtype="float32",
        strict=True,
        single_replica_sharding="replica_0",
    )
    assert sr_args.restore_type() == "jax"
    assert sr_args.single_replica_sharding == "replica_0"

    async_opts = zcp.AsyncOptions(timeout_secs=120, async_origin="worker_0")
    assert async_opts.timeout_secs == 120
    assert async_opts.async_origin == "worker_0"

    mgr_opts = zcp.CheckpointManagerOptions(save_interval_steps=2, max_to_keep=2)
    assert mgr_opts.save_interval_steps == 2

    mgr_dir = tmp_path / "mgr"
    with zcp.CheckpointManager(mgr_dir, options=mgr_opts) as mgr:
        assert mgr.should_save(2) is True
        assert mgr.should_save(1) is False
        assert mgr.save(2, {"x": 42}) is True
        assert mgr.restore(2) == {"x": 42}

    assert check_compliance.main() == 0


def test_random_key_handler_and_default_registry(tmp_path: pathlib.Path) -> None:
    """Test RandomKeyCheckpointHandler and create_default_handler_registry.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    rk_handler = zcp.RandomKeyCheckpointHandler(key_name="rng_state")
    assert rk_handler.checkpoint_save_args(None) == (None, None)
    key_item, s_args = rk_handler.checkpoint_save_args([1, 2])
    assert key_item == [1, 2]
    assert isinstance(s_args, zcp.SaveArgs)
    r_args = zcp.RestoreArgs(restore_type=list)
    assert rk_handler.checkpoint_restore_args([1, 2], r_args) is r_args
    auto_r_args = rk_handler.checkpoint_restore_args([1, 2])
    assert isinstance(auto_r_args, zcp.RestoreArgs) and auto_r_args.restore_type is list
    auto_arr_r_args = rk_handler.checkpoint_restore_args(jnp.asarray([1, 2], dtype=jnp.int32))
    assert isinstance(auto_arr_r_args, zcp.ArrayRestoreArgs)
    assert rk_handler.checkpoint_restore_args(None) is None
    assert rk_handler.post_restore(None) is None
    assert rk_handler.post_restore([1, 2], metadata=zcp.RestoreArgs(restore_type=tuple)) == (1, 2)
    assert rk_handler.post_restore((1, 2), metadata=zcp.RestoreArgs(restore_type=list)) == [1, 2]
    reconstructed_jnp = rk_handler.post_restore(
        [1, 2],
        metadata=zcp.RestoreArgs(restore_type=jnp.ndarray, dtype="int32"),
    )
    assert isinstance(reconstructed_jnp, jnp.ndarray)
    reconstructed_from_tmpl = rk_handler.post_restore(
        [3, 4],
        metadata=jnp.asarray([0, 0], dtype=jnp.int32),
    )
    assert isinstance(reconstructed_from_tmpl, jnp.ndarray)
    assert rk_handler.post_restore([1, 2]) == [1, 2]

    rk_dir = tmp_path / "rk_ckpt"
    rk_handler.save(None, [9, 9])
    assert rk_handler.restore(None, item=[7, 7]) == [7, 7]
    assert rk_handler.restore(rk_dir, item=[8, 8]) == [8, 8]
    rk_handler.save(rk_dir, [10, 20])
    assert rk_handler.restore(rk_dir) == [10, 20]

    json_h = zcp.JsonCheckpointHandler()
    reg = zcp.create_default_handler_registry(metrics=json_h, ignored=None)
    assert reg.has("metrics") is True
    assert reg.get("metrics") is json_h
    assert reg.has(None, zcp.args.StandardSave) is True
    assert reg.has(None, zcp.args.StandardRestore) is True


def test_checkpoint_utils_and_msgpack_utils(tmp_path: pathlib.Path) -> None:
    """Test construct_restore_args, restore_args_from_target, wait_for_new_checkpoint, and msgpack_utils.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    arr = jnp.asarray([1.0, 2.0])
    ser_arr = SerializedArray(data=b"\x00\x01", shape=(2,), dtype="uint8")
    flax_var = nnx.Param(jnp.asarray([3.0, 4.0]))
    flax_state = nnx.State({"w": flax_var})
    fdict = FrozenDict({"k": 5})
    nt = _SamplePair(alpha=1, beta=2.5)

    complex_target: dict[str, PyTree] = {
        "arr": arr,
        "ser": ser_arr,
        "var": flax_var,
        "state": flax_state,
        "fdict": fdict,
        "nt": nt,
        "list": [10, [20]],
        "tuple": (30, None),
    }
    sharding_tree: dict[str, PyTree] = {
        "arr": PartitionSpec("x"),
        "ser": "replicated",
        "state": {"w": PartitionSpec("y")},
        "fdict": {"k": "scalar"},
        "nt": ["s0", "s1"],
        "list": ["l0", ["l1"]],
        "tuple": ("t0", "t1"),
    }

    built = zcp.checkpoint_utils.restore_args_from_target(complex_target, sharding_tree)
    assert isinstance(built, dict)
    assert isinstance(built["arr"], zcp.ArrayRestoreArgs)
    assert built["arr"].global_shape == (2,)
    assert isinstance(built["ser"], zcp.ArrayRestoreArgs)
    assert built["ser"].global_shape == (2,)
    assert isinstance(built["var"], zcp.ArrayRestoreArgs)
    assert isinstance(built["state"], dict)
    assert isinstance(built["fdict"], dict)
    assert isinstance(built["nt"], dict)
    assert isinstance(built["list"], list)
    assert isinstance(built["tuple"], tuple)

    built_no_shape = zcp.checkpoint_utils.construct_restore_args(
        {"arr": arr, "ser": ser_arr, "nt": nt},
        sharding_tree=None,
        set_global_shape=False,
    )
    assert isinstance(built_no_shape, dict)
    assert isinstance(built_no_shape["arr"], zcp.ArrayRestoreArgs)
    assert built_no_shape["arr"].global_shape is None
    assert isinstance(built_no_shape["ser"], zcp.ArrayRestoreArgs)
    assert built_no_shape["ser"].global_shape is None

    wait_dir = tmp_path / "wait_ckpts"
    wait_dir.mkdir()
    (wait_dir / "README.txt").write_text("not a step", encoding="utf-8")
    assert zcp.checkpoint_utils.wait_for_new_checkpoint(wait_dir, timeout=0) == -1
    assert (
        zcp.checkpoint_utils.wait_for_new_checkpoint(
            wait_dir,
            until_step=5,
            seconds_to_sleep=0,
            timeout=1,
        )
        == -1
    )

    (wait_dir / "1").mkdir()
    assert zcp.checkpoint_utils.wait_for_new_checkpoint(wait_dir, until_step=5, timeout=0) == -1
    assert (
        zcp.checkpoint_utils.wait_for_new_checkpoint(
            wait_dir,
            until_step=5,
            seconds_to_sleep=1,
            timeout=0,
        )
        == -1
    )
    (wait_dir / "5").mkdir()
    assert zcp.checkpoint_utils.wait_for_new_checkpoint(wait_dir, until_step=5, timeout=1) == 5
    assert (
        zcp.checkpoint_utils.wait_for_new_checkpoint(
            wait_dir,
            step_name_format=zcp.step_lib.standard_name_format(),
        )
        == 5
    )

    # Test polling loop sleep branch when timeout > 0 and no checkpoint matches
    empty_wait = tmp_path / "empty_wait"
    empty_wait.mkdir()
    assert (
        zcp.checkpoint_utils.wait_for_new_checkpoint(
            empty_wait,
            seconds_to_sleep=0,
            timeout=1,
        )
        == -1
    )

    # Test checkpoints_iterator skipping non-step entries, multihost, tree.to_flat_dict, and test_utils.sync_global_processes
    assert list(zcp.checkpoint_utils.checkpoints_iterator(wait_dir)) == [1, 5]
    assert list(zcp.checkpoint_utils.checkpoints_iterator(wait_dir, timeout_fn=lambda: True)) == []
    assert list(zcp.checkpoint_utils.checkpoints_iterator(wait_dir, timeout_fn=lambda: False)) == [
        1,
        5,
    ]
    assert zcp.multihost.process_index() == 0
    assert zcp.multihost.process_count() == 1
    zcp.multihost.sync_global_processes("test_barrier_mh")
    zcp.test_utils.sync_global_processes("test_barrier_tu")
    assert zcp.tree.to_flat_dict("not_a_dict") == {}
    assert zcp.tree.to_flat_dict({"a": {"b": 1, "c": {}}, "d": 2}) == {
        "a/b": 1,
        "a/c": {},
        "d": 2,
    }
    assert (
        zcp.checkpoint_utils.wait_for_new_checkpoint(
            empty_wait,
            until_step=1,
            seconds_to_sleep=1,
            timeout=1,
        )
        == -1
    )

    encoded = zcp.msgpack_utils.msgpack_serialize({"a": [1, 2, 3]}, in_place=True)
    decoded = zcp.msgpack_utils.msgpack_restore(encoded)
    assert decoded == {"a": [1, 2, 3]}


def test_v1_functional_apis(tmp_path: pathlib.Path) -> None:
    """Test v1.save_pytree, save_pytree_async, load_pytree, save_checkpointables, and load_checkpointables.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    p1 = tmp_path / "v1_sync"
    assert zcp.v1.custom_metadata(p1) is None
    zcp.v1.save_pytree(p1, {"w": [1, 2]}, custom_metadata={"author": "orbax"})
    assert zcp.v1.load_pytree(p1) == {"w": [1, 2]}
    assert zcp.v1.custom_metadata(p1) == {"author": "orbax"}

    with pytest.raises(ValueError, match="already exists"):
        zcp.v1.save_pytree(p1, {"w": [3, 4]}, overwrite=False)

    zcp.v1.save_pytree(p1, {"w": [3, 4]}, overwrite=True)
    assert zcp.v1.load_pytree(p1) == {"w": [3, 4]}

    p2 = tmp_path / "v1_async"
    fut = zcp.v1.save_pytree_async(p2, {"k": 10}, custom_metadata={"tag": "async"})
    fut.result()
    assert zcp.v1.load_pytree(p2) == {"k": 10}
    assert zcp.v1.custom_metadata(p2) == {"tag": "async"}

    with pytest.raises(ValueError, match="already exists"):
        zcp.v1.save_pytree_async(p2, {"k": 20}, overwrite=False)

    fut2 = zcp.v1.save_pytree_async(p2, {"k": 20}, overwrite=True)
    fut2.result()
    assert zcp.v1.load_pytree(p2) == {"k": 20}

    p3 = tmp_path / "v1_comp"
    zcp.v1.save_checkpointables(
        p3,
        {"model": {"x": 1}, "opt": {"step": 2}},
        custom_metadata={"ver": 1},
    )
    loaded_comp = zcp.v1.load_checkpointables(p3)
    assert loaded_comp == {"model": {"x": 1}, "opt": {"step": 2}}
    assert zcp.v1.custom_metadata(p3) == {"ver": 1}

    with pytest.raises(ValueError, match="already exists"):
        zcp.v1.save_checkpointables(p3, {"model": {"x": 2}}, overwrite=False)

    zcp.v1.save_checkpointables(p3, {"model": {"x": 5}}, overwrite=True)
    assert zcp.v1.load_checkpointables(p3) == {"model": {"x": 5}}


def test_orbax_export_module(tmp_path: pathlib.Path) -> None:
    """Test ObmExportOptions, ServingConfig, JaxModule, and ExportManager in zero_orbax.export.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    assert zero_orbax.export is zexp
    assert zcp.export is zexp
    assert zcp.orbax.export is zexp

    obm_opts = zexp.ObmExportOptions(enable_auto_layout=True, checkpoint_path="ckpts/0")
    assert obm_opts.enable_auto_layout is True
    assert obm_opts.checkpoint_path == "ckpts/0"

    def _linear_apply(params: PyTree, inputs: PyTree) -> PyTree:
        """Compute simple affine transformation on scalar/dict inputs.

        Args:
            params: Parameter dictionary with ``scale`` and ``bias``.
            inputs: Input number.

        Returns:
            PyTree: Output number.
        """
        if isinstance(params, dict) and isinstance(inputs, (int, float)):
            scale = params.get("scale", 1)
            bias = params.get("bias", 0)
            if isinstance(scale, (int, float)) and isinstance(bias, (int, float)):
                return inputs * scale + bias
        return inputs

    assert _linear_apply({"scale": "bad"}, 3) == 3
    assert _linear_apply("not_dict", 3) == 3

    jmod = zexp.JaxModule(
        params={"scale": 2, "bias": 1},
        apply_fn=_linear_apply,
        trainable=True,
    )
    assert "__call__" in jmod.apply_fn_map
    assert jmod.model_params == {"scale": 2, "bias": 1}
    assert jmod.methods["__call__"](3) == 7
    assert jmod.jax_methods["__call__"](4) == 9

    jmod.update_variables({"scale": 3, "bias": 2})
    assert jmod.model_params == {"scale": 3, "bias": 2}
    assert jmod.methods["__call__"](4) == 14

    cfg_single = zexp.ServingConfig(
        signature_key="serving_default",
        tf_preprocessor=lambda x: x + 1 if isinstance(x, int) else x,
        tf_postprocessor=lambda y: y * 10 if isinstance(y, int) else y,
        obm_export_options=obm_opts,
    )
    assert cfg_single.get_signature_keys() == ["serving_default"]
    bound_direct = cfg_single.bind(lambda x: x * 2 if isinstance(x, int) else x)
    assert bound_direct["serving_default"](3) == 80

    cfg_multi_keys = zexp.ServingConfig(
        signature_key=["sig_a", "sig_b"],
        method_key="predict",
    )
    assert cfg_multi_keys.get_signature_keys() == ["sig_a", "sig_b"]

    jmod_multi = zexp.JaxModule(
        params={"scale": 5, "bias": 0},
        apply_fn={"predict": _linear_apply, "encode": _linear_apply},
    )
    mgr = zexp.ExportManager(jmod_multi, [cfg_multi_keys])
    assert mgr.tf_module is jmod_multi
    sigs = mgr.serving_signatures
    assert sigs["sig_a"](2) == 10
    assert sigs["sig_b"](3) == 15

    # Test ServingConfig.bind error branches
    cfg_missing_key = zexp.ServingConfig(signature_key="bad", method_key="nonexistent")
    with pytest.raises(ValueError, match="not found"):
        cfg_missing_key.bind(jmod_multi.methods)

    cfg_ambiguous = zexp.ServingConfig(signature_key="ambiguous", method_key=None)
    with pytest.raises(ValueError, match="method_key"):
        cfg_ambiguous.bind(jmod_multi.methods)

    cfg_auto_single = zexp.ServingConfig(signature_key="auto", method_key=None)
    bound_auto = cfg_auto_single.bind(jmod.methods)
    assert bound_auto["auto"](2) == 8

    export_dir = tmp_path / "exported_model"
    mgr.save(export_dir)
    mgr.save(
        export_dir,
        save_options={"format": "obm"},
        signature_overrides={"custom_sig": lambda x: x},
    )
    loaded = mgr.load(export_dir)
    assert isinstance(loaded, dict)
    assert loaded["params"] == {"scale": 5, "bias": 0}
    assert "custom_sig" in loaded["signature_keys"]

    with pytest.raises(FileNotFoundError, match="not found"):
        mgr.load(tmp_path / "missing_export_dir")
