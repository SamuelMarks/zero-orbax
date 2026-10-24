"""Tests for concrete CheckpointHandler subclasses and DefaultCheckpointHandlerRegistry."""

from __future__ import annotations

import pathlib

import pytest

import zero_orbax.checkpoint as zcp


def test_save_args() -> None:
    """Test SaveArgs initialization and attributes."""
    sa = zcp.SaveArgs(aggregate=True, dtype="float32", chunk_byte_size=1024, shard_axes=(0, 1))
    assert sa.aggregate is True
    assert sa.dtype == "float32"
    assert sa.chunk_byte_size == 1024
    assert sa.shard_axes == (0, 1)


def test_restore_args() -> None:
    """Test RestoreArgs and ArrayRestoreArgs initialization and methods."""
    ra = zcp.RestoreArgs(restore_type=int, dtype="int32")
    assert ra.restore_type is int
    assert ra.dtype == "int32"

    ara = zcp.ArrayRestoreArgs(
        restore_type=list,
        global_shape=(4, 4),
        shape=(2, 4),
        strict=False,
        dtype="float32",
    )
    assert ara.restore_type() is list
    assert ara.global_shape == (4, 4)
    assert ara.shape == (2, 4)
    assert ara.strict is False


def test_array_checkpoint_handler(tmp_path: pathlib.Path) -> None:
    """Test ArrayCheckpointHandler save and restore with TensorStoreDriver on disk.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    handler = zcp.ArrayCheckpointHandler(checkpoint_name="my_arr")
    handler.save(tmp_path, zcp.args.ArraySave(item=[1, 2, 3]))
    assert (tmp_path / "my_arr.ts").exists()
    assert handler.restore(tmp_path) == [1, 2, 3]
    assert handler.restore(
        tmp_path,
        restore_args=zcp.ArrayRestoreArgs(restore_type=tuple),
    ) == (1, 2, 3)
    assert handler.restore(
        tmp_path,
        restore_args=zcp.ArrayRestoreArgs(restore_type=list),
    ) == [1, 2, 3]

    # Fallback when .ts file does not exist on disk
    empty_dir = tmp_path / "empty_arr"
    empty_dir.mkdir()
    assert handler.restore(empty_dir, item=[9, 9]) == [9, 9]
    assert handler.restore(None, item=[7]) == [7]
    handler.save(None, [1])
    handler.close()


def test_json_checkpoint_handler(tmp_path: pathlib.Path) -> None:
    """Test JsonCheckpointHandler save and restore from disk JSON files.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    handler = zcp.JsonCheckpointHandler(filename="config.json")
    handler.save(tmp_path, zcp.args.JsonSave(item={"lr": 0.01, "epochs": 10}))
    assert (tmp_path / "config.json").exists()
    assert not (tmp_path / "checkpoint.msgpack").exists()
    assert handler.restore(tmp_path) == {"lr": 0.01, "epochs": 10}
    assert handler.restore(tmp_path / "nonexistent", item={"default": 1}) == {"default": 1}
    assert handler.restore(None, item={"none_dir": 2}) == {"none_dir": 2}
    handler.save(None, {"a": 1})
    handler.close()


def test_proto_checkpoint_handler(tmp_path: pathlib.Path) -> None:
    """Test ProtoCheckpointHandler save and restore on disk.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    handler = zcp.ProtoCheckpointHandler(filename="model.pb")
    assert handler.filename == "model.pb"
    handler.save(tmp_path, zcp.args.ProtoSave(item={"proto_field": "v1"}))
    assert (tmp_path / "model.pb").exists()
    assert handler.restore(tmp_path) == {"proto_field": "v1"}
    assert (
        handler.restore(
            tmp_path,
            zcp.args.ProtoRestore(item_type=str),
        )
        == "{'proto_field': 'v1'}"
    )
    assert handler.restore(tmp_path / "missing", item="fallback") == "fallback"
    assert handler.restore(None, item="none_fb") == "none_fb"
    handler.save(None, "ignored")
    handler.close()


def test_random_key_handlers(tmp_path: pathlib.Path) -> None:
    """Test JaxRandomKeyCheckpointHandler and NumpyRandomKeyCheckpointHandler on disk.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    jax_h = zcp.JaxRandomKeyCheckpointHandler(key_name="jax_key")
    assert jax_h.checkpoint_save_args(None) == (None, None)
    item, s_args = jax_h.checkpoint_save_args([0, 42])
    assert item == [0, 42]
    assert isinstance(s_args, zcp.SaveArgs)
    ra = zcp.RestoreArgs(dtype="uint32")
    assert jax_h.checkpoint_restore_args([0, 42], ra) is ra
    assert jax_h.post_restore([0, 42]) == [0, 42]

    jax_dir = tmp_path / "jax_rng"
    jax_h.save(jax_dir, zcp.args.JaxRandomKeySave(item=[0, 42]))
    assert (jax_dir / "jax_key.msgpack").exists()
    assert jax_h.restore(jax_dir, zcp.args.JaxRandomKeyRestore()) == [0, 42]
    assert jax_h.restore(tmp_path / "missing_jax", item=[1, 1]) == [1, 1]
    assert jax_h.restore(None, item=[2, 2]) == [2, 2]
    jax_h.save(None, [0, 0])

    np_h = zcp.NumpyRandomKeyCheckpointHandler(key_name="np_key")
    assert np_h.checkpoint_save_args(None) == (None, None)
    np_item, np_s_args = np_h.checkpoint_save_args([1, 2, 3])
    assert np_item == [1, 2, 3]
    assert isinstance(np_s_args, zcp.SaveArgs)
    assert np_h.checkpoint_restore_args([1, 2, 3], ra) is ra
    assert np_h.post_restore([1, 2, 3]) == [1, 2, 3]

    np_dir = tmp_path / "np_rng"
    np_h.save(np_dir, zcp.args.NumpyRandomKeySave(item=[7, 8, 9]))
    assert (np_dir / "np_key.msgpack").exists()
    assert np_h.restore(np_dir, zcp.args.NumpyRandomKeyRestore()) == [7, 8, 9]
    assert np_h.restore(tmp_path / "missing_np", item=[9]) == [9]
    assert np_h.restore(None, item=[8]) == [8]
    np_h.save(None, [0])


def test_composite_checkpoint_handler_and_registry(tmp_path: pathlib.Path) -> None:
    """Test CompositeCheckpointHandler and DefaultCheckpointHandlerRegistry on disk.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    registry = zcp.DefaultCheckpointHandlerRegistry()
    h = zcp.StandardCheckpointHandler()
    registry.add("state", zcp.args.StandardSave, h)
    registry.add("state", zcp.args.StandardRestore, h)
    registry.add("unused", zcp.args.StandardSave, None)
    assert registry.has("state", zcp.args.StandardSave) is True
    assert registry.get("state", zcp.args.StandardSave) is h

    with pytest.raises(ValueError, match="already registered"):
        registry.add("state", zcp.args.StandardSave, h, override=False)
    registry.add("state", zcp.args.StandardSave, zcp.StandardCheckpointHandler, override=True)

    copied = zcp.DefaultCheckpointHandlerRegistry(other_registry=registry)
    assert copied.has("state", zcp.args.StandardSave) is True

    comp = zcp.CompositeCheckpointHandler(
        "state",
        handler_registry=copied,
        meta=zcp.JsonCheckpointHandler(),
    )
    assert "state" in comp.item_names
    assert "meta" in comp.item_names

    comp.save(None, {"state": {"x": 1}})
    assert comp.async_save(None, {"state": {"x": 1}}) is None
    assert comp.restore(None) is None
    assert comp.structure(None) is None
    assert comp.metadata(None) is None

    futs = comp.async_save(
        tmp_path,
        zcp.args.Composite(
            state=zcp.args.StandardSave({"w": [1, 2]}),
            meta=zcp.args.JsonSave({"version": 3}),
            extra={"raw": 99},
        ),
    )
    assert futs is not None
    for f in futs:
        f.result()
    comp.finalize(tmp_path)
    assert comp.metadata(tmp_path) is not None
    assert comp.structure(tmp_path) == {
        "state": {"w": [1, 2]},
        "meta": {"version": 3},
        "extra": {"raw": 99},
    }
    comp.close()
