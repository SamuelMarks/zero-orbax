"""Tests for missing handlers."""

import pytest
import zero_orbax.checkpoint as zcp


def test_save_args():
    """Test function."""
    args = zcp.SaveArgs(aggregate=True, chunk_byte_size=1024)
    assert args.aggregate is True
    assert args.chunk_byte_size == 1024


def test_array_restore_args():
    """Test function."""
    args = zcp.ArrayRestoreArgs(restore_type=int)
    assert args.restore_type is int


def test_async_checkpoint_handler():
    """Test function."""
    handler = zcp.AsyncCheckpointHandler()
    assert handler.async_save("dir") is None
    handler.close()
    handler.finalize("dir")
    assert handler.metadata("dir") is None
    assert handler.restore("dir") is None
    handler.save("dir")


def test_array_checkpoint_handler():
    """Test function."""
    handler = zcp.ArrayCheckpointHandler(checkpoint_name="test")
    assert handler.checkpoint_name == "test"


def test_base_pytree_checkpoint_handler():
    """Test function."""
    handler = zcp.BasePyTreeCheckpointHandler(use_zarr3=True)
    assert handler.get_param_names(None) is None


def test_composite_checkpoint_handler():
    """Test function."""
    handler = zcp.CompositeCheckpointHandler("item")
    assert handler


def test_jax_random_key_checkpoint_handler():
    """Test function."""
    handler = zcp.JaxRandomKeyCheckpointHandler(key_name="key")
    assert handler.checkpoint_restore_args(None) is None
    assert handler.checkpoint_save_args(None) == (None, None)
    assert handler.post_restore("item", None) == "item"


def test_json_checkpoint_handler():
    """Test function."""
    handler = zcp.JsonCheckpointHandler(filename="test.json")
    assert handler.filename == "test.json"


def test_numpy_random_key_checkpoint_handler():
    """Test function."""
    handler = zcp.NumpyRandomKeyCheckpointHandler(key_name="key")
    assert handler.checkpoint_restore_args(None) is None
    assert handler.checkpoint_save_args(None) == (None, None)
    assert handler.post_restore("item", None) == "item"


def test_proto_checkpoint_handler():
    """Test function."""
    handler = zcp.ProtoCheckpointHandler(filename="test.proto")
    assert handler.filename == "test.proto"


def test_pytree_checkpoint_handler():
    """Test function."""
    handler = zcp.PyTreeCheckpointHandler(aggregate_filename="agg")
    assert handler


def test_standard_checkpoint_handler():
    """Test function."""
    handler = zcp.StandardCheckpointHandler(save_concurrent_gb=10)
    assert handler


def test_default_checkpoint_handler_registry():
    """Test function."""
    registry = zcp.DefaultCheckpointHandlerRegistry()
    registry.add("item", None, None)
    assert registry.get("item", None) is None
    assert registry.get_all_entries() == {}
    assert registry.has("item", None) is False
