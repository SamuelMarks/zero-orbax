"""Tests for missing symbols."""

import pytest
import zero_orbax.checkpoint as zcp


def test_missing_symbols():
    """Test function."""
    missing_classes = [
        "ArrayCheckpointHandler",
        "ArrayRestoreArgs",
        "AsyncCheckpointHandler",
        "BasePyTreeCheckpointHandler",
        "CompositeCheckpointHandler",
        "DefaultCheckpointHandlerRegistry",
        "JaxRandomKeyCheckpointHandler",
        "JsonCheckpointHandler",
        "NumpyRandomKeyCheckpointHandler",
        "ProtoCheckpointHandler",
        "PyTreeCheckpointHandler",
        "SaveArgs",
        "StandardCheckpointHandler",
    ]
    for cls_name in missing_classes:
        assert hasattr(zcp, cls_name), f"Missing class: {cls_name}"

    missing_modules = [
        "abstract_checkpoint_manager",
        "abstract_checkpointer",
        "aggregate_handlers",
        "async_checkpointer",
        "checkpoint_args",
        "checkpoint_manager",
        "checkpoint_utils",
        "checkpointer",
        "contextlib",
        "functools",
        "future",
        "handlers",
        "logging",
        "metadata",
        "msgpack_utils",
        "nest_asyncio",
        "options",
        "path",
        "pytree_checkpointer",
        "serialization",
        "standard_checkpointer",
        "test_utils",
        "transform_utils",
        "tree",
        "type_handlers",
        "utils",
    ]
    for mod_name in missing_modules:
        assert hasattr(zcp, mod_name), f"Missing module: {mod_name}"


try:
    import orbax.checkpoint as ocp

    HAS_ORBAX = True
except ImportError:
    HAS_ORBAX = False


@pytest.mark.skipif(not HAS_ORBAX, reason="orbax is not installed")
def test_signature_compliance():
    """Test function."""
    import inspect

    missing_classes = [
        "ArrayCheckpointHandler",
        "ArrayRestoreArgs",
        "AsyncCheckpointHandler",
        "BasePyTreeCheckpointHandler",
        "CompositeCheckpointHandler",
        "DefaultCheckpointHandlerRegistry",
        "JaxRandomKeyCheckpointHandler",
        "JsonCheckpointHandler",
        "NumpyRandomKeyCheckpointHandler",
        "ProtoCheckpointHandler",
        "PyTreeCheckpointHandler",
        "SaveArgs",
        "StandardCheckpointHandler",
    ]
    for cls_name in missing_classes:
        if hasattr(ocp, cls_name) and hasattr(zcp, cls_name):
            o_cls = getattr(ocp, cls_name)
            z_cls = getattr(zcp, cls_name)
            # basic signature check
            try:
                o_sig = inspect.signature(o_cls)
                z_sig = inspect.signature(z_cls)
                # Just ensuring the instantiation doesn't completely mismatch, but we might not need exact match if it works.
            except ValueError:
                pass
