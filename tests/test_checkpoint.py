"""Tests for zero_orbax.checkpoint."""

import pytest
from zero_orbax.checkpoint import (
    AbstractCheckpointManager,
    AbstractCheckpointer,
    AsyncCheckpointer,
    AsyncOptions,
    CheckpointManager,
    CheckpointManagerOptions,
    Checkpointer,
    Future,
    PyTreeCheckpointer,
    RestoreTransform,
    StandardCheckpointer,
    Transform,
    apply_transformations,
    merge_trees,
)


def test_abstract_checkpoint_manager() -> None:
    """Test AbstractCheckpointManager initialization."""
    manager = AbstractCheckpointManager()
    assert isinstance(manager, AbstractCheckpointManager)


def test_abstract_checkpointer() -> None:
    """Test AbstractCheckpointer initialization."""
    checkpointer = AbstractCheckpointer()
    assert isinstance(checkpointer, AbstractCheckpointer)


def test_checkpoint_manager() -> None:
    """Test CheckpointManager initialization."""
    manager = CheckpointManager(directory=None)
    assert isinstance(manager, CheckpointManager)


def test_checkpoint_manager_options() -> None:
    """Test CheckpointManagerOptions initialization."""
    options = CheckpointManagerOptions()
    assert isinstance(options, CheckpointManagerOptions)


def test_checkpointer() -> None:
    """Test Checkpointer initialization."""
    checkpointer = Checkpointer(handler=None)
    assert isinstance(checkpointer, Checkpointer)


def test_pytree_checkpointer() -> None:
    """Test PyTreeCheckpointer initialization."""
    checkpointer = PyTreeCheckpointer()
    assert isinstance(checkpointer, PyTreeCheckpointer)


def test_restore_transform() -> None:
    """Test RestoreTransform initialization."""
    rt = RestoreTransform(
        original_key=("b", "c"),
        use_fallback=False,
        multi_value_fn=lambda t: t.get("b", {}).get("c"),
    )
    assert rt.original_key == ("b", "c")
    assert rt.use_fallback is False
    assert rt.multi_value_fn({"b": {"c": 42}}) == 42


def test_standard_checkpointer() -> None:
    """Test StandardCheckpointer initialization."""
    checkpointer = StandardCheckpointer()
    assert isinstance(checkpointer, StandardCheckpointer)


def test_transform() -> None:
    """Test Transform initialization."""
    t = Transform(original_key="a", use_fallback=True, value_fn=lambda x: x * 2)
    assert t.original_key == "a"
    assert t.use_fallback is True
    assert t.value_fn(5) == 10


def test_apply_transformations() -> None:
    """Test apply_transformations."""
    orig = {"a": 1, "b": {"c": 2}, "d": 3}
    new_t = {"a": None, "b": {"c": None}, "x": 9}

    # No transformations, default to original
    res = apply_transformations(orig, {}, new_t)
    assert res == {"a": 1, "b": {"c": 2}, "x": 9}

    # No transformations, not default to original
    res = apply_transformations(orig, {}, new_t, default_to_original=False)
    assert res == {"a": None, "b": {"c": None}, "x": 9}

    # With transformations
    trans = {
        "a": Transform(value_fn=lambda x: x * 10),
        "b": {"c": Transform(original_key="c", value_fn=lambda x: x * 2)},
        "x": 99,  # Literal
    }
    res = apply_transformations(orig, trans, new_t)
    assert res == {"a": 10, "b": {"c": 4}, "x": 99}

    # Transform with multi_value_fn
    trans2 = {"a": Transform(multi_value_fn=lambda t: t["d"] * 2)}
    res2 = apply_transformations(orig, trans2, {"a": None})
    assert res2 == {"a": 6}

    # Transform with tuple original_key
    trans3 = {"a": Transform(original_key=("b", "c"))}
    res3 = apply_transformations(orig, trans3, {"a": None})
    assert res3 == {"a": 2}

    # Non-dict new_tree
    assert apply_transformations(orig, {}, 5) == 5

    # Transform with missing orig_key tuple mapping
    trans4 = {"a": Transform(original_key=("b", "z"))}
    res4 = apply_transformations(orig, trans4, {"a": None})
    assert res4 == {"a": None}


def test_merge_trees() -> None:
    """Test merge_trees."""
    assert merge_trees() == {}

    t1 = {"a": {"b": 1}, "c": 2}
    t2 = {"a": {"d": 3}, "e": 4}
    res = merge_trees(t1, t2)
    assert res == {"a": {"b": 1, "d": 3}, "c": 2, "e": 4}

    # Test target dict
    target = {"z": 9}
    res2 = merge_trees(t1, target=target)
    assert res2 == {"z": 9, "a": {"b": 1}, "c": 2}

    with pytest.raises(TypeError):
        merge_trees(1)


"""Tests for Phase 2 semantics."""


def test_checkpoint_manager_options_validation() -> None:
    with pytest.raises(ValueError, match="save_interval_steps must be positive"):
        CheckpointManagerOptions(save_interval_steps=0)

    with pytest.raises(ValueError, match="max_to_keep must be positive"):
        CheckpointManagerOptions(max_to_keep=0)

    opts = CheckpointManagerOptions(save_interval_steps=2, max_to_keep=5)
    assert opts.save_interval_steps == 2
    assert opts.max_to_keep == 5


def test_abstract_checkpoint_manager_methods() -> None:
    manager = AbstractCheckpointManager()
    with pytest.raises(NotImplementedError):
        manager.save(1, {"a": 1})
    with pytest.raises(NotImplementedError):
        manager.restore(1)
    with pytest.raises(NotImplementedError):
        manager.latest_step()
    with pytest.raises(NotImplementedError):
        manager.all_steps()


def test_checkpoint_manager_save_restore_and_latest() -> None:
    opts = CheckpointManagerOptions(save_interval_steps=1, max_to_keep=2)
    manager = CheckpointManager(directory="fake_dir", options=opts, metadata=42)

    assert manager.latest_step() is None
    assert manager.all_steps() == []

    # Save step 1
    assert manager.save(1, {"val": 10}) is True
    assert manager.latest_step() == 1
    assert manager.all_steps() == [1]
    assert manager.restore(1) == {"val": 10}

    # Save step 2
    assert manager.save(2, {"val": 20}) is True
    assert manager.latest_step() == 2
    assert manager.all_steps() == [1, 2]

    # Save step 3 (triggers max_to_keep rotation)
    assert manager.save(3, {"val": 30}) is True
    assert manager.latest_step() == 3
    assert manager.all_steps() == [2, 3]  # 1 should be deleted

    # Check deleted
    with pytest.raises(ValueError, match="Checkpoint for step 1 not found."):
        manager.restore(1)


def test_checkpoint_manager_read_only() -> None:
    opts = CheckpointManagerOptions(read_only=True)
    manager = CheckpointManager(directory="fake_dir", options=opts)

    with pytest.raises(ValueError, match="Cannot save checkpoint in read_only mode."):
        manager.save(1, {"val": 10})


def test_checkpoint_manager_save_interval() -> None:
    opts = CheckpointManagerOptions(save_interval_steps=2)
    manager = CheckpointManager(directory="fake_dir", options=opts)

    assert manager.save(1, {"val": 10}) is False
    assert manager.save(2, {"val": 20}) is True

    # Test save_on_steps override
    opts2 = CheckpointManagerOptions(save_interval_steps=10, save_on_steps=[3])
    manager2 = CheckpointManager(directory="fake_dir", options=opts2)
    assert manager2.save(3, {"val": 30}) is True
    assert manager2.save(10, {"val": 100}) is True
    assert manager2.save(1, {"val": 10}) is False


def test_checkpoint_manager_should_save_fn() -> None:
    def should_save(step, latest_step):
        return step > 5

    opts = CheckpointManagerOptions(should_save_fn=should_save)
    manager = CheckpointManager(directory="fake_dir", options=opts)

    assert manager.save(4, {"val": 40}) is False
    assert manager.save(6, {"val": 60}) is True


class MockHandler:
    def __init__(self):
        self.saved = None

    def save(self, path, item, *args, **kwargs):
        self.saved = (path, item)

    def restore(self, path, item=None, *args, **kwargs):
        return {"restored_from": path, "item": item}


def test_abstract_checkpointer_methods() -> None:
    cp = AbstractCheckpointer()
    with pytest.raises(NotImplementedError):
        cp.save("path", {"a": 1})
    with pytest.raises(NotImplementedError):
        cp.restore("path")


def test_checkpointer_save_restore() -> None:
    handler = MockHandler()
    cp = Checkpointer(handler=handler)

    cp.save("some_path", {"x": 42})
    assert handler.saved == ("some_path", {"x": 42})

    res = cp.restore("other_path", {"y": 1})
    assert res == {"restored_from": "other_path", "item": {"y": 1}}


def test_checkpointer_no_handler_methods() -> None:
    cp = Checkpointer(handler=EmptyHandler2())
    # Should not crash if handler lacks save/restore methods
    cp.save("path", {"x": 1})
    res = cp.restore("path")
    assert res is None


def test_pytree_checkpointer_save_restore() -> None:
    cp = PyTreeCheckpointer()
    # Save is a pass-through
    cp.save("path", {"a": 1})

    # Restore returns item
    res = cp.restore("path", {"b": 2})
    assert res == {"b": 2}


def test_standard_checkpointer_save_restore() -> None:
    cp = StandardCheckpointer()
    # Save is a pass-through
    cp.save("path", {"a": 1})

    # Restore returns item
    res = cp.restore("path", {"b": 2})
    assert res == {"b": 2}


class MockAsyncHandler:
    def __init__(self):
        self.saved = None

    def save(self, path, item, *args, **kwargs):
        self.saved = (path, item)

    def restore(self, path, item=None, *args, **kwargs):
        return {"restored_from": path, "item": item}


class EmptyHandler2:
    pass


def test_async_options_semantic() -> None:
    opts = AsyncOptions(timeout_secs=100)
    assert opts.timeout_secs == 100
    assert opts.barrier_sync_fn is None
    assert opts.post_finalization_callback is None


def test_future_semantic() -> None:
    f = Future(result=42)
    assert f.result() == 42

    f2 = Future()
    assert f2.result() is None


def test_async_checkpointer_semantic() -> None:
    handler = MockAsyncHandler()
    cp = AsyncCheckpointer(handler=handler)

    f_save = cp.save("some_path", {"x": 42})
    assert isinstance(f_save, Future)
    assert f_save.result() is None
    assert handler.saved == ("some_path", {"x": 42})

    f_restore = cp.restore("other_path", {"y": 1})
    assert isinstance(f_restore, Future)
    res = f_restore.result()
    assert res == {"restored_from": "other_path", "item": {"y": 1}}


def test_async_checkpointer_no_handler_methods() -> None:
    cp = AsyncCheckpointer(handler=EmptyHandler2())
    f_save = cp.save("path", {"x": 1})
    assert f_save.result() is None

    f_restore = cp.restore("path")
    assert f_restore.result() is None


def test_checkpoint_manager_complex_pytree() -> None:
    opts = CheckpointManagerOptions(save_interval_steps=1)
    manager = CheckpointManager(directory="fake_dir", options=opts)

    # Complex pytree with nested lists/dicts
    # In zero-orbax tracer, PyTrees are usually dicts, but let's test a deeply nested dict
    complex_tree = {
        "params": {
            "layer1": {"weights": [1, 2, 3], "bias": 0},
            "layer2": {"weights": [4, 5, 6], "bias": 1},
        },
        "opt_state": {"step": 100, "momentum": {"layer1": 0.9, "layer2": 0.9}},
    }

    assert manager.save(1, complex_tree) is True
    restored = manager.restore(1)
    assert restored == complex_tree


def test_checkpoint_manager_background_delete_mock() -> None:
    # Just verify that passing the flag doesn't break initialization and state is kept
    opts = CheckpointManagerOptions(enable_background_delete=True, max_to_keep=1)
    manager = CheckpointManager(directory="fake_dir", options=opts)

    manager.save(1, {"a": 1})
    manager.save(2, {"b": 2})

    assert (
        manager.all_steps() == [2]
    )  # 1 is deleted (synchronously in our mock, verifying the max_to_keep still works when bg delete is enabled)
