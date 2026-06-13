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


def test_namespace_classes() -> None:
    """Test that all namespace classes and their properties are correctly initialized.

    Returns:
        None
    """
    from zero_orbax.checkpoint import (
        async_checkpoint_handler,
        orbax,
        atomicity,
        checkpoint,
        multihost,
        epath,
        abstract_logger,
        step_lib,
        checkpoint_handler,
        args,
    )

    assert async_checkpoint_handler.AsyncCheckpointHandler is not None
    assert orbax.checkpoint.options.MultiprocessingOptions is not None
    assert atomicity.TemporaryPath is not None
    assert checkpoint.CheckpointMetadataStore is not None
    assert multihost.BarrierSyncFn is not None
    assert epath.PathLike is not None
    assert abstract_logger.AbstractLogger is not None
    assert step_lib.Metadata() is not None
    assert checkpoint_handler.CheckpointHandler is not None
    assert args.ArrayRestore() is not None
    assert args.ArraySave() is not None
    assert args.Composite() is not None
    assert args.JsonRestore() is not None
    assert args.JsonSave() is not None
    assert args.ProtoRestore() is not None
    assert args.ProtoSave() is not None
    assert args.PyTreeRestore() is not None
    assert args.PyTreeSave() is not None
    assert args.StandardRestore() is not None
    assert args.StandardSave() is not None
    assert args.JaxRandomKeySave() is not None
    assert args.JaxRandomKeyRestore() is not None
    assert args.NumpyRandomKeySave() is not None
    assert args.NumpyRandomKeyRestore() is not None
    assert args.CheckpointArgs() is not None
    assert args.get_registered_handler_cls() is None
    assert args.get_registered_args_cls() is None
    assert args.has_registered_args() is False
    assert args.register_with_handler()(int) is int


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
    checkpointer = Checkpointer(handler=None)  # type: ignore
    assert isinstance(checkpointer, Checkpointer)


def test_pytree_checkpointer() -> None:
    """Test PyTreeCheckpointer initialization."""
    checkpointer = PyTreeCheckpointer()
    assert isinstance(checkpointer, PyTreeCheckpointer)


def test_restore_transform() -> None:
    """Test RestoreTransform initialization."""
    rt = RestoreTransform(
        original_key="b/c",
        use_fallback=False,
        multi_value_fn=lambda t: t.get("b", {}).get("c"),
    )
    assert rt.original_key == "b/c"
    assert rt.use_fallback is False
    assert rt.multi_value_fn({"b": {"c": 42}}) == 42  # type: ignore


def test_standard_checkpointer() -> None:
    """Test StandardCheckpointer initialization."""
    checkpointer = StandardCheckpointer()
    assert isinstance(checkpointer, StandardCheckpointer)


def test_transform() -> None:
    """Test Transform initialization."""
    t = Transform(original_key="a", use_fallback=True, value_fn=lambda x: x * 2)
    assert t.original_key == "a"
    assert t.use_fallback is True
    assert t.value_fn(5) == 10  # type: ignore


def test_apply_transformations() -> None:
    """Test apply_transformations."""
    orig = {"a": 1, "b": {"c": 2}, "d": 3}
    new_t = {"a": None, "b": {"c": None}, "x": 9}
    res = apply_transformations(orig, {}, new_t)
    assert res == {"a": 1, "b": {"c": 2}, "x": 9}
    res = apply_transformations(orig, {}, new_t, default_to_original=False)
    assert res == {"a": None, "b": {"c": None}, "x": 9}
    trans = {
        "a": Transform(value_fn=lambda x: x * 10),
        "b": {"c": Transform(original_key="c", value_fn=lambda x: x * 2)},
        "x": 99,
    }
    res = apply_transformations(orig, trans, new_t)
    assert res == {"a": 10, "b": {"c": 4}, "x": 99}
    trans2 = {"a": Transform(multi_value_fn=lambda t: t["d"] * 2)}
    res2 = apply_transformations(orig, trans2, {"a": None})
    assert res2 == {"a": 6}
    trans3 = {"a": Transform(original_key=("b", "c"))}
    res3 = apply_transformations(orig, trans3, {"a": None})
    assert res3 == {"a": 2}
    assert apply_transformations(orig, {}, {"y": 1}) == {"y": 1}
    assert apply_transformations(orig, {}, 5) == 5
    assert apply_transformations({"b": 1}, {"b": {"c": 1}}, {"b": 5}) == {"b": 5}
    trans4 = {"a": Transform(original_key="b/z")}
    res4 = apply_transformations(orig, trans4, {"a": None})
    assert res4 == {"a": None}


def test_merge_trees() -> None:
    """Test merge_trees."""
    assert merge_trees() == {}
    t1 = {"a": {"b": 1}, "c": 2}
    t2 = {"a": {"d": 3}, "e": 4}
    res = merge_trees(t1, t2)
    assert res == {"a": {"b": 1, "d": 3}, "c": 2, "e": 4}
    target = {"z": 9}
    res2 = merge_trees(t1, target=target)
    assert res2 == {"z": 9, "a": {"b": 1}, "c": 2}
    with pytest.raises(TypeError):
        merge_trees(1)


"Tests for Phase 2 semantics."


def test_checkpoint_manager_options_validation() -> None:
    """Test/Mock documentation."""
    with pytest.raises(ValueError, match="save_interval_steps must be positive"):
        CheckpointManagerOptions(save_interval_steps=0)
    with pytest.raises(ValueError, match="must be None or non-negative"):
        CheckpointManagerOptions(max_to_keep=-1)
    with pytest.raises(ValueError, match="best_mode"):
        CheckpointManagerOptions(best_mode="invalid")
    with pytest.raises(ValueError, match="todelete_subdir and todelete_full_path"):
        CheckpointManagerOptions(todelete_subdir="subdir", todelete_full_path="path")
    opts_ro = CheckpointManagerOptions(read_only=True, todelete_full_path="path")
    assert opts_ro.todelete_full_path is None
    opts = CheckpointManagerOptions(save_interval_steps=2, max_to_keep=5)
    assert opts.save_interval_steps == 2
    assert opts.max_to_keep == 5
    opts2 = opts.replace(save_interval_steps=3)
    assert opts2.save_interval_steps == 3
    assert opts2.max_to_keep == 5
    from zero_orbax.checkpoint import step_lib

    nf = step_lib.NameFormat()
    assert not nf == "some string"


def test_abstract_checkpoint_manager_methods() -> None:
    """Test/Mock documentation."""
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
    """Test/Mock documentation."""
    opts = CheckpointManagerOptions(save_interval_steps=1, max_to_keep=2)
    manager = CheckpointManager(directory="fake_dir", options=opts, metadata=42)
    assert manager.latest_step() is None
    assert manager.all_steps() == []
    assert manager.save(1, {"val": 10}) is True
    assert manager.latest_step() == 1
    assert manager.all_steps() == [1]
    assert manager.restore(1) == {"val": 10}
    assert manager.save(2, {"val": 20}) is True
    assert manager.latest_step() == 2
    assert manager.all_steps() == [1, 2]
    assert manager.save(3, {"val": 30}) is True
    assert manager.latest_step() == 3
    assert manager.all_steps() == [2, 3]
    with pytest.raises(ValueError, match="Checkpoint for step 1 not found."):
        manager.restore(1)


def test_checkpoint_manager_read_only() -> None:
    """Test/Mock documentation."""
    opts = CheckpointManagerOptions(read_only=True)
    manager = CheckpointManager(directory="fake_dir", options=opts)
    with pytest.raises(ValueError, match="Cannot save checkpoint in read_only mode."):
        manager.save(1, {"val": 10})


def test_checkpoint_manager_save_interval() -> None:
    """Test/Mock documentation."""
    opts = CheckpointManagerOptions(save_interval_steps=2)
    manager = CheckpointManager(directory="fake_dir", options=opts)
    assert manager.save(1, {"val": 10}) is False
    assert manager.save(2, {"val": 20}) is True
    opts2 = CheckpointManagerOptions(save_interval_steps=10, save_on_steps=[3])
    manager2 = CheckpointManager(directory="fake_dir", options=opts2)
    assert manager2.save(3, {"val": 30}) is True
    assert manager2.save(10, {"val": 100}) is True
    assert manager2.save(1, {"val": 10}) is False


def test_checkpoint_manager_should_save_fn() -> None:
    """Test/Mock documentation."""

    def should_save(step, latest_step):
        """Test/Mock documentation."""
        return step > 5

    opts = CheckpointManagerOptions(should_save_fn=should_save)
    manager = CheckpointManager(directory="fake_dir", options=opts)
    assert manager.save(4, {"val": 40}) is False
    assert manager.save(6, {"val": 60}) is True


class MockHandler:
    """Test/Mock documentation."""

    def __init__(self):
        """Test/Mock documentation."""
        self.saved = None

    def save(self, path, item, *args, **kwargs):
        """Test/Mock documentation."""
        self.saved = (path, item)

    def restore(self, path, item=None, *args, **kwargs):
        """Test/Mock documentation."""
        return {"restored_from": path, "item": item}


def test_abstract_checkpointer_methods() -> None:
    """Test/Mock documentation."""
    cp = AbstractCheckpointer()
    with pytest.raises(NotImplementedError):
        cp.save("path", {"a": 1})
    with pytest.raises(NotImplementedError):
        cp.restore("path")


def test_checkpointer_save_restore() -> None:
    """Test/Mock documentation."""
    handler = MockHandler()
    cp = Checkpointer(handler=handler)  # type: ignore  # type: ignore
    cp.save("some_path", {"x": 42})
    assert handler.saved == ("some_path", {"x": 42})
    res = cp.restore("other_path", {"y": 1})
    assert res == {"restored_from": "other_path", "item": {"y": 1}}


def test_checkpointer_no_handler_methods() -> None:
    """Test/Mock documentation."""
    cp = Checkpointer(handler=EmptyHandler2())  # type: ignore[arg-type]
    cp.save("path", {"x": 1})
    res = cp.restore("path")
    assert res is None


def test_pytree_checkpointer_save_restore() -> None:
    """Test/Mock documentation."""
    cp = PyTreeCheckpointer()
    cp.save("path", {"a": 1})
    res = cp.restore("path", {"b": 2})
    assert res == {"b": 2}


def test_standard_checkpointer_save_restore() -> None:
    """Test/Mock documentation."""
    cp = StandardCheckpointer()
    cp.save("path", {"a": 1})
    res = cp.restore("path", {"b": 2})
    assert res == {"b": 2}


class MockAsyncHandler:
    """Test/Mock documentation."""

    def __init__(self):
        """Test/Mock documentation."""
        self.saved = None

    def save(self, path, item, *args, **kwargs):
        """Test/Mock documentation."""
        self.saved = (path, item)

    def restore(self, path, item=None, *args, **kwargs):
        """Test/Mock documentation."""
        return {"restored_from": path, "item": item}


class EmptyHandler2:
    """Test/Mock documentation."""

    pass


def test_async_options_semantic() -> None:
    """Test/Mock documentation."""
    opts = AsyncOptions(timeout_secs=100)
    assert opts.timeout_secs == 100
    assert opts.barrier_sync_fn is None
    assert opts.post_finalization_callback is None


def test_future_semantic() -> None:
    """Test/Mock documentation."""
    f = Future(result=42)
    assert f.result() == 42
    f2 = Future()
    assert f2.result() is None


def test_async_checkpointer_semantic() -> None:
    """Test/Mock documentation."""
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
    """Test/Mock documentation."""
    cp = AsyncCheckpointer(handler=EmptyHandler2())
    f_save = cp.save("path", {"x": 1})
    assert f_save.result() is None
    f_restore = cp.restore("path")
    assert f_restore.result() is None


def test_checkpoint_manager_complex_pytree() -> None:
    """Test/Mock documentation."""
    opts = CheckpointManagerOptions(save_interval_steps=1)
    manager = CheckpointManager(directory="fake_dir", options=opts)
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
    """Test/Mock documentation."""
    opts = CheckpointManagerOptions(enable_background_delete=True, max_to_keep=1)
    manager = CheckpointManager(directory="fake_dir", options=opts)
    manager.save(1, {"a": 1})
    manager.save(2, {"b": 2})
    assert manager.all_steps() == [2]


def test_apply_transformations_invalid_path() -> None:
    """Test that applying transformations handles invalid paths gracefully.

    Returns:
        None
    """
    from zero_orbax.checkpoint import apply_transformations, Transform

    orig = {"a": 1, "b": 2}
    trans = {"a": Transform(original_key=("b", "c"))}
    res = apply_transformations(orig, trans, {"a": None})
    assert res == {"a": 1}
