"""Comprehensive unit tests for zero_orbax.checkpoint classes, managers, and transforms."""

from __future__ import annotations

import pathlib

import pytest

from zero_orbax.checkpoint import (
    AbstractCheckpointManager,
    AbstractCheckpointer,
    AsyncCheckpointHandler,
    AsyncCheckpointer,
    CheckpointHandler,
    CheckpointManager,
    CheckpointManagerOptions,
    Checkpointer,
    Future,
    PyTreeCheckpointHandler,
    PyTreeCheckpointer,
    RestoreArgs,
    StandardCheckpointer,
    Transform,
    apply_transformations,
    args,
    merge_trees,
)
from zero_orbax.core import MsgpackEngine
from zero_orbax.types import PyTree


def test_abstract_checkpointer() -> None:
    """Test AbstractCheckpointer raises NotImplementedError for save and restore."""
    cp = AbstractCheckpointer()
    with pytest.raises(NotImplementedError):
        cp.save("/tmp")
    with pytest.raises(NotImplementedError):
        cp.restore("/tmp")


def test_checkpointer_and_handlers(tmp_path: pathlib.Path) -> None:
    """Test Checkpointer, AsyncCheckpointer, StandardCheckpointer, and PyTreeCheckpointer.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    handler = CheckpointHandler()
    cp = Checkpointer(handler)
    assert cp.save(tmp_path / "sync", {"a": 1}) is None
    assert cp.restore(tmp_path / "sync") == {"a": 1}

    acp = AsyncCheckpointer(AsyncCheckpointHandler())
    fut = acp.save(tmp_path / "async", {"a": 1})
    assert isinstance(fut, Future)
    assert fut.result() is None
    with pytest.raises(ValueError, match="non-negative"):
        fut.result(timeout=-1.0)
    rfut = acp.restore(tmp_path / "async")
    assert isinstance(rfut, Future)
    assert rfut.result() == {"a": 1}

    scp = StandardCheckpointer()
    sfut = scp.save(tmp_path / "std", {"a": 1})
    assert isinstance(sfut, Future)
    assert scp.restore(tmp_path / "std") == {"a": 1}

    pcp = PyTreeCheckpointer()
    assert pcp.save(tmp_path / "pt", {"a": 1}) is None
    assert pcp.restore(tmp_path / "pt") == {"a": 1}


def test_pytree_checkpoint_handler(tmp_path: pathlib.Path) -> None:
    """Test PyTreeCheckpointHandler with transforms and PyTreeRestore args.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    handler = PyTreeCheckpointHandler()
    handler.save(tmp_path, {"a": 1, "b": 2})
    assert handler.restore(tmp_path) == {"a": 1, "b": 2}

    def _mul10(x: PyTree) -> PyTree:
        """Multiply integer leaf by 10.

        Args:
            x: Input leaf.

        Returns:
            PyTree: Multiplied value.
        """
        return x * 10 if isinstance(x, int) else x

    def _add5(x: PyTree) -> PyTree:
        """Add 5 to integer leaf.

        Args:
            x: Input leaf.

        Returns:
            PyTree: Incremented value.
        """
        return x + 5 if isinstance(x, int) else x

    transforms: PyTree = {"a": Transform(value_fn=_mul10)}
    res = handler.restore(tmp_path, item={"a": 0, "b": 0}, transforms=transforms)
    assert res == {"a": 10, "b": 2}

    restore_arg = args.PyTreeRestore(
        item={"a": 0, "b": 0},
        transforms={"b": Transform(value_fn=_add5)},
    )
    res2 = handler.restore(tmp_path, item=restore_arg)
    assert res2 == {"a": 1, "b": 7}


def test_abstract_checkpoint_manager() -> None:
    """Test AbstractCheckpointManager raises NotImplementedError for core abstract methods."""
    mgr = AbstractCheckpointManager()
    with pytest.raises(NotImplementedError):
        mgr.save(1, {})
    with pytest.raises(NotImplementedError):
        mgr.restore(1)
    with pytest.raises(NotImplementedError):
        mgr.latest_step()
    with pytest.raises(NotImplementedError):
        mgr.all_steps()


def test_checkpoint_manager_and_empty_dict_roundtrip(tmp_path: pathlib.Path) -> None:
    """Test CheckpointManager intervals, max_to_keep, empty dict round-trip, and errors.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    assert MsgpackEngine().unpackb(MsgpackEngine().packb({})) == {}

    opts = CheckpointManagerOptions(save_interval_steps=2, max_to_keep=2)
    mgr = CheckpointManager(tmp_path, Checkpointer(CheckpointHandler()), options=opts)

    assert mgr.latest_step() is None
    assert mgr.all_steps() == []

    # Step 1 skipped because save_interval_steps=2
    assert not mgr.save(1, {"a": 1})
    # Force save empty dict at step 1
    assert mgr.save(1, {}, force=True)
    assert mgr.restore(1) == {}

    assert mgr.save(2, {"a": 2})
    assert mgr.save(4, {"a": 4})
    assert mgr.all_steps() == [2, 4]
    assert mgr.latest_step() == 4
    assert mgr.restore() == {"a": 4}
    assert mgr.restore(2) == {"a": 2}

    with pytest.raises(ValueError):
        mgr.restore(1)

    mgr2 = CheckpointManager(tmp_path)
    assert mgr2.save(1, args=args.StandardSave({"x": 10}))
    assert mgr2.restore(1) == {"x": 10}
    assert mgr2.restore(1, items={"x": 0}) == {"x": 10}

    mgr2.save(2, None)
    assert mgr2.restore(2, items={"fallback": 99}) == {"fallback": 99}

    ro_opts = CheckpointManagerOptions(read_only=True)
    ro_mgr = CheckpointManager(tmp_path, options=ro_opts)
    with pytest.raises(ValueError, match="read_only"):
        ro_mgr.save(1, {"a": 1})


def test_checkpoint_manager_options_errors() -> None:
    """Test validation errors on CheckpointManagerOptions."""
    with pytest.raises(ValueError, match="best_mode"):
        CheckpointManagerOptions(best_mode="invalid")
    with pytest.raises(ValueError, match="max_to_keep"):
        CheckpointManagerOptions(max_to_keep=-1)
    with pytest.raises(ValueError, match="save_interval_steps"):
        CheckpointManagerOptions(save_interval_steps=0)
    with pytest.raises(ValueError, match="todelete"):
        CheckpointManagerOptions(todelete_subdir="a", todelete_full_path="b")

    opts = CheckpointManagerOptions(save_on_steps=[1, 5])
    mgr = CheckpointManager("/tmp/steps_test", options=opts)
    assert mgr.should_save(5) is True


def test_args() -> None:
    """Test args wrappers and registration lookup helpers."""
    c = args.Composite(a=args.StandardSave({"x": 1}), b=args.JsonSave({"y": 2}))
    assert isinstance(c["a"], args.StandardSave)
    assert c["a"].item == {"x": 1}
    assert list(c.keys()) == ["a", "b"]
    assert len(c.values()) == 2
    assert len(c.items()) == 2

    s_restore = args.StandardRestore({"x": 0})
    assert s_restore.item == {"x": 0}

    p_save = args.PyTreeSave({"x": 1})
    assert p_save.item == {"x": 1}

    class DummyHandler(CheckpointHandler):
        """Dummy handler for registration testing."""

    @args.register_with_handler(DummyHandler, for_save=True, for_restore=True)
    class DummyArgs(args.CheckpointArgs):
        """Dummy args for registration testing."""

    assert args.get_registered_handler_cls(DummyArgs) is DummyHandler
    assert args.get_registered_handler_cls(DummyArgs()) is DummyHandler
    assert args.get_registered_args_cls(DummyHandler) is DummyArgs
    assert args.get_registered_args_cls(DummyHandler(), for_save=False) is DummyArgs
    assert args.has_registered_args(DummyHandler)
    assert args.has_registered_args(DummyArgs)


def test_merge_trees() -> None:
    """Test merge_trees across empty, nested, and invalid inputs."""
    assert merge_trees() == {}
    t1: dict[str, PyTree] = {"a": 1, "b": {"x": 10}}
    t2: dict[str, PyTree] = {"b": {"y": 20}, "c": 3}
    assert merge_trees(t1, t2) == {"a": 1, "b": {"x": 10, "y": 20}, "c": 3}

    with pytest.raises(TypeError):
        merge_trees(t1, [1, 2])


def test_apply_transformations() -> None:
    """Test apply_transformations across value_fn, multi_value_fn, regex, and path lookups."""
    assert apply_transformations({"a": 1}, {}, [1, 2]) == [1, 2]

    def _add1(x: PyTree) -> PyTree:
        """Add 1 to integer leaf.

        Args:
            x: Input leaf.

        Returns:
            PyTree: Incremented leaf.
        """
        return x + 1 if isinstance(x, int) else x

    def _sum_ab(tree: PyTree) -> PyTree:
        """Sum 'a' and 'b/c' from tree.

        Args:
            tree: Full original PyTree.

        Returns:
            PyTree: Computed sum.
        """
        if isinstance(tree, dict):
            a_val = tree.get("a")
            b_val = tree.get("b")
            if isinstance(a_val, int) and isinstance(b_val, dict):
                c_val = b_val.get("c")
                if isinstance(c_val, int):
                    return a_val + c_val
        return 0

    orig: PyTree = {"a": 1, "b": {"c": 2}, "d": 4, "layer_0": 99}
    trans: PyTree = {
        "a_new": Transform(original_key="a", value_fn=_add1),
        "b": {"c_new": Transform(original_key="b/c")},
        "combined": Transform(
            multi_value_fn=_sum_ab,
            multi_value_fn_input_args={"a": RestoreArgs(restore_type=int)},
        ),
        "literal": 99,
        "missing_orig": Transform(original_key="nonexistent/path", use_fallback=True),
        "missing_sub": Transform(original_key=("a", "deep", "missing")),
        "regex_match": Transform(original_key=r"layer_\d+"),
        "regex_nested": Transform(original_key=("b", r"c.*")),
        "nested_untransformed": {},
    }
    new_tree: PyTree = {
        "a_new": None,
        "b": {"c_new": None},
        "combined": None,
        "d": None,
        "literal": None,
        "missing_orig": None,
        "missing_sub": None,
        "regex_match": None,
        "regex_nested": None,
        "nested_untransformed": {"sub": 123},
        "fallback_default": 55,
    }
    res = apply_transformations(orig, trans, new_tree, default_to_original=True)
    assert res == {
        "a_new": 2,
        "b": {"c_new": 2},
        "combined": 3,
        "d": 4,
        "literal": 99,
        "missing_orig": None,
        "missing_sub": None,
        "regex_match": 99,
        "regex_nested": 2,
        "nested_untransformed": {"sub": 123},
        "fallback_default": 55,
    }

    res_no_default = apply_transformations(orig, {}, {"d": 0}, default_to_original=False)
    assert res_no_default == {"d": 0}

    res_nested_non_dict = apply_transformations(
        {"a": {"x": 1}},
        {"a": {"x": Transform()}},
        {"a": 123},
    )
    assert res_nested_non_dict == {"a": 123}


def test_apply_transformations_tuple_path_and_restore_args() -> None:
    """Test apply_transformations with sequence path and RestoreArgs."""
    orig: PyTree = {"a": 1, "b": 2}
    trans: PyTree = {"a": Transform(original_key=("b", "c"))}
    res = apply_transformations(orig, trans, {"a": None})
    assert res == {"a": 1}
    assert RestoreArgs(restore_type=int).restore_type is int
