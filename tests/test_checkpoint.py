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


def test_flax_state_and_frozendict_checkpointing(tmp_path: pathlib.Path) -> None:
    """Test saving and restoring zero_flax State, Variable, and FrozenDict with zero_chex validation.

    Args:
        tmp_path: Temporary pytest directory.
    """
    import zero_chex as chex
    from zero_flax.core import FrozenDict
    import zero_flax.nnx as nnx
    from zero_jax import numpy as jnp
    from zero_orbax.checkpoint import (
        ArrayRestoreArgs,
        BasePyTreeCheckpointHandler,
        PyTreeCheckpointHandler,
        PyTreeCheckpointer,
        StandardCheckpointer,
        args,
    )
    from zero_orbax.core import SerializedArray
    from zero_orbax.types import ArrayProtocol, DType, array_to_nested_list

    state = nnx.State(
        {
            "kernel": nnx.Param(jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32), tag="w"),
            "sub": nnx.State(
                {
                    "running_mean": nnx.BatchStat(jnp.array([0.5, -0.5], dtype=jnp.float32)),
                    "scalar_val": 7,
                }
            ),
        }
    )

    pt_cp = PyTreeCheckpointer()
    ckpt_dir = tmp_path / "flax_state_pt"
    pt_cp.save(ckpt_dir, state)

    restored_no_tmpl = pt_cp.restore(ckpt_dir)
    assert isinstance(restored_no_tmpl, nnx.State)
    assert isinstance(restored_no_tmpl["kernel"], nnx.Param)
    assert restored_no_tmpl["kernel"].metadata.get("tag") == "w"
    assert isinstance(restored_no_tmpl["sub"], nnx.State)
    assert isinstance(restored_no_tmpl["sub"]["running_mean"], nnx.BatchStat)
    chex.assert_trees_all_close(restored_no_tmpl, state)

    target_state = nnx.State(
        {
            "kernel": nnx.Param(jnp.zeros((2, 2), dtype=jnp.float32)),
            "sub": nnx.State(
                {
                    "running_mean": nnx.BatchStat(jnp.zeros((2,), dtype=jnp.float32)),
                    "scalar_val": 0,
                }
            ),
        }
    )
    std_cp = StandardCheckpointer()
    std_dir = tmp_path / "flax_state_std"
    std_cp.save(std_dir, args.StandardSave(state))
    restored_with_tmpl = std_cp.restore(std_dir, item=target_state)
    assert isinstance(restored_with_tmpl, nnx.State)
    chex.assert_trees_all_close(restored_with_tmpl, state)
    chex.assert_trees_all_close(target_state, state)

    restore_args_tree: PyTree = {
        "kernel": ArrayRestoreArgs(shape=(2, 2), dtype="float64", strict=True),
        "sub": {
            "running_mean": ArrayRestoreArgs(global_shape=(2,), strict=True),
            "scalar_val": None,
        },
    }
    handler = PyTreeCheckpointHandler()
    restored_coerced = handler.restore(std_dir, item=target_state, restore_args=restore_args_tree)
    assert isinstance(restored_coerced, nnx.State)
    assert handler.structure(std_dir) is not None
    assert handler.structure(None) is None

    fd = FrozenDict(
        {
            "weights": jnp.array([10.0, 20.0], dtype=jnp.float32),
            "nested": FrozenDict({"bias": jnp.array([1.0], dtype=jnp.float32)}),
            "var_leaf": nnx.Cache(jnp.array([2.0], dtype=jnp.float32)),
        }
    )
    fd_dir = tmp_path / "frozendict_ckpt"
    pt_cp.save(fd_dir, fd)
    restored_fd = pt_cp.restore(fd_dir)
    assert isinstance(restored_fd, FrozenDict)
    assert isinstance(restored_fd["nested"], FrozenDict)
    chex.assert_trees_all_close(restored_fd.unfreeze(), fd.unfreeze())

    tmpl_fd = FrozenDict(
        {
            "weights": jnp.zeros((2,), dtype=jnp.float32),
            "nested": FrozenDict({"bias": jnp.zeros((1,), dtype=jnp.float32)}),
            "var_leaf": nnx.Cache(jnp.zeros((1,), dtype=jnp.float32)),
        }
    )
    restored_fd_tmpl = pt_cp.restore(fd_dir, item=tmpl_fd)
    assert isinstance(restored_fd_tmpl, FrozenDict)
    chex.assert_trees_all_close(restored_fd_tmpl.unfreeze(), fd.unfreeze())

    base_h = BasePyTreeCheckpointHandler()
    base_dir = tmp_path / "base_pytree"
    base_h.save(base_dir, state)
    assert isinstance(base_h.restore(base_dir), nnx.State)
    assert base_h.restore(None, item=state) is state
    assert isinstance(base_h.structure(base_dir), nnx.State)
    assert base_h.structure(None) is None

    unpacked_plain_to_state = base_h._unpack_pytree(
        {"kernel": [1.0, 2.0]}, nnx.State({"kernel": [0.0, 0.0]})
    )
    assert isinstance(unpacked_plain_to_state, nnx.State)
    unpacked_plain_to_fd = base_h._unpack_pytree(
        {"a": {"b": 1}},
        FrozenDict({"a": FrozenDict({"b": 0})}),
    )
    assert isinstance(unpacked_plain_to_fd, FrozenDict)

    sa = SerializedArray([1.0, 2.0], shape=(2,), dtype="float32")
    packed_sa = base_h._pack_pytree(sa)
    assert isinstance(base_h._unpack_pytree(packed_sa, None), SerializedArray)
    assert isinstance(base_h._unpack_pytree(sa, sa), SerializedArray)
    assert isinstance(base_h._unpack_pytree(sa, None), jnp.ndarray)
    assert isinstance(
        base_h._pack_pytree(nnx.Param(sa)),
        dict,
    )
    unpacked_var_from_sa = base_h._unpack_pytree(
        {"__zero_flax_variable__": True, "var_type": "Param", "value": sa, "metadata": {"k": 1}},
        None,
    )
    assert isinstance(unpacked_var_from_sa, nnx.Param)
    unpacked_var_non_dict_meta = base_h._unpack_pytree(
        {"__zero_flax_variable__": True, "var_type": "Param", "value": 1.0, "metadata": None},
        None,
    )
    assert isinstance(unpacked_var_non_dict_meta, nnx.Param)

    non_strict_args: PyTree = {
        "kernel": ArrayRestoreArgs(shape=(2, 2), dtype="float32", strict=False),
        "sub": {
            "running_mean": ArrayRestoreArgs(strict=False),
            "scalar_val": None,
        },
    }
    assert isinstance(
        handler.restore(std_dir, item=target_state, restore_args=non_strict_args),
        nnx.State,
    )

    from zero_chex.types import ZeroArray

    zero_arr = ZeroArray([1.0, 2.0], shape=(2,), dtype=DType("float32"))
    assert array_to_nested_list(zero_arr) == [1.0, 2.0]

    class _BareArray:
        """Minimal ArrayProtocol without tolist or to_list."""

        @property
        def shape(self) -> tuple[int, ...]:
            """Return shape.

            Returns:
                tuple[int, ...]: Shape tuple.
            """
            return (0,)

        @property
        def dtype(self) -> str:
            """Return dtype.

            Returns:
                str: Dtype name.
            """
            return "float32"

    bare = _BareArray()
    assert isinstance(bare, ArrayProtocol)
    assert array_to_nested_list(bare) == []


def test_optax_and_grain_composite_checkpointing(tmp_path: pathlib.Path) -> None:
    """Test saving and restoring zero_optax OptState and zero_grain DataLoaderIterator with CheckpointManager.

    Args:
        tmp_path: Temporary pytest directory.
    """
    import zero_chex as chex
    import zero_flax.nnx as nnx
    import zero_grain as grain
    from zero_jax import numpy as jnp
    import zero_optax as optax
    from zero_orbax.checkpoint import (
        BasePyTreeCheckpointHandler,
        CheckpointManager,
        PyGrainCheckpointHandler,
        PyTreeCheckpointer,
        args,
    )

    params = {"w": jnp.array([1.0, -2.0], dtype=jnp.float32)}
    tx = optax.adamw(1e-3)
    opt_state = tx.init(params)
    _, updated_opt_state = tx.update(
        {"w": jnp.array([0.25, -0.25], dtype=jnp.float32)}, opt_state, params
    )

    pt_cp = PyTreeCheckpointer()
    opt_dir = tmp_path / "opt_ckpt"
    pt_cp.save(opt_dir, updated_opt_state)

    restored_opt_dict = pt_cp.restore(opt_dir)
    assert isinstance(restored_opt_dict, dict)

    fresh_opt_state = tx.init(params)
    restored_opt = pt_cp.restore(opt_dir, item=fresh_opt_state)
    assert isinstance(restored_opt, optax.AdamwState)
    chex.assert_trees_all_close(restored_opt.adam_state.mu, updated_opt_state.adam_state.mu)
    chex.assert_trees_all_close(restored_opt.adam_state.nu, updated_opt_state.adam_state.nu)

    base_h = BasePyTreeCheckpointHandler()
    unpacked_plain_opt = base_h._unpack_pytree(
        {"count": 5},
        updated_opt_state.scale_state,
    )
    assert type(unpacked_plain_opt) is type(updated_opt_state.scale_state)

    assert args.has_registered_args(grain.PyGrainCheckpointHandler)
    assert (
        args.get_registered_args_cls(grain.PyGrainCheckpointHandler, for_save=True)
        is args.PyGrainCheckpointSave
    )
    assert (
        args.get_registered_args_cls(grain.PyGrainCheckpointHandler, for_save=False)
        is args.PyGrainCheckpointRestore
    )

    loader = grain.DataLoader(
        data_source=grain.RangeDataSource(0, 10, 1),
        sampler=grain.SequentialSampler(num_records=10),
    )
    data_iter = iter(loader)
    assert next(data_iter) == 0
    assert next(data_iter) == 1

    model_state = nnx.State({"w": nnx.Param(jnp.array([2.0, 4.0], dtype=jnp.float32))})
    comp_dir = tmp_path / "composite_mgr"
    mgr = CheckpointManager(comp_dir)
    assert mgr.save(
        1,
        args=args.Composite(
            model=args.StandardSave(model_state),
            opt=args.PyTreeSave(updated_opt_state),
            data=args.PyGrainCheckpointSave(data_iter),
        ),
    )
    assert next(data_iter) == 2
    assert next(data_iter) == 3

    target_model = nnx.State({"w": nnx.Param(jnp.zeros((2,), dtype=jnp.float32))})
    target_opt = tx.init(params)
    restored_comp = mgr.restore(
        1,
        items={
            "model": target_model,
            "opt": target_opt,
            "data": data_iter,
        },
    )
    assert isinstance(restored_comp, dict)
    assert isinstance(restored_comp["model"], nnx.State)
    chex.assert_trees_all_close(restored_comp["model"], model_state)
    assert type(restored_comp["opt"]) is type(updated_opt_state)
    assert next(data_iter) == 2

    restored_comp_no_items = mgr.restore(1)
    assert isinstance(restored_comp_no_items, dict)
    assert isinstance(restored_comp_no_items["data"], dict)

    gh = PyGrainCheckpointHandler()
    gh.save(None, data_iter)
    gh.save(tmp_path / "grain_ignored", {"not_iter": 1})
    assert gh.restore(None, data_iter) is data_iter
    assert gh.restore(tmp_path / "grain_missing", data_iter) is data_iter
    gh_dir = tmp_path / "grain_explicit"
    gh.save(gh_dir, args.PyGrainCheckpointSave(data_iter))
    assert isinstance(gh.restore(gh_dir, args.PyGrainCheckpointRestore(None)), dict)
    assert gh.restore(gh_dir, args.PyGrainCheckpointRestore(data_iter)) is data_iter

    empty_mgr_dir = tmp_path / "empty_step_mgr"
    empty_mgr = CheckpointManager(empty_mgr_dir)
    empty_mgr.steps.append(99)
    empty_mgr.checkpoints[99] = {"fallback": 1}
    assert empty_mgr.restore(99) == {"fallback": 1}
