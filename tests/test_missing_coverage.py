"""Tests verifying method behavior across managers, checkpointers, handlers, and registries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import datetime
import pathlib

import pytest
from ml_switcheroo_compiler import ops as compiler_ops
from ml_switcheroo_compiler.core.dtype import DType as CompilerDType
from ml_switcheroo_compiler.core.tensor import as_tensor_obj
from ml_switcheroo_ir import LogicalMesh, PartitionSpec
from zero_flax.core import FrozenDict
from zero_jax import numpy as jnp

from zero_orbax.checkpoint import (
    AbstractCheckpointManager,
    AbstractCheckpointer,
    ArrayCheckpointHandler,
    ArrayRestoreArgs,
    AsyncCheckpointHandler,
    AsyncCheckpointer,
    AsyncOptions,
    BasePyTreeCheckpointHandler,
    CheckpointHandler,
    CheckpointHandlerRegistry,
    CheckpointManager,
    CheckpointManagerOptions,
    Checkpointer,
    CompositeCheckpointHandler,
    Future,
    JsonCheckpointHandler,
    ProtoCheckpointHandler,
    PyTreeCheckpointHandler,
    RestoreArgs,
    RestoreTransform,
    SaveArgs,
    StandardCheckpointHandler,
    StandardCheckpointer,
    args,
    checkpoint_utils,
    merge_trees,
    nest_asyncio,
    step_lib,
    test_utils,
    _apply_restore_args_tree,
    _apply_save_args_tree,
    _infer_array_shape,
)
from zero_orbax.core import AtomicityTemporaryPath, SerializedArray
from zero_orbax.types import (
    AbstractLoggerProtocol,
    ArrayMetadataValidatorProtocol,
    CheckpointMetadataStoreProtocol,
    FileOptions,
    JsonValue,
    PathLike,
    PreservationPolicyProtocol,
    PyTree,
    SaveDecisionPolicyProtocol,
    TemporaryPathProtocol,
    TypeHandlerProtocol,
    TypeHandlerRegistryProtocol,
)


class CustomSavePolicy(SaveDecisionPolicyProtocol):
    """Test implementation of SaveDecisionPolicyProtocol."""

    def should_save(self, step: int, latest_step: int | None) -> bool:
        """Save only on step 42.

        Args:
            step: Candidate step.
            latest_step: Previous step.

        Returns:
            bool: True if step == 42.
        """
        return step == 42 and latest_step != 42


class CustomPreservationPolicy(PreservationPolicyProtocol):
    """Test implementation of PreservationPolicyProtocol."""

    def should_preserve(self, steps: Sequence[int]) -> Sequence[int]:
        """Preserve step 1 always.

        Args:
            steps: Existing steps.

        Returns:
            Sequence[int]: Preserved steps.
        """
        return [s for s in steps if s == 1]


class CustomLogger(AbstractLoggerProtocol):
    """Test implementation of AbstractLoggerProtocol."""

    def __init__(self) -> None:
        """Initialize the logger with an empty entries list."""
        self.entries: list[tuple[str, int | None]] = []

    def log_entry(self, msg: str, step: int | None = None) -> None:
        """Append a log entry.

        Args:
            msg: Message string.
            step: Step number.
        """
        self.entries.append((msg, step))


class InMemoryMetadataStore(CheckpointMetadataStoreProtocol):
    """Concrete test implementation of CheckpointMetadataStoreProtocol."""

    def __init__(self) -> None:
        """Initialize the metadata store."""
        self.store: dict[str, dict[str, JsonValue]] = {}
        self.closed: bool = False

    def write(self, path: PathLike, metadata: Mapping[str, JsonValue]) -> None:
        """Write metadata for ``path``.

        Args:
            path: Target path.
            metadata: Metadata mapping.
        """
        self.store[str(path)] = dict(metadata)

    def read(self, path: PathLike) -> dict[str, JsonValue] | None:
        """Read metadata for ``path``.

        Args:
            path: Source path.

        Returns:
            dict[str, JsonValue] | None: Stored metadata or None.
        """
        return self.store.get(str(path))

    def close(self) -> None:
        """Mark store as closed."""
        self.closed = True


class IntTypeHandler(TypeHandlerProtocol):
    """Custom TypeHandlerProtocol for integers."""

    def typestr(self) -> str:
        """Return type identifier.

        Returns:
            str: Type string.
        """
        return "custom_int"

    def serialize(self, value: PyTree) -> bytes:
        """Serialize integer to ASCII bytes.

        Args:
            value: Value to serialize.

        Returns:
            bytes: Encoded bytes.
        """
        return str(value).encode("utf-8")

    def deserialize(self, data: bytes) -> PyTree:
        """Deserialize ASCII bytes to integer.

        Args:
            data: Raw bytes.

        Returns:
            PyTree: Integer value.
        """
        return int(data.decode("utf-8"))


class SimpleTypeHandlerRegistry(TypeHandlerRegistryProtocol):
    """Concrete implementation of TypeHandlerRegistryProtocol."""

    def __init__(self) -> None:
        """Initialize registry."""
        self._map: dict[type, TypeHandlerProtocol] = {}

    def add(self, ty: type, handler: TypeHandlerProtocol, override: bool = False) -> None:
        """Register a type handler.

        Args:
            ty: Python type.
            handler: Handler instance.
            override: Whether to override.
        """
        if ty not in self._map or override:
            self._map[ty] = handler

    def get(self, ty: type) -> TypeHandlerProtocol | None:
        """Get a registered type handler.

        Args:
            ty: Python type.

        Returns:
            TypeHandlerProtocol | None: Handler or None.
        """
        return self._map.get(ty)

    def has(self, ty: type) -> bool:
        """Check if ``ty`` is registered.

        Args:
            ty: Python type.

        Returns:
            bool: True if registered.
        """
        return ty in self._map

    def get_by_typestr(self, typestr: str) -> TypeHandlerProtocol | None:
        """Look up a registered type handler by its ``typestr()``.

        Args:
            typestr: Canonical type string identifier.

        Returns:
            TypeHandlerProtocol | None: Matching handler or None.
        """
        for handler in self._map.values():
            if handler.typestr() == typestr:
                return handler
        return None


class RecordingValidator(ArrayMetadataValidatorProtocol):
    """Concrete ArrayMetadataValidatorProtocol recording validated snapshots."""

    def __init__(self) -> None:
        """Initialize the validator."""
        self.validated: list[dict[str, JsonValue]] = []

    def validate(self, metadata: Mapping[str, JsonValue]) -> bool:
        """Record and validate ``metadata``.

        Args:
            metadata: Metadata mapping.

        Returns:
            bool: Always True.
        """
        self.validated.append(dict(metadata))
        return True


class RecordingTempPath(TemporaryPathProtocol):
    """Concrete TemporaryPathProtocol implementation for testing Checkpointer staging."""

    created_count: int = 0
    finalized_count: int = 0
    base_dir: pathlib.Path = pathlib.Path(".")

    def get(self) -> pathlib.Path:
        """Return the temporary directory path.

        Returns:
            pathlib.Path: Temporary directory path.
        """
        return self.base_dir / "temp_stage"

    def get_final(self) -> pathlib.Path:
        """Return the final directory path.

        Returns:
            pathlib.Path: Final directory path.
        """
        return self.base_dir / "final_dest"

    def create(self) -> pathlib.Path:
        """Create the temporary directory.

        Returns:
            pathlib.Path: Created temporary directory.
        """
        RecordingTempPath.created_count += 1
        p = self.get()
        p.mkdir(parents=True, exist_ok=True)
        return p

    def finalize(self) -> None:
        """Record finalization."""
        RecordingTempPath.finalized_count += 1


class SyncOnlyHandler(CheckpointHandler):
    """Non-async CheckpointHandler for testing synchronous fallback branches."""


def test_abstract_checkpoint_manager_methods() -> None:
    """Test default methods on AbstractCheckpointManager."""
    mgr = AbstractCheckpointManager()
    assert mgr.best_step() is None
    assert mgr.should_save(0) is False
    mgr.delete(0)
    assert mgr.reached_preemption(0) is False
    mgr.check_for_errors()
    mgr.wait_until_finished()
    assert mgr.metadata() is None
    assert mgr.item_metadata() is None
    assert mgr.metrics() is None
    mgr.reload()
    mgr.close()


def test_abstract_checkpointer_methods() -> None:
    """Test default methods and context manager on AbstractCheckpointer."""
    with AbstractCheckpointer() as cp:
        assert cp.metadata() is None
        assert cp.structure() is None


def test_array_checkpoint_handler_methods(tmp_path: pathlib.Path) -> None:
    """Test ArrayCheckpointHandler async_save, strict shape validation, mesh sharding, and close.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    assert _infer_array_shape(SerializedArray([1, 2], shape=(2,))) == (2,)
    assert _infer_array_shape([]) == (0,)
    assert _infer_array_shape(5) == ()

    h = ArrayCheckpointHandler()
    assert h.async_save(None) is None
    futs = h.async_save(tmp_path, [10, 20, 30, 40])
    assert futs is not None and len(futs) == 1
    futs[0].result()
    assert h.metadata(tmp_path) == {"finalized": True, "directory": str(tmp_path)}
    assert h.structure(tmp_path) == [10, 20, 30, 40]

    # Strict shape validation pass and failure
    assert h.restore(tmp_path, restore_args=ArrayRestoreArgs(shape=(4,), strict=True)) == [
        10,
        20,
        30,
        40,
    ]
    with pytest.raises(ValueError, match="Strict shape mismatch"):
        h.restore(tmp_path, restore_args=ArrayRestoreArgs(shape=(2,), strict=True))

    # Mesh + mesh_axes sharding
    mesh = LogicalMesh(shape={"data": 2})
    sharded = h.restore(
        tmp_path,
        restore_args=ArrayRestoreArgs(
            mesh=mesh,
            mesh_axes=("data",),
            strict=False,
        ),
    )
    assert sharded == [10, 20]
    sharded_pspec = h.restore(
        tmp_path,
        restore_args=ArrayRestoreArgs(
            mesh=mesh,
            mesh_axes=PartitionSpec(axes=("data",)),
            strict=False,
        ),
    )
    assert sharded_pspec == [10, 20]
    h.close()


def test_array_restore_args_methods() -> None:
    """Test ArrayRestoreArgs.restore_type()."""
    ara = ArrayRestoreArgs(restore_type=int)
    assert ara.restore_type() is int


def test_async_checkpoint_handler_methods(tmp_path: pathlib.Path) -> None:
    """Test AsyncCheckpointHandler methods.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    h = AsyncCheckpointHandler()
    h.save(tmp_path, {"k": 1})
    assert h.restore(tmp_path) == {"k": 1}
    assert h.metadata(tmp_path) == {"directory": str(tmp_path)}
    assert h.structure(tmp_path) == {"k": 1}
    h.finalize(tmp_path)
    assert h.metadata() == {"finalized": True, "directory": str(tmp_path)}
    h.close()


def test_async_checkpointer_methods(tmp_path: pathlib.Path) -> None:
    """Test AsyncCheckpointer check_for_errors, wait_until_finished, temporary_path_class, and callbacks.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    barrier_called: list[tuple[str, int]] = []
    post_called: list[bool] = []

    opts = AsyncOptions(
        timeout_secs=60,
        barrier_sync_fn=lambda key, t: barrier_called.append((key, t)),
        post_finalization_callback=lambda: post_called.append(True),
    )
    meta_store = InMemoryMetadataStore()
    RecordingTempPath.base_dir = tmp_path / "async_temp_test"
    RecordingTempPath.created_count = 0
    RecordingTempPath.finalized_count = 0

    cp = AsyncCheckpointer(
        handler=StandardCheckpointHandler(),
        async_options=opts,
        checkpoint_metadata_store=meta_store,
        temporary_path_class=RecordingTempPath,
    )
    assert cp.timeout_secs == 60
    fut = cp.save(tmp_path, {"x": 1})
    assert isinstance(fut, Future)
    assert RecordingTempPath.created_count == 1
    cp.check_for_errors()
    cp.wait_until_finished()
    assert RecordingTempPath.finalized_count == 1
    assert len(barrier_called) == 1
    assert post_called == [True]
    assert (
        cp.create_temporary_path(tmp_path)
        == tmp_path.parent / f"{tmp_path.name}.orbax-checkpoint-tmp"
    )
    assert cp.create_temporary_path(None) is None
    assert cp.structure(RecordingTempPath().get()) == {"x": 1}
    cp.close()
    assert meta_store.closed is True

    # Test AsyncCheckpointer with real AtomicityTemporaryPath promotion
    atomic_dest = tmp_path / "atomic_async_dest"
    cp_atomic = AsyncCheckpointer(
        handler=StandardCheckpointHandler(),
        temporary_path_class=AtomicityTemporaryPath,
        file_options=FileOptions(path_permission_mode=0o755),
    )
    fut_atomic = cp_atomic.save(atomic_dest, {"real_atomic": 42})
    assert (tmp_path / "atomic_async_dest.orbax-checkpoint-tmp").exists()
    assert not atomic_dest.exists()
    fut_atomic.result()
    assert atomic_dest.exists()
    assert not (tmp_path / "atomic_async_dest.orbax-checkpoint-tmp").exists()
    restored_fut_atomic = cp_atomic.restore(atomic_dest)
    assert isinstance(restored_fut_atomic, Future)
    assert restored_fut_atomic.result() == {"real_atomic": 42}
    cp_atomic.close()

    # Test AsyncCheckpointer with file_options (no temporary_path_class)
    perm_async_dir = tmp_path / "perm_async_dir"
    cp_perm_async = AsyncCheckpointer(
        handler=StandardCheckpointHandler(),
        file_options=FileOptions(path_permission_mode=0o755),
    )
    cp_perm_async.save(perm_async_dir, {"perm": 1}).result()
    cp_perm_async.close()

    # Test AsyncCheckpointer with synchronous handler and barrier_sync_fn
    sync_barriers: list[tuple[str, int]] = []
    sync_cp = AsyncCheckpointer(
        handler=SyncOnlyHandler(),
        async_options=AsyncOptions(
            timeout_secs=30,
            barrier_sync_fn=lambda k, t: sync_barriers.append((k, t)),
        ),
    )
    sync_fut = sync_cp.save(tmp_path / "sync_dir", {"s": 2})
    assert sync_fut.done() is False
    sync_cp.wait_until_finished()
    assert sync_fut.done() is True
    assert len(sync_barriers) == 1

    sync_cp_no_barrier = AsyncCheckpointer(handler=SyncOnlyHandler())
    sync_cp_no_barrier.save(tmp_path / "sync_dir_nb", {"s": 3}).result()

    # Test Future exception capture and re-raise
    def _failing_commit() -> None:
        """Raise a RuntimeError during commit.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("background commit failed")

    bad_fut = Future(result=None, commit_callback=_failing_commit)
    with pytest.raises(RuntimeError, match="background commit failed"):
        bad_fut.result()
    with pytest.raises(RuntimeError, match="background commit failed"):
        bad_fut.result()


def test_base_pytree_checkpoint_handler_methods() -> None:
    """Test BasePyTreeCheckpointHandler.get_param_names and base methods."""
    h = BasePyTreeCheckpointHandler()
    assert h.get_param_names(None) is None
    names = h.get_param_names({"layer1": {"w": [1, 2], "b": 0}, "step": 1})
    assert names == {"layer1": {"w": "layer1.w", "b": "layer1.b"}, "step": "step"}
    h.finalize(None)
    assert h.metadata(None) is None
    assert h.structure(None) is None
    h.close()


def test_checkpoint_handler_methods(tmp_path: pathlib.Path) -> None:
    """Test CheckpointHandler disk fallback when restoring from msgpack.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    h = CheckpointHandler()
    h.save(tmp_path, args=args.StandardSave({"disk_key": 99}))
    assert h.restore(tmp_path) == {"disk_key": 99}
    assert h.restore(tmp_path / "empty_sub", item={"fallback": 1}) == {"fallback": 1}
    assert h.restore(None, item={"default": 0}) == {"default": 0}


def test_checkpoint_handler_registry_methods() -> None:
    """Test CheckpointHandlerRegistry add, get, has, and get_all_entries."""
    reg = CheckpointHandlerRegistry()
    h = CheckpointHandler()
    reg.add("item_a", args.StandardSave, h)
    assert reg.has("item_a", args.StandardSave) is True
    assert reg.get("item_a", args.StandardSave()) is h
    assert len(reg.get_all_entries()) == 1

    # Wildcard item=None or args_ty=None lookups
    reg.add("item_only", None, h)
    assert reg.get("item_only", args.StandardSave) is h
    reg.add(None, args.JsonSave, h)
    assert reg.get("any_item", args.JsonSave) is h
    assert reg.get("unregistered", args.ArraySave) is None


def test_checkpoint_manager_advanced_features(tmp_path: pathlib.Path) -> None:
    """Test CheckpointManager custom policies, logger, todelete_subdir, single/mapping handlers, and reload.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    # Create a tmp directory to verify cleanup_tmp_directories=True removes it on init
    stale_tmp = tmp_path / "0.orbax-checkpoint-tmp"
    stale_tmp.mkdir(parents=True)
    kept_dir = tmp_path / "regular_note"
    kept_dir.mkdir(parents=True)

    logger = CustomLogger()
    opts = CheckpointManagerOptions(
        save_decision_policy=CustomSavePolicy(),
        preservation_policy=CustomPreservationPolicy(),
        best_fn=lambda m: m.get("loss", 0.0),
        best_mode="min",
        max_to_keep=1,
        file_options=FileOptions(path_permission_mode=0o755),
        todelete_subdir="trash",
        cleanup_tmp_directories=True,
    )
    cp = Checkpointer(StandardCheckpointHandler())
    with CheckpointManager(
        tmp_path,
        checkpointers=cp,
        options=opts,
        metadata={"exp": "test"},
        logger=logger,
    ) as mgr:
        assert not stale_tmp.exists()
        assert kept_dir.exists()
        assert mgr.should_save(1) is False
        assert mgr.should_save(42) is True
        assert mgr.save(1, {"v": 1}, metrics={"loss": 0.5}, force=True) is True
        assert mgr.save(42, {"v": 42}, metrics={"loss": 0.1}) is True
        assert mgr.save(43, {"v": 43}, metrics={"loss": 0.9}, force=True) is True
        # Step 1 preserved by CustomPreservationPolicy, Step 42 preserved as best_step (min loss=0.1), Step 43 moved to trash
        assert (tmp_path / "trash" / "43").exists()
        assert mgr.best_step() == 42
        assert mgr.metrics(42) == {"loss": 0.1}
        assert mgr.metrics() == {"loss": 0.1}
        assert mgr.metadata() == {"exp": "test"}
        assert mgr.item_metadata(42) == {"v": 42}
        assert mgr.item_metadata() == {"v": 42}
        assert mgr.is_saving_in_progress() is False
        assert len(logger.entries) == 3

        # Lightweight initialize skips auto-reload until all_steps(read=True)
        mgr_light = CheckpointManager(
            tmp_path,
            options=CheckpointManagerOptions(lightweight_initialize=True),
        )
        assert mgr_light.metrics() is None
        assert mgr_light.item_metadata() is None
        assert mgr_light.best_step() is None

        # Default initialize auto-reloads from disk immediately
        mgr2 = CheckpointManager(tmp_path)
        assert mgr2.all_steps() == [1, 42]
        assert mgr2.all_steps(read=True) == [1, 42]
        assert mgr2.metadata() == {"exp": "test"}
        assert mgr2.metrics(42) == {"loss": 0.1}
        mgr2.delete(1)
        assert mgr2.all_steps() == [42]

    # Test CheckpointManager with todelete_full_path and item_handlers mapping/single
    trash_full = tmp_path / "external_trash"
    h_map_dir = tmp_path / "h_map_dir"
    opts_trash = CheckpointManagerOptions(
        max_to_keep=1,
        todelete_full_path=str(trash_full),
        should_keep_fn=lambda s, ts: s == 10,
        enable_should_save_is_saving_in_progress_check=True,
    )
    with CheckpointManager(
        h_map_dir,
        item_handlers={"a": StandardCheckpointHandler(), "b": JsonCheckpointHandler()},
        options=opts_trash,
    ) as mgr_h:
        mgr_h._saving_in_progress = True
        assert mgr_h.should_save(10) is False
        mgr_h._saving_in_progress = False
        mgr_h.save(10, {"a": {"x": 1}, "b": {"y": 2}})
        mgr_h.save(11, {"a": {"x": 3}, "b": {"y": 4}})
        mgr_h.save(12, {"a": {"x": 5}, "b": {"y": 6}})
        assert (trash_full / "11").exists()
        # Trigger overwrite in trash
        mgr_h.steps.append(11)
        mgr_h.save(11, {"a": {"x": 3}, "b": {"y": 4}})
        mgr_h.delete(11)
        assert mgr_h.restore(10) == {"a": {"x": 1}, "b": {"y": 2}}

    # Test single item_handler and directory override on restore (including step not in current manager)
    single_h_dir = tmp_path / "single_h"
    with CheckpointManager(
        single_h_dir,
        item_handlers=StandardCheckpointHandler(),
    ) as mgr_single:
        mgr_single.save(1, {"s": 10})
        assert mgr_single.restore(1, directory=single_h_dir) == {"s": 10}

    fresh_unloaded_mgr = CheckpointManager(
        tmp_path / "empty_other",
        options=CheckpointManagerOptions(lightweight_initialize=True),
    )
    with pytest.raises(ValueError, match="not found"):
        fresh_unloaded_mgr.restore()
    assert fresh_unloaded_mgr.restore(1, directory=single_h_dir) == {"s": 10}
    with pytest.raises(ValueError, match="not found"):
        fresh_unloaded_mgr.restore(999, directory=single_h_dir)

    # Test keep_time_interval and AsyncCheckpointer future tracking in CheckpointManager
    time_dir = tmp_path / "time_mgr"
    async_cp_map = {"main": AsyncCheckpointer(StandardCheckpointHandler())}
    with CheckpointManager(
        time_dir,
        checkpointers=async_cp_map,
        options=CheckpointManagerOptions(
            max_to_keep=1,
            keep_time_interval=datetime.timedelta(hours=1),
        ),
    ) as mgr_time:
        t0 = datetime.datetime(2026, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
        mgr_time._step_timestamps[1] = t0
        mgr_time.save(1, {"main": {"v": 1}})
        assert mgr_time.is_saving_in_progress() is True
        mgr_time.check_for_errors()
        mgr_time.wait_until_finished()
        assert mgr_time.is_saving_in_progress() is False
        # Step 2 is only 10 minutes after step 1 (< 1 hour interval), so step 1 is protected and step 2 is pruned when step 3 arrives
        mgr_time._step_timestamps[2] = t0 + datetime.timedelta(minutes=10)
        mgr_time.save(2, {"main": {"v": 2}})
        mgr_time._step_timestamps[3] = t0 + datetime.timedelta(minutes=20)
        mgr_time.save(3, {"main": {"v": 3}})
        assert 1 in mgr_time.all_steps()

    # Single AsyncCheckpointer in CheckpointManager
    single_async_dir = tmp_path / "single_async_mgr"
    with CheckpointManager(
        single_async_dir,
        checkpointers=AsyncCheckpointer(StandardCheckpointHandler()),
    ) as mgr_sa:
        mgr_sa.save(1, {"v": 10})
        mgr_sa.check_for_errors()
        mgr_sa.wait_until_finished()

    # Test get_registered_args_cls on unregistered CheckpointHandler subclass
    assert args.get_registered_args_cls(SyncOnlyHandler, for_save=True) is None
    assert args.get_registered_args_cls(SyncOnlyHandler, for_save=False) is None

    # Test CheckpointManager check_for_errors and wait_until_finished with non-async Checkpointer mapping and None checkpointers
    sync_cp_map_dir = tmp_path / "sync_cp_map"
    with CheckpointManager(
        sync_cp_map_dir,
        checkpointers={"sync": Checkpointer(StandardCheckpointHandler())},
    ) as mgr_sync_map:
        mgr_sync_map.save(1, {"sync": {"v": 1}})
        mgr_sync_map.check_for_errors()
        mgr_sync_map.wait_until_finished()
    with CheckpointManager(tmp_path / "no_cp_mgr") as mgr_no_cp:
        mgr_no_cp.check_for_errors()
        mgr_no_cp.wait_until_finished()

    # Test Composite args with CheckpointManager
    comp_mgr_dir = tmp_path / "comp_mgr"
    with CheckpointManager(
        comp_mgr_dir,
        item_names=("state", "cfg"),
        item_handlers={"state": StandardCheckpointHandler(), "cfg": JsonCheckpointHandler()},
    ) as mgr_comp:
        mgr_comp.save(
            1,
            args=args.Composite(
                state=args.StandardSave({"w": 1}),
                cfg=args.JsonSave({"lr": 0.01}),
            ),
        )
        restored_comp = mgr_comp.restore(
            1,
            args=args.Composite(
                state=args.StandardRestore(),
                cfg=args.JsonRestore(),
            ),
        )
        assert restored_comp == {"state": {"w": 1}, "cfg": {"lr": 0.01}}

    # Test CheckpointManager with registered and unregistered args class directly
    reg_mgr_dir = tmp_path / "reg_mgr"
    with CheckpointManager(
        reg_mgr_dir,
        options=CheckpointManagerOptions(
            step_name_format=step_lib.standard_name_format(step_prefix="s"),
            should_save_fn=lambda s, latest: s == 1,
        ),
    ) as mgr_reg:
        assert mgr_reg.should_save(1) is True
        mgr_reg.save(1, args=args.JsonSave({"direct_json": True}))
        assert mgr_reg.restore(1, args=args.JsonRestore()) == {"direct_json": True}
        # Unregistered CheckpointArgs fallback
        mgr_reg.save(2, args=args.CheckpointArgs({"unreg": 1}), force=True)
        assert mgr_reg.restore(2, args=args.CheckpointArgs()) == {"unreg": 1}

    # Test CheckpointManager with directory=None, read_only should_save, best_fn without metrics
    mem_mgr = CheckpointManager(
        None,
        options=CheckpointManagerOptions(
            max_to_keep=1,
            best_fn=lambda m: m.get("score", 0),
            keep_checkpoints_without_metrics=True,
        ),
    )
    mem_mgr.reload()
    mem_mgr.save(1, {"a": 1})
    mem_mgr.save(2, {"a": 2})
    assert mem_mgr.best_step() == 2
    assert mem_mgr.restore(2) == {"a": 2}
    with pytest.raises(ValueError, match="not found"):
        mem_mgr.restore(999)
    mem_mgr.delete(1)

    ro_mgr = CheckpointManager(
        None,
        options=CheckpointManagerOptions(read_only=True),
    )
    assert ro_mgr.should_save(1) is False

    # Test Composite args on manager without item_handlers mapping
    comp_no_h_dir = tmp_path / "comp_no_h"
    with CheckpointManager(
        comp_no_h_dir,
        options=CheckpointManagerOptions(
            temporary_path_class=AtomicityTemporaryPath,
            enable_async_checkpointing=False,
        ),
    ) as mgr_c2:
        mgr_c2.save(1, args=args.Composite(a=args.StandardSave({"v": 1})))
        assert mgr_c2.best_step() == 1
        assert mgr_c2.restore(1, args=args.Composite(a=args.StandardRestore())) == {"a": {"v": 1}}
        assert mgr_c2.item_metadata(1) == {"a": {"v": 1}}
        assert mgr_c2.metadata(1) is None
        # When composite subdirs are removed, restore falls back to items
        import shutil as _shutil

        _shutil.rmtree(comp_no_h_dir / "1" / "a")
        assert mgr_c2.restore(1, items={"empty_fb": 1}) == {"empty_fb": 1}

    # Test CheckpointManager.metadata(step) when _orbax_metadata.json exists, and item_metadata when step not in steps
    meta_step_dir = tmp_path / "meta_step_mgr"
    with CheckpointManager(
        meta_step_dir,
        item_handlers=StandardCheckpointHandler(),
    ) as mgr_ms:
        mgr_ms.save(5, {"w": 10})
        step_meta = mgr_ms.metadata(5)
        assert isinstance(step_meta, dict) and "directory" in step_meta
        assert mgr_ms.item_metadata(999) is None
        # When disk file is removed after save, restore falls back to items
        (meta_step_dir / "5" / "checkpoint.msgpack").unlink(missing_ok=True)
        (meta_step_dir / "5" / "_orbax_metadata.json").unlink(missing_ok=True)
        assert mgr_ms.restore(5, items={"fallback_tmpl": 7}) == {"fallback_tmpl": 7}


def test_checkpoint_manager_options_methods() -> None:
    """Test CheckpointManagerOptions.replace()."""
    opts = CheckpointManagerOptions(save_interval_steps=2, max_to_keep=3)
    opts2 = opts.replace(save_interval_steps=5, max_to_keep=10, best_mode="min")
    assert opts2.save_interval_steps == 5
    assert opts2.max_to_keep == 10
    assert opts2.best_mode == "min"


def test_checkpointer_methods(tmp_path: pathlib.Path) -> None:
    """Test Checkpointer without handler, with file_options, temporary_path_class, and metadata store.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    cp = Checkpointer(file_options=FileOptions(path_permission_mode=0o755))
    assert cp.save(tmp_path / "perm_dir", {"no_handler": 1}) is None
    assert cp.restore(tmp_path / "perm_dir") == {"no_handler": 1}
    assert cp.restore(tmp_path / "missing", item={"default": 2}) == {"default": 2}
    assert cp.metadata(tmp_path / "perm_dir") is None
    assert cp.structure(tmp_path / "perm_dir") is None
    cp.close()

    RecordingTempPath.base_dir = tmp_path / "sync_temp"
    RecordingTempPath.created_count = 0
    RecordingTempPath.finalized_count = 0
    cp_temp = Checkpointer(
        StandardCheckpointHandler(),
        temporary_path_class=RecordingTempPath,
    )
    cp_temp.save(tmp_path / "ignored_dest", {"staged": 1})
    assert RecordingTempPath.created_count == 1
    assert RecordingTempPath.finalized_count == 1
    cp_temp.close()

    # Test Checkpointer with real AtomicityTemporaryPath promotion and barrier_sync_fn
    sync_barriers: list[str] = []
    real_dest = tmp_path / "real_staged_dest"
    cp_real_temp = Checkpointer(
        StandardCheckpointHandler(),
        temporary_path_class=AtomicityTemporaryPath,
        file_options=FileOptions(path_permission_mode=0o755),
        barrier_sync_fn=lambda k, t: sync_barriers.append(k),
    )
    cp_real_temp.save(real_dest, {"real_staged": 99})
    assert real_dest.exists()
    assert cp_real_temp.restore(real_dest) == {"real_staged": 99}
    assert sync_barriers == [str(real_dest)]
    cp_real_temp.close()

    meta_store = InMemoryMetadataStore()
    cp_meta = Checkpointer(StandardCheckpointHandler(), checkpoint_metadata_store=meta_store)
    cp_meta.save(tmp_path / "with_meta", {"m": 1})
    assert cp_meta.metadata(tmp_path / "with_meta") == {"path": str(tmp_path / "with_meta")}
    cp_meta.close()


def test_composite_checkpoint_handler_extra_branches(tmp_path: pathlib.Path) -> None:
    """Test CompositeCheckpointHandler with sync sub-handler, None handler, dict item, and metadata branches.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    comp = CompositeCheckpointHandler(sync_item=SyncOnlyHandler(), ignored_item=None)
    assert comp.metadata(tmp_path) is None
    comp.finalize(None)
    futs = comp.async_save(tmp_path, {"sync_item": {"v": 1}})
    assert futs == []
    assert comp.restore(tmp_path) == {"sync_item": {"v": 1}, "ignored_item": None}
    # Test unregistered CheckpointArgs subclass inside Composite
    comp.save(tmp_path, args.Composite(unreg=args.CheckpointArgs({"u": 2})))
    comp.close()


def test_pytree_checkpoint_handler_custom_impl_and_type_registry(tmp_path: pathlib.Path) -> None:
    """Test PyTreeCheckpointHandler with handler_impl, type_handler_registry, validator, priority keys, and SaveArgs/RestoreArgs.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    assert _apply_save_args_tree(None, SaveArgs(dtype=int)) is None
    assert _apply_save_args_tree(10, "not_a_save_args") == 10
    coerced_save_tree = _apply_save_args_tree(
        {
            "a": "10",
            "b": 2,
            "c": 3,
            "d": [1],
            "seq_list": ["5", "6"],
            "seq_tuple": ("7", "8"),
            "jnp_arr": jnp.asarray([1.0, 2.0], dtype="float32"),
            "ser_arr": SerializedArray([1.0, 2.0], shape=(2,), dtype="float32"),
            "ct_arr": as_tensor_obj(compiler_ops.asarray([1, 2], dtype=CompilerDType.int32)),
            "empty_sa": 9,
        },
        {
            "a": SaveArgs(dtype=int),
            "b": SaveArgs(dtype=float),
            "c": SaveArgs(dtype=str),
            "d": SaveArgs(dtype=bool),
            "seq_list": [SaveArgs(dtype=int), SaveArgs(dtype=int)],
            "seq_tuple": (SaveArgs(dtype=int), SaveArgs(dtype=int)),
            "jnp_arr": SaveArgs(dtype="int32"),
            "ser_arr": SaveArgs(dtype="int32"),
            "ct_arr": SaveArgs(dtype=float),
            "empty_sa": {"nested_non_save_args": None},
        },
    )
    assert isinstance(coerced_save_tree, dict) and coerced_save_tree["seq_list"] == [5, 6]
    assert _apply_restore_args_tree(
        FrozenDict({"x": "10"}),
        {"x": RestoreArgs(restore_type=int)},
    ) == FrozenDict({"x": 10})
    assert _apply_restore_args_tree(
        ["1", ("2",)],
        [RestoreArgs(restore_type=int), (RestoreArgs(restore_type=int),)],
    ) == [1, (2,)]

    custom_impl = BasePyTreeCheckpointHandler()
    pt_with_impl = PyTreeCheckpointHandler(handler_impl=custom_impl)
    pt_with_impl.save(tmp_path / "impl", {"a": 1})
    assert pt_with_impl.restore(tmp_path / "impl") == {"a": 1}

    validator = RecordingValidator()
    th_reg = SimpleTypeHandlerRegistry()
    th_reg.add(int, IntTypeHandler())
    th_reg.add(int, IntTypeHandler(), override=False)
    assert th_reg.has(int) is True
    assert RecordingTempPath().get_final() == RecordingTempPath.base_dir / "final_dest"
    pt_th = PyTreeCheckpointHandler(
        type_handler_registry=th_reg,
        array_metadata_validator=validator,
        is_prioritized_key_fn=lambda k: "prio" in k,
    )
    pt_th.save(
        tmp_path / "th",
        {"z_normal": {"nested": [1, (2,)]}, "a_prio": 42},
        save_args={"a_prio": SaveArgs(dtype=int), "z_normal": {"nested": None}},
    )
    assert len(validator.validated) == 1
    restored_th = pt_th.restore(tmp_path / "th")
    assert restored_th == {"z_normal": {"nested": [1, (2,)]}, "a_prio": 42}
    assert th_reg.get_by_typestr("unknown_type") is None

    # Test when type_handler_registry does not have the typestr during restore
    empty_reg = SimpleTypeHandlerRegistry()
    pt_empty_reg = PyTreeCheckpointHandler(type_handler_registry=empty_reg)
    raw_unmatched = pt_empty_reg.restore(tmp_path / "th")
    assert isinstance(raw_unmatched, dict)

    # Test _apply_restore_args_tree for float, str, list, tuple, and unknown restore_type
    pt_plain = PyTreeCheckpointHandler()
    pt_plain.save(
        tmp_path / "coercions",
        {"f": 1, "s": 123, "lst": (1, 2), "tup": [3, 4], "keep": "ok", "other": 5},
    )
    coerced = pt_plain.restore(
        tmp_path / "coercions",
        restore_args={
            "f": RestoreArgs(restore_type=float),
            "s": RestoreArgs(restore_type=str),
            "lst": RestoreArgs(restore_type=list),
            "tup": ArrayRestoreArgs(restore_type=tuple),
            "keep": None,
            "other": RestoreArgs(restore_type=bool),
        },
    )
    assert coerced == {
        "f": 1.0,
        "s": "123",
        "lst": [1, 2],
        "tup": (3, 4),
        "keep": "ok",
        "other": 5,
    }

    # Test PyTreeCheckpointHandler aggregate_filename missing on disk fallback and unregistered leaf in type_handler_registry
    pt_agg_missing = PyTreeCheckpointHandler(
        aggregate_filename="missing_agg.msgpack",
        type_handler_registry=th_reg,
    )
    pt_agg_missing.save(None, {"num": 1, "str_unregistered": "hello"})
    assert pt_agg_missing.restore(tmp_path / "coercions") is not None


def test_proto_handler_default_and_custom_filename(tmp_path: pathlib.Path) -> None:
    """Test ProtoCheckpointHandler with default proto.pb and custom args.ProtoSave(filename=...).

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    proto_h = ProtoCheckpointHandler()
    proto_h.save(tmp_path, args.ProtoSave({"msg": "hello"}, filename="custom.pb"))
    assert (tmp_path / "custom.pb").exists()
    assert proto_h.restore(tmp_path, args.ProtoRestore(filename="custom.pb")) == {"msg": "hello"}


def test_standard_checkpointer_methods(tmp_path: pathlib.Path) -> None:
    """Test StandardCheckpointer save, restore, check_for_errors, metadata, structure, and close.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    cp = StandardCheckpointer()
    cp.check_for_errors()
    assert cp.metadata() is None
    assert cp.create_temporary_path() is None
    cp.wait_until_finished()
    assert cp.structure() is None
    std_dir = tmp_path / "std_cp_ckpt"
    cp.save(std_dir, {"weights": [1, 2, 3]})
    cp.wait_until_finished()
    assert cp.metadata(std_dir) == {"finalized": True, "directory": str(std_dir)}
    assert cp.structure(std_dir) == {"weights": [1, 2, 3]}
    cp.close()


def test_utils_and_submodules(tmp_path: pathlib.Path) -> None:
    """Test test_utils, checkpoint_utils, nest_asyncio, step_lib, and RestoreTransform.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    nest_asyncio.apply()

    nf = step_lib.standard_name_format(step_prefix="ckpt", step_format_fixed_length=4)
    assert nf.build_name(7) == "ckpt_0007"
    assert nf.parse_step("ckpt_0007") == 7
    assert nf.parse_step("other_0007") is None
    assert nf.parse_step("ckpt_abc") is None

    nf_plain = step_lib.standard_name_format()
    assert nf_plain.build_name(7) == "7"
    assert nf_plain.parse_step("7") == 7

    target_dir = tmp_path / "iter_dir"
    test_utils.erase_and_create_empty(target_dir)
    (target_dir / "1").mkdir()
    (target_dir / "3").mkdir()
    (target_dir / "not_a_step").mkdir()
    assert list(checkpoint_utils.checkpoints_iterator(target_dir)) == [1, 3]
    assert list(checkpoint_utils.checkpoints_iterator(target_dir, min_interval_secs=-1)) == []

    rt = RestoreTransform(
        value_fn=lambda x: x,
        original_key="a",
        use_fallback=True,
        multi_value_fn_input_args={"a": RestoreArgs(restore_type=int)},
    )
    assert rt.original_key == "a"
    assert rt.use_fallback is True
    assert merge_trees({"a": 1}, target={"b": 2}) == {"a": 1, "b": 2}
