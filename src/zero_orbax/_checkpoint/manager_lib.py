"""CheckpointManager, CheckpointManagerOptions, and AbstractCheckpointManager."""

from __future__ import annotations

import dataclasses
import datetime
import json
from collections.abc import Container, Mapping, Sequence
from types import TracebackType
from typing import Literal
from typing_extensions import TypeAlias

from zero_orbax._checkpoint.args_lib import _CheckpointArgsBase, _CompositeArgsCls, args
from zero_orbax._checkpoint.checkpointer_lib import (
    AbstractCheckpointer,
    AsyncCheckpointer,
    _instantiate_temporary_path,
)
from zero_orbax._checkpoint.future_lib import Future
from zero_orbax._checkpoint.handlers_lib import (
    BasePyTreeCheckpointHandler,
    CheckpointHandler,
    CheckpointHandlerRegistry,
    CompositeCheckpointHandler,
)
from zero_orbax._checkpoint.step_lib import step_lib
from zero_orbax.core import AtomicWrite, FileSystemIO, MsgpackEngine
from zero_orbax.types import (
    AbstractLoggerProtocol,
    AsyncOptions,
    AtomicityOptions,
    BestFn,
    FileOptions,
    GrainIteratorProtocol,
    JsonValue,
    MultiprocessingOptions,
    PathLike,
    PreservationPolicyProtocol,
    PyTree,
    SaveDecisionPolicyProtocol,
    ShouldKeepFn,
    ShouldSaveFn,
    TemporaryPathProtocol,
    is_pytree_dict,
)


@dataclasses.dataclass
class CheckpointManagerOptions:
    """Options configuring the lifecycle, retention, and I/O behavior of ``CheckpointManager``."""

    save_interval_steps: int = 1
    max_to_keep: int | None = None
    keep_time_interval: datetime.timedelta | None = None
    keep_period: int | None = None
    should_save_fn: ShouldSaveFn | None = None
    best_fn: BestFn | None = None
    best_mode: Literal["min", "max"] | str = "max"
    keep_checkpoints_without_metrics: bool = True
    step_prefix: str | None = None
    step_format_fixed_length: int | None = None
    step_name_format: step_lib.NameFormat | None = None
    create: bool = True
    cleanup_tmp_directories: bool = False
    save_on_steps: Container[int] | None = None
    single_host_load_and_broadcast: bool = False
    todelete_subdir: str | None = None
    todelete_full_path: str | None = None
    enable_background_delete: bool = False
    num_deletion_threads: int = 1
    read_only: bool = False
    enable_async_checkpointing: bool = True
    async_options: AsyncOptions | None = None
    multiprocessing_options: MultiprocessingOptions | None = None
    should_keep_fn: ShouldKeepFn | None = None
    file_options: FileOptions | None = None
    atomicity_options: AtomicityOptions | None = None
    save_root_metadata: bool = True
    temporary_path_class: type[TemporaryPathProtocol] | None = None
    save_decision_policy: SaveDecisionPolicyProtocol | None = None
    preservation_policy: PreservationPolicyProtocol | None = None
    prevent_write_metrics: bool = False
    enable_should_save_is_saving_in_progress_check: bool = False
    enable_per_process_directory_creation: bool = False
    lightweight_initialize: bool = False

    def __post_init__(self) -> None:
        """Validate option combinations and normalize ``save_on_steps`` after initialization.

        Raises:
            ValueError: If an invalid or contradictory option combination is provided.
        """
        step_name_format_shlb = (
            isinstance(self.step_name_format, step_lib.StandardNameFormat)
            and self.step_name_format.single_host_load_and_broadcast
        )
        if self.single_host_load_and_broadcast and self.step_name_format is not None:
            if not step_name_format_shlb:
                raise ValueError(
                    "`CheckpointManagerOptions.single_host_load_and_broadcast=True` requires "
                    "`step_name_format.single_host_load_and_broadcast` to be set to True."
                )
        if step_name_format_shlb and not self.single_host_load_and_broadcast:
            raise ValueError(
                "`step_name_format.single_host_load_and_broadcast=True` but "
                "`CheckpointManagerOptions.single_host_load_and_broadcast=False`."
            )
        if self.best_mode not in ("min", "max"):
            raise ValueError(
                "`CheckpointManagerOptions.best_mode` must be one of None, 'min' or 'max'. "
                f"Got {self.best_mode}."
            )
        if (
            self.preservation_policy is None
            and self.max_to_keep is not None
            and self.max_to_keep < 0
        ):
            raise ValueError("Setting of `max_to_keep` must be None or non-negative.")
        if self.save_interval_steps <= 0 and not self.read_only:
            raise ValueError("save_interval_steps must be positive")
        if self.read_only:
            if self.save_interval_steps > 0:
                self.save_interval_steps = 0
            if self.max_to_keep is not None:
                self.max_to_keep = None
            if self.keep_time_interval is not None:
                self.keep_time_interval = None
            if self.keep_period is not None:
                self.keep_period = None
                self.should_keep_fn = None
            if self.create:
                self.create = False
            if self.cleanup_tmp_directories:
                self.cleanup_tmp_directories = False
            if self.save_on_steps:
                self.save_on_steps = None
            if self.todelete_subdir is not None:
                self.todelete_subdir = None
            if self.todelete_full_path is not None:
                self.todelete_full_path = None
            if self.should_save_fn is not None:
                self.should_save_fn = None
        if self.todelete_subdir is not None and self.todelete_full_path is not None:
            raise ValueError("todelete_subdir and todelete_full_path both cannot be set togther")
        if self.preservation_policy is None and self.should_keep_fn is not None:
            self.keep_period = None
        if isinstance(self.save_on_steps, (list, tuple, set, frozenset)):
            self.save_on_steps = frozenset(self.save_on_steps)
        else:
            self.save_on_steps = frozenset()

    def replace(
        self,
        *,
        save_interval_steps: int | None = None,
        max_to_keep: int | None = None,
        keep_period: int | None = None,
        read_only: bool | None = None,
        best_mode: Literal["min", "max"] | str | None = None,
        create: bool | None = None,
        step_prefix: str | None = None,
        step_format_fixed_length: int | None = None,
    ) -> CheckpointManagerOptions:
        """Return a new ``CheckpointManagerOptions`` instance with updated fields.

        Args:
            save_interval_steps: Optional replacement ``save_interval_steps``. Defaults to None.
            max_to_keep: Optional replacement ``max_to_keep``. Defaults to None.
            keep_period: Optional replacement ``keep_period``. Defaults to None.
            read_only: Optional replacement ``read_only``. Defaults to None.
            best_mode: Optional replacement ``best_mode``. Defaults to None.
            create: Optional replacement ``create``. Defaults to None.
            step_prefix: Optional replacement ``step_prefix``. Defaults to None.
            step_format_fixed_length: Optional replacement ``step_format_fixed_length``. Defaults to None.

        Returns:
            CheckpointManagerOptions: A new options instance with the requested updates applied.
        """
        return dataclasses.replace(
            self,
            save_interval_steps=(
                save_interval_steps if save_interval_steps is not None else self.save_interval_steps
            ),
            max_to_keep=max_to_keep if max_to_keep is not None else self.max_to_keep,
            keep_period=keep_period if keep_period is not None else self.keep_period,
            read_only=read_only if read_only is not None else self.read_only,
            best_mode=best_mode if best_mode is not None else self.best_mode,
            create=create if create is not None else self.create,
            step_prefix=step_prefix if step_prefix is not None else self.step_prefix,
            step_format_fixed_length=(
                step_format_fixed_length
                if step_format_fixed_length is not None
                else self.step_format_fixed_length
            ),
        )


CheckpointersDict: TypeAlias = dict[str, AbstractCheckpointer]
CheckpointHandlersDict: TypeAlias = dict[str, CheckpointHandler]


class AbstractCheckpointManager:
    """Abstract interface for managing multi-step checkpoints over training runs."""

    def __init__(self, *args: PyTree, **kwargs: PyTree) -> None:
        """Initialize the AbstractCheckpointManager.

        Args:
            *args: Optional positional initialization arguments.
            **kwargs: Optional keyword initialization arguments.
        """
        _ = (args, kwargs)
        self._closed: bool = False

    def save(
        self,
        step: int,
        items: PyTree = None,
        save_kwargs: Mapping[str, JsonValue] | None = None,
        metrics: Mapping[str, float | int] | None = None,
        force: bool = False,
        args: args.CheckpointArgs | None = None,
        **kwargs: PyTree,
    ) -> bool:
        """Save a checkpoint at the given training step.

        Args:
            step: The training step number at which to save.
            items: The PyTree or mapping of items to save. Defaults to None.
            save_kwargs: Optional keyword arguments for the underlying checkpointer. Defaults to None.
            metrics: Optional dictionary of metric values for best-step tracking. Defaults to None.
            force: If True, bypass ``should_save(step)`` interval checks. Defaults to False.
            args: Optional structured ``CheckpointArgs`` payload. Defaults to None.
            **kwargs: Additional options for saving.

        Returns:
            bool: True if the save was executed, False if skipped.

        Raises:
            NotImplementedError: If not implemented by subclasses.
        """
        raise NotImplementedError

    def restore(
        self,
        step: int | None = None,
        items: PyTree = None,
        restore_kwargs: Mapping[str, JsonValue] | None = None,
        directory: PathLike | None = None,
        args: args.CheckpointArgs | None = None,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore a checkpoint from a specific training step.

        Args:
            step: The step number to restore (or None for latest). Defaults to None.
            items: The target structure to restore into. Defaults to None.
            restore_kwargs: Optional keyword arguments for restoration. Defaults to None.
            directory: Optional override base directory. Defaults to None.
            args: Optional structured ``CheckpointArgs`` for restoration. Defaults to None.
            **kwargs: Additional options for restoring.

        Returns:
            PyTree: The restored objects.

        Raises:
            NotImplementedError: If not implemented by subclasses.
        """
        raise NotImplementedError

    def latest_step(self) -> int | None:
        """Return the most recently saved step number.

        Returns:
            int | None: The latest step number, or ``None`` if no checkpoints exist.

        Raises:
            NotImplementedError: If not implemented by subclasses.
        """
        raise NotImplementedError

    def all_steps(self, read: bool = False) -> Sequence[int]:
        """Return all steps that currently have saved checkpoints.

        Args:
            read: If True, force reading step directories from storage. Defaults to False.

        Returns:
            Sequence[int]: Sequence of available step numbers.

        Raises:
            NotImplementedError: If not implemented by subclasses.
        """
        raise NotImplementedError

    def best_step(self) -> int | None:
        """Return the step with the best metric score.

        Returns:
            int | None: Best step number, or ``None``.
        """
        return None

    def should_save(self, step: int) -> bool:
        """Return whether a checkpoint should be saved at ``step``.

        Args:
            step: Candidate training step.

        Returns:
            bool: True if a checkpoint should be saved.
        """
        return False

    def delete(self, step: int) -> None:
        """Delete the checkpoint for a given step.

        Args:
            step: Step number to delete.
        """
        return None

    def reached_preemption(self, step: int = 0) -> bool:
        """Return whether a preemption signal has been received at ``step``.

        Args:
            step: Current training step. Defaults to 0.

        Returns:
            bool: True if preemption was reached, False otherwise.
        """
        return False

    def check_for_errors(self) -> None:
        """Check for any asynchronous background errors."""
        return None

    def wait_until_finished(self) -> None:
        """Block until any in-flight background operations complete."""
        return None

    def metadata(self, step: int | None = None) -> Mapping[str, JsonValue] | int | None:
        """Return root or step-level metadata.

        Args:
            step: Optional step number. Defaults to None.

        Returns:
            Mapping[str, JsonValue] | int | None: Metadata value.
        """
        return None

    def item_metadata(self, step: int | None = None) -> PyTree:
        """Return item metadata for the given ``step``.

        Args:
            step: Optional step number. Defaults to None.

        Returns:
            PyTree: Item metadata or None.
        """
        return None

    def metrics(self, step: int | None = None) -> Mapping[str, float | int] | None:
        """Return recorded evaluation metrics for the given ``step``.

        Args:
            step: Optional step number. Defaults to None.

        Returns:
            Mapping[str, float | int] | None: Metrics dictionary or None.
        """
        return None

    def reload(self) -> None:
        """Reload internal step tracking state from the underlying directory."""
        return None

    def close(self) -> None:
        """Close the manager and release any held checkpointers."""
        self._closed = True


class CheckpointManager(AbstractCheckpointManager):
    """Manages saving, restoring, retention, and best-step tracking of checkpoints."""

    def __init__(
        self,
        directory: PathLike | None = None,
        checkpointers: AbstractCheckpointer | Mapping[str, AbstractCheckpointer] | None = None,
        options: CheckpointManagerOptions | None = None,
        metadata: Mapping[str, JsonValue] | int | None = None,
        item_names: Sequence[str] | None = None,
        item_handlers: CheckpointHandler | Mapping[str, CheckpointHandler] | None = None,
        logger: AbstractLoggerProtocol | None = None,
        handler_registry: CheckpointHandlerRegistry | None = None,
    ) -> None:
        """Initialize the CheckpointManager.

        Args:
            directory: Base directory for storing step checkpoints. Defaults to None.
            checkpointers: Single checkpointer or mapping of item checkpointers. Defaults to None.
            options: ``CheckpointManagerOptions`` configuring save/retention policies. Defaults to None.
            metadata: Optional root metadata mapping or integer tag. Defaults to None.
            item_names: Optional sequence of item names managed by this instance. Defaults to None.
            item_handlers: Optional handler or mapping of item handlers. Defaults to None.
            logger: Optional structured logger implementing ``AbstractLoggerProtocol``. Defaults to None.
            handler_registry: Optional ``CheckpointHandlerRegistry`` instance. Defaults to None.
        """
        super().__init__()
        self.directory = directory
        self.checkpointers = checkpointers
        self.options = options
        self._metadata = metadata
        self.item_names = item_names
        self.item_handlers = item_handlers
        self.logger = logger
        self.handler_registry = handler_registry
        self.steps: list[int] = []
        self.checkpoints: dict[int, PyTree] = {}
        self._step_metrics: dict[int, dict[str, float | int]] = {}
        self._step_timestamps: dict[int, datetime.datetime] = {}
        self._pending_futures: list[Future] = []
        self.latest: int | None = None
        self._saving_in_progress: bool = False

        if self.directory is not None and self.options is not None and self.options.create:
            perm = (
                self.options.file_options.path_permission_mode
                if self.options.file_options is not None
                else None
            )
            FileSystemIO().makedirs(self.directory, exist_ok=True, permission_mode=perm)
        if (
            self.directory is not None
            and self.options is not None
            and self.options.cleanup_tmp_directories
        ):
            io_clean = FileSystemIO()
            base_clean = str(self.directory).rstrip("/")
            for entry in io_clean.listdir(self.directory):
                if entry.endswith(".orbax-checkpoint-tmp") or ".tmp." in entry:
                    io_clean.remove(f"{base_clean}/{entry}")
        if (
            self.directory is not None
            and self._metadata is not None
            and (self.options is None or self.options.save_root_metadata)
            and (self.options is None or not self.options.read_only)
        ):
            AtomicWrite().atomic_write_bytes(
                f"{str(self.directory).rstrip('/')}/_root_metadata.json",
                json.dumps({"metadata": self._metadata}).encode("utf-8"),
            )
        if self.directory is not None and (
            self.options is None or not self.options.lightweight_initialize
        ):
            self.reload()

    def _get_name_format(self) -> step_lib.NameFormat:
        """Resolve the active ``step_lib.NameFormat`` from ``self.options``.

        Returns:
            step_lib.NameFormat: Step directory name formatter.
        """
        if self.options is not None:
            if self.options.step_name_format is not None:
                return self.options.step_name_format
            return step_lib.standard_name_format(
                single_host_load_and_broadcast=self.options.single_host_load_and_broadcast,
                step_prefix=self.options.step_prefix,
                step_format_fixed_length=self.options.step_format_fixed_length,
            )
        return step_lib.standard_name_format()

    def _step_dir(self, step_num: int, base_dir: PathLike | None = None) -> str:
        """Build the full directory path for ``step_num``.

        Args:
            step_num: Step integer.
            base_dir: Optional override base directory. Defaults to ``self.directory``.

        Returns:
            str: Full step directory path string.
        """
        root = base_dir if base_dir is not None else self.directory
        formatted = self._get_name_format().build_name(step_num)
        return f"{str(root).rstrip('/')}/{formatted}"

    def should_save(self, step: int) -> bool:
        """Determine whether a checkpoint should be saved at the given ``step``.

        Args:
            step: Candidate training step number.

        Returns:
            bool: True if saving should occur at ``step``, False otherwise.
        """
        if self.options is None:
            return True
        if self.options.read_only:
            return False
        if self.options.enable_should_save_is_saving_in_progress_check and self._saving_in_progress:
            return False
        if self.options.save_decision_policy is not None:
            return self.options.save_decision_policy.should_save(step, self.latest)
        if self.options.should_save_fn is not None:
            return self.options.should_save_fn(step, self.latest)
        if self.options.save_on_steps and step in self.options.save_on_steps:
            return True
        return (
            self.options.save_interval_steps > 0 and (step % self.options.save_interval_steps) == 0
        )

    def _record_future(self, maybe_fut: Future | PyTree) -> None:
        """Record an asynchronous ``Future`` returned by a checkpointer.

        Args:
            maybe_fut: Candidate ``Future`` or return value from ``save()``.
        """
        if isinstance(maybe_fut, Future):
            self._pending_futures.append(maybe_fut)
            self._saving_in_progress = True

    def _persist_step_to_disk(
        self,
        step_dir: str,
        items: PyTree,
        args_obj: _CheckpointArgsBase | PyTree,
        save_kwargs: Mapping[str, JsonValue] | None,
    ) -> None:
        """Persist a step's items to ``step_dir`` using configured checkpointers or handlers.

        Args:
            step_dir: Step directory path string.
            items: Positional items passed to ``save()``.
            args_obj: ``args`` parameter passed to ``save()``.
            save_kwargs: Optional ``save_kwargs`` mapping.
        """
        extra_kw: dict[str, PyTree] = dict(save_kwargs) if save_kwargs is not None else {}
        perm_mode = (
            self.options.file_options.path_permission_mode
            if (self.options is not None and self.options.file_options is not None)
            else None
        )
        temp_obj: TemporaryPathProtocol | None = None
        active_step_dir = step_dir
        if self.options is not None and self.options.temporary_path_class is not None:
            temp_obj = _instantiate_temporary_path(
                self.options.temporary_path_class,
                step_dir,
                perm_mode,
            )
            temp_obj.create()
            active_step_dir = str(temp_obj.get())
        elif perm_mode is not None:
            FileSystemIO().makedirs(step_dir, exist_ok=True, permission_mode=perm_mode)

        try:
            if isinstance(args_obj, _CompositeArgsCls) or (
                is_pytree_dict(items)
                and (
                    bool(self.item_names)
                    or any(
                        isinstance(v, (_CheckpointArgsBase, GrainIteratorProtocol))
                        for v in items.values()
                    )
                )
            ):
                comp_handler = CompositeCheckpointHandler(
                    *(self.item_names or ()),
                    handler_registry=self.handler_registry,
                )
                if isinstance(self.item_handlers, Mapping):
                    for k, h in self.item_handlers.items():
                        comp_handler._item_handlers[k] = h
                comp_target = args_obj if isinstance(args_obj, _CompositeArgsCls) else items
                comp_handler.save(active_step_dir, comp_target)
                return

            if isinstance(self.checkpointers, Mapping) and is_pytree_dict(items):
                for item_name, cp in self.checkpointers.items():
                    res = cp.save(
                        f"{active_step_dir}/{item_name}",
                        items.get(item_name),
                        **extra_kw,
                    )
                    self._record_future(res)
                return
            if isinstance(self.checkpointers, AbstractCheckpointer):
                target_val = items if items is not None else args_obj
                res = self.checkpointers.save(active_step_dir, target_val, **extra_kw)
                self._record_future(res)
                return

            if isinstance(self.item_handlers, Mapping) and is_pytree_dict(items):
                for item_name, h in self.item_handlers.items():
                    h.save(f"{active_step_dir}/{item_name}", items.get(item_name), **extra_kw)
                return
            if isinstance(self.item_handlers, CheckpointHandler):
                target_val = items if items is not None else args_obj
                self.item_handlers.save(active_step_dir, target_val, **extra_kw)
                return

            if isinstance(args_obj, _CheckpointArgsBase):
                reg_h_cls = args.get_registered_handler_cls(args_obj)
                if reg_h_cls is not None and issubclass(reg_h_cls, CheckpointHandler):
                    reg_h_cls().save(active_step_dir, args_obj, **extra_kw)
                    return

            payload = (
                items
                if items is not None
                else (args_obj.item if isinstance(args_obj, _CheckpointArgsBase) else args_obj)
            )
            packed = BasePyTreeCheckpointHandler()._pack_pytree(payload)
            encoded = MsgpackEngine().packb(packed)
            AtomicWrite().atomic_write_bytes(
                f"{active_step_dir}/checkpoint.msgpack",
                encoded,
                permission_mode=perm_mode,
            )
        finally:
            if temp_obj is not None:
                temp_obj.finalize()

    def _compute_protected_steps(self, opts: CheckpointManagerOptions) -> set[int]:
        """Compute the set of step numbers exempt from ``max_to_keep`` eviction.

        Args:
            opts: Active ``CheckpointManagerOptions``.

        Returns:
            set[int]: Protected step numbers.
        """
        protected: set[int] = set()
        if opts.preservation_policy is not None:
            preserved = opts.preservation_policy.should_preserve(self.steps)
            protected.update(preserved)
        if opts.keep_period is not None and opts.keep_period > 0:
            for s in self.steps:
                if s % opts.keep_period == 0:
                    protected.add(s)
        if opts.keep_time_interval is not None:
            default_epoch = datetime.datetime.now(tz=datetime.timezone.utc)
            last_kept_time: datetime.datetime | None = None
            for s in self.steps:
                ts = self._step_timestamps.get(s, default_epoch)
                if last_kept_time is None or (ts - last_kept_time) >= opts.keep_time_interval:
                    protected.add(s)
                    last_kept_time = ts
        if opts.should_keep_fn is not None:
            default_now = datetime.datetime.now(tz=datetime.timezone.utc)
            for s in self.steps:
                step_ts = self._step_timestamps.get(s, default_now)
                if opts.should_keep_fn(s, step_ts):
                    protected.add(s)
        if opts.best_fn is not None and opts.max_to_keep is not None:
            best_fn = opts.best_fn
            scored = [
                (best_fn(self._step_metrics[s]), s) for s in self.steps if s in self._step_metrics
            ]
            scored.sort(
                key=lambda pair: pair[0],
                reverse=(opts.best_mode != "min"),
            )
            for _, s in scored[: opts.max_to_keep]:
                protected.add(s)
            if opts.keep_checkpoints_without_metrics:
                for s in self.steps:
                    if s not in self._step_metrics:
                        protected.add(s)
        return protected

    def _remove_step_storage(self, step_num: int) -> None:
        """Delete or move to ``todelete_subdir`` / ``todelete_full_path`` the directory for ``step_num``.

        Args:
            step_num: Step integer to remove from storage.
        """
        if self.directory is None:
            return
        src_dir = self._step_dir(step_num)
        io_handler = FileSystemIO()
        if self.options is not None and (
            self.options.todelete_subdir is not None or self.options.todelete_full_path is not None
        ):
            if self.options.todelete_full_path is not None:
                trash_base = self.options.todelete_full_path.rstrip("/")
            else:
                trash_base = f"{str(self.directory).rstrip('/')}/{self.options.todelete_subdir}"
            formatted = self._get_name_format().build_name(step_num)
            dest_dir = f"{trash_base}/{formatted}"
            io_handler.makedirs(trash_base, exist_ok=True)
            io_handler.move(src_dir, dest_dir)
            return
        io_handler.remove(src_dir)

    def save(
        self,
        step: int,
        items: PyTree = None,
        save_kwargs: Mapping[str, JsonValue] | None = None,
        metrics: Mapping[str, float | int] | None = None,
        force: bool = False,
        args: args.CheckpointArgs | None = None,
        **kwargs: PyTree,
    ) -> bool:
        """Save a checkpoint at ``step`` if retention/interval conditions are satisfied.

        Args:
            step: Training step number at which to save.
            items: PyTree or mapping of items to save. Defaults to None.
            save_kwargs: Optional keyword arguments for the underlying checkpointer. Defaults to None.
            metrics: Optional dictionary of metric values for best-step tracking. Defaults to None.
            force: If True, bypass ``should_save(step)`` interval checks. Defaults to False.
            args: Optional structured ``CheckpointArgs`` payload. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if the checkpoint was saved, False if skipped.

        Raises:
            ValueError: If ``options.read_only`` is True.
        """
        if self.options is not None and self.options.read_only:
            raise ValueError("Cannot save checkpoint in read_only mode.")
        if not force and not self.should_save(step):
            return False

        if isinstance(args, _CompositeArgsCls):
            payload: PyTree = {
                k: (v.item if isinstance(v, _CheckpointArgsBase) else v) for k, v in args.items()
            }
        else:
            payload = (
                items
                if items is not None
                else (args.item if isinstance(args, _CheckpointArgsBase) else args)
            )
        if step not in self.steps:
            self.steps.append(step)
        if self.directory is None:
            self.checkpoints[step] = payload
        else:
            self.checkpoints.pop(step, None)
        if step not in self._step_timestamps:
            self._step_timestamps[step] = datetime.datetime.now(tz=datetime.timezone.utc)
        if metrics is not None:
            self._step_metrics[step] = dict(metrics)
        self.latest = max(self.steps)

        if self.directory is not None:
            step_dir = self._step_dir(step)
            self._persist_step_to_disk(step_dir, items, args, save_kwargs)
            if metrics is not None and (
                self.options is None or not self.options.prevent_write_metrics
            ):
                AtomicWrite().atomic_write_bytes(
                    f"{step_dir}/metrics.json",
                    json.dumps(dict(metrics)).encode("utf-8"),
                )

        if self.options is not None and not self.options.enable_async_checkpointing:
            self.wait_until_finished()

        if self.logger is not None:
            self.logger.log_entry(f"Saved checkpoint at step {step}", step=step)

        if self.options is not None and self.options.max_to_keep is not None:
            protected = self._compute_protected_steps(self.options)
            unprotected_candidates = [s for s in self.steps if s not in protected]
            while len(self.steps) > self.options.max_to_keep and unprotected_candidates:
                to_remove = unprotected_candidates.pop(0)
                self.steps.remove(to_remove)
                self.checkpoints.pop(to_remove, None)
                self._step_metrics.pop(to_remove, None)
                self._step_timestamps.pop(to_remove, None)
                self._remove_step_storage(to_remove)
            self.latest = max(self.steps) if self.steps else None
        return True

    def _restore_step_from_disk(
        self,
        step_dir: str,
        items: PyTree,
        args_obj: _CheckpointArgsBase | PyTree,
        restore_kwargs: Mapping[str, JsonValue] | None,
    ) -> PyTree:
        """Restore a step from ``step_dir`` using configured checkpointers or handlers.

        Args:
            step_dir: Step directory path string.
            items: Optional target structure.
            args_obj: Optional ``CheckpointArgs`` for restoration.
            restore_kwargs: Optional ``restore_kwargs`` mapping.

        Returns:
            PyTree: Restored step payload, or None if not found on disk.
        """
        extra_kw: dict[str, PyTree] = dict(restore_kwargs) if restore_kwargs is not None else {}
        if isinstance(args_obj, _CompositeArgsCls):
            comp_handler = CompositeCheckpointHandler(
                *(self.item_names or ()),
                handler_registry=self.handler_registry,
            )
            if isinstance(self.item_handlers, Mapping):
                for k, h in self.item_handlers.items():
                    comp_handler._item_handlers[k] = h
            return comp_handler.restore(step_dir, args_obj)

        if isinstance(self.checkpointers, Mapping):
            result_map: dict[str, PyTree] = {}
            for item_name, cp in self.checkpointers.items():
                tmpl = items.get(item_name) if is_pytree_dict(items) else None
                restored_sub = cp.restore(f"{step_dir}/{item_name}", tmpl, **extra_kw)
                result_map[item_name] = (
                    restored_sub.result() if isinstance(restored_sub, Future) else restored_sub
                )
            return result_map
        if isinstance(self.checkpointers, AbstractCheckpointer):
            target_val = items if items is not None else args_obj
            restored_cp = self.checkpointers.restore(step_dir, target_val, **extra_kw)
            return restored_cp.result() if isinstance(restored_cp, Future) else restored_cp

        if isinstance(self.item_handlers, Mapping):
            result_h_map: dict[str, PyTree] = {}
            for item_name, h in self.item_handlers.items():
                tmpl = items.get(item_name) if is_pytree_dict(items) else None
                result_h_map[item_name] = h.restore(f"{step_dir}/{item_name}", tmpl, **extra_kw)
            return result_h_map
        if isinstance(self.item_handlers, CheckpointHandler):
            target_val = items if items is not None else args_obj
            return self.item_handlers.restore(step_dir, target_val, **extra_kw)

        if isinstance(args_obj, _CheckpointArgsBase):
            reg_h_cls = args.get_registered_handler_cls(args_obj)
            if reg_h_cls is not None and issubclass(reg_h_cls, CheckpointHandler):
                return reg_h_cls().restore(step_dir, args_obj, **extra_kw)

        raw = FileSystemIO().read(f"{step_dir}/checkpoint.msgpack")
        if raw:
            return BasePyTreeCheckpointHandler()._unpack_pytree(
                MsgpackEngine().unpackb(raw),
                items,
            )
        discovered_subdirs = [
            entry
            for entry in FileSystemIO().listdir(step_dir)
            if not entry.startswith("_") and entry != "metrics.json"
        ]
        if discovered_subdirs:
            comp_handler = CompositeCheckpointHandler(
                *(self.item_names or ()),
                handler_registry=self.handler_registry,
            )
            return comp_handler.restore(step_dir, items)
        return None

    def restore(
        self,
        step: int | None = None,
        items: PyTree = None,
        restore_kwargs: Mapping[str, JsonValue] | None = None,
        directory: PathLike | None = None,
        args: args.CheckpointArgs | None = None,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore a checkpoint for ``step`` (or the latest step if ``step is None``).

        Args:
            step: Step number to restore. Defaults to ``latest_step()`` when None.
            items: Optional target structure to restore into. Defaults to None.
            restore_kwargs: Optional keyword arguments for restoration. Defaults to None.
            directory: Optional override base directory. Defaults to None.
            args: Optional structured ``CheckpointArgs`` for restoration. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            PyTree: The restored checkpoint payload.

        Raises:
            ValueError: If no checkpoint exists for the requested step.
        """
        target_step = self.latest if step is None else step
        base_dir = directory if directory is not None else self.directory
        if target_step is None:
            raise ValueError(f"Checkpoint for step {step} not found.")
        if base_dir is not None:
            step_dir = self._step_dir(target_step, base_dir=base_dir)
            if target_step not in self.checkpoints and not FileSystemIO().exists(step_dir):
                raise ValueError(f"Checkpoint for step {step} not found.")
            from_disk = self._restore_step_from_disk(step_dir, items, args, restore_kwargs)
            if from_disk is not None:
                return from_disk
        elif target_step not in self.checkpoints:
            raise ValueError(f"Checkpoint for step {step} not found.")
        stored = self.checkpoints.get(target_step)
        return stored if stored is not None else items

    def latest_step(self) -> int | None:
        """Return the most recently saved step number.

        Returns:
            int | None: Most recent step number, or ``None`` if no checkpoints exist.
        """
        return self.latest

    def all_steps(self, read: bool = False) -> Sequence[int]:
        """Return a sequence of all currently available checkpoint step numbers.

        Args:
            read: If True, reload step directories from disk first. Defaults to False.

        Returns:
            Sequence[int]: List of available step numbers.
        """
        if read:
            self.reload()
        return list(self.steps)

    def best_step(self) -> int | None:
        """Return the step with the best metric according to ``options.best_fn`` and ``best_mode``.

        Returns:
            int | None: Best step number, or ``latest_step()`` if no metrics/best_fn are configured.
        """
        if not self.steps:
            return None
        if self.options is not None and self.options.best_fn is not None:
            best_fn = self.options.best_fn
            scored_steps = [
                (best_fn(self._step_metrics[s]), s) for s in self.steps if s in self._step_metrics
            ]
            if scored_steps:
                if self.options.best_mode == "min":
                    return min(scored_steps, key=lambda pair: pair[0])[1]
                return max(scored_steps, key=lambda pair: pair[0])[1]
        return self.latest

    def delete(self, step: int) -> None:
        """Delete the checkpoint at ``step`` from memory and storage.

        Args:
            step: Step number to delete.
        """
        if step in self.steps:
            self.steps.remove(step)
            self.checkpoints.pop(step, None)
            self._step_metrics.pop(step, None)
            self._step_timestamps.pop(step, None)
            self.latest = max(self.steps) if self.steps else None
            self._remove_step_storage(step)

    def check_for_errors(self) -> None:
        """Check for any asynchronous background errors across pending futures and checkpointers."""
        for fut in self._pending_futures:
            fut.result()
        if isinstance(self.checkpointers, AsyncCheckpointer):
            self.checkpointers.check_for_errors()
        elif isinstance(self.checkpointers, Mapping):
            for cp in self.checkpointers.values():
                if isinstance(cp, AsyncCheckpointer):
                    cp.check_for_errors()

    def wait_until_finished(self) -> None:
        """Block until any in-flight background operations complete."""
        for fut in self._pending_futures:
            fut.result()
        self._pending_futures.clear()
        if isinstance(self.checkpointers, AsyncCheckpointer):
            self.checkpointers.wait_until_finished()
        elif isinstance(self.checkpointers, Mapping):
            for cp in self.checkpointers.values():
                if isinstance(cp, AsyncCheckpointer):
                    cp.wait_until_finished()
        self._saving_in_progress = False

    def is_saving_in_progress(self) -> bool:
        """Return whether an asynchronous save operation is currently in progress.

        Returns:
            bool: True if saving is in progress, False otherwise.
        """
        return self._saving_in_progress

    def metrics(self, step: int | None = None) -> Mapping[str, float | int] | None:
        """Return recorded metrics for ``step`` (or ``latest_step()`` when ``step is None``).

        Args:
            step: Step number to query. Defaults to None.

        Returns:
            Mapping[str, float | int] | None: Metrics dictionary for the step, or ``None``.
        """
        target = self.latest if step is None else step
        if target is None:
            return None
        return self._step_metrics.get(target)

    def metadata(self, step: int | None = None) -> Mapping[str, JsonValue] | int | None:
        """Return the configured metadata for this checkpoint manager or for a specific ``step``.

        Args:
            step: Optional step number. Defaults to None.

        Returns:
            Mapping[str, JsonValue] | int | None: Step metadata if ``step`` is provided and present on disk, otherwise root metadata.
        """
        if step is not None and self.directory is not None:
            step_dir = self._step_dir(step)
            raw_meta = FileSystemIO().read(f"{step_dir}/_orbax_metadata.json")
            if raw_meta:
                loaded_meta: dict[str, JsonValue] = json.loads(raw_meta.decode("utf-8"))
                return loaded_meta
        return self._metadata

    def item_metadata(self, step: int | None = None) -> PyTree:
        """Return the stored item structure/metadata for ``step``.

        Args:
            step: Optional step number (defaults to latest step). Defaults to None.

        Returns:
            PyTree: Stored item structure for the step, or ``None``.
        """
        target = self.latest if step is None else step
        if target is None:
            return None
        if target in self.checkpoints:
            return self.checkpoints[target]
        if self.directory is not None and target in self.steps:
            step_dir = self._step_dir(target)
            return self._restore_step_from_disk(step_dir, None, None, None)
        return None

    def reload(self) -> None:
        """Rescan the checkpoint base directory to discover saved step subdirectories and metadata."""
        if self.directory is None:
            return
        io_handler = FileSystemIO()
        base_str = str(self.directory).rstrip("/")
        raw_root = io_handler.read(f"{base_str}/_root_metadata.json")
        if self._metadata is None and raw_root:
            parsed_root: dict[str, dict[str, JsonValue] | int] = json.loads(
                raw_root.decode("utf-8")
            )
            self._metadata = parsed_root.get("metadata")

        name_format = self._get_name_format()
        for entry in io_handler.listdir(self.directory):
            step_num = name_format.parse_step(entry)
            if step_num is not None and step_num not in self.steps:
                step_dir = f"{base_str}/{entry}"
                self.steps.append(step_num)
                self._step_timestamps.setdefault(
                    step_num,
                    datetime.datetime.now(tz=datetime.timezone.utc),
                )
                raw_m = io_handler.read(f"{step_dir}/metrics.json")
                if raw_m:
                    loaded_m: dict[str, float | int] = json.loads(raw_m.decode("utf-8"))
                    self._step_metrics[step_num] = {
                        str(mk): float(mv) if isinstance(mv, float) else int(mv)
                        for mk, mv in loaded_m.items()
                    }
        self.steps.sort()
        self.latest = max(self.steps) if self.steps else None

    def close(self) -> None:
        """Close the manager and any underlying checkpointers or item handlers."""
        super().close()
        if isinstance(self.checkpointers, Mapping):
            for cp in self.checkpointers.values():
                cp.close()
        elif isinstance(self.checkpointers, AbstractCheckpointer):
            self.checkpointers.close()
        if isinstance(self.item_handlers, Mapping):
            for h in self.item_handlers.values():
                h.close()
        elif isinstance(self.item_handlers, CheckpointHandler):
            self.item_handlers.close()

    def __enter__(self) -> CheckpointManager:
        """Enter the runtime context for the ``CheckpointManager``.

        Returns:
            CheckpointManager: Self.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the runtime context, waiting for completion and closing the manager.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception instance if raised.
            exc_tb: Exception traceback if raised.
        """
        self.wait_until_finished()
        self.close()


class abstract_checkpoint_manager:
    """Namespace exporting ``AbstractCheckpointManager``."""

    AbstractCheckpointManager = AbstractCheckpointManager


class checkpoint_manager:
    """Namespace exporting ``CheckpointManager`` and ``CheckpointManagerOptions``."""

    CheckpointManager = CheckpointManager
    CheckpointManagerOptions = CheckpointManagerOptions


__all__ = [
    "AbstractCheckpointManager",
    "CheckpointHandlersDict",
    "CheckpointManager",
    "CheckpointManagerOptions",
    "CheckpointersDict",
    "abstract_checkpoint_manager",
    "checkpoint_manager",
]
