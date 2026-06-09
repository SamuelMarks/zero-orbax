"""Orbax checkpointing API."""

from __future__ import annotations

import datetime
from typing import (
    Any,
    Callable,
    Container,
    Optional,
    Sequence,
)


class async_checkpoint_handler:
    """Type hint wrapper."""

    AsyncCheckpointHandler = Any


class orbax:
    """Type hint wrapper."""

    class checkpoint:
        """Type hint wrapper."""

        class options:
            """Type hint wrapper."""

            MultiprocessingOptions = Any
            AsyncOptions = Any
            FileOptions = Any


class atomicity:
    """Type hint wrapper."""

    TemporaryPath = Any


class checkpoint:
    """Type hint wrapper."""

    CheckpointMetadataStore = Any


class multihost:
    """Type hint wrapper."""

    BarrierSyncFn = Any


class epath:
    """Type hint wrapper."""

    PathLike = Any


class abstract_logger:
    """Type hint wrapper."""

    AbstractLogger = Any


class step_lib:
    """Type hint wrapper."""

    Metadata = Any
    NameFormat = Any


CheckpointHandler = Any


class checkpoint_handler:
    """Type hint wrapper."""

    CheckpointHandler = Any


MultiprocessingOptions = Any
FileOptions = Any
CheckpointersDict = Any
CheckpointHandlerRegistry = Any
CheckpointHandlersDict = Any
PyTree = Any
RestoreArgs = Any
NoneType = type(None)
ValueFn = Any
MultiValueFn = Any


class AbstractCheckpointManager:
    """Interface to manage checkpoints."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the manager."""
        pass

    def save(self, step: int, items: Any, **kwargs: Any) -> bool:
        """Saves a checkpoint."""
        raise NotImplementedError

    def restore(self, step: int, items: Any | None = None, **kwargs: Any) -> Any:
        """Restores a checkpoint."""
        raise NotImplementedError

    def latest_step(self) -> int | None:
        """Returns the latest checkpoint step."""
        raise NotImplementedError

    def all_steps(self) -> Sequence[int]:
        """Returns all checkpoint steps."""
        raise NotImplementedError


class CheckpointManager(AbstractCheckpointManager):
    """A generic, synchronous AbstractCheckpointManager implementation."""

    def __init__(
        self,
        directory: epath.PathLike,
        checkpointers: AbstractCheckpointer | CheckpointersDict | None = None,
        options: CheckpointManagerOptions | None = None,
        metadata: int = None,
        item_names: Sequence[str] | None = None,
        item_handlers: CheckpointHandler | CheckpointHandlersDict | None = None,
        logger: abstract_logger.AbstractLogger | None = None,
        handler_registry: CheckpointHandlerRegistry | None = None,
    ) -> None:
        """Initializes the CheckpointManager."""
        self.directory = directory
        self.checkpointers = checkpointers
        self.options = options or CheckpointManagerOptions()
        self.metadata = metadata
        self.item_names = item_names
        self.item_handlers = item_handlers
        self.logger = logger
        self.handler_registry = handler_registry

        self._checkpoints: dict[int, Any] = {}

    def save(self, step: int, items: Any, **kwargs: Any) -> bool:
        """Saves a checkpoint at the given step."""
        if self.options.read_only:
            raise ValueError("Cannot save checkpoint in read_only mode.")

        if self.options.should_save_fn and not self.options.should_save_fn(
            step, self.latest_step()
        ):
            return False

        if (
            self.options.save_on_steps is not None
            and step in self.options.save_on_steps
        ):
            pass  # force save
        elif step % self.options.save_interval_steps != 0:
            return False

        # Dummy serialization logic for zero-orbax testing
        self._checkpoints[step] = {"items": items, "metadata": self.metadata}

        # Directory rotation logic
        if self.options.max_to_keep is not None:
            sorted_steps = sorted(self._checkpoints.keys())
            while len(sorted_steps) > self.options.max_to_keep:
                step_to_delete = sorted_steps.pop(0)
                del self._checkpoints[step_to_delete]

        return True

    def restore(self, step: int, items: Any | None = None, **kwargs: Any) -> Any:
        """Restores a checkpoint from the given step."""
        if step not in self._checkpoints:
            raise ValueError(f"Checkpoint for step {step} not found.")
        return self._checkpoints[step]["items"]

    def latest_step(self) -> int | None:
        """Returns the latest checkpoint step."""
        if not self._checkpoints:
            return None
        return max(self._checkpoints.keys())

    def all_steps(self) -> Sequence[int]:
        """Returns all checkpoint steps."""
        return sorted(self._checkpoints.keys())


class AbstractCheckpointer:
    """An interface allowing atomic save and restore for a single object."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the checkpointer."""
        pass

    def save(self, path: epath.PathLike, item: Any, *args: Any, **kwargs: Any) -> None:
        """Saves an item to the given path."""
        raise NotImplementedError

    def restore(
        self, path: epath.PathLike, item: Any | None = None, *args: Any, **kwargs: Any
    ) -> Any:
        """Restores an item from the given path."""
        raise NotImplementedError


class AsyncCheckpointer(AbstractCheckpointer):
    """An asynchronous implementation of Checkpointer."""

    def __init__(
        self,
        _handler: async_checkpoint_handler.AsyncCheckpointHandler = None,
        *,
        multiprocessing_options: MultiprocessingOptions = "```(options_lib.MultiprocessingOptions())```",
        timeout_secs: int | None = None,
        handler: async_checkpoint_handler.AsyncCheckpointHandler = None,
        temporary_path_class: type[atomicity.TemporaryPath] | None = None,
        async_options: orbax.checkpoint.options.AsyncOptions = "```(options_lib.AsyncOptions())```",
        file_options: FileOptions = "```(options_lib.FileOptions())```",
        checkpoint_metadata_store: checkpoint.CheckpointMetadataStore | None = None,
    ) -> None:
        """Initializes the AsyncCheckpointer."""
        self.handler = handler or _handler
        self.multiprocessing_options = multiprocessing_options
        self.timeout_secs = timeout_secs
        self.temporary_path_class = temporary_path_class
        self.async_options = async_options
        self.file_options = file_options
        self.checkpoint_metadata_store = checkpoint_metadata_store

    def save(
        self, path: epath.PathLike, item: Any, *args: Any, **kwargs: Any
    ) -> Future:
        """Saves an item asynchronously."""
        if hasattr(self.handler, "save"):
            self.handler.save(path, item, *args, **kwargs)
        return Future(result=None)

    def restore(
        self, path: epath.PathLike, item: Any | None = None, *args: Any, **kwargs: Any
    ) -> Future:
        """Restores an item asynchronously."""
        res = None
        if hasattr(self.handler, "restore"):
            res = self.handler.restore(path, item, *args, **kwargs)
        return Future(result=res)


class AsyncOptions:
    """Options used to configure async behavior."""

    def __init__(
        self,
        timeout_secs: int = 300,
        barrier_sync_fn: multihost.BarrierSyncFn | None = None,
        post_finalization_callback: Callable[[], None] | None = None,
    ) -> None:
        """Initializes the AsyncOptions."""
        self.timeout_secs = timeout_secs
        self.barrier_sync_fn = barrier_sync_fn
        self.post_finalization_callback = post_finalization_callback


class CheckpointManagerOptions:
    """Optional arguments for CheckpointManager."""

    def __init__(
        self,
        save_interval_steps: int = 1,
        max_to_keep: int | None = None,
        keep_time_interval: datetime.timedelta | None = None,
        keep_period: int | None = None,
        best_fn: Callable[[PyTree], float] | None = None,
        best_mode: str = "max",
        keep_checkpoints_without_metrics: bool = True,
        step_prefix: str | None = None,
        step_format_fixed_length: int | None = None,
        step_name_format: step_lib.NameFormat[step_lib.Metadata] | None = None,
        create: bool = True,
        cleanup_tmp_directories: bool = False,
        save_on_steps: Container[int] | None = None,
        single_host_load_and_broadcast: bool = False,
        todelete_subdir: str | None = None,
        enable_background_delete: bool = False,
        read_only: bool = False,
        enable_async_checkpointing: bool = True,
        async_options: AsyncOptions | None = None,
        multiprocessing_options: MultiprocessingOptions = "dataclasses.field(default_factory=MultiprocessingOptions)",
        should_save_fn: Callable[[int, int | None], bool] | None = None,
        file_options: FileOptions = "dataclasses.field(default_factory=FileOptions)",
        temporary_path_class: type[atomicity.TemporaryPath] | None = None,
    ) -> None:
        """Initializes the CheckpointManagerOptions."""
        if save_interval_steps < 1:
            raise ValueError("save_interval_steps must be positive")
        if max_to_keep is not None and max_to_keep <= 0:
            raise ValueError("max_to_keep must be positive")

        self.save_interval_steps = save_interval_steps
        self.max_to_keep = max_to_keep
        self.keep_time_interval = keep_time_interval
        self.keep_period = keep_period
        self.best_fn = best_fn
        self.best_mode = best_mode
        self.keep_checkpoints_without_metrics = keep_checkpoints_without_metrics
        self.step_prefix = step_prefix
        self.step_format_fixed_length = step_format_fixed_length
        self.step_name_format = step_name_format
        self.create = create
        self.cleanup_tmp_directories = cleanup_tmp_directories
        self.save_on_steps = save_on_steps
        self.single_host_load_and_broadcast = single_host_load_and_broadcast
        self.todelete_subdir = todelete_subdir
        self.enable_background_delete = enable_background_delete
        self.read_only = read_only
        self.enable_async_checkpointing = enable_async_checkpointing
        self.async_options = async_options
        self.multiprocessing_options = multiprocessing_options
        self.should_save_fn = should_save_fn
        self.file_options = file_options
        self.temporary_path_class = temporary_path_class

        pass


class Checkpointer(AbstractCheckpointer):
    """A synchronous implementation of AbstractCheckpointer."""

    def __init__(
        self,
        handler: checkpoint_handler.CheckpointHandler,
        *,
        multiprocessing_options: MultiprocessingOptions = "```(options_lib.MultiprocessingOptions())```",
        file_options: FileOptions = "```(options_lib.FileOptions())```",
        checkpoint_metadata_store: checkpoint.CheckpointMetadataStore | None = None,
        temporary_path_class: type[atomicity.TemporaryPath] | None = None,
    ) -> None:
        """Initializes the Checkpointer."""
        self.handler = handler
        self.multiprocessing_options = multiprocessing_options
        self.file_options = file_options
        self.checkpoint_metadata_store = checkpoint_metadata_store
        self.temporary_path_class = temporary_path_class

    def save(self, path: epath.PathLike, item: Any, *args: Any, **kwargs: Any) -> None:
        """Saves an item using the handler."""
        if hasattr(self.handler, "save"):
            self.handler.save(path, item, *args, **kwargs)

    def restore(
        self, path: epath.PathLike, item: Any | None = None, *args: Any, **kwargs: Any
    ) -> Any:
        """Restores an item using the handler."""
        if hasattr(self.handler, "restore"):
            return self.handler.restore(path, item, *args, **kwargs)
        return None


class Future:
    """Abstracted Orbax Future class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializes the future."""
        self._result = kwargs.get("result")

    def result(self) -> Any:
        """Returns the result of the future."""
        return self._result


class PyTreeCheckpointer(AbstractCheckpointer):
    """Shorthand class."""

    def __init__(
        self,
        primary_host: Optional[int] = 0,
        use_ocdbt: bool = True,
        use_zarr3: bool = False,
    ) -> None:
        """Initializes the PyTreeCheckpointer."""
        self.primary_host = primary_host
        self.use_ocdbt = use_ocdbt
        self.use_zarr3 = use_zarr3

    def save(self, path: epath.PathLike, item: Any, *args: Any, **kwargs: Any) -> None:
        """Saves a PyTree."""
        pass

    def restore(
        self, path: epath.PathLike, item: Any | None = None, *args: Any, **kwargs: Any
    ) -> Any:
        """Restores a PyTree."""
        return item


class RestoreTransform:
    """Transform subclass used only during restoration from checkpoint."""

    def __init__(
        self,
        value_fn: Callable[[Any, RestoreArgs], Any] | None = None,
        multi_value_fn: Callable[[str, PyTree, RestoreArgs], Any] | None = None,
        multi_value_fn_input_args: dict[str, Any] | None = None,
        original_key: str | tuple[str] | NoneType = None,
        use_fallback: bool = False,
    ) -> None:
        """Initializes the RestoreTransform."""
        self.value_fn = value_fn
        self.multi_value_fn = multi_value_fn
        self.multi_value_fn_input_args = multi_value_fn_input_args
        self.original_key = original_key
        self.use_fallback = use_fallback


class StandardCheckpointer(AbstractCheckpointer):
    """Shorthand class."""

    def __init__(
        self,
        *,
        async_options: orbax.checkpoint.options.AsyncOptions = "```(options_lib.AsyncOptions())```",
        multiprocessing_options: MultiprocessingOptions = "```(options_lib.MultiprocessingOptions())```",
        file_options: FileOptions = "```(options_lib.FileOptions())```",
        checkpoint_metadata_store: checkpoint.CheckpointMetadataStore | None = None,
        temporary_path_class: type[atomicity.TemporaryPath] | None = None,
        **kwargs: dict | None,
    ) -> None:
        """Initializes the StandardCheckpointer."""
        self.async_options = async_options
        self.multiprocessing_options = multiprocessing_options
        self.file_options = file_options
        self.checkpoint_metadata_store = checkpoint_metadata_store
        self.temporary_path_class = temporary_path_class
        self.kwargs = kwargs

    def save(self, path: epath.PathLike, item: Any, *args: Any, **kwargs: Any) -> None:
        """Saves an item."""
        pass

    def restore(
        self, path: epath.PathLike, item: Any | None = None, *args: Any, **kwargs: Any
    ) -> Any:
        """Restores an item."""
        return item


class Transform:
    """A representation of a transformation applied to pytree keys/values."""

    def __init__(
        self,
        original_key: str | tuple[str] | None = None,
        use_fallback: bool = False,
        value_fn: ValueFn | None = None,
        multi_value_fn: MultiValueFn | None = None,
    ) -> None:
        """Initializes the Transform."""
        self.original_key = original_key
        self.use_fallback = use_fallback
        self.value_fn = value_fn
        self.multi_value_fn = multi_value_fn


def apply_transformations(
    original_tree: PyTree,
    transformations: PyTree,
    new_tree: PyTree,
    default_to_original: bool | None = True,
) -> Any:
    """Applies transformations to a pytree."""
    if not isinstance(new_tree, dict):
        return new_tree

    result: dict[str, Any] = {}
    for key, new_value in new_tree.items():
        if isinstance(transformations, dict) and key in transformations:
            trans = transformations[key]

            if isinstance(trans, dict) and isinstance(new_value, dict):
                orig_sub = (
                    original_tree.get(key, {})
                    if isinstance(original_tree, dict)
                    else {}
                )
                result[key] = apply_transformations(
                    orig_sub, trans, new_value, default_to_original
                )
                continue

            if isinstance(trans, (Transform, RestoreTransform)):
                orig_key = getattr(trans, "original_key", None) or key
                val = None

                if getattr(trans, "multi_value_fn", None):
                    val = trans.multi_value_fn(original_tree)
                else:
                    if isinstance(orig_key, tuple):
                        val = original_tree
                        for k in orig_key:
                            val = val.get(k) if isinstance(val, dict) else None
                    else:
                        val = (
                            original_tree.get(orig_key)
                            if isinstance(original_tree, dict)
                            else None
                        )

                    if getattr(trans, "value_fn", None):
                        val = trans.value_fn(val)
                result[key] = val
            else:
                result[key] = trans
        else:
            if isinstance(new_value, dict):
                orig_sub = (
                    original_tree.get(key, {})
                    if isinstance(original_tree, dict)
                    else {}
                )
                result[key] = apply_transformations(
                    orig_sub, {}, new_value, default_to_original
                )
            else:
                if (
                    default_to_original
                    and isinstance(original_tree, dict)
                    and key in original_tree
                ):
                    result[key] = original_tree[key]
                else:
                    result[key] = new_value

    return result


def merge_trees(*trees: tuple, target: dict = "```(None)```") -> Any:
    """Merges the provided PyTrees into a single result."""
    merged = dict(target) if target is not None and target != "```(None)```" else {}
    for tree in trees:
        if not isinstance(tree, dict):
            raise TypeError("merge_trees currently only supports dictionary PyTrees.")
        for k, v in tree.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = merge_trees(merged[k], v)
            else:
                merged[k] = v
    return merged
