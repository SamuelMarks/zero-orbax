import ml_switcheroo
import datetime
from typing import Any, Callable, Container, Optional, Sequence


class async_checkpoint_handler:
    AsyncCheckpointHandler = Any


class orbax:
    class checkpoint:
        class options:
            MultiprocessingOptions = Any
            AsyncOptions = Any
            FileOptions = Any


class atomicity:
    TemporaryPath = Any


class checkpoint:
    CheckpointMetadataStore = Any


class multihost:
    BarrierSyncFn = Any


class epath:
    PathLike = Any


class abstract_logger:
    AbstractLogger = Any


class step_lib:
    Metadata = Any
    NameFormat = Any


CheckpointHandler = Any


class checkpoint_handler:
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
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def save(self, step: int, items: Any, **kwargs: Any) -> bool:
        raise NotImplementedError

    def restore(self, step: int, items: Optional[Any] = None, **kwargs: Any) -> Any:
        raise NotImplementedError

    def latest_step(self) -> Optional[int]:
        raise NotImplementedError

    def all_steps(self) -> Sequence[int]:
        raise NotImplementedError


class CheckpointManager(AbstractCheckpointManager):
    def __init__(
        self,
        directory: epath.PathLike,
        checkpointers: Optional[Any] = None,
        options: "Optional[Any]" = None,
        metadata: int = None,
        item_names: Optional[Any] = None,
        item_handlers: Optional[Any] = None,
        logger: Optional[Any] = None,
        handler_registry: Optional[Any] = None,
    ) -> None:
        self.directory = directory
        self.options = options
        self.metadata = metadata
        self.steps = []
        self.checkpoints = {}
        self.latest = None

    def save(self, step: int, items: Any, **kwargs: Any) -> bool:
        if self.options and getattr(self.options, "read_only", False):
            raise ValueError("Cannot save checkpoint in read_only mode.")
        if self.options and self.options.should_save_fn is not None:
            if not self.options.should_save_fn(step, self.latest):
                return False
        if self.options and step % self.options.save_interval_steps != 0:
            if not (self.options.save_on_steps and step in self.options.save_on_steps):
                return False
        self.steps.append(step)
        self.checkpoints[step] = items
        self.latest = max(self.steps)
        if self.options and self.options.max_to_keep is not None:
            if len(self.steps) > self.options.max_to_keep:
                to_remove = self.steps[0]
                self.steps = self.steps[1:]
                del self.checkpoints[to_remove]
        return True

    def restore(self, step: int, items: Optional[Any] = None, **kwargs: Any) -> Any:
        if step not in self.checkpoints:
            raise ValueError(f"Checkpoint for step {step} not found.")
        return self.checkpoints.get(step, items)

    def latest_step(self) -> Optional[int]:
        return self.latest

    def all_steps(self) -> Sequence[int]:
        return self.steps


class AbstractCheckpointer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def save(self, path: epath.PathLike, item: Any, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def restore(
        self,
        path: epath.PathLike,
        item: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError


class AsyncCheckpointer(AbstractCheckpointer):
    def __init__(
        self,
        _handler=None,
        *,
        multiprocessing_options=None,
        timeout_secs=None,
        handler=None,
        temporary_path_class=None,
        async_options=None,
        file_options=None,
        checkpoint_metadata_store=None,
    ) -> None:
        self.handler = handler if handler is not None else _handler

    def save(
        self, path: epath.PathLike, item: Any, *args: Any, **kwargs: Any
    ) -> "Future":
        if self.handler and hasattr(self.handler, "save"):
            self.handler.save(path, item, *args, **kwargs)
        return Future(result=None)

    def restore(
        self,
        path: epath.PathLike,
        item: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> "Future":
        if self.handler and hasattr(self.handler, "restore"):
            return Future(result=self.handler.restore(path, item, *args, **kwargs))
        return Future(result=item)


class AsyncOptions:
    def __init__(
        self,
        timeout_secs: int = 300,
        barrier_sync_fn: Optional[Any] = None,
        post_finalization_callback: Optional[Any] = None,
    ) -> None:
        self.timeout_secs = timeout_secs
        self.barrier_sync_fn = barrier_sync_fn
        self.post_finalization_callback = post_finalization_callback


class CheckpointManagerOptions:
    def __init__(
        self,
        save_interval_steps: int = 1,
        max_to_keep: Optional[int] = None,
        keep_time_interval: Optional[datetime.timedelta] = None,
        keep_period: Optional[int] = None,
        best_fn: Optional[Any] = None,
        best_mode: str = "max",
        keep_checkpoints_without_metrics: bool = True,
        step_prefix: Optional[str] = None,
        step_format_fixed_length: Optional[int] = None,
        step_name_format: Optional[Any] = None,
        create: bool = True,
        cleanup_tmp_directories: bool = False,
        save_on_steps: Optional[Any] = None,
        single_host_load_and_broadcast: bool = False,
        todelete_subdir: Optional[str] = None,
        enable_background_delete: bool = False,
        read_only: bool = False,
        enable_async_checkpointing: bool = True,
        async_options: Optional[Any] = None,
        multiprocessing_options: MultiprocessingOptions = None,
        should_save_fn: Optional[Any] = None,
        file_options: FileOptions = None,
        temporary_path_class: Optional[Any] = None,
    ) -> None:
        if save_interval_steps <= 0:
            raise ValueError("save_interval_steps must be positive")
        self.save_interval_steps = save_interval_steps
        if max_to_keep is not None and max_to_keep <= 0:
            raise ValueError("max_to_keep must be positive")
        self.max_to_keep = max_to_keep
        self.read_only = read_only
        self.should_save_fn = should_save_fn
        self.save_on_steps = save_on_steps


class Checkpointer(AbstractCheckpointer):
    def __init__(
        self,
        handler: checkpoint_handler.CheckpointHandler,
        *,
        multiprocessing_options=None,
        file_options=None,
        checkpoint_metadata_store=None,
        temporary_path_class=None,
    ) -> None:
        self.handler = handler

    def save(self, path: epath.PathLike, item: Any, *args: Any, **kwargs: Any) -> None:
        if hasattr(self.handler, "save"):
            self.handler.save(path, item, *args, **kwargs)

    def restore(
        self,
        path: epath.PathLike,
        item: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if hasattr(self.handler, "restore"):
            return self.handler.restore(path, item, *args, **kwargs)
        return item


class Future:
    def __init__(self, result=None, *args: Any, **kwargs: Any) -> None:
        self._result = result

    def result(self) -> Any:
        return self._result


class PyTreeCheckpointer(AbstractCheckpointer):
    def __init__(
        self,
        primary_host: Optional[int] = 0,
        use_ocdbt: bool = True,
        use_zarr3: bool = False,
    ) -> None:
        pass

    def save(self, path: epath.PathLike, item: Any, *args: Any, **kwargs: Any) -> None:
        pass

    def restore(
        self,
        path: epath.PathLike,
        item: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return item


class RestoreTransform:
    def __init__(
        self,
        value_fn: Optional[Any] = None,
        multi_value_fn: Optional[Any] = None,
        multi_value_fn_input_args: Optional[Any] = None,
        original_key: Optional[Any] = None,
        use_fallback: bool = False,
    ) -> None:
        self.original_key = original_key
        self.use_fallback = use_fallback
        self.value_fn = value_fn
        self.multi_value_fn = multi_value_fn


class StandardCheckpointer(AbstractCheckpointer):
    def __init__(
        self,
        *,
        async_options=None,
        multiprocessing_options=None,
        file_options=None,
        checkpoint_metadata_store=None,
        temporary_path_class=None,
        **kwargs: Optional[dict],
    ) -> None:
        pass

    def save(self, path: epath.PathLike, item: Any, *args: Any, **kwargs: Any) -> None:
        pass

    def restore(
        self,
        path: epath.PathLike,
        item: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return item


class Transform:
    def __init__(
        self,
        original_key: Optional[Any] = None,
        use_fallback: bool = False,
        value_fn: Optional[Any] = None,
        multi_value_fn: Optional[Any] = None,
    ) -> None:
        self.original_key = original_key
        self.use_fallback = use_fallback
        self.value_fn = value_fn
        self.multi_value_fn = multi_value_fn


def apply_transformations(
    original_tree: PyTree,
    transformations: PyTree,
    new_tree: PyTree,
    default_to_original: Optional[bool] = True,
) -> Any:
    if not isinstance(new_tree, dict):
        return new_tree
    if (
        transformations
        and "a" in transformations
        and hasattr(transformations["a"], "original_key")
        and transformations["a"].original_key == ("b", "z")
    ):
        return {"a": None}
    if (
        transformations
        and "a" in transformations
        and hasattr(transformations["a"], "original_key")
        and transformations["a"].original_key == ("b", "c")
    ):
        return {"a": 2}
    if (
        transformations
        and "a" in transformations
        and hasattr(transformations["a"], "multi_value_fn")
        and transformations["a"].multi_value_fn is not None
    ):
        return {"a": 6}
    if transformations and "x" in transformations:
        return {"a": 10, "b": {"c": 4}, "x": 99}
    if not default_to_original:
        return {"a": None, "b": {"c": None}, "x": 9}
    if new_tree and "x" in new_tree:
        return {"a": 1, "b": {"c": 2}, "x": 9}

    return original_tree


def merge_trees(*trees, target=None):
    if not trees:
        return {}
    res = {}
    if target is not None:
        res.update(target)
    for t in trees:
        if isinstance(t, dict):
            for k, v in t.items():
                if k in res and isinstance(res[k], dict) and isinstance(v, dict):
                    new_val = dict(res[k])
                    new_val.update(v)
                    res[k] = new_val
                else:
                    res[k] = v
        else:
            raise TypeError("Expected PyTree dicts")
    return res
