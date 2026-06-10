"""Provide module-level functionality."""

import ml_switcheroo  # type: ignore[import-untyped]
import datetime
from typing import Any, Callable, Container, Optional, Sequence


class args:
    """Represent the class."""

    class ArrayRestore:
        """Represent the class."""

        def __init__(self, *args, **kwargs):
            """Execute the function."""

    class ArraySave:
        """Represent the class."""

        def __init__(self, *args, **kwargs):
            """Execute the function."""

    class Composite:
        """Represent the class."""

        def __init__(self, *args, **kwargs):
            """Execute the function."""

    class JsonRestore:
        """Represent the class."""

        def __init__(self, *args, **kwargs):
            """Execute the function."""

    class JsonSave:
        """Represent the class."""

        def __init__(self, *args, **kwargs):
            """Execute the function."""

    class ProtoRestore:
        """Represent the class."""

        def __init__(self, *args, **kwargs):
            """Execute the function."""

    class ProtoSave:
        """Represent the class."""

        def __init__(self, *args, **kwargs):
            """Execute the function."""

    class PyTreeRestore:
        """Represent the class."""

        def __init__(self, *args, **kwargs):
            """Execute the function."""

    class PyTreeSave:
        """Represent the class."""

        def __init__(self, *args, **kwargs):
            """Execute the function."""

    class StandardRestore:
        """Represent the class."""

        def __init__(self, *args, **kwargs):
            """Execute the function."""

    class StandardSave:
        """Represent the class."""

        def __init__(self, *args, **kwargs):
            """Execute the function."""

    class JaxRandomKeySave:
        """Represent the class."""

        def __init__(self, *args, **kwargs):
            """Execute the function."""

    class JaxRandomKeyRestore:
        """Represent the class."""

        def __init__(self, *args, **kwargs):
            """Execute the function."""

    class NumpyRandomKeySave:
        """Represent the class."""

        def __init__(self, *args, **kwargs):
            """Execute the function."""

    class NumpyRandomKeyRestore:
        """Represent the class."""

        def __init__(self, *args, **kwargs):
            """Execute the function."""

    class CheckpointArgs:
        """Represent the class."""

        pass

    @staticmethod
    def get_registered_handler_cls(*a, **kw):
        """Execute the function."""
        return None

    @staticmethod
    def get_registered_args_cls(*a, **kw):
        """Execute the function."""
        return None

    @staticmethod
    def has_registered_args(*a, **kw):
        """Execute the function."""
        return False

    @staticmethod
    def register_with_handler(*a, **kw):
        """Execute the function."""
        return lambda cls: cls


class async_checkpoint_handler:
    """Represent the class."""

    AsyncCheckpointHandler = Any


class orbax:
    """Represent the class."""

    class checkpoint:
        """Represent the class."""

        class options:
            """Represent the class."""

            MultiprocessingOptions = Any
            AsyncOptions = Any
            FileOptions = Any


class atomicity:
    """Represent the class."""

    TemporaryPath = Any


class checkpoint:
    """Represent the class."""

    CheckpointMetadataStore = Any


class multihost:
    """Represent the class."""

    BarrierSyncFn = Any


class epath:
    """Represent the class."""

    PathLike = Any


class abstract_logger:
    """Represent the class."""

    AbstractLogger = Any


class step_lib:
    """Represent the class."""

    class Metadata:
        """Represent the class."""

        pass

    class NameFormat:
        """Represent the class."""

        def __init__(self, single_host_load_and_broadcast=False):
            """Execute the function."""
            self.single_host_load_and_broadcast = single_host_load_and_broadcast

        def __eq__(self, other):
            """Execute the function."""
            if isinstance(other, step_lib.NameFormat):
                return (
                    self.single_host_load_and_broadcast
                    == other.single_host_load_and_broadcast
                )
            return False

    @staticmethod
    def standard_name_format(
        single_host_load_and_broadcast=False, *args, **kwargs
    ) -> Any:
        """Execute the function."""
        return step_lib.NameFormat(
            single_host_load_and_broadcast=single_host_load_and_broadcast
        )


step = step_lib
CheckpointHandler = Any


class checkpoint_handler:
    """Represent the class."""

    class CheckpointHandler:
        """Represent the class."""

        pass


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
    """Represent the class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Execute the function."""
        pass

    def save(self, step: int, items: Any, **kwargs: Any) -> bool:
        """Execute the function."""
        raise NotImplementedError

    def restore(self, step: int, items: Optional[Any] = None, **kwargs: Any) -> Any:
        """Execute the function."""
        raise NotImplementedError

    def latest_step(self) -> Optional[int]:
        """Execute the function."""
        raise NotImplementedError

    def all_steps(self) -> Sequence[int]:
        """Execute the function."""
        raise NotImplementedError


class CheckpointManager(AbstractCheckpointManager):
    """Represent the class."""

    def __init__(
        self,
        directory: Any,
        checkpointers: Optional[Any] = None,
        options: "Optional[Any]" = None,
        metadata: Optional[int] = None,
        item_names: Optional[Any] = None,
        item_handlers: Optional[Any] = None,
        logger: Optional[Any] = None,
        handler_registry: Optional[Any] = None,
    ) -> None:
        """Execute the function."""
        self.directory = directory
        self.options = options
        self.metadata = metadata
        self.steps: list[int] = []
        self.checkpoints: dict[int, Any] = {}
        self.latest: Optional[int] = None

    def save(self, step: int, items: Any, **kwargs: Any) -> bool:
        """Execute the function."""
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
        """Execute the function."""
        if step not in self.checkpoints:
            raise ValueError(f"Checkpoint for step {step} not found.")
        return self.checkpoints.get(step, items)

    def latest_step(self) -> Optional[int]:
        """Execute the function."""
        return self.latest

    def all_steps(self) -> Sequence[int]:
        """Execute the function."""
        return self.steps


class AbstractCheckpointer:
    """Represent the class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Execute the function."""
        pass

    def save(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute the function."""
        raise NotImplementedError

    def restore(
        self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute the function."""
        raise NotImplementedError


class AsyncCheckpointer(AbstractCheckpointer):
    """Represent the class."""

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
        """Execute the function."""
        self.handler = handler if handler is not None else _handler

    def save(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> "Future":
        """Execute the function."""
        if self.handler and hasattr(self.handler, "save"):
            self.handler.save(path, item, *args, **kwargs)
        return Future(result=None)

    def restore(
        self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any
    ) -> "Future":
        """Execute the function."""
        res = None
        if self.handler and hasattr(self.handler, "restore"):
            res = self.handler.restore(path, item, *args, **kwargs)
        return Future(result=res)

    def wait_until_finished(self):
        """Execute the function."""
        pass


class AsyncOptions:
    """Represent the class."""

    def __init__(
        self,
        timeout_secs: int = 300,
        barrier_sync_fn: Optional[Any] = None,
        post_finalization_callback: Optional[Any] = None,
    ) -> None:
        """Execute the function."""
        self.timeout_secs = timeout_secs
        self.barrier_sync_fn = barrier_sync_fn
        self.post_finalization_callback = post_finalization_callback


import dataclasses


@dataclasses.dataclass
class CheckpointManagerOptions:
    """Represent the class."""

    save_interval_steps: int = 1
    max_to_keep: Optional[int] = None
    keep_time_interval: Optional[datetime.timedelta] = None
    keep_period: Optional[int] = None
    best_fn: Optional[Any] = None
    best_mode: str = "max"
    keep_checkpoints_without_metrics: bool = True
    step_prefix: Optional[str] = None
    step_format_fixed_length: Optional[int] = None
    step_name_format: Optional[Any] = None
    create: bool = True
    cleanup_tmp_directories: bool = False
    save_on_steps: Optional[Any] = None
    single_host_load_and_broadcast: bool = False
    todelete_subdir: Optional[str] = None
    enable_background_delete: bool = False
    read_only: bool = False
    enable_async_checkpointing: bool = True
    async_options: Optional[Any] = None
    multiprocessing_options: MultiprocessingOptions = None
    should_save_fn: Optional[Any] = None
    should_keep_fn: Optional[Any] = None
    file_options: FileOptions = None
    temporary_path_class: Optional[Any] = None
    todelete_full_path: Optional[str] = None
    preservation_policy: Optional[Any] = None

    def __post_init__(self):
        """Execute the function."""
        step_name_format_single_host_load_and_broadcast = (
            hasattr(self.step_name_format, "single_host_load_and_broadcast")
            and self.step_name_format.single_host_load_and_broadcast
        )
        if self.single_host_load_and_broadcast and self.step_name_format:
            if not step_name_format_single_host_load_and_broadcast:
                raise ValueError(
                    "`CheckpointManagerOptions.single_host_load_and_broadcast=True` requires `step_name_format.single_host_load_and_broadcast` to be set to True."
                )
        if step_name_format_single_host_load_and_broadcast and (
            not self.single_host_load_and_broadcast
        ):
            raise ValueError(
                "`step_name_format.single_host_load_and_broadcast=True` but `CheckpointManagerOptions.single_host_load_and_broadcast=False`."
            )
        if self.best_mode not in ("min", "max"):
            raise ValueError(
                f"`CheckpointManagerOptions.best_mode` must be one of None, 'min' or 'max'. Got {self.best_mode}."
            )
        if (
            self.preservation_policy is None
            and self.max_to_keep is not None
            and (self.max_to_keep < 0)
        ):
            raise ValueError("Setting of `max_to_keep` must be None or non-negative.")
        if self.save_interval_steps <= 0 and (not self.read_only):
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
            raise ValueError(
                "todelete_subdir and todelete_full_path both cannot be set togther"
            )
        if self.preservation_policy is None and self.should_keep_fn is not None:
            self.keep_period = None
        self.save_on_steps = frozenset(self.save_on_steps or ())

    def replace(self, **kwargs):
        """Execute the function."""
        attrs = dict(self.__dict__)
        attrs.update(kwargs)
        return CheckpointManagerOptions(**attrs)


class Checkpointer(AbstractCheckpointer):
    """Represent the class."""

    def __init__(
        self,
        handler: checkpoint_handler.CheckpointHandler,
        *,
        multiprocessing_options=None,
        file_options=None,
        checkpoint_metadata_store=None,
        temporary_path_class=None,
    ) -> None:
        """Execute the function."""
        self.handler = handler

    def save(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> None:
        """Execute the function."""
        if hasattr(self.handler, "save"):
            self.handler.save(path, item, *args, **kwargs)

    def restore(
        self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute the function."""
        if hasattr(self.handler, "restore"):
            return self.handler.restore(path, item, *args, **kwargs)
        return item


class Future:
    """Represent the class."""

    def __init__(self, result=None, *args: Any, **kwargs: Any) -> None:
        """Execute the function."""
        self._result = result

    def result(self) -> Any:
        """Execute the function."""
        return self._result


class PyTreeCheckpointer(AbstractCheckpointer):
    """Represent the class."""

    def __init__(
        self,
        primary_host: Optional[int] = 0,
        use_ocdbt: bool = True,
        use_zarr3: bool = False,
    ) -> None:
        """Execute the function."""
        pass

    def save(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute the function."""
        pass

    def restore(
        self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute the function."""
        return item


class RestoreTransform:
    """Represent the class."""

    def __init__(
        self,
        value_fn: Optional[Any] = None,
        multi_value_fn: Optional[Any] = None,
        multi_value_fn_input_args: Optional[Any] = None,
        original_key: Optional[Any] = None,
        use_fallback: bool = False,
    ) -> None:
        """Execute the function."""
        self.original_key = original_key
        self.use_fallback = use_fallback
        self.value_fn = value_fn
        self.multi_value_fn = multi_value_fn


class StandardCheckpointer(AbstractCheckpointer):
    """Represent the class."""

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
        """Execute the function."""
        pass

    def save(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute the function."""
        pass

    def restore(
        self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute the function."""
        return item


class Transform:
    """Represent the class."""

    def __init__(
        self,
        original_key: Optional[Any] = None,
        use_fallback: bool = False,
        value_fn: Optional[Any] = None,
        multi_value_fn: Optional[Any] = None,
    ) -> None:
        """Execute the function."""
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
    """Execute the function."""
    if not isinstance(new_tree, dict):
        return new_tree

    def get_by_path(tree, path, default_val):
        if isinstance(path, str):
            res = tree.get(path)
            return res if res is not None else default_val
        for k in path:
            if not isinstance(tree, dict):
                return default_val
            tree = tree.get(k)
        return tree

    def process(orig, trans, new, orig_full):
        """Execute the function."""
        if not isinstance(new, dict):
            return new
        res = {}
        for k in new.keys():
            t = trans.get(k) if isinstance(trans, dict) else None
            o = orig.get(k) if isinstance(orig, dict) else None
            if t is not None:
                if isinstance(t, dict):
                    res[k] = process(o if o is not None else {}, t, new[k], orig_full)
                elif hasattr(t, "multi_value_fn") and getattr(t, "multi_value_fn"):
                    res[k] = t.multi_value_fn(orig_full)
                elif type(t).__name__ == "Transform":
                    if hasattr(t, "original_key") and getattr(t, "original_key"):
                        o = get_by_path(orig_full, t.original_key, o)
                    if hasattr(t, "value_fn") and getattr(t, "value_fn"):
                        res[k] = t.value_fn(o)
                    else:
                        res[k] = o
                else:
                    res[k] = t
            elif not default_to_original:
                res[k] = new[k]
            else:
                res[k] = o if o is not None else new[k]
        return res

    return process(original_tree, transformations, new_tree, original_tree)


def merge_trees(*trees, target=None):
    """Execute the function."""
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
