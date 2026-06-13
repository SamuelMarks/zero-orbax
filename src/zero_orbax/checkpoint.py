"Provide module-level functionality for checkpointing.\n\nThis module provides various classes and utilities to save and restore\nmachine learning models, trees, and other artifacts in a structured way.\n"

import datetime
import asyncio
import contextlib
import functools
import logging
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)


class args:
    "Namespace for various checkpoint argument classes.\n\n    Contains classes that specify how different types of items\n    should be saved or restored.\n"

    class ArrayRestore:
        "Arguments for restoring an array."

        def __init__(self, *args, **kwargs):
            "Initialize the ArrayRestore arguments.\n\n            Args:\n                *args: Variable length argument list.\n                **kwargs: Arbitrary keyword arguments.\n"

    class ArraySave:
        "Arguments for saving an array."

        def __init__(self, *args, **kwargs):
            "Initialize the ArraySave arguments.\n\n            Args:\n                *args: Variable length argument list.\n                **kwargs: Arbitrary keyword arguments.\n"

    class Composite:
        "Arguments for a composite checkpoint handler."

        def __init__(self, *args, **kwargs):
            "Initialize the Composite arguments.\n\n            Args:\n                *args: Variable length argument list.\n                **kwargs: Arbitrary keyword arguments.\n"

    class JsonRestore:
        "Arguments for restoring from a JSON file."

        def __init__(self, *args, **kwargs):
            "Initialize the JsonRestore arguments.\n\n            Args:\n                *args: Variable length argument list.\n                **kwargs: Arbitrary keyword arguments.\n"

    class JsonSave:
        "Arguments for saving to a JSON file."

        def __init__(self, *args, **kwargs):
            "Initialize the JsonSave arguments.\n\n            Args:\n                *args: Variable length argument list.\n                **kwargs: Arbitrary keyword arguments.\n"

    class ProtoRestore:
        "Arguments for restoring from a Protocol Buffer."

        def __init__(self, *args, **kwargs):
            "Initialize the ProtoRestore arguments.\n\n            Args:\n                *args: Variable length argument list.\n                **kwargs: Arbitrary keyword arguments.\n"

    class ProtoSave:
        "Arguments for saving to a Protocol Buffer."

        def __init__(self, *args, **kwargs):
            "Initialize the ProtoSave arguments.\n\n            Args:\n                *args: Variable length argument list.\n                **kwargs: Arbitrary keyword arguments.\n"

    class PyTreeRestore:
        "Arguments for restoring a PyTree structure."

        def __init__(self, *args, **kwargs):
            "Initialize the PyTreeRestore arguments.\n\n            Args:\n                *args: Variable length argument list.\n                **kwargs: Arbitrary keyword arguments.\n"

    class PyTreeSave:
        "Arguments for saving a PyTree structure."

        def __init__(self, *args, **kwargs):
            "Initialize the PyTreeSave arguments.\n\n            Args:\n                *args: Variable length argument list.\n                **kwargs: Arbitrary keyword arguments.\n"

    class StandardRestore:
        "Arguments for standard restoration processes."

        def __init__(self, *args, **kwargs):
            "Initialize the StandardRestore arguments.\n\n            Args:\n                *args: Variable length argument list.\n                **kwargs: Arbitrary keyword arguments.\n"

    class StandardSave:
        "Arguments for standard saving processes."

        def __init__(self, *args, **kwargs):
            "Initialize the StandardSave arguments.\n\n            Args:\n                *args: Variable length argument list.\n                **kwargs: Arbitrary keyword arguments.\n"

    class JaxRandomKeySave:
        "Arguments for saving a JAX random key."

        def __init__(self, *args, **kwargs):
            "Initialize the JaxRandomKeySave arguments.\n\n            Args:\n                *args: Variable length argument list.\n                **kwargs: Arbitrary keyword arguments.\n"

    class JaxRandomKeyRestore:
        "Arguments for restoring a JAX random key."

        def __init__(self, *args, **kwargs):
            "Initialize the JaxRandomKeyRestore arguments.\n\n            Args:\n                *args: Variable length argument list.\n                **kwargs: Arbitrary keyword arguments.\n"

    class NumpyRandomKeySave:
        "Arguments for saving a NumPy random key."

        def __init__(self, *args, **kwargs):
            "Initialize the NumpyRandomKeySave arguments.\n\n            Args:\n                *args: Variable length argument list.\n                **kwargs: Arbitrary keyword arguments.\n"

    class NumpyRandomKeyRestore:
        "Arguments for restoring a NumPy random key."

        def __init__(self, *args, **kwargs):
            "Initialize the NumpyRandomKeyRestore arguments.\n\n            Args:\n                *args: Variable length argument list.\n                **kwargs: Arbitrary keyword arguments.\n"

    class CheckpointArgs:
        "Base arguments for a checkpoint handler."

        pass

    @staticmethod
    def get_registered_handler_cls(*a, **kw):
        "Get the registered handler class for the given arguments.\n\n        Args:\n            *a: Positional arguments for the registry lookup.\n            **kw: Keyword arguments for the registry lookup.\n\n        Returns:\n            The registered handler class, or None if not found.\n"
        return None

    @staticmethod
    def get_registered_args_cls(*a, **kw):
        "Get the registered arguments class.\n\n        Args:\n            *a: Positional arguments for the registry lookup.\n            **kw: Keyword arguments for the registry lookup.\n\n        Returns:\n            The registered args class, or None if not found.\n"
        return None

    @staticmethod
    def has_registered_args(*a, **kw):
        "Check if there are registered arguments.\n\n        Args:\n            *a: Positional arguments for the registry lookup.\n            **kw: Keyword arguments for the registry lookup.\n\n        Returns:\n            bool: True if registered arguments exist, False otherwise.\n"
        return False

    @staticmethod
    def register_with_handler(*a, **kw):
        "Register a handler class.\n\n        Args:\n            *a: Positional arguments for the registration.\n            **kw: Keyword arguments for the registration.\n\n        Returns:\n            Callable: A decorator function that returns the class unchanged.\n"
        return lambda cls: cls


class async_checkpoint_handler:
    "Namespace for asynchronous checkpoint handler functionality."

    AsyncCheckpointHandler = Any


class orbax:
    "Namespace for orbax classes."

    class checkpoint:
        "Namespace for orbax checkpointing."

        class options:
            "Namespace for orbax checkpointing options."

            MultiprocessingOptions = Any
            AsyncOptions = Any
            FileOptions = Any


class atomicity:
    "Namespace for atomic operations and paths."

    TemporaryPath = Any


class checkpoint:
    "Namespace for general checkpoint operations."

    CheckpointMetadataStore = Any


class multihost:
    "Namespace for multi-host synchronization."

    BarrierSyncFn = Any


class epath:
    "Namespace for path-like interfaces."

    PathLike = Any


class abstract_logger:
    "Namespace for abstract logging interfaces."

    AbstractLogger = Any


class step_lib:
    "Namespace for step-related metadata and formatting."

    class Metadata:
        "Metadata associated with a checkpoint step."

        pass

    class NameFormat:
        "Formatting logic for step names."

        def __init__(self, single_host_load_and_broadcast=False):
            "Initialize the NameFormat.\n\n            Args:\n                single_host_load_and_broadcast (bool): Whether to format for single host load and broadcast.\n"
            self.single_host_load_and_broadcast = single_host_load_and_broadcast

        def __eq__(self, other):
            "Check equality between two NameFormat instances.\n\n            Args:\n                other (Any): The other object to compare against.\n\n            Returns:\n                bool: True if the objects are equal, False otherwise.\n"
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
        "Create a standard step name format.\n\n        Args:\n            single_host_load_and_broadcast (bool): Whether to format for single host load and broadcast.\n            *args: Additional positional arguments.\n            **kwargs: Additional keyword arguments.\n\n        Returns:\n            step_lib.NameFormat: A populated NameFormat instance.\n"
        return step_lib.NameFormat(
            single_host_load_and_broadcast=single_host_load_and_broadcast
        )


step = step_lib


class CheckpointHandler:
    "Mock implementation for CheckpointHandler."

    def metadata(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def finalize(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def restore(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def close(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def save(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    pass


class checkpoint_handler:
    "Namespace for checkpoint handler."

    class CheckpointHandler:
        "Base class for handlers that manage checkpoint reading and writing."

        pass


MultiprocessingOptions = Any
FileOptions = Any
CheckpointersDict = Any


class CheckpointHandlerRegistry:
    "Mock implementation for CheckpointHandlerRegistry."

    def get_all_entries(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def get(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def has(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def add(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    pass


CheckpointHandlersDict = Any
PyTree = Any


class RestoreArgs:
    "Mock implementation for RestoreArgs."

    pass


NoneType = type(None)
ValueFn = Any
MultiValueFn = Any


class AbstractCheckpointManager:
    "Mock implementation for AbstractCheckpointManager."

    def reached_preemption(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def check_for_errors(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def metadata(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def best_step(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def close(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def metrics(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def reload(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def wait_until_finished(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def should_save(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def delete(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def item_metadata(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        "Initialize the AbstractCheckpointManager.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n"
        pass

    def save(self, step: int, items: Any, **kwargs: Any) -> bool:
        "Save a checkpoint at the given step.\n\n        Args:\n            step (int): The step number at which to save.\n            items (Any): The objects to save.\n            **kwargs (Any): Additional options for saving.\n\n        Returns:\n            bool: True if the save was successful, False otherwise.\n\n        Raises:\n            NotImplementedError: If not implemented by subclasses.\n"
        raise NotImplementedError

    def restore(self, step: int, items: Optional[Any] = None, **kwargs: Any) -> Any:
        "Restore a checkpoint from a specific step.\n\n        Args:\n            step (int): The step number to restore.\n            items (Optional[Any]): The target structure to restore into. Defaults to None.\n            **kwargs (Any): Additional options for restoring.\n\n        Returns:\n            Any: The restored objects.\n\n        Raises:\n            NotImplementedError: If not implemented by subclasses.\n"
        raise NotImplementedError

    def latest_step(self) -> Optional[int]:
        "Get the latest saved step.\n\n        Returns:\n            Optional[int]: The most recent step number, or None if no checkpoints exist.\n\n        Raises:\n            NotImplementedError: If not implemented by subclasses.\n"
        raise NotImplementedError

    def all_steps(self) -> Sequence[int]:
        "Get all steps with available checkpoints.\n\n        Returns:\n            Sequence[int]: A sequence of all step numbers available for restoration.\n\n        Raises:\n            NotImplementedError: If not implemented by subclasses.\n"
        raise NotImplementedError


class CheckpointManager(AbstractCheckpointManager):
    "Mock implementation for CheckpointManager."

    def reached_preemption(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def check_for_errors(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def metadata(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def best_step(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def close(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def is_saving_in_progress(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def metrics(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def reload(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def wait_until_finished(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def should_save(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def delete(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def item_metadata(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

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
        "Initialize the CheckpointManager.\n\n        Args:\n            directory (Any): The base directory for storing checkpoints.\n            checkpointers (Optional[Any]): Mappings of checkpointer objects. Defaults to None.\n            options (Optional[Any]): Options for managing checkpoint retention and save conditions. Defaults to None.\n            metadata (Optional[int]): Additional metadata for the checkpoints. Defaults to None.\n            item_names (Optional[Any]): Specific names of items to track. Defaults to None.\n            item_handlers (Optional[Any]): Custom handlers for individual items. Defaults to None.\n            logger (Optional[Any]): A logger instance. Defaults to None.\n            handler_registry (Optional[Any]): Registry mapping items to their handlers. Defaults to None.\n"
        self.directory = directory
        self.options = options
        self._metadata = metadata
        self.steps: list[int] = []
        self.checkpoints: dict[(int, Any)] = {}
        self.latest: Optional[int] = None

    def save(self, step: int, items: Any, **kwargs: Any) -> bool:
        "Save a checkpoint at the given step if conditions are met.\n\n        Args:\n            step (int): The current training step.\n            items (Any): The items to save.\n            **kwargs (Any): Additional options for the save operation.\n\n        Returns:\n            bool: True if the checkpoint was successfully saved, False otherwise.\n\n        Raises:\n            ValueError: If attempting to save in read-only mode.\n"
        if self.options and getattr(self.options, "read_only", False):
            raise ValueError("Cannot save checkpoint in read_only mode.")
        if self.options and (self.options.should_save_fn is not None):
            if not self.options.should_save_fn(step, self.latest):
                return False
        if self.options and ((step % self.options.save_interval_steps) != 0):
            if not (
                self.options.save_on_steps and (step in self.options.save_on_steps)
            ):
                return False
        self.steps.append(step)
        self.checkpoints[step] = items
        self.latest = max(self.steps)
        if self.options and (self.options.max_to_keep is not None):
            if len(self.steps) > self.options.max_to_keep:
                to_remove = self.steps[0]
                self.steps = self.steps[1:]
                del self.checkpoints[to_remove]
        return True

    def restore(self, step: int, items: Optional[Any] = None, **kwargs: Any) -> Any:
        "Restore a checkpoint for a specific step.\n\n        Args:\n            step (int): The step number to restore.\n            items (Optional[Any]): The structure to restore into. Defaults to None.\n            **kwargs (Any): Additional options for the restore operation.\n\n        Returns:\n            Any: The restored objects.\n\n        Raises:\n            ValueError: If the requested step is not found in the stored checkpoints.\n"
        if step not in self.checkpoints:
            raise ValueError(f"Checkpoint for step {step} not found.")
        return self.checkpoints.get(step, items)

    def latest_step(self) -> Optional[int]:
        "Return the most recently saved step number.\n\n        Returns:\n            Optional[int]: The most recent step number, or None if no checkpoints exist.\n"
        return self.latest

    def all_steps(self) -> Sequence[int]:
        "Return a sequence of all step numbers that have a checkpoint.\n\n        Returns:\n            Sequence[int]: A sequence of step numbers.\n"
        return self.steps


class AbstractCheckpointer:
    "Mock implementation for AbstractCheckpointer."

    def close(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def metadata(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def structure(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        "Initialize the AbstractCheckpointer.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n"
        pass

    def save(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> Any:
        "Save an item to a given path.\n\n        Args:\n            path (Any): The destination path.\n            item (Any): The item to save.\n            *args (Any): Additional positional arguments.\n            **kwargs (Any): Additional keyword arguments.\n\n        Returns:\n            Any: Result of the save operation.\n\n        Raises:\n            NotImplementedError: If not implemented by subclasses.\n"
        raise NotImplementedError

    def restore(
        self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any
    ) -> Any:
        "Restore an item from a given path.\n\n        Args:\n            path (Any): The source path to restore from.\n            item (Optional[Any]): The target structure to restore into. Defaults to None.\n            *args (Any): Additional positional arguments.\n            **kwargs (Any): Additional keyword arguments.\n\n        Returns:\n            Any: The restored item.\n\n        Raises:\n            NotImplementedError: If not implemented by subclasses.\n"
        raise NotImplementedError


class AsyncCheckpointer(AbstractCheckpointer):
    "Mock implementation for AsyncCheckpointer."

    def check_for_errors(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def metadata(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def close(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def create_temporary_path(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def structure(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

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
        "Initialize the AsyncCheckpointer.\n\n        Args:\n            _handler (Any, optional): Legacy positional handler argument. Defaults to None.\n            multiprocessing_options (Any, optional): Options for multiprocessing. Defaults to None.\n            timeout_secs (int, optional): Timeout in seconds. Defaults to None.\n            handler (Any, optional): The checkpoint handler. Defaults to None.\n            temporary_path_class (Any, optional): Class used to generate temporary paths. Defaults to None.\n            async_options (Any, optional): Options configuring async behavior. Defaults to None.\n            file_options (Any, optional): File-specific options. Defaults to None.\n            checkpoint_metadata_store (Any, optional): Store for checkpoint metadata. Defaults to None.\n"
        self.handler = handler if (handler is not None) else _handler

    def save(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> "Future":
        "Initiate an asynchronous save operation.\n\n        Args:\n            path (Any): The destination path.\n            item (Any): The item to save.\n            *args (Any): Additional positional arguments.\n            **kwargs (Any): Additional keyword arguments.\n\n        Returns:\n            Future: A future object representing the pending save operation.\n"
        if self.handler and hasattr(self.handler, "save"):
            self.handler.save(path, item, *args, **kwargs)
        return Future(result=None)

    def restore(
        self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any
    ) -> "Future":
        "Initiate an asynchronous restore operation.\n\n        Args:\n            path (Any): The source path.\n            item (Optional[Any]): The target structure. Defaults to None.\n            *args (Any): Additional positional arguments.\n            **kwargs (Any): Additional keyword arguments.\n\n        Returns:\n            Future: A future object representing the pending restore operation, yielding the restored item.\n"
        res = None
        if self.handler and hasattr(self.handler, "restore"):
            res = self.handler.restore(path, item, *args, **kwargs)
        return Future(result=res)

    def wait_until_finished(self):
        "Block until all background operations are complete.\n\n        Returns:\n            None\n"
        pass


class AsyncOptions:
    "Configuration options for asynchronous checkpointing."

    def __init__(
        self,
        timeout_secs: int = 300,
        barrier_sync_fn: Optional[Any] = None,
        post_finalization_callback: Optional[Any] = None,
    ) -> None:
        "Initialize the AsyncOptions.\n\n        Args:\n            timeout_secs (int): Timeout duration in seconds. Defaults to 300.\n            barrier_sync_fn (Optional[Any]): Function for syncing across hosts. Defaults to None.\n            post_finalization_callback (Optional[Any]): Callback executed after finalization. Defaults to None.\n"
        self.timeout_secs = timeout_secs
        self.barrier_sync_fn = barrier_sync_fn
        self.post_finalization_callback = post_finalization_callback


import dataclasses


@dataclasses.dataclass
class CheckpointManagerOptions:
    "Options to configure the behavior of a CheckpointManager."

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
        "Perform validation of options after initialization.\n\n        Raises:\n            ValueError: If the configuration combination is invalid.\n"
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
            (self.preservation_policy is None)
            and (self.max_to_keep is not None)
            and (self.max_to_keep < 0)
        ):
            raise ValueError("Setting of `max_to_keep` must be None or non-negative.")
        if (self.save_interval_steps <= 0) and (not self.read_only):
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
        if (self.todelete_subdir is not None) and (self.todelete_full_path is not None):
            raise ValueError(
                "todelete_subdir and todelete_full_path both cannot be set togther"
            )
        if (self.preservation_policy is None) and (self.should_keep_fn is not None):
            self.keep_period = None
        self.save_on_steps = frozenset((self.save_on_steps or ()))

    def replace(self, **kwargs):
        "Create a new options instance with updated attributes.\n\n        Args:\n            **kwargs: Attributes to update.\n\n        Returns:\n            CheckpointManagerOptions: A new options instance with the modifications.\n"
        attrs = dict(self.__dict__)
        attrs.update(kwargs)
        return CheckpointManagerOptions(**attrs)


class Checkpointer(AbstractCheckpointer):
    "Mock implementation for Checkpointer."

    def close(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def metadata(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def structure(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def create_temporary_path(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def __init__(
        self,
        handler: checkpoint_handler.CheckpointHandler,
        *,
        multiprocessing_options=None,
        file_options=None,
        checkpoint_metadata_store=None,
        temporary_path_class=None,
    ) -> None:
        "Initialize the Checkpointer.\n\n        Args:\n            handler (checkpoint_handler.CheckpointHandler): The handler used to save and restore.\n            multiprocessing_options (Any, optional): Multiprocessing options. Defaults to None.\n            file_options (Any, optional): File-specific options. Defaults to None.\n            checkpoint_metadata_store (Any, optional): Metadata store for the checkpoints. Defaults to None.\n            temporary_path_class (Any, optional): Class used to generate temporary paths. Defaults to None.\n"
        self.handler = handler

    def save(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> None:
        "Save the given item to the specified path synchronously.\n\n        Args:\n            path (Any): The destination path.\n            item (Any): The item to save.\n            *args (Any): Additional positional arguments for the handler.\n            **kwargs (Any): Additional keyword arguments for the handler.\n\n        Returns:\n            None\n"
        if hasattr(self.handler, "save"):
            self.handler.save(path, item, *args, **kwargs)

    def restore(
        self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any
    ) -> Any:
        "Restore the given item from the specified path synchronously.\n\n        Args:\n            path (Any): The source path.\n            item (Optional[Any]): The target structure. Defaults to None.\n            *args (Any): Additional positional arguments for the handler.\n            **kwargs (Any): Additional keyword arguments for the handler.\n\n        Returns:\n            Any: The restored item.\n"
        if hasattr(self.handler, "restore"):
            return self.handler.restore(path, item, *args, **kwargs)
        return item


class Future:
    "A simple representation of an asynchronous result."

    def __init__(self, result=None, *args: Any, **kwargs: Any) -> None:
        "Initialize the Future.\n\n        Args:\n            result (Any, optional): The result to wrap. Defaults to None.\n            *args (Any): Additional positional arguments.\n            **kwargs (Any): Additional keyword arguments.\n"
        self._result = result

    def result(self) -> Any:
        "Retrieve the wrapped result.\n\n        Returns:\n            Any: The result stored in this future.\n"
        return self._result


class PyTreeCheckpointer(AbstractCheckpointer):
    "Mock implementation for PyTreeCheckpointer."

    def close(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def metadata(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def structure(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def create_temporary_path(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def __init__(
        self,
        primary_host: Optional[int] = 0,
        use_ocdbt: bool = True,
        use_zarr3: bool = False,
    ) -> None:
        "Initialize the PyTreeCheckpointer.\n\n        Args:\n            primary_host (Optional[int]): The primary host index. Defaults to 0.\n            use_ocdbt (bool): Whether to use OCDBT format. Defaults to True.\n            use_zarr3 (bool): Whether to use Zarr3 format. Defaults to False.\n"
        pass

    def save(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> Any:
        "Save a PyTree to the specified path.\n\n        Args:\n            path (Any): The destination path.\n            item (Any): The PyTree item to save.\n            *args (Any): Additional positional arguments.\n            **kwargs (Any): Additional keyword arguments.\n\n        Returns:\n            Any: The result of the save operation.\n"
        pass

    def restore(
        self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any
    ) -> Any:
        "Restore a PyTree from the specified path.\n\n        Args:\n            path (Any): The source path.\n            item (Optional[Any]): The target PyTree structure. Defaults to None.\n            *args (Any): Additional positional arguments.\n            **kwargs (Any): Additional keyword arguments.\n\n        Returns:\n            Any: The restored PyTree.\n"
        return item


class RestoreTransform:
    "Rule defining how to transform data during a restore operation."

    def __init__(
        self,
        value_fn: Optional[Any] = None,
        multi_value_fn: Optional[Any] = None,
        multi_value_fn_input_args: Optional[Any] = None,
        original_key: Optional[Any] = None,
        use_fallback: bool = False,
    ) -> None:
        "Initialize the RestoreTransform.\n\n        Args:\n            value_fn (Optional[Any]): A function to transform a single value. Defaults to None.\n            multi_value_fn (Optional[Any]): A function to derive a value from multiple sources. Defaults to None.\n            multi_value_fn_input_args (Optional[Any]): Arguments for the multi_value_fn. Defaults to None.\n            original_key (Optional[Any]): The key in the original structure to map from. Defaults to None.\n            use_fallback (bool): Whether to use fallback logic. Defaults to False.\n"
        self.original_key = original_key
        self.use_fallback = use_fallback
        self.value_fn = value_fn
        self.multi_value_fn = multi_value_fn


class StandardCheckpointer(AbstractCheckpointer):
    "Mock implementation for StandardCheckpointer."

    def check_for_errors(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def metadata(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def close(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def create_temporary_path(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def wait_until_finished(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def structure(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

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
        "Initialize the StandardCheckpointer.\n\n        Args:\n            async_options (Any, optional): Asynchronous options. Defaults to None.\n            multiprocessing_options (Any, optional): Multiprocessing options. Defaults to None.\n            file_options (Any, optional): File-specific options. Defaults to None.\n            checkpoint_metadata_store (Any, optional): Metadata store for the checkpoints. Defaults to None.\n            temporary_path_class (Any, optional): Class used to generate temporary paths. Defaults to None.\n            **kwargs (Optional[dict]): Additional keyword arguments.\n"
        pass

    def save(self, path: Any, item: Any, *args: Any, **kwargs: Any) -> Any:
        "Save a common item to the specified path.\n\n        Args:\n            path (Any): The destination path.\n            item (Any): The item to save.\n            *args (Any): Additional positional arguments.\n            **kwargs (Any): Additional keyword arguments.\n\n        Returns:\n            Any: The result of the save operation.\n"
        pass

    def restore(
        self, path: Any, item: Optional[Any] = None, *args: Any, **kwargs: Any
    ) -> Any:
        "Restore a common item from the specified path.\n\n        Args:\n            path (Any): The source path.\n            item (Optional[Any]): The target item structure. Defaults to None.\n            *args (Any): Additional positional arguments.\n            **kwargs (Any): Additional keyword arguments.\n\n        Returns:\n            Any: The restored item.\n"
        return item


class Transform:
    "Defines a general transformation on data values."

    def __init__(
        self,
        original_key: Optional[Any] = None,
        use_fallback: bool = False,
        value_fn: Optional[Any] = None,
        multi_value_fn: Optional[Any] = None,
    ) -> None:
        "Initialize the Transform.\n\n        Args:\n            original_key (Optional[Any]): The key in the original structure. Defaults to None.\n            use_fallback (bool): Whether to fallback if the transformation fails. Defaults to False.\n            value_fn (Optional[Any]): Function to apply to a single value. Defaults to None.\n            multi_value_fn (Optional[Any]): Function to apply to multiple values. Defaults to None.\n"
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
    "Apply a set of transformations to map from an original PyTree to a new PyTree structure.\n\n    Args:\n        original_tree (PyTree): The source tree structure.\n        transformations (PyTree): The tree of transformation rules.\n        new_tree (PyTree): The target structure that defines the expected output shape.\n        default_to_original (Optional[bool]): If True, fields missing in the transformation\n            fall back to the original tree's values. Defaults to True.\n\n    Returns:\n        Any: The transformed PyTree.\n"
    if not isinstance(new_tree, dict):
        return new_tree

    def get_by_path(tree, path, default_val):
        "Retrieve a nested dictionary value by string or iterable path.\n\n        Args:\n            tree (dict): The dictionary to search.\n            path (Union[str, Iterable]): The path to follow.\n            default_val (Any): The value to return if the path is not found.\n\n        Returns:\n            Any: The value at the specified path, or default_val.\n"
        if isinstance(path, str):
            if "/" in path:
                path = path.split("/")
            else:
                res = tree.get(path)
                return res if (res is not None) else default_val
        for k in path:
            if not isinstance(tree, dict):
                return default_val
            tree = tree.get(k)
        return tree

    def process(orig, trans, new, orig_full):
        "Recursively process transformations for each node in the PyTree.\n\n        Args:\n            orig: The current subtree in the original tree.\n            trans: The current subtree in the transformations.\n            new: The current subtree in the new tree.\n            orig_full: The full original tree for global path references.\n\n        Returns:\n            The processed sub-tree.\n"
        if not isinstance(new, dict):
            return new
        res = {}
        for k in new.keys():
            t = trans.get(k) if isinstance(trans, dict) else None
            o = orig.get(k) if isinstance(orig, dict) else None
            if t is not None:
                if isinstance(t, dict):
                    res[k] = process(
                        (o if (o is not None) else {}), t, new[k], orig_full
                    )
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
            elif (new[k] is not None) and isinstance(new[k], dict):
                res[k] = process((o if (o is not None) else {}), {}, new[k], orig_full)
            elif not default_to_original:
                res[k] = new[k]
            else:
                res[k] = o if (o is not None) else new[k]
        return res

    return process(original_tree, transformations, new_tree, original_tree)


def merge_trees(*trees, target=None):
    "Merge multiple PyTrees into a single PyTree dict.\n\n    Args:\n        *trees: A variable number of PyTree dictionaries to merge.\n        target (Optional[dict]): An initial target dictionary to merge into. Defaults to None.\n\n    Returns:\n        dict: The deeply merged PyTree dictionary.\n\n    Raises:\n        TypeError: If any of the provided trees are not dictionaries.\n"
    if not trees:
        return {}
    res = {}
    if target is not None:
        res.update(target)
    for t in trees:
        if isinstance(t, dict):
            for k, v in t.items():
                if (k in res) and isinstance(res[k], dict) and isinstance(v, dict):
                    new_val = dict(res[k])
                    new_val.update(v)
                    res[k] = new_val
                else:
                    res[k] = v
        else:
            raise TypeError("Expected PyTree dicts")
    return res


class nest_asyncio:
    "Namespace for nest_asyncio."

    pass


class tree:
    "Namespace for tree."

    pass


class abstract_checkpoint_manager:
    "Namespace for abstract_checkpoint_manager."

    pass


class abstract_checkpointer:
    "Namespace for abstract_checkpointer."

    pass


class aggregate_handlers:
    "Namespace for aggregate_handlers."

    pass


class async_checkpointer_module:
    "Namespace for async_checkpointer."

    pass


class checkpoint_args:
    "Namespace for checkpoint_args."

    pass


class checkpoint_manager_module:
    "Namespace for checkpoint_manager."

    pass


class checkpoint_utils:
    "Namespace for checkpoint_utils."

    pass


class checkpointer_module:
    "Namespace for checkpointer."

    pass


class future_module:
    "Namespace for future."

    pass


class handlers:
    "Namespace for handlers."

    pass


class metadata:
    "Namespace for metadata."

    pass


class msgpack_utils:
    "Namespace for msgpack_utils."

    pass


class path:
    "Namespace for path."

    pass


class pytree_checkpointer_module:
    "Namespace for pytree_checkpointer."

    pass


class serialization:
    "Namespace for serialization."

    pass


class standard_checkpointer_module:
    "Namespace for standard_checkpointer."

    pass


class test_utils:
    "Namespace for test_utils."

    pass


class transform_utils:
    "Namespace for transform_utils."

    pass


class type_handlers:
    "Namespace for type_handlers."

    pass


class utils:
    "Namespace for utils."

    pass


async_checkpointer = async_checkpointer_module
checkpoint_manager = checkpoint_manager_module
checkpointer = checkpointer_module
future = future_module
pytree_checkpointer = pytree_checkpointer_module
standard_checkpointer = standard_checkpointer_module


class SaveArgs:
    "Arguments for saving a checkpoint."

    def __init__(
        self,
        aggregate: bool = False,
        dtype: Optional[Any] = None,
        chunk_byte_size: Optional[int] = None,
    ) -> None:
        "Initialize."
        self.aggregate = aggregate
        self.dtype = dtype
        self.chunk_byte_size = chunk_byte_size


class ArrayRestoreArgs:
    "Mock implementation for ArrayRestoreArgs."

    def restore_type(self, *args: Any, **kwargs: Any) -> Any:
        "Provide a mock implementation for parity.\n\n        Args:\n            *args (Any): Positional arguments.\n            **kwargs (Any): Keyword arguments.\n\n        Returns:\n            Any: The result."
        pass

    def __init__(
        self,
        restore_type: Optional[Any] = None,
        dtype: Optional[Any] = None,
        mesh: Optional[Any] = None,
        mesh_axes: Optional[Any] = None,
        sharding: Optional[Any] = None,
        global_shape: Optional[Tuple[(int, ...)]] = None,
    ) -> None:
        "Initialize."
        self._restore_type = restore_type
        self.dtype = dtype
        self.mesh = mesh
        self.mesh_axes = mesh_axes
        self.sharding = sharding
        self.global_shape = global_shape


class AsyncCheckpointHandler(checkpoint_handler.CheckpointHandler):
    "Base class for asynchronous handlers."

    def async_save(
        self, directory: Any, *args: Any, **kwargs: Any
    ) -> Optional[List[Any]]:
        "Asynchronously save an item."
        pass

    def close(self) -> None:
        "Close the handler."
        pass

    def finalize(self, directory: Any) -> None:
        "Finalize the checkpoint."
        pass

    def metadata(self, directory: Any) -> Optional[Any]:
        "Get metadata."
        return None

    def restore(self, directory: Any, *args: Any, **kwargs: Any) -> Any:
        "Restore an item."
        return None

    def save(self, directory: Any, *args: Any, **kwargs: Any) -> None:
        "Save an item synchronously."
        pass


class ArrayCheckpointHandler(AsyncCheckpointHandler):
    "Handler for array checkpoints."

    def __init__(self, checkpoint_name: Optional[str] = None) -> None:
        "Initialize."
        self.checkpoint_name = checkpoint_name


class BasePyTreeCheckpointHandler(AsyncCheckpointHandler):
    "Base handler for PyTree checkpoints."

    def __init__(
        self,
        *,
        save_concurrent_bytes: Optional[int] = None,
        restore_concurrent_bytes: Optional[int] = None,
        use_ocdbt: bool = True,
        use_zarr3: bool = False,
        multiprocessing_options: Any = None,
        type_handler_registry: Any = None,
        enable_post_merge_validation: bool = True,
    ) -> None:
        "Initialize."
        pass

    def get_param_names(self, item: Any) -> Any:
        "Get parameter names."
        return None


class CompositeCheckpointHandler(AsyncCheckpointHandler):
    "Handler for composite checkpoints."

    def __init__(
        self,
        *item_names: str,
        composite_options: Any = None,
        handler_registry: Any = None,
        **items_and_handlers: Any,
    ) -> None:
        "Initialize."
        pass


class JaxRandomKeyCheckpointHandler(AsyncCheckpointHandler):
    "Handler for JAX random keys."

    def __init__(self, key_name: Optional[str] = None) -> None:
        "Initialize."
        self.key_name = key_name

    def checkpoint_restore_args(self, args: Any) -> Any:
        "Get restore arguments."
        return None

    def checkpoint_save_args(self, args: Any) -> Any:
        "Get save arguments."
        return (None, None)

    def post_restore(self, item: Any, metadata: Any) -> Any:
        "Post-restore hook."
        return item


class JsonCheckpointHandler(AsyncCheckpointHandler):
    "Handler for JSON checkpoints."

    def __init__(
        self, filename: Optional[str] = None, *, multiprocessing_options: Any = None
    ) -> None:
        "Initialize."
        self.filename = filename


class NumpyRandomKeyCheckpointHandler(AsyncCheckpointHandler):
    "Handler for NumPy random keys."

    def __init__(self, key_name: Optional[str] = None) -> None:
        "Initialize."
        self.key_name = key_name

    def checkpoint_restore_args(self, args: Any) -> Any:
        "Get restore arguments."
        return None

    def checkpoint_save_args(self, args: Any) -> Any:
        "Get save arguments."
        return (None, None)

    def post_restore(self, item: Any, metadata: Any) -> Any:
        "Post-restore hook."
        return item


class ProtoCheckpointHandler(AsyncCheckpointHandler):
    "Handler for Proto checkpoints."

    def __init__(self, filename: str, *, multiprocessing_options: Any = None) -> None:
        "Initialize."
        self.filename = filename


class PyTreeCheckpointHandler(AsyncCheckpointHandler):
    "Handler for PyTree checkpoints."

    def __init__(
        self,
        aggregate_filename: Optional[str] = None,
        *,
        save_concurrent_gb: Optional[int] = None,
        restore_concurrent_gb: Optional[int] = None,
        use_ocdbt: bool = True,
        use_zarr3: bool = False,
        multiprocessing_options: Any = None,
        type_handler_registry: Any = None,
        handler_impl: Optional[Any] = None,
    ) -> None:
        "Initialize."
        pass


class StandardCheckpointHandler(AsyncCheckpointHandler):
    "Standard checkpoint handler."

    def __init__(
        self,
        *,
        save_concurrent_gb: int = 96,
        restore_concurrent_gb: int = 96,
        multiprocessing_options: Any = None,
    ) -> None:
        "Initialize."
        pass


class DefaultCheckpointHandlerRegistry:
    "Registry for checkpoint handlers."

    def __init__(self, other_registry: Optional[Any] = None) -> None:
        "Initialize."
        self.other_registry = other_registry

    def add(self, item: Optional[str], args: Any, handler: Any) -> None:
        "Add a handler to the registry."
        pass

    def get(self, item: Optional[str], args: Any) -> Any:
        "Get a handler from the registry."
        return None

    def get_all_entries(self) -> Any:
        "Get all entries."
        return {}

    def has(self, item: Optional[str], args: Any) -> bool:
        "Check if an item exists in the registry."
        return False


options = orbax.checkpoint.options
