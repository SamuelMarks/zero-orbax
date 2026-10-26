"""Provide module-level functionality for checkpointing.

This module provides strongly typed, zero-dependency classes and utilities to
save and restore machine learning models, PyTrees, arrays, and metadata in an
Orbax-compatible manner without third-party runtime dependencies.
"""

from __future__ import annotations

import asyncio as _stdlib_asyncio
import contextlib as _stdlib_contextlib
import dataclasses
import datetime
import functools as _stdlib_functools
import json
import logging as _stdlib_logging
import pathlib
import re
import shutil
import time
from collections.abc import Callable, Container, Iterator, Mapping, Sequence
from types import TracebackType
from typing import Literal, TypeVar
from typing_extensions import TypeAlias

from ml_switcheroo_compiler import ops as compiler_ops
from ml_switcheroo_compiler.core.dtype import DType as CompilerDType
from ml_switcheroo_compiler.core.tensor import Tensor as CompilerTensor, as_tensor_obj
from ml_switcheroo_ir import DType, LogicalMesh, PartitionSpec
from zero_flax.nnx.variables import (
    BatchStat,
    Cache,
    Intermediate,
    LoRAParam,
    Param,
    Rng,
    RngState,
    Variable as FlaxVariable,
    VariableValue,
)
from zero_flax.types import MetadataValue
from zero_jax import numpy as jnp

from zero_orbax import export as _export_module
from zero_orbax.core import (
    AtomicWrite,
    BarrierSync,
    FileSystemIO,
    GlobalArraySharding,
    MsgpackEngine,
    SerializedArray,
    TensorStoreDriver,
)
from zero_orbax.types import (
    AbstractLoggerProtocol,
    ArrayMetadataValidatorProtocol,
    AsyncOptions,
    AtomicityOptions,
    BarrierSyncFn,
    BestFn,
    CheckpointMetadataStoreProtocol,
    CheckpointStateDict,
    CompositeOptions,
    FileOptions,
    FlaxStateProtocol,
    FlaxVariableProtocol,
    FrozenDictProtocol,
    GrainIteratorProtocol,
    IsPrioritizedKeyFn,
    JsonValue,
    MemoryLimitOptions,
    MultiValueTransformFn,
    MultiprocessingOptions,
    NamedTupleProtocol,
    PathLike,
    PreservationPolicyProtocol,
    PyTree,
    PyTreeMetadataOptions,
    SaveDecisionPolicyProtocol,
    ShouldKeepFn,
    ShouldSaveFn,
    TemporaryPathProtocol,
    TypeHandlerProtocol,
    TypeHandlerRegistryProtocol,
    ValueTransformFn,
    array_to_nested_list,
    is_array_protocol,
    is_pytree_dict,
    is_pytree_mapping,
    normalize_path,
)

__version__: str = "0.6.4"

# Compatibility re-exports verified by snapshot compliance
asyncio = _stdlib_asyncio
contextlib = _stdlib_contextlib
functools = _stdlib_functools
logging = _stdlib_logging

NoneType: TypeAlias = None
ValueFn: TypeAlias = ValueTransformFn
MultiValueFn: TypeAlias = MultiValueTransformFn


class _CheckpointArgsBase:
    """Base class for all structured checkpoint handler argument wrappers."""

    def __init__(self, item: PyTree = None) -> None:
        """Initialize the base CheckpointArgs wrapper.

        Args:
            item: Optional item payload wrapped by these arguments. Defaults to None.
        """
        self.item = item


_ArgsClsT = TypeVar("_ArgsClsT", bound=type[_CheckpointArgsBase])
_RegisteredHandlerCls: TypeAlias = type["CheckpointHandler"]
_RegisteredArgsCls: TypeAlias = type[_CheckpointArgsBase]


class _ArgsHandlerCatalog:
    """Encapsulated catalog mapping ``CheckpointArgs`` classes to ``CheckpointHandler`` classes."""

    def __init__(self) -> None:
        """Initialize the empty catalog mappings."""
        self.save_args_to_handler: dict[_RegisteredArgsCls, _RegisteredHandlerCls] = {}
        self.restore_args_to_handler: dict[_RegisteredArgsCls, _RegisteredHandlerCls] = {}
        self.handler_to_save_args: dict[_RegisteredHandlerCls, _RegisteredArgsCls] = {}
        self.handler_to_restore_args: dict[_RegisteredHandlerCls, _RegisteredArgsCls] = {}

    def register(
        self,
        args_cls: _RegisteredArgsCls,
        handler_cls: _RegisteredHandlerCls,
        *,
        for_save: bool = False,
        for_restore: bool = False,
    ) -> None:
        """Register an association between ``args_cls`` and ``handler_cls``.

        Args:
            args_cls: ``CheckpointArgs`` subclass.
            handler_cls: ``CheckpointHandler`` subclass.
            for_save: Whether ``args_cls`` is used for save operations. Defaults to False.
            for_restore: Whether ``args_cls`` is used for restore operations. Defaults to False.
        """
        if for_save:
            self.save_args_to_handler[args_cls] = handler_cls
            self.handler_to_save_args[handler_cls] = args_cls
        if for_restore:
            self.restore_args_to_handler[args_cls] = handler_cls
            self.handler_to_restore_args[handler_cls] = args_cls

    def get_handler_cls(
        self,
        arg_or_cls: _CheckpointArgsBase | type,
    ) -> _RegisteredHandlerCls | None:
        """Look up the registered ``CheckpointHandler`` subclass for ``arg_or_cls``.

        Args:
            arg_or_cls: ``CheckpointArgs`` instance or subclass.

        Returns:
            _RegisteredHandlerCls | None: Registered handler class, or None if not found.
        """
        target_cls: type = arg_or_cls if isinstance(arg_or_cls, type) else type(arg_or_cls)
        if issubclass(target_cls, _CheckpointArgsBase):
            if target_cls in self.save_args_to_handler:
                return self.save_args_to_handler[target_cls]
            if target_cls in self.restore_args_to_handler:
                return self.restore_args_to_handler[target_cls]
        return None

    def get_args_cls(
        self,
        handler_or_cls: CheckpointHandler | type,
        *,
        for_save: bool = True,
    ) -> _RegisteredArgsCls | None:
        """Look up the registered ``CheckpointArgs`` subclass for ``handler_or_cls``.

        Args:
            handler_or_cls: ``CheckpointHandler`` instance or subclass.
            for_save: Whether to query save args or restore args. Defaults to True.

        Returns:
            _RegisteredArgsCls | None: Registered ``CheckpointArgs`` subclass, or None.
        """
        target_cls: type = (
            handler_or_cls if isinstance(handler_or_cls, type) else type(handler_or_cls)
        )
        if target_cls.__module__.startswith("zero_grain"):
            target_cls = PyGrainCheckpointHandler
        if issubclass(target_cls, CheckpointHandler):
            if for_save and target_cls in self.handler_to_save_args:
                return self.handler_to_save_args[target_cls]
            if not for_save and target_cls in self.handler_to_restore_args:
                return self.handler_to_restore_args[target_cls]
        return None

    def has_registered(
        self, handler_or_cls: CheckpointHandler | _CheckpointArgsBase | type
    ) -> bool:
        """Return whether ``handler_or_cls`` is registered for save or restore.

        Args:
            handler_or_cls: Handler or args class/instance.

        Returns:
            bool: True if registered.
        """
        target_cls: type = (
            handler_or_cls if isinstance(handler_or_cls, type) else type(handler_or_cls)
        )
        if target_cls.__module__.startswith("zero_grain"):
            target_cls = PyGrainCheckpointHandler
        return (
            target_cls in self.handler_to_save_args
            or target_cls in self.handler_to_restore_args
            or target_cls in self.save_args_to_handler
            or target_cls in self.restore_args_to_handler
        )


_CATALOG = _ArgsHandlerCatalog()


@dataclasses.dataclass
class SaveArgs:
    """Arguments configuring how an individual leaf or PyTree node is saved.

    Attributes:
        aggregate: Whether to aggregate the leaf into a consolidated msgpack file. Defaults to False.
        dtype: Optional target data type to cast to when saving. Defaults to None.
        chunk_byte_size: Optional target byte size per storage chunk. Defaults to None.
        ocdbt_target_data_file_size: Optional target OCDBT data file size in bytes. Defaults to None.
        shard_axes: Tuple of axis indices along which to shard. Defaults to ``()``.
    """

    aggregate: bool = False
    dtype: str | DType | type | None = None
    chunk_byte_size: int | None = None
    ocdbt_target_data_file_size: int | None = None
    shard_axes: tuple[int, ...] = ()


@dataclasses.dataclass
class RestoreArgs:
    """Arguments configuring how an individual leaf or PyTree node is restored.

    Attributes:
        restore_type: Target Python type or type identifier for restoration. Defaults to None.
        dtype: Target element data type for restoration. Defaults to None.
        strict: Whether to enforce strict restoration validation. Defaults to True.
    """

    restore_type: type | str | None = None
    dtype: str | DType | type | None = None
    strict: bool = True


@dataclasses.dataclass
class ArrayRestoreArgs:
    """Arguments configuring how an array or sharded tensor is restored."""

    _restore_type: type | str | None = None
    mesh: LogicalMesh | None = None
    mesh_axes: PartitionSpec | Sequence[str | None] | None = None
    sharding: PartitionSpec | str | None = None
    global_shape: tuple[int, ...] | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | DType | type | None = None
    strict: bool = True

    def __init__(
        self,
        restore_type: type | str | None = None,
        mesh: LogicalMesh | None = None,
        mesh_axes: PartitionSpec | Sequence[str | None] | None = None,
        sharding: PartitionSpec | str | None = None,
        global_shape: tuple[int, ...] | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: str | DType | type | None = None,
        strict: bool = True,
    ) -> None:
        """Initialize the ArrayRestoreArgs instance.

        Args:
            restore_type: Target array type or identifier for restoration. Defaults to None.
            mesh: Optional ``LogicalMesh`` device topology. Defaults to None.
            mesh_axes: Optional ``PartitionSpec`` or axis names for mesh sharding. Defaults to None.
            sharding: Optional explicit sharding descriptor. Defaults to None.
            global_shape: Optional global array shape across all hosts. Defaults to None.
            shape: Optional local or target array shape. Defaults to None.
            dtype: Optional target element data type. Defaults to None.
            strict: Whether to strictly validate shape and sharding metadata. Defaults to True.
        """
        self._restore_type = restore_type
        self.mesh = mesh
        self.mesh_axes = mesh_axes
        self.sharding = sharding
        self.global_shape = global_shape
        self.shape = shape
        self.dtype = dtype
        self.strict = strict

    def restore_type(self) -> type | str | None:
        """Return the configured restore type for the array.

        Returns:
            type | str | None: The configured restore type.
        """
        return self._restore_type


@dataclasses.dataclass
class SingleReplicaArrayRestoreArgs(ArrayRestoreArgs):
    """Restore arguments reading array shards on a single replica and broadcasting across replicas."""

    single_replica_sharding: PartitionSpec | str | None = None

    def __init__(
        self,
        restore_type: type | str | None = None,
        mesh: LogicalMesh | None = None,
        mesh_axes: PartitionSpec | Sequence[str | None] | None = None,
        sharding: PartitionSpec | str | None = None,
        global_shape: tuple[int, ...] | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: str | DType | type | None = None,
        strict: bool = True,
        single_replica_sharding: PartitionSpec | str | None = None,
    ) -> None:
        """Initialize the SingleReplicaArrayRestoreArgs instance.

        Args:
            restore_type: Target array type or identifier for restoration. Defaults to None.
            mesh: Optional ``LogicalMesh`` device topology. Defaults to None.
            mesh_axes: Optional ``PartitionSpec`` or axis names for mesh sharding. Defaults to None.
            sharding: Optional explicit sharding descriptor. Defaults to None.
            global_shape: Optional global array shape across all hosts. Defaults to None.
            shape: Optional local or target array shape. Defaults to None.
            dtype: Optional target element data type. Defaults to None.
            strict: Whether to strictly validate shape and sharding metadata. Defaults to True.
            single_replica_sharding: Optional sharding on the single reading replica. Defaults to None.
        """
        super().__init__(
            restore_type=restore_type,
            mesh=mesh,
            mesh_axes=mesh_axes,
            sharding=sharding,
            global_shape=global_shape,
            shape=shape,
            dtype=dtype,
            strict=strict,
        )
        self.single_replica_sharding = single_replica_sharding


class args:
    """Namespace for checkpoint argument classes and handler registration utilities.

    Contains structured argument wrappers that specify how different types of
    checkpoint items should be saved or restored, along with handler registration
    lookup helpers.
    """

    class CheckpointArgs(_CheckpointArgsBase):
        """Base class for all structured checkpoint handler argument wrappers."""

    class ArraySave(CheckpointArgs):
        """Arguments for saving a single array or tensor."""

        def __init__(
            self,
            item: PyTree = None,
            save_args: SaveArgs | None = None,
        ) -> None:
            """Initialize the ArraySave arguments.

            Args:
                item: The array or tensor to save. Defaults to None.
                save_args: Optional ``SaveArgs`` configuration. Defaults to None.
            """
            super().__init__(item=item)
            self.save_args = save_args

    class ArrayRestore(CheckpointArgs):
        """Arguments for restoring a single array or tensor."""

        def __init__(
            self,
            item: PyTree = None,
            restore_args: ArrayRestoreArgs | RestoreArgs | None = None,
        ) -> None:
            """Initialize the ArrayRestore arguments.

            Args:
                item: Optional template array or item structure. Defaults to None.
                restore_args: Optional ``ArrayRestoreArgs`` or ``RestoreArgs``. Defaults to None.
            """
            super().__init__(item=item)
            self.restore_args = restore_args

    class Composite(CheckpointArgs):
        """Arguments for saving or restoring multiple named items via ``CompositeCheckpointHandler``."""

        def __init__(
            self,
            **items: _CheckpointArgsBase | PyTree,
        ) -> None:
            """Initialize the Composite arguments mapping.

            Args:
                **items: Named checkpoint items or ``CheckpointArgs`` instances.
            """
            super().__init__(item=None)
            self._items: dict[str, _CheckpointArgsBase | PyTree] = dict(items)

        def __getitem__(self, key: str) -> _CheckpointArgsBase | PyTree:
            """Retrieve a named composite item by key.

            Args:
                key: Item name.

            Returns:
                _CheckpointArgsBase | PyTree: Associated argument wrapper or item.
            """
            return self._items[key]

        def keys(self) -> Sequence[str]:
            """Return the registered item keys in this composite argument set.

            Returns:
                Sequence[str]: List of item keys.
            """
            return list(self._items.keys())

        def values(self) -> Sequence[_CheckpointArgsBase | PyTree]:
            """Return the values stored in this composite argument set.

            Returns:
                Sequence[_CheckpointArgsBase | PyTree]: List of item values.
            """
            return list(self._items.values())

        def items(self) -> Sequence[tuple[str, _CheckpointArgsBase | PyTree]]:
            """Return ``(key, value)`` pairs stored in this composite argument set.

            Returns:
                Sequence[tuple[str, _CheckpointArgsBase | PyTree]]: List of key-value pairs.
            """
            return list(self._items.items())

        def __len__(self) -> int:
            """Return the number of items in the composite argument set.

            Returns:
                int: Number of composite items.
            """
            return len(self._items)

    class JsonSave(CheckpointArgs):
        """Arguments for saving a JSON-serializable dictionary or value."""

        def __init__(self, item: PyTree = None) -> None:
            """Initialize the JsonSave arguments.

            Args:
                item: The JSON-compatible item to save. Defaults to None.
            """
            super().__init__(item=item)

    class JsonRestore(CheckpointArgs):
        """Arguments for restoring from a JSON checkpoint file."""

        def __init__(self, item: PyTree = None) -> None:
            """Initialize the JsonRestore arguments.

            Args:
                item: Optional template structure for JSON restoration. Defaults to None.
            """
            super().__init__(item=item)

    class ProtoSave(CheckpointArgs):
        """Arguments for saving a Protocol Buffer message."""

        def __init__(
            self,
            item: PyTree = None,
            filename: str = "proto.pb",
        ) -> None:
            """Initialize the ProtoSave arguments.

            Args:
                item: Protocol Buffer message or representation to save. Defaults to None.
                filename: Target protobuf filename inside the step directory. Defaults to ``"proto.pb"``.
            """
            super().__init__(item=item)
            self.filename = filename

    class ProtoRestore(CheckpointArgs):
        """Arguments for restoring a Protocol Buffer message."""

        def __init__(
            self,
            item: PyTree = None,
            item_type: type | None = None,
            filename: str = "proto.pb",
        ) -> None:
            """Initialize the ProtoRestore arguments.

            Args:
                item: Optional template message instance. Defaults to None.
                item_type: Optional Protocol Buffer message class. Defaults to None.
                filename: Source protobuf filename inside the step directory. Defaults to ``"proto.pb"``.
            """
            super().__init__(item=item)
            self.item_type = item_type
            self.filename = filename

    class PyTreeSave(CheckpointArgs):
        """Arguments for saving a PyTree structure."""

        def __init__(
            self,
            item: PyTree = None,
            save_args: PyTree = None,
            ocdbt_target_data_file_size: int | None = None,
        ) -> None:
            """Initialize the PyTreeSave arguments.

            Args:
                item: PyTree structure to save. Defaults to None.
                save_args: Optional PyTree of ``SaveArgs``. Defaults to None.
                ocdbt_target_data_file_size: Optional target OCDBT file size in bytes. Defaults to None.
            """
            super().__init__(item=item)
            self.save_args = save_args
            self.ocdbt_target_data_file_size = ocdbt_target_data_file_size

    class PyTreeRestore(CheckpointArgs):
        """Arguments for restoring a PyTree structure."""

        def __init__(
            self,
            item: PyTree = None,
            save_args: PyTree = None,
            restore_args: PyTree = None,
            transforms: PyTree = None,
            transforms_default_to_original: bool = True,
        ) -> None:
            """Initialize the PyTreeRestore arguments.

            Args:
                item: Optional target PyTree structure. Defaults to None.
                save_args: Optional PyTree of ``SaveArgs``. Defaults to None.
                restore_args: Optional PyTree of ``RestoreArgs``. Defaults to None.
                transforms: Optional PyTree of ``Transform`` / ``RestoreTransform`` rules. Defaults to None.
                transforms_default_to_original: Whether untransformed keys default to original values. Defaults to True.
            """
            super().__init__(item=item)
            self.save_args = save_args
            self.restore_args = restore_args
            self.transforms = transforms
            self.transforms_default_to_original = transforms_default_to_original

    class StandardSave(CheckpointArgs):
        """Arguments for standard PyTree saving operations."""

        def __init__(
            self,
            item: PyTree = None,
            save_args: PyTree = None,
        ) -> None:
            """Initialize the StandardSave arguments.

            Args:
                item: PyTree structure to save. Defaults to None.
                save_args: Optional per-leaf ``SaveArgs`` tree. Defaults to None.
            """
            super().__init__(item=item)
            self.save_args = save_args

    class StandardRestore(CheckpointArgs):
        """Arguments for standard PyTree restoration operations."""

        def __init__(
            self,
            item: PyTree = None,
            restore_args: PyTree = None,
            strict: bool = True,
        ) -> None:
            """Initialize the StandardRestore arguments.

            Args:
                item: Optional target PyTree structure to restore into. Defaults to None.
                restore_args: Optional per-leaf ``RestoreArgs`` tree. Defaults to None.
                strict: Whether to enforce strict PyTree structure validation. Defaults to True.
            """
            super().__init__(item=item)
            self.restore_args = restore_args
            self.strict = strict

    class JaxRandomKeySave(CheckpointArgs):
        """Arguments for saving a JAX PRNG random key."""

        def __init__(self, item: PyTree = None) -> None:
            """Initialize the JaxRandomKeySave arguments.

            Args:
                item: JAX PRNG key array or state to save. Defaults to None.
            """
            super().__init__(item=item)

    class JaxRandomKeyRestore(CheckpointArgs):
        """Arguments for restoring a JAX PRNG random key."""

        def __init__(self, item: PyTree = None) -> None:
            """Initialize the JaxRandomKeyRestore arguments.

            Args:
                item: Optional template JAX PRNG key. Defaults to None.
            """
            super().__init__(item=item)

    class NumpyRandomKeySave(CheckpointArgs):
        """Arguments for saving a NumPy random generator state."""

        def __init__(self, item: PyTree = None) -> None:
            """Initialize the NumpyRandomKeySave arguments.

            Args:
                item: NumPy random key or state dictionary to save. Defaults to None.
            """
            super().__init__(item=item)

    class NumpyRandomKeyRestore(CheckpointArgs):
        """Arguments for restoring a NumPy random generator state."""

        def __init__(self, item: PyTree = None) -> None:
            """Initialize the NumpyRandomKeyRestore arguments.

            Args:
                item: Optional template NumPy random state. Defaults to None.
            """
            super().__init__(item=item)

    class PyGrainCheckpointSave(CheckpointArgs):
        """Arguments for saving a ``zero_grain`` dataset or data loader iterator."""

        def __init__(self, item: PyTree = None) -> None:
            """Initialize the PyGrainCheckpointSave arguments.

            Args:
                item: ``GrainIteratorProtocol`` instance to checkpoint. Defaults to None.
            """
            super().__init__(item=item)

    class PyGrainCheckpointRestore(CheckpointArgs):
        """Arguments for restoring a ``zero_grain`` dataset or data loader iterator."""

        def __init__(self, item: PyTree = None) -> None:
            """Initialize the PyGrainCheckpointRestore arguments.

            Args:
                item: ``GrainIteratorProtocol`` instance to restore in-place. Defaults to None.
            """
            super().__init__(item=item)

    @staticmethod
    def get_registered_handler_cls(
        arg_or_cls: _CheckpointArgsBase | type,
    ) -> _RegisteredHandlerCls | None:
        """Retrieve the registered ``CheckpointHandler`` class for an args class or instance.

        Args:
            arg_or_cls: A ``CheckpointArgs`` instance or subclass.

        Returns:
            _RegisteredHandlerCls | None: The registered handler class, or ``None`` if not registered.
        """
        return _CATALOG.get_handler_cls(arg_or_cls)

    @staticmethod
    def get_registered_args_cls(
        handler_or_cls: CheckpointHandler | type,
        for_save: bool = True,
    ) -> _RegisteredArgsCls | None:
        """Retrieve the registered ``CheckpointArgs`` class for a handler.

        Args:
            handler_or_cls: A ``CheckpointHandler`` instance or class.
            for_save: If True, look up save args; otherwise look up restore args. Defaults to True.

        Returns:
            _RegisteredArgsCls | None: The registered ``CheckpointArgs`` class, or ``None`` if not registered.
        """
        return _CATALOG.get_args_cls(handler_or_cls, for_save=for_save)

    @staticmethod
    def has_registered_args(
        handler_or_cls: CheckpointHandler | _CheckpointArgsBase | type,
    ) -> bool:
        """Check whether a handler class or args class has registered arguments.

        Args:
            handler_or_cls: Handler class/instance or args class to check.

        Returns:
            bool: True if registered for save or restore, False otherwise.
        """
        return _CATALOG.has_registered(handler_or_cls)

    @staticmethod
    def register_with_handler(
        handler_cls: type[CheckpointHandler],
        for_save: bool = False,
        for_restore: bool = False,
    ) -> Callable[[_ArgsClsT], _ArgsClsT]:
        """Create a class decorator registering a ``CheckpointArgs`` class with a handler.

        Args:
            handler_cls: The ``CheckpointHandler`` class to associate with the decorated args class.
            for_save: Whether to register the decorated class for save operations. Defaults to False.
            for_restore: Whether to register the decorated class for restore operations. Defaults to False.

        Returns:
            Callable[[_ArgsClsT], _ArgsClsT]: Decorator that registers and returns the class unchanged.
        """

        def _decorator(target_cls: _ArgsClsT) -> _ArgsClsT:
            """Register ``target_cls`` in the catalog handler/args registries.

            Args:
                target_cls: The argument class being decorated.

            Returns:
                _ArgsClsT: The unmodified argument class.
            """
            _CATALOG.register(
                target_cls,
                handler_cls,
                for_save=for_save,
                for_restore=for_restore,
            )
            return target_cls

        return _decorator


_CompositeArgsCls = args.Composite
_PyTreeSaveArgsCls = args.PyTreeSave
_PyTreeRestoreArgsCls = args.PyTreeRestore
_StandardSaveArgsCls = args.StandardSave
_StandardRestoreArgsCls = args.StandardRestore
_ArrayRestoreArgsWrapperCls = args.ArrayRestore
_ProtoSaveArgsCls = args.ProtoSave
_ProtoRestoreArgsCls = args.ProtoRestore


class step_lib:
    """Namespace for step-related metadata and directory name formatting."""

    @dataclasses.dataclass(frozen=True)
    class Metadata:
        """Metadata associated with an individual checkpoint step.

        Attributes:
            step: Numeric training step.
            path: Directory path for the step checkpoint.
        """

        step: int = 0
        path: str = ""

    @dataclasses.dataclass
    class NameFormat:
        """Formatting rules for checkpoint step directory names.

        Attributes:
            step_prefix: Optional string prefix prepended to step numbers. Defaults to None.
            step_format_fixed_length: Optional zero-padding width for step numbers. Defaults to None.
        """

        step_prefix: str | None = None
        step_format_fixed_length: int | None = None

        def build_name(self, step_num: int) -> str:
            """Format a step number into its directory name string.

            Args:
                step_num: Step integer to format.

            Returns:
                str: Formatted directory name.
            """
            num_str = (
                f"{step_num:0{self.step_format_fixed_length}d}"
                if self.step_format_fixed_length is not None
                else str(step_num)
            )
            if self.step_prefix:
                return f"{self.step_prefix}_{num_str}"
            return num_str

        def parse_step(self, dir_name: str) -> int | None:
            """Parse a step integer from a formatted step directory name.

            Args:
                dir_name: Candidate directory name.

            Returns:
                int | None: Parsed step integer, or None if ``dir_name`` does not match.
            """
            if self.step_prefix:
                prefix_with_sep = f"{self.step_prefix}_"
                if not dir_name.startswith(prefix_with_sep):
                    return None
                num_part = dir_name[len(prefix_with_sep) :]
            else:
                num_part = dir_name
            if num_part.isdigit():
                return int(num_part)
            return None

    @dataclasses.dataclass
    class StandardNameFormat(NameFormat):
        """Standard step directory name format supporting single-host load and broadcast.

        Attributes:
            single_host_load_and_broadcast: Whether to format for single-host load and broadcast.
            step_prefix: Optional string prefix prepended to step numbers.
            step_format_fixed_length: Optional zero-padding width for step numbers.
        """

        single_host_load_and_broadcast: bool = False
        step_prefix: str | None = None
        step_format_fixed_length: int | None = None

    @staticmethod
    def standard_name_format(
        single_host_load_and_broadcast: bool = False,
        step_prefix: str | None = None,
        step_format_fixed_length: int | None = None,
    ) -> step_lib.StandardNameFormat:
        """Create a standard step ``StandardNameFormat`` instance.

        Args:
            single_host_load_and_broadcast: Whether to format for single-host load and broadcast. Defaults to False.
            step_prefix: Optional step directory name prefix. Defaults to None.
            step_format_fixed_length: Optional fixed digit width for step numbers. Defaults to None.

        Returns:
            step_lib.StandardNameFormat: Configured ``StandardNameFormat`` instance.
        """
        return step_lib.StandardNameFormat(
            single_host_load_and_broadcast=single_host_load_and_broadcast,
            step_prefix=step_prefix,
            step_format_fixed_length=step_format_fixed_length,
        )


step = step_lib


class Future:
    """Representation of an in-flight or completed asynchronous checkpoint operation."""

    def __init__(
        self,
        result: PyTree = None,
        commit_callback: Callable[[], None] | None = None,
        *args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Initialize the Future wrapper.

        Args:
            result: The result value to wrap. Defaults to None.
            commit_callback: Optional callback invoked once when ``result()`` is awaited. Defaults to None.
            *args: Optional additional positional arguments.
            **kwargs: Optional additional keyword arguments.
        """
        _ = (args, kwargs)
        self._result = result
        self._commit_callback = commit_callback
        self._done = commit_callback is None

    def result(self, timeout: float | None = None) -> PyTree:
        """Wait for completion and return the wrapped result.

        Args:
            timeout: Optional timeout in seconds (must be non-negative if provided). Defaults to None.

        Returns:
            PyTree: The result stored in this future.

        Raises:
            ValueError: If ``timeout`` is negative.
        """
        if timeout is not None and timeout < 0:
            raise ValueError(f"Future timeout must be non-negative, got {timeout}.")
        if not self._done and self._commit_callback is not None:
            self._commit_callback()
            self._done = True
        return self._result

    def done(self) -> bool:
        """Return whether the asynchronous operation has completed.

        Returns:
            bool: True if finalized, False if a commit callback is still pending.
        """
        return self._done


def _unwrap_checkpoint_arg(
    item: PyTree,
    kwargs: Mapping[str, PyTree],
) -> PyTree:
    """Extract the underlying payload from ``item`` or ``kwargs['args']``.

    Args:
        item: Positional item or ``_CheckpointArgsBase`` wrapper.
        kwargs: Keyword arguments mapping potentially containing ``args``.

    Returns:
        PyTree: Extracted payload.
    """
    if isinstance(item, _CheckpointArgsBase):
        return item.item
    if item is None and "args" in kwargs:
        arg_obj = kwargs["args"]
        return arg_obj.item if isinstance(arg_obj, _CheckpointArgsBase) else arg_obj
    return item


class CheckpointHandler:
    """Base class for handlers that manage checkpoint reading, writing, and metadata."""

    def __init__(self, *args: PyTree, **kwargs: PyTree) -> None:
        """Initialize the base CheckpointHandler.

        Args:
            *args: Optional positional initialization arguments.
            **kwargs: Optional keyword initialization arguments.
        """
        _ = (args, kwargs)
        self._closed: bool = False
        self._last_metadata: dict[str, JsonValue] | None = None

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Save an item synchronously to the specified directory.

        Args:
            directory: Target checkpoint directory path. Defaults to None.
            item: The item or ``CheckpointArgs`` payload to save. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.
        """
        payload = _unwrap_checkpoint_arg(item, kwargs)
        if directory is not None:
            key = str(directory).rstrip("/")
            encoded = MsgpackEngine().packb(payload)
            AtomicWrite().atomic_write_bytes(f"{key}/checkpoint.msgpack", encoded)
            meta: dict[str, JsonValue] = {"directory": str(directory)}
            AtomicWrite().atomic_write_bytes(
                f"{key}/_orbax_metadata.json",
                json.dumps(meta).encode("utf-8"),
            )

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore an item from the specified directory.

        Args:
            directory: Source checkpoint directory path. Defaults to None.
            item: Optional target structure or ``CheckpointArgs`` wrapper. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: The restored PyTree or item structure.
        """
        template = _unwrap_checkpoint_arg(item, kwargs)
        if directory is not None:
            key = str(directory).rstrip("/")
            raw = FileSystemIO().read(f"{key}/checkpoint.msgpack")
            if raw:
                return MsgpackEngine().unpackb(raw)
        return template

    def metadata(self, directory: PathLike | None = None) -> dict[str, JsonValue] | None:
        """Return metadata associated with the checkpoint in ``directory``.

        Args:
            directory: Checkpoint directory path. Defaults to None.

        Returns:
            dict[str, JsonValue] | None: Metadata dictionary if available, otherwise None.
        """
        if directory is not None:
            key = str(directory).rstrip("/")
            raw = FileSystemIO().read(f"{key}/_orbax_metadata.json")
            if raw:
                loaded: dict[str, JsonValue] = json.loads(raw.decode("utf-8"))
                return loaded
        return self._last_metadata

    def structure(self, directory: PathLike | None = None) -> PyTree:
        """Return the structural skeleton of the saved checkpoint.

        Args:
            directory: Checkpoint directory path. Defaults to None.

        Returns:
            PyTree: Structure of the saved item, or None.
        """
        if directory is not None:
            return self.restore(directory)
        return None

    def finalize(self, directory: PathLike | None = None) -> None:
        """Finalize the checkpoint directory after writing completes.

        Args:
            directory: Checkpoint directory path to finalize. Defaults to None.
        """
        if directory is not None:
            key = str(directory).rstrip("/")
            meta: dict[str, JsonValue] = {"finalized": True, "directory": str(directory)}
            self._last_metadata = meta
            AtomicWrite().atomic_write_bytes(
                f"{key}/_orbax_metadata.json",
                json.dumps(meta).encode("utf-8"),
            )

    def close(self) -> None:
        """Release any resources held by the handler."""
        self._closed = True


class AsyncCheckpointHandler(CheckpointHandler):
    """Base class for checkpoint handlers supporting asynchronous save operations."""

    def __init__(self, *args: PyTree, **kwargs: PyTree) -> None:
        """Initialize the base AsyncCheckpointHandler.

        Args:
            *args: Optional positional initialization arguments.
            **kwargs: Optional keyword initialization arguments.
        """
        super().__init__(*args, **kwargs)

    def async_save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> list[Future] | None:
        """Initiate an asynchronous save operation and return commit futures.

        Args:
            directory: Target checkpoint directory path. Defaults to None.
            item: The item or ``CheckpointArgs`` to save. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.

        Returns:
            list[Future] | None: List of futures representing pending commit tasks, or None.
        """
        if directory is None:
            return None
        self.save(directory, item, *pos_args, **kwargs)
        return [Future(result=None, commit_callback=lambda: self.finalize(directory))]


def _infer_array_shape(val: PyTree) -> tuple[int, ...]:
    """Infer the shape tuple of an array-like object or nested sequence.

    Args:
        val: ArrayProtocol, SerializedArray, or nested sequence.

    Returns:
        tuple[int, ...]: Inferred shape tuple.
    """
    if is_array_protocol(val):
        return tuple(int(d) for d in val.shape)
    dims: list[int] = []
    curr: PyTree = val
    while isinstance(curr, (list, tuple)):
        dims.append(len(curr))
        if len(curr) == 0:
            break
        curr = curr[0]
    return tuple(dims)


class ArrayCheckpointHandler(AsyncCheckpointHandler):
    """Handler for saving and restoring single arrays or sharded tensors."""

    def __init__(
        self,
        checkpoint_name: str | None = None,
        *,
        multiprocessing_options: MultiprocessingOptions | None = None,
    ) -> None:
        """Initialize the ArrayCheckpointHandler.

        Args:
            checkpoint_name: Optional filename for the serialized array inside the step directory. Defaults to None.
            multiprocessing_options: Optional ``MultiprocessingOptions``. Defaults to None.
        """
        super().__init__()
        self.checkpoint_name = checkpoint_name
        self.multiprocessing_options = multiprocessing_options

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Save an array to ``directory`` using ``TensorStoreDriver``.

        Args:
            directory: Target directory path. Defaults to None.
            item: Array or ``args.ArraySave`` instance to save. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.
        """
        if directory is not None:
            name = self.checkpoint_name or "array"
            driver = TensorStoreDriver()
            driver.open({"driver": "zarr", "path": f"{str(directory).rstrip('/')}/{name}.ts"})
            payload = _unwrap_checkpoint_arg(item, kwargs)
            driver.write(MsgpackEngine().packb(payload))

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        restore_args: ArrayRestoreArgs | RestoreArgs | PyTree = None,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore an array from ``directory`` using ``TensorStoreDriver``.

        Args:
            directory: Source directory path. Defaults to None.
            item: Optional template array or ``args.ArrayRestore`` wrapper. Defaults to None.
            *pos_args: Additional positional restore arguments.
            restore_args: Optional ``ArrayRestoreArgs`` or ``RestoreArgs``. Defaults to None.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: Restored array or template value.

        Raises:
            ValueError: If ``strict=True`` on ``ArrayRestoreArgs`` and the restored shape mismatches.
        """
        active_restore_args = restore_args
        raw_arg = item if isinstance(item, _CheckpointArgsBase) else kwargs.get("args")
        if isinstance(raw_arg, _ArrayRestoreArgsWrapperCls) and active_restore_args is None:
            active_restore_args = raw_arg.restore_args
        template = _unwrap_checkpoint_arg(item, kwargs)
        if directory is not None:
            name = self.checkpoint_name or "array"
            ts_path = f"{str(directory).rstrip('/')}/{name}.ts"
            driver = TensorStoreDriver()
            driver.open({"driver": "zarr", "path": ts_path})
            raw = driver.read()
            if raw:
                restored = MsgpackEngine().unpackb(raw)
                if isinstance(active_restore_args, ArrayRestoreArgs):
                    expected_shape = active_restore_args.shape or active_restore_args.global_shape
                    if active_restore_args.strict and expected_shape is not None:
                        actual_shape = _infer_array_shape(restored)
                        if actual_shape != expected_shape:
                            raise ValueError(
                                f"Strict shape mismatch in ArrayCheckpointHandler: expected "
                                f"{expected_shape}, got {actual_shape}."
                            )
                    if (
                        active_restore_args.mesh is not None
                        and active_restore_args.mesh_axes is not None
                    ):
                        pspec = (
                            active_restore_args.mesh_axes
                            if isinstance(active_restore_args.mesh_axes, PartitionSpec)
                            else PartitionSpec(axes=tuple(active_restore_args.mesh_axes))
                        )
                        restored = GlobalArraySharding().get_local_shard(
                            restored,
                            host_id=0,
                            num_hosts=1,
                            partition_spec=pspec,
                            mesh=active_restore_args.mesh,
                        )
                    rtype = active_restore_args.restore_type()
                    if rtype is tuple and isinstance(restored, list):
                        return tuple(restored)
                return restored
            return super().restore(directory, template, *pos_args, **kwargs)
        return template


_COMPILER_DTYPE_MAP: dict[str, CompilerDType] = {member.name: member for member in CompilerDType}
_FLAX_VARIABLE_CLASSES: dict[str, type[FlaxVariable[VariableValue]]] = {
    "Param": Param,
    "BatchStat": BatchStat,
    "Cache": Cache,
    "Intermediate": Intermediate,
    "LoRAParam": LoRAParam,
    "Rng": Rng,
    "RngState": RngState,
    "Variable": FlaxVariable,
}


def _resolve_compiler_dtype(dtype_str: str) -> CompilerDType:
    """Resolve a string dtype identifier into a ``CompilerDType`` enum member.

    Args:
        dtype_str: String representation of the data type (e.g. ``"float32"`` or ``"DType.float32"``).

    Returns:
        CompilerDType: Corresponding compiler data type enum member.
    """
    clean = dtype_str[6:] if dtype_str.startswith("DType.") else dtype_str
    return _COMPILER_DTYPE_MAP.get(clean, CompilerDType.float32)


def _coerce_flax_variable_value(val: PyTree) -> VariableValue:
    """Coerce an unpacked PyTree leaf into a strongly typed ``VariableValue``.

    Args:
        val: Unpacked PyTree value for a Flax ``Variable``.

    Returns:
        VariableValue: Typed JAX array or scalar leaf.
    """
    if val is None or isinstance(val, (jnp.ndarray, int, float, bool, str)):
        return val
    return jnp.asarray(array_to_nested_list(val) if is_array_protocol(val) else val)


def _coerce_flax_metadata(meta: PyTree) -> dict[str, MetadataValue]:
    """Coerce an unpacked metadata mapping into a typed ``dict[str, MetadataValue]``.

    Args:
        meta: Raw unpacked metadata dictionary.

    Returns:
        dict[str, MetadataValue]: Strongly typed variable metadata mapping.
    """
    if not is_pytree_dict(meta):
        return {}
    return {
        str(k): v for k, v in meta.items() if v is None or isinstance(v, (int, float, bool, str))
    }


def _reconstruct_array_leaf(
    data: PyTree,
    shape: tuple[int, ...],
    dtype_str: str,
    array_kind: str,
    template: PyTree = None,
) -> PyTree:
    """Reconstruct a typed array leaf from serialized data, shape, and dtype.

    Args:
        data: Nested list or scalar data payload.
        shape: Target array shape tuple.
        dtype_str: Target data type name.
        array_kind: Recorded array kind (``"tensor"``, ``"serialized_array"``, or ``"jax_array"``).
        template: Optional template leaf guiding array type selection. Defaults to None.

    Returns:
        PyTree: Reconstructed ``zero_jax.numpy.ndarray``, ``CompilerTensor``, or ``SerializedArray``.
    """
    resolved_dt = _resolve_compiler_dtype(dtype_str)
    raw_input = data if isinstance(data, (int, float, bool, list, tuple)) else 0.0
    if isinstance(template, CompilerTensor) or (template is None and array_kind == "tensor"):
        return as_tensor_obj(compiler_ops.asarray(raw_input, dtype=resolved_dt))
    if isinstance(template, SerializedArray) or (
        template is None and array_kind == "serialized_array"
    ):
        return SerializedArray(data=data, shape=shape, dtype=dtype_str)
    return jnp.reshape(jnp.asarray(raw_input, dtype=resolved_dt), shape)


def _build_flax_state(entries: Mapping[str, PyTree], template: PyTree = None) -> PyTree:
    """Construct a ``zero_flax.nnx.State`` from unpacked entries and optionally update ``template``.

    Args:
        entries: Mapping of state keys to unpacked values.
        template: Optional target ``State`` template to update in-place. Defaults to None.

    Returns:
        PyTree: Reconstructed ``zero_flax.nnx.State`` instance.
    """
    from zero_flax.nnx.state import State as FlaxState, StateEntry

    typed_entries: dict[str, StateEntry] = {}
    for k, v in entries.items():
        if isinstance(v, (FlaxState, FlaxVariable)):
            typed_entries[str(k)] = v
        else:
            typed_entries[str(k)] = _coerce_flax_variable_value(v)
    state_obj = FlaxState(typed_entries)
    if isinstance(template, FlaxState):
        template.update(state_obj)
    return state_obj


def _build_frozen_dict(entries: Mapping[str, PyTree]) -> PyTree:
    """Construct a ``zero_flax.core.FrozenDict`` from unpacked entries.

    Args:
        entries: Mapping of keys to unpacked PyTree values.

    Returns:
        PyTree: Reconstructed ``FrozenDict`` instance.
    """
    from zero_flax.core import FrozenDict, FrozenEntry

    typed_entries: dict[str, FrozenEntry] = {}
    for k, v in entries.items():
        if isinstance(v, (FrozenDict, FlaxVariable)):
            typed_entries[str(k)] = v
        else:
            typed_entries[str(k)] = _coerce_flax_variable_value(v)
    return FrozenDict(typed_entries)


class BasePyTreeCheckpointHandler(AsyncCheckpointHandler):
    """Base checkpoint handler for structured PyTree serialization."""

    def __init__(
        self,
        *,
        save_concurrent_bytes: int | None = None,
        restore_concurrent_bytes: int | None = None,
        save_device_host_concurrent_bytes: int | None = None,
        memory_limit_options: MemoryLimitOptions | None = None,
        use_ocdbt: bool = True,
        use_zarr3: bool = False,
        use_compression: bool = True,
        multiprocessing_options: MultiprocessingOptions | None = None,
        type_handler_registry: TypeHandlerRegistryProtocol | None = None,
        enable_post_merge_validation: bool = True,
        pytree_metadata_options: PyTreeMetadataOptions | None = None,
        array_metadata_validator: ArrayMetadataValidatorProtocol | None = None,
        enable_pinned_host_transfer: bool | None = None,
        is_prioritized_key_fn: IsPrioritizedKeyFn | None = None,
    ) -> None:
        """Initialize the BasePyTreeCheckpointHandler.

        Args:
            save_concurrent_bytes: Maximum concurrent bytes during save. Defaults to None.
            restore_concurrent_bytes: Maximum concurrent bytes during restore. Defaults to None.
            save_device_host_concurrent_bytes: Max concurrent device-to-host bytes. Defaults to None.
            memory_limit_options: Optional ``MemoryLimitOptions`` configuration. Defaults to None.
            use_ocdbt: Whether to use OCDBT storage driver. Defaults to True.
            use_zarr3: Whether to use Zarr v3 format. Defaults to False.
            use_compression: Whether to compress stored chunks. Defaults to True.
            multiprocessing_options: Optional ``MultiprocessingOptions``. Defaults to None.
            type_handler_registry: Optional leaf ``TypeHandlerRegistryProtocol``. Defaults to None.
            enable_post_merge_validation: Whether to validate shards after OCDBT merge. Defaults to True.
            pytree_metadata_options: Optional ``PyTreeMetadataOptions``. Defaults to None.
            array_metadata_validator: Optional ``ArrayMetadataValidatorProtocol``. Defaults to None.
            enable_pinned_host_transfer: Whether to use pinned host memory transfers. Defaults to None.
            is_prioritized_key_fn: Optional callback prioritizing specific key paths. Defaults to None.
        """
        super().__init__()
        self.save_concurrent_bytes = save_concurrent_bytes
        self.restore_concurrent_bytes = restore_concurrent_bytes
        self.save_device_host_concurrent_bytes = save_device_host_concurrent_bytes
        self.memory_limit_options = memory_limit_options
        self.use_ocdbt = use_ocdbt
        self.use_zarr3 = use_zarr3
        self.use_compression = use_compression
        self.multiprocessing_options = multiprocessing_options
        self.type_handler_registry = type_handler_registry
        self.enable_post_merge_validation = enable_post_merge_validation
        self.pytree_metadata_options = pytree_metadata_options
        self.array_metadata_validator = array_metadata_validator
        self.enable_pinned_host_transfer = enable_pinned_host_transfer
        self.is_prioritized_key_fn = is_prioritized_key_fn

    def _pack_pytree(self, node: PyTree) -> PyTree:
        """Pack rich ecosystem PyTree nodes into tagged serializable dictionaries.

        Args:
            node: Input PyTree node (``Variable``, ``State``, ``FrozenDict``, ``OptState``, array, or container).

        Returns:
            PyTree: Tagged PyTree ready for MessagePack serialization.
        """
        if isinstance(node, FlaxVariableProtocol):
            return {
                "__zero_flax_variable__": True,
                "var_type": type(node).__name__,
                "value": self._pack_pytree(node.value),
                "metadata": self._pack_pytree(dict(node.metadata)),
            }
        if isinstance(node, FlaxStateProtocol):
            return {
                "__zero_flax_state__": True,
                "entries": {str(k): self._pack_pytree(v) for k, v in node.items()},
            }
        if isinstance(node, FrozenDictProtocol):
            return {
                "__zero_flax_frozendict__": True,
                "entries": {str(k): self._pack_pytree(v) for k, v in node.items()},
            }
        if isinstance(node, NamedTupleProtocol):
            field_dict = node._asdict()
            return {
                "__zero_optax_opt_state__": True,
                "state_type": type(node).__name__,
                "fields": [str(f) for f in node._fields],
                "values": {str(f): self._pack_pytree(field_dict.get(f)) for f in node._fields},
            }
        if is_array_protocol(node):
            array_kind = (
                "tensor"
                if isinstance(node, CompilerTensor)
                else ("serialized_array" if isinstance(node, SerializedArray) else "jax_array")
            )
            return {
                "__zero_orbax_array__": True,
                "array_kind": array_kind,
                "shape": [int(d) for d in node.shape],
                "dtype": str(node.dtype),
                "data": array_to_nested_list(node),
            }
        if isinstance(node, tuple):
            return tuple(self._pack_pytree(item) for item in node)
        if isinstance(node, list):
            return [self._pack_pytree(item) for item in node]
        if is_pytree_mapping(node):
            return {str(k): self._pack_pytree(v) for k, v in node.items()}
        return node

    def _unpack_pytree(self, node: PyTree, template: PyTree = None) -> PyTree:
        """Unpack tagged PyTree dictionaries back into rich ecosystem containers and arrays.

        Args:
            node: Deserialized PyTree node.
            template: Optional target template guiding container and array reconstruction. Defaults to None.

        Returns:
            PyTree: Reconstructed ecosystem PyTree.
        """
        if isinstance(node, FlaxStateProtocol):
            tmpl_map = (
                dict(template.items())
                if isinstance(template, FlaxStateProtocol)
                else (template if is_pytree_dict(template) else {})
            )
            return _build_flax_state(
                {str(k): self._unpack_pytree(v, tmpl_map.get(str(k))) for k, v in node.items()},
                template,
            )
        if is_pytree_dict(node):
            if node.get("__zero_flax_variable__") is True:
                var_type = str(node.get("var_type", "Variable"))
                tmpl_val = template.value if isinstance(template, FlaxVariableProtocol) else None
                val = self._unpack_pytree(node.get("value"), tmpl_val)
                raw_meta = self._unpack_pytree(node.get("metadata"), None)
                coerced_val = _coerce_flax_variable_value(val)
                coerced_meta = _coerce_flax_metadata(raw_meta)
                if isinstance(template, FlaxVariableProtocol):
                    template.value = coerced_val
                    return template.replace(value=coerced_val, **coerced_meta)
                var_cls = _FLAX_VARIABLE_CLASSES.get(var_type, FlaxVariable)
                return var_cls(coerced_val, **coerced_meta)
            if node.get("__zero_flax_state__") is True:
                raw_entries = node.get("entries")
                entries_map = raw_entries if is_pytree_dict(raw_entries) else {}
                tmpl_map = (
                    dict(template.items())
                    if isinstance(template, FlaxStateProtocol)
                    else (template if is_pytree_dict(template) else {})
                )
                ordered_keys = [*tmpl_map.keys(), *(k for k in entries_map if k not in tmpl_map)]
                unpacked_entries = {
                    str(k): self._unpack_pytree(entries_map[k], tmpl_map.get(str(k)))
                    for k in ordered_keys
                    if k in entries_map
                }
                return _build_flax_state(unpacked_entries, template)
            if node.get("__zero_flax_frozendict__") is True:
                raw_entries = node.get("entries")
                entries_map = raw_entries if is_pytree_dict(raw_entries) else {}
                tmpl_map = (
                    dict(template.items())
                    if isinstance(template, FrozenDictProtocol)
                    else (template if is_pytree_dict(template) else {})
                )
                ordered_keys = [*tmpl_map.keys(), *(k for k in entries_map if k not in tmpl_map)]
                unpacked_entries = {
                    str(k): self._unpack_pytree(entries_map[k], tmpl_map.get(str(k)))
                    for k in ordered_keys
                    if k in entries_map
                }
                return _build_frozen_dict(unpacked_entries)
            if node.get("__zero_optax_opt_state__") is True:
                raw_fields = node.get("fields")
                raw_values = node.get("values")
                field_names = (
                    [str(f) for f in raw_fields] if isinstance(raw_fields, (list, tuple)) else []
                )
                values_map = raw_values if is_pytree_dict(raw_values) else {}
                tmpl_fields = (
                    dict(template._asdict())
                    if isinstance(template, NamedTupleProtocol)
                    else (template if is_pytree_dict(template) else {})
                )
                unpacked_vals = [
                    self._unpack_pytree(values_map.get(f), tmpl_fields.get(f)) for f in field_names
                ]
                if isinstance(template, NamedTupleProtocol):
                    return template._make(unpacked_vals)
                return {f: v for f, v in zip(field_names, unpacked_vals)}
            if node.get("__zero_orbax_array__") is True:
                raw_shape = node.get("shape")
                shape_tuple = (
                    tuple(int(d) for d in raw_shape if isinstance(d, int))
                    if isinstance(raw_shape, (list, tuple))
                    else ()
                )
                dtype_str = str(node.get("dtype", "float32"))
                array_kind = str(node.get("array_kind", "jax_array"))
                return _reconstruct_array_leaf(
                    node.get("data"),
                    shape_tuple,
                    dtype_str,
                    array_kind,
                    template,
                )
            if isinstance(template, FlaxStateProtocol):
                tmpl_map = dict(template.items())
                unpacked_s = {
                    str(k): self._unpack_pytree(v, tmpl_map.get(str(k))) for k, v in node.items()
                }
                return _build_flax_state(unpacked_s, template)
            if isinstance(template, FrozenDictProtocol):
                tmpl_map = dict(template.items())
                unpacked_fd = {
                    str(k): self._unpack_pytree(v, tmpl_map.get(str(k))) for k, v in node.items()
                }
                return _build_frozen_dict(unpacked_fd)
            if isinstance(template, NamedTupleProtocol):
                tmpl_map = dict(template._asdict())
                vals = [self._unpack_pytree(node.get(f), tmpl_map.get(f)) for f in template._fields]
                return template._make(vals)
            tmpl_dict = template if is_pytree_dict(template) else {}
            return {k: self._unpack_pytree(v, tmpl_dict.get(k)) for k, v in node.items()}
        if isinstance(node, SerializedArray):
            return _reconstruct_array_leaf(
                node.tolist(),
                node.shape,
                str(node.dtype),
                "jax_array",
                template,
            )
        if isinstance(node, tuple):
            tmpl_seq = template if isinstance(template, (list, tuple)) else ()
            return tuple(
                self._unpack_pytree(item, tmpl_seq[idx] if idx < len(tmpl_seq) else None)
                for idx, item in enumerate(node)
            )
        if isinstance(node, list):
            tmpl_seq = template if isinstance(template, (list, tuple)) else ()
            return [
                self._unpack_pytree(item, tmpl_seq[idx] if idx < len(tmpl_seq) else None)
                for idx, item in enumerate(node)
            ]
        return node

    def _extract_structure(self, node: PyTree) -> PyTree:
        """Extract the unpacked structural tree of a saved PyTree payload.

        Args:
            node: Packed or unpacked PyTree node.

        Returns:
            PyTree: Reconstructed structural tree.
        """
        return self._unpack_pytree(node, None)

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Save a PyTree to ``directory`` after packing rich ecosystem nodes.

        Args:
            directory: Target checkpoint directory path. Defaults to None.
            item: PyTree or ``CheckpointArgs`` payload to save. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.
        """
        payload = _unwrap_checkpoint_arg(item, kwargs)
        packed = self._pack_pytree(payload)
        super().save(directory, packed, *pos_args, **kwargs)

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore and unpack a PyTree from ``directory``.

        Args:
            directory: Source checkpoint directory path. Defaults to None.
            item: Optional target template or ``CheckpointArgs`` wrapper. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: Unpacked PyTree structure or fallback template.
        """
        template = _unwrap_checkpoint_arg(item, kwargs)
        sentinel = "__zero_orbax_missing__"
        raw_restored = super().restore(directory, sentinel, *pos_args, **kwargs)
        if raw_restored == sentinel:
            return template
        return self._unpack_pytree(raw_restored, template)

    def structure(self, directory: PathLike | None = None) -> PyTree:
        """Return the structural skeleton of the saved PyTree checkpoint.

        Args:
            directory: Checkpoint directory path. Defaults to None.

        Returns:
            PyTree: Extracted PyTree structure, or None.
        """
        return self._extract_structure(super().structure(directory))

    def get_param_names(self, item: PyTree) -> PyTree:
        """Derive dotted parameter names for each leaf in a PyTree dictionary.

        Args:
            item: PyTree mapping to extract parameter names from.

        Returns:
            PyTree: Nested dictionary of dotted parameter names, or ``None`` if ``item`` is not a mapping.
        """
        if not is_pytree_mapping(item):
            return None

        def _build_names(subtree: Mapping[str, PyTree], prefix: str) -> dict[str, PyTree]:
            """Recursively construct dotted parameter names for a mapping subtree.

            Args:
                subtree: Current dictionary subtree.
                prefix: Accumulated dotted key prefix.

            Returns:
                dict[str, PyTree]: Nested dictionary of parameter name strings.
            """
            result: dict[str, PyTree] = {}
            for k, v in subtree.items():
                full_key = f"{prefix}.{k}" if prefix else str(k)
                if is_pytree_mapping(v):
                    result[str(k)] = _build_names(v, full_key)
                else:
                    result[str(k)] = full_key
            return result

        return _build_names(item, "")


def _apply_save_args_tree(item: PyTree, save_args_tree: PyTree) -> PyTree:
    """Recursively apply a tree of ``SaveArgs`` to ``item`` before serialization.

    Args:
        item: PyTree structure to save.
        save_args_tree: Matching tree of ``SaveArgs``.

    Returns:
        PyTree: PyTree with leaf ``dtype`` coercions applied.
    """
    if save_args_tree is None or item is None:
        return item
    if is_pytree_dict(item) and is_pytree_dict(save_args_tree):
        return {k: _apply_save_args_tree(v, save_args_tree.get(k)) for k, v in item.items()}
    if isinstance(save_args_tree, SaveArgs):
        dt = save_args_tree.dtype
        if (dt is int or dt == "int" or dt == "int32" or dt == "int64") and isinstance(
            item, (int, float, str, bool)
        ):
            return int(item)
        if (dt is float or dt == "float" or dt == "float32" or dt == "float64") and isinstance(
            item, (int, float, str, bool)
        ):
            return float(item)
        if dt is str or dt == "str":
            return str(item)
    return item


def _apply_restore_args_tree(restored: PyTree, restore_args_tree: PyTree) -> PyTree:
    """Recursively apply a tree of ``RestoreArgs`` / ``ArrayRestoreArgs`` to ``restored``.

    Args:
        restored: Restored PyTree structure.
        restore_args_tree: Matching tree of ``RestoreArgs`` or ``ArrayRestoreArgs``.

    Returns:
        PyTree: Coerced and validated PyTree structure.
    """
    if restore_args_tree is None:
        return restored
    if isinstance(restored, FlaxStateProtocol) and is_pytree_dict(restore_args_tree):
        return _build_flax_state(
            {k: _apply_restore_args_tree(v, restore_args_tree.get(k)) for k, v in restored.items()},
            restored,
        )
    if is_pytree_dict(restored) and is_pytree_dict(restore_args_tree):
        return {
            k: _apply_restore_args_tree(v, restore_args_tree.get(k)) for k, v in restored.items()
        }
    if isinstance(restored, FlaxVariableProtocol) and isinstance(
        restore_args_tree, ArrayRestoreArgs
    ):
        coerced_inner = _apply_restore_args_tree(restored.value, restore_args_tree)
        return restored.replace(value=_coerce_flax_variable_value(coerced_inner))
    if isinstance(restore_args_tree, ArrayRestoreArgs) and is_array_protocol(restored):
        import zero_chex.assertions as _chex_assertions

        expected_shape = (
            restore_args_tree.shape
            if restore_args_tree.shape is not None
            else restore_args_tree.global_shape
        )
        if restore_args_tree.strict and expected_shape is not None:
            _chex_assertions.assert_shape(restored, expected_shape)
        if restore_args_tree.dtype is not None:
            resolved_dt = _resolve_compiler_dtype(str(restore_args_tree.dtype))
            nested_data = array_to_nested_list(restored)
            raw_in = (
                nested_data if isinstance(nested_data, (int, float, bool, list, tuple)) else 0.0
            )
            restored = jnp.reshape(jnp.asarray(raw_in, dtype=resolved_dt), restored.shape)
            if restore_args_tree.strict:
                _chex_assertions.assert_type(restored, resolved_dt)
        return restored
    target_type: type | str | None = (
        restore_args_tree.restore_type
        if isinstance(restore_args_tree, RestoreArgs)
        else (
            restore_args_tree.restore_type()
            if isinstance(restore_args_tree, ArrayRestoreArgs)
            else None
        )
    )
    target_dtype: str | DType | type | None = (
        restore_args_tree.dtype
        if isinstance(restore_args_tree, (RestoreArgs, ArrayRestoreArgs))
        else None
    )
    effective_target = target_type if target_type is not None else target_dtype

    if (effective_target is int or effective_target in ("int", "int32", "int64")) and isinstance(
        restored, (int, float, str, bool)
    ):
        return int(restored)
    if (
        effective_target is float or effective_target in ("float", "float32", "float64")
    ) and isinstance(restored, (int, float, str, bool)):
        return float(restored)
    if (effective_target is str or effective_target == "str") and restored is not None:
        return str(restored)
    if effective_target is list and isinstance(restored, (tuple, list)):
        return list(restored)
    if effective_target is tuple and isinstance(restored, (tuple, list)):
        return tuple(restored)
    return restored


class PyTreeCheckpointHandler(BasePyTreeCheckpointHandler):
    """Handler for saving and restoring PyTree structures with transformations."""

    def __init__(
        self,
        aggregate_filename: str | None = None,
        use_ocdbt: bool = True,
        use_zarr3: bool = False,
        primary_host: int | None = 0,
        type_handler_registry: TypeHandlerRegistryProtocol | None = None,
        handler_impl: BasePyTreeCheckpointHandler | None = None,
        save_concurrent_gb: int | None = None,
        restore_concurrent_gb: int | None = None,
        *,
        save_device_host_concurrent_gb: int | None = None,
        memory_limit_options: MemoryLimitOptions | None = None,
        use_compression: bool = True,
        multiprocessing_options: MultiprocessingOptions | None = None,
        pytree_metadata_options: PyTreeMetadataOptions | None = None,
        array_metadata_validator: ArrayMetadataValidatorProtocol | None = None,
        enable_pinned_host_transfer: bool | None = None,
        is_prioritized_key_fn: IsPrioritizedKeyFn | None = None,
    ) -> None:
        """Initialize the PyTreeCheckpointHandler.

        Args:
            aggregate_filename: Optional filename for aggregated msgpack checkpoints. Defaults to None.
            use_ocdbt: Whether to enable OCDBT storage. Defaults to True.
            use_zarr3: Whether to enable Zarr v3 storage. Defaults to False.
            primary_host: Primary host ID for metadata writes. Defaults to 0.
            type_handler_registry: Optional ``TypeHandlerRegistryProtocol``. Defaults to None.
            handler_impl: Optional custom ``BasePyTreeCheckpointHandler`` delegate. Defaults to None.
            save_concurrent_gb: Max concurrent GB during save. Defaults to None.
            restore_concurrent_gb: Max concurrent GB during restore. Defaults to None.
            save_device_host_concurrent_gb: Max concurrent device-to-host GB. Defaults to None.
            memory_limit_options: Optional ``MemoryLimitOptions``. Defaults to None.
            use_compression: Whether to enable chunk compression. Defaults to True.
            multiprocessing_options: Optional ``MultiprocessingOptions``. Defaults to None.
            pytree_metadata_options: Optional ``PyTreeMetadataOptions``. Defaults to None.
            array_metadata_validator: Optional ``ArrayMetadataValidatorProtocol``. Defaults to None.
            enable_pinned_host_transfer: Whether to enable pinned host memory transfer. Defaults to None.
            is_prioritized_key_fn: Optional key prioritization function. Defaults to None.
        """
        resolved_mp = multiprocessing_options or MultiprocessingOptions(primary_host=primary_host)
        super().__init__(
            memory_limit_options=memory_limit_options,
            use_ocdbt=use_ocdbt,
            use_zarr3=use_zarr3,
            use_compression=use_compression,
            multiprocessing_options=resolved_mp,
            type_handler_registry=type_handler_registry,
            pytree_metadata_options=pytree_metadata_options,
            array_metadata_validator=array_metadata_validator,
            enable_pinned_host_transfer=enable_pinned_host_transfer,
            is_prioritized_key_fn=is_prioritized_key_fn,
        )
        self.aggregate_filename = aggregate_filename
        self.primary_host = primary_host
        self.save_concurrent_gb = save_concurrent_gb
        self.restore_concurrent_gb = restore_concurrent_gb
        self.save_device_host_concurrent_gb = save_device_host_concurrent_gb
        self.handler_impl = handler_impl

    def _order_prioritized_keys(self, node: PyTree) -> PyTree:
        """Order dictionary keys placing prioritized keys first when ``self.is_prioritized_key_fn`` is set.

        Args:
            node: PyTree node to order.

        Returns:
            PyTree: Ordered PyTree dictionary.
        """
        if self.is_prioritized_key_fn is None or not is_pytree_dict(node):
            return node
        prio_fn = self.is_prioritized_key_fn
        sorted_keys = sorted(node.keys(), key=lambda k: (0 if prio_fn((str(k),)) else 1, k))
        return {k: self._order_prioritized_keys(node[k]) for k in sorted_keys}

    def _encode_custom_leaves(self, node: PyTree) -> PyTree:
        """Encode leaves registered in ``self.type_handler_registry`` before packing.

        Args:
            node: PyTree node to inspect.

        Returns:
            PyTree: Tree with custom leaves converted to tagged dictionaries.
        """
        if self.type_handler_registry is None or node is None:
            return node
        if is_pytree_dict(node):
            return {k: self._encode_custom_leaves(v) for k, v in node.items()}
        th = self.type_handler_registry.get(type(node))
        if th is not None:
            return {
                "__custom_type__": th.typestr(),
                "__custom_hex__": th.serialize(node).hex(),
            }
        return node

    def _decode_custom_leaves(self, node: PyTree) -> PyTree:
        """Decode leaves tagged with ``__custom_type__`` using ``self.type_handler_registry``.

        Args:
            node: Restored PyTree node to inspect.

        Returns:
            PyTree: Tree with custom leaves deserialized via their registered ``TypeHandlerProtocol``.
        """
        if self.type_handler_registry is None or node is None:
            return node
        if is_pytree_dict(node):
            custom_type = node.get("__custom_type__")
            custom_hex = node.get("__custom_hex__")
            if isinstance(custom_type, str) and isinstance(custom_hex, str):
                th = self.type_handler_registry.get_by_typestr(custom_type)
                if th is not None:
                    return th.deserialize(bytes.fromhex(custom_hex))
            return {k: self._decode_custom_leaves(v) for k, v in node.items()}
        return node

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        save_args: PyTree = None,
        **kwargs: PyTree,
    ) -> None:
        """Save a PyTree to ``directory``, respecting ``handler_impl``, ``save_args``, and ``aggregate_filename``.

        Args:
            directory: Target checkpoint directory. Defaults to None.
            item: PyTree or ``args.PyTreeSave`` / ``args.StandardSave`` wrapper to save. Defaults to None.
            *pos_args: Additional positional save arguments.
            save_args: Optional PyTree of ``SaveArgs``. Defaults to None.
            **kwargs: Additional keyword save arguments.
        """
        if self.handler_impl is not None:
            self.handler_impl.save(directory, item, *pos_args, **kwargs)
            return
        active_save_args = save_args
        raw_arg = item if isinstance(item, _CheckpointArgsBase) else kwargs.get("args")
        if (
            isinstance(raw_arg, (_PyTreeSaveArgsCls, _StandardSaveArgsCls))
            and active_save_args is None
        ):
            active_save_args = raw_arg.save_args
        payload = _unwrap_checkpoint_arg(item, kwargs)
        if active_save_args is not None:
            payload = _apply_save_args_tree(payload, active_save_args)
        payload = self._order_prioritized_keys(payload)
        if (
            self.array_metadata_validator is not None
            and self.enable_post_merge_validation
            and is_pytree_mapping(payload)
        ):
            meta_snapshot: dict[str, JsonValue] = {
                str(k): str(type(v).__name__) for k, v in payload.items()
            }
            self.array_metadata_validator.validate(meta_snapshot)
        encoded_payload = self._encode_custom_leaves(payload)
        super().save(directory, encoded_payload, *pos_args, **kwargs)
        if directory is not None and self.aggregate_filename is not None:
            key = str(directory).rstrip("/")
            encoded = MsgpackEngine().packb(self._pack_pytree(encoded_payload))
            AtomicWrite().atomic_write_bytes(f"{key}/{self.aggregate_filename}", encoded)

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        transforms: PyTree = None,
        transforms_default_to_original: bool | PyTree = True,
        restore_args: PyTree = None,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore a PyTree from ``directory``, optionally applying transformations.

        Args:
            directory: Source checkpoint directory. Defaults to None.
            item: Target structure or ``args.PyTreeRestore`` / ``args.StandardRestore``. Defaults to None.
            *pos_args: Additional positional restore arguments.
            transforms: Optional transformation rules PyTree. Defaults to None.
            transforms_default_to_original: Whether untransformed fields fall back to original values. Defaults to True.
            restore_args: Optional tree of ``RestoreArgs``. Defaults to None.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: The restored (and optionally transformed) PyTree.
        """
        if self.handler_impl is not None:
            return self.handler_impl.restore(directory, item, *pos_args, **kwargs)
        default_orig = bool(transforms_default_to_original)
        active_restore_args = restore_args
        raw_arg = item if isinstance(item, _CheckpointArgsBase) else kwargs.get("args")
        if isinstance(raw_arg, _PyTreeRestoreArgsCls):
            transforms = raw_arg.transforms if transforms is None else transforms
            default_orig = raw_arg.transforms_default_to_original
            active_restore_args = (
                raw_arg.restore_args if active_restore_args is None else active_restore_args
            )
            target_item = raw_arg.item
        elif isinstance(raw_arg, _StandardRestoreArgsCls):
            active_restore_args = (
                raw_arg.restore_args if active_restore_args is None else active_restore_args
            )
            target_item = raw_arg.item
        else:
            target_item = _unwrap_checkpoint_arg(item, kwargs)

        restored: PyTree = None
        if directory is not None and self.aggregate_filename is not None:
            agg_path = f"{str(directory).rstrip('/')}/{self.aggregate_filename}"
            raw_agg = FileSystemIO().read(agg_path)
            if raw_agg:
                restored = self._unpack_pytree(MsgpackEngine().unpackb(raw_agg), target_item)
        if restored is None:
            restored = super().restore(directory, target_item, *pos_args, **kwargs)
        if restored is not None:
            restored = self._decode_custom_leaves(restored)
        if active_restore_args is not None and restored is not None:
            restored = _apply_restore_args_tree(restored, active_restore_args)
        if transforms is not None and target_item is not None and restored is not None:
            return apply_transformations(
                restored,
                transforms,
                target_item,
                default_to_original=default_orig,
            )
        if (
            transforms is None
            and isinstance(target_item, (FlaxStateProtocol, FrozenDictProtocol, NamedTupleProtocol))
            and isinstance(restored, (FlaxStateProtocol, FrozenDictProtocol, NamedTupleProtocol))
        ):
            import zero_chex.trees as _chex_trees

            _chex_trees.assert_trees_all_equal_structs(restored, target_item)
        return restored


class StandardCheckpointHandler(PyTreeCheckpointHandler):
    """Standard handler for saving and restoring typical training state PyTrees."""

    def __init__(
        self,
        save_concurrent_gb: int | None = 96,
        restore_concurrent_gb: int | None = 96,
        *,
        multiprocessing_options: MultiprocessingOptions | None = None,
        pytree_metadata_options: PyTreeMetadataOptions | None = None,
        use_ocdbt: bool = True,
    ) -> None:
        """Initialize the StandardCheckpointHandler.

        Args:
            save_concurrent_gb: Max concurrent GB during save. Defaults to 96.
            restore_concurrent_gb: Max concurrent GB during restore. Defaults to 96.
            multiprocessing_options: Optional ``MultiprocessingOptions``. Defaults to None.
            pytree_metadata_options: Optional ``PyTreeMetadataOptions``. Defaults to None.
            use_ocdbt: Whether to use OCDBT storage format. Defaults to True.
        """
        super().__init__(
            save_concurrent_gb=save_concurrent_gb,
            restore_concurrent_gb=restore_concurrent_gb,
            multiprocessing_options=multiprocessing_options,
            pytree_metadata_options=pytree_metadata_options,
            use_ocdbt=use_ocdbt,
        )


class JsonCheckpointHandler(AsyncCheckpointHandler):
    """Handler for saving and restoring JSON-formatted checkpoint files."""

    def __init__(
        self,
        filename: str | None = None,
        *,
        multiprocessing_options: MultiprocessingOptions | None = None,
    ) -> None:
        """Initialize the JsonCheckpointHandler.

        Args:
            filename: Optional target JSON filename inside the checkpoint directory. Defaults to None.
            multiprocessing_options: Optional ``MultiprocessingOptions``. Defaults to None.
        """
        super().__init__()
        self.filename = filename
        self.multiprocessing_options = multiprocessing_options

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Save a JSON-serializable object to ``directory/<filename>``.

        Args:
            directory: Target checkpoint directory. Defaults to None.
            item: JSON-compatible object or ``args.JsonSave`` wrapper. Defaults to None.
            *pos_args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        if directory is not None:
            payload = _unwrap_checkpoint_arg(item, kwargs)
            fname = self.filename or "metadata.json"
            target_path = f"{str(directory).rstrip('/')}/{fname}"
            AtomicWrite().atomic_write_bytes(
                target_path,
                json.dumps(payload, default=str).encode("utf-8"),
            )

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore a JSON-formatted checkpoint from ``directory/<filename>``.

        Args:
            directory: Source checkpoint directory. Defaults to None.
            item: Optional fallback template or ``args.JsonRestore`` wrapper. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: Parsed JSON object or fallback template.
        """
        template = _unwrap_checkpoint_arg(item, kwargs)
        if directory is not None:
            fname = self.filename or "metadata.json"
            raw = FileSystemIO().read(f"{str(directory).rstrip('/')}/{fname}")
            if raw:
                loaded: PyTree = json.loads(raw.decode("utf-8"))
                return loaded
        return template


class ProtoCheckpointHandler(AsyncCheckpointHandler):
    """Handler for saving and restoring Protocol Buffer checkpoints."""

    def __init__(
        self,
        filename: str = "proto.pb",
        *,
        multiprocessing_options: MultiprocessingOptions | None = None,
    ) -> None:
        """Initialize the ProtoCheckpointHandler.

        Args:
            filename: Target filename for the serialized protobuf artifact. Defaults to ``"proto.pb"``.
            multiprocessing_options: Optional ``MultiprocessingOptions``. Defaults to None.
        """
        super().__init__()
        self.filename = filename
        self.multiprocessing_options = multiprocessing_options

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Save a protobuf payload to ``directory/<self.filename>``.

        Args:
            directory: Target checkpoint directory. Defaults to None.
            item: Protobuf payload or ``args.ProtoSave`` wrapper. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.
        """
        if directory is not None:
            raw_arg = item if isinstance(item, _CheckpointArgsBase) else kwargs.get("args")
            fname = (
                raw_arg.filename
                if isinstance(raw_arg, _ProtoSaveArgsCls) and self.filename == "proto.pb"
                else self.filename
            )
            payload = _unwrap_checkpoint_arg(item, kwargs)
            target_path = f"{str(directory).rstrip('/')}/{fname}"
            AtomicWrite().atomic_write_bytes(target_path, MsgpackEngine().packb(payload))

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore a protobuf payload from ``directory/<self.filename>``.

        Args:
            directory: Source checkpoint directory. Defaults to None.
            item: Optional template or ``args.ProtoRestore`` wrapper. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: Restored protobuf structure or template.
        """
        raw_arg = item if isinstance(item, _CheckpointArgsBase) else kwargs.get("args")
        item_type = raw_arg.item_type if isinstance(raw_arg, _ProtoRestoreArgsCls) else None
        fname = (
            raw_arg.filename
            if isinstance(raw_arg, _ProtoRestoreArgsCls) and self.filename == "proto.pb"
            else self.filename
        )
        template = _unwrap_checkpoint_arg(item, kwargs)
        if directory is not None:
            raw = FileSystemIO().read(f"{str(directory).rstrip('/')}/{fname}")
            if raw:
                unpacked = MsgpackEngine().unpackb(raw)
                if item_type is str and unpacked is not None:
                    return str(unpacked)
                return unpacked
        return template


class JaxRandomKeyCheckpointHandler(AsyncCheckpointHandler):
    """Handler for saving and restoring JAX PRNG random keys."""

    def __init__(self, key_name: str | None = None) -> None:
        """Initialize the JaxRandomKeyCheckpointHandler.

        Args:
            key_name: Optional filename/key identifier for the saved random key. Defaults to None.
        """
        super().__init__()
        self.key_name = key_name

    def checkpoint_restore_args(
        self,
        item: PyTree,
        restore_args: RestoreArgs | None = None,
    ) -> RestoreArgs | None:
        """Construct ``RestoreArgs`` for restoring a JAX random key.

        Args:
            item: Target random key item or structure.
            restore_args: Optional explicit ``RestoreArgs``. Defaults to None.

        Returns:
            RestoreArgs | None: The resolved ``RestoreArgs``, or ``None``.
        """
        return restore_args

    def checkpoint_save_args(
        self,
        item: PyTree,
    ) -> tuple[PyTree, SaveArgs | None]:
        """Construct ``(item, SaveArgs)`` tuple for saving a JAX random key.

        Args:
            item: Random key item to save.

        Returns:
            tuple[PyTree, SaveArgs | None]: Tuple of ``(None, None)`` or ``(item, SaveArgs)``.
        """
        if item is None:
            return (None, None)
        return (item, SaveArgs())

    def post_restore(self, item: PyTree, metadata: PyTree = None) -> PyTree:
        """Apply post-restoration formatting to a restored JAX random key.

        Args:
            item: Restored raw key data.
            metadata: Optional key metadata. Defaults to None.

        Returns:
            PyTree: Post-processed random key.
        """
        return item

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Save a JAX random key into ``directory/<key_name>.msgpack``.

        Args:
            directory: Target directory path. Defaults to None.
            item: JAX random key or ``args.JaxRandomKeySave``. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.
        """
        if directory is not None:
            raw_payload = _unwrap_checkpoint_arg(item, kwargs)
            key_item, _ = self.checkpoint_save_args(raw_payload)
            fname = f"{self.key_name or 'jax_random_key'}.msgpack"
            AtomicWrite().atomic_write_bytes(
                f"{str(directory).rstrip('/')}/{fname}",
                MsgpackEngine().packb(key_item),
            )

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore a JAX random key from ``directory/<key_name>.msgpack``.

        Args:
            directory: Source directory path. Defaults to None.
            item: Optional template key or ``args.JaxRandomKeyRestore``. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: Restored JAX random key.
        """
        template = _unwrap_checkpoint_arg(item, kwargs)
        self.checkpoint_restore_args(template)
        if directory is not None:
            fname = f"{self.key_name or 'jax_random_key'}.msgpack"
            raw = FileSystemIO().read(f"{str(directory).rstrip('/')}/{fname}")
            if raw:
                return self.post_restore(MsgpackEngine().unpackb(raw))
        return self.post_restore(template)


class NumpyRandomKeyCheckpointHandler(AsyncCheckpointHandler):
    """Handler for saving and restoring NumPy PRNG random keys and states."""

    def __init__(self, key_name: str | None = None) -> None:
        """Initialize the NumpyRandomKeyCheckpointHandler.

        Args:
            key_name: Optional filename/key identifier for the saved NumPy random key. Defaults to None.
        """
        super().__init__()
        self.key_name = key_name

    def checkpoint_restore_args(
        self,
        item: PyTree,
        restore_args: RestoreArgs | None = None,
    ) -> RestoreArgs | None:
        """Construct ``RestoreArgs`` for restoring a NumPy random key.

        Args:
            item: Target random key item or structure.
            restore_args: Optional explicit ``RestoreArgs``. Defaults to None.

        Returns:
            RestoreArgs | None: The resolved ``RestoreArgs``, or ``None``.
        """
        return restore_args

    def checkpoint_save_args(
        self,
        item: PyTree,
    ) -> tuple[PyTree, SaveArgs | None]:
        """Construct ``(item, SaveArgs)`` tuple for saving a NumPy random key.

        Args:
            item: Random key item to save.

        Returns:
            tuple[PyTree, SaveArgs | None]: Tuple of ``(None, None)`` or ``(item, SaveArgs)``.
        """
        if item is None:
            return (None, None)
        return (item, SaveArgs())

    def post_restore(self, item: PyTree, metadata: PyTree = None) -> PyTree:
        """Apply post-restoration formatting to a restored NumPy random key.

        Args:
            item: Restored raw key data.
            metadata: Optional key metadata. Defaults to None.

        Returns:
            PyTree: Post-processed random key.
        """
        return item

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Save a NumPy random key into ``directory/<key_name>.msgpack``.

        Args:
            directory: Target directory path. Defaults to None.
            item: NumPy random key or ``args.NumpyRandomKeySave``. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.
        """
        if directory is not None:
            raw_payload = _unwrap_checkpoint_arg(item, kwargs)
            key_item, _ = self.checkpoint_save_args(raw_payload)
            fname = f"{self.key_name or 'numpy_random_key'}.msgpack"
            AtomicWrite().atomic_write_bytes(
                f"{str(directory).rstrip('/')}/{fname}",
                MsgpackEngine().packb(key_item),
            )

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore a NumPy random key from ``directory/<key_name>.msgpack``.

        Args:
            directory: Source directory path. Defaults to None.
            item: Optional template key or ``args.NumpyRandomKeyRestore``. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: Restored NumPy random key.
        """
        template = _unwrap_checkpoint_arg(item, kwargs)
        self.checkpoint_restore_args(template)
        if directory is not None:
            fname = f"{self.key_name or 'numpy_random_key'}.msgpack"
            raw = FileSystemIO().read(f"{str(directory).rstrip('/')}/{fname}")
            if raw:
                return self.post_restore(MsgpackEngine().unpackb(raw))
        return self.post_restore(template)


class RandomKeyCheckpointHandler(AsyncCheckpointHandler):
    """Handler for saving and restoring JAX or NumPy PRNG random keys and states."""

    def __init__(self, key_name: str | None = None) -> None:
        """Initialize the RandomKeyCheckpointHandler.

        Args:
            key_name: Optional filename/key identifier for the saved random key. Defaults to None.
        """
        super().__init__()
        self.key_name = key_name

    def checkpoint_restore_args(
        self,
        item: PyTree,
        restore_args: RestoreArgs | None = None,
    ) -> RestoreArgs | None:
        """Construct ``RestoreArgs`` for restoring a random key.

        Args:
            item: Target random key item or structure.
            restore_args: Optional explicit ``RestoreArgs``. Defaults to None.

        Returns:
            RestoreArgs | None: The resolved ``RestoreArgs``, or ``None``.
        """
        _ = item
        return restore_args

    def checkpoint_save_args(
        self,
        item: PyTree,
    ) -> tuple[PyTree, SaveArgs | None]:
        """Construct ``(item, SaveArgs)`` tuple for saving a random key.

        Args:
            item: Random key item to save.

        Returns:
            tuple[PyTree, SaveArgs | None]: Tuple of ``(None, None)`` or ``(item, SaveArgs)``.
        """
        if item is None:
            return (None, None)
        return (item, SaveArgs())

    def post_restore(self, item: PyTree, metadata: PyTree = None) -> PyTree:
        """Apply post-restoration formatting to a restored random key.

        Args:
            item: Restored raw key data.
            metadata: Optional key metadata. Defaults to None.

        Returns:
            PyTree: Post-processed random key.
        """
        _ = metadata
        return item

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Save a random key into ``directory/<key_name>.msgpack``.

        Args:
            directory: Target directory path. Defaults to None.
            item: Random key or ``CheckpointArgs`` wrapper. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.
        """
        _ = pos_args
        if directory is not None:
            raw_payload = _unwrap_checkpoint_arg(item, kwargs)
            key_item, _ = self.checkpoint_save_args(raw_payload)
            fname = f"{self.key_name or 'random_key'}.msgpack"
            AtomicWrite().atomic_write_bytes(
                f"{str(directory).rstrip('/')}/{fname}",
                MsgpackEngine().packb(key_item),
            )

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore a random key from ``directory/<key_name>.msgpack``.

        Args:
            directory: Source directory path. Defaults to None.
            item: Optional template key or ``CheckpointArgs`` wrapper. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: Restored random key.
        """
        _ = pos_args
        template = _unwrap_checkpoint_arg(item, kwargs)
        self.checkpoint_restore_args(template)
        if directory is not None:
            fname = f"{self.key_name or 'random_key'}.msgpack"
            raw = FileSystemIO().read(f"{str(directory).rstrip('/')}/{fname}")
            if raw:
                return self.post_restore(MsgpackEngine().unpackb(raw))
        return self.post_restore(template)


class PyGrainCheckpointHandler(AsyncCheckpointHandler):
    """Handler for saving and restoring ``zero_grain`` dataset and data loader iterators."""

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Save a ``GrainIteratorProtocol`` state into ``directory/process_0-of-1.json``.

        Args:
            directory: Target checkpoint directory path. Defaults to None.
            item: ``GrainIteratorProtocol`` or ``args.PyGrainCheckpointSave`` wrapper. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.
        """
        if directory is not None:
            payload = _unwrap_checkpoint_arg(item, kwargs)
            if isinstance(payload, GrainIteratorProtocol):
                state = payload.get_state()
                target_path = f"{str(directory).rstrip('/')}/process_0-of-1.json"
                AtomicWrite().atomic_write_bytes(
                    target_path,
                    json.dumps(state, indent=2).encode("utf-8"),
                )

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore a ``GrainIteratorProtocol`` state from ``directory/process_0-of-1.json``.

        Args:
            directory: Source checkpoint directory path. Defaults to None.
            item: Optional target ``GrainIteratorProtocol`` or ``args.PyGrainCheckpointRestore``. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: The restored iterator (mutated in-place) or raw state dictionary if ``item is None``.
        """
        template = _unwrap_checkpoint_arg(item, kwargs)
        if directory is not None:
            target_path = f"{str(directory).rstrip('/')}/process_0-of-1.json"
            raw = FileSystemIO().read(target_path)
            if raw:
                state: CheckpointStateDict = json.loads(raw.decode("utf-8"))
                if isinstance(template, GrainIteratorProtocol):
                    template.set_state(state)
                    return template
                state_tree: PyTree = dict(state)
                return state_tree
        return template


class CheckpointHandlerRegistry:
    """Base interface for mapping checkpoint items and argument types to handlers."""

    def __init__(self, *args: PyTree, **kwargs: PyTree) -> None:
        """Initialize the CheckpointHandlerRegistry.

        Args:
            *args: Optional positional initialization arguments.
            **kwargs: Optional keyword initialization arguments.
        """
        _ = (args, kwargs)
        self._entries: dict[tuple[str | None, type | None], CheckpointHandler] = {}

    def add(
        self,
        item: str | None = None,
        args_ty: type | _CheckpointArgsBase | None = None,
        handler: CheckpointHandler | type[CheckpointHandler] | None = None,
        override: bool = False,
    ) -> None:
        """Register a handler for a given ``(item, args)`` pair.

        Args:
            item: Optional item name string (e.g. ``"state"``). Defaults to None.
            args_ty: Optional ``CheckpointArgs`` subclass or instance. Defaults to None.
            handler: Handler instance or class to register. Defaults to None.
            override: Whether to overwrite an existing registration. Defaults to False.

        Raises:
            ValueError: If the entry already exists and ``override=False``.
        """
        resolved_args_cls: type | None = (
            args_ty
            if isinstance(args_ty, type)
            else (type(args_ty) if args_ty is not None else None)
        )
        key = (item, resolved_args_cls)
        if key in self._entries and not override:
            raise ValueError(f"Handler already registered for {key}.")
        if handler is None:
            return
        handler_inst = handler() if isinstance(handler, type) else handler
        self._entries[key] = handler_inst

    def get(
        self,
        item: str | None = None,
        args_ty: type | _CheckpointArgsBase | None = None,
    ) -> CheckpointHandler | None:
        """Retrieve a registered handler for the given ``(item, args)`` pair.

        Args:
            item: Optional item name string. Defaults to None.
            args_ty: Optional ``CheckpointArgs`` subclass or instance. Defaults to None.

        Returns:
            CheckpointHandler | None: Registered handler instance, or ``None`` if not found.
        """
        resolved_args_cls: type | None = (
            args_ty
            if isinstance(args_ty, type)
            else (type(args_ty) if args_ty is not None else None)
        )
        if (item, resolved_args_cls) in self._entries:
            return self._entries[(item, resolved_args_cls)]
        if (item, None) in self._entries:
            return self._entries[(item, None)]
        if (None, resolved_args_cls) in self._entries:
            return self._entries[(None, resolved_args_cls)]
        return None

    def has(
        self,
        item: str | None = None,
        args_ty: type | _CheckpointArgsBase | None = None,
    ) -> bool:
        """Check whether a handler is registered for the given ``(item, args)`` pair.

        Args:
            item: Optional item name string. Defaults to None.
            args_ty: Optional ``CheckpointArgs`` subclass or instance. Defaults to None.

        Returns:
            bool: True if a matching handler is registered, False otherwise.
        """
        return self.get(item, args_ty) is not None

    def get_all_entries(
        self,
    ) -> dict[tuple[str | None, type | None], CheckpointHandler]:
        """Return a copy of all registered ``(item, args_cls) -> handler`` entries.

        Returns:
            dict[tuple[str | None, type | None], CheckpointHandler]: Registered handler mapping.
        """
        return dict(self._entries)


class DefaultCheckpointHandlerRegistry(CheckpointHandlerRegistry):
    """Default concrete implementation of ``CheckpointHandlerRegistry``."""

    def __init__(
        self,
        other_registry: CheckpointHandlerRegistry | None = None,
    ) -> None:
        """Initialize the DefaultCheckpointHandlerRegistry, optionally copying entries.

        Args:
            other_registry: Optional existing registry whose entries should be copied. Defaults to None.
        """
        super().__init__()
        self.other_registry = other_registry
        if other_registry is not None:
            self._entries.update(other_registry.get_all_entries())


def create_default_handler_registry(
    **items_and_handlers: CheckpointHandler | None,
) -> CheckpointHandlerRegistry:
    """Create a ``DefaultCheckpointHandlerRegistry`` populated with default and custom item handlers.

    Args:
        **items_and_handlers: Optional keyword mapping from item names to ``CheckpointHandler`` instances.

    Returns:
        CheckpointHandlerRegistry: Initialized registry with standard and custom handlers registered.
    """
    registry = DefaultCheckpointHandlerRegistry()
    for save_cls, handler_cls in _CATALOG.save_args_to_handler.items():
        registry.add(None, save_cls, handler_cls, override=True)
    for restore_cls, handler_cls in _CATALOG.restore_args_to_handler.items():
        registry.add(None, restore_cls, handler_cls, override=True)
    for item_name, handler_inst in items_and_handlers.items():
        if handler_inst is not None:
            registry.add(item_name, None, handler_inst, override=True)
    return registry


class CompositeCheckpointHandler(AsyncCheckpointHandler):
    """Handler coordinating saving and restoring of multiple named sub-items."""

    def __init__(
        self,
        *item_names: str,
        composite_options: CompositeOptions | None = None,
        handler_registry: CheckpointHandlerRegistry | None = None,
        **items_and_handlers: CheckpointHandler | None,
    ) -> None:
        """Initialize the CompositeCheckpointHandler.

        Args:
            *item_names: Positional names of composite sub-items to manage.
            composite_options: Optional ``CompositeOptions`` configuration. Defaults to None.
            handler_registry: Optional ``CheckpointHandlerRegistry`` for resolving item handlers. Defaults to None.
            **items_and_handlers: Keyword mapping of item names to explicit ``CheckpointHandler`` instances.
        """
        super().__init__()
        self.item_names: tuple[str, ...] = item_names + tuple(items_and_handlers.keys())
        self.composite_options = composite_options or CompositeOptions()
        self.handler_registry = handler_registry
        self._item_handlers: dict[str, CheckpointHandler] = {
            name: StandardCheckpointHandler() for name in item_names
        }
        for k, v in items_and_handlers.items():
            if v is not None:
                self._item_handlers[k] = v

    def _resolve_handler_for_item(
        self,
        item_name: str,
        item_arg: _CheckpointArgsBase | PyTree,
    ) -> CheckpointHandler:
        """Resolve the ``CheckpointHandler`` for a composite sub-item.

        Args:
            item_name: Name of the composite sub-item.
            item_arg: Sub-item value or ``CheckpointArgs`` instance.

        Returns:
            CheckpointHandler: Resolved handler for ``item_name``.
        """
        if self.handler_registry is not None:
            arg_hint = item_arg if isinstance(item_arg, _CheckpointArgsBase) else None
            from_reg = self.handler_registry.get(item_name, arg_hint)
            if from_reg is not None:
                self._item_handlers[item_name] = from_reg
                return from_reg
        if isinstance(item_arg, _CheckpointArgsBase):
            reg_cls = args.get_registered_handler_cls(item_arg)
            if reg_cls is not None and issubclass(reg_cls, CheckpointHandler):
                existing = self._item_handlers.get(item_name)
                if existing is not None and isinstance(existing, reg_cls):
                    return existing
                inst: CheckpointHandler = reg_cls()
                self._item_handlers[item_name] = inst
                return inst
        unwrapped_item = item_arg.item if isinstance(item_arg, _CheckpointArgsBase) else item_arg
        if isinstance(unwrapped_item, GrainIteratorProtocol):
            grain_handler = PyGrainCheckpointHandler()
            self._item_handlers[item_name] = grain_handler
            return grain_handler
        if item_name in self._item_handlers:
            return self._item_handlers[item_name]
        default_handler = StandardCheckpointHandler()
        self._item_handlers[item_name] = default_handler
        return default_handler

    def _extract_composite_map(
        self,
        item: PyTree,
        kwargs: Mapping[str, PyTree],
    ) -> dict[str, _CheckpointArgsBase | PyTree]:
        """Extract a mapping of composite item names to payloads from ``item`` or ``kwargs``.

        Args:
            item: Positional ``args.Composite`` or mapping.
            kwargs: Keyword arguments mapping.

        Returns:
            dict[str, _CheckpointArgsBase | PyTree]: Mapping of sub-item names to values/args.
        """
        raw_arg = item if isinstance(item, _CheckpointArgsBase) else kwargs.get("args", item)
        if isinstance(raw_arg, _CompositeArgsCls):
            return dict(raw_arg.items())
        if is_pytree_dict(raw_arg):
            return dict(raw_arg)
        return {k: v for k, v in kwargs.items() if k != "args"}

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Save each named composite item into ``<directory>/<item_name>``.

        Args:
            directory: Target step checkpoint directory. Defaults to None.
            item: ``args.Composite`` or dictionary of named items. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.
        """
        if directory is None:
            return
        base_dir = str(directory).rstrip("/")
        comp_map = self._extract_composite_map(item, kwargs)
        for name, sub_arg in comp_map.items():
            sub_handler = self._resolve_handler_for_item(name, sub_arg)
            sub_dir = f"{base_dir}/{name}"
            FileSystemIO().makedirs(sub_dir, exist_ok=True)
            sub_handler.save(sub_dir, sub_arg)

    def async_save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> list[Future] | None:
        """Asynchronously save each named composite item and aggregate futures.

        Args:
            directory: Target step checkpoint directory. Defaults to None.
            item: ``args.Composite`` or dictionary of named items. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.

        Returns:
            list[Future] | None: Aggregated list of futures, or None if ``directory`` is None.
        """
        if directory is None:
            return None
        base_dir = str(directory).rstrip("/")
        comp_map = self._extract_composite_map(item, kwargs)
        futures: list[Future] = []
        for name, sub_arg in comp_map.items():
            sub_handler = self._resolve_handler_for_item(name, sub_arg)
            sub_dir = f"{base_dir}/{name}"
            FileSystemIO().makedirs(sub_dir, exist_ok=True)
            if isinstance(sub_handler, AsyncCheckpointHandler):
                futures.extend(sub_handler.async_save(sub_dir, sub_arg) or [])
            else:
                sub_handler.save(sub_dir, sub_arg)
        return futures

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore all composite items from ``<directory>/<item_name>``.

        Args:
            directory: Source step checkpoint directory. Defaults to None.
            item: Optional ``args.Composite`` or template dictionary. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: Dictionary mapping each item name to its restored payload.
        """
        if directory is None:
            return None
        base_dir = str(directory).rstrip("/")
        comp_map = self._extract_composite_map(item, kwargs)
        discovered_dirs = [
            entry for entry in FileSystemIO().listdir(base_dir) if not entry.startswith("_")
        ]
        target_names: list[str] = (
            list(comp_map.keys())
            if comp_map
            else list(dict.fromkeys([*self.item_names, *discovered_dirs]))
        )
        restored_dict: dict[str, PyTree] = {}
        for name in target_names:
            sub_arg = comp_map.get(name)
            sub_dir = f"{base_dir}/{name}"
            if (
                sub_arg is None
                and name not in self._item_handlers
                and FileSystemIO().exists(f"{sub_dir}/process_0-of-1.json")
            ):
                self._item_handlers[name] = PyGrainCheckpointHandler()
            sub_handler = self._resolve_handler_for_item(name, sub_arg)
            restored_dict[name] = sub_handler.restore(sub_dir, sub_arg)
        return restored_dict

    def metadata(self, directory: PathLike | None = None) -> dict[str, JsonValue] | None:
        """Return aggregated metadata across all composite sub-items.

        Args:
            directory: Step checkpoint directory. Defaults to None.

        Returns:
            dict[str, JsonValue] | None: Mapping of item name to sub-handler metadata, or None.
        """
        if directory is None:
            return self._last_metadata
        base_dir = str(directory).rstrip("/")
        meta_map: dict[str, JsonValue] = {}
        for name, sub_handler in self._item_handlers.items():
            sub_meta = sub_handler.metadata(f"{base_dir}/{name}")
            if sub_meta is not None:
                meta_map[name] = sub_meta
        return meta_map if meta_map else self._last_metadata

    def structure(self, directory: PathLike | None = None) -> PyTree:
        """Return the composite structure across all sub-handlers.

        Args:
            directory: Step checkpoint directory. Defaults to None.

        Returns:
            PyTree: Composite structure mapping, or None.
        """
        if directory is None:
            return None
        return self.restore(directory)

    def finalize(self, directory: PathLike | None = None) -> None:
        """Finalize all composite sub-item directories.

        Args:
            directory: Step checkpoint directory. Defaults to None.
        """
        super().finalize(directory)
        if directory is not None:
            base_dir = str(directory).rstrip("/")
            for name, sub_handler in self._item_handlers.items():
                sub_handler.finalize(f"{base_dir}/{name}")

    def close(self) -> None:
        """Close all managed sub-handlers."""
        super().close()
        for sub_handler in self._item_handlers.values():
            sub_handler.close()


for _s_cls, _h_cls in (
    (args.ArraySave, ArrayCheckpointHandler),
    (args.JsonSave, JsonCheckpointHandler),
    (args.ProtoSave, ProtoCheckpointHandler),
    (args.PyTreeSave, PyTreeCheckpointHandler),
    (args.StandardSave, StandardCheckpointHandler),
    (args.JaxRandomKeySave, JaxRandomKeyCheckpointHandler),
    (args.NumpyRandomKeySave, NumpyRandomKeyCheckpointHandler),
    (args.PyGrainCheckpointSave, PyGrainCheckpointHandler),
    (args.Composite, CompositeCheckpointHandler),
):
    _CATALOG.register(_s_cls, _h_cls, for_save=True)

for _r_cls, _h_cls in (
    (args.ArrayRestore, ArrayCheckpointHandler),
    (args.JsonRestore, JsonCheckpointHandler),
    (args.ProtoRestore, ProtoCheckpointHandler),
    (args.PyTreeRestore, PyTreeCheckpointHandler),
    (args.StandardRestore, StandardCheckpointHandler),
    (args.JaxRandomKeyRestore, JaxRandomKeyCheckpointHandler),
    (args.NumpyRandomKeyRestore, NumpyRandomKeyCheckpointHandler),
    (args.PyGrainCheckpointRestore, PyGrainCheckpointHandler),
    (args.Composite, CompositeCheckpointHandler),
):
    _CATALOG.register(_r_cls, _h_cls, for_restore=True)


class AbstractCheckpointer:
    """Abstract base class for synchronous and asynchronous checkpointers."""

    def __init__(self, *args: PyTree, **kwargs: PyTree) -> None:
        """Initialize the AbstractCheckpointer.

        Args:
            *args: Optional positional initialization arguments.
            **kwargs: Optional keyword initialization arguments.
        """
        _ = (args, kwargs)
        self._closed = False

    def save(
        self,
        path: PathLike,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> Future | PyTree:
        """Save an item to the specified path.

        Args:
            path: Destination path for the checkpoint.
            item: Item to save. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.

        Returns:
            Future | PyTree: Result of the save operation.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError

    def restore(
        self,
        path: PathLike,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> Future | PyTree:
        """Restore an item from the specified path.

        Args:
            path: Source path to restore from.
            item: Target structure to restore into. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: The restored item.

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        raise NotImplementedError

    def metadata(self, path: PathLike | None = None) -> dict[str, JsonValue] | None:
        """Return checkpoint metadata for the specified path.

        Args:
            path: Checkpoint path. Defaults to None.

        Returns:
            dict[str, JsonValue] | None: Metadata dictionary, or ``None``.
        """
        return None

    def structure(self, path: PathLike | None = None) -> PyTree:
        """Return the structural tree of the checkpoint at ``path``.

        Args:
            path: Checkpoint path. Defaults to None.

        Returns:
            PyTree: Checkpoint structure, or ``None``.
        """
        return None

    def close(self) -> None:
        """Close the checkpointer and release resources."""
        self._closed = True

    def __enter__(self) -> AbstractCheckpointer:
        """Enter the runtime context for the checkpointer.

        Returns:
            AbstractCheckpointer: Self.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the runtime context and close the checkpointer.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception instance if raised.
            exc_tb: Exception traceback if raised.
        """
        self.close()


class Checkpointer(AbstractCheckpointer):
    """Synchronous checkpointer wrapping a ``CheckpointHandler``."""

    def __init__(
        self,
        handler: CheckpointHandler | None = None,
        *,
        primary_host: int | None = 0,
        active_processes: set[int] | None = None,
        barrier_sync_fn: BarrierSyncFn | None = None,
        path_permission_mode: int | None = None,
        checkpoint_metadata_store: CheckpointMetadataStoreProtocol | None = None,
        temporary_path_cls: type[TemporaryPathProtocol] | None = None,
        multiprocessing_options: MultiprocessingOptions | None = None,
        file_options: FileOptions | None = None,
        atomicity_options: AtomicityOptions | None = None,
        temporary_path_class: type[TemporaryPathProtocol] | None = None,
    ) -> None:
        """Initialize the synchronous Checkpointer.

        Args:
            handler: The ``CheckpointHandler`` used to save and restore items. Defaults to None.
            primary_host: Primary host index for multi-process metadata writes. Defaults to 0.
            active_processes: Optional set of active process indices. Defaults to None.
            barrier_sync_fn: Optional barrier synchronization callback. Defaults to None.
            path_permission_mode: Optional octal permission mode for created directories. Defaults to None.
            checkpoint_metadata_store: Optional metadata store implementation. Defaults to None.
            temporary_path_cls: Optional class used to manage temporary staging paths. Defaults to None.
            multiprocessing_options: Optional ``MultiprocessingOptions``. Defaults to None.
            file_options: Optional ``FileOptions``. Defaults to None.
            atomicity_options: Optional ``AtomicityOptions``. Defaults to None.
            temporary_path_class: Optional alias for ``temporary_path_cls``. Defaults to None.
        """
        super().__init__()
        self.handler = handler
        self.primary_host = primary_host
        self.active_processes = active_processes
        self.barrier_sync_fn = barrier_sync_fn
        self.path_permission_mode = path_permission_mode
        self.multiprocessing_options = multiprocessing_options or (
            MultiprocessingOptions(
                primary_host=primary_host,
                active_processes=active_processes,
            )
            if (primary_host != 0 or active_processes is not None)
            else None
        )
        self.file_options = file_options or (
            FileOptions(path_permission_mode=path_permission_mode)
            if path_permission_mode is not None
            else None
        )
        self.atomicity_options = atomicity_options
        self.checkpoint_metadata_store = checkpoint_metadata_store
        self.temporary_path_cls = (
            temporary_path_cls if temporary_path_cls is not None else temporary_path_class
        )
        self.temporary_path_class = self.temporary_path_cls

    def save(
        self,
        path: PathLike,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> Future | None:
        """Synchronously save ``item`` to ``path`` using the configured handler.

        Args:
            path: Destination path for the checkpoint.
            item: Item to save. Defaults to None.
            *pos_args: Additional positional arguments forwarded to the handler.
            **kwargs: Additional keyword arguments forwarded to the handler.

        Returns:
            Future | None: Always returns ``None`` for synchronous ``Checkpointer``.
        """
        perm_mode = (
            self.file_options.path_permission_mode if self.file_options is not None else None
        )
        temp_obj: TemporaryPathProtocol | None = None
        active_path: PathLike = path
        if self.temporary_path_class is not None:
            temp_obj = self.temporary_path_class()
            temp_obj.create()
            active_path = temp_obj.get()
        elif perm_mode is not None:
            FileSystemIO().makedirs(path, exist_ok=True, permission_mode=perm_mode)

        if self.handler is not None:
            self.handler.save(active_path, item, *pos_args, **kwargs)
            self.handler.finalize(active_path)
        else:
            payload = _unwrap_checkpoint_arg(item, kwargs)
            key = str(active_path).rstrip("/")
            AtomicWrite().atomic_write_bytes(
                f"{key}/checkpoint.msgpack",
                MsgpackEngine().packb(payload),
                permission_mode=perm_mode,
            )
        if temp_obj is not None:
            temp_obj.finalize()
        if self.checkpoint_metadata_store is not None:
            self.checkpoint_metadata_store.write(path, {"path": str(path)})
        return None

    def restore(
        self,
        path: PathLike,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> Future | PyTree:
        """Synchronously restore an item from ``path`` using the configured handler.

        Args:
            path: Source path for the checkpoint.
            item: Optional target structure to restore into. Defaults to None.
            *pos_args: Additional positional arguments forwarded to the handler.
            **kwargs: Additional keyword arguments forwarded to the handler.

        Returns:
            Future | PyTree: The restored item.
        """
        if self.handler is not None:
            return self.handler.restore(path, item, *pos_args, **kwargs)
        key = str(path).rstrip("/")
        raw = FileSystemIO().read(f"{key}/checkpoint.msgpack")
        if raw:
            return MsgpackEngine().unpackb(raw)
        return _unwrap_checkpoint_arg(item, kwargs)

    def metadata(self, path: PathLike | None = None) -> dict[str, JsonValue] | None:
        """Retrieve checkpoint metadata from the underlying handler or metadata store.

        Args:
            path: Checkpoint directory path. Defaults to None.

        Returns:
            dict[str, JsonValue] | None: Metadata dictionary, or ``None``.
        """
        if self.checkpoint_metadata_store is not None and path is not None:
            return self.checkpoint_metadata_store.read(path)
        if self.handler is not None:
            return self.handler.metadata(path)
        return None

    def structure(self, path: PathLike | None = None) -> PyTree:
        """Retrieve the structural skeleton of the checkpoint from the handler.

        Args:
            path: Checkpoint directory path. Defaults to None.

        Returns:
            PyTree: Structure of the checkpoint, or ``None``.
        """
        if self.handler is not None:
            return self.handler.structure(path)
        return None

    def create_temporary_path(self, path: PathLike | None = None) -> pathlib.Path | None:
        """Create a temporary staging path for atomic checkpoint writes.

        Args:
            path: Optional final destination path. Defaults to None.

        Returns:
            pathlib.Path | None: Temporary staging path if ``path`` is provided, otherwise ``None``.
        """
        if path is None:
            return None
        norm = normalize_path(path)
        return norm.parent / f"{norm.name}.orbax-checkpoint-tmp"

    def close(self) -> None:
        """Close the checkpointer and its underlying handler."""
        super().close()
        if self.handler is not None:
            self.handler.close()
        if self.checkpoint_metadata_store is not None:
            self.checkpoint_metadata_store.close()


class AsyncCheckpointer(Checkpointer):
    """Asynchronous checkpointer coordinating background saves and futures."""

    def __init__(
        self,
        handler: CheckpointHandler | None = None,
        timeout_secs: int = 300,
        *,
        _handler: CheckpointHandler | None = None,
        primary_host: int | None = 0,
        active_processes: set[int] | None = None,
        barrier_sync_fn: BarrierSyncFn | None = None,
        post_finalization_callback: Callable[[], None] | None = None,
        path_permission_mode: int | None = None,
        async_options: AsyncOptions | None = None,
        multiprocessing_options: MultiprocessingOptions | None = None,
        temporary_path_cls: type[TemporaryPathProtocol] | None = None,
        file_options: FileOptions | None = None,
        checkpoint_metadata_store: CheckpointMetadataStoreProtocol | None = None,
        atomicity_options: AtomicityOptions | None = None,
        temporary_path_class: type[TemporaryPathProtocol] | None = None,
    ) -> None:
        """Initialize the AsyncCheckpointer.

        Args:
            handler: ``CheckpointHandler`` instance for saving and restoring. Defaults to None.
            timeout_secs: Timeout in seconds for async operations. Defaults to 300.
            _handler: Optional alternate ``CheckpointHandler`` keyword parameter. Defaults to None.
            primary_host: Primary host index for multi-process metadata writes. Defaults to 0.
            active_processes: Optional set of active process indices. Defaults to None.
            barrier_sync_fn: Optional barrier synchronization callback. Defaults to None.
            post_finalization_callback: Optional callback invoked after async finalization. Defaults to None.
            path_permission_mode: Optional octal permission mode for created directories. Defaults to None.
            async_options: Optional ``AsyncOptions``. Defaults to None.
            multiprocessing_options: Optional ``MultiprocessingOptions``. Defaults to None.
            temporary_path_cls: Optional temporary path class. Defaults to None.
            file_options: Optional ``FileOptions``. Defaults to None.
            checkpoint_metadata_store: Optional metadata store implementation. Defaults to None.
            atomicity_options: Optional ``AtomicityOptions``. Defaults to None.
            temporary_path_class: Optional alias for ``temporary_path_cls``. Defaults to None.
        """
        resolved_handler = handler if handler is not None else _handler
        super().__init__(
            handler=resolved_handler,
            primary_host=primary_host,
            active_processes=active_processes,
            barrier_sync_fn=barrier_sync_fn,
            path_permission_mode=path_permission_mode,
            checkpoint_metadata_store=checkpoint_metadata_store,
            temporary_path_cls=temporary_path_cls,
            multiprocessing_options=multiprocessing_options,
            file_options=file_options,
            atomicity_options=atomicity_options,
            temporary_path_class=temporary_path_class,
        )
        self.async_options = async_options
        self.timeout_secs = timeout_secs
        self.post_finalization_callback = post_finalization_callback
        self._pending_futures: list[Future] = []

    def save(
        self,
        path: PathLike,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> Future:
        """Initiate an asynchronous save operation and return a ``Future``.

        Args:
            path: Destination path for the checkpoint.
            item: Item to save. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.

        Returns:
            Future: Future representing the asynchronous save operation.
        """
        if isinstance(self.handler, AsyncCheckpointHandler):
            temp_obj: TemporaryPathProtocol | None = None
            active_path: PathLike = path
            if self.temporary_path_class is not None:
                temp_obj = self.temporary_path_class()
                temp_obj.create()
                active_path = temp_obj.get()
            sub_futures = self.handler.async_save(active_path, item, *pos_args, **kwargs) or []

            def _commit() -> None:
                """Finalize all handler futures and run barrier/post-finalization hooks."""
                for sf in sub_futures:
                    sf.result(timeout=float(self.timeout_secs))
                if temp_obj is not None:
                    temp_obj.finalize()
                active_barrier = (
                    self.async_options.barrier_sync_fn
                    if (
                        self.async_options is not None
                        and self.async_options.barrier_sync_fn is not None
                    )
                    else self.barrier_sync_fn
                )
                if active_barrier is not None:
                    active_barrier(str(path), self.timeout_secs)

            fut = Future(result=None, commit_callback=_commit)
        else:
            super().save(path, item, *pos_args, **kwargs)
            fut = Future(result=None)
        self._pending_futures.append(fut)
        return fut

    def restore(
        self,
        path: PathLike,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> Future | PyTree:
        """Initiate an asynchronous restore operation and return a ``Future``.

        Args:
            path: Source path for the checkpoint.
            item: Optional target structure to restore into. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            Future | PyTree: Future wrapping the restored item.
        """
        restored = super().restore(path, item, *pos_args, **kwargs)
        fut = Future(result=restored)
        self._pending_futures.append(fut)
        return fut

    def check_for_errors(self) -> None:
        """Check whether any background save operations raised an error."""
        for fut in self._pending_futures:
            fut.result()

    def wait_until_finished(self) -> None:
        """Block until all pending asynchronous save operations have finalized."""
        for fut in self._pending_futures:
            fut.result()
        self._pending_futures.clear()
        active_post_cb = (
            self.async_options.post_finalization_callback
            if (
                self.async_options is not None
                and self.async_options.post_finalization_callback is not None
            )
            else self.post_finalization_callback
        )
        if active_post_cb is not None:
            active_post_cb()


class PyTreeCheckpointer(Checkpointer):
    """Convenience checkpointer pre-configured with a ``PyTreeCheckpointHandler``."""

    def __init__(
        self,
        primary_host: int | None = 0,
        use_ocdbt: bool = True,
        use_zarr3: bool = False,
        use_compression: bool = True,
    ) -> None:
        """Initialize the PyTreeCheckpointer.

        Args:
            primary_host: Primary host index for metadata writes. Defaults to 0.
            use_ocdbt: Whether to use OCDBT storage format. Defaults to True.
            use_zarr3: Whether to use Zarr v3 format. Defaults to False.
            use_compression: Whether to enable compression. Defaults to True.
        """
        mp_opts = MultiprocessingOptions(primary_host=primary_host)
        super().__init__(
            handler=PyTreeCheckpointHandler(
                use_ocdbt=use_ocdbt,
                use_zarr3=use_zarr3,
                use_compression=use_compression,
                multiprocessing_options=mp_opts,
            ),
            multiprocessing_options=mp_opts,
        )
        self.primary_host = primary_host
        self.use_ocdbt = use_ocdbt
        self.use_zarr3 = use_zarr3
        self.use_compression = use_compression


class StandardCheckpointer(AsyncCheckpointer):
    """Asynchronous checkpointer pre-configured with ``StandardCheckpointHandler``."""

    def __init__(
        self,
        primary_host: int | None = 0,
        *,
        async_options: AsyncOptions | None = None,
        multiprocessing_options: MultiprocessingOptions | None = None,
        file_options: FileOptions | None = None,
        atomicity_options: AtomicityOptions | None = None,
        checkpoint_metadata_store: CheckpointMetadataStoreProtocol | None = None,
        temporary_path_class: type[TemporaryPathProtocol] | None = None,
        save_concurrent_gb: int = 96,
        restore_concurrent_gb: int = 96,
        pytree_metadata_options: PyTreeMetadataOptions | None = None,
        use_ocdbt: bool = True,
        **kwargs: PyTree,
    ) -> None:
        """Initialize the StandardCheckpointer.

        Args:
            primary_host: Primary host index for metadata writes. Defaults to 0.
            async_options: Optional ``AsyncOptions``. Defaults to None.
            multiprocessing_options: Optional ``MultiprocessingOptions``. Defaults to None.
            file_options: Optional ``FileOptions``. Defaults to None.
            atomicity_options: Optional ``AtomicityOptions``. Defaults to None.
            checkpoint_metadata_store: Optional metadata store implementation. Defaults to None.
            temporary_path_class: Optional temporary path class. Defaults to None.
            save_concurrent_gb: Max concurrent GB during save. Defaults to 96.
            restore_concurrent_gb: Max concurrent GB during restore. Defaults to 96.
            pytree_metadata_options: Optional ``PyTreeMetadataOptions``. Defaults to None.
            use_ocdbt: Whether to use OCDBT storage format. Defaults to True.
            **kwargs: Additional keyword options for ``StandardCheckpointHandler``.
        """
        resolved_mp = multiprocessing_options or MultiprocessingOptions(primary_host=primary_host)
        super().__init__(
            handler=StandardCheckpointHandler(
                save_concurrent_gb=save_concurrent_gb,
                restore_concurrent_gb=restore_concurrent_gb,
                multiprocessing_options=resolved_mp,
                pytree_metadata_options=pytree_metadata_options,
                use_ocdbt=use_ocdbt,
            ),
            primary_host=primary_host,
            multiprocessing_options=resolved_mp,
            async_options=async_options,
            file_options=file_options,
            atomicity_options=atomicity_options,
            checkpoint_metadata_store=checkpoint_metadata_store,
            temporary_path_class=temporary_path_class,
        )
        self.extra_kwargs: dict[str, PyTree] = dict(kwargs)

    def restore(
        self,
        path: PathLike,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> Future | PyTree:
        """Restore a standard PyTree item from ``path``.

        Args:
            path: Source checkpoint path.
            item: Optional target PyTree template. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            Future | PyTree: The restored PyTree item.
        """
        return Checkpointer.restore(self, path, item, *pos_args, **kwargs)


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
        if isinstance(args_obj, _CompositeArgsCls):
            comp_handler = CompositeCheckpointHandler(
                *(self.item_names or ()),
                handler_registry=self.handler_registry,
            )
            if isinstance(self.item_handlers, Mapping):
                for k, h in self.item_handlers.items():
                    comp_handler._item_handlers[k] = h
            comp_handler.save(step_dir, args_obj)
            return

        if isinstance(self.checkpointers, Mapping) and is_pytree_dict(items):
            for item_name, cp in self.checkpointers.items():
                res = cp.save(f"{step_dir}/{item_name}", items.get(item_name), **extra_kw)
                self._record_future(res)
            return
        if isinstance(self.checkpointers, AbstractCheckpointer):
            target_val = items if items is not None else args_obj
            res = self.checkpointers.save(step_dir, target_val, **extra_kw)
            self._record_future(res)
            return

        if isinstance(self.item_handlers, Mapping) and is_pytree_dict(items):
            for item_name, h in self.item_handlers.items():
                h.save(f"{step_dir}/{item_name}", items.get(item_name), **extra_kw)
            return
        if isinstance(self.item_handlers, CheckpointHandler):
            target_val = items if items is not None else args_obj
            self.item_handlers.save(step_dir, target_val, **extra_kw)
            return

        if isinstance(args_obj, _CheckpointArgsBase):
            reg_h_cls = args.get_registered_handler_cls(args_obj)
            if reg_h_cls is not None and issubclass(reg_h_cls, CheckpointHandler):
                reg_h_cls().save(step_dir, args_obj, **extra_kw)
                return

        payload = (
            items
            if items is not None
            else (args_obj.item if isinstance(args_obj, _CheckpointArgsBase) else args_obj)
        )
        packed = BasePyTreeCheckpointHandler()._pack_pytree(payload)
        encoded = MsgpackEngine().packb(packed)
        AtomicWrite().atomic_write_bytes(f"{step_dir}/checkpoint.msgpack", encoded)

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
            src_path = normalize_path(src_dir)
            dest_path = normalize_path(dest_dir)
            io_handler.makedirs(trash_base, exist_ok=True)
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.move(str(src_path), str(dest_path))
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
        self.checkpoints[step] = payload
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

        if self.logger is not None:
            self.logger.log_entry(f"Saved checkpoint at step {step}", step=step)

        if self.options is not None and self.options.max_to_keep is not None:
            protected = self._compute_protected_steps(self.options)
            unprotected_candidates = [
                s
                for s in self.steps
                if s not in protected
                and not (
                    self.options.keep_period is not None
                    and self.options.keep_period > 0
                    and s % self.options.keep_period == 0
                )
            ]
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
        """Return the configured metadata for this checkpoint manager.

        Args:
            step: Optional step number. Defaults to None.

        Returns:
            Mapping[str, JsonValue] | int | None: Configured metadata.
        """
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
        return self.checkpoints.get(target)

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
                restored_val = self._restore_step_from_disk(step_dir, None, None, None)
                self.checkpoints[step_num] = restored_val
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


@dataclasses.dataclass
class Transform:
    """Rule defining how to map or transform a PyTree node from an original tree.

    Attributes:
        original_key: Key, slash-delimited path, regex pattern, or key sequence in the original PyTree.
        use_fallback: Whether to fall back to the same key in the original tree if ``original_key`` is missing.
        value_fn: Single-value transformation function ``(val) -> new_val``.
        multi_value_fn: Multi-value transformation function ``(orig_tree) -> new_val``.
        multi_value_fn_input_args: Optional mapping of input keys to ``RestoreArgs``.
    """

    original_key: str | Sequence[str] | None = None
    use_fallback: bool = False
    value_fn: ValueTransformFn | None = None
    multi_value_fn: MultiValueTransformFn | None = None
    multi_value_fn_input_args: Mapping[str, RestoreArgs] | None = None


class RestoreTransform(Transform):
    """Rule defining how to transform data during a checkpoint restore operation."""

    def __init__(
        self,
        original_key: str | Sequence[str] | None = None,
        use_fallback: bool = False,
        value_fn: ValueTransformFn | None = None,
        multi_value_fn: MultiValueTransformFn | None = None,
        multi_value_fn_input_args: Mapping[str, RestoreArgs] | None = None,
    ) -> None:
        """Initialize the RestoreTransform rule.

        Args:
            original_key: Key or path in the original structure to map from. Defaults to None.
            use_fallback: Whether to use fallback resolution logic. Defaults to False.
            value_fn: Function applied to a single restored value. Defaults to None.
            multi_value_fn: Function deriving a restored value from the full original tree. Defaults to None.
            multi_value_fn_input_args: Optional mapping of input keys to ``RestoreArgs``. Defaults to None.
        """
        super().__init__(
            original_key=original_key,
            use_fallback=use_fallback,
            value_fn=value_fn,
            multi_value_fn=multi_value_fn,
            multi_value_fn_input_args=multi_value_fn_input_args,
        )


def _get_by_path(
    tree: dict[str, PyTree],
    path: str | Sequence[str],
    default_val: PyTree,
) -> PyTree:
    """Retrieve a nested dictionary value by string key, slash-delimited path, regex, or key sequence.

    Args:
        tree: The root dictionary to search.
        path: String key, ``"a/b"`` path, regex pattern, or sequence of keys ``("a", "b")``.
        default_val: Fallback value returned if the path cannot be resolved.

    Returns:
        PyTree: The resolved value at ``path``, or ``default_val``.
    """
    if isinstance(path, str):
        if "/" in path:
            parts: Sequence[str] = path.split("/")
        else:
            if path in tree:
                return tree[path]
            for k, v in tree.items():
                if re.fullmatch(path, k):
                    return v
            return default_val
    else:
        parts = path

    curr: PyTree = tree
    for key in parts:
        if not is_pytree_dict(curr):
            return default_val
        if key in curr:
            curr = curr[key]
        else:
            matched_val: PyTree = None
            found = False
            for ck, cv in curr.items():
                if re.fullmatch(key, ck):
                    matched_val = cv
                    found = True
                    break
            if not found:
                return default_val
            curr = matched_val
    return curr


def apply_transformations(
    original_tree: PyTree,
    transformations: PyTree,
    new_tree: PyTree,
    default_to_original: bool | None = True,
) -> PyTree:
    """Apply a tree of ``Transform`` rules to map ``original_tree`` into ``new_tree``.

    Args:
        original_tree: The source PyTree dictionary.
        transformations: The PyTree of ``Transform`` / ``RestoreTransform`` rules.
        new_tree: The target PyTree structure defining expected output keys.
        default_to_original: If True, keys omitted from ``transformations`` fall back
            to ``original_tree`` values. Defaults to True.

    Returns:
        PyTree: The transformed PyTree matching the structure of ``new_tree``.
    """
    if not is_pytree_dict(new_tree):
        return new_tree

    orig_root: dict[str, PyTree] = original_tree if is_pytree_dict(original_tree) else {}

    def _process(
        orig: dict[str, PyTree],
        trans: dict[str, PyTree],
        new: dict[str, PyTree],
        orig_full: dict[str, PyTree],
    ) -> dict[str, PyTree]:
        """Recursively apply transformation rules for each key in ``new``.

        Args:
            orig: Current dictionary node in the original tree.
            trans: Current dictionary node in the transformations tree.
            new: Current dictionary node in the target tree.
            orig_full: Full original root dictionary for global path lookups.

        Returns:
            dict[str, PyTree]: Transformed sub-dictionary.
        """
        res: dict[str, PyTree] = {}
        for k, new_val in new.items():
            t = trans.get(k)
            o = orig.get(k)
            orig_sub: dict[str, PyTree] = o if is_pytree_dict(o) else {}
            if t is not None:
                if is_pytree_dict(t):
                    res[k] = (
                        _process(orig_sub, t, new_val, orig_full)
                        if is_pytree_dict(new_val)
                        else new_val
                    )
                elif isinstance(t, Transform):
                    if t.multi_value_fn is not None:
                        coerced_full = (
                            _apply_restore_args_tree(
                                orig_full,
                                dict(t.multi_value_fn_input_args),
                            )
                            if t.multi_value_fn_input_args is not None
                            else orig_full
                        )
                        res[k] = t.multi_value_fn(coerced_full)
                    else:
                        fallback_val = (
                            o
                            if (t.use_fallback or t.original_key is None)
                            else (new_val if new_val is not None else o)
                        )
                        resolved_o = (
                            _get_by_path(orig_full, t.original_key, fallback_val)
                            if t.original_key is not None
                            else o
                        )
                        if t.value_fn is not None:
                            res[k] = t.value_fn(resolved_o)
                        else:
                            res[k] = resolved_o
                else:
                    res[k] = t
            elif not default_to_original:
                res[k] = new_val
            else:
                res[k] = o if o is not None else new_val
        return res

    trans_root: dict[str, PyTree] = transformations if is_pytree_dict(transformations) else {}
    return _process(orig_root, trans_root, new_tree, orig_root)


def merge_trees(
    *trees: PyTree,
    target: dict[str, PyTree] | None = None,
) -> dict[str, PyTree]:
    """Deeply merge multiple PyTree dictionaries into a single dictionary.

    Args:
        *trees: Variable number of PyTree dictionaries to merge from left to right.
        target: Optional initial target dictionary to seed the merge. Defaults to None.

    Returns:
        dict[str, PyTree]: Deeply merged PyTree dictionary.

    Raises:
        TypeError: If any positional argument in ``trees`` is not a ``dict``.
    """
    if not trees:
        return {}
    res: dict[str, PyTree] = {}
    if target is not None:
        res.update(target)
    for t in trees:
        if is_pytree_dict(t):
            for k, v in t.items():
                existing = res.get(k)
                if is_pytree_dict(existing) and is_pytree_dict(v):
                    res[k] = merge_trees(existing, v)
                else:
                    res[k] = v
        else:
            raise TypeError("Expected PyTree dicts")
    return res


class async_checkpoint_handler:
    """Namespace for asynchronous checkpoint handler classes."""

    AsyncCheckpointHandler = AsyncCheckpointHandler


class options:
    """Namespace for Orbax checkpointing option dataclasses."""

    AsyncOptions = AsyncOptions
    MultiprocessingOptions = MultiprocessingOptions
    FileOptions = FileOptions
    AtomicityOptions = AtomicityOptions
    MemoryLimitOptions = MemoryLimitOptions
    PyTreeMetadataOptions = PyTreeMetadataOptions
    CompositeOptions = CompositeOptions


class orbax:
    """Compat namespace mirroring the top-level ``orbax`` package hierarchy."""

    export = _export_module

    class checkpoint:
        """Compat namespace mirroring ``orbax.checkpoint``."""

        options = options


export = _export_module


class atomicity:
    """Namespace for atomic path protocols and helpers."""

    TemporaryPath = TemporaryPathProtocol


class checkpoint:
    """Namespace for checkpoint metadata store protocols."""

    CheckpointMetadataStore = CheckpointMetadataStoreProtocol


class multihost:
    """Namespace for multi-host barrier synchronization helpers."""

    BarrierSyncFn = BarrierSyncFn
    BarrierSync = BarrierSync


class epath:
    """Namespace for path-like type aliases and normalization utilities."""

    PathLike = PathLike
    Path = pathlib.Path


class abstract_logger:
    """Namespace for structured checkpoint logger protocols."""

    AbstractLogger = AbstractLoggerProtocol


class checkpoint_handler:
    """Namespace exporting the base ``CheckpointHandler`` class."""

    CheckpointHandler = CheckpointHandler
    AsyncCheckpointHandler = AsyncCheckpointHandler


class nest_asyncio:
    """Namespace stub for ``nest_asyncio`` loop compatibility."""

    @staticmethod
    def apply() -> None:
        """Apply nested event-loop compatibility (no-op in zero-orbax)."""
        return None


class tree:
    """Namespace for PyTree flattening and inspection utilities."""

    is_pytree_dict = staticmethod(is_pytree_dict)
    is_pytree_mapping = staticmethod(is_pytree_mapping)


class abstract_checkpoint_manager:
    """Namespace exporting ``AbstractCheckpointManager``."""

    AbstractCheckpointManager = AbstractCheckpointManager


class abstract_checkpointer:
    """Namespace exporting ``AbstractCheckpointer``."""

    AbstractCheckpointer = AbstractCheckpointer


class aggregate_handlers:
    """Namespace exporting msgpack aggregation handlers."""

    MsgpackHandler = MsgpackEngine


class async_checkpointer:
    """Namespace exporting ``AsyncCheckpointer``."""

    AsyncCheckpointer = AsyncCheckpointer


class checkpoint_args:
    """Namespace exporting ``args`` and ``CheckpointArgs``."""

    CheckpointArgs = args.CheckpointArgs
    register_with_handler = staticmethod(args.register_with_handler)
    get_registered_handler_cls = staticmethod(args.get_registered_handler_cls)
    get_registered_args_cls = staticmethod(args.get_registered_args_cls)
    has_registered_args = staticmethod(args.has_registered_args)


class checkpoint_manager:
    """Namespace exporting ``CheckpointManager`` and ``CheckpointManagerOptions``."""

    CheckpointManager = CheckpointManager
    CheckpointManagerOptions = CheckpointManagerOptions
    AsyncOptions = AsyncOptions


class checkpoint_utils:
    """Namespace for checkpoint discovery and iteration utilities."""

    @staticmethod
    def checkpoints_iterator(
        checkpoint_dir: PathLike,
        min_interval_secs: int = 0,
        timeout: int | None = None,
        step_prefix: str | None = None,
        step_name_format: step_lib.NameFormat | None = None,
    ) -> Iterator[int]:
        """Yield available checkpoint steps discovered in ``checkpoint_dir``.

        Args:
            checkpoint_dir: Base directory containing step checkpoints.
            min_interval_secs: Minimum polling interval in seconds. Defaults to 0.
            timeout: Optional polling timeout in seconds. Defaults to None.
            step_prefix: Optional step directory prefix. Defaults to None.
            step_name_format: Optional ``step_lib.NameFormat`` instance. Defaults to None.

        Yields:
            int: Discovered checkpoint step numbers in ascending order.
        """
        if min_interval_secs < 0 or (timeout is not None and timeout < 0):
            return
        active_format = (
            step_name_format
            if step_name_format is not None
            else step_lib.standard_name_format(step_prefix=step_prefix)
        )
        steps: list[int] = []
        for name in FileSystemIO().listdir(checkpoint_dir):
            parsed = active_format.parse_step(name)
            if parsed is not None:
                steps.append(parsed)
        yield from sorted(steps)

    @staticmethod
    def construct_restore_args(
        target: PyTree,
        sharding_tree: PyTree = None,
        set_global_shape: bool = True,
    ) -> PyTree:
        """Construct a PyTree of ``RestoreArgs`` or ``ArrayRestoreArgs`` matching ``target``.

        Args:
            target: Target PyTree providing leaf types, shapes, and dtypes.
            sharding_tree: Optional PyTree of sharding specifications matching ``target``. Defaults to None.
            set_global_shape: Whether to populate ``global_shape`` and ``shape`` on ``ArrayRestoreArgs``. Defaults to True.

        Returns:
            PyTree: PyTree of ``RestoreArgs`` / ``ArrayRestoreArgs`` matching the structure of ``target``.
        """
        if isinstance(target, FlaxStateProtocol):
            sharding_dict = sharding_tree if is_pytree_dict(sharding_tree) else {}
            return {
                str(k): checkpoint_utils.construct_restore_args(
                    v,
                    sharding_dict.get(str(k)),
                    set_global_shape=set_global_shape,
                )
                for k, v in target.items()
            }
        if isinstance(target, FrozenDictProtocol):
            raw_fd = target.unfreeze()
            sharding_fd = sharding_tree if is_pytree_dict(sharding_tree) else {}
            return {
                str(k): checkpoint_utils.construct_restore_args(
                    v,
                    sharding_fd.get(str(k)),
                    set_global_shape=set_global_shape,
                )
                for k, v in raw_fd.items()
            }
        if isinstance(target, NamedTupleProtocol):
            nt_dict = target._asdict()
            sharding_seq = (
                sharding_tree
                if isinstance(sharding_tree, (list, tuple))
                else [None] * len(target._fields)
            )
            return {
                field_name: checkpoint_utils.construct_restore_args(
                    nt_dict.get(field_name),
                    sharding_seq[idx] if idx < len(sharding_seq) else None,
                    set_global_shape=set_global_shape,
                )
                for idx, field_name in enumerate(target._fields)
            }
        if is_pytree_dict(target):
            sharding_sub = sharding_tree if is_pytree_dict(sharding_tree) else {}
            return {
                k: checkpoint_utils.construct_restore_args(
                    v,
                    sharding_sub.get(k),
                    set_global_shape=set_global_shape,
                )
                for k, v in target.items()
            }
        if isinstance(target, list):
            sharding_list = sharding_tree if isinstance(sharding_tree, (list, tuple)) else []
            return [
                checkpoint_utils.construct_restore_args(
                    v,
                    sharding_list[idx] if idx < len(sharding_list) else None,
                    set_global_shape=set_global_shape,
                )
                for idx, v in enumerate(target)
            ]
        if isinstance(target, tuple):
            sharding_tup = sharding_tree if isinstance(sharding_tree, (list, tuple)) else ()
            return tuple(
                checkpoint_utils.construct_restore_args(
                    v,
                    sharding_tup[idx] if idx < len(sharding_tup) else None,
                    set_global_shape=set_global_shape,
                )
                for idx, v in enumerate(target)
            )
        if isinstance(target, FlaxVariableProtocol):
            return checkpoint_utils.construct_restore_args(
                target.value,
                sharding_tree,
                set_global_shape=set_global_shape,
            )
        sharding_val: PartitionSpec | str | None = (
            sharding_tree if isinstance(sharding_tree, (PartitionSpec, str)) else None
        )
        if isinstance(target, SerializedArray):
            shape_val = target.shape if set_global_shape else None
            return ArrayRestoreArgs(
                restore_type=SerializedArray,
                sharding=sharding_val,
                global_shape=shape_val,
                shape=shape_val,
                dtype=target.dtype,
            )
        if is_array_protocol(target) or isinstance(target, CompilerTensor):
            arr_shape = tuple(int(d) for d in target.shape) if set_global_shape else None
            return ArrayRestoreArgs(
                restore_type=type(target),
                sharding=sharding_val,
                global_shape=arr_shape,
                shape=arr_shape,
                dtype=str(target.dtype),
            )
        if target is None:
            return RestoreArgs(restore_type=None, dtype=None)
        return RestoreArgs(restore_type=type(target), dtype=type(target))

    @staticmethod
    def restore_args_from_target(
        target: PyTree,
        sharding_tree: PyTree = None,
    ) -> PyTree:
        """Construct ``RestoreArgs`` or ``ArrayRestoreArgs`` from a target PyTree structure.

        Args:
            target: Target PyTree providing leaf types, shapes, and dtypes.
            sharding_tree: Optional PyTree of sharding specifications matching ``target``. Defaults to None.

        Returns:
            PyTree: PyTree of ``RestoreArgs`` / ``ArrayRestoreArgs`` matching ``target``.
        """
        return checkpoint_utils.construct_restore_args(
            target=target,
            sharding_tree=sharding_tree,
            set_global_shape=True,
        )

    @staticmethod
    def wait_for_new_checkpoint(
        checkpoint_dir: PathLike,
        until_step: int | None = None,
        seconds_to_sleep: int = 1,
        timeout: int | None = None,
        step_prefix: str | None = None,
        step_format_fixed_length: int | None = None,
        step_name_format: step_lib.NameFormat | None = None,
    ) -> int:
        """Wait until a new checkpoint step is present in ``checkpoint_dir``.

        Args:
            checkpoint_dir: Directory containing step checkpoint subdirectories.
            until_step: Optional minimum step number to wait for. Defaults to None.
            seconds_to_sleep: Sleep interval in seconds between directory checks. Defaults to 1.
            timeout: Optional timeout in seconds before returning ``-1``. Defaults to None.
            step_prefix: Optional step prefix for directory parsing. Defaults to None.
            step_format_fixed_length: Optional zero-padded digit width for step directories. Defaults to None.
            step_name_format: Optional custom ``step_lib.NameFormat``. Defaults to None.

        Returns:
            int: Discovered checkpoint step number, or ``-1`` if no matching checkpoint is found before timeout.
        """
        active_format = (
            step_name_format
            if step_name_format is not None
            else step_lib.standard_name_format(
                step_prefix=step_prefix,
                step_format_fixed_length=step_format_fixed_length,
            )
        )
        start_time = time.monotonic()
        while True:
            steps: list[int] = []
            for name in FileSystemIO().listdir(checkpoint_dir):
                parsed = active_format.parse_step(name)
                if parsed is not None:
                    steps.append(parsed)
            if steps:
                latest = max(steps)
                if until_step is None or latest >= until_step:
                    return latest
            if (
                timeout is None
                or (time.monotonic() - start_time) >= timeout
                or seconds_to_sleep <= 0
            ):
                return -1
            time.sleep(min(seconds_to_sleep, 0.01))


class checkpointer:
    """Namespace exporting synchronous ``Checkpointer``."""

    Checkpointer = Checkpointer


class future:
    """Namespace exporting ``Future``."""

    Future = Future


class handlers:
    """Namespace exporting all built-in ``CheckpointHandler`` classes and registries."""

    CheckpointHandler = CheckpointHandler
    AsyncCheckpointHandler = AsyncCheckpointHandler
    ArrayCheckpointHandler = ArrayCheckpointHandler
    BasePyTreeCheckpointHandler = BasePyTreeCheckpointHandler
    CompositeCheckpointHandler = CompositeCheckpointHandler
    JaxRandomKeyCheckpointHandler = JaxRandomKeyCheckpointHandler
    JsonCheckpointHandler = JsonCheckpointHandler
    NumpyRandomKeyCheckpointHandler = NumpyRandomKeyCheckpointHandler
    ProtoCheckpointHandler = ProtoCheckpointHandler
    PyGrainCheckpointHandler = PyGrainCheckpointHandler
    PyTreeCheckpointHandler = PyTreeCheckpointHandler
    RandomKeyCheckpointHandler = RandomKeyCheckpointHandler
    StandardCheckpointHandler = StandardCheckpointHandler
    CheckpointHandlerRegistry = CheckpointHandlerRegistry
    DefaultCheckpointHandlerRegistry = DefaultCheckpointHandlerRegistry
    create_default_handler_registry = staticmethod(create_default_handler_registry)


class metadata:
    """Namespace exporting checkpoint metadata protocols and structures."""

    CheckpointMetadataStore = CheckpointMetadataStoreProtocol
    StepMetadata = step_lib.Metadata


class msgpack_utils:
    """Namespace exporting Msgpack serialization and restoration functions."""

    packb = staticmethod(MsgpackEngine().packb)
    unpackb = staticmethod(MsgpackEngine().unpackb)

    @staticmethod
    def msgpack_serialize(pytree: PyTree, in_place: bool = False) -> bytes:
        """Serialize a Python PyTree into MessagePack binary format.

        Args:
            pytree: The PyTree structure to encode.
            in_place: Whether to serialize array leaves in-place. Defaults to False.

        Returns:
            bytes: Encoded MessagePack byte string.
        """
        _ = in_place
        return MsgpackEngine().packb(pytree)

    @staticmethod
    def msgpack_restore(encoded_pytree: bytes) -> PyTree:
        """Restore a Python PyTree from MessagePack binary bytes.

        Args:
            encoded_pytree: MessagePack byte payload produced by ``msgpack_serialize``.

        Returns:
            PyTree: Decoded Python PyTree structure.
        """
        return MsgpackEngine().unpackb(encoded_pytree)


class v1:
    """Namespace providing Orbax v1 functional checkpointing APIs."""

    @staticmethod
    def save_pytree(
        path: PathLike,
        pytree: PyTree,
        *,
        overwrite: bool = False,
        custom_metadata: Mapping[str, JsonValue] | None = None,
    ) -> None:
        """Synchronously save a PyTree to ``path`` using the Orbax v1 functional interface.

        Args:
            path: Target checkpoint directory path.
            pytree: PyTree object to save.
            overwrite: Whether to overwrite an existing checkpoint at ``path``. Defaults to False.
            custom_metadata: Optional custom JSON metadata dictionary to store alongside the checkpoint. Defaults to None.

        Raises:
            ValueError: If ``path`` already exists and ``overwrite`` is False.
        """
        io = FileSystemIO()
        if io.exists(path):
            if not overwrite:
                raise ValueError(f"Checkpoint path already exists: {path}")
            io.remove(path)
        PyTreeCheckpointer().save(path, pytree, force=overwrite, custom_metadata=custom_metadata)
        if custom_metadata is not None:
            meta_file = f"{str(path).rstrip('/')}/_custom_metadata.json"
            AtomicWrite().atomic_write_bytes(
                meta_file,
                json.dumps(dict(custom_metadata)).encode("utf-8"),
            )

    @staticmethod
    def save_pytree_async(
        path: PathLike,
        pytree: PyTree,
        *,
        overwrite: bool = False,
        custom_metadata: Mapping[str, JsonValue] | None = None,
    ) -> Future:
        """Asynchronously save a PyTree to ``path`` and return a ``Future``.

        Args:
            path: Target checkpoint directory path.
            pytree: PyTree object to save.
            overwrite: Whether to overwrite an existing checkpoint at ``path``. Defaults to False.
            custom_metadata: Optional custom JSON metadata dictionary. Defaults to None.

        Returns:
            Future: Asynchronous completion handle for the background save operation.

        Raises:
            ValueError: If ``path`` already exists and ``overwrite`` is False.
        """
        io = FileSystemIO()
        if io.exists(path):
            if not overwrite:
                raise ValueError(f"Checkpoint path already exists: {path}")
            io.remove(path)
        checkpointer_async = AsyncCheckpointer(PyTreeCheckpointHandler())
        fut = checkpointer_async.save(
            path, pytree, force=overwrite, custom_metadata=custom_metadata
        )
        if custom_metadata is not None:
            meta_file = f"{str(path).rstrip('/')}/_custom_metadata.json"
            AtomicWrite().atomic_write_bytes(
                meta_file,
                json.dumps(dict(custom_metadata)).encode("utf-8"),
            )
        return fut

    @staticmethod
    def load_pytree(
        path: PathLike,
        abstract_pytree: PyTree = None,
    ) -> PyTree:
        """Load and restore a PyTree from ``path``.

        Args:
            path: Source checkpoint directory path.
            abstract_pytree: Optional target PyTree structure or ``RestoreArgs`` template. Defaults to None.

        Returns:
            PyTree: Restored PyTree matching ``abstract_pytree`` if provided.
        """
        return PyTreeCheckpointer().restore(path, item=abstract_pytree)

    @staticmethod
    def save_checkpointables(
        path: PathLike,
        checkpointables: Mapping[str, PyTree],
        *,
        overwrite: bool = False,
        custom_metadata: Mapping[str, JsonValue] | None = None,
    ) -> None:
        """Save a dictionary of named checkpointable items to ``path``.

        Args:
            path: Target checkpoint directory path.
            checkpointables: Mapping from item name to checkpointable PyTree or state object.
            overwrite: Whether to overwrite an existing checkpoint at ``path``. Defaults to False.
            custom_metadata: Optional custom JSON metadata dictionary. Defaults to None.

        Raises:
            ValueError: If ``path`` already exists and ``overwrite`` is False.
        """
        io = FileSystemIO()
        if io.exists(path):
            if not overwrite:
                raise ValueError(f"Checkpoint path already exists: {path}")
            io.remove(path)
        comp_handler = CompositeCheckpointHandler(*checkpointables.keys())
        comp_handler.save(path, args.Composite(**dict(checkpointables)))
        if custom_metadata is not None:
            meta_file = f"{str(path).rstrip('/')}/_custom_metadata.json"
            AtomicWrite().atomic_write_bytes(
                meta_file,
                json.dumps(dict(custom_metadata)).encode("utf-8"),
            )

    @staticmethod
    def load_checkpointables(
        path: PathLike,
        abstract_checkpointables: Mapping[str, PyTree] | None = None,
    ) -> dict[str, PyTree]:
        """Load a dictionary of named checkpointable items from ``path``.

        Args:
            path: Source checkpoint directory path.
            abstract_checkpointables: Optional mapping of item names to target templates. Defaults to None.

        Returns:
            dict[str, PyTree]: Dictionary mapping item names to restored objects.
        """
        comp_handler = CompositeCheckpointHandler()
        restored = comp_handler.restore(path, item=abstract_checkpointables)
        return dict(restored) if is_pytree_dict(restored) else {}


save_pytree = v1.save_pytree
save_pytree_async = v1.save_pytree_async
load_pytree = v1.load_pytree
save_checkpointables = v1.save_checkpointables
load_checkpointables = v1.load_checkpointables


class path:
    """Namespace exporting step path formatting and atomic write utilities."""

    step = step_lib
    AtomicWrite = AtomicWrite
    FileSystemIO = FileSystemIO


class pytree_checkpointer:
    """Namespace exporting ``PyTreeCheckpointer``."""

    PyTreeCheckpointer = PyTreeCheckpointer


class serialization:
    """Namespace exporting tensor and PyTree serialization primitives."""

    TensorStoreDriver = TensorStoreDriver
    MsgpackEngine = MsgpackEngine
    SerializedArray = SerializedArray


class standard_checkpointer:
    """Namespace exporting ``StandardCheckpointer``."""

    StandardCheckpointer = StandardCheckpointer


class test_utils:
    """Namespace providing test directory setup and synchronization utilities."""

    @staticmethod
    def erase_and_create_empty(directory: PathLike) -> pathlib.Path:
        """Delete ``directory`` if present and recreate it as an empty directory.

        Args:
            directory: Target directory path to reset.

        Returns:
            pathlib.Path: Normalized path to the newly created empty directory.
        """
        io = FileSystemIO()
        io.remove(directory)
        io.makedirs(directory, exist_ok=True)
        return normalize_path(directory)


class transform_utils:
    """Namespace exporting PyTree transformation classes and functions."""

    Transform = Transform
    RestoreTransform = RestoreTransform
    apply_transformations = staticmethod(apply_transformations)
    merge_trees = staticmethod(merge_trees)


class type_handlers:
    """Namespace exporting leaf type handler protocols and argument classes."""

    SaveArgs = SaveArgs
    RestoreArgs = RestoreArgs
    ArrayRestoreArgs = ArrayRestoreArgs
    SingleReplicaArrayRestoreArgs = SingleReplicaArrayRestoreArgs
    TypeHandler = TypeHandlerProtocol
    TypeHandlerRegistry = TypeHandlerRegistryProtocol


class utils:
    """Namespace exporting general filesystem and PyTree helper utilities."""

    normalize_path = staticmethod(normalize_path)
    cleanup_tmp_directories = staticmethod(FileSystemIO().remove)


__all__ = [
    "AbstractCheckpointManager",
    "AbstractCheckpointer",
    "ArrayCheckpointHandler",
    "ArrayRestoreArgs",
    "AsyncCheckpointHandler",
    "AsyncCheckpointer",
    "AsyncOptions",
    "BasePyTreeCheckpointHandler",
    "CheckpointHandler",
    "CheckpointHandlerRegistry",
    "CheckpointManager",
    "CheckpointManagerOptions",
    "Checkpointer",
    "CompositeCheckpointHandler",
    "CompositeOptions",
    "DefaultCheckpointHandlerRegistry",
    "Future",
    "JaxRandomKeyCheckpointHandler",
    "JsonCheckpointHandler",
    "MultiprocessingOptions",
    "NumpyRandomKeyCheckpointHandler",
    "ProtoCheckpointHandler",
    "PyGrainCheckpointHandler",
    "PyTreeCheckpointHandler",
    "PyTreeCheckpointer",
    "RandomKeyCheckpointHandler",
    "RestoreArgs",
    "RestoreTransform",
    "SaveArgs",
    "SingleReplicaArrayRestoreArgs",
    "StandardCheckpointHandler",
    "StandardCheckpointer",
    "Transform",
    "__version__",
    "abstract_checkpoint_manager",
    "abstract_checkpointer",
    "abstract_logger",
    "aggregate_handlers",
    "apply_transformations",
    "args",
    "async_checkpoint_handler",
    "async_checkpointer",
    "asyncio",
    "atomicity",
    "checkpoint",
    "checkpoint_args",
    "checkpoint_handler",
    "checkpoint_manager",
    "checkpoint_utils",
    "checkpointer",
    "contextlib",
    "create_default_handler_registry",
    "epath",
    "export",
    "functools",
    "future",
    "handlers",
    "load_checkpointables",
    "load_pytree",
    "logging",
    "merge_trees",
    "metadata",
    "msgpack_utils",
    "multihost",
    "nest_asyncio",
    "options",
    "orbax",
    "path",
    "pytree_checkpointer",
    "save_checkpointables",
    "save_pytree",
    "save_pytree_async",
    "serialization",
    "standard_checkpointer",
    "step",
    "step_lib",
    "test_utils",
    "transform_utils",
    "tree",
    "type_handlers",
    "utils",
    "v1",
]
