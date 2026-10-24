"""Domain types, protocols, and type guards for zero-orbax.

This module provides strongly typed data structures, callback signatures,
and runtime-checkable protocols used throughout ``zero_orbax`` without relying
on ``typing.Any``, ``object``, or dynamic attribute reflection.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import pathlib
from collections.abc import Callable, Mapping, Sequence
from typing import Optional, Protocol, Union, runtime_checkable
from typing_extensions import TypeAlias, TypeGuard

from ml_switcheroo_ir import AttributeValue, DType, LogicalGraph, LogicalMesh, PartitionSpec


@runtime_checkable
class PathLikeProtocol(Protocol):
    """Protocol for objects implementing ``__fspath__``."""

    def __fspath__(self) -> str:
        """Return the file system path representation of the object.

        Returns:
            str: String file system path.
        """
        ...


ScalarLeaf: TypeAlias = Union[int, float, bool, str, bytes, complex, None]
JsonScalar: TypeAlias = Union[int, float, bool, str, None]
JsonValue: TypeAlias = Union[
    JsonScalar,
    list["JsonValue"],
    dict[str, "JsonValue"],
    tuple["JsonValue", ...],
]
PathLike: TypeAlias = Union[str, PathLikeProtocol, pathlib.Path]


@runtime_checkable
class ArrayProtocol(Protocol):
    """Runtime-checkable protocol for array and tensor leaves.

    Attributes:
        shape: Tuple of integer dimensions describing the array shape.
        dtype: Data type descriptor or string name.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape dimensions of the array.

        Returns:
            tuple[int, ...]: Dimensions of the array.
        """
        ...

    @property
    def dtype(self) -> str | DType | type:
        """Return the element data type of the array.

        Returns:
            str | DType | type: Data type identifier.
        """
        ...

    def tolist(self) -> PyTree:
        """Convert the array into nested Python lists or a scalar leaf.

        Returns:
            PyTree: Nested list or scalar representation.
        """
        ...


@runtime_checkable
class BytesSerializableProtocol(Protocol):
    """Protocol for objects providing a raw ``tobytes()`` serialization method."""

    def tobytes(self) -> bytes:
        """Serialize the underlying buffer into raw bytes.

        Returns:
            bytes: Raw byte representation.
        """
        ...


@runtime_checkable
class GraphProtocol(Protocol):
    """Protocol for intermediate representation graphs such as IRGraph."""

    name: str
    inputs: list[str]
    outputs: list[str]


PyTreeLeaf: TypeAlias = Union[ScalarLeaf, ArrayProtocol, LogicalGraph, GraphProtocol]
PyTreeDict: TypeAlias = dict[str, "PyTree"]
PyTreeList: TypeAlias = list["PyTree"]
PyTreeTuple: TypeAlias = tuple["PyTree", ...]
PyTree: TypeAlias = Union[
    PyTreeLeaf,
    PyTreeDict,
    PyTreeList,
    PyTreeTuple,
    Mapping[str, "PyTree"],
    Sequence["PyTree"],
    AttributeValue,
]


def is_scalar_leaf(value: PyTree) -> TypeGuard[ScalarLeaf]:
    """Check whether a PyTree value is a primitive scalar leaf.

    Args:
        value: The PyTree value to inspect.

    Returns:
        TypeGuard[ScalarLeaf]: True if the value is a primitive scalar or None.
    """
    return value is None or isinstance(value, (int, float, bool, str, bytes, complex))


def is_pytree_dict(value: PyTree) -> TypeGuard[dict[str, PyTree]]:
    """Check whether a PyTree node is a mutable string-keyed dictionary.

    Args:
        value: The PyTree node to inspect.

    Returns:
        TypeGuard[dict[str, PyTree]]: True if the value is a ``dict``.
    """
    return isinstance(value, dict)


def is_pytree_mapping(value: PyTree) -> TypeGuard[Mapping[str, PyTree]]:
    """Check whether a PyTree node is a mapping with string keys.

    Args:
        value: The PyTree node to inspect.

    Returns:
        TypeGuard[Mapping[str, PyTree]]: True if the value is a ``Mapping``.
    """
    return isinstance(value, Mapping)


def is_array_protocol(value: PyTree) -> TypeGuard[ArrayProtocol]:
    """Check whether a PyTree node satisfies ``ArrayProtocol``.

    Args:
        value: The PyTree node to inspect.

    Returns:
        TypeGuard[ArrayProtocol]: True if the value implements ``ArrayProtocol``.
    """
    return isinstance(value, ArrayProtocol)


def normalize_path(path: PathLike) -> pathlib.Path:
    """Normalize a ``PathLike`` input into a ``pathlib.Path`` instance.

    Args:
        path: String or path-like object to normalize.

    Returns:
        pathlib.Path: Normalized path instance.
    """
    if isinstance(path, pathlib.Path):
        return path
    return pathlib.Path(os.fspath(path))


BarrierSyncFn: TypeAlias = Callable[[str, int], None]
PostFinalizationCallback: TypeAlias = Callable[[], None]
BestFn: TypeAlias = Callable[
    [Mapping[str, Union[float, int]]],
    Union[float, int],
]
ShouldSaveFn: TypeAlias = Callable[[int, Optional[int]], bool]
ShouldKeepFn: TypeAlias = Callable[[int, Optional[datetime.datetime]], bool]
ValueTransformFn: TypeAlias = Callable[[PyTree], PyTree]
MultiValueTransformFn: TypeAlias = Callable[[PyTree], PyTree]
IsPrioritizedKeyFn: TypeAlias = Callable[[tuple[str, ...]], bool]


@runtime_checkable
class TemporaryPathProtocol(Protocol):
    """Protocol for atomic temporary path management during checkpointing."""

    def get(self) -> pathlib.Path:
        """Return the temporary staging directory path.

        Returns:
            pathlib.Path: Temporary path.
        """
        ...

    def get_final(self) -> pathlib.Path:
        """Return the final committed destination path.

        Returns:
            pathlib.Path: Final path.
        """
        ...

    def create(self) -> pathlib.Path:
        """Create and return the temporary staging directory.

        Returns:
            pathlib.Path: Created temporary path.
        """
        ...

    def finalize(self) -> None:
        """Atomically promote the temporary path to the final destination."""
        ...


@runtime_checkable
class CheckpointMetadataStoreProtocol(Protocol):
    """Protocol for reading and writing checkpoint step/root metadata."""

    def write(self, path: PathLike, metadata: Mapping[str, JsonValue]) -> None:
        """Write metadata dictionary to the given path.

        Args:
            path: Target metadata file or directory path.
            metadata: Metadata key-value mapping to persist.
        """
        ...

    def read(self, path: PathLike) -> dict[str, JsonValue] | None:
        """Read metadata dictionary from the given path.

        Args:
            path: Source metadata file or directory path.

        Returns:
            dict[str, JsonValue] | None: Loaded metadata dictionary, or None if missing.
        """
        ...

    def close(self) -> None:
        """Close the metadata store and flush any pending writes."""
        ...


@runtime_checkable
class AbstractLoggerProtocol(Protocol):
    """Protocol for structured checkpoint event logging."""

    def log_entry(self, msg: str, step: int | None = None) -> None:
        """Record a checkpointing log message.

        Args:
            msg: Message string to log.
            step: Optional training step associated with the log entry.
        """
        ...


@runtime_checkable
class SaveDecisionPolicyProtocol(Protocol):
    """Protocol for deciding whether a checkpoint should be saved at a step."""

    def should_save(self, step: int, latest_step: int | None) -> bool:
        """Determine whether to save at ``step``.

        Args:
            step: Candidate training step.
            latest_step: Most recently saved step, or None.

        Returns:
            bool: True if a checkpoint should be saved.
        """
        ...


@runtime_checkable
class PreservationPolicyProtocol(Protocol):
    """Protocol for deciding which historical checkpoints to preserve."""

    def should_preserve(self, steps: Sequence[int]) -> Sequence[int]:
        """Return the subset of steps that should be preserved on disk.

        Args:
            steps: Ordered sequence of existing checkpoint steps.

        Returns:
            Sequence[int]: Steps to retain.
        """
        ...


@runtime_checkable
class ArrayMetadataValidatorProtocol(Protocol):
    """Protocol for validating array shard metadata after save or merge."""

    def validate(self, metadata: Mapping[str, JsonValue]) -> bool:
        """Validate the given array metadata mapping.

        Args:
            metadata: Metadata mapping to validate.

        Returns:
            bool: True if the metadata is valid.
        """
        ...


@runtime_checkable
class TypeHandlerProtocol(Protocol):
    """Protocol for custom leaf type serialization handlers."""

    def typestr(self) -> str:
        """Return the canonical type identifier string.

        Returns:
            str: Type identifier string.
        """
        ...

    def serialize(self, value: PyTree) -> bytes:
        """Serialize a PyTree leaf value into bytes.

        Args:
            value: Leaf value to serialize.

        Returns:
            bytes: Serialized byte representation.
        """
        ...

    def deserialize(self, data: bytes) -> PyTree:
        """Deserialize bytes back into a PyTree leaf value.

        Args:
            data: Raw serialized bytes.

        Returns:
            PyTree: Reconstructed leaf value.
        """
        ...


@runtime_checkable
class TypeHandlerRegistryProtocol(Protocol):
    """Protocol for registering and querying leaf ``TypeHandlerProtocol`` instances."""

    def add(self, ty: type, handler: TypeHandlerProtocol, override: bool = False) -> None:
        """Register a handler for a specific Python type.

        Args:
            ty: Target Python type.
            handler: Type handler instance.
            override: Whether to overwrite an existing registration.
        """
        ...

    def get(self, ty: type) -> TypeHandlerProtocol | None:
        """Retrieve the registered handler for a Python type.

        Args:
            ty: Target Python type.

        Returns:
            TypeHandlerProtocol | None: Registered handler, or None if not found.
        """
        ...

    def has(self, ty: type) -> bool:
        """Check whether a Python type has a registered handler.

        Args:
            ty: Target Python type.

        Returns:
            bool: True if registered.
        """
        ...


@dataclasses.dataclass
class MultiprocessingOptions:
    """Configuration options for multi-process / multi-host coordination.

    Attributes:
        primary_host: Host ID responsible for writing shared metadata. Defaults to 0.
        active_processes: Optional set of active process indices.
        barrier_sync_key_prefix: Optional prefix for distributed barrier keys.
    """

    primary_host: int | None = 0
    active_processes: set[int] | None = None
    barrier_sync_key_prefix: str | None = None


@dataclasses.dataclass
class FileOptions:
    """File-system permission options for created checkpoint directories and files.

    Attributes:
        path_permission_mode: Optional octal permission mode (e.g. ``0o750``).
    """

    path_permission_mode: int | None = None


@dataclasses.dataclass
class AtomicityOptions:
    """Options controlling atomic checkpoint finalization behavior.

    Attributes:
        use_snapshot: Whether to use filesystem snapshot semantics where supported.
    """

    use_snapshot: bool = False


@dataclasses.dataclass
class MemoryLimitOptions:
    """Options controlling host/device memory transfer concurrency limits.

    Attributes:
        max_transfer_concurrent_gb: Maximum concurrent transfer size in gigabytes.
    """

    max_transfer_concurrent_gb: int | None = None


@dataclasses.dataclass
class PyTreeMetadataOptions:
    """Options controlling PyTree metadata serialization fidelity.

    Attributes:
        support_rich_types: Whether to preserve rich type metadata for custom leaves.
    """

    support_rich_types: bool = False


@dataclasses.dataclass
class AsyncOptions:
    """Configuration options for asynchronous checkpointing operations.

    Attributes:
        timeout_secs: Timeout duration in seconds for async completion. Defaults to 300.
        barrier_sync_fn: Function for synchronizing barriers across hosts. Defaults to None.
        post_finalization_callback: Callback executed after finalization completes. Defaults to None.
        create_directories_asynchronously: Whether directories may be created asynchronously. Defaults to True.
    """

    timeout_secs: int = 300
    barrier_sync_fn: BarrierSyncFn | None = None
    post_finalization_callback: PostFinalizationCallback | None = None
    create_directories_asynchronously: bool = True


@dataclasses.dataclass
class CompositeOptions:
    """Configuration options for ``CompositeCheckpointHandler``.

    Attributes:
        multiprocessing_options: Multi-host coordination options.
        temporary_path_class: Optional temporary path implementation class.
        file_options: Optional file permission options.
        atomicity_options: Optional atomicity behavior options.
        async_options: Optional asynchronous execution options.
    """

    multiprocessing_options: MultiprocessingOptions = dataclasses.field(
        default_factory=MultiprocessingOptions
    )
    temporary_path_class: type[TemporaryPathProtocol] | None = None
    file_options: FileOptions | None = None
    atomicity_options: AtomicityOptions | None = None
    async_options: AsyncOptions | None = None


__all__ = [
    "AbstractLoggerProtocol",
    "ArrayMetadataValidatorProtocol",
    "ArrayProtocol",
    "AsyncOptions",
    "AtomicityOptions",
    "BarrierSyncFn",
    "BestFn",
    "BytesSerializableProtocol",
    "CheckpointMetadataStoreProtocol",
    "CompositeOptions",
    "DType",
    "FileOptions",
    "GraphProtocol",
    "IsPrioritizedKeyFn",
    "JsonScalar",
    "JsonValue",
    "LogicalGraph",
    "LogicalMesh",
    "MemoryLimitOptions",
    "MultiValueTransformFn",
    "MultiprocessingOptions",
    "PartitionSpec",
    "PathLike",
    "PostFinalizationCallback",
    "PreservationPolicyProtocol",
    "PyTree",
    "PyTreeDict",
    "PyTreeLeaf",
    "PyTreeList",
    "PyTreeMetadataOptions",
    "PyTreeTuple",
    "SaveDecisionPolicyProtocol",
    "ScalarLeaf",
    "ShouldKeepFn",
    "ShouldSaveFn",
    "TemporaryPathProtocol",
    "TypeHandlerProtocol",
    "TypeHandlerRegistryProtocol",
    "ValueTransformFn",
    "is_array_protocol",
    "is_pytree_dict",
    "is_pytree_mapping",
    "is_scalar_leaf",
    "normalize_path",
]
