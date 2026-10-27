"""Leaf type handler arguments and namespace for zero_orbax.checkpoint.type_handlers."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from ml_switcheroo_ir import DType, LogicalMesh, PartitionSpec
from zero_jax.sharding import (
    Mesh as JaxMesh,
    NamedSharding as JaxNamedSharding,
    PartitionSpec as JaxPartitionSpec,
    Sharding as JaxSharding,
)

from zero_orbax.types import TypeHandlerProtocol, TypeHandlerRegistryProtocol


@dataclasses.dataclass
class SaveArgs:
    """Arguments configuring how an individual leaf or PyTree node is saved.

    Attributes:
        aggregate: Whether to aggregate the leaf into a consolidated msgpack file. Defaults to False.
        dtype: Optional target data type to cast to when saving. Defaults to None.
        chunk_byte_size: Optional target byte size per storage chunk. Defaults to None.
        ocdbt_target_data_file_size: Optional target OCDBT data file size in bytes. Defaults to None.
        shard_axes: Tuple of axis indices along which to shard. Defaults to ``()``.
        check_finite: Whether to validate that all array values are finite on save via ``zero_chex``. Defaults to False.
    """

    aggregate: bool = False
    dtype: str | DType | type | None = None
    chunk_byte_size: int | None = None
    ocdbt_target_data_file_size: int | None = None
    shard_axes: tuple[int, ...] = ()
    check_finite: bool = False


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
    mesh: LogicalMesh | JaxMesh | None = None
    mesh_axes: PartitionSpec | JaxPartitionSpec | Sequence[str | None] | None = None
    sharding: JaxSharding | JaxNamedSharding | PartitionSpec | JaxPartitionSpec | str | None = None
    global_shape: tuple[int, ...] | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | DType | type | None = None
    strict: bool = True

    def __init__(
        self,
        restore_type: type | str | None = None,
        mesh: LogicalMesh | JaxMesh | None = None,
        mesh_axes: PartitionSpec | JaxPartitionSpec | Sequence[str | None] | None = None,
        sharding: JaxSharding | JaxNamedSharding | PartitionSpec | JaxPartitionSpec | str | None = (
            None
        ),
        global_shape: tuple[int, ...] | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: str | DType | type | None = None,
        strict: bool = True,
    ) -> None:
        """Initialize the ArrayRestoreArgs instance.

        Args:
            restore_type: Target array type or identifier for restoration. Defaults to None.
            mesh: Optional ``LogicalMesh`` or ``zero_jax.sharding.Mesh`` device topology. Defaults to None.
            mesh_axes: Optional ``PartitionSpec`` or axis names for mesh sharding. Defaults to None.
            sharding: Optional explicit ``zero_jax.sharding.Sharding`` or ``PartitionSpec`` descriptor. Defaults to None.
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

    single_replica_sharding: (
        JaxSharding | JaxNamedSharding | PartitionSpec | JaxPartitionSpec | str | None
    ) = None

    def __init__(
        self,
        restore_type: type | str | None = None,
        mesh: LogicalMesh | JaxMesh | None = None,
        mesh_axes: PartitionSpec | JaxPartitionSpec | Sequence[str | None] | None = None,
        sharding: JaxSharding | JaxNamedSharding | PartitionSpec | JaxPartitionSpec | str | None = (
            None
        ),
        global_shape: tuple[int, ...] | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: str | DType | type | None = None,
        strict: bool = True,
        single_replica_sharding: (
            JaxSharding | JaxNamedSharding | PartitionSpec | JaxPartitionSpec | str | None
        ) = None,
    ) -> None:
        """Initialize the SingleReplicaArrayRestoreArgs instance.

        Args:
            restore_type: Target array type or identifier for restoration. Defaults to None.
            mesh: Optional ``LogicalMesh`` or ``zero_jax.sharding.Mesh`` device topology. Defaults to None.
            mesh_axes: Optional ``PartitionSpec`` or axis names for mesh sharding. Defaults to None.
            sharding: Optional explicit ``zero_jax.sharding.Sharding`` or ``PartitionSpec`` descriptor. Defaults to None.
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


class type_handlers:
    """Namespace exporting leaf type handler protocols and argument classes."""

    SaveArgs = SaveArgs
    RestoreArgs = RestoreArgs
    ArrayRestoreArgs = ArrayRestoreArgs
    SingleReplicaArrayRestoreArgs = SingleReplicaArrayRestoreArgs
    TypeHandler = TypeHandlerProtocol
    TypeHandlerRegistry = TypeHandlerRegistryProtocol


__all__ = [
    "ArrayRestoreArgs",
    "RestoreArgs",
    "SaveArgs",
    "SingleReplicaArrayRestoreArgs",
    "type_handlers",
]
