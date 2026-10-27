"""Auxiliary submodule namespaces for zero_orbax.checkpoint."""

from __future__ import annotations

import pathlib

import zero_jax

from zero_orbax._checkpoint.handlers_lib import PyTreeCheckpointHandler
from zero_orbax._checkpoint.transform_lib import apply_transformations, merge_trees
from zero_orbax.core import AtomicityTemporaryPath, BarrierSync, MsgpackEngine, SyncBarrier
from zero_orbax.types import (
    AbstractLoggerProtocol,
    ArrayMetadataValidatorProtocol,
    CheckpointMetadataStoreProtocol,
    PreservationPolicyProtocol,
    PyTree,
    SaveDecisionPolicyProtocol,
    is_pytree_dict,
)


class abstract_logger:
    """Namespace for structured checkpoint logger protocols."""

    AbstractLogger = AbstractLoggerProtocol


class aggregate_handlers:
    """Namespace for aggregate PyTree checkpoint handlers."""

    MsgpackHandler = PyTreeCheckpointHandler


class atomicity:
    """Namespace for atomic path protocols and helpers."""

    TemporaryPath = AtomicityTemporaryPath


class checkpoint:
    """Namespace for checkpoint metadata store protocols."""

    CheckpointMetadataStore = CheckpointMetadataStoreProtocol


class epath:
    """Namespace providing ``etils.epath.Path`` compatibility via ``pathlib.Path``."""

    Path = pathlib.Path


class metadata:
    """Namespace for array and checkpoint metadata validation protocols."""

    ArrayMetadataValidator = ArrayMetadataValidatorProtocol
    CheckpointMetadataStore = CheckpointMetadataStoreProtocol


class msgpack_utils:
    """Namespace for MessagePack binary packing and unpacking utilities."""

    @staticmethod
    def msgpack_serialize(pytree: PyTree, in_place: bool = False) -> bytes:
        """Serialize a PyTree into MessagePack binary bytes.

        Args:
            pytree: PyTree to serialize.
            in_place: Whether to serialize in-place (retained for Orbax API parity). Defaults to False.

        Returns:
            bytes: Packed MessagePack bytes.
        """
        _ = in_place
        return MsgpackEngine().packb(pytree)

    @staticmethod
    def msgpack_restore(encoded_pytree: bytes) -> PyTree:
        """Deserialize MessagePack binary bytes back into a PyTree.

        Args:
            encoded_pytree: Packed MessagePack bytes.

        Returns:
            PyTree: Reconstructed PyTree.
        """
        return MsgpackEngine().unpackb(encoded_pytree)


class multihost:
    """Namespace for multi-host process indices and barrier synchronization."""

    SyncBarrier = SyncBarrier
    BarrierSync = BarrierSync

    @staticmethod
    def process_index() -> int:
        """Return the current host process index via ``zero_jax.process_index()``.

        Returns:
            int: Current host process index.
        """
        return zero_jax.process_index()

    @staticmethod
    def process_count() -> int:
        """Return the total number of host processes via ``zero_jax.process_count()``.

        Returns:
            int: Total process count.
        """
        return zero_jax.process_count()

    @staticmethod
    def sync_global_processes(name: str) -> None:
        """Synchronize all host processes at a named barrier.

        Args:
            name: Barrier identifier string.
        """
        BarrierSync().wait(name)


class nest_asyncio:
    """Namespace stub for ``nest_asyncio`` loop compatibility."""

    @staticmethod
    def apply() -> None:
        """Apply nested event-loop compatibility (no-op in zero-orbax)."""
        return None


class path:
    """Namespace for path and atomicity utilities."""

    atomicity = atomicity


class policies:
    """Namespace for save-decision and preservation policy protocols."""

    SaveDecisionPolicy = SaveDecisionPolicyProtocol
    PreservationPolicy = PreservationPolicyProtocol


class serialization:
    """Namespace for low-level tensor and msgpack serialization utilities."""

    msgpack_serialize = staticmethod(msgpack_utils.msgpack_serialize)
    msgpack_restore = staticmethod(msgpack_utils.msgpack_restore)


class tree:
    """Namespace for PyTree inspection and manipulation helpers."""

    @staticmethod
    def to_flat_dict(pytree: PyTree, sep: str = "/") -> dict[str, PyTree]:
        """Flatten a nested PyTree dictionary into a single-level dictionary.

        Args:
            pytree: Nested PyTree dictionary to flatten.
            sep: Key path separator string. Defaults to ``"/"``.

        Returns:
            dict[str, PyTree]: Flattened dictionary mapping joined paths to leaves.
        """
        flat: dict[str, PyTree] = {}
        if not is_pytree_dict(pytree):
            return flat
        for k, v in pytree.items():
            if is_pytree_dict(v) and v:
                for sub_k, sub_v in tree.to_flat_dict(v, sep=sep).items():
                    flat[f"{k}{sep}{sub_k}"] = sub_v
            else:
                flat[k] = v
        return flat


class utils:
    """Namespace for general checkpoint path and transformation utilities."""

    apply_transformations = staticmethod(apply_transformations)
    merge_trees = staticmethod(merge_trees)


__all__ = [
    "abstract_logger",
    "aggregate_handlers",
    "atomicity",
    "checkpoint",
    "epath",
    "metadata",
    "msgpack_utils",
    "multihost",
    "nest_asyncio",
    "path",
    "policies",
    "serialization",
    "tree",
    "utils",
]
