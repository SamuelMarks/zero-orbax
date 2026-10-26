"""Functional v1 checkpointing APIs for zero_orbax.checkpoint.v1."""

from __future__ import annotations

import json
from collections.abc import Mapping

from zero_orbax._checkpoint.future_lib import Future
from zero_orbax._checkpoint.handlers_lib import (
    CompositeCheckpointHandler,
    PyTreeCheckpointHandler,
)
from zero_orbax.core import AtomicWrite, FileSystemIO
from zero_orbax.types import JsonValue, PathLike, PyTree, is_pytree_dict


class v1:
    """Orbax Checkpoint v1 functional API namespace (`orbax.checkpoint.v1`)."""

    @staticmethod
    def save_pytree(
        path: PathLike,
        pytree: PyTree,
        *,
        overwrite: bool = False,
        custom_metadata: Mapping[str, JsonValue] | None = None,
    ) -> None:
        """Synchronously save a PyTree to ``path`` using the v1 functional interface.

        Args:
            path: Target checkpoint directory path.
            pytree: The PyTree structure to save.
            overwrite: Whether to overwrite ``path`` if it already exists. Defaults to False.
            custom_metadata: Optional custom metadata dictionary to store alongside the checkpoint. Defaults to None.

        Raises:
            ValueError: If ``path`` already exists and ``overwrite=False``.
        """
        io = FileSystemIO()
        if io.exists(path):
            if not overwrite:
                raise ValueError(f"Checkpoint path '{path}' already exists and overwrite=False.")
            io.remove(path)
        handler = PyTreeCheckpointHandler()
        handler.save(path, pytree)
        handler.finalize(path)
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
            pytree: The PyTree structure to save.
            overwrite: Whether to overwrite ``path`` if it already exists. Defaults to False.
            custom_metadata: Optional custom metadata dictionary. Defaults to None.

        Returns:
            Future: Future representing the asynchronous save operation.

        Raises:
            ValueError: If ``path`` already exists and ``overwrite=False``.
        """
        io = FileSystemIO()
        if io.exists(path):
            if not overwrite:
                raise ValueError(f"Checkpoint path '{path}' already exists and overwrite=False.")
            io.remove(path)
        handler = PyTreeCheckpointHandler()
        sub_futures = handler.async_save(path, pytree) or []

        def _commit() -> None:
            """Finalize all PyTree handler futures and write custom metadata."""
            for sf in sub_futures:
                sf.result()
            if custom_metadata is not None:
                meta_file = f"{str(path).rstrip('/')}/_custom_metadata.json"
                AtomicWrite().atomic_write_bytes(
                    meta_file,
                    json.dumps(dict(custom_metadata)).encode("utf-8"),
                )

        return Future(result=None, commit_callback=_commit)

    @staticmethod
    def load_pytree(
        path: PathLike,
        abstract_pytree: PyTree = None,
    ) -> PyTree:
        """Load a PyTree from ``path`` using the v1 functional interface.

        Args:
            path: Source checkpoint directory path.
            abstract_pytree: Optional target template guiding container and array restoration. Defaults to None.

        Returns:
            PyTree: Restored PyTree structure.
        """
        return PyTreeCheckpointHandler().restore(path, item=abstract_pytree)

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
            checkpointables: Mapping of item names to checkpointable objects.
            overwrite: Whether to overwrite ``path`` if it already exists. Defaults to False.
            custom_metadata: Optional custom metadata dictionary. Defaults to None.

        Raises:
            ValueError: If ``path`` already exists and ``overwrite=False``.
        """
        io = FileSystemIO()
        if io.exists(path):
            if not overwrite:
                raise ValueError(f"Checkpoint path '{path}' already exists and overwrite=False.")
            io.remove(path)
        comp_handler = CompositeCheckpointHandler(*checkpointables.keys())
        comp_handler.save(path, dict(checkpointables))
        comp_handler.finalize(path)
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

    @staticmethod
    def custom_metadata(path: PathLike) -> dict[str, JsonValue] | None:
        """Read custom metadata persisted alongside a v1 checkpoint at ``path``.

        Args:
            path: Source checkpoint directory path.

        Returns:
            dict[str, JsonValue] | None: Custom metadata dictionary, or None if not present.
        """
        meta_file = f"{str(path).rstrip('/')}/_custom_metadata.json"
        raw = FileSystemIO().read(meta_file)
        if not raw:
            return None
        parsed: dict[str, JsonValue] = json.loads(raw.decode("utf-8"))
        return parsed


save_pytree = v1.save_pytree
save_pytree_async = v1.save_pytree_async
load_pytree = v1.load_pytree
save_checkpointables = v1.save_checkpointables
load_checkpointables = v1.load_checkpointables

__all__ = [
    "load_checkpointables",
    "load_pytree",
    "save_checkpointables",
    "save_pytree",
    "save_pytree_async",
    "v1",
]
