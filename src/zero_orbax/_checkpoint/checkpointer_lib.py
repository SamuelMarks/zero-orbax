"""Checkpointers and checkpointer namespaces for zero_orbax.checkpoint."""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from types import TracebackType

from zero_orbax._checkpoint.args_lib import _unwrap_checkpoint_arg
from zero_orbax._checkpoint.future_lib import Future
from zero_orbax._checkpoint.handlers_lib import (
    AsyncCheckpointHandler,
    CheckpointHandler,
    PyTreeCheckpointHandler,
    StandardCheckpointHandler,
)
from zero_orbax.core import (
    AtomicWrite,
    AtomicityTemporaryPath,
    FileSystemIO,
    MsgpackEngine,
)
from zero_orbax.types import (
    AsyncOptions,
    AtomicityOptions,
    BarrierSyncFn,
    CheckpointMetadataStoreProtocol,
    FileOptions,
    JsonValue,
    MultiprocessingOptions,
    PathLike,
    PyTree,
    PyTreeMetadataOptions,
    TemporaryPathProtocol,
    normalize_path,
)


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


def _instantiate_temporary_path(
    temp_cls: type[TemporaryPathProtocol],
    path: PathLike,
    permission_mode: int | None = None,
) -> TemporaryPathProtocol:
    """Instantiate ``temp_cls`` with ``final_path=path`` or fall back to zero-arg construction.

    Args:
        temp_cls: ``TemporaryPathProtocol`` class to instantiate.
        path: Target destination path.
        permission_mode: Optional octal permission mode. Defaults to None.

    Returns:
        TemporaryPathProtocol: Instantiated temporary path object.
    """
    if issubclass(temp_cls, AtomicityTemporaryPath):
        return AtomicityTemporaryPath(final_path=path, permission_mode=permission_mode)
    return temp_cls()


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
            temp_obj = _instantiate_temporary_path(self.temporary_path_class, path, perm_mode)
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
        if self.barrier_sync_fn is not None:
            self.barrier_sync_fn(str(path), 300)
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
        self.timeout_secs = (
            async_options.timeout_secs
            if (async_options is not None and timeout_secs == 300)
            else timeout_secs
        )
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
        perm_mode = (
            self.file_options.path_permission_mode if self.file_options is not None else None
        )
        active_barrier = (
            self.async_options.barrier_sync_fn
            if (self.async_options is not None and self.async_options.barrier_sync_fn is not None)
            else self.barrier_sync_fn
        )
        if isinstance(self.handler, AsyncCheckpointHandler):
            temp_obj: TemporaryPathProtocol | None = None
            active_path: PathLike = path
            if self.temporary_path_class is not None:
                temp_obj = _instantiate_temporary_path(
                    self.temporary_path_class,
                    path,
                    perm_mode,
                )
                temp_obj.create()
                active_path = temp_obj.get()
            elif perm_mode is not None:
                FileSystemIO().makedirs(path, exist_ok=True, permission_mode=perm_mode)
            sub_futures = self.handler.async_save(active_path, item, *pos_args, **kwargs) or []

            def _commit() -> None:
                """Finalize all handler futures and run barrier/post-finalization hooks."""
                for sf in sub_futures:
                    sf.result(timeout=float(self.timeout_secs))
                if temp_obj is not None:
                    temp_obj.finalize()
                if active_barrier is not None:
                    active_barrier(str(path), self.timeout_secs)
                if self.checkpoint_metadata_store is not None:
                    self.checkpoint_metadata_store.write(path, {"path": str(path)})

            fut = Future(result=None, commit_callback=_commit)
        else:
            super().save(path, item, *pos_args, **kwargs)

            def _sync_commit() -> None:
                """Run async_options barrier synchronization after synchronous handler save."""
                if (
                    self.async_options is not None
                    and self.async_options.barrier_sync_fn is not None
                ):
                    self.async_options.barrier_sync_fn(str(path), self.timeout_secs)

            fut = Future(result=None, commit_callback=_sync_commit)
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


class abstract_checkpointer:
    """Namespace exporting ``AbstractCheckpointer``."""

    AbstractCheckpointer = AbstractCheckpointer


class async_checkpointer:
    """Namespace exporting ``AsyncCheckpointer``."""

    AsyncCheckpointer = AsyncCheckpointer


class checkpointer:
    """Namespace exporting ``Checkpointer``."""

    Checkpointer = Checkpointer


class pytree_checkpointer:
    """Namespace exporting ``PyTreeCheckpointer``."""

    PyTreeCheckpointer = PyTreeCheckpointer


class standard_checkpointer:
    """Namespace exporting ``StandardCheckpointer``."""

    StandardCheckpointer = StandardCheckpointer


__all__ = [
    "AbstractCheckpointer",
    "AsyncCheckpointer",
    "Checkpointer",
    "PyTreeCheckpointer",
    "StandardCheckpointer",
    "_instantiate_temporary_path",
    "abstract_checkpointer",
    "async_checkpointer",
    "checkpointer",
    "pytree_checkpointer",
    "standard_checkpointer",
]
