"""Future completion handle for asynchronous checkpointing in zero_orbax.checkpoint."""

from __future__ import annotations

from collections.abc import Callable

from zero_orbax.types import PyTree


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
        self._exception: BaseException | None = None

    def result(self, timeout: float | None = None) -> PyTree:
        """Wait for completion and return the wrapped result.

        Args:
            timeout: Optional timeout in seconds (must be non-negative if provided). Defaults to None.

        Returns:
            PyTree: The result stored in this future.

        Raises:
            ValueError: If ``timeout`` is negative.
            BaseException: Any exception raised by the underlying commit callback.
        """
        if timeout is not None and timeout < 0:
            raise ValueError(f"Future timeout must be non-negative, got {timeout}.")
        if self._exception is not None:
            raise self._exception
        if not self._done and self._commit_callback is not None:
            try:
                self._commit_callback()
            except BaseException as exc:
                self._exception = exc
                self._done = True
                raise
            self._done = True
        return self._result

    def done(self) -> bool:
        """Return whether the asynchronous operation has completed.

        Returns:
            bool: True if finalized, False if a commit callback is still pending.
        """
        return self._done


class future:
    """Namespace exporting ``Future``."""

    Future = Future


__all__ = ["Future", "future"]
