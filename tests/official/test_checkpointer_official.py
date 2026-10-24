"""Official-style unit tests for Checkpointer, AsyncCheckpointer, and specialized variants."""

from __future__ import annotations

import pathlib
import tempfile

from absl.testing import parameterized

import zero_orbax.checkpoint as ocp
from zero_orbax.types import PathLike, PyTree


class RecordingCheckpointHandler(ocp.AsyncCheckpointHandler):
    """Concrete test handler recording save and restore invocations on disk."""

    def __init__(self) -> None:
        """Initialize the RecordingCheckpointHandler."""
        super().__init__()
        self.save_calls: list[tuple[str, PyTree, dict[str, PyTree]]] = []
        self.restore_calls: list[tuple[str, PyTree, dict[str, PyTree]]] = []

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Record save invocation and persist to disk.

        Args:
            directory: Target directory.
            item: Item to save.
            *pos_args: Extra positional args.
            **kwargs: Extra keyword args.
        """
        self.save_calls.append((str(directory), item, dict(kwargs)))
        super().save(directory, item, *pos_args, **kwargs)

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> PyTree:
        """Record restore invocation and restore from disk.

        Args:
            directory: Source directory.
            item: Template item.
            *pos_args: Extra positional args.
            **kwargs: Extra keyword args.

        Returns:
            PyTree: Restored item.
        """
        self.restore_calls.append((str(directory), item, dict(kwargs)))
        return super().restore(directory, item, *pos_args, **kwargs)


class CheckpointerTest(parameterized.TestCase):
    """Test suite for the Checkpointer class and its variants."""

    def test_checkpointer_save_and_restore(self) -> None:
        """Test synchronous saving and restoring using Checkpointer against real disk storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = RecordingCheckpointHandler()
            checkpointer = ocp.Checkpointer(handler)
            checkpointer.save(tmpdir, {"item": 1}, force=True)
            self.assertEqual(len(handler.save_calls), 1)
            self.assertEqual(handler.save_calls[0], (tmpdir, {"item": 1}, {"force": True}))
            self.assertTrue((pathlib.Path(tmpdir) / "checkpoint.msgpack").is_file())

            restored = checkpointer.restore(tmpdir, item={"item": 0}, fallback=True)
            self.assertEqual(len(handler.restore_calls), 1)
            self.assertEqual(
                handler.restore_calls[0],
                (tmpdir, {"item": 0}, {"fallback": True}),
            )
            self.assertEqual(restored, {"item": 1})
            checkpointer.close()

    def test_async_checkpointer_save_and_restore(self) -> None:
        """Test asynchronous saving and restoring using AsyncCheckpointer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = RecordingCheckpointHandler()
            barrier_calls: list[tuple[str, int]] = []
            finalized: list[bool] = []
            async_opts = ocp.AsyncOptions(
                timeout_secs=60,
                barrier_sync_fn=lambda k, t: barrier_calls.append((k, t)),
                post_finalization_callback=lambda: finalized.append(True),
            )
            checkpointer = ocp.AsyncCheckpointer(
                handler,
                async_options=async_opts,
                timeout_secs=60,
            )
            future = checkpointer.save(tmpdir, {"item": 1})
            self.assertFalse(future.done())
            checkpointer.wait_until_finished()
            self.assertTrue(future.done())
            self.assertEqual(len(barrier_calls), 1)
            self.assertEqual(finalized, [True])

            restored = checkpointer.restore(tmpdir)
            self.assertIsInstance(restored, ocp.Future)
            assert isinstance(restored, ocp.Future)
            self.assertEqual(restored.result(), {"item": 1})
            checkpointer.close()

    def test_pytree_checkpointer(self) -> None:
        """Test PyTreeCheckpointer save and restore round-trip on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpointer = ocp.PyTreeCheckpointer()
            checkpointer.save(tmpdir, {"layer": {"w": [1, 2, 3], "b": 0.5}})
            restored = checkpointer.restore(tmpdir)
            self.assertEqual(restored, {"layer": {"w": [1, 2, 3], "b": 0.5}})
            checkpointer.close()

    def test_standard_checkpointer(self) -> None:
        """Test StandardCheckpointer save and restore round-trip on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpointer = ocp.StandardCheckpointer()
            fut = checkpointer.save(tmpdir, {"step": 10, "params": {"x": 42}})
            checkpointer.wait_until_finished()
            self.assertTrue(fut.done())
            restored = checkpointer.restore(tmpdir)
            self.assertEqual(restored, {"step": 10, "params": {"x": 42}})
            checkpointer.close()
