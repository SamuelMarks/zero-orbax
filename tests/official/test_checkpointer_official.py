"""Provide module docstring."""

import pytest
from absl.testing import parameterized
import zero_orbax.checkpoint as ocp
from unittest import mock


class CheckpointerTest(parameterized.TestCase):
    """Represent the class."""

    def setUp(self):
        """Execute the function."""
        super().setUp()
        self.directory = "/tmp/mock_checkpointer_dir"
        self.mock_handler = mock.MagicMock()

    def test_checkpointer_save_and_restore(self):
        """Execute the function."""
        checkpointer = ocp.Checkpointer(self.mock_handler)
        checkpointer.save(self.directory, {"item": 1}, force=True)
        self.mock_handler.save.assert_called_once_with(
            self.directory, {"item": 1}, force=True
        )
        self.mock_handler.restore.return_value = {"item": 1}
        restored = checkpointer.restore(self.directory, item={"item": 0}, fallback=True)
        self.mock_handler.restore.assert_called_once_with(
            self.directory, {"item": 0}, fallback=True
        )
        self.assertEqual(restored, {"item": 1})

    def test_async_checkpointer_save_and_restore(self):
        """Execute the function."""
        checkpointer = ocp.AsyncCheckpointer(self.mock_handler)
        future = checkpointer.save(self.directory, {"item": 1})
        self.mock_handler.save.assert_called_once_with(self.directory, {"item": 1})
        checkpointer.wait_until_finished()
        self.mock_handler.restore.return_value = {"item": 1}
        restored = checkpointer.restore(self.directory)
        self.assertEqual(restored.result(), {"item": 1})

    def test_pytree_checkpointer(self):
        """Execute the function."""
        checkpointer = ocp.PyTreeCheckpointer()
        self.assertIsInstance(checkpointer, ocp.PyTreeCheckpointer)

    def test_standard_checkpointer(self):
        """Execute the function."""
        checkpointer = ocp.StandardCheckpointer()
        self.assertIsInstance(checkpointer, ocp.StandardCheckpointer)
