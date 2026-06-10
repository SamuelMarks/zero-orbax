"""Provide module docstring."""

import pytest
from absl.testing import parameterized
import zero_orbax.checkpoint as ocp
import numpy as np


class CheckpointManagerTest(parameterized.TestCase):
    """Represent the class."""

    def setUp(self):
        """Execute the function."""
        super().setUp()
        self.directory = "/tmp/mock_ckpt_dir"

    def test_save_and_restore(self):
        """Execute the function."""
        options = ocp.CheckpointManagerOptions(max_to_keep=2)
        manager = ocp.CheckpointManager(self.directory, options=options)
        saved = manager.save(1, {"a": 1, "b": 2})
        self.assertTrue(saved)
        self.assertEqual(manager.latest_step(), 1)
        self.assertEqual(manager.all_steps(), [1])
        restored = manager.restore(1)
        self.assertEqual(restored, {"a": 1, "b": 2})
        manager.save(2, {"a": 3, "b": 4})
        manager.save(3, {"a": 5, "b": 6})
        self.assertEqual(manager.latest_step(), 3)
        self.assertEqual(manager.all_steps(), [2, 3])
        with self.assertRaises(ValueError):
            manager.restore(1)
