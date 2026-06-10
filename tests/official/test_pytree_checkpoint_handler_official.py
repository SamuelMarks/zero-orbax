"""Provide module docstring."""

import pytest
from absl.testing import parameterized
import zero_orbax.checkpoint as ocp


class PyTreeCheckpointHandlerTest(parameterized.TestCase):
    """Represent the class."""

    def test_handler_initialization(self):
        """Execute the function."""
        handler = ocp.checkpoint_handler.CheckpointHandler()
        self.assertIsNotNone(handler)

    def test_merge_trees(self):
        """Execute the function."""
        tree1 = {"a": 1, "b": {"c": 2}}
        tree2 = {"b": {"d": 3}, "e": 4}
        merged = ocp.merge_trees(tree1, tree2)
        self.assertEqual(merged, {"a": 1, "b": {"c": 2, "d": 3}, "e": 4})

    def test_apply_transformations(self):
        """Execute the function."""
        orig = {"a": 1, "b": {"c": 2}}
        transformations = {
            "a": ocp.Transform(original_key="a", value_fn=lambda x: x * 10),
            "b": {"c": ocp.Transform(original_key="c", value_fn=lambda x: x * 20)},
        }
        new_t = {"a": None, "b": {"c": None}}
        res = ocp.apply_transformations(orig, transformations, new_t)
        self.assertEqual(res, {"a": 10, "b": {"c": 40}})
