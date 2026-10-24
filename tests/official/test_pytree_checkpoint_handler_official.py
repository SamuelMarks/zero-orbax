"""Official-style unit tests for PyTreeCheckpointHandler, merge_trees, and transformations."""

from __future__ import annotations

import tempfile

from absl.testing import parameterized

import zero_orbax.checkpoint as ocp
from zero_orbax.types import PyTree


class PyTreeCheckpointHandlerTest(parameterized.TestCase):
    """Test suite for handling and transforming PyTree structures during checkpointing."""

    def test_handler_save_and_restore_with_transforms(self) -> None:
        """Test saving and restoring nested PyTrees with PyTreeCheckpointHandler and transforms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = ocp.PyTreeCheckpointHandler(aggregate_filename="custom_pytree.msgpack")
            original_tree: PyTree = {
                "encoder": {"weights": [1, 2, 3], "bias": 10},
                "step": 5,
            }
            handler.save(tmpdir, original_tree)
            self.assertEqual(
                handler.get_param_names(original_tree),
                {
                    "encoder": {"weights": "encoder.weights", "bias": "encoder.bias"},
                    "step": "step",
                },
            )
            transforms: PyTree = {
                "decoder_bias": ocp.Transform(
                    original_key="encoder/bias",
                    value_fn=lambda v: v * 2 if isinstance(v, int) else v,
                ),
                "step": ocp.Transform(original_key="step"),
            }
            target_tree: PyTree = {"decoder_bias": 0, "step": 0}
            restored = handler.restore(
                tmpdir,
                item=target_tree,
                transforms=transforms,
                transforms_default_to_original=False,
            )
            self.assertEqual(restored, {"decoder_bias": 20, "step": 5})
            handler.close()

    def test_merge_trees(self) -> None:
        """Test merging multiple overlapping PyTree dictionaries."""
        tree1: dict[str, PyTree] = {"a": 1, "b": {"c": 2}}
        tree2: dict[str, PyTree] = {"b": {"d": 3}, "e": 4}
        merged = ocp.merge_trees(tree1, tree2)
        self.assertEqual(merged, {"a": 1, "b": {"c": 2, "d": 3}, "e": 4})

    def test_apply_transformations(self) -> None:
        """Test applying a set of transformation rules to a PyTree."""

        def _scale10(val: PyTree) -> PyTree:
            """Multiply an integer leaf by 10.

            Args:
                val: Input PyTree leaf.

            Returns:
                PyTree: Scaled integer leaf.
            """
            return val * 10 if isinstance(val, int) else val

        def _scale20(val: PyTree) -> PyTree:
            """Multiply an integer leaf by 20.

            Args:
                val: Input PyTree leaf.

            Returns:
                PyTree: Scaled integer leaf.
            """
            return val * 20 if isinstance(val, int) else val

        orig: PyTree = {"a": 1, "b": {"c": 2}}
        transformations: PyTree = {
            "a": ocp.Transform(original_key="a", value_fn=_scale10),
            "b": {"c": ocp.Transform(original_key="c", value_fn=_scale20)},
        }
        new_t: PyTree = {"a": None, "b": {"c": None}}
        res = ocp.apply_transformations(orig, transformations, new_t)
        self.assertEqual(res, {"a": 10, "b": {"c": 40}})
