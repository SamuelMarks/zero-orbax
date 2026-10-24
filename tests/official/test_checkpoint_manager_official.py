"""Official-style unit tests for CheckpointManager multi-step save and restore."""

from __future__ import annotations

import pathlib
import tempfile

from absl.testing import parameterized

import zero_orbax.checkpoint as ocp


class CheckpointManagerTest(parameterized.TestCase):
    """Test suite for the CheckpointManager."""

    def test_save_and_restore(self) -> None:
        """Test full save, restore, pruning, and reload workflows over multiple steps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            options = ocp.CheckpointManagerOptions(max_to_keep=2)
            manager = ocp.CheckpointManager(tmpdir, options=options)
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

            reloaded = ocp.CheckpointManager(tmpdir, options=options)
            reloaded.reload()
            self.assertEqual(reloaded.all_steps(), [2, 3])
            self.assertEqual(reloaded.restore(3), {"a": 5, "b": 6})

    def test_multi_item_checkpointers_and_step_formatting(self) -> None:
        """Test multi-item checkpointers, step_prefix, keep_period, and best_fn retention."""
        with tempfile.TemporaryDirectory() as tmpdir:
            options = ocp.CheckpointManagerOptions(
                max_to_keep=1,
                keep_period=2,
                step_prefix="ckpt",
                step_format_fixed_length=4,
                best_fn=lambda m: float(m["acc"]),
                best_mode="max",
                keep_checkpoints_without_metrics=False,
            )
            checkpointers: dict[str, ocp.AbstractCheckpointer] = {
                "params": ocp.PyTreeCheckpointer(),
                "config": ocp.Checkpointer(ocp.JsonCheckpointHandler()),
            }
            manager = ocp.CheckpointManager(
                tmpdir,
                checkpointers=checkpointers,
                options=options,
            )
            manager.save(
                1,
                {"params": {"w": 1}, "config": {"lr": 0.1}},
                metrics={"acc": 0.99},
            )
            manager.save(
                2,
                {"params": {"w": 2}, "config": {"lr": 0.05}},
                metrics={"acc": 0.50},
            )
            manager.save(
                3,
                {"params": {"w": 3}, "config": {"lr": 0.01}},
                metrics={"acc": 0.60},
            )
            # Step 1 is protected as best_step (acc=0.99); Step 2 is protected by keep_period=2; Step 3 is evicted
            self.assertEqual(manager.best_step(), 1)
            self.assertIn(1, manager.all_steps())
            self.assertIn(2, manager.all_steps())
            self.assertTrue((pathlib.Path(tmpdir) / "ckpt_0001").is_dir())
            self.assertEqual(
                manager.restore(1),
                {"params": {"w": 1}, "config": {"lr": 0.1}},
            )
            manager.close()
