"""Official-style unit tests for checkpoint argument wrappers and handler registration."""

from __future__ import annotations

import tempfile

from absl.testing import parameterized

import zero_orbax.checkpoint as ocp


class ArgsTest(parameterized.TestCase):
    """Test suite for checkpoint arguments and handlers registration."""

    def test_args_composite(self) -> None:
        """Test Composite args save and restore round-trip through CompositeCheckpointHandler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            composite_save = ocp.args.Composite(
                state=ocp.args.StandardSave({"step": 5, "weights": [1.0, 2.0]}),
                meta=ocp.args.JsonSave({"epoch": 2, "tag": "v1"}),
            )
            self.assertEqual(len(composite_save), 2)
            self.assertEqual(list(composite_save.keys()), ["state", "meta"])

            handler = ocp.CompositeCheckpointHandler("state", "meta")
            handler.save(tmpdir, composite_save)

            composite_restore = ocp.args.Composite(
                state=ocp.args.StandardRestore(),
                meta=ocp.args.JsonRestore(),
            )
            restored = handler.restore(tmpdir, composite_restore)
            self.assertEqual(
                restored,
                {
                    "state": {"step": 5, "weights": [1.0, 2.0]},
                    "meta": {"epoch": 2, "tag": "v1"},
                },
            )
            handler.close()

    def test_args_save_restore(self) -> None:
        """Test PyTreeSave, PyTreeRestore, StandardSave, StandardRestore, ArraySave, and ArrayRestore round-trips."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pt_dir = f"{tmpdir}/pytree"
            pt_handler = ocp.PyTreeCheckpointHandler()
            pt_handler.save(pt_dir, ocp.args.PyTreeSave(item={"w": 3.5}))
            restored_pt = pt_handler.restore(
                pt_dir,
                ocp.args.PyTreeRestore(
                    item={"w": 0},
                    restore_args={"w": ocp.RestoreArgs(restore_type=int)},
                ),
            )
            self.assertEqual(restored_pt, {"w": 3})

            std_dir = f"{tmpdir}/std"
            std_handler = ocp.StandardCheckpointHandler()
            std_handler.save(std_dir, ocp.args.StandardSave(item={"k": [1, 2]}))
            restored_std = std_handler.restore(std_dir, ocp.args.StandardRestore())
            self.assertEqual(restored_std, {"k": [1, 2]})

            arr_dir = f"{tmpdir}/arr"
            arr_handler = ocp.ArrayCheckpointHandler()
            arr_handler.save(arr_dir, ocp.args.ArraySave(item=[10, 20, 30]))
            restored_arr = arr_handler.restore(
                arr_dir,
                ocp.args.ArrayRestore(
                    restore_args=ocp.ArrayRestoreArgs(restore_type=tuple),
                ),
            )
            self.assertEqual(restored_arr, (10, 20, 30))

    def test_args_registration(self) -> None:
        """Test argument and handler registry mechanisms."""
        self.assertFalse(ocp.args.has_registered_args(int))
        self.assertIsNone(ocp.args.get_registered_handler_cls(int))
        self.assertIsNone(ocp.args.get_registered_args_cls(int))

        class CustomHandler(ocp.CheckpointHandler):
            """Custom handler class for testing registration."""

        @ocp.args.register_with_handler(CustomHandler, for_save=True)
        class CustomSaveArgs(ocp.args.CheckpointArgs):
            """Custom save args class for testing registration."""

        @ocp.args.register_with_handler(CustomHandler, for_restore=True)
        class CustomRestoreArgs(ocp.args.CheckpointArgs):
            """Custom restore args class for testing registration."""

        self.assertEqual(ocp.args.get_registered_handler_cls(CustomSaveArgs), CustomHandler)
        self.assertEqual(ocp.args.get_registered_handler_cls(CustomRestoreArgs), CustomHandler)
        self.assertEqual(
            ocp.args.get_registered_args_cls(CustomHandler, for_save=True),
            CustomSaveArgs,
        )
        self.assertEqual(
            ocp.args.get_registered_args_cls(CustomHandler, for_save=False),
            CustomRestoreArgs,
        )
        self.assertTrue(ocp.args.has_registered_args(CustomHandler))
