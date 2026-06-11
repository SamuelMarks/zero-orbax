"""Provide module docstring."""

import pytest
from absl.testing import parameterized
import zero_orbax.checkpoint as ocp


class ArgsTest(parameterized.TestCase):
    """Test suite for checkpoint arguments and handlers registration."""

    def test_args_composite(self):
        """Test the Composite args class instantiation.

        Returns:
            None
        """
        composite = ocp.args.Composite(a=1, b=2)
        self.assertIsNotNone(composite)

    def test_args_save_restore(self):
        """Test various predefined save and restore args.

        Returns:
            None
        """
        self.assertIsNotNone(ocp.args.PyTreeSave(item=1))
        self.assertIsNotNone(ocp.args.PyTreeRestore(item=1))
        self.assertIsNotNone(ocp.args.StandardSave(item=1))
        self.assertIsNotNone(ocp.args.StandardRestore(item=1))

    def test_args_registration(self):
        """Test argument and handler registry mechanisms.

        Returns:
            None
        """
        self.assertFalse(ocp.args.has_registered_args(int))
        self.assertIsNone(ocp.args.get_registered_handler_cls(int))
        self.assertIsNone(ocp.args.get_registered_args_cls(int))

        @ocp.args.register_with_handler(int, for_save=True)
        class CustomSaveArgs:
            """Dummy args class for testing registration."""

            pass

        self.assertIsNotNone(CustomSaveArgs)
