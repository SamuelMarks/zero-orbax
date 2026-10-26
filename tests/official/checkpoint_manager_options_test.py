"""Official-style unit tests for CheckpointManagerOptions validation and configuration."""

from __future__ import annotations

import dataclasses
import datetime
from typing import TypedDict

from absl.testing import parameterized

import zero_orbax.checkpoint as ocp
from zero_orbax.types import ShouldSaveFn


class CustomNameFormat(ocp.step.NameFormat):
    """Concrete non-StandardNameFormat step naming implementation for testing."""


CUSTOM_STEP_NAME_FORMAT = CustomNameFormat(step_prefix="custom")


class ReadOnlyOptionSpec(TypedDict, total=False):
    """Typed dictionary for parameterized read_only option test cases."""

    save_interval_steps: int
    max_to_keep: int | None
    keep_time_interval: datetime.timedelta | None
    keep_period: int | None
    create: bool
    cleanup_tmp_directories: bool
    save_on_steps: list[int] | None
    todelete_subdir: str | None
    todelete_full_path: str | None
    should_save_fn: ShouldSaveFn | None


def _save_every_even_step(step: int, latest: int | None) -> bool:
    """Return True when ``step`` is even and strictly greater than ``latest``.

    Args:
        step: Candidate step.
        latest: Latest step.

    Returns:
        bool: True if ``step`` is even and greater than ``latest`` (or ``latest is None``).
    """
    return step % 2 == 0 and (latest is None or step > latest)


class CheckpointManagerOptionsTest(parameterized.TestCase):
    """Test suite for validating the CheckpointManagerOptions dataclass."""

    @parameterized.parameters(
        (ReadOnlyOptionSpec(save_interval_steps=1),),
        (ReadOnlyOptionSpec(save_interval_steps=0, max_to_keep=1),),
        (
            ReadOnlyOptionSpec(
                save_interval_steps=0,
                keep_time_interval=datetime.timedelta(seconds=1),
            ),
        ),
        (ReadOnlyOptionSpec(save_interval_steps=0, keep_period=1),),
        (ReadOnlyOptionSpec(save_interval_steps=0, create=True),),
        (
            ReadOnlyOptionSpec(
                save_interval_steps=0,
                create=False,
                cleanup_tmp_directories=True,
            ),
        ),
        (
            ReadOnlyOptionSpec(
                save_interval_steps=0,
                create=False,
                cleanup_tmp_directories=False,
                save_on_steps=[1],
            ),
        ),
        (
            ReadOnlyOptionSpec(
                save_interval_steps=0,
                create=False,
                cleanup_tmp_directories=False,
                save_on_steps=None,
                todelete_subdir="ttl=1h",
            ),
        ),
        (
            ReadOnlyOptionSpec(
                save_interval_steps=0,
                create=False,
                cleanup_tmp_directories=False,
                save_on_steps=None,
                todelete_full_path="/tmp/trash",
            ),
        ),
        (
            ReadOnlyOptionSpec(
                save_interval_steps=1,
                save_on_steps=[1],
                should_save_fn=_save_every_even_step,
            ),
        ),
    )
    def test_side_effect_options_update_for_read_only(
        self,
        spec: ReadOnlyOptionSpec,
    ) -> None:
        """Test that read_only flag correctly nullifies other incompatible options.

        Args:
            spec: The typed configuration options dictionary to initialize with.
        """
        self.assertTrue(_save_every_even_step(2, None))
        self.assertFalse(_save_every_even_step(3, 2))
        options = ocp.CheckpointManagerOptions(
            read_only=True,
            save_interval_steps=spec.get("save_interval_steps", 1),
            max_to_keep=spec.get("max_to_keep"),
            keep_time_interval=spec.get("keep_time_interval"),
            keep_period=spec.get("keep_period"),
            create=spec.get("create", True),
            cleanup_tmp_directories=spec.get("cleanup_tmp_directories", False),
            save_on_steps=spec.get("save_on_steps"),
            todelete_subdir=spec.get("todelete_subdir"),
            todelete_full_path=spec.get("todelete_full_path"),
            should_save_fn=spec.get("should_save_fn"),
        )
        self.assertEqual(options.save_interval_steps, 0)
        self.assertIsNone(options.max_to_keep)
        self.assertIsNone(options.keep_time_interval)
        self.assertIsNone(options.keep_period)
        self.assertFalse(options.create)
        self.assertFalse(options.cleanup_tmp_directories)
        self.assertEmpty(options.save_on_steps)
        self.assertIsNone(options.todelete_subdir)
        self.assertIsNone(options.todelete_full_path)
        self.assertIsNone(options.should_save_fn)
        self.assertIsNone(options.should_keep_fn)

    def test_replace_for_read_only(self) -> None:
        """Test that dataclasses.replace and options.replace preserve states for read-only options."""
        options = ocp.CheckpointManagerOptions(read_only=True, create=False, save_interval_steps=0)
        self.assertEmpty(options.save_on_steps)
        updated_options = dataclasses.replace(options, step_prefix="prefix")
        self.assertEmpty(updated_options.save_on_steps)
        replaced = options.replace(step_prefix="step", step_format_fixed_length=4)
        self.assertEqual(replaced.step_prefix, "step")
        self.assertEqual(replaced.step_format_fixed_length, 4)

    def test_replace_for_should_keep_fn(self) -> None:
        """Test that keep_period is nulled when should_keep_fn is set on initialization."""
        options = ocp.CheckpointManagerOptions(keep_period=1, should_keep_fn=lambda step, ts: True)
        self.assertIsNone(options.keep_period)
        self.assertIsNotNone(options.should_keep_fn)

    @parameterized.named_parameters(
        dict(
            testcase_name="error_step_name_format_false",
            single_host_load_and_broadcast=True,
            step_name_format=ocp.step.standard_name_format(single_host_load_and_broadcast=False),
            expected_single_host_load_and_broadcast=None,
            expected_step_name_format=None,
        ),
        dict(
            testcase_name="error_step_name_format_true",
            single_host_load_and_broadcast=False,
            step_name_format=ocp.step.standard_name_format(single_host_load_and_broadcast=True),
            expected_single_host_load_and_broadcast=None,
            expected_step_name_format=None,
        ),
        dict(
            testcase_name="error_enabled_with_non_supporting_step_name_format",
            single_host_load_and_broadcast=True,
            step_name_format=CUSTOM_STEP_NAME_FORMAT,
            expected_single_host_load_and_broadcast=None,
            expected_step_name_format=None,
        ),
        dict(
            testcase_name="both_true",
            single_host_load_and_broadcast=True,
            step_name_format=ocp.step.standard_name_format(single_host_load_and_broadcast=True),
            expected_single_host_load_and_broadcast=True,
            expected_step_name_format=ocp.step.standard_name_format(
                single_host_load_and_broadcast=True
            ),
        ),
        dict(
            testcase_name="both_false",
            single_host_load_and_broadcast=False,
            step_name_format=ocp.step.standard_name_format(single_host_load_and_broadcast=False),
            expected_single_host_load_and_broadcast=False,
            expected_step_name_format=ocp.step.standard_name_format(
                single_host_load_and_broadcast=False
            ),
        ),
        dict(
            testcase_name="only_single_host_load_and_broadcast_true",
            single_host_load_and_broadcast=True,
            step_name_format=None,
            expected_single_host_load_and_broadcast=True,
            expected_step_name_format=None,
        ),
        dict(
            testcase_name="only_single_host_load_and_broadcast_false",
            single_host_load_and_broadcast=False,
            step_name_format=None,
            expected_single_host_load_and_broadcast=False,
            expected_step_name_format=None,
        ),
        dict(
            testcase_name="disabled_with_non_supporting_step_name_format",
            single_host_load_and_broadcast=False,
            step_name_format=CUSTOM_STEP_NAME_FORMAT,
            expected_single_host_load_and_broadcast=False,
            expected_step_name_format=CUSTOM_STEP_NAME_FORMAT,
        ),
    )
    def test_single_host_load_and_broadcast(
        self,
        single_host_load_and_broadcast: bool,
        step_name_format: ocp.step.NameFormat | None,
        expected_single_host_load_and_broadcast: bool | None,
        expected_step_name_format: ocp.step.NameFormat | None,
    ) -> None:
        """Test single_host_load_and_broadcast interplay with step_name_format.

        Args:
            single_host_load_and_broadcast: Input boolean for manager options.
            step_name_format: Step NameFormat instance to use.
            expected_single_host_load_and_broadcast: Expected boolean value, or None if error.
            expected_step_name_format: Expected NameFormat, or None if error.
        """
        if expected_single_host_load_and_broadcast is None and expected_step_name_format is None:
            with self.assertRaises(ValueError):
                ocp.CheckpointManagerOptions(
                    single_host_load_and_broadcast=single_host_load_and_broadcast,
                    step_name_format=step_name_format,
                )
        else:
            options = ocp.CheckpointManagerOptions(
                single_host_load_and_broadcast=single_host_load_and_broadcast,
                step_name_format=step_name_format,
            )
            self.assertEqual(
                options.single_host_load_and_broadcast,
                expected_single_host_load_and_broadcast,
            )
            self.assertEqual(options.step_name_format, expected_step_name_format)
