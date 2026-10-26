"""Step metadata and directory name formatting for zero_orbax.checkpoint.step."""

from __future__ import annotations

import dataclasses


class step_lib:
    """Namespace for step-related metadata and directory name formatting."""

    @dataclasses.dataclass(frozen=True)
    class Metadata:
        """Metadata associated with an individual checkpoint step.

        Attributes:
            step: Numeric training step.
            path: Directory path for the step checkpoint.
        """

        step: int = 0
        path: str = ""

    @dataclasses.dataclass
    class NameFormat:
        """Formatting rules for checkpoint step directory names.

        Attributes:
            step_prefix: Optional string prefix prepended to step numbers. Defaults to None.
            step_format_fixed_length: Optional zero-padding width for step numbers. Defaults to None.
        """

        step_prefix: str | None = None
        step_format_fixed_length: int | None = None

        def build_name(self, step_num: int) -> str:
            """Format a step number into its directory name string.

            Args:
                step_num: Step integer to format.

            Returns:
                str: Formatted directory name.
            """
            num_str = (
                f"{step_num:0{self.step_format_fixed_length}d}"
                if self.step_format_fixed_length is not None
                else str(step_num)
            )
            if self.step_prefix:
                return f"{self.step_prefix}_{num_str}"
            return num_str

        def parse_step(self, dir_name: str) -> int | None:
            """Parse a step integer from a formatted step directory name.

            Args:
                dir_name: Candidate directory name.

            Returns:
                int | None: Parsed step integer, or None if ``dir_name`` does not match.
            """
            if self.step_prefix:
                prefix_with_sep = f"{self.step_prefix}_"
                if not dir_name.startswith(prefix_with_sep):
                    return None
                num_part = dir_name[len(prefix_with_sep) :]
            else:
                num_part = dir_name
            if num_part.isdigit():
                return int(num_part)
            return None

    @dataclasses.dataclass
    class StandardNameFormat(NameFormat):
        """Standard step directory name format supporting single-host load and broadcast.

        Attributes:
            single_host_load_and_broadcast: Whether to format for single-host load and broadcast.
            step_prefix: Optional string prefix prepended to step numbers.
            step_format_fixed_length: Optional zero-padding width for step numbers.
        """

        single_host_load_and_broadcast: bool = False
        step_prefix: str | None = None
        step_format_fixed_length: int | None = None

    @staticmethod
    def standard_name_format(
        single_host_load_and_broadcast: bool = False,
        step_prefix: str | None = None,
        step_format_fixed_length: int | None = None,
    ) -> step_lib.StandardNameFormat:
        """Create a standard step ``StandardNameFormat`` instance.

        Args:
            single_host_load_and_broadcast: Whether to format for single-host load and broadcast. Defaults to False.
            step_prefix: Optional step directory name prefix. Defaults to None.
            step_format_fixed_length: Optional fixed digit width for step numbers. Defaults to None.

        Returns:
            step_lib.StandardNameFormat: Configured ``StandardNameFormat`` instance.
        """
        return step_lib.StandardNameFormat(
            single_host_load_and_broadcast=single_host_load_and_broadcast,
            step_prefix=step_prefix,
            step_format_fixed_length=step_format_fixed_length,
        )


step = step_lib

__all__ = ["step", "step_lib"]
