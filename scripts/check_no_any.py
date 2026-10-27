"""Delegate wrapper for scripts/lint_typing.py."""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run the strict static typing and anti-reflection AST check.

    Args:
        argv: Optional CLI argument sequence. Defaults to None.

    Returns:
        int: Process exit code (0 on zero violations, 1 on failure).
    """
    sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1])]
    from scripts.lint_typing import main as _lint_typing_main

    return _lint_typing_main(argv)


if __name__ == "__main__":
    sys.exit(main())
