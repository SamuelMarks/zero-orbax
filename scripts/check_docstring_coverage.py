"""Delegate wrapper for scripts/check_docs.py."""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence


def main(target_dirs: Sequence[str] = ("src", "scripts", "tests")) -> int:
    """Run the AST docstring coverage verifier across zero_orbax, scripts, and tests.

    Args:
        target_dirs: Directories to inspect. Defaults to ``("src", "scripts", "tests")``.

    Returns:
        int: Process exit code (0 on 100% docstring coverage, 1 on failure).
    """
    sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1])]
    from scripts.check_docs import main as _check_docs_main

    return _check_docs_main(target_dirs=target_dirs)


if __name__ == "__main__":
    sys.exit(main())
