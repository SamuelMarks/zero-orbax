"""Delegate wrapper for scripts/check_compliance.py."""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run the API parity and compliance check across zero_orbax.

    Args:
        argv: Optional CLI argument sequence. Defaults to None.

    Returns:
        int: Process exit code (0 on 100% parity, 1 on failure).
    """
    _ = argv
    sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1])]
    from scripts.check_compliance import main as _check_compliance_main

    return _check_compliance_main()


if __name__ == "__main__":
    sys.exit(main())
