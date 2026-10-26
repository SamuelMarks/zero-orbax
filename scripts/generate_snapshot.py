"""Generate the structural API snapshot JSON for orbax.checkpoint / zero_orbax.checkpoint."""

from __future__ import annotations

import json
import pathlib
import sys
import types

sys.path[:0] = [
    str(pathlib.Path(__file__).resolve().parent.parent),
    str(pathlib.Path(__file__).resolve().parent.parent / "src"),
]

from scripts.check_compliance import dump_api  # noqa: E402


def generate_snapshot_to_file(
    module: types.ModuleType,
    output_path: str | None = None,
) -> dict[str, dict[str, str | list[str]]]:
    """Introspect ``module`` and optionally write its API snapshot dictionary to ``output_path``.

    Args:
        module: Module to introspect (e.g. ``zero_orbax.checkpoint`` or ``orbax.checkpoint``).
        output_path: Optional destination JSON file path. Defaults to None.

    Returns:
        dict[str, dict[str, str | list[str]]]: The generated snapshot dictionary.
    """
    snapshot = dump_api(module)
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, sort_keys=True)
    print("Snapshot generated.")
    return snapshot


def main(output_path: str | None = None) -> int:
    """Generate the API snapshot from ``zero_orbax.checkpoint`` and optionally write to disk.

    Args:
        output_path: Optional destination JSON file path. Defaults to None.

    Returns:
        int: Process exit code (always 0).
    """
    import zero_orbax.checkpoint as ocp

    generate_snapshot_to_file(ocp, output_path=output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
