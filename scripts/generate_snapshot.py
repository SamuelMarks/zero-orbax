"""Generate the structural API snapshot JSON for orbax.checkpoint / zero_orbax.checkpoint."""

from __future__ import annotations

import json
import sys
import types

from scripts.check_compliance import dump_api


def generate_snapshot_to_file(
    module: types.ModuleType,
    output_path: str = "tests/orbax_api_snapshot.json",
) -> dict[str, dict[str, str | list[str]]]:
    """Introspect ``module`` and write its API snapshot dictionary to ``output_path``.

    Args:
        module: Module to introspect (e.g. ``zero_orbax.checkpoint`` or ``orbax.checkpoint``).
        output_path: Destination JSON file path. Defaults to ``"tests/orbax_api_snapshot.json"``.

    Returns:
        dict[str, dict[str, str | list[str]]]: The generated snapshot dictionary.
    """
    snapshot = dump_api(module)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
    print("Snapshot generated.")
    return snapshot


def main(output_path: str = "tests/orbax_api_snapshot.json") -> int:
    """Generate the API snapshot from ``zero_orbax.checkpoint`` and write to disk.

    Args:
        output_path: Destination JSON file path. Defaults to ``"tests/orbax_api_snapshot.json"``.

    Returns:
        int: Process exit code (always 0).
    """
    import zero_orbax.checkpoint as ocp

    generate_snapshot_to_file(ocp, output_path=output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
