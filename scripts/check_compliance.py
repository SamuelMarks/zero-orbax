"""API compliance verification script against local and ml-ecosystem-snapshots v0.0.4 schemas."""

from __future__ import annotations

import inspect
import json
import os
import sys
import types
from collections.abc import Mapping


def dump_api(module: types.ModuleType) -> dict[str, dict[str, str | list[str]]]:
    """Introspect a Python module and build a structural dictionary of its public symbols.

    Args:
        module: The imported module object to inspect.

    Returns:
        dict[str, dict[str, str | list[str]]]: Mapping of symbol name to ``{"type": ..., "methods": ...}``.
    """
    api: dict[str, dict[str, str | list[str]]] = {}
    for name, obj in inspect.getmembers(module):
        if name.startswith("_") and name not in ("__version__",):
            continue
        if inspect.isclass(obj):
            methods: list[str] = [
                m_name
                for m_name, m_obj in inspect.getmembers(obj)
                if not m_name.startswith("_") and callable(m_obj)
            ]
            api[name] = {"type": "class", "methods": sorted(methods)}
        elif isinstance(obj, types.ModuleType):
            api[name] = {"type": "namespace"}
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
            api[name] = {"type": "function"}
        else:
            api[name] = {"type": "other"}
    return api


def verify_snapshot_compliance(
    current_api: Mapping[str, Mapping[str, str | list[str]]],
    snapshot: Mapping[str, Mapping[str, str | list[str]]],
) -> list[str]:
    """Compare the introspected API against the expected snapshot dictionary.

    Args:
        current_api: Dictionary produced by ``dump_api``.
        snapshot: Expected snapshot dictionary loaded from ``orbax_api_snapshot.json``.

    Returns:
        list[str]: List of mismatch error strings (empty if 100% compliant).
    """
    errors: list[str] = []
    ignored_names = {
        "__version__",
        "logging",
        "contextlib",
        "functools",
        "asyncio",
        "nest_asyncio",
    }

    for name, expected in snapshot.items():
        if name in ignored_names:
            continue
        if name not in current_api:
            errors.append(f"Missing symbol in zero_orbax.checkpoint: {name}")
            continue

        actual = current_api[name]
        expected_type = expected.get("type")
        actual_type = actual.get("type")

        if (expected_type == "namespace" and actual_type == "class") or (
            expected_type == "class" and actual_type == "namespace"
        ):
            continue
        if expected_type != actual_type:
            errors.append(f"Type mismatch for {name}: expected {expected_type}, got {actual_type}")
            continue

        if expected_type == "class" and actual_type == "class":
            raw_exp_methods = expected.get("methods", [])
            raw_act_methods = actual.get("methods", [])
            expected_methods = set(raw_exp_methods) if isinstance(raw_exp_methods, list) else set()
            actual_methods = set(raw_act_methods) if isinstance(raw_act_methods, list) else set()
            missing_methods = sorted(expected_methods - actual_methods)
            if missing_methods:
                errors.append(f"Class {name} is missing methods: {', '.join(missing_methods)}")

    return errors


def check_compliance(snapshot_path: str | None = None) -> tuple[bool, list[str]]:
    """Run the API compliance check against ``zero_orbax.checkpoint``.

    Args:
        snapshot_path: Optional override path to the JSON snapshot file. Defaults to None.

    Returns:
        tuple[bool, list[str]]: Tuple of ``(is_compliant, error_messages)``.
    """
    import zero_orbax.checkpoint as zcp

    resolved_path = snapshot_path or os.path.join(
        os.path.dirname(__file__), "..", "tests", "orbax_api_snapshot.json"
    )
    if not os.path.exists(resolved_path):
        return False, [f"Error: Snapshot not found at {resolved_path}"]

    with open(resolved_path, encoding="utf-8") as f:
        snapshot: dict[str, dict[str, str | list[str]]] = json.load(f)

    current_api = dump_api(zcp)
    errors = verify_snapshot_compliance(current_api, snapshot)
    return len(errors) == 0, errors


def main(snapshot_path: str | None = None) -> int:
    """Execute the compliance verification CLI.

    Args:
        snapshot_path: Optional override path to the JSON snapshot file. Defaults to None.

    Returns:
        int: Exit code (0 on 100% compliance, 1 on failure).
    """
    ok, errors = check_compliance(snapshot_path=snapshot_path)
    if not ok:
        print("Compliance check failed!")
        for error in errors:
            print(f" - {error}")
        return 1

    print("Compliance Check Passed! 100% compliant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
