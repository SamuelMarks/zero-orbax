import sys
import json
import os
import inspect
import types


def dump_api(module):
    api = {}
    for name in dir(module):
        if name.startswith("_") and name not in ("__version__",):
            continue
        obj = getattr(module, name)

        if inspect.isclass(obj):
            # Could be a class or a namespace masquerading as a class
            methods = []
            for m_name in dir(obj):
                if not m_name.startswith("_") and callable(getattr(obj, m_name)):
                    methods.append(m_name)
            api[name] = {"type": "class", "methods": methods}
        elif isinstance(obj, types.ModuleType):
            api[name] = {"type": "namespace"}
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
            api[name] = {"type": "function"}
        else:
            api[name] = {"type": "other"}
    return api


def main():
    try:
        import zero_orbax.checkpoint as zcp
    except ImportError:
        print(
            "Error: Could not import zero_orbax.checkpoint. Make sure it's in PYTHONPATH."
        )
        sys.exit(1)

    snapshot_path = os.path.join(
        os.path.dirname(__file__), "..", "tests", "orbax_api_snapshot.json"
    )
    if not os.path.exists(snapshot_path):
        print(f"Error: Snapshot not found at {snapshot_path}")
        sys.exit(1)

    with open(snapshot_path, "r") as f:
        snapshot = json.load(f)

    current_api = dump_api(zcp)
    errors = []

    for name, expected in snapshot.items():
        if name in (
            "__version__",
            "logging",
            "contextlib",
            "functools",
            "asyncio",
            "nest_asyncio",
        ):
            continue

        if name not in current_api:
            errors.append(f"Missing symbol in zero_orbax.checkpoint: {name}")
            continue

        actual = current_api[name]

        expected_type = expected["type"]
        actual_type = actual["type"]

        # Relax distinction between class and namespace since zero_orbax uses classes for namespaces
        if expected_type == "namespace" and actual_type == "class":
            pass  # Acceptable
        elif expected_type == "class" and actual_type == "namespace":
            pass  # Acceptable
        elif expected_type != actual_type:
            errors.append(
                f"Type mismatch for {name}: expected {expected_type}, got {actual_type}"
            )
            continue

        if expected_type == "class" and actual_type == "class":
            expected_methods = set(expected["methods"])
            actual_methods = set(actual["methods"])
            # In zero_orbax we want to ensure all expected methods are present
            missing_methods = expected_methods - actual_methods
            if missing_methods:
                errors.append(
                    f"Class {name} is missing methods: {', '.join(missing_methods)}"
                )

    if errors:
        print("Compliance check failed!")
        for error in errors:
            print(f" - {error}")
        sys.exit(1)

    print("Compliance Check Passed! 100% compliant.")
    sys.exit(0)


if __name__ == "__main__":
    main()
