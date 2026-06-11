"""AST parser to ensure no disallowed 3rd party imports."""

import ast
import sys
import os

ALLOWED_3RD_PARTY = {"numpy", "pydantic"}
ALLOWED_INTERNAL = {
    "ml_switcheroo",
    "cdd",
    "ml_switcheroo_ir",
    "ml_switcheroo_compiler",
    "zero_jax",
    "zero_orbax",
}


def check_file(filepath: str) -> bool:
    """Check a file for disallowed imports.

    Args:
        filepath (str): The file to check.

    Returns:
        bool: True if clean, False otherwise.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        print(f"Syntax error in {filepath}")
        return False

    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if (
                    base_module not in ALLOWED_3RD_PARTY
                    and base_module not in ALLOWED_INTERNAL
                    and base_module not in sys.stdlib_module_names
                ):
                    errors.append((node.lineno, base_module))
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:  # Relative import
                continue
            if node.module:
                base_module = node.module.split(".")[0]
                if (
                    base_module not in ALLOWED_3RD_PARTY
                    and base_module not in ALLOWED_INTERNAL
                    and base_module not in sys.stdlib_module_names
                ):
                    errors.append((node.lineno, base_module))

    if errors:
        for lineno, module in errors:
            print(
                f"{filepath}:{lineno}: Disallowed 3rd party import detected: '{module}'"
            )
        return False
    return True


if __name__ == "__main__":
    if not hasattr(sys, "stdlib_module_names"):
        # Polyfill for python < 3.10
        import sysconfig
        import pathlib

        try:
            from stdlib_list import stdlib_list

            names = set(stdlib_list(".".join(map(str, sys.version_info[:2]))))
        except ImportError:
            # Fallback
            import distutils.sysconfig as ds

            stdlib_paths = [ds.get_python_lib(standard_lib=True)]
            names = set(sys.builtin_module_names)
            for p in stdlib_paths:
                path = pathlib.Path(p)
                if path.exists():
                    for item in path.iterdir():
                        if item.is_file() and item.suffix == ".py":
                            names.add(item.stem)
                        elif item.is_dir():
                            names.add(item.name)
        sys.stdlib_module_names = frozenset(names)

    success = True
    for arg in sys.argv[1:]:
        if arg.endswith(".py"):
            if not check_file(arg):
                success = False

    if not success:
        sys.exit(1)
