"""AST parser to ensure no disallowed 3rd party imports exist in zero-orbax source files."""

from __future__ import annotations

import ast
import os
import pkgutil
import sys
from collections.abc import Sequence

ALLOWED_3RD_PARTY: frozenset[str] = frozenset({"pydantic", "typing_extensions"})
ALLOWED_INTERNAL: frozenset[str] = frozenset(
    {
        "cdd",
        "ml_switcheroo_ir",
        "ml_switcheroo_compiler",
        "ml_ecosystem_snapshots",
        "zero_chex",
        "zero_flax",
        "zero_grain",
        "zero_optax",
        "zero_jax",
        "zero_orbax",
    }
)


def _default_stdlib_names() -> frozenset[str]:
    """Build the set of standard library module names across Python versions.

    Returns:
        frozenset[str]: Standard library module names.
    """
    names: set[str] = set(sys.builtin_module_names)
    for mod in pkgutil.iter_modules():
        if "site-packages" not in str(mod.module_finder):
            names.add(mod.name)
    names.update(
        {
            "abc",
            "argparse",
            "ast",
            "asyncio",
            "collections",
            "contextlib",
            "copy",
            "dataclasses",
            "datetime",
            "enum",
            "functools",
            "inspect",
            "io",
            "itertools",
            "json",
            "logging",
            "math",
            "os",
            "pathlib",
            "pkgutil",
            "re",
            "shutil",
            "struct",
            "subprocess",
            "sys",
            "tempfile",
            "threading",
            "time",
            "types",
            "typing",
            "unittest",
            "uuid",
            "warnings",
        }
    )
    return frozenset(names)


DEFAULT_STDLIB_NAMES: frozenset[str] = _default_stdlib_names()


def _is_allowed_module(base_module: str, stdlib_names: frozenset[str]) -> bool:
    """Check whether an imported root module name is permitted.

    Args:
        base_module: Top-level package name extracted from an import node.
        stdlib_names: Set of standard library module names.

    Returns:
        bool: True if the module is allowed, False otherwise.
    """
    return (
        base_module in ALLOWED_3RD_PARTY
        or base_module in ALLOWED_INTERNAL
        or base_module in stdlib_names
    )


def check_file(filepath: str, stdlib_names: frozenset[str] | None = None) -> bool:
    """Check a Python source file for disallowed third-party imports.

    Args:
        filepath: Path to the Python file to inspect.
        stdlib_names: Optional override set of standard library module names. Defaults to None.

    Returns:
        bool: True if the file contains only allowed imports, False if violations or syntax errors occur.
    """
    active_stdlib = stdlib_names if stdlib_names is not None else DEFAULT_STDLIB_NAMES
    with open(filepath, encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        print(f"Syntax error in {filepath}")
        return False

    errors: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if not _is_allowed_module(base_module, active_stdlib):
                    errors.append((node.lineno, base_module))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                base_module = node.module.split(".")[0]
                if not _is_allowed_module(base_module, active_stdlib):
                    errors.append((node.lineno, base_module))

    if errors:
        for lineno, module in errors:
            print(f"{filepath}:{lineno}: Disallowed 3rd party import detected: '{module}'")
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """Run the allowed-imports check across CLI file or directory arguments.

    Args:
        argv: Optional sequence of file or directory paths to inspect. Defaults to ``sys.argv[1:]``.

    Returns:
        int: Exit code (0 if all files pass, 1 if any file fails).
    """
    args_list = list(argv) if argv is not None else sys.argv[1:]
    if not args_list:
        args_list = ["src/zero_orbax"]
    success = True
    for arg in args_list:
        if os.path.isdir(arg):
            for root, _, files in sorted(os.walk(arg)):
                for file in sorted(files):
                    if file.endswith(".py") and not check_file(os.path.join(root, file)):
                        success = False
        elif arg.endswith(".py") and not check_file(arg):
            success = False
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
