"""AST-based static typing and anti-reflection linter for zero-orbax.

Verifies that across ``src/``, ``scripts/``, and ``tests/``:
- ``Any`` and ``object`` are never imported or used in type annotations.
- ``typing.cast``, ``getattr``, ``setattr``, ``hasattr``, ``delattr``, ``globals``, ``locals``,
  ``vars``, ``exec``, ``eval``, ``compile``, ``__import__``, ``__dict__``, ``__class__``,
  ``__bases__``, ``__mro__``, ``__subclasses__``, ``__globals__``, ``__code__``, ``__closure__``,
  ``__func__``, and ``__self__`` are not used.
- Inline type-ignore suppression comments are not used.
- Every function and method has explicit type annotations on all parameters and return values.
"""

from __future__ import annotations

import argparse
import ast
import io
import os
import sys
import tokenize
from collections.abc import Sequence

DEFAULT_TARGET_DIRS: tuple[str, ...] = ("src", "scripts", "tests")
FORBIDDEN_TYPE_NAMES: frozenset[str] = frozenset({"Any", "object", "AnyType"})
FORBIDDEN_REFLECTION_CALLS: frozenset[str] = frozenset(
    {
        "cast",
        "getattr",
        "setattr",
        "hasattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "exec",
        "eval",
        "compile",
        "__import__",
    }
)
FORBIDDEN_ATTR_CALLS: frozenset[str] = frozenset(
    {
        "cast",
        "getattr",
        "setattr",
        "hasattr",
        "delattr",
    }
)
FORBIDDEN_DUNDER_ATTRS: frozenset[str] = frozenset(
    {
        "__dict__",
        "__class__",
        "__bases__",
        "__mro__",
        "__subclasses__",
        "__globals__",
        "__code__",
        "__closure__",
        "__func__",
        "__self__",
        "__module__",
        "__qualname__",
        "__getattribute__",
        "__setattr__",
    }
)
FORBIDDEN_IMPORT_NAMES: frozenset[str] = frozenset(
    {
        "Any",
        "cast",
        "getattr",
        "setattr",
        "hasattr",
        "delattr",
        "vars",
        "globals",
        "locals",
        "exec",
        "eval",
    }
)


def _annotation_violations(node: ast.AST) -> list[str]:
    """Inspect an annotation AST subtree for prohibited ``Any`` or ``object`` references.

    Args:
        node: Annotation AST node to inspect.

    Returns:
        list[str]: List of prohibited type names detected in the annotation.
    """
    hits: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in FORBIDDEN_TYPE_NAMES:
            hits.append(sub.id)
        elif isinstance(sub, ast.Attribute) and sub.attr in FORBIDDEN_TYPE_NAMES:
            hits.append(f"typing.{sub.attr}")
    return hits


def _check_function_node(
    file_path: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[str]:
    """Check a function or method AST node for missing or prohibited type annotations.

    Args:
        file_path: Source file path being scanned.
        node: Function or async function AST node.

    Returns:
        list[str]: List of human-readable violation messages.
    """
    violations: list[str] = []
    all_args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    for arg in all_args:
        if arg.arg in ("self", "cls"):
            continue
        if arg.annotation is None:
            violations.append(
                f"{file_path}:{arg.lineno}: missing type annotation for parameter '{arg.arg}' in '{node.name}'"
            )
        else:
            for hit in _annotation_violations(arg.annotation):
                violations.append(
                    f"{file_path}:{arg.lineno}: prohibited parameter annotation '{hit}' on '{arg.arg}'"
                )

    for default_val in [d for d in [*node.args.defaults, *node.args.kw_defaults] if d is not None]:
        for hit in _annotation_violations(default_val):
            violations.append(
                f"{file_path}:{node.lineno}: prohibited default value reference '{hit}' in '{node.name}'"
            )

    for var_arg in (node.args.vararg, node.args.kwarg):
        if var_arg is not None:
            if var_arg.annotation is None:
                violations.append(
                    f"{file_path}:{var_arg.lineno}: missing type annotation for '*{var_arg.arg}' in '{node.name}'"
                )
            else:
                for hit in _annotation_violations(var_arg.annotation):
                    violations.append(
                        f"{file_path}:{var_arg.lineno}: prohibited variadic annotation '{hit}' on '{var_arg.arg}'"
                    )

    if node.returns is None:
        violations.append(
            f"{file_path}:{node.lineno}: missing return type annotation on '{node.name}'"
        )
    else:
        for hit in _annotation_violations(node.returns):
            violations.append(
                f"{file_path}:{node.lineno}: prohibited return annotation '{hit}' on '{node.name}'"
            )
    return violations


def scan_file(file_path: str) -> list[str]:
    """Scan a single Python source file for typing and reflection violations.

    Args:
        file_path: Path to the Python file to scan.

    Returns:
        list[str]: List of detected violations.
    """
    with open(file_path, encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as exc:
        return [f"{file_path}:{exc.lineno}: syntax error: {exc.msg}"]

    violations: list[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            normalized_comment = tok.string.replace(" ", "")
            if normalized_comment.startswith("#" + "type:ignore"):
                violations.append(f"{file_path}:{tok.start[0]}: prohibited type-ignore comment")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "builtins":
                    violations.append(f"{file_path}:{node.lineno}: prohibited import 'builtins'")
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORT_NAMES or (
                    alias.asname is not None and alias.asname in FORBIDDEN_IMPORT_NAMES
                ):
                    violations.append(
                        f"{file_path}:{node.lineno}: prohibited import '{alias.name}'"
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.extend(_check_function_node(file_path, node))
        elif isinstance(node, ast.AnnAssign):
            for hit in _annotation_violations(node.annotation):
                violations.append(
                    f"{file_path}:{node.lineno}: prohibited variable annotation '{hit}'"
                )
        elif isinstance(node, ast.Assign):
            for hit in _annotation_violations(node.value):
                violations.append(f"{file_path}:{node.lineno}: prohibited type alias value '{hit}'")
        elif isinstance(node, ast.ClassDef):
            for base_node in node.bases:
                for hit in _annotation_violations(base_node):
                    violations.append(
                        f"{file_path}:{node.lineno}: prohibited class base reference '{hit}'"
                    )
            for kw_node in node.keywords:
                for hit in _annotation_violations(kw_node.value):
                    violations.append(
                        f"{file_path}:{node.lineno}: prohibited class keyword reference '{hit}'"
                    )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_REFLECTION_CALLS:
                violations.append(
                    f"{file_path}:{node.lineno}: prohibited call '{node.func.id}(...)'"
                )
            elif isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_ATTR_CALLS:
                violations.append(
                    f"{file_path}:{node.lineno}: prohibited attribute call '.{node.func.attr}(...)'"
                )
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_DUNDER_ATTRS:
            violations.append(
                f"{file_path}:{node.lineno}: prohibited dunder reflection '.{node.attr}'"
            )

    return violations


def scan_directories(target_dirs: Sequence[str] = DEFAULT_TARGET_DIRS) -> list[str]:
    """Scan all Python files in ``target_dirs`` for typing and reflection violations.

    Args:
        target_dirs: Directories to scan recursively. Defaults to ``("src", "scripts", "tests")``.

    Returns:
        list[str]: Aggregated list of violations across all scanned files.
    """
    all_violations: list[str] = []
    for target_dir in target_dirs:
        if not os.path.exists(target_dir):
            continue
        for root, _, files in sorted(os.walk(target_dir)):
            for fname in sorted(files):
                if fname.endswith(".py"):
                    fpath = os.path.join(root, fname)
                    all_violations.extend(scan_file(fpath))
    return all_violations


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the typing and anti-reflection linter CLI.

    Args:
        argv: Optional command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        int: Exit code (0 if zero violations, 1 otherwise).
    """
    parser = argparse.ArgumentParser(
        description="Lint Python ASTs for strict typing and zero reflection."
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        default=list(DEFAULT_TARGET_DIRS),
        help="Directories to scan.",
    )
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    violations = scan_directories(parsed.dirs)
    if violations:
        for v in violations:
            print(v)
        print(f"Found {len(violations)} typing/reflection violation(s).")
        return 1
    print("Strict typing & reflection check passed! 0 violations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
