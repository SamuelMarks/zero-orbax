"""Granular AST docstring coverage verifier for modules, classes, functions, and arguments."""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from collections.abc import Sequence

BANNED_STUB_DOCSTRINGS: frozenset[str] = frozenset(
    {
        "Initialize.",
        "Test function.",
        "Provide module docstring.",
        "Provide a mock implementation for parity.",
        "Represent the class.",
        "Execute the function.",
    }
)


def _extract_documented_args(docstring: str) -> set[str]:
    """Extract parameter names documented inside the ``Args:`` section of a docstring.

    Args:
        docstring: Raw docstring text from a function or method.

    Returns:
        set[str]: Set of parameter names (stripped of leading ``*``) documented in ``Args:``.
    """
    documented: set[str] = set()
    lines = docstring.splitlines()
    in_args = False
    args_indent = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if stripped == "Args:":
            in_args = True
            args_indent = indent
            continue
        if in_args:
            if indent <= args_indent and stripped.endswith(":"):
                break
            match = re.match(r"^\*{0,2}([A-Za-z_][A-Za-z0-9_]*)\s*(\([^)]*\))?\s*:", stripped)
            if match:
                documented.add(match.group(1))
    return documented


def _required_param_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Collect parameter names requiring documentation on a function or method AST node.

    Args:
        node: Function or async function AST node.

    Returns:
        list[str]: Parameter names excluding ``self`` and ``cls``.
    """
    params = [
        arg.arg
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        if arg.arg not in ("self", "cls")
    ]
    if node.args.vararg is not None:
        params.append(node.args.vararg.arg)
    if node.args.kwarg is not None:
        params.append(node.args.kwarg.arg)
    return params


def _is_non_none_return(returns_node: ast.AST | None) -> bool:
    """Determine whether a return annotation represents a non-``None`` return value.

    Args:
        returns_node: Return annotation AST node on a function definition.

    Returns:
        bool: True if the return annotation is present and not ``None``.
    """
    if returns_node is None:
        return False
    return not (
        (isinstance(returns_node, ast.Constant) and returns_node.value is None)
        or (isinstance(returns_node, ast.Name) and returns_node.id == "None")
    )


def _has_direct_raise(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Determine whether a function body contains a ``raise`` statement outside nested definitions.

    Args:
        node: Function or async function AST node to inspect.

    Returns:
        bool: True if any statement in the function body (including nested control flow) raises.
    """
    stack: list[ast.AST] = list(node.body)
    while stack:
        current = stack.pop()
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(current, ast.Raise):
            return True
        stack.extend(ast.iter_child_nodes(current))
    return False


def _check_function_args_doc(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    doc: str,
) -> list[str]:
    """Verify that all required parameters, returns, and raises on a function AST node are documented.

    Also rejects extra documented parameters not in the function signature, and
    verifies ``Returns:`` / ``Yields:`` and ``Raises:`` sections.

    Args:
        node: Function or async function AST node.
        doc: Non-empty docstring of the function.

    Returns:
        list[str]: Missing or extra argument/section error messages.
    """
    errors: list[str] = []
    req_params = _required_param_names(node)
    doc_args = _extract_documented_args(doc)
    if req_params:
        undocumented = [p for p in req_params if p not in doc_args]
        if undocumented:
            errors.append(
                f"{type(node).__name__} '{node.name}' at line {node.lineno} "
                f"(MISSING ARGS DOC: {', '.join(undocumented)})"
            )
    extra_documented = sorted(doc_args - set(req_params))
    if extra_documented:
        errors.append(
            f"{type(node).__name__} '{node.name}' at line {node.lineno} "
            f"(EXTRA ARGS DOC: {', '.join(extra_documented)})"
        )
    if _is_non_none_return(node.returns) and "Returns:" not in doc and "Yields:" not in doc:
        errors.append(
            f"{type(node).__name__} '{node.name}' at line {node.lineno} (MISSING RETURNS DOC)"
        )
    if _has_direct_raise(node) and "Raises:" not in doc:
        errors.append(
            f"{type(node).__name__} '{node.name}' at line {node.lineno} (MISSING RAISES DOC)"
        )
    return errors


def check_file(path: str, *, strict_sections: bool = True) -> tuple[int, int]:
    """Check a single Python file for module, class, function, and argument docstring coverage.

    Args:
        path: Path to the Python file to inspect.
        strict_sections: Retained for backward compatibility; strict sections are always enforced. Defaults to True.

    Returns:
        tuple[int, int]: Tuple of ``(missing_count, total_items_count)``.
    """
    _ = strict_sections
    with open(path, encoding="utf-8") as f:
        content = f.read()

    tree = ast.parse(content, filename=path)
    missing: list[str] = []
    total = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Module):
            total += 1
            doc = ast.get_docstring(node)
            if not doc or not doc.strip():
                missing.append("Module 'module' at line 1")
            elif doc.strip().splitlines()[0].strip() in BANNED_STUB_DOCSTRINGS:
                missing.append("Module 'module' at line 1 (STUB DOCSTRING)")
        elif isinstance(node, ast.ClassDef):
            total += 1
            doc = ast.get_docstring(node)
            if not doc or not doc.strip():
                missing.append(f"ClassDef '{node.name}' at line {node.lineno}")
            elif (
                doc.strip().splitlines()[0].strip() in BANNED_STUB_DOCSTRINGS
                or "Represent the class." in doc
            ):
                missing.append(f"ClassDef '{node.name}' at line {node.lineno} (STUB DOCSTRING)")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            total += 1
            doc = ast.get_docstring(node)
            if not doc or not doc.strip():
                missing.append(f"{type(node).__name__} '{node.name}' at line {node.lineno}")
            elif (
                doc.strip().splitlines()[0].strip() in BANNED_STUB_DOCSTRINGS
                or "Execute the function." in doc
                or "Provide a mock implementation for parity." in doc
            ):
                missing.append(
                    f"{type(node).__name__} '{node.name}' at line {node.lineno} (STUB DOCSTRING)"
                )
            else:
                missing.extend(_check_function_args_doc(node, doc))

    if missing:
        print(f"Missing/Stub in {path}:")
        for m in missing:
            print("  - " + m)
        return len(missing), total
    return 0, total


def run_doc_check(
    target_dirs: Sequence[str] = ("src", "scripts", "tests"),
    output_json_path: str | None = None,
) -> tuple[float, int]:
    """Scan target directories and compute overall documentation coverage percentage.

    Args:
        target_dirs: Sequence of directory paths to scan recursively. Defaults to ``("src", "scripts", "tests")``.
        output_json_path: Optional path where the coverage JSON report is written. Defaults to None.

    Returns:
        tuple[float, int]: Tuple of ``(coverage_percentage, total_missing_count)``.
    """
    total_missing = 0
    total_items = 0

    for directory in target_dirs:
        if not os.path.exists(directory):
            continue
        for root, _, files in sorted(os.walk(directory)):
            for file in sorted(files):
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    m, t = check_file(full_path, strict_sections=True)
                    total_missing += m
                    total_items += t

    coverage = 100.0 if total_items == 0 else ((total_items - total_missing) / total_items) * 100.0
    if output_json_path is not None:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump({"doc_coverage": coverage}, f)
    return coverage, total_missing


def main(
    target_dirs: Sequence[str] = ("src", "scripts", "tests"),
    output_json_path: str | None = None,
) -> int:
    """Execute the documentation coverage check CLI.

    Args:
        target_dirs: Directories to inspect. Defaults to ``("src", "scripts", "tests")``.
        output_json_path: Optional destination JSON file for badge reporting. Defaults to None.

    Returns:
        int: Process exit code (0 when 100% compliant, 1 otherwise).
    """
    coverage, total_missing = run_doc_check(
        target_dirs=target_dirs,
        output_json_path=output_json_path,
    )
    if total_missing > 0:
        print(
            f"Conformance is below 100%. Doc coverage is {coverage:.1f}%. "
            f"Missing/Stub docstrings: {total_missing}"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
