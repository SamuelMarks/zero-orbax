import ast
import os
import sys
import json


def check_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    tree = ast.parse(content)
    missing = []
    total = 0

    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            total += 1
            if not ast.get_docstring(node):
                missing.append(
                    f"{type(node).__name__} '{getattr(node, 'name', 'module')}' at line {getattr(node, 'lineno', 1)}"
                )
            elif "Represent the class." in ast.get_docstring(
                node
            ) or "Execute the function." in ast.get_docstring(node):
                missing.append(
                    f"{type(node).__name__} '{getattr(node, 'name', 'module')}' at line {node.lineno} (STUB DOCSTRING)"
                )

    if missing:
        print(f"\nMissing/Stub in {path}:")
        for m in missing:
            print("  - " + m)
        return len(missing), total
    return 0, total


total_missing = 0
total_items = 0

for root, _, files in os.walk("src"):
    for file in files:
        if file.endswith(".py"):
            m, t = check_file(os.path.join(root, file))
            total_missing += m
            total_items += t

for root, _, files in os.walk("tests"):
    for file in files:
        if file.endswith(".py"):
            m, t = check_file(os.path.join(root, file))
            total_missing += m
            total_items += t

for file in os.listdir("."):
    if file.endswith(".py") and os.path.isfile(file) and file != "check_docs.py":
        m, t = check_file(file)
        total_missing += m
        total_items += t

if total_items == 0:
    coverage = 100.0
else:
    coverage = ((total_items - total_missing) / total_items) * 100.0

with open("doc_coverage.json", "w") as f:
    json.dump({"doc_coverage": coverage}, f)

if total_missing > 0:
    print(
        f"\nConformance is below 100%. Doc coverage is {coverage:.1f}%. Missing/Stub docstrings: {total_missing}"
    )
    sys.exit(1)
