"""Unit tests for all quality, compliance, documentation, and typing scripts in scripts/."""

from __future__ import annotations

import importlib.util
import os
import pathlib

from scripts import (
    check_allowed_imports,
    check_compliance,
    check_docs,
    generate_snapshot,
    lint_typing,
    update_badges,
)

NL = os.linesep


def test_check_allowed_imports(tmp_path: pathlib.Path) -> None:
    """Test check_allowed_imports across valid files, disallowed imports, relative imports, and syntax errors.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    good_file = tmp_path / "good.py"
    good_lines = [
        "import os",
        "from pydantic import BaseModel",
        "from . import local_mod",
        "from .sub import x",
    ]
    good_file.write_text(NL.join(good_lines) + NL, encoding="utf-8")
    assert check_allowed_imports.check_file(str(good_file)) is True
    assert check_allowed_imports.main([str(good_file), "ignore.txt"]) == 0

    bad_import = tmp_path / "bad_import.py"
    bad_lines = [
        "import torch",
        "from tensorflow import keras",
    ]
    bad_import.write_text(NL.join(bad_lines) + NL, encoding="utf-8")
    assert check_allowed_imports.check_file(str(bad_import)) is False
    assert check_allowed_imports.main([str(bad_import)]) == 1

    syntax_err = tmp_path / "syntax_err.py"
    syntax_err.write_text("def broken(:" + NL, encoding="utf-8")
    assert check_allowed_imports.check_file(str(syntax_err)) is False


def test_check_docs(tmp_path: pathlib.Path) -> None:
    """Test check_docs across fully documented files, missing docstrings, stub docstrings, and missing Args.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    valid_dir = tmp_path / "valid_pkg"
    valid_dir.mkdir()
    (valid_dir / "README.txt").write_text("ignored non-py file", encoding="utf-8")
    valid_py = valid_dir / "mod.py"
    valid_lines = [
        '"""Valid module docstring."""',
        "",
        "class Sample:",
        '    """Sample class docstring."""',
        "",
        "def zero_params() -> int:",
        '    """Return constant one."""',
        "    return 1",
        "",
        "async def add(a: int, *args: int, **kwargs: int) -> int:",
        '    """Add integers.',
        "",
        "    Args:",
        "        a: First number",
        "            continued on a second line.",
        "        *args: Extra positional numbers.",
        "        **kwargs: Extra keyword numbers.",
        "",
        "    Returns:",
        "        int: Sum.",
        '    """',
        "    return a",
        "",
        "CONST_VAL = 1",
    ]
    valid_py.write_text(NL.join(valid_lines) + NL, encoding="utf-8")
    out_json = tmp_path / "doc_cov.json"
    assert (
        check_docs.main(
            target_dirs=[str(valid_dir), str(tmp_path / "nonexistent_dir")],
            output_json_path=str(out_json),
        )
        == 0
    )

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    cov, missing = check_docs.run_doc_check(
        target_dirs=[str(empty_dir)],
        output_json_path=str(out_json),
    )
    assert cov == 100.0
    assert missing == 0

    invalid_dir = tmp_path / "invalid_pkg"
    invalid_dir.mkdir()
    bad_doc_lines = [
        "class UndocClass:",
        "    pass",
        "",
        "class StubClass:",
        '    """Represent the class."""',
        "",
        "def no_doc(x: int) -> int:",
        "    return x",
        "",
        "def stub_doc() -> None:",
        '    """Test function."""',
        "    return None",
        "",
        "def missing_arg_doc(x: int, y: int) -> int:",
        '    """Compute value.',
        "",
        "    Args:",
        "        x: Documented x.",
        '    """',
        "    return x + y",
    ]
    (invalid_dir / "bad_docs.py").write_text(NL.join(bad_doc_lines) + NL, encoding="utf-8")
    (invalid_dir / "stub_mod.py").write_text(
        '"""Provide module docstring."""' + NL,
        encoding="utf-8",
    )
    (invalid_dir / "empty_mod.py").write_text("X = 1" + NL, encoding="utf-8")
    assert (
        check_docs.main(
            target_dirs=[str(invalid_dir)],
            output_json_path=str(out_json),
        )
        == 1
    )


def test_lint_typing(tmp_path: pathlib.Path) -> None:
    """Test lint_typing on clean files, Any/object annotations, missing annotations, and reflection calls.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    assert lint_typing.main(["src", "scripts", "tests"]) == 0

    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    clean_lines = [
        '"""Clean module."""',
        "",
        "class Box:",
        '    """Box class."""',
        "",
        "    def get(self, x: int) -> int:",
        '        """Get x.',
        "",
        "        Args:",
        "            x: Input.",
        '        """',
        "        return x",
    ]
    (clean_dir / "ok.py").write_text(NL.join(clean_lines) + NL, encoding="utf-8")
    assert lint_typing.main([str(clean_dir), str(tmp_path / "missing_dir")]) == 0

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    bad_file = bad_dir / "violations.py"
    bad_lines = [
        "from typing import Any, cast",
        "import typing",
        "x: Any = 1  #" + " type: ignore",
        "y: typing.Any = 2",
        "def untyped(a, *args, **kwargs):",
        "    return getattr(a, 'x')",
        "def bad_types(a: object, *args: Any, **kwargs: object) -> Any:",
        "    typing.cast(int, a)",
        "    globals()",
        "    return a.__dict__",
    ]
    bad_file.write_text(NL.join(bad_lines) + NL, encoding="utf-8")
    assert lint_typing.main([str(bad_dir)]) == 1

    syntax_bad = bad_dir / "syntax_bad.py"
    syntax_bad.write_text("def broken(:" + NL, encoding="utf-8")
    errs = lint_typing.scan_file(str(syntax_bad))
    assert len(errs) == 1


def test_check_compliance_and_generate_snapshot(tmp_path: pathlib.Path) -> None:
    """Test check_compliance and generate_snapshot across compliant and mismatched snapshots.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    assert check_compliance.main() == 0

    snap_out = tmp_path / "snap.json"
    assert generate_snapshot.main(output_path=str(snap_out)) == 0
    assert check_compliance.main(snapshot_path=str(snap_out)) == 0

    assert check_compliance.main(snapshot_path=str(tmp_path / "missing_snap.json")) == 1

    dummy_mod_file = tmp_path / "dummy_mod_sample.py"
    dummy_mod_lines = [
        "import types",
        "submod = types.ModuleType('submod')",
        "CONST = 123",
    ]
    dummy_mod_file.write_text(NL.join(dummy_mod_lines) + NL, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("dummy_mod_sample", dummy_mod_file)
    assert spec is not None
    assert spec.loader is not None
    dummy_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dummy_mod)
    api = check_compliance.dump_api(dummy_mod)
    assert api["submod"]["type"] == "namespace"
    assert api["CONST"]["type"] == "other"

    mismatch_snap: dict[str, dict[str, str | list[str]]] = {
        "__version__": {"type": "other"},
        "MissingClass": {"type": "class", "methods": ["foo"]},
        "submod": {"type": "class", "methods": []},
        "CONST": {"type": "function"},
        "SampleClass": {"type": "class", "methods": ["missing_method"]},
    }
    current_mock: dict[str, dict[str, str | list[str]]] = {
        "submod": {"type": "namespace"},
        "CONST": {"type": "other"},
        "SampleClass": {"type": "class", "methods": ["existing_method"]},
    }
    errs = check_compliance.verify_snapshot_compliance(current_mock, mismatch_snap)
    assert len(errs) == 3

    bad_snap_file = tmp_path / "bad_snap.json"
    bad_snap_file.write_text(
        '{"NonExistentSymbol": {"type": "class", "methods": []}}',
        encoding="utf-8",
    )
    assert check_compliance.main(snapshot_path=str(bad_snap_file)) == 1


def test_update_badges(tmp_path: pathlib.Path) -> None:
    """Test update_badges color mapping, number formatting, and README badge replacement.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    assert update_badges.get_color(100.0) == "brightgreen"
    assert update_badges.get_color(95.0) == "green"
    assert update_badges.get_color(85.0) == "yellowgreen"
    assert update_badges.get_color(75.0) == "yellow"
    assert update_badges.get_color(55.0) == "orange"
    assert update_badges.get_color(20.0) == "red"

    assert update_badges.format_badge_number(100.0) == "100"
    assert update_badges.format_badge_number(98.5) == "98.5"

    assert update_badges.update_readme(str(tmp_path / "missing_readme.md")) is False

    readme_1 = tmp_path / "README_1.md"
    readme_lines = [
        "# Title",
        "[![Test Coverage](https://img.shields.io/badge/test_coverage-90%25-green.svg)](#)",
        "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-90%25-green.svg)](#)",
    ]
    readme_1.write_text(NL.join(readme_lines) + NL, encoding="utf-8")
    assert update_badges.main(readme_path=str(readme_1)) == 0
    text_1 = readme_1.read_text(encoding="utf-8")
    assert "api_compliance-100%25-brightgreen.svg" in text_1

    assert (
        update_badges.update_readme(
            str(readme_1),
            test_cov=95.5,
            doc_cov=100.0,
            api_compliance=100.0,
        )
        is True
    )
    text_2 = readme_1.read_text(encoding="utf-8")
    assert "test_coverage-95.5%25-green.svg" in text_2

    readme_plain = tmp_path / "README_plain.md"
    readme_plain.write_text("# Plain README" + NL, encoding="utf-8")
    assert update_badges.update_readme(str(readme_plain)) is True
