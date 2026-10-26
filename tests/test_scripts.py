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
    (tmp_path / "ignore.txt").write_text("txt", encoding="utf-8")
    assert check_allowed_imports.main([str(tmp_path)]) == 1
    assert check_allowed_imports.main([]) == 0

    syntax_err = tmp_path / "syntax_err.py"
    syntax_err.write_text("def broken(:" + NL, encoding="utf-8")
    assert check_allowed_imports.check_file(str(syntax_err)) is False


def test_check_docs(tmp_path: pathlib.Path) -> None:
    """Test check_docs across fully documented files, missing docstrings, stub docstrings, and missing/extra Args/Returns/Raises.

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
        '    """Return constant one.',
        "",
        "    Returns:",
        "        int: One.",
        '    """',
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
        "    def _nested() -> int:",
        '        """Nested helper.',
        "",
        "        Returns:",
        "            int: Zero.",
        "",
        "        Raises:",
        "            ValueError: Never.",
        '        """',
        "        if False:",
        '            raise ValueError("nested")',
        "        return 0",
        "    return a + _nested()",
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
        output_json_path=None,
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
        "        z: Extra documented z.",
        '    """',
        '    raise ValueError("err")',
        "",
        "def no_annotation_fn():",
        '    """No return annotation."""',
        "    return None",
    ]
    bad_doc_path = invalid_dir / "bad_docs.py"
    bad_doc_path.write_text(NL.join(bad_doc_lines) + NL, encoding="utf-8")
    m_strict, _ = check_docs.check_file(str(bad_doc_path), strict_sections=True)
    assert m_strict >= 6

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
        "class SubBox(Box, metaclass=type):",
        '    """SubBox class."""',
        "",
        "    def get(self, x: int = 1) -> int:",
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
        "import builtins",
        "from typing import Any, cast",
        "import typing",
        "BadAlias = Any",
        "class BadClass(object, metaclass=Any):",
        "    pass",
        "x: Any = 1  #" + " type: ignore",
        "y: typing.Any = 2",
        "def untyped(a, *args, **kwargs):",
        "    return getattr(a, 'x')",
        "def bad_types(a: object = object, *args: Any, **kwargs: object) -> Any:",
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
    assert generate_snapshot.main(output_path=None) == 0

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
        "args": {"type": "namespace"},
        "NonNamespaceSubmod": {"type": "class", "methods": []},
        "CONST": {"type": "function"},
        "SampleClass": {"type": "class", "methods": ["missing_method"]},
    }
    current_mock: dict[str, dict[str, str | list[str]]] = {
        "args": {"type": "class", "methods": []},
        "NonNamespaceSubmod": {"type": "namespace"},
        "CONST": {"type": "other"},
        "SampleClass": {"type": "class", "methods": ["existing_method"]},
    }
    errs = check_compliance.verify_snapshot_compliance(current_mock, mismatch_snap)
    assert len(errs) == 4

    bad_snap_file = tmp_path / "bad_snap.json"
    bad_snap_file.write_text(
        '{"NonExistentSymbol": {"type": "class", "methods": []}}',
        encoding="utf-8",
    )
    assert check_compliance.main(snapshot_path=str(bad_snap_file)) == 1

    # Test collect_ast_definitions with non-existent file and custom AST source
    custom_src = tmp_path / "custom_defs.py"
    custom_src.write_text(
        NL.join(
            [
                "def Checkpointer():",
                "    pass",
                "class AsyncCheckpointer:",
                "    def __init__(self, handler, timeout_secs: int = 300):",
                "        pass",
                "class PyTreeCheckpointer:",
                '    """Doc without args."""',
                "    def __init__(self, primary_host: int = 0, use_ocdbt: bool = True, *, use_zarr3: bool = False):",
                "        pass",
            ]
        )
        + NL,
        encoding="utf-8",
    )
    defs = check_compliance.collect_ast_definitions(
        [str(tmp_path / "does_not_exist.py"), str(custom_src)]
    )
    v005_errs_missing_json = check_compliance.verify_v005_compliance(
        defs,
        v005_release_snapshot_path=str(tmp_path / "missing_v005.json"),
    )
    assert any("Kind mismatch" in e for e in v005_errs_missing_json)
    assert any("Missing docstring on" in e for e in v005_errs_missing_json)
    assert any("Missing parameter" in e for e in v005_errs_missing_json)
    assert any("Missing type annotation" in e for e in v005_errs_missing_json)
    assert any("Missing docstring entry" in e for e in v005_errs_missing_json)
    assert any("v0.0.5 release snapshot not found" in e for e in v005_errs_missing_json)

    bad_v005_json = tmp_path / "bad_v005.json"
    bad_v005_json.write_text(
        '{"categories": {"array": ['
        '{"api_path": "orbax.checkpoint.MissingRel", "name": "MissingRel", "params": []},'
        '{"api_path": "orbax.checkpoint.AsyncCheckpointer", "name": "AsyncCheckpointer", "params": [{"name": "missing_req_param"}]}'
        "]}}",
        encoding="utf-8",
    )
    v005_errs_rel = check_compliance.verify_v005_compliance(
        defs,
        v005_release_snapshot_path=str(bad_v005_json),
    )
    assert any("Missing v0.12.8 release symbol" in e for e in v005_errs_rel)
    assert any("Missing docstring on v0.12.8 symbol" in e for e in v005_errs_rel)
    assert any("Missing v0.12.8 parameter" in e for e in v005_errs_rel)


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

    readme_2 = tmp_path / "README_2.md"
    readme_2.write_text(
        "# Title"
        + NL
        + "[![Test Coverage](https://img.shields.io/badge/test_coverage-90%25-green.svg)](#)"
        + NL
        + "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-90%25-green.svg)](#)"
        + NL
        + "[![API Compliance](https://img.shields.io/badge/api_compliance-95%25-green.svg)](#)"
        + NL,
        encoding="utf-8",
    )
    assert (
        update_badges.update_readme(
            readme_path=str(readme_2),
            test_cov=88.4,
            doc_cov=92.0,
            api_compliance=100.0,
        )
        is True
    )
    text_2 = readme_2.read_text(encoding="utf-8")
    assert "test_coverage-88.4%25-yellowgreen.svg" in text_2
    assert "doc_coverage-92%25-green.svg" in text_2
    assert "api_compliance-100%25-brightgreen.svg" in text_2

    readme_no_doc = tmp_path / "README_no_doc.md"
    readme_no_doc.write_text("# Only header" + NL, encoding="utf-8")
    assert update_badges.update_readme(readme_path=str(readme_no_doc)) is True


def test_requirements_txt_contains_griffe() -> None:
    """Verify requirements.txt pins griffelib, griffecli, and sibling zero-* dependencies."""
    root_dir = pathlib.Path(__file__).resolve().parent.parent
    req_path = root_dir / "requirements.txt"
    req_txt = req_path.read_text(encoding="utf-8")
    assert (
        "griffelib @ git+https://github.com/SamuelMarks/griffe.git@rest#subdirectory=packages/griffelib"
        in req_txt
    )
    assert (
        "griffecli @ git+https://github.com/SamuelMarks/griffe.git@rest#subdirectory=packages/griffecli"
        in req_txt
    )
    assert (
        "ml-switcheroo-ir @ https://github.com/SamuelMarks/ml-switcheroo-ir/releases/download/v0.0.4/ml_switcheroo_ir-0.0.4-py3-none-any.whl"
        in req_txt
    )
    assert (
        "ml-switcheroo-compiler @ https://github.com/SamuelMarks/ml-switcheroo-compiler/releases/download/v0.0.1/ml_switcheroo_compiler-0.0.1-py3-none-any.whl"
        in req_txt
    )
    assert (
        "ml-ecosystem-snapshots @ https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/download/v0.0.5/ml_ecosystem_snapshots-0.0.5-py3-none-any.whl"
        in req_txt
    )
    for sibling_pkg in ("zero-jax", "zero-chex", "zero-flax", "zero-grain", "zero-optax"):
        assert f"{sibling_pkg} @ git+https://github.com/SamuelMarks/{sibling_pkg}.git@master" in (
            req_txt
        )
    ci_txt = (root_dir / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "pip install --no-deps" in ci_txt
    assert "pip install --no-deps -e ." in ci_txt
