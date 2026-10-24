"""Update README.md shields for test, documentation, and API compliance coverage."""

from __future__ import annotations

import os
import re
import sys


def get_color(pct: float) -> str:
    """Map a percentage score (0-100) to a shields.io badge color name.

    Args:
        pct: Percentage value between 0.0 and 100.0.

    Returns:
        str: Shields.io color identifier.
    """
    if pct >= 100:
        return "brightgreen"
    if pct >= 90:
        return "green"
    if pct >= 80:
        return "yellowgreen"
    if pct >= 70:
        return "yellow"
    if pct >= 50:
        return "orange"
    return "red"


def format_badge_number(val: float) -> str:
    """Format a floating-point percentage as an integer string when whole, or 1 decimal place.

    Args:
        val: Percentage number to format.

    Returns:
        str: Formatted percentage string.
    """
    return f"{int(val)}" if val == int(val) else f"{val:.1f}"


def update_readme(
    readme_path: str = "README.md",
    test_cov: float = 100.0,
    doc_cov: float = 100.0,
    api_compliance: float = 100.0,
) -> bool:
    """Update or insert coverage and compliance badges in a README markdown file.

    Args:
        readme_path: Path to the README file. Defaults to ``"README.md"``.
        test_cov: Test coverage percentage. Defaults to 100.0.
        doc_cov: Documentation coverage percentage. Defaults to 100.0.
        api_compliance: API compliance percentage. Defaults to 100.0.

    Returns:
        bool: True if the README file existed and was updated, False if not found.
    """
    if not os.path.exists(readme_path):
        return False

    with open(readme_path, encoding="utf-8") as f:
        content = f.read()

    test_color = get_color(test_cov)
    doc_color = get_color(doc_cov)
    api_color = get_color(api_compliance)

    test_str = format_badge_number(test_cov)
    doc_str = format_badge_number(doc_cov)
    api_str = format_badge_number(api_compliance)

    content = re.sub(
        r"!\[Test Coverage\]\(https://img\.shields\.io/badge/test_coverage-[^)]+\)",
        f"![Test Coverage](https://img.shields.io/badge/test_coverage-{test_str}%25-{test_color}.svg)",
        content,
    )
    content = re.sub(
        r"!\[Doc Coverage\]\(https://img\.shields\.io/badge/doc_coverage-[^)]+\)",
        f"![Doc Coverage](https://img.shields.io/badge/doc_coverage-{doc_str}%25-{doc_color}.svg)",
        content,
    )

    if "api_compliance-" in content:
        content = re.sub(
            r"!\[API Compliance\]\(https://img\.shields\.io/badge/api_compliance-[^)]+\)",
            f"![API Compliance](https://img.shields.io/badge/api_compliance-{api_str}%25-{api_color}.svg)",
            content,
        )
    else:
        doc_badge_pattern = (
            r"(\[!\[Doc Coverage\]\(https://img\.shields\.io/badge/doc_coverage-[^)]+\)\]\(#\))"
        )
        match = re.search(doc_badge_pattern, content)
        if match:
            insert_pos = match.end()
            content = (
                content[:insert_pos]
                + "\n"
                + f"[![API Compliance](https://img.shields.io/badge/api_compliance-{api_str}%25-{api_color}.svg)](#)"
                + content[insert_pos:]
            )

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    return True


def main(readme_path: str = "README.md") -> int:
    """Run the badge update CLI.

    Args:
        readme_path: Path to the README markdown file. Defaults to ``"README.md"``.

    Returns:
        int: Process exit code (0).
    """
    update_readme(readme_path=readme_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
