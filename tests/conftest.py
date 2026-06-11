"""Provide module docstring."""

import pytest
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")
    ),
)
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../zero-jax/src")),
)
import ml_switcheroo  # type: ignore[import-untyped]


@pytest.fixture(autouse=True)
def switcheroo_config():
    """Configure the ML switcheroo library to run in eager mode during testing.

    Yields:
        None: Yields control back to the test function while keeping eager mode active.
    """
    with ml_switcheroo.EagerMode():
        yield
