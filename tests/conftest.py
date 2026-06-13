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
