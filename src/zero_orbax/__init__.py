"""zero_orbax framework root module.

This module exports the main public APIs, such as the `checkpoint` submodule,
which provides functionality for saving and restoring machine learning models
and their associated states.
"""

import ml_switcheroo  # type: ignore[import-untyped]

from zero_orbax import checkpoint

__all__ = ["checkpoint"]
