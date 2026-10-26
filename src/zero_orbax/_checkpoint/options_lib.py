"""Option dataclasses and namespace for zero_orbax.checkpoint.options."""

from __future__ import annotations

from zero_orbax.types import (
    AsyncOptions,
    AtomicityOptions,
    CompositeOptions,
    FileOptions,
    MemoryLimitOptions,
    MultiprocessingOptions,
    PyTreeMetadataOptions,
)


class options:
    """Namespace for Orbax checkpointing option dataclasses."""

    AsyncOptions = AsyncOptions
    MultiprocessingOptions = MultiprocessingOptions
    FileOptions = FileOptions
    AtomicityOptions = AtomicityOptions
    MemoryLimitOptions = MemoryLimitOptions
    PyTreeMetadataOptions = PyTreeMetadataOptions
    CompositeOptions = CompositeOptions


__all__ = [
    "AsyncOptions",
    "AtomicityOptions",
    "CompositeOptions",
    "FileOptions",
    "MemoryLimitOptions",
    "MultiprocessingOptions",
    "PyTreeMetadataOptions",
    "options",
]
