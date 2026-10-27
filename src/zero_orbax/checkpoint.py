"""Orbax Checkpoint API (`zero_orbax.checkpoint`).

This module provides a zero-dependency, strongly typed implementation of the
``orbax.checkpoint`` public API, re-exporting decomposed domain modules from
``zero_orbax._checkpoint``.
"""

from __future__ import annotations

import asyncio as _stdlib_asyncio
import contextlib as _stdlib_contextlib
import functools as _stdlib_functools
import logging as _stdlib_logging
from typing_extensions import TypeAlias

from zero_orbax import export as _export_module
from zero_orbax._checkpoint.future_lib import Future, future
from zero_orbax._checkpoint.args_lib import _CheckpointArgsBase, args, checkpoint_args
from zero_orbax._checkpoint.options_lib import (
    AsyncOptions,
    AtomicityOptions,
    CompositeOptions,
    FileOptions,
    MemoryLimitOptions,
    MultiprocessingOptions,
    PyTreeMetadataOptions,
    options,
)
from zero_orbax._checkpoint.step_lib import step, step_lib
from zero_orbax._checkpoint.type_handlers_lib import (
    ArrayRestoreArgs,
    RestoreArgs,
    SaveArgs,
    SingleReplicaArrayRestoreArgs,
    type_handlers,
)
from zero_orbax._checkpoint.transform_lib import (
    RestoreTransform,
    Transform,
    _apply_restore_args_tree,
    _apply_save_args_tree,
    apply_transformations,
    merge_trees,
    transform_utils,
)
from zero_orbax._checkpoint.handlers_lib import (
    ArrayCheckpointHandler,
    AsyncCheckpointHandler,
    BasePyTreeCheckpointHandler,
    CheckpointHandler,
    CheckpointHandlerRegistry,
    CompositeCheckpointHandler,
    DefaultCheckpointHandlerRegistry,
    JaxRandomKeyCheckpointHandler,
    JsonCheckpointHandler,
    NumpyRandomKeyCheckpointHandler,
    ProtoCheckpointHandler,
    PyGrainCheckpointHandler,
    PyTreeCheckpointHandler,
    RandomKeyCheckpointHandler,
    StandardCheckpointHandler,
    async_checkpoint_handler,
    checkpoint_handler,
    create_default_handler_registry,
    handlers,
)
from zero_orbax._checkpoint.checkpointer_lib import (
    AbstractCheckpointer,
    AsyncCheckpointer,
    Checkpointer,
    PyTreeCheckpointer,
    StandardCheckpointer,
    abstract_checkpointer,
    async_checkpointer,
    checkpointer,
    pytree_checkpointer,
    standard_checkpointer,
)
from zero_orbax._checkpoint.manager_lib import (
    AbstractCheckpointManager,
    CheckpointHandlersDict,
    CheckpointManager,
    CheckpointManagerOptions,
    CheckpointersDict,
    abstract_checkpoint_manager,
    checkpoint_manager,
)
from zero_orbax._checkpoint.namespaces_lib import (
    abstract_logger,
    aggregate_handlers,
    atomicity,
    checkpoint,
    epath,
    metadata,
    msgpack_utils,
    multihost,
    nest_asyncio,
    path,
    policies,
    serialization,
    tree,
    utils,
)
from zero_orbax._checkpoint.utils_lib import checkpoint_utils, test_utils
from zero_orbax._checkpoint.v1_lib import (
    load_checkpointables,
    load_pytree,
    save_checkpointables,
    save_pytree,
    save_pytree_async,
    v1,
)
from zero_orbax.types import (
    MultiValueTransformFn,
    ValueTransformFn,
    infer_array_shape as _infer_array_shape,
)

__version__ = "0.6.4"

# Re-exported standard library modules for orbax.checkpoint snapshot compliance
asyncio = _stdlib_asyncio
contextlib = _stdlib_contextlib
functools = _stdlib_functools
logging = _stdlib_logging

NoneType: TypeAlias = None
ValueFn: TypeAlias = ValueTransformFn
MultiValueFn: TypeAlias = MultiValueTransformFn

export = _export_module


class orbax:
    """Namespace providing ``orbax.export`` and ``orbax.checkpoint`` compatibility."""

    export = _export_module
    checkpoint = checkpoint


__all__ = [
    "AbstractCheckpointManager",
    "AbstractCheckpointer",
    "ArrayCheckpointHandler",
    "ArrayRestoreArgs",
    "AsyncCheckpointHandler",
    "AsyncCheckpointer",
    "AsyncOptions",
    "AtomicityOptions",
    "BasePyTreeCheckpointHandler",
    "CheckpointHandler",
    "CheckpointHandlerRegistry",
    "CheckpointHandlersDict",
    "CheckpointManager",
    "CheckpointManagerOptions",
    "Checkpointer",
    "CheckpointersDict",
    "CompositeCheckpointHandler",
    "CompositeOptions",
    "DefaultCheckpointHandlerRegistry",
    "FileOptions",
    "Future",
    "JaxRandomKeyCheckpointHandler",
    "JsonCheckpointHandler",
    "MemoryLimitOptions",
    "MultiprocessingOptions",
    "NumpyRandomKeyCheckpointHandler",
    "ProtoCheckpointHandler",
    "PyGrainCheckpointHandler",
    "PyTreeCheckpointHandler",
    "PyTreeCheckpointer",
    "PyTreeMetadataOptions",
    "RandomKeyCheckpointHandler",
    "RestoreArgs",
    "RestoreTransform",
    "SaveArgs",
    "SingleReplicaArrayRestoreArgs",
    "StandardCheckpointHandler",
    "StandardCheckpointer",
    "Transform",
    "__version__",
    "_CheckpointArgsBase",
    "_apply_restore_args_tree",
    "_apply_save_args_tree",
    "_infer_array_shape",
    "abstract_checkpoint_manager",
    "abstract_checkpointer",
    "abstract_logger",
    "aggregate_handlers",
    "apply_transformations",
    "args",
    "async_checkpoint_handler",
    "async_checkpointer",
    "asyncio",
    "atomicity",
    "checkpoint",
    "checkpoint_args",
    "checkpoint_handler",
    "checkpoint_manager",
    "checkpoint_utils",
    "checkpointer",
    "contextlib",
    "create_default_handler_registry",
    "epath",
    "export",
    "functools",
    "future",
    "handlers",
    "load_checkpointables",
    "load_pytree",
    "logging",
    "merge_trees",
    "metadata",
    "msgpack_utils",
    "multihost",
    "nest_asyncio",
    "options",
    "orbax",
    "path",
    "policies",
    "pytree_checkpointer",
    "save_checkpointables",
    "save_pytree",
    "save_pytree_async",
    "serialization",
    "standard_checkpointer",
    "step",
    "step_lib",
    "test_utils",
    "transform_utils",
    "tree",
    "type_handlers",
    "utils",
    "v1",
]
