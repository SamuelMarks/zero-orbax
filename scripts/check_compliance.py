"""API compliance verification script against local and ml-ecosystem-snapshots v0.0.5 schemas."""

from __future__ import annotations

import ast
import inspect
import json
import os
import pathlib
import sys
import types
from collections.abc import Mapping, Sequence

sys.path[:0] = [
    str(pathlib.Path(__file__).resolve().parent.parent),
    str(pathlib.Path(__file__).resolve().parent.parent / "src"),
]

from scripts.check_docs import _extract_documented_args  # noqa: E402

NAMESPACE_EMULATING_CLASSES: frozenset[str] = frozenset(
    {
        "abstract_checkpoint_manager",
        "abstract_checkpointer",
        "abstract_logger",
        "aggregate_handlers",
        "args",
        "async_checkpoint_handler",
        "async_checkpointer",
        "atomicity",
        "checkpoint",
        "checkpoint_args",
        "checkpoint_handler",
        "checkpoint_manager",
        "checkpoint_utils",
        "checkpointer",
        "epath",
        "export",
        "future",
        "handlers",
        "metadata",
        "msgpack_utils",
        "multihost",
        "nest_asyncio",
        "options",
        "orbax",
        "path",
        "pytree_checkpointer",
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
    }
)

CANONICAL_V005_OP_DEFS: tuple[
    tuple[str, str, str, str, tuple[tuple[str, str, str | None], ...], str],
    ...,
] = (
    (
        "AbstractCheckpointer",
        "orbax.checkpoint.AbstractCheckpointer",
        "class",
        "checkpoint",
        (),
        "Abstract interface for synchronous and asynchronous Orbax checkpointers.",
    ),
    (
        "Checkpointer",
        "orbax.checkpoint.Checkpointer",
        "class",
        "checkpoint",
        (
            ("handler", "POSITIONAL_OR_KEYWORD", None),
            ("primary_host", "KEYWORD_ONLY", "0"),
            ("active_processes", "KEYWORD_ONLY", "None"),
            ("barrier_sync_fn", "KEYWORD_ONLY", "None"),
            ("path_permission_mode", "KEYWORD_ONLY", "None"),
            ("checkpoint_metadata_store", "KEYWORD_ONLY", "None"),
            ("temporary_path_cls", "KEYWORD_ONLY", "None"),
        ),
        "Synchronous checkpointer wrapping a `CheckpointHandler` for atomic step saves and restores.",
    ),
    (
        "AsyncCheckpointer",
        "orbax.checkpoint.AsyncCheckpointer",
        "class",
        "checkpoint",
        (
            ("handler", "POSITIONAL_OR_KEYWORD", None),
            ("timeout_secs", "POSITIONAL_OR_KEYWORD", "300"),
            ("primary_host", "KEYWORD_ONLY", "0"),
            ("active_processes", "KEYWORD_ONLY", "None"),
            ("barrier_sync_fn", "KEYWORD_ONLY", "None"),
            ("post_finalization_callback", "KEYWORD_ONLY", "None"),
            ("path_permission_mode", "KEYWORD_ONLY", "None"),
            ("async_options", "KEYWORD_ONLY", "None"),
            ("multiprocessing_options", "KEYWORD_ONLY", "None"),
            ("temporary_path_cls", "KEYWORD_ONLY", "None"),
        ),
        "Asynchronous checkpointer executing background device-to-host and disk I/O transfers.",
    ),
    (
        "PyTreeCheckpointer",
        "orbax.checkpoint.PyTreeCheckpointer",
        "class",
        "checkpoint",
        (
            ("primary_host", "POSITIONAL_OR_KEYWORD", "0"),
            ("use_ocdbt", "POSITIONAL_OR_KEYWORD", "True"),
            ("use_zarr3", "KEYWORD_ONLY", "False"),
        ),
        "Convenience checkpointer pre-configured with `PyTreeCheckpointHandler`.",
    ),
    (
        "StandardCheckpointer",
        "orbax.checkpoint.StandardCheckpointer",
        "class",
        "checkpoint",
        (
            ("primary_host", "POSITIONAL_OR_KEYWORD", "0"),
            ("multiprocessing_options", "KEYWORD_ONLY", "None"),
            ("async_options", "KEYWORD_ONLY", "None"),
        ),
        "Async checkpointer pre-configured with `StandardCheckpointHandler` for JAX PyTrees.",
    ),
    (
        "CheckpointManager",
        "orbax.checkpoint.CheckpointManager",
        "class",
        "checkpoint",
        (
            ("directory", "POSITIONAL_OR_KEYWORD", None),
            ("checkpointers", "POSITIONAL_OR_KEYWORD", "None"),
            ("options", "POSITIONAL_OR_KEYWORD", "None"),
            ("metadata", "POSITIONAL_OR_KEYWORD", "None"),
            ("item_names", "POSITIONAL_OR_KEYWORD", "None"),
            ("item_handlers", "POSITIONAL_OR_KEYWORD", "None"),
            ("logger", "POSITIONAL_OR_KEYWORD", "None"),
            ("handler_registry", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "High-level manager orchestrating step-numbered checkpoints, retention policies, metrics, and composite items.",
    ),
    (
        "CheckpointManagerOptions",
        "orbax.checkpoint.CheckpointManagerOptions",
        "class",
        "checkpoint",
        (
            ("save_interval_steps", "POSITIONAL_OR_KEYWORD", "1"),
            ("max_to_keep", "POSITIONAL_OR_KEYWORD", "None"),
            ("keep_time_interval", "POSITIONAL_OR_KEYWORD", "None"),
            ("keep_period", "POSITIONAL_OR_KEYWORD", "None"),
            ("should_save_fn", "POSITIONAL_OR_KEYWORD", "None"),
            ("best_fn", "POSITIONAL_OR_KEYWORD", "None"),
            ("best_mode", "POSITIONAL_OR_KEYWORD", "'max'"),
            ("keep_checkpoints_without_metrics", "POSITIONAL_OR_KEYWORD", "True"),
            ("step_prefix", "POSITIONAL_OR_KEYWORD", "None"),
            ("step_format_fixed_length", "POSITIONAL_OR_KEYWORD", "None"),
            ("step_name_format", "POSITIONAL_OR_KEYWORD", "None"),
            ("create", "POSITIONAL_OR_KEYWORD", "True"),
            ("cleanup_tmp_directories", "POSITIONAL_OR_KEYWORD", "False"),
            ("save_on_steps", "POSITIONAL_OR_KEYWORD", "None"),
            ("single_host_load_and_broadcast", "POSITIONAL_OR_KEYWORD", "False"),
            ("todelete_subdir", "POSITIONAL_OR_KEYWORD", "None"),
            ("enable_background_delete", "POSITIONAL_OR_KEYWORD", "False"),
            ("read_only", "POSITIONAL_OR_KEYWORD", "False"),
            ("enable_async_checkpointing", "POSITIONAL_OR_KEYWORD", "True"),
            ("async_options", "POSITIONAL_OR_KEYWORD", "None"),
            ("multiprocessing_options", "POSITIONAL_OR_KEYWORD", "None"),
            ("save_decision_policy", "POSITIONAL_OR_KEYWORD", "None"),
            ("preservation_policy", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Configuration options controlling `CheckpointManager` save frequency, retention, naming, and async execution.",
    ),
    (
        "AsyncOptions",
        "orbax.checkpoint.AsyncOptions",
        "class",
        "checkpoint",
        (
            ("timeout_secs", "POSITIONAL_OR_KEYWORD", "600"),
            ("barrier_sync_fn", "POSITIONAL_OR_KEYWORD", "None"),
            ("post_finalization_callback", "POSITIONAL_OR_KEYWORD", "None"),
            ("create_directories_asynchronously", "POSITIONAL_OR_KEYWORD", "True"),
            ("async_origin", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Options configuring asynchronous checkpointing timeouts and synchronization callbacks.",
    ),
    (
        "MultiprocessingOptions",
        "orbax.checkpoint.MultiprocessingOptions",
        "class",
        "checkpoint",
        (
            ("primary_host", "POSITIONAL_OR_KEYWORD", "0"),
            ("active_processes", "POSITIONAL_OR_KEYWORD", "None"),
            ("barrier_sync_key_prefix", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Options configuring multi-host JAX process coordination during checkpointing.",
    ),
    (
        "CheckpointHandler",
        "orbax.checkpoint.CheckpointHandler",
        "class",
        "handlers",
        (),
        "Base interface for synchronous serialization and deserialization of checkpointable items.",
    ),
    (
        "AsyncCheckpointHandler",
        "orbax.checkpoint.AsyncCheckpointHandler",
        "class",
        "handlers",
        (),
        "Base interface for asynchronous checkpoint handlers returning background commit futures.",
    ),
    (
        "PyTreeCheckpointHandler",
        "orbax.checkpoint.PyTreeCheckpointHandler",
        "class",
        "handlers",
        (
            ("aggregate_filename", "POSITIONAL_OR_KEYWORD", "None"),
            ("use_ocdbt", "POSITIONAL_OR_KEYWORD", "True"),
            ("use_zarr3", "POSITIONAL_OR_KEYWORD", "False"),
            ("primary_host", "POSITIONAL_OR_KEYWORD", "0"),
            ("type_handler_registry", "POSITIONAL_OR_KEYWORD", "None"),
            ("handler_impl", "POSITIONAL_OR_KEYWORD", "None"),
            ("save_concurrent_gb", "POSITIONAL_OR_KEYWORD", "None"),
            ("restore_concurrent_gb", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Handler for saving and restoring arbitrary JAX PyTrees using TensorStore, OCDBT, and Zarr.",
    ),
    (
        "StandardCheckpointHandler",
        "orbax.checkpoint.StandardCheckpointHandler",
        "class",
        "handlers",
        (
            ("save_concurrent_gb", "POSITIONAL_OR_KEYWORD", "None"),
            ("restore_concurrent_gb", "POSITIONAL_OR_KEYWORD", "None"),
            ("multiprocessing_options", "KEYWORD_ONLY", "None"),
        ),
        "Simplified PyTree handler using `StandardSave` and `StandardRestore` argument wrappers.",
    ),
    (
        "JsonCheckpointHandler",
        "orbax.checkpoint.JsonCheckpointHandler",
        "class",
        "handlers",
        (
            ("filename", "POSITIONAL_OR_KEYWORD", "None"),
            ("multiprocessing_options", "KEYWORD_ONLY", "None"),
        ),
        "Handler for saving and restoring nested JSON-serializable metadata dictionaries.",
    ),
    (
        "ArrayCheckpointHandler",
        "orbax.checkpoint.ArrayCheckpointHandler",
        "class",
        "handlers",
        (
            ("checkpoint_name", "POSITIONAL_OR_KEYWORD", "None"),
            ("multiprocessing_options", "KEYWORD_ONLY", "None"),
        ),
        "Handler for saving and restoring individual NumPy or `jax.Array` instances.",
    ),
    (
        "ProtoCheckpointHandler",
        "orbax.checkpoint.ProtoCheckpointHandler",
        "class",
        "handlers",
        (
            ("filename", "POSITIONAL_OR_KEYWORD", None),
            ("multiprocessing_options", "KEYWORD_ONLY", "None"),
        ),
        "Handler for saving and restoring Protocol Buffer messages in text format.",
    ),
    (
        "RandomKeyCheckpointHandler",
        "orbax.checkpoint.RandomKeyCheckpointHandler",
        "class",
        "handlers",
        (),
        "Handler for saving and restoring JAX PRNG keys.",
    ),
    (
        "CompositeCheckpointHandler",
        "orbax.checkpoint.CompositeCheckpointHandler",
        "class",
        "handlers",
        (
            ("item_names", "VAR_POSITIONAL", None),
            ("composite_options", "KEYWORD_ONLY", "None"),
            ("items_and_handlers", "VAR_KEYWORD", None),
        ),
        "Handler delegating multiple named checkpoint items to specialized sub-handlers.",
    ),
    (
        "CheckpointHandlerRegistry",
        "orbax.checkpoint.CheckpointHandlerRegistry",
        "class",
        "handlers",
        (),
        "Protocol for mapping checkpoint item names and `CheckpointArgs` types to `CheckpointHandler` instances.",
    ),
    (
        "DefaultCheckpointHandlerRegistry",
        "orbax.checkpoint.DefaultCheckpointHandlerRegistry",
        "class",
        "handlers",
        (("other_registry", "POSITIONAL_OR_KEYWORD", "None"),),
        "Default mutable registry mapping `CheckpointArgs` types to their corresponding handlers.",
    ),
    (
        "create_default_handler_registry",
        "orbax.checkpoint.create_default_handler_registry",
        "function",
        "handlers",
        (("items_and_handlers", "VAR_KEYWORD", None),),
        "Factory creating a `CheckpointHandlerRegistry` populated with `(item_name, handler)` pairs.",
    ),
    (
        "CheckpointArgs",
        "orbax.checkpoint.args.CheckpointArgs",
        "class",
        "args",
        (),
        "Base dataclass for declarative save and restore arguments.",
    ),
    (
        "StandardSave",
        "orbax.checkpoint.args.StandardSave",
        "class",
        "args",
        (
            ("item", "POSITIONAL_OR_KEYWORD", None),
            ("save_args", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Declarative `CheckpointArgs` for saving a JAX PyTree via `StandardCheckpointHandler`.",
    ),
    (
        "StandardRestore",
        "orbax.checkpoint.args.StandardRestore",
        "class",
        "args",
        (
            ("item", "POSITIONAL_OR_KEYWORD", "None"),
            ("strict", "POSITIONAL_OR_KEYWORD", "True"),
        ),
        "Declarative `CheckpointArgs` for restoring a JAX PyTree via `StandardCheckpointHandler`.",
    ),
    (
        "PyTreeSave",
        "orbax.checkpoint.args.PyTreeSave",
        "class",
        "args",
        (
            ("item", "POSITIONAL_OR_KEYWORD", None),
            ("save_args", "POSITIONAL_OR_KEYWORD", "None"),
            ("ocdbt_target_data_file_size", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Declarative `CheckpointArgs` for saving a PyTree via `PyTreeCheckpointHandler`.",
    ),
    (
        "PyTreeRestore",
        "orbax.checkpoint.args.PyTreeRestore",
        "class",
        "args",
        (
            ("item", "POSITIONAL_OR_KEYWORD", "None"),
            ("save_args", "POSITIONAL_OR_KEYWORD", "None"),
            ("restore_args", "POSITIONAL_OR_KEYWORD", "None"),
            ("transforms", "POSITIONAL_OR_KEYWORD", "None"),
            ("transforms_default_to_original", "POSITIONAL_OR_KEYWORD", "True"),
        ),
        "Declarative `CheckpointArgs` for restoring a PyTree via `PyTreeCheckpointHandler`.",
    ),
    (
        "JsonSave",
        "orbax.checkpoint.args.JsonSave",
        "class",
        "args",
        (("item", "POSITIONAL_OR_KEYWORD", None),),
        "Declarative `CheckpointArgs` for saving JSON metadata via `JsonCheckpointHandler`.",
    ),
    (
        "JsonRestore",
        "orbax.checkpoint.args.JsonRestore",
        "class",
        "args",
        (("item", "POSITIONAL_OR_KEYWORD", "None"),),
        "Declarative `CheckpointArgs` for restoring JSON metadata via `JsonCheckpointHandler`.",
    ),
    (
        "ArraySave",
        "orbax.checkpoint.args.ArraySave",
        "class",
        "args",
        (
            ("item", "POSITIONAL_OR_KEYWORD", None),
            ("save_args", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Declarative `CheckpointArgs` for saving a single array via `ArrayCheckpointHandler`.",
    ),
    (
        "ArrayRestore",
        "orbax.checkpoint.args.ArrayRestore",
        "class",
        "args",
        (
            ("item", "POSITIONAL_OR_KEYWORD", "None"),
            ("restore_args", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Declarative `CheckpointArgs` for restoring a single array via `ArrayCheckpointHandler`.",
    ),
    (
        "ProtoSave",
        "orbax.checkpoint.args.ProtoSave",
        "class",
        "args",
        (("item", "POSITIONAL_OR_KEYWORD", None),),
        "Declarative `CheckpointArgs` for saving a protobuf message via `ProtoCheckpointHandler`.",
    ),
    (
        "ProtoRestore",
        "orbax.checkpoint.args.ProtoRestore",
        "class",
        "args",
        (("item", "POSITIONAL_OR_KEYWORD", None),),
        "Declarative `CheckpointArgs` for restoring a protobuf message via `ProtoCheckpointHandler`.",
    ),
    (
        "JaxRandomKeySave",
        "orbax.checkpoint.args.JaxRandomKeySave",
        "class",
        "args",
        (("item", "POSITIONAL_OR_KEYWORD", None),),
        "Declarative `CheckpointArgs` for saving a JAX PRNG key.",
    ),
    (
        "JaxRandomKeyRestore",
        "orbax.checkpoint.args.JaxRandomKeyRestore",
        "class",
        "args",
        (),
        "Declarative `CheckpointArgs` for restoring a JAX PRNG key.",
    ),
    (
        "NumpyRandomKeySave",
        "orbax.checkpoint.args.NumpyRandomKeySave",
        "class",
        "args",
        (("item", "POSITIONAL_OR_KEYWORD", None),),
        "Declarative `CheckpointArgs` for saving a NumPy random generator state.",
    ),
    (
        "NumpyRandomKeyRestore",
        "orbax.checkpoint.args.NumpyRandomKeyRestore",
        "class",
        "args",
        (),
        "Declarative `CheckpointArgs` for restoring a NumPy random generator state.",
    ),
    (
        "Composite",
        "orbax.checkpoint.args.Composite",
        "class",
        "args",
        (("items", "VAR_KEYWORD", None),),
        "Container mapping composite item names to their respective `CheckpointArgs` specs.",
    ),
    (
        "SaveArgs",
        "orbax.checkpoint.SaveArgs",
        "class",
        "type_handlers",
        (
            ("aggregate", "POSITIONAL_OR_KEYWORD", "False"),
            ("dtype", "POSITIONAL_OR_KEYWORD", "None"),
            ("chunk_byte_size", "POSITIONAL_OR_KEYWORD", "None"),
            ("ocdbt_target_data_file_size", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Per-leaf configuration controlling dtype casting and chunk sizes during save.",
    ),
    (
        "RestoreArgs",
        "orbax.checkpoint.RestoreArgs",
        "class",
        "type_handlers",
        (
            ("restore_type", "POSITIONAL_OR_KEYWORD", "None"),
            ("dtype", "POSITIONAL_OR_KEYWORD", "None"),
            ("strict", "POSITIONAL_OR_KEYWORD", "True"),
        ),
        "Per-leaf configuration specifying target `restore_type` and `dtype` during restore.",
    ),
    (
        "ArrayRestoreArgs",
        "orbax.checkpoint.ArrayRestoreArgs",
        "class",
        "type_handlers",
        (
            ("restore_type", "POSITIONAL_OR_KEYWORD", "None"),
            ("mesh", "POSITIONAL_OR_KEYWORD", "None"),
            ("mesh_axes", "POSITIONAL_OR_KEYWORD", "None"),
            ("sharding", "POSITIONAL_OR_KEYWORD", "None"),
            ("global_shape", "POSITIONAL_OR_KEYWORD", "None"),
            ("shape", "POSITIONAL_OR_KEYWORD", "None"),
            ("dtype", "POSITIONAL_OR_KEYWORD", "None"),
            ("strict", "POSITIONAL_OR_KEYWORD", "True"),
        ),
        "Per-leaf restore arguments specifying sharding and global_shape for array leaves.",
    ),
    (
        "SingleReplicaArrayRestoreArgs",
        "orbax.checkpoint.SingleReplicaArrayRestoreArgs",
        "class",
        "type_handlers",
        (
            ("restore_type", "POSITIONAL_OR_KEYWORD", "None"),
            ("mesh", "POSITIONAL_OR_KEYWORD", "None"),
            ("mesh_axes", "POSITIONAL_OR_KEYWORD", "None"),
            ("sharding", "POSITIONAL_OR_KEYWORD", "None"),
            ("global_shape", "POSITIONAL_OR_KEYWORD", "None"),
            ("shape", "POSITIONAL_OR_KEYWORD", "None"),
            ("dtype", "POSITIONAL_OR_KEYWORD", "None"),
            ("strict", "POSITIONAL_OR_KEYWORD", "True"),
            ("single_replica_sharding", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Restore arguments reading array shards on a single replica and broadcasting across replicas.",
    ),
    (
        "Transform",
        "orbax.checkpoint.Transform",
        "class",
        "transform_utils",
        (
            ("original_key", "POSITIONAL_OR_KEYWORD", "None"),
            ("use_fallback", "POSITIONAL_OR_KEYWORD", "False"),
            ("value_fn", "POSITIONAL_OR_KEYWORD", "None"),
            ("multi_value_fn", "POSITIONAL_OR_KEYWORD", "None"),
            ("multi_value_fn_input_args", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Rule describing structural key renaming or value transformation between PyTrees.",
    ),
    (
        "RestoreTransform",
        "orbax.checkpoint.RestoreTransform",
        "class",
        "transform_utils",
        (
            ("original_key", "POSITIONAL_OR_KEYWORD", "None"),
            ("use_fallback", "POSITIONAL_OR_KEYWORD", "False"),
            ("value_fn", "POSITIONAL_OR_KEYWORD", "None"),
            ("multi_value_fn", "POSITIONAL_OR_KEYWORD", "None"),
            ("multi_value_fn_input_args", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Transformation rule applied during PyTree checkpoint restoration.",
    ),
    (
        "apply_transformations",
        "orbax.checkpoint.apply_transformations",
        "function",
        "transform_utils",
        (
            ("original_tree", "POSITIONAL_OR_KEYWORD", None),
            ("transformations", "POSITIONAL_OR_KEYWORD", None),
            ("new_tree", "POSITIONAL_OR_KEYWORD", None),
            ("default_to_original", "POSITIONAL_OR_KEYWORD", "True"),
        ),
        "Apply a tree of `Transform` rules to map `original_tree` into the structure of `new_tree`.",
    ),
    (
        "construct_restore_args",
        "orbax.checkpoint.checkpoint_utils.construct_restore_args",
        "function",
        "checkpoint_utils",
        (
            ("target", "POSITIONAL_OR_KEYWORD", None),
            ("sharding_tree", "POSITIONAL_OR_KEYWORD", "None"),
            ("set_global_shape", "POSITIONAL_OR_KEYWORD", "True"),
        ),
        "Construct a PyTree of `RestoreArgs` / `ArrayRestoreArgs` matching a target abstract PyTree.",
    ),
    (
        "restore_args_from_target",
        "orbax.checkpoint.checkpoint_utils.restore_args_from_target",
        "function",
        "checkpoint_utils",
        (
            ("target", "POSITIONAL_OR_KEYWORD", None),
            ("sharding_tree", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Create a PyTree of `RestoreArgs` inferred from `target` and `sharding_tree`.",
    ),
    (
        "wait_for_new_checkpoint",
        "orbax.checkpoint.checkpoint_utils.wait_for_new_checkpoint",
        "function",
        "checkpoint_utils",
        (
            ("checkpoint_dir", "POSITIONAL_OR_KEYWORD", None),
            ("until_step", "POSITIONAL_OR_KEYWORD", "None"),
            ("seconds_to_sleep", "POSITIONAL_OR_KEYWORD", "1"),
            ("timeout", "POSITIONAL_OR_KEYWORD", "None"),
            ("step_prefix", "POSITIONAL_OR_KEYWORD", "None"),
            ("step_format_fixed_length", "POSITIONAL_OR_KEYWORD", "None"),
            ("step_name_format", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Block until a new checkpoint step appears in `checkpoint_dir`.",
    ),
    (
        "msgpack_serialize",
        "orbax.checkpoint.msgpack_utils.msgpack_serialize",
        "function",
        "msgpack_utils",
        (
            ("pytree", "POSITIONAL_OR_KEYWORD", None),
            ("in_place", "POSITIONAL_OR_KEYWORD", "False"),
        ),
        "Serialize a PyTree into MsgPack bytes.",
    ),
    (
        "msgpack_restore",
        "orbax.checkpoint.msgpack_utils.msgpack_restore",
        "function",
        "msgpack_utils",
        (("encoded_pytree", "POSITIONAL_OR_KEYWORD", None),),
        "Deserialize MsgPack bytes back into a Python structure.",
    ),
    (
        "save_pytree",
        "orbax.checkpoint.v1.save_pytree",
        "function",
        "v1",
        (
            ("path", "POSITIONAL_OR_KEYWORD", None),
            ("pytree", "POSITIONAL_OR_KEYWORD", None),
            ("overwrite", "KEYWORD_ONLY", "False"),
            ("custom_metadata", "KEYWORD_ONLY", "None"),
        ),
        "Synchronously save a JAX PyTree to `path` using the Orbax v1 functional API.",
    ),
    (
        "save_pytree_async",
        "orbax.checkpoint.v1.save_pytree_async",
        "function",
        "v1",
        (
            ("path", "POSITIONAL_OR_KEYWORD", None),
            ("pytree", "POSITIONAL_OR_KEYWORD", None),
            ("overwrite", "KEYWORD_ONLY", "False"),
            ("custom_metadata", "KEYWORD_ONLY", "None"),
        ),
        "Asynchronously save a JAX PyTree to `path` using the Orbax v1 functional API.",
    ),
    (
        "load_pytree",
        "orbax.checkpoint.v1.load_pytree",
        "function",
        "v1",
        (
            ("path", "POSITIONAL_OR_KEYWORD", None),
            ("abstract_pytree", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Load a JAX PyTree from `path` using optional `abstract_pytree` specifications.",
    ),
    (
        "save_checkpointables",
        "orbax.checkpoint.v1.save_checkpointables",
        "function",
        "v1",
        (
            ("path", "POSITIONAL_OR_KEYWORD", None),
            ("checkpointables", "POSITIONAL_OR_KEYWORD", None),
            ("overwrite", "KEYWORD_ONLY", "False"),
            ("custom_metadata", "KEYWORD_ONLY", "None"),
        ),
        "Save a dictionary of named checkpointables to `path`.",
    ),
    (
        "load_checkpointables",
        "orbax.checkpoint.v1.load_checkpointables",
        "function",
        "v1",
        (
            ("path", "POSITIONAL_OR_KEYWORD", None),
            ("abstract_checkpointables", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Load a dictionary of named checkpointables from `path`.",
    ),
    (
        "ExportManager",
        "orbax.export.ExportManager",
        "class",
        "export",
        (
            ("module", "POSITIONAL_OR_KEYWORD", None),
            ("serving_configs", "POSITIONAL_OR_KEYWORD", None),
        ),
        "Manager exporting a `JaxModule` across `serving_configs`.",
    ),
    (
        "JaxModule",
        "orbax.export.JaxModule",
        "class",
        "export",
        (
            ("params", "POSITIONAL_OR_KEYWORD", None),
            ("apply_fn", "POSITIONAL_OR_KEYWORD", None),
            ("trainable", "POSITIONAL_OR_KEYWORD", "None"),
            ("input_polymorphic_shape", "POSITIONAL_OR_KEYWORD", "None"),
            ("jax2tf_kwargs", "POSITIONAL_OR_KEYWORD", "None"),
            ("jit_compile", "POSITIONAL_OR_KEYWORD", "True"),
            ("pspecs", "POSITIONAL_OR_KEYWORD", "None"),
            ("allow_multi_axis_sharding_consolidation", "POSITIONAL_OR_KEYWORD", "None"),
            ("export_version", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Exportable wrapper encapsulating JAX model parameters and pure `apply_fn` callables.",
    ),
    (
        "ServingConfig",
        "orbax.export.ServingConfig",
        "class",
        "export",
        (
            ("signature_key", "POSITIONAL_OR_KEYWORD", None),
            ("input_signature", "POSITIONAL_OR_KEYWORD", "None"),
            ("tf_preprocessor", "POSITIONAL_OR_KEYWORD", "None"),
            ("tf_postprocessor", "POSITIONAL_OR_KEYWORD", "None"),
            ("extra_trackable_resources", "POSITIONAL_OR_KEYWORD", "None"),
            ("method_key", "POSITIONAL_OR_KEYWORD", "None"),
            ("obm_export_options", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Configuration binding a serving `signature_key` and `input_signature` with optional pre/postprocessors.",
    ),
    (
        "ObmExportOptions",
        "orbax.export.ObmExportOptions",
        "class",
        "export",
        (
            ("enable_auto_layout", "POSITIONAL_OR_KEYWORD", "False"),
            ("checkpoint_path", "POSITIONAL_OR_KEYWORD", "None"),
        ),
        "Options configuring native Orbax Model (OBM) export.",
    ),
)

DEFAULT_SOURCE_FILES: tuple[str, ...] = (
    "src/zero_orbax/__init__.py",
    "src/zero_orbax/types.py",
    "src/zero_orbax/core.py",
    "src/zero_orbax/checkpoint.py",
    "src/zero_orbax/export.py",
    "src/zero_orbax/_checkpoint/__init__.py",
    "src/zero_orbax/_checkpoint/args_lib.py",
    "src/zero_orbax/_checkpoint/checkpointer_lib.py",
    "src/zero_orbax/_checkpoint/future_lib.py",
    "src/zero_orbax/_checkpoint/handlers_lib.py",
    "src/zero_orbax/_checkpoint/manager_lib.py",
    "src/zero_orbax/_checkpoint/namespaces_lib.py",
    "src/zero_orbax/_checkpoint/options_lib.py",
    "src/zero_orbax/_checkpoint/step_lib.py",
    "src/zero_orbax/_checkpoint/transform_lib.py",
    "src/zero_orbax/_checkpoint/type_handlers_lib.py",
    "src/zero_orbax/_checkpoint/utils_lib.py",
    "src/zero_orbax/_checkpoint/v1_lib.py",
)

DEFAULT_STRUCTURAL_SNAPSHOT: dict[str, dict[str, str | list[str]]] = {
    "AbstractCheckpointManager": {
        "type": "class",
        "methods": [
            "all_steps",
            "best_step",
            "check_for_errors",
            "close",
            "delete",
            "item_metadata",
            "latest_step",
            "metadata",
            "metrics",
            "reached_preemption",
            "reload",
            "restore",
            "save",
            "should_save",
            "wait_until_finished",
        ],
    },
    "AbstractCheckpointer": {
        "type": "class",
        "methods": ["close", "metadata", "restore", "save", "structure"],
    },
    "ArrayCheckpointHandler": {
        "type": "class",
        "methods": ["async_save", "close", "finalize", "metadata", "restore", "save"],
    },
    "ArrayRestoreArgs": {"type": "class", "methods": ["restore_type"]},
    "AsyncCheckpointHandler": {
        "type": "class",
        "methods": ["async_save", "close", "finalize", "metadata", "restore", "save"],
    },
    "AsyncCheckpointer": {
        "type": "class",
        "methods": [
            "check_for_errors",
            "close",
            "create_temporary_path",
            "metadata",
            "restore",
            "save",
            "structure",
            "wait_until_finished",
        ],
    },
    "AsyncOptions": {"type": "class", "methods": []},
    "BasePyTreeCheckpointHandler": {
        "type": "class",
        "methods": [
            "async_save",
            "close",
            "finalize",
            "get_param_names",
            "metadata",
            "restore",
            "save",
        ],
    },
    "CheckpointHandler": {
        "type": "class",
        "methods": ["close", "finalize", "metadata", "restore", "save"],
    },
    "CheckpointHandlerRegistry": {
        "type": "class",
        "methods": ["add", "get", "get_all_entries", "has"],
    },
    "CheckpointManager": {
        "type": "class",
        "methods": [
            "all_steps",
            "best_step",
            "check_for_errors",
            "close",
            "delete",
            "is_saving_in_progress",
            "item_metadata",
            "latest_step",
            "metadata",
            "metrics",
            "reached_preemption",
            "reload",
            "restore",
            "save",
            "should_save",
            "wait_until_finished",
        ],
    },
    "CheckpointManagerOptions": {"type": "class", "methods": []},
    "Checkpointer": {
        "type": "class",
        "methods": [
            "close",
            "create_temporary_path",
            "metadata",
            "restore",
            "save",
            "structure",
        ],
    },
    "CompositeCheckpointHandler": {
        "type": "class",
        "methods": ["async_save", "close", "finalize", "metadata", "restore", "save"],
    },
    "CompositeOptions": {"type": "class", "methods": []},
    "DefaultCheckpointHandlerRegistry": {
        "type": "class",
        "methods": ["add", "get", "get_all_entries", "has"],
    },
    "Future": {"type": "class", "methods": ["result"]},
    "JaxRandomKeyCheckpointHandler": {
        "type": "class",
        "methods": [
            "async_save",
            "checkpoint_restore_args",
            "checkpoint_save_args",
            "close",
            "finalize",
            "metadata",
            "post_restore",
            "restore",
            "save",
        ],
    },
    "JsonCheckpointHandler": {
        "type": "class",
        "methods": ["async_save", "close", "finalize", "metadata", "restore", "save"],
    },
    "NumpyRandomKeyCheckpointHandler": {
        "type": "class",
        "methods": [
            "async_save",
            "checkpoint_restore_args",
            "checkpoint_save_args",
            "close",
            "finalize",
            "metadata",
            "post_restore",
            "restore",
            "save",
        ],
    },
    "ProtoCheckpointHandler": {
        "type": "class",
        "methods": ["async_save", "close", "finalize", "metadata", "restore", "save"],
    },
    "PyTreeCheckpointHandler": {
        "type": "class",
        "methods": ["async_save", "close", "finalize", "metadata", "restore", "save"],
    },
    "PyTreeCheckpointer": {
        "type": "class",
        "methods": [
            "close",
            "create_temporary_path",
            "metadata",
            "restore",
            "save",
            "structure",
        ],
    },
    "RestoreArgs": {"type": "class", "methods": []},
    "RestoreTransform": {"type": "class", "methods": []},
    "SaveArgs": {"type": "class", "methods": []},
    "StandardCheckpointHandler": {
        "type": "class",
        "methods": ["async_save", "close", "finalize", "metadata", "restore", "save"],
    },
    "StandardCheckpointer": {
        "type": "class",
        "methods": [
            "check_for_errors",
            "close",
            "create_temporary_path",
            "metadata",
            "restore",
            "save",
            "structure",
            "wait_until_finished",
        ],
    },
    "Transform": {"type": "class", "methods": []},
    "__version__": {"type": "other"},
    "abstract_checkpoint_manager": {"type": "namespace"},
    "abstract_checkpointer": {"type": "namespace"},
    "aggregate_handlers": {"type": "namespace"},
    "apply_transformations": {"type": "function"},
    "args": {"type": "namespace"},
    "async_checkpointer": {"type": "namespace"},
    "asyncio": {"type": "namespace"},
    "checkpoint_args": {"type": "namespace"},
    "checkpoint_manager": {"type": "namespace"},
    "checkpoint_utils": {"type": "namespace"},
    "checkpointer": {"type": "namespace"},
    "contextlib": {"type": "namespace"},
    "functools": {"type": "namespace"},
    "future": {"type": "namespace"},
    "handlers": {"type": "namespace"},
    "logging": {"type": "namespace"},
    "merge_trees": {"type": "function"},
    "metadata": {"type": "namespace"},
    "msgpack_utils": {"type": "namespace"},
    "multihost": {"type": "namespace"},
    "nest_asyncio": {"type": "namespace"},
    "options": {"type": "namespace"},
    "path": {"type": "namespace"},
    "pytree_checkpointer": {"type": "namespace"},
    "serialization": {"type": "namespace"},
    "standard_checkpointer": {"type": "namespace"},
    "step": {"type": "namespace"},
    "test_utils": {"type": "namespace"},
    "transform_utils": {"type": "namespace"},
    "tree": {"type": "namespace"},
    "type_handlers": {"type": "namespace"},
    "utils": {"type": "namespace"},
}


def _resolve_v005_release_snapshot_path(
    v005_release_snapshot_path: str | None = None,
) -> str:
    """Resolve the path to ``orbax_checkpoint_v0.12.8.json`` from ``ml-ecosystem-snapshots`` v0.0.5.

    Args:
        v005_release_snapshot_path: Optional explicit path to the snapshot JSON. Defaults to None.

    Returns:
        str: Resolved filesystem path to ``orbax_checkpoint_v0.12.8.json``.
    """
    if v005_release_snapshot_path is not None:
        return v005_release_snapshot_path
    from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path

    return str(resolve_snapshot_path("orbax_checkpoint_v0.12.8.json", must_exist=False))


def dump_api(module: types.ModuleType) -> dict[str, dict[str, str | list[str]]]:
    """Introspect a Python module and build a structural dictionary of its public symbols.

    Args:
        module: The imported module object to inspect.

    Returns:
        dict[str, dict[str, str | list[str]]]: Mapping of symbol name to ``{"type": ..., "methods": ...}``.
    """
    api: dict[str, dict[str, str | list[str]]] = {}
    for name, obj in inspect.getmembers(module):
        if name.startswith("_") and name not in ("__version__",):
            continue
        if inspect.isclass(obj):
            methods: list[str] = [
                m_name
                for m_name, m_obj in inspect.getmembers(obj)
                if not m_name.startswith("_") and callable(m_obj)
            ]
            api[name] = {"type": "class", "methods": sorted(methods)}
        elif isinstance(obj, types.ModuleType):
            api[name] = {"type": "namespace"}
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
            api[name] = {"type": "function"}
        else:
            api[name] = {"type": "other"}
    return api


def verify_snapshot_compliance(
    current_api: Mapping[str, Mapping[str, str | list[str]]],
    snapshot: Mapping[str, Mapping[str, str | list[str]]],
) -> list[str]:
    """Compare the introspected API against the expected structural snapshot dictionary.

    Args:
        current_api: Dictionary produced by ``dump_api``.
        snapshot: Expected snapshot dictionary loaded from ``orbax_api_snapshot.json``.

    Returns:
        list[str]: List of mismatch error strings (empty if 100% compliant).
    """
    errors: list[str] = []
    ignored_names = {
        "__version__",
        "logging",
        "contextlib",
        "functools",
        "asyncio",
        "nest_asyncio",
    }

    for name, expected in snapshot.items():
        if name in ignored_names:
            continue
        if name not in current_api:
            errors.append(f"Missing symbol in zero_orbax.checkpoint: {name}")
            continue

        actual = current_api[name]
        expected_type = expected.get("type")
        actual_type = actual.get("type")

        if (
            name in NAMESPACE_EMULATING_CLASSES
            and expected_type == "namespace"
            and actual_type in ("class", "namespace")
        ):
            continue
        if expected_type != actual_type:
            errors.append(f"Type mismatch for {name}: expected {expected_type}, got {actual_type}")
            continue

        if expected_type == "class" and actual_type == "class":
            raw_exp_methods = expected.get("methods", [])
            raw_act_methods = actual.get("methods", [])
            expected_methods = set(raw_exp_methods) if isinstance(raw_exp_methods, list) else set()
            actual_methods = set(raw_act_methods) if isinstance(raw_act_methods, list) else set()
            missing_methods = sorted(expected_methods - actual_methods)
            if missing_methods:
                errors.append(f"Class {name} is missing methods: {', '.join(missing_methods)}")

    return errors


def _extract_func_param_specs(
    fn_node: ast.FunctionDef,
    skip_self: bool,
) -> list[tuple[str, str, str | None, str | None]]:
    """Extract parameter ``(name, kind, default, annotation)`` tuples from a function AST node.

    Args:
        fn_node: ``ast.FunctionDef`` node to inspect.
        skip_self: Whether to omit leading ``self`` or ``cls`` parameter.

    Returns:
        list[tuple[str, str, str | None, str | None]]: Extracted parameter specifications.
    """
    args = fn_node.args
    pos_args = list(args.posonlyargs) + list(args.args)
    if skip_self and pos_args and pos_args[0].arg in ("self", "cls"):
        pos_args = pos_args[1:]
    non_def_count = len(pos_args) - len(args.defaults)
    specs: list[tuple[str, str, str | None, str | None]] = []
    for idx, arg_node in enumerate(pos_args):
        kind = "POSITIONAL_ONLY" if idx < len(args.posonlyargs) else "POSITIONAL_OR_KEYWORD"
        default_str = (
            ast.unparse(args.defaults[idx - non_def_count]) if idx >= non_def_count else None
        )
        ann_str = ast.unparse(arg_node.annotation) if arg_node.annotation is not None else None
        specs.append((arg_node.arg, kind, default_str, ann_str))
    if args.vararg is not None:
        var_ann = (
            ast.unparse(args.vararg.annotation) if args.vararg.annotation is not None else None
        )
        specs.append((args.vararg.arg, "VAR_POSITIONAL", None, var_ann))
    for idx, kw_arg in enumerate(args.kwonlyargs):
        kw_def = args.kw_defaults[idx]
        def_str = ast.unparse(kw_def) if kw_def is not None else None
        ann_str = ast.unparse(kw_arg.annotation) if kw_arg.annotation is not None else None
        specs.append((kw_arg.arg, "KEYWORD_ONLY", def_str, ann_str))
    if args.kwarg is not None:
        kw_ann = ast.unparse(args.kwarg.annotation) if args.kwarg.annotation is not None else None
        specs.append((args.kwarg.arg, "VAR_KEYWORD", None, kw_ann))
    return specs


def _extract_node_params_and_doc(
    node: ast.ClassDef | ast.FunctionDef,
) -> tuple[list[tuple[str, str, str | None, str | None]], str, set[str]]:
    """Extract parameter specs, docstring, and documented parameter names from a class or function AST node.

    Args:
        node: ``ast.ClassDef`` or ``ast.FunctionDef`` node.

    Returns:
        tuple[list[tuple[str, str, str | None, str | None]], str, set[str]]:
            Tuple of ``(param_specs, docstring, documented_arg_names)``.
    """
    doc = ast.get_docstring(node) or ""
    if isinstance(node, ast.FunctionDef):
        return _extract_func_param_specs(node, skip_self=False), doc, _extract_documented_args(doc)

    for child in node.body:
        if isinstance(child, ast.FunctionDef) and child.name == "__init__":
            init_doc = ast.get_docstring(child) or ""
            combined_doc = " ".join((doc, init_doc)).strip()
            return (
                _extract_func_param_specs(child, skip_self=True),
                combined_doc,
                _extract_documented_args(init_doc) | _extract_documented_args(doc),
            )

    field_specs: list[tuple[str, str, str | None, str | None]] = []
    documented_fields: set[str] = set(_extract_documented_args(doc))
    for child in node.body:
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            fname = child.target.id
            def_str = ast.unparse(child.value) if child.value is not None else None
            ann_str = ast.unparse(child.annotation)
            field_specs.append((fname, "POSITIONAL_OR_KEYWORD", def_str, ann_str))
            documented_fields.add(fname)
    return field_specs, doc, documented_fields


def collect_ast_definitions(
    source_files: Sequence[str] = DEFAULT_SOURCE_FILES,
) -> dict[str, ast.ClassDef | ast.FunctionDef]:
    """Parse ``source_files`` into a mapping of symbol names and qualified names to AST nodes.

    Args:
        source_files: Sequence of Python file paths to parse. Defaults to ``DEFAULT_SOURCE_FILES``.

    Returns:
        dict[str, ast.ClassDef | ast.FunctionDef]: Mapping of symbol/qualified names to AST definitions.
    """
    definitions: dict[str, ast.ClassDef | ast.FunctionDef] = {}
    for rel_path in source_files:
        if not os.path.exists(rel_path):
            continue
        with open(rel_path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=rel_path)
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                definitions[node.name] = node
                if isinstance(node, ast.ClassDef):
                    for sub in node.body:
                        if isinstance(sub, (ast.ClassDef, ast.FunctionDef)):
                            definitions[f"{node.name}.{sub.name}"] = sub
    return definitions


def verify_v005_compliance(
    definitions: Mapping[str, ast.ClassDef | ast.FunctionDef],
    v005_release_snapshot_path: str | None = None,
) -> list[str]:
    """Verify AST definitions against ``ml-ecosystem-snapshots`` v0.0.5 canonical ops and release JSON.

    Args:
        definitions: Mapping of symbol names to ``ast.ClassDef`` or ``ast.FunctionDef`` nodes.
        v005_release_snapshot_path: Optional path to ``orbax_checkpoint_v0.12.8.json``. Defaults to None.

    Returns:
        list[str]: List of compliance error strings (empty if 100% compliant).
    """
    errors: list[str] = []

    for name, api_path, expected_kind, submod, expected_params, _ in CANONICAL_V005_OP_DEFS:
        rel_key = (
            api_path.replace("orbax.checkpoint.", "")
            .replace("orbax.export.", "")
            .replace("orbax.", "")
        )
        node = definitions.get(rel_key) or definitions.get(name)
        if node is None:
            errors.append(f"Missing v0.0.5 canonical symbol: {api_path} (submodule {submod})")
            continue
        actual_kind = "class" if isinstance(node, ast.ClassDef) else "function"
        if actual_kind != expected_kind:
            errors.append(
                f"Kind mismatch for {api_path}: expected {expected_kind}, got {actual_kind}"
            )
            continue

        param_specs, doc, doc_args = _extract_node_params_and_doc(node)
        if not doc.strip():
            errors.append(f"Missing docstring on {api_path}")
        actual_param_map = {p[0]: p for p in param_specs}
        for p_name, _, _ in expected_params:
            if p_name not in actual_param_map:
                errors.append(f"Missing parameter '{p_name}' on {api_path}")
                continue
            _, _, _, p_ann = actual_param_map[p_name]
            if not p_ann:
                errors.append(f"Missing type annotation for parameter '{p_name}' on {api_path}")
            if p_name not in doc_args:
                errors.append(f"Missing docstring entry for parameter '{p_name}' on {api_path}")

    resolved_v005_json = _resolve_v005_release_snapshot_path(v005_release_snapshot_path)
    if not os.path.exists(resolved_v005_json):
        errors.append(f"Error: v0.0.5 release snapshot not found at {resolved_v005_json}")
        return errors

    with open(resolved_v005_json, encoding="utf-8") as f:
        release_data = json.load(f)

    categories = release_data.get("categories", {})
    array_entries = categories.get("array", []) if isinstance(categories, dict) else []
    for entry in array_entries:
        api_path_val = str(entry.get("api_path", ""))
        sym_name = str(entry.get("name", ""))
        node = definitions.get(sym_name)
        if node is None:
            errors.append(f"Missing v0.12.8 release symbol: {api_path_val}")
            continue
        param_specs, doc, _ = _extract_node_params_and_doc(node)
        if not doc.strip():
            errors.append(f"Missing docstring on v0.12.8 symbol: {api_path_val}")
        actual_names = {p[0] for p in param_specs}
        raw_params = entry.get("params") or []
        for p_item in raw_params:
            req_name = str(p_item.get("name", ""))
            if req_name not in actual_names:
                errors.append(f"Missing v0.12.8 parameter '{req_name}' on {api_path_val}")

    return errors


def check_compliance(
    snapshot_path: str | None = None,
    v005_release_snapshot_path: str | None = None,
    source_files: Sequence[str] = DEFAULT_SOURCE_FILES,
) -> tuple[bool, list[str]]:
    """Run the full API and ``ml-ecosystem-snapshots`` v0.0.5 compliance check.

    Args:
        snapshot_path: Optional override path to a structural snapshot JSON. Defaults to None.
        v005_release_snapshot_path: Optional override path to ``orbax_checkpoint_v0.12.8.json``. Defaults to None.
        source_files: Sequence of source files to parse for AST compliance. Defaults to ``DEFAULT_SOURCE_FILES``.

    Returns:
        tuple[bool, list[str]]: Tuple of ``(is_compliant, error_messages)``.
    """
    import zero_orbax.checkpoint as zcp

    if snapshot_path is not None:
        if not os.path.exists(snapshot_path):
            return False, [f"Error: Snapshot not found at {snapshot_path}"]
        with open(snapshot_path, encoding="utf-8") as f:
            snapshot: Mapping[str, Mapping[str, str | list[str]]] = json.load(f)
    else:
        snapshot = DEFAULT_STRUCTURAL_SNAPSHOT

    current_api = dump_api(zcp)
    errors = verify_snapshot_compliance(current_api, snapshot)
    if snapshot_path is None or v005_release_snapshot_path is not None:
        ast_defs = collect_ast_definitions(source_files)
        errors.extend(
            verify_v005_compliance(
                ast_defs,
                v005_release_snapshot_path=v005_release_snapshot_path,
            )
        )
    return len(errors) == 0, errors


def main(
    snapshot_path: str | None = None,
    v005_release_snapshot_path: str | None = None,
) -> int:
    """Execute the compliance verification CLI.

    Args:
        snapshot_path: Optional override path to ``orbax_api_snapshot.json``. Defaults to None.
        v005_release_snapshot_path: Optional override path to ``orbax_checkpoint_v0.12.8.json``. Defaults to None.

    Returns:
        int: Exit code (0 on 100% compliance, 1 on failure).
    """
    ok, errors = check_compliance(
        snapshot_path=snapshot_path,
        v005_release_snapshot_path=v005_release_snapshot_path,
    )
    if not ok:
        print("Compliance check failed!")
        for error in errors:
            print(f" - {error}")
        return 1

    print("Compliance Check Passed! 100% compliant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
