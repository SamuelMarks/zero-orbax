# Exhaustive Codebase Audit & Conformance Plan (`zero-orbax` -> `ml-ecosystem-snapshots @ v0.0.5`)

This document captures the comprehensive architectural, API surface, module hierarchy, signature, docstring, typing, and pre-commit compliance audit of `zero-orbax` against `ml-ecosystem-snapshots` v0.0.5 (`https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/tag/v0.0.5`), followed by an exhaustive, itemized checkbox execution plan.

---

## Global Invariants & Quality Gates

Every phase and checkbox item in this plan MUST maintain the following repository-wide invariants at all times:

- **100% Documentation Coverage**: Every file, module, class, method, function, and parameter (`Args:`, `Returns:`, `Yields:`, `Raises:`, `Attributes:`) across `src/`, `scripts/`, and `tests/` must be documented (`interrogate -vv -f 100 -M -i -c pyproject.toml src scripts tests` at 100.0%, `ruff check --select D` with Google convention, and `python3 scripts/check_docs.py` passing with zero missing/stub docstrings).
- **100% Test Coverage**: 100% function, line, and branch coverage across `src/zero_orbax` and `scripts` (`pytest --cov=src/zero_orbax --cov=scripts --cov-branch --cov-report=term-missing --cov-fail-under=100`).
- **Strong Typing & Zero Reflection**: Explicit type annotations on every parameter, return type, and attribute across `src/`, `scripts/`, and `tests/`. Zero `Any` or `object` annotations; zero `typing.cast`, `getattr`, `setattr`, `hasattr`, `delattr`, `vars`, `globals`, `locals`, `exec`, `eval`, or dunder reflection (`__dict__`, `__class__`, `__bases__`, `__mro__`, `__subclasses__`, etc.); zero `# type: ignore` comments (`mypy --strict src tests scripts` and `python3 scripts/lint_typing.py` passing with 0 violations).
- **No New External Dependencies**: Only standard library modules and explicitly permitted ecosystem packages (`pydantic`, `ml-switcheroo-ir`, `ml-switcheroo-compiler`, `cdd-python` / `cdd`, `ml-ecosystem-snapshots`, and sibling `../zero-*` packages: `zero-jax`, `zero-chex`, `zero-flax`, `zero-grain`, `zero-optax`) are allowed (`python3 scripts/check_allowed_imports.py` passing).
- **All Pre-Commit Hooks Passing**: `ruff-format`, `ruff-check`, `mypy`, `lint-typing`, `interrogate`, `check-docs`, `check-allowed-imports`, `check-api-compliance`, `pytest-coverage`, and `update-badges` must all pass cleanly.
- **Never `git push`**: All work remains local; never execute `git push`.

---

## Architecture & Audit Overview

```text
+-----------------------------------------------------------------------------------+
|                      ml-ecosystem-snapshots @ v0.0.5                              |
|  1. Release Asset: orbax_checkpoint_v0.12.8.json (31 Griffe-extracted APIs)       |
|  2. Canonical Extractor: CANONICAL_ORBAX_OPS / _ORBAX_OP_DEFS (58 APIs)           |
|     - orbax.checkpoint.* (Managers, Checkpointers, Options, Handlers, Transforms) |
|     - orbax.checkpoint.args.* (16 Declarative Save/Restore CheckpointArgs)        |
|     - orbax.checkpoint.checkpoint_utils.* (Iterator, Wait, RestoreArgs Helpers)   |
|     - orbax.checkpoint.msgpack_utils.* (msgpack_serialize, msgpack_restore)       |
|     - orbax.checkpoint.v1.* (save/load_pytree, save_pytree_async, checkpointables)|
|     - orbax.export.* (ExportManager, JaxModule, ServingConfig, ObmExportOptions)  |
|  3. Legacy Structural Snapshot: tests/orbax_api_snapshot.json (62 symbols)        |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                    scripts/check_compliance.py (Pre-Commit Gate)                  |
|  - Validates module hierarchy: zero_orbax.{checkpoint,export,core,types}          |
|    and submodules: args, checkpoint_utils, msgpack_utils, v1, handlers, options,  |
|    type_handlers, transform_utils, step, step_lib, path, serialization, etc.      |
|  - Validates minimum required classes, fields, methods, and functions exist       |
|  - Validates function/constructor parameters: names, ordering, kinds, annotations,|
|    and default values (allowing additional optional parameters in zero_orbax)     |
|  - Validates non-empty Google-style docstrings and parameter documentation        |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                              src/zero_orbax/                                      |
|  +-- __init__.py   (exports checkpoint, core, export, types)                      |
|  +-- types.py      (Protocols, Options dataclasses, TypeGuards, Callbacks)        |
|  +-- core.py       (VFS, FileSystemIO, AtomicWrite, MsgpackEngine, Sharding)      |
|  +-- checkpoint.py (Handlers, Checkpointers, CheckpointManager, args, v1, utils)  |
|  +-- export.py     (ExportManager, JaxModule, ServingConfig, ObmExportOptions)    |
+-----------------------------------------------------------------------------------+
```

### Summary of Audit Findings

| ID | Area / Component | Current State in `zero-orbax` | Target Conformance State (`v0.0.5`) | Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **A1** | **Dependency & Snapshot Pinning** | `requirements.txt` and `setup_dev_env.sh` reference `ml-ecosystem-snapshots` `v0.0.4`; `orbax_checkpoint_v0.12.8.json` from `v0.0.5` is not bundled/checked. | Upgrade `requirements.txt`, `setup_dev_env.sh`, and `.github/workflows/ci.yml` to `v0.0.5`; bundle/load `v0.0.5` Orbax schema & `CANONICAL_ORBAX_OPS`. | Low |
| **A2** | **Cross-Repo Circular Import (`zero_chex` <-> `zero_grain` <-> `zero_orbax`)** | Top-level `import zero_chex as _chex` in `checkpoint.py` before `AsyncCheckpointHandler` is defined triggers circular `ImportError` when `zero_chex` imports `zero_grain` which imports `zero_orbax.checkpoint`. | Reorder imports or defer `zero_chex` / `zero_flax` / `zero_grain` imports after core handler definitions (or into helper functions) while keeping `check_allowed_imports.py` and `lint_typing.py` 100% clean. | Medium |
| **A3** | **Missing Submodules & Hierarchies (`v1`, `export`)** | `zero_orbax.checkpoint.v1` and `zero_orbax.export` (`orbax.export`) do not exist. | Implement `v1` namespace (`save_pytree`, `save_pytree_async`, `load_pytree`, `save_checkpointables`, `load_checkpointables`) and `zero_orbax.export` (`ExportManager`, `JaxModule`, `ServingConfig`, `ObmExportOptions`). | High |
| **A4** | **Missing Classes & Functions in `checkpoint`** | Missing `MultiprocessingOptions` top-level export in `checkpoint.__all__`, `RandomKeyCheckpointHandler`, `SingleReplicaArrayRestoreArgs`, `create_default_handler_registry`, `checkpoint_utils.{construct_restore_args, restore_args_from_target, wait_for_new_checkpoint}`, and `msgpack_utils.{msgpack_serialize, msgpack_restore}`. | Implement and export all missing classes and functions with full semantic behavior, strict typing, and 100% test/doc coverage. | High |
| **A5** | **Constructor & Function Signature / Field / Default Alignment** | Several classes (`Checkpointer`, `AsyncCheckpointer`, `PyTreeCheckpointer`, `StandardCheckpointer`, `CheckpointManagerOptions`, `AsyncOptions`, `PyTreeCheckpointHandler`, `StandardCheckpointHandler`, `ArrayCheckpointHandler`, `ProtoCheckpointHandler`, `SaveArgs`, `RestoreArgs`, `ArrayRestoreArgs`, `RestoreTransform`, `args.StandardRestore`) have missing parameters, keyword-only vs positional-or-keyword differences, or field ordering differences relative to `v0.0.5`. | Align all parameter names, ordering, kinds, type annotations, and default values so both `v0.0.5` canonical specs and existing official tests work seamlessly. | High |
| **A6** | **Pre-Commit Hook & AST Compliance Validator (`scripts/check_compliance.py`)** | `scripts/check_compliance.py` only checks symbol names and method names against `tests/orbax_api_snapshot.json` using `inspect.getmembers` (which is forbidden by `lint_typing.py` if `getattr` is used, and misses module hierarchy, class fields, function args, types, defaults, and docstrings). | Upgrade `scripts/check_compliance.py` and `.pre-commit-config.yaml` to AST-validate module hierarchy, classes, fields, methods, functions, parameter names/kinds/types/defaults, and docstrings against `ml-ecosystem-snapshots @ v0.0.5` + `tests/orbax_api_snapshot.json`. | High |
| **A7** | **Exhaustive Test & Docstring Coverage Verification** | Existing tests do not cover the new `v0.0.5` symbols (`v1`, `export`, `RandomKeyCheckpointHandler`, `SingleReplicaArrayRestoreArgs`, `checkpoint_utils` helpers, `msgpack_utils` helpers, upgraded `check_compliance.py`). | Add comprehensive unit and integration tests achieving 100% line/branch/function coverage and 100% docstring coverage across all files. | Medium |

---

## Phase 1: Environment, Dependency Upgrade & Circular Import Resolution

### 1.1 Upgrade `ml-ecosystem-snapshots` to `v0.0.5`
- [x] Update `requirements.txt` to pin `ml-ecosystem-snapshots @ https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/download/v0.0.5/ml_ecosystem_snapshots-0.0.5-py3-none-any.whl`.
- [x] Update `setup_dev_env.sh` to install `https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/download/v0.0.5/ml_ecosystem_snapshots-0.0.5-py3-none-any.whl || pip install -e ../ml-ecosystem-snapshots`.
- [x] Check `.github/workflows/ci.yml` and update any `v0.0.4` references or snapshot setup steps to `v0.0.5`.
- [x] Store or reference the official `v0.0.5` snapshot payload (`tests/orbax_checkpoint_v0.12.8.json` downloaded from `https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/download/v0.0.5/orbax_checkpoint_v0.12.8.json`) alongside `tests/orbax_api_snapshot.json` so offline pre-commit hooks can deterministically validate both the `v0.0.5` release dump and the canonical `v0.0.5` operator definitions.

### 1.2 Resolve Circular Import Between `zero_orbax`, `zero_chex`, and `zero_grain`
- [x] Audit import order in `src/zero_orbax/checkpoint.py`:
  - `src/zero_orbax/__init__.py` imports `checkpoint, core, types`.
  - `src/zero_orbax/checkpoint.py` imports `zero_chex as _chex` at line 29, before `AsyncCheckpointHandler`, `PyGrainCheckpointHandler`, and `args` are defined.
  - `zero_chex` imports `zero_grain.shared_memory_array`, which triggers `zero_grain/__init__.py` -> `zero_grain/python/__init__.py` -> `zero_grain/checkpoint.py` -> `from zero_orbax.checkpoint import AsyncCheckpointHandler, PyGrainCheckpointHandler, args`.
- [x] Fix the circular import cleanly without reflection or lazy `getattr`:
  - Move `from zero_orbax import core, types, checkpoint` order in `src/zero_orbax/__init__.py` so `types` and `core` load first.
  - In `src/zero_orbax/checkpoint.py`, place `import zero_chex as _chex` after `AsyncCheckpointHandler`, `PyGrainCheckpointHandler`, and `args` are defined (or import `zero_chex._src.asserts as _chex` / import inside validation helpers), ensuring `zero_grain.checkpoint` can import `AsyncCheckpointHandler`, `PyGrainCheckpointHandler`, and `args` from `zero_orbax.checkpoint` without partial-initialization `ImportError`.
- [x] Verify `pytest` collects and runs all existing tests without `ImportError`.

---

## Phase 2: Module Hierarchy & Missing Submodules (`v1`, `export`, `checkpoint_utils`, `msgpack_utils`)

### 2.1 Implement `zero_orbax.export` (`orbax.export` Parity)
- [x] Create `src/zero_orbax/export.py` with module docstring and `__all__ = ["ExportManager", "JaxModule", "ObmExportOptions", "ServingConfig"]`.
- [x] Export `export` from `src/zero_orbax/__init__.py` (`__all__ = ["checkpoint", "core", "export", "types"]`) and expose `export` under `zero_orbax.checkpoint.orbax.export` and `zero_orbax.checkpoint.export` for full namespace hierarchy parity.
- [x] Implement `ObmExportOptions` dataclass in `src/zero_orbax/export.py`:
  - [x] Fields and defaults matching `v0.0.5`:
    - `enable_auto_layout: bool = False`
    - `checkpoint_path: str | None = None`
  - [x] Full Google-style class and attribute docstrings.
- [x] Implement `ServingConfig` dataclass in `src/zero_orbax/export.py`:
  - [x] Fields and defaults matching `v0.0.5`:
    - `signature_key: str`
    - `input_signature: Sequence[PyTree] | PyTree = None`
    - `tf_preprocessor: Callable[[PyTree], PyTree] | None = None`
    - `tf_postprocessor: Callable[[PyTree], PyTree] | None = None`
    - `extra_trackable_resources: PyTree = None`
    - `method_key: str | None = None`
    - `obm_export_options: ObmExportOptions | None = None`
  - [x] Implement helper methods (`get_signature_keys() -> Sequence[str]`, `get_infer_step(infer_fn: ...) -> Callable[[PyTree], PyTree]`) with full type annotations and docstrings.
- [x] Implement `JaxModule` class in `src/zero_orbax/export.py`:
  - [x] Constructor `__init__` signature matching `v0.0.5`:
    - `params: PyTree`
    - `apply_fn: Callable[[PyTree, PyTree], PyTree] | Mapping[str, Callable[[PyTree, PyTree], PyTree]]`
    - `trainable: bool | PyTree = False`
    - `input_polymorphic_shape: PyTree = None`
    - `jax2tf_kwargs: Mapping[str, PyTree] | None = None`
    - `jit_compile: bool | Mapping[str, bool] = True`
    - `pspecs: PyTree = None`
    - `allow_multi_axis_sharding_consolidation: bool | None = None`
    - `export_version: str = "TF_SAVEDMODEL"`
  - [x] Implement properties and methods:
    - `apply_fn_map -> Mapping[str, Callable[[PyTree, PyTree], PyTree]]`
    - `model_params -> PyTree`
    - `methods -> Mapping[str, Callable[[PyTree], PyTree]]`
    - `jax_methods -> Mapping[str, Callable[[PyTree], PyTree]]`
    - `update_variables(params: PyTree) -> None`
- [x] Implement `ExportManager` class in `src/zero_orbax/export.py`:
  - [x] Constructor `__init__` signature matching `v0.0.5`:
    - `module: JaxModule`
    - `serving_configs: Sequence[ServingConfig]`
  - [x] Implement methods:
    - `tf_module -> JaxModule` property
    - `serving_signatures -> Mapping[str, Callable[[PyTree], PyTree]]` property
    - `save(model_path: PathLike, save_options: PyTree = None, signature_overrides: Mapping[str, Callable[[PyTree], PyTree]] | None = None) -> None` (persisting serialized module parameters and serving config metadata via `AtomicWrite` / `MsgpackEngine` and `FileSystemIO`)
    - `load(model_path: PathLike, **kwargs: PyTree) -> PyTree` (restoring the persisted artifact from `model_path`)

### 2.2 Implement `zero_orbax.checkpoint.v1` (`orbax.checkpoint.v1` Functional APIs)
- [x] Implement the `v1` namespace class/submodule in `src/zero_orbax/checkpoint.py` and include `"v1"` in `zero_orbax.checkpoint.__all__` and `NAMESPACE_EMULATING_CLASSES`.
- [x] Implement `v1.save_pytree`:
  - [x] Signature matching `v0.0.5`: `save_pytree(path: PathLike, pytree: PyTree, *, overwrite: bool = False, custom_metadata: Mapping[str, JsonValue] | None = None) -> None`
  - [x] Semantics: Check if `path` already exists when `overwrite=False` (raise `ValueError` if it exists); if `overwrite=True` and `path` exists, remove existing checkpoint first; save `pytree` via `PyTreeCheckpointer` / `PyTreeCheckpointHandler` and persist `custom_metadata` if provided.
- [x] Implement `v1.save_pytree_async`:
  - [x] Signature matching `v0.0.5`: `save_pytree_async(path: PathLike, pytree: PyTree, *, overwrite: bool = False, custom_metadata: Mapping[str, JsonValue] | None = None) -> Future`
  - [x] Semantics: Validate `overwrite`, initiate an asynchronous save via `AsyncCheckpointer(PyTreeCheckpointHandler())`, write `custom_metadata` on commit, and return the `Future` instance.
- [x] Implement `v1.load_pytree`:
  - [x] Signature matching `v0.0.5`: `load_pytree(path: PathLike, abstract_pytree: PyTree = None) -> PyTree`
  - [x] Semantics: Restore and unpack the PyTree from `path` using `PyTreeCheckpointer().restore(path, item=abstract_pytree)`.
- [x] Implement `v1.save_checkpointables`:
  - [x] Signature matching `v0.0.5`: `save_checkpointables(path: PathLike, checkpointables: Mapping[str, PyTree], *, overwrite: bool = False, custom_metadata: Mapping[str, JsonValue] | None = None) -> None`
  - [x] Semantics: Validate `overwrite`, save named dictionary of checkpointables via `CompositeCheckpointHandler`, and persist `custom_metadata` if provided.
- [x] Implement `v1.load_checkpointables`:
  - [x] Signature matching `v0.0.5`: `load_checkpointables(path: PathLike, abstract_checkpointables: Mapping[str, PyTree] | None = None) -> dict[str, PyTree]`
  - [x] Semantics: Restore named composite checkpointables from `path` using `CompositeCheckpointHandler().restore(path, item=abstract_checkpointables)`.
- [x] Export `save_pytree`, `save_pytree_async`, `load_pytree`, `save_checkpointables`, and `load_checkpointables` both on `zero_orbax.checkpoint.v1` and at top-level `zero_orbax.checkpoint` if needed.

### 2.3 Expand `zero_orbax.checkpoint.checkpoint_utils`
- [x] Implement `checkpoint_utils.restore_args_from_target`:
  - [x] Signature matching `v0.0.5`: `restore_args_from_target(target: PyTree, sharding_tree: PyTree = None) -> PyTree`
  - [x] Semantics: Recursively traverse `target` (and matching `sharding_tree`) to construct `ArrayRestoreArgs` (for array leaves with shape/dtype/sharding) or `RestoreArgs(restore_type=type(leaf), dtype=...)` for scalar leaves.
- [x] Implement `checkpoint_utils.construct_restore_args`:
  - [x] Signature matching `v0.0.5`: `construct_restore_args(target: PyTree, sharding_tree: PyTree = None, set_global_shape: bool = True) -> PyTree`
  - [x] Semantics: Recursively construct `ArrayRestoreArgs` / `RestoreArgs` from `target` (including `FlaxStateProtocol`, `FrozenDictProtocol`, `NamedTupleProtocol`, `ArrayProtocol`, `SerializedArray`, dicts, lists, tuples, and scalars), setting `global_shape` and `shape` when `set_global_shape=True`.
- [x] Implement `checkpoint_utils.wait_for_new_checkpoint`:
  - [x] Signature matching `v0.0.5`: `wait_for_new_checkpoint(checkpoint_dir: PathLike, *, until_step: int | None = None, seconds_to_sleep: int = 1, timeout: int | None = None, step_prefix: str | None = None, step_format_fixed_length: int | None = None, step_name_format: step_lib.NameFormat | None = None) -> int`
  - [x] Semantics: Poll `checkpoint_dir` using `step_name_format` (or `standard_name_format(step_prefix=step_prefix, step_format_fixed_length=step_format_fixed_length)`) until a step `>= until_step` (or any step if `until_step is None`) is found, returning the discovered latest step integer (or `-1` if timeout expires without finding a matching step).
- [x] Ensure `checkpoint_utils.checkpoints_iterator` remains intact and tested.

### 2.4 Expand `zero_orbax.checkpoint.msgpack_utils`
- [x] Implement `msgpack_utils.msgpack_serialize`:
  - [x] Signature matching `v0.0.5`: `msgpack_serialize(pytree: PyTree, in_place: bool = False) -> bytes`
  - [x] Semantics: Serialize `pytree` into RFC 8949 / MessagePack bytes via `MsgpackEngine().packb(pytree)`.
- [x] Implement `msgpack_utils.msgpack_restore`:
  - [x] Signature matching `v0.0.5`: `msgpack_restore(encoded_pytree: bytes) -> PyTree`
  - [x] Semantics: Deserialize `encoded_pytree` bytes back into a Python `PyTree` via `MsgpackEngine().unpackb(encoded_pytree)`.
- [x] Keep `msgpack_utils.packb` and `msgpack_utils.unpackb` for backward compatibility.

---

## Phase 3: Classes, Fields, Signatures & Defaults Conformance (`orbax.checkpoint`)

### 3.1 Missing Classes & Factory Functions in `zero_orbax.checkpoint`
- [x] Export `MultiprocessingOptions` in `zero_orbax.checkpoint.__all__` (currently imported from `zero_orbax.types` and exposed on `options.MultiprocessingOptions`, but missing from `checkpoint.__all__`).
- [x] Implement `RandomKeyCheckpointHandler(AsyncCheckpointHandler)`:
  - [x] Constructor signature matching `v0.0.5`: `__init__(self, key_name: str | None = None) -> None`
  - [x] Semantics: Save and restore both JAX and NumPy random keys under `key_name or "random_key"`.
  - [x] Expose on `zero_orbax.checkpoint`, `zero_orbax.checkpoint.handlers`, and `zero_orbax.checkpoint.__all__`.
- [x] Implement `SingleReplicaArrayRestoreArgs(ArrayRestoreArgs)`:
  - [x] Fields/constructor signature matching `v0.0.5`:
    - `restore_type: type | str | None = None`
    - `mesh: LogicalMesh | None = None`
    - `mesh_axes: PartitionSpec | Sequence[str | None] | None = None`
    - `sharding: PartitionSpec | str | None = None`
    - `global_shape: tuple[int, ...] | None = None`
    - `shape: tuple[int, ...] | None = None`
    - `dtype: str | DType | type | None = None`
    - `strict: bool = True`
    - `single_replica_sharding: PartitionSpec | str | None = None`
  - [x] Expose on `zero_orbax.checkpoint`, `zero_orbax.checkpoint.type_handlers`, and `zero_orbax.checkpoint.__all__`.
- [x] Implement `create_default_handler_registry`:
  - [x] Signature matching `v0.0.5`: `create_default_handler_registry(**items_and_handlers: CheckpointHandler | None) -> CheckpointHandlerRegistry`
  - [x] Semantics: Create a `DefaultCheckpointHandlerRegistry` pre-populated with any passed `items_and_handlers` mappings and default standard handlers.
  - [x] Expose on `zero_orbax.checkpoint`, `zero_orbax.checkpoint.handlers`, and `zero_orbax.checkpoint.__all__`.

### 3.2 Align Constructor & Function Signatures, Types, and Defaults with `v0.0.5`
- [x] **`AbstractCheckpointManager` & `AbstractCheckpointer`**:
  - [x] Update `AbstractCheckpointManager.__init__(self, *args: PyTree, **kwargs: PyTree) -> None` and `AbstractCheckpointer.__init__(self, *args: PyTree, **kwargs: PyTree) -> None` so they satisfy `orbax_checkpoint_v0.12.8.json` (`*args`, `**kwargs`) while remaining 100% compatible with zero-arg `super().__init__()` calls.
- [x] **`Checkpointer`**:
  - [x] Update `Checkpointer.__init__` to support all `v0.0.5` canonical parameters as well as existing options:
    - `handler: CheckpointHandler | None = None`
    - `*,`
    - `primary_host: int | None = 0`
    - `active_processes: set[int] | None = None`
    - `barrier_sync_fn: BarrierSyncFn | None = None`
    - `path_permission_mode: int | None = None`
    - `checkpoint_metadata_store: CheckpointMetadataStoreProtocol | None = None`
    - `temporary_path_cls: type[TemporaryPathProtocol] | None = None`
    - `multiprocessing_options: MultiprocessingOptions | None = None`
    - `file_options: FileOptions | None = None`
    - `atomicity_options: AtomicityOptions | None = None`
    - `temporary_path_class: type[TemporaryPathProtocol] | None = None`
  - [x] Wire `primary_host`, `active_processes`, `path_permission_mode`, and `temporary_path_cls` into `self.multiprocessing_options`, `self.file_options`, and `self.temporary_path_class` when provided.
- [x] **`AsyncCheckpointer`**:
  - [x] Update `AsyncCheckpointer.__init__` so the first positional parameter is named `handler: CheckpointHandler | None = None` and the second positional-or-keyword parameter is `timeout_secs: int = 300`, followed by keyword-only arguments matching `v0.0.5`:
    - `handler: CheckpointHandler | None = None`
    - `timeout_secs: int = 300`
    - `*,`
    - `primary_host: int | None = 0`
    - `active_processes: set[int] | None = None`
    - `barrier_sync_fn: BarrierSyncFn | None = None`
    - `post_finalization_callback: Callable[[], None] | None = None`
    - `path_permission_mode: int | None = None`
    - `async_options: AsyncOptions | None = None`
    - `multiprocessing_options: MultiprocessingOptions | None = None`
    - `temporary_path_cls: type[TemporaryPathProtocol] | None = None`
    - `file_options: FileOptions | None = None`
    - `checkpoint_metadata_store: CheckpointMetadataStoreProtocol | None = None`
    - `atomicity_options: AtomicityOptions | None = None`
    - `temporary_path_class: type[TemporaryPathProtocol] | None = None`
  - [x] Ensure `barrier_sync_fn` and `post_finalization_callback` passed directly to `AsyncCheckpointer` are invoked during `save()` commit and `wait_until_finished()` even when `async_options` is `None`.
- [x] **`PyTreeCheckpointer`**:
  - [x] Update `PyTreeCheckpointer.__init__` signature to support positional or keyword `primary_host: int | None = 0`, `use_ocdbt: bool = True`, `use_zarr3: bool = False`, `use_compression: bool = True`.
- [x] **`StandardCheckpointer`**:
  - [x] Update `StandardCheckpointer.__init__` to accept `primary_host: int | None = 0` as the first positional-or-keyword parameter (defaulting to `0`), followed by keyword-only `async_options`, `multiprocessing_options`, `file_options`, `atomicity_options`, `checkpoint_metadata_store`, `temporary_path_class`, `save_concurrent_gb`, `restore_concurrent_gb`, `pytree_metadata_options`, `use_ocdbt`, and `**kwargs: PyTree`.
- [x] **`AsyncOptions`**:
  - [x] Add `async_origin: str | None = None` field to `AsyncOptions` in `src/zero_orbax/types.py` (after `create_directories_asynchronously: bool = True`), and update `timeout_secs: int = 600` (matching `v0.0.5` canonical default `'600'`).
- [x] **`CheckpointManagerOptions`**:
  - [x] Ensure `should_save_fn: ShouldSaveFn | None = None` is ordered at position 5 (immediately after `keep_period`, before `best_fn`) to match `v0.0.5` canonical parameter ordering (`save_interval_steps`, `max_to_keep`, `keep_time_interval`, `keep_period`, `should_save_fn`, `best_fn`, `best_mode`, `keep_checkpoints_without_metrics`, `step_prefix`, `step_format_fixed_length`, `step_name_format`, `create`, `cleanup_tmp_directories`, `save_on_steps`, `single_host_load_and_broadcast`, `todelete_subdir`, `enable_background_delete`, `read_only`, `enable_async_checkpointing`, `async_options`, `multiprocessing_options`, `save_decision_policy`, `preservation_policy`, plus additional optional fields).
- [x] **`PyTreeCheckpointHandler`**:
  - [x] Update `PyTreeCheckpointHandler.__init__` to accept `primary_host: int | None = 0` and allow `use_ocdbt`, `use_zarr3`, `primary_host`, `type_handler_registry`, `handler_impl`, `save_concurrent_gb`, `restore_concurrent_gb` to be passed either positionally or by keyword while preserving all 15 `v0.12.8` parameters (`save_device_host_concurrent_gb`, `memory_limit_options`, `use_compression`, `multiprocessing_options`, `pytree_metadata_options`, `array_metadata_validator`, `enable_pinned_host_transfer`, `is_prioritized_key_fn`).
- [x] **`StandardCheckpointHandler`**:
  - [x] Allow `save_concurrent_gb: int | None = 96` and `restore_concurrent_gb: int | None = 96` to be passed either positionally or by keyword, followed by keyword-only `multiprocessing_options`, `pytree_metadata_options`, and `use_ocdbt`.
- [x] **`ArrayCheckpointHandler`**:
  - [x] Add `*, multiprocessing_options: MultiprocessingOptions | None = None` to `ArrayCheckpointHandler.__init__(self, checkpoint_name: str | None = None, *, multiprocessing_options: MultiprocessingOptions | None = None) -> None` and store `self.multiprocessing_options = multiprocessing_options`.
- [x] **`SaveArgs`**:
  - [x] Add `ocdbt_target_data_file_size: int | None = None` to `SaveArgs`:
    - `aggregate: bool = False`
    - `dtype: str | DType | type | None = None`
    - `chunk_byte_size: int | None = None`
    - `ocdbt_target_data_file_size: int | None = None`
    - `shard_axes: tuple[int, ...] = ()`
- [x] **`RestoreArgs`**:
  - [x] Add `strict: bool = True` to `RestoreArgs`:
    - `restore_type: type | str | None = None`
    - `dtype: str | DType | type | None = None`
    - `strict: bool = True`
- [x] **`ArrayRestoreArgs`**:
  - [x] Update parameter/field ordering of `ArrayRestoreArgs` to match `v0.0.5` (`restore_type`, `mesh`, `mesh_axes`, `sharding`, `global_shape`, `shape`, `dtype`, `strict`).
- [x] **`RestoreTransform`**:
  - [x] Align `RestoreTransform.__init__` parameter order with `Transform` and `v0.0.5`:
    - `original_key: str | Sequence[str] | None = None`
    - `use_fallback: bool = False`
    - `value_fn: ValueTransformFn | None = None`
    - `multi_value_fn: MultiValueTransformFn | None = None`
    - `multi_value_fn_input_args: Mapping[str, RestoreArgs] | None = None`
- [x] **`args.StandardRestore`**:
  - [x] Add `strict: bool = True` parameter to `args.StandardRestore.__init__(self, item: PyTree = None, restore_args: PyTree = None, strict: bool = True) -> None` (or `strict: bool = True` alongside `restore_args`) and store `self.strict = strict`.

---

## Phase 4: Pre-Commit Hooks & AST Compliance Verification (`scripts/check_compliance.py`)

### 4.1 Upgrade `scripts/check_compliance.py` for `ml-ecosystem-snapshots @ v0.0.5`
- [x] Refactor `scripts/check_compliance.py` so it performs a multi-layered compliance audit without any reflection (`getattr`, `hasattr`, `__dict__`):
  - [x] **Layer 1: Legacy Structural Snapshot Parity (`tests/orbax_api_snapshot.json`)**:
    - Retain `dump_api` and `verify_snapshot_compliance` so all 62 top-level symbols, classes, methods, and namespaces in `tests/orbax_api_snapshot.json` continue to be verified.
  - [x] **Layer 2: Official `v0.0.5` Release Asset (`tests/orbax_checkpoint_v0.12.8.json`)**:
    - Verify all 31 classes and functions in `orbax_checkpoint_v0.12.8.json` exist in `zero_orbax.checkpoint`.
    - Verify via AST / `inspect.signature` that every required parameter name in `orbax_checkpoint_v0.12.8.json` is accepted by the implementation (allowing `zero_orbax` to define additional optional parameters beyond the minimum required).
    - Verify that every class and function has a non-empty docstring and that all parameters are documented in `Args:` / `Attributes:`.
  - [x] **Layer 3: Official `v0.0.5` Canonical API & Module Hierarchy (`CANONICAL_ORBAX_OPS` / `_ORBAX_OP_DEFS` from `ml_ecosystem_snapshots.frameworks.orbax_checkpoint`)**:
    - Verify module hierarchy existence:
      - `zero_orbax.checkpoint` (`orbax.checkpoint`)
      - `zero_orbax.checkpoint.args` (`orbax.checkpoint.args`)
      - `zero_orbax.checkpoint.checkpoint_utils` (`orbax.checkpoint.checkpoint_utils`)
      - `zero_orbax.checkpoint.msgpack_utils` (`orbax.checkpoint.msgpack_utils`)
      - `zero_orbax.checkpoint.v1` (`orbax.checkpoint.v1`)
      - `zero_orbax.export` (`orbax.export`)
      - All 28 sub-namespaces (`handlers`, `options`, `type_handlers`, `transform_utils`, `step`, `step_lib`, `path`, `serialization`, etc.)
    - Verify all 58 canonical operations (`class`es and `function`s) exist at their canonical hierarchical paths.
    - Verify that every canonical parameter (`name`, `kind`, `default` presence, and explicit type annotation) is present on the corresponding class constructor / dataclass fields / function signature (while permitting additional optional parameters in `zero_orbax`).
    - Verify docstring presence and parameter documentation on every canonical class and function.
- [x] Update `scripts/generate_snapshot.py` if needed to stay synchronized with `scripts/check_compliance.py`.

### 4.2 Update `.pre-commit-config.yaml` and `scripts/check_allowed_imports.py`
- [x] Update `.pre-commit-config.yaml`:
  - [x] Update `check-allowed-imports` entry to include `src/zero_orbax/export.py`:
    `python3 scripts/check_allowed_imports.py src/zero_orbax/__init__.py src/zero_orbax/types.py src/zero_orbax/core.py src/zero_orbax/checkpoint.py src/zero_orbax/export.py`
  - [x] Ensure `check-api-compliance` (`python3 scripts/check_compliance.py`) validates the `ml-ecosystem-snapshots @ v0.0.5` schema and module hierarchy.
- [x] Update `.github/workflows/ci.yml` if `check_allowed_imports.py` is invoked with explicit file paths there.

---

## Phase 5: Exhaustive Unit & Integration Testing (100% Line, Branch & Function Coverage)

### 5.1 Unit Tests for New `v0.0.5` APIs
- [x] Expand `tests/test_missing_symbols.py` and `tests/test_missing_handlers.py`:
  - [x] Test `RandomKeyCheckpointHandler` save, restore, `checkpoint_save_args`, `checkpoint_restore_args`, and `post_restore` (both with `directory` and with `directory=None`).
  - [x] Test `SingleReplicaArrayRestoreArgs` construction, field defaults, `single_replica_sharding`, and `restore_type()`.
  - [x] Test `create_default_handler_registry` with and without custom keyword handlers.
  - [x] Test `checkpoint_utils.restore_args_from_target` and `checkpoint_utils.construct_restore_args` across nested PyTrees, `ArrayProtocol` leaves, `SerializedArray` leaves, `FlaxState`, `FrozenDict`, `NamedTuple`, lists, tuples, and scalars (with `set_global_shape=True` and `False`, and with/without `sharding_tree`).
  - [x] Test `checkpoint_utils.wait_for_new_checkpoint` across existing checkpoints, `until_step` filtering, empty directories with `timeout=0`, and custom `step_name_format` / `step_prefix`.
  - [x] Test `msgpack_utils.msgpack_serialize` and `msgpack_utils.msgpack_restore` round-trips.
  - [x] Test `v1.save_pytree`, `v1.save_pytree_async`, `v1.load_pytree`, `v1.save_checkpointables`, and `v1.load_checkpointables` (including `overwrite=False` raising `ValueError` on existing path, `overwrite=True` replacing existing checkpoint, and `custom_metadata` persistence).
  - [x] Test `zero_orbax.export` (`ObmExportOptions`, `ServingConfig`, `JaxModule`, `ExportManager`) covering single `apply_fn` and mapping `apply_fn`, `trainable` flags, `update_variables`, `get_signature_keys`, `get_infer_step` (with and without `tf_preprocessor` / `tf_postprocessor`), `ExportManager.save`, `ExportManager.load`, `tf_module`, and `serving_signatures`.
  - [x] Test updated `Checkpointer`, `AsyncCheckpointer`, `StandardCheckpointer`, `PyTreeCheckpointHandler`, `ArrayCheckpointHandler`, `SaveArgs`, `RestoreArgs`, `ArrayRestoreArgs`, `RestoreTransform`, and `args.StandardRestore` parameter combinations.

### 5.2 Unit Tests for Updated Compliance Script (`tests/test_scripts.py`)
- [x] Update `tests/test_scripts.py` to test all branches of the upgraded `scripts/check_compliance.py`:
  - [x] Passing compliance check against both `tests/orbax_api_snapshot.json` and `tests/orbax_checkpoint_v0.12.8.json` / `v0.0.5` canonical definitions.
  - [x] Failure branches for missing module/submodule, missing class/function, missing required parameter, missing docstring, and missing snapshot file.
  - [x] Verify `test_requirements_txt_contains_griffe` also asserts `ml-ecosystem-snapshots` `v0.0.5` is pinned in `requirements.txt`.

---

## Phase 6: Final Quality Gates & Pre-Commit Verification

- [x] Run `ruff format --check .` and `ruff check .` (0 errors, including `D` Google docstring rules).
- [x] Run `mypy --strict src tests scripts` (0 errors across all files).
- [x] Run `python3 scripts/lint_typing.py` (0 typing or reflection violations).
- [x] Run `interrogate -vv -f 100 -M -i -c pyproject.toml src scripts tests` (100.0% docstring coverage).
- [x] Run `python3 scripts/check_docs.py` (100.0% granular docstring & argument coverage, 0 missing/stub items).
- [x] Run `python3 scripts/check_allowed_imports.py src/zero_orbax/__init__.py src/zero_orbax/types.py src/zero_orbax/core.py src/zero_orbax/checkpoint.py src/zero_orbax/export.py` (0 disallowed imports).
- [x] Run `python3 scripts/check_compliance.py` (100% API & `v0.0.5` snapshot compliance).
- [x] Run `pytest --cov=src/zero_orbax --cov=scripts --cov-branch --cov-report=term-missing --cov-fail-under=100` (100.0% line, branch, and function coverage).
- [x] Run `python3 scripts/update_badges.py` and verify `README.md` badges are 100% up to date.
- [x] Confirm `git status` and `git diff` are clean and never run `git push`.
