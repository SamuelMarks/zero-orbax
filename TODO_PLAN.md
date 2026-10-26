# `zero-orbax` Exhaustive Audit & Refactoring Plan (`TODO_PLAN.md`)

This document provides an exhaustive, file-by-file, class-by-class, and test-by-test engineering plan to eliminate all code smells, architectural anti-patterns, silent no-op options, typing/dunder loopholes, and tautological tests across `zero-orbax`.

All refactoring phases **must strictly preserve**:
- **100% Documentation Coverage**: Every module, file, class, function, method, and argument (including `Args:`, `Returns:` / `Yields:`, `Raises:`, and `Attributes:`) verified by `scripts/check_docs.py` and `interrogate`.
- **100% Test Coverage**: 100% function, line, and branch coverage verified by `pytest --cov=src/zero_orbax --cov=scripts --cov=tests --cov-branch --cov-fail-under=100`.
- **Strong Static Typing**: Explicit type annotations everywhere; zero `Any`, `object`, `typing.cast`, `getattr`, `setattr`, `hasattr`, `delattr`, `vars`, `globals`, `locals`, or dunder reflection tricks (`__dict__`, `__class__`, `__module__`, etc.), verified by `mypy --strict` and `scripts/lint_typing.py`.
- **Zero Disallowed Dependencies**: Only Python standard library, `pydantic`, `ml-switcheroo-ir`, `ml-switcheroo-compiler`, `cdd-python`, `ml-ecosystem-snapshots`, and sibling `../zero-*` packages (`zero-jax`, `zero-flax`, `zero-optax`, `zero-grain`, `zero-chex`), verified by `scripts/check_allowed_imports.py`.
- **100% Snapshot & Pre-Commit Compliance**: All hooks in `.pre-commit-config.yaml` and `scripts/check_compliance.py` passing.

---

## Phase 1: Decompose Monolithic `src/zero_orbax/checkpoint.py` (5,554 Lines) & Fix Submodule Abstractions

`src/zero_orbax/checkpoint.py` was previously a 5,554-line monolithic module. It has been decomposed into clean domain submodules under `src/zero_orbax/_checkpoint/` and re-exported cleanly in `src/zero_orbax/checkpoint.py`.

### 1.1 Package Conversion & Submodule Decomposition
- [x] Convert `src/zero_orbax/checkpoint.py` into a modular architecture backed by `src/zero_orbax/_checkpoint/` so that `orbax.checkpoint.*` namespaces are cleanly factored units.
- [x] Extract `src/zero_orbax/_checkpoint/options_lib.py` containing `AsyncOptions`, `MultiprocessingOptions`, `FileOptions`, `AtomicityOptions`, `MemoryLimitOptions`, `PyTreeMetadataOptions`, and `CompositeOptions`.
- [x] Extract `src/zero_orbax/_checkpoint/step_lib.py` (aliased as `step` and `step_lib`) containing `Metadata`, `NameFormat`, `StandardNameFormat`, and `standard_name_format`.
- [x] Extract `src/zero_orbax/_checkpoint/args_lib.py` (and `checkpoint_args`) containing `CheckpointArgs`, `ArraySave`, `ArrayRestore`, `Composite`, `JsonSave`, `JsonRestore`, `ProtoSave`, `ProtoRestore`, `PyTreeSave`, `PyTreeRestore`, `StandardSave`, `StandardRestore`, `JaxRandomKeySave`, `JaxRandomKeyRestore`, `NumpyRandomKeySave`, `NumpyRandomKeyRestore`, `PyGrainCheckpointSave`, `PyGrainCheckpointRestore`, `_ArgsHandlerCatalog`, `register_with_handler`, `get_registered_handler_cls`, `get_registered_args_cls`, and `has_registered_args`.
- [x] Extract `src/zero_orbax/_checkpoint/type_handlers_lib.py` containing `SaveArgs`, `RestoreArgs`, `ArrayRestoreArgs`, `SingleReplicaArrayRestoreArgs`, `TypeHandler`, `TypeHandlerRegistry`, and default leaf type handlers.
- [x] Extract `src/zero_orbax/_checkpoint/future_lib.py` containing `Future` with asynchronous / background execution semantics and exception propagation.
- [x] Extract `src/zero_orbax/_checkpoint/handlers_lib.py` (and `checkpoint_handler`, `async_checkpoint_handler`) containing `CheckpointHandler`, `AsyncCheckpointHandler`, `ArrayCheckpointHandler`, `BasePyTreeCheckpointHandler`, `PyTreeCheckpointHandler`, `StandardCheckpointHandler`, `JsonCheckpointHandler`, `ProtoCheckpointHandler`, `RandomKeyCheckpointHandler`, `JaxRandomKeyCheckpointHandler`, `NumpyRandomKeyCheckpointHandler`, `PyGrainCheckpointHandler`, `CompositeCheckpointHandler`, `CheckpointHandlerRegistry`, `DefaultCheckpointHandlerRegistry`, and `create_default_handler_registry`.
- [x] Extract `src/zero_orbax/_checkpoint/checkpointer_lib.py` (and `abstract_checkpointer`, `async_checkpointer`, `pytree_checkpointer`, `standard_checkpointer`) containing `AbstractCheckpointer`, `Checkpointer`, `AsyncCheckpointer`, `PyTreeCheckpointer`, and `StandardCheckpointer`.
- [x] Extract `src/zero_orbax/_checkpoint/manager_lib.py` (and `abstract_checkpoint_manager`) containing `CheckpointManagerOptions`, `AbstractCheckpointManager`, and `CheckpointManager`.
- [x] Extract `src/zero_orbax/_checkpoint/transform_lib.py` containing `Transform`, `RestoreTransform`, `apply_transformations`, `merge_trees`, and `_get_by_path`.
- [x] Extract `src/zero_orbax/_checkpoint/utils_lib.py` containing `checkpoints_iterator`, `construct_restore_args`, `restore_args_from_target`, `wait_for_new_checkpoint`, and `test_utils`.
- [x] Extract `src/zero_orbax/_checkpoint/namespaces_lib.py` and `src/zero_orbax/_checkpoint/v1_lib.py` containing `msgpack_utils`, `atomicity`, `metadata`, `multihost`, `epath`, `abstract_logger`, `aggregate_handlers`, `path`, `serialization`, `tree`, `utils`, and `v1`.
- [x] Isolate stdlib module re-exports (`asyncio`, `contextlib`, `functools`, `logging`) in the `zero_orbax.checkpoint` compatibility layer with explicit documentation.
- [x] Ensure `scripts/check_compliance.py` `collect_ast_definitions` scans all decomposed submodule files under `src/zero_orbax/` and verifies 100% compliance with `CANONICAL_V005_OP_DEFS`, `DEFAULT_STRUCTURAL_SNAPSHOT`, and `orbax_checkpoint_v0.12.8.json`.

---

## Phase 2: Eliminate Duplicate & Incorrect Handler Abstractions

### 2.1 Consolidate Triplicated Random Key Handlers
- [x] Refactor `RandomKeyCheckpointHandler` into the canonical base class with a class-level `_default_key_name: str = "random_key"` attribute.
- [x] Make `JaxRandomKeyCheckpointHandler(RandomKeyCheckpointHandler)` and `NumpyRandomKeyCheckpointHandler(RandomKeyCheckpointHandler)` inherit directly from `RandomKeyCheckpointHandler`, overriding `_default_key_name = "jax_random_key"` and `_default_key_name = "numpy_random_key"`.
- [x] Make `checkpoint_restore_args` construct a typed `RestoreArgs` / `ArrayRestoreArgs` from `item` when `restore_args is None` (instead of tautologically returning `restore_args` and ignoring `item`).
- [x] Make `post_restore` apply key/state reconstruction using `metadata` and `item` template (reconstructing JAX PRNG key arrays or sequence types) instead of returning `item` unchanged.
- [x] Make `RandomKeyCheckpointHandler.restore` pass the result of `checkpoint_restore_args(template)` to `post_restore(..., metadata=resolved_rargs)` instead of discarding its return value.

### 2.2 Unify Dual Array & PyTree Serialization Pipelines
- [x] Unify array, `FlaxVariable`, `FlaxState`, `FrozenDict`, `NamedTuple`/`OptState`, and `LogicalGraph` tagging between `zero_orbax.core.MsgpackEngine` and `BasePyTreeCheckpointHandler` so both tag schemes are recognized seamlessly.
- [x] Eliminate double-packing in `PyTreeCheckpointHandler.save` when `aggregate_filename` is set by packing `encoded_bytes` once and writing to `self.aggregate_filename or "checkpoint.msgpack"`.
- [x] Support recursive container traversal in `PyTreeCheckpointHandler._encode_custom_leaves` and `_decode_custom_leaves` for `list` and `tuple` sequences in addition to `dict`.
- [x] Support `list`, `tuple`, and `FrozenDictProtocol` traversal in `_order_prioritized_keys`, `_apply_save_args_tree`, and `_apply_restore_args_tree`.
- [x] Fix `_apply_save_args_tree` so that `SaveArgs(dtype=...)` also casts array/tensor leaves (`ArrayProtocol` / `jnp.ndarray` / `CompilerTensor` / `SerializedArray`), not just Python scalar primitives (`int`, `float`, `str`, `bool`).
- [x] Fix `SaveArgs(aggregate=...)`, `chunk_byte_size`, and `shard_axes` so they are extracted via `_extract_save_args_metadata` and recorded in checkpoint metadata (`_orbax_metadata.json`).

### 2.3 Fix `Checkpointer` & `AsyncCheckpointer` Temporary Path Staging Bug
- [x] Implement a concrete `AtomicityTemporaryPath(TemporaryPathProtocol)` in `zero_orbax.core` (and exposed as `atomicity.TemporaryPath`) that accepts `(temporary_path, final_path)` (and supports zero-arg construction for custom test classes), creates `<final_path>.orbax-checkpoint-tmp`, writes files into the staging directory, and atomically moves the staging directory to `final_path` upon `finalize()`.
- [x] Use atomic temporary directory staging in `Checkpointer.save` and `AsyncCheckpointer.save` via `_instantiate_temporary_path` so that partial/unfinalized checkpoints are staged in `.orbax-checkpoint-tmp` before `finalize()` promotes them to the final path.
- [x] Fix `AsyncCheckpointer.__init__` so `timeout_secs` synchronization with `async_options.timeout_secs` is consistent (`if async_options is not None and timeout_secs == 300: self.timeout_secs = async_options.timeout_secs`).
- [x] Fix `AsyncCheckpointer.save` when `self.handler` is a synchronous `CheckpointHandler` (not `AsyncCheckpointHandler`): ensure `barrier_sync_fn` and `temporary_path_class` staging are respected.
- [x] Fix `Future` and `AsyncCheckpointer.check_for_errors()` so that exceptions raised during background/commit callbacks are captured and re-raised on `check_for_errors()` and `result()`.

---

## Phase 3: Fix `CheckpointManager` State Duplication & Silent No-Op Options

### 3.1 Remove Unbounded In-Memory Checkpoint Cache (`self.checkpoints`)
- [x] Replace `self.checkpoints: dict[int, PyTree]` eager full-payload caching when `self.directory is not None`, only keeping an in-memory fallback store when `self.directory is None` (pure in-memory manager mode).
- [x] Fix `CheckpointManager.reload()` so it discovers step directories, step metadata, and `metrics.json` without eagerly reading and unpacking every historical step's full tensor/PyTree weights into RAM.
- [x] Fix `CheckpointManager.item_metadata(step)` to restore/inspect the step structure from disk when not cached in memory.
- [x] Fix `CheckpointManager.metadata(step)` so that when `step is not None`, it reads step-level `_orbax_metadata.json` from disk if present, and when `step is None`, returns the root metadata mapping.

### 3.2 Implement Ignored `CheckpointManagerOptions` & `CompositeCheckpointHandler` Options
- [x] Wire `CheckpointManagerOptions.enable_async_checkpointing`: when `enable_async_checkpointing=False`, force synchronous completion inside `CheckpointManager.save()` (calling `wait_until_finished()` before returning).
- [x] Wire `CheckpointManagerOptions.async_options`: support async options and checkpointer error/wait propagation in `CheckpointManager`.
- [x] Wire `CheckpointManagerOptions.temporary_path_class`, `file_options`, and `atomicity_options` into `CheckpointManager._persist_step_to_disk` so directory permissions (`file_options.path_permission_mode`) and atomic staging (`temporary_path_class`) apply to every step directory created during `save()`.
- [x] Wire `CheckpointManagerOptions.enable_per_process_directory_creation` and `multiprocessing_options` into step directory creation and barrier synchronization.
- [x] Fix duplicate condition in `CheckpointManager.save`: remove the redundant inline `keep_period` check when filtering `unprotected_candidates` since `_compute_protected_steps` already computes `keep_period`.
- [x] Fix `CheckpointManager._remove_step_storage` so `todelete_subdir` and `todelete_full_path` work seamlessly with both local paths and virtual URIs (`gs://`, `s3://`, `memory://`) via `FileSystemIO.move()`.
- [x] Wire `CompositeOptions` (`file_options`) inside `CompositeCheckpointHandler.save` and `async_save`.
- [x] Ensure `CompositeCheckpointHandler._resolve_handler_for_item` resolves handlers consistently across save and restore operations.

---

## Phase 4: Fix Core Storage, Sharding, and Serialization Edge Cases (`src/zero_orbax/core.py` & `types.py`)

### 4.1 `GlobalArraySharding` Multi-Axis & Array Type Preservation
- [x] Fix `GlobalArraySharding.get_local_shard` in `src/zero_orbax/core.py`: implement full multi-axis coordinate sharding across all partitioned axes in `partition_spec.axes` instead of breaking after the first axis.
- [x] Fix `GlobalArraySharding.get_local_shard` for `jnp.ndarray` and `CompilerTensor` inputs so `get_local_shard` reconstructs a `jnp.ndarray` or `CompilerTensor` of the same type and dtype instead of returning a raw Python `list`.
- [x] Consolidate `_infer_nested_shape` in `src/zero_orbax/core.py` and `_infer_array_shape` in `src/zero_orbax/checkpoint.py` into a single `infer_array_shape(val: PyTree) -> tuple[int, ...]` helper in `zero_orbax.types`.

### 4.2 `MsgpackEngine` Strictness & Error Handling
- [x] In `MsgpackEngine.unpackb` (`src/zero_orbax/core.py`): add `strict: bool = False` parameter that raises `ValueError` on corrupted binary MessagePack or legacy `_MSGPACK_MAGIC` payloads while keeping backward compatibility when `strict=False`.
- [x] In `_encode_pytree_to_json_value` and `_decode_json_value_to_pytree` (`src/zero_orbax/core.py`): explicitly support `set`, `frozenset`, and `pathlib.Path` nodes inside PyTrees with `__ztype__` tags.

### 4.3 `types.py` Type Guard Precision
- [x] Strengthen `is_pytree_dict` and `is_pytree_mapping` in `src/zero_orbax/types.py` to validate string keys (`all(isinstance(k, str) for k in value)`).
- [x] Maintain modern PEP 604 `X | Y` union syntax across all function/method/class annotations while keeping runtime `TypeAlias` assignments compatible with Python 3.9+.

---

## Phase 5: Harden Static Analysis & Linter Scripts (`scripts/`)

### 5.1 Close Reflection & Dunder Loopholes in `scripts/lint_typing.py`
- [x] Remove `target_cls.__module__.startswith("zero_grain")` dunder inspection from `_ArgsHandlerCatalog` and add `"__module__"`, `"__qualname__"`, `"__getattribute__"`, and `"__setattr__"` to `FORBIDDEN_DUNDER_ATTRS` in `scripts/lint_typing.py`.
- [x] Update `scan_file` in `scripts/lint_typing.py` to inspect **all** `ast.Assign` value subtrees for `FORBIDDEN_TYPE_NAMES` (not just variables ending with `"Alias"`).
- [x] Update `scan_file` in `scripts/lint_typing.py` to check `ast.ClassDef.bases` and `ast.ClassDef.keywords` for `FORBIDDEN_TYPE_NAMES` (`Any`, `object`).

### 5.2 Improve `scripts/check_docs.py` & `scripts/check_compliance.py`
- [x] Update `_has_direct_raise` in `scripts/check_docs.py` to walk the entire function body AST (including `if` / `for` / `with` / `try` blocks, excluding nested function/class definitions) and require `Raises:` in docstrings whenever any branch raises.
- [x] Enforce strict docstring section checks (`Args:`, `Returns:` / `Yields:`, `Raises:`) across `src/`, `scripts/`, and `tests/` in `scripts/check_docs.py`.
- [x] Ensure `scripts/check_compliance.py` reuses `_extract_documented_args` from `scripts.check_docs` without duplication.

---

## Phase 6: Eliminate Tautological, Shallow, and Mirrored Tests (`tests/`)

### 6.1 Fix Tautological & Shallow Assertions in `tests/test_missing_symbols.py`
- [x] Replace manual `try: ... raise AssertionError(...) except ValueError: pass` blocks in `tests/test_missing_symbols.py` with `pytest.raises(ValueError, match=...)` and `pytest.raises(FileNotFoundError, match=...)`.
- [x] Update `test_random_key_handler_and_default_registry` in `tests/test_missing_symbols.py` to verify real `RestoreArgs` / `ArrayRestoreArgs` construction and `post_restore` array/tuple reconstruction.
- [x] Add `v1.custom_metadata(path)` and verify in `test_v1_functional_apis` that `custom_metadata` persisted by `v1.save_pytree`, `v1.save_pytree_async`, and `v1.save_checkpointables` round-trips from disk.

### 6.2 Fix Tautological & Mirrored Implementation Tests in `tests/test_missing_coverage.py`
- [x] Update `test_async_checkpointer_methods`, `test_checkpointer_methods`, and `test_checkpoint_manager_advanced_features` in `tests/test_missing_coverage.py` to test real `AtomicityTemporaryPath` staging (verifying `.orbax-checkpoint-tmp` creation before commit and promotion to final destination upon `finalize()`).
- [x] Update `test_utils_and_submodules` and `test_checkpoint_utils_and_msgpack_utils` to test real multi-step iteration and `timeout_fn` predicates on `checkpoints_iterator`.
- [x] Update `test_standard_checkpointer_methods` in `tests/test_missing_coverage.py` to verify `StandardCheckpointer.save(path, ...)`, `metadata(path)`, and `structure(path)` on an actual saved standard checkpoint on disk.

### 6.3 Fix Tautological Tests in `tests/official/` & `tests/test_checkpoint.py`
- [x] Replace tautological `_always_save` in `tests/official/checkpoint_manager_options_test.py` with a meaningful `_save_every_even_step` predicate and verify both True and False evaluations before `read_only=True` nullifies it.
- [x] Replace private state mutation (`empty_mgr.steps.append(99); empty_mgr.checkpoints[99] = ...`) in `tests/test_checkpoint.py` with public `save`/`restore`/`item_metadata` calls on an in-memory `CheckpointManager(None)`.
- [x] Add multi-axis 2D sharding tests (`PartitionSpec("data", "model")`) and `jnp.ndarray` / `CompilerTensor` type preservation tests in `tests/test_orbax_primitives.py`.

---

## Phase 7: Final Verification & Quality Gate Checklist

- [x] Run `ruff format --check .` and `ruff check .` with 0 warnings/errors.
- [x] Run `mypy src scripts tests` with `--strict` and 0 errors across all source, script, and test files.
- [x] Run `python3 scripts/lint_typing.py` and confirm 0 typing or reflection violations (`Any`, `object`, `cast`, `getattr`, `setattr`, `hasattr`, `delattr`, `vars`, `globals`, `locals`, dunder attributes, or missing annotations).
- [x] Run `python3 scripts/check_allowed_imports.py src/zero_orbax` (across all submodule files) and confirm 0 disallowed third-party imports.
- [x] Run `python3 scripts/check_docs.py` and `interrogate -v -f 100 src scripts tests` and confirm 100.0% documentation coverage across all files, modules, classes, functions, and arguments (including nested `Raises:` checks).
- [x] Run `python3 scripts/check_compliance.py` and confirm 100% API and `ml-ecosystem-snapshots` v0.0.5 / v0.12.8 compliance.
- [x] Run `pytest --cov=src/zero_orbax --cov=scripts --cov=tests --cov-report=term-missing --cov-branch --cov-fail-under=100` and confirm 100% function, line, and branch test coverage.
- [x] Run `python3 scripts/update_badges.py` and verify `README.md` badges reflect 100% test coverage, 100% doc coverage, and 100% API compliance.
- [x] Verify `git status` and `git diff` are clean and **do not** run `git push`.
