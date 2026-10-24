# Zero-Orbax Exhaustive Audit & Remediation Plan (`TODO_PLAN.md`)

This document outlines the exhaustive engineering plan to remediate all code smells, broken abstractions, dunder/global hacks, weak typing, and tautological tests across `zero-orbax`, while strictly preserving:
- **100% Documentation Coverage** (module, file, class, function, method, and `Args:` parameter docstrings verified via `interrogate` and `scripts/check_docs.py`).
- **100% Test Coverage** (100% function, line, and branch coverage verified via `pytest --cov=src/zero_orbax --cov=scripts --cov-branch --cov-fail-under=100`).
- **Strong Typing** (explicit type annotations everywhere; zero `Any`, `object`, `typing.cast`, `getattr`, `setattr`, `hasattr`, `globals()`, `vars()`, `exec()`, or `__dict__` tricks).
- **Zero New Unapproved Dependencies** (relying strictly on Python stdlib, `pydantic`, `ml-switcheroo-ir`, `cdd-python`, `ml-ecosystem-snapshots`, and `../zero-*` / `../ml-switcheroo-*`).
- **All Pre-Commit Hooks Passing** (`ruff-format`, `ruff-check`, `mypy`, `lint-typing`, `interrogate`, `check-docs`, `check-allowed-imports`, `check-api-compliance`, `pytest-coverage`, `update-badges`).

---

## Phase 1: Eliminate Global Mutable Singletons & Scope-Shadowing Hacks

### 1.1 Replace Module-Level Mutable State in `src/zero_orbax/core.py` and `src/zero_orbax/checkpoint.py`
- [x] Replace module-level `_VIRTUAL_FS: dict[str, bytes]` in `src/zero_orbax/core.py` with an encapsulated `VirtualFileSystem` store class (with explicit `reset()` / scoped context support for test isolation instead of tests importing and mutating private `_VIRTUAL_FS`).
- [x] Replace module-level `_BARRIER_COUNTS: dict[str, int]` in `src/zero_orbax/core.py` with encapsulated state on `BarrierSync` (with an optional shared default store and explicit `reset()` method).
- [x] Remove module-level `_IN_MEMORY_CHECKPOINTS: dict[str, PyTree]` in `src/zero_orbax/checkpoint.py` so `CheckpointHandler` and `Checkpointer` persist through `FileSystemIO` / `AtomicWrite` (using `memory://` or local disk paths) rather than leaking state across unrelated test cases and handler instances.
- [x] Remove all test imports of private `_VIRTUAL_FS` and `_IN_MEMORY_CHECKPOINTS` in `tests/test_orbax_primitives.py` and `tests/test_missing_coverage.py`.

### 1.2 Eliminate `globals()["args"]` & Argument Shadowing in `src/zero_orbax/checkpoint.py`
- [x] Rename variadic `*pos_args: PyTree` in `CheckpointHandler.save`, `CheckpointHandler.restore`, `AsyncCheckpointHandler.async_save`, `ArrayCheckpointHandler.save`, `PyTreeCheckpointHandler.restore`, `JsonCheckpointHandler.save`, `AbstractCheckpointer.save`, `AbstractCheckpointer.restore`, `Checkpointer.save`, `Checkpointer.restore`, `AsyncCheckpointer.save`, `AsyncCheckpointer.restore`, and `StandardCheckpointer.restore` so that `*args` no longer shadows the module-level `args` namespace class.
- [x] Rename the `args: args.CheckpointArgs | PyTree = None` parameter internal references or extract a private `_CheckpointArgsBase` class defined above `args` so `isinstance(..., args.CheckpointArgs)` works directly without `globals()["args"]`.
- [x] Remove all 6 occurrences of `globals()["args"].CheckpointArgs` and `globals()["args"].PyTreeRestore` in `src/zero_orbax/checkpoint.py`.
- [x] Remove unused re-exported stdlib modules (`asyncio`, `contextlib`, `functools`, `logging`) from the top of `src/zero_orbax/checkpoint.py` if not required by snapshot, or explicitly bind them in dedicated compatibility aliases with clear typing.
- [x] Fix `NoneType: TypeAlias = None` in `src/zero_orbax/checkpoint.py` to `types.NoneType` (or `type(None)`).

---

## Phase 2: Strengthen Static Typing, Pydantic Models & AST Linter (`src/zero_orbax/types.py` & `scripts/lint_typing.py`)

### 2.1 Modernize Type Annotations & Eliminate `# type: ignore`
- [x] Replace legacy `typing.Union` and `typing.Optional` in `src/zero_orbax/types.py` with PEP 604 `|` union syntax across all method/function signatures while maintaining Python 3.9 runtime `TypeAlias` compatibility.
- [x] Narrow `CheckpointHandler.__init__`, `AbstractCheckpointer.__init__`, `AbstractCheckpointManager.__init__`, and `CheckpointHandlerRegistry.__init__` signatures so they do not accept arbitrary untyped `*args: PyTree, **kwargs: PyTree` where concrete parameters exist.
- [x] Narrow `PyTreeCheckpointHandler.restore` parameter `transforms_default_to_original: bool | PyTree = True`.
- [x] Convert `SaveArgs`, `RestoreArgs`, `ArrayRestoreArgs`, `AsyncOptions`, and `step_lib.NameFormat` into `@dataclasses.dataclass` classes with explicit field types and validation while preserving `ArrayRestoreArgs.restore_type()` method compatibility.
- [x] Remove `# type: ignore[arg-type]` in `tests/official/checkpoint_manager_options_test.py` by using a strongly typed `TypedDict` (`ReadOnlyOptionSpec`) for parameterized `CheckpointManagerOptions` kwargs.
- [x] Remove all `unittest.mock.MagicMock` and `unittest.mock.create_autospec` usages in `tests/official/test_checkpointer_official.py` and `tests/official/checkpoint_manager_options_test.py`, replacing them with strongly typed concrete test doubles (`RecordingCheckpointHandler` and `CustomNameFormat`).
- [x] Remove `warn_unreachable = false` and restrict `disallow_subclassing_any = false` in `pyproject.toml`.

### 2.2 Harden `scripts/lint_typing.py` Against Reflection & Dunder Tricks
- [x] Expand `FORBIDDEN_REFLECTION_CALLS` in `scripts/lint_typing.py` to also forbid `globals`, `locals`, `vars`, `exec`, `eval`, `compile`, `delattr`, and `__import__`.
- [x] Expand `scripts/lint_typing.py` attribute checks to forbid `.__dict__`, `.__class__`, `.__bases__`, and `.__mro__` manipulation across `src/`, `scripts/`, and `tests/`.
- [x] Add a token-based check in `scripts/lint_typing.py` that forbids `# type: ignore` comments across `src/`, `scripts/`, and `tests/`.
- [x] Remove `exec(...)` and `vars(...)` from `tests/test_scripts.py` by importing test fixture modules via `importlib.util.spec_from_file_location`.

---

## Phase 3: Fix Core Serialization & I/O Abstractions (`src/zero_orbax/core.py`)

### 3.1 Real Binary Msgpack / Lossless PyTree Serialization in `MsgpackEngine`
- [x] Fix `MsgpackEngine.packb({})` so packing an empty dictionary `{}` produces a valid serialized empty-map payload (`b"\x80"`) that round-trips via `unpackb(packb({})) == {}` instead of returning `b""` and unpacking to `None`.
- [x] Implement RFC 8949 / MessagePack spec-compliant binary framing in `MsgpackEngine` (supporting fixmap/map16/map32, fixarray/array16/array32, fixstr/str8/str16/str32, int/uint 8/16/32/64, float64, booleans, nil, and tagged structures for `tuple`, `complex`, `ArrayProtocol`, and `LogicalGraph`) while retaining backward-compatible decoding for `_MSGPACK_MAGIC` payloads.
- [x] Preserve `ArrayProtocol` metadata (`shape`, `dtype`, and `data`/`hex`) on round-trip and reconstruct a concrete `SerializedArray` implementing `ArrayProtocol` and `BytesSerializableProtocol` when `preserve_array=True`.
- [x] Eliminate silent `str(node)` coercion of unknown objects in `_encode_pytree_to_json_value` and support `GraphProtocol` leaves explicitly.

### 3.2 Fix `AtomicWrite` & `FileSystemIO` Parent Directory Semantics
- [x] Fix `AtomicWrite.atomic_write_bytes(path, data)` in `src/zero_orbax/core.py` so that when targeting a local filesystem path whose parent directory does not yet exist (e.g., `step_dir/checkpoint.msgpack`), it creates `parent.mkdir(parents=True, exist_ok=True)` and performs the atomic temp-file write + `os.replace` to disk, rather than silently skipping disk persistence when `parent.exists()` is `False`.
- [x] Support `FileOptions.path_permission_mode` in `FileSystemIO.makedirs`, `FileSystemIO.write`, and `AtomicWrite.atomic_write_bytes` when specified.
- [x] Ensure `TensorStoreDriver` writes and reads chunked array metadata (`dtype`, `shape`, `chunks`) consistently on disk and in `memory://` / `gs://` / `s3://` URIs.
- [x] Enhance `GlobalArraySharding.get_local_shard` to respect multi-axis `PartitionSpec` and `LogicalMesh` shapes from `ml_switcheroo_ir` in addition to 1D host count slicing.

---

## Phase 4: Fix Broken Abstractions in Handlers, Checkpointers & Manager (`src/zero_orbax/checkpoint.py`)

### 4.1 Implement Real Concrete `CheckpointHandler` Subclasses
- [x] **`CompositeCheckpointHandler`**:
  - [x] Implement `save(directory, args)` to iterate over `args.Composite` items (or `kwargs`), resolve each sub-item's handler via `self._item_handlers` or `self.handler_registry` or `args.get_registered_handler_cls`, create `<directory>/<item_name>`, and delegate `sub_handler.save(...)`.
  - [x] Implement `async_save(directory, args)` to aggregate `Future` lists across all sub-handlers.
  - [x] Implement `restore(directory, args)` to restore each sub-item from `<directory>/<item_name>` using its registered/configured handler and return a `dict[str, PyTree]` (or composite structure).
  - [x] Implement `metadata(directory)`, `structure(directory)`, `finalize(directory)`, and `close()` to delegate across all managed sub-handlers.
- [x] **`JsonCheckpointHandler`**:
  - [x] Override `restore(directory, item)` to read and parse `<directory>/<self.filename or 'metadata.json'>` directly using `json.loads` (instead of inheriting `CheckpointHandler.restore` which reads `checkpoint.msgpack`).
  - [x] Avoid writing redundant `checkpoint.msgpack` in `JsonCheckpointHandler.save` when saving JSON artifacts.
- [x] **`ArrayCheckpointHandler`**:
  - [x] Override `restore(directory, item)` to read from `<directory>/<self.checkpoint_name or 'array'>.ts` via `TensorStoreDriver`, applying `ArrayRestoreArgs` (`dtype`, `shape`, `global_shape`, `restore_type`) when provided.
- [x] **`ProtoCheckpointHandler`**:
  - [x] Override `save(directory, item)` and `restore(directory, item)` to read and write `<directory>/<self.filename>` deterministically using `AtomicWrite` and `FileSystemIO`.
- [x] **`JaxRandomKeyCheckpointHandler` & `NumpyRandomKeyCheckpointHandler`**:
  - [x] Implement `save(directory, item)` and `restore(directory, item)` using `self.key_name` and invoking `checkpoint_save_args`, `checkpoint_restore_args`, and `post_restore`.
- [x] **`BasePyTreeCheckpointHandler` & `PyTreeCheckpointHandler`**:
  - [x] Delegate to `self.handler_impl` in `PyTreeCheckpointHandler` when `handler_impl` is provided.
  - [x] Support custom leaf serialization via `self.type_handler_registry` (`TypeHandlerRegistryProtocol`) when saving and restoring PyTree leaves.
  - [x] Respect `aggregate_filename` when writing aggregated msgpack files.
  - [x] Support `SaveArgs` (`dtype`, `aggregate`) and `RestoreArgs` / `ArrayRestoreArgs` (`restore_type`, `dtype`, `shape`) per-leaf trees passed via `PyTreeSave` / `PyTreeRestore` / `StandardSave` / `StandardRestore`.

### 4.2 Fix `Checkpointer`, `AsyncCheckpointer`, `PyTreeCheckpointer`, and `StandardCheckpointer`
- [x] **`Checkpointer`**:
  - [x] Ensure handler-less fallback persists to disk via `AtomicWrite` / `FileSystemIO`.
  - [x] Call `self.handler.finalize(path)` during `save`.
  - [x] Read and write metadata via `self.checkpoint_metadata_store` (`CheckpointMetadataStoreProtocol`) when configured.
- [x] **`AsyncCheckpointer`**:
  - [x] Delegate `save` to `self.handler.async_save(...)` when `isinstance(self.handler, AsyncCheckpointHandler)` and wire a real `commit_callback` on the returned `Future` that invokes `self.handler.finalize(path)`, `barrier_sync_fn`, and `post_finalization_callback`.
  - [x] Ensure `AsyncCheckpointer.restore` and `StandardCheckpointer.restore` operate deterministically against on-disk checkpoints.

### 4.3 Fix `CheckpointManager` Delegation & Retention Policies
- [x] **Step Naming Format Integration**:
  - [x] Compute step directory names in `CheckpointManager` using `options.step_name_format.build_name(step)` or `step_lib.standard_name_format(step_prefix=options.step_prefix, step_format_fixed_length=options.step_format_fixed_length).build_name(step)` instead of hardcoding `f"{directory}/{step}"`.
  - [x] Update `CheckpointManager.reload()` and `checkpoint_utils.checkpoints_iterator()` to parse formatted step directory names (including `step_prefix` and zero-padded step numbers).
- [x] **Checkpointer & Handler Delegation**:
  - [x] Route `CheckpointManager.save()` and `CheckpointManager.restore()` through configured `self.checkpointers`, `self.item_handlers`, `self.handler_registry`, or `CompositeCheckpointHandler` (when `item_names` / multiple items or `args.Composite` are used) instead of bypassing them and calling `MsgpackEngine().packb` directly.
  - [x] Propagate `save_kwargs` and `restore_kwargs` to the underlying checkpointer/handler.
  - [x] Support the `directory` override parameter in `CheckpointManager.restore(step, items, restore_kwargs, directory=...)`.
  - [x] Persist and reload root metadata (`save_root_metadata`) and per-step metrics (`metrics.json` unless `prevent_write_metrics=True`) on disk so `reload()` restores `_step_metrics` and `_metadata`.
- [x] **Retention & Preservation Policies**:
  - [x] Implement `keep_period` in `CheckpointManager.save()` so steps where `step % keep_period == 0` are exempt from `max_to_keep` pruning.
  - [x] Implement `should_keep_fn(step, timestamp)` and `preservation_policy.should_preserve(steps)` in `CheckpointManager` pruning logic.
  - [x] Implement `keep_checkpoints_without_metrics` and `best_fn` interaction during pruning (protecting the top `max_to_keep` best steps when `best_fn` is set, rather than FIFO-evicting the best step).
  - [x] Support `todelete_subdir` and `todelete_full_path` by moving pruned checkpoints into the deletion subdirectory when configured.
  - [x] Implement `enable_should_save_is_saving_in_progress_check` in `CheckpointManager.should_save()`.
  - [x] Expand `CheckpointManagerOptions.replace(**kwargs)` to accept `step_prefix`, `step_format_fixed_length`, `keep_period`, `read_only`, `best_mode`, `create`, `save_interval_steps`, and `max_to_keep`.
- [x] **Transformations (`apply_transformations`)**:
  - [x] Implement `Transform.use_fallback` and `Transform.multi_value_fn_input_args` in `apply_transformations`.
  - [x] Support regex key matching in `Transform.original_key` and key paths when transforming PyTrees.

---

## Phase 5: Replace Tautological Tests with Behavioral & Property Assertions (`tests/`)

### 5.1 Overhaul `tests/test_missing_symbols.py`
- [x] Replace `assert cls_name in exported_members` and `assert inspect.signature(cls_obj) is not None` with behavioral contract tests that instantiate exported classes and invoke their methods to verify post-conditions.
- [x] Add a test comparing `zero_orbax.checkpoint` against `ml_ecosystem_snapshots` v0.0.4 (in addition to `tests/orbax_api_snapshot.json`) via `scripts/check_compliance.py`.

### 5.2 Overhaul `tests/official/` Tautological Tests
- [x] **`tests/official/test_args_official.py`**:
  - [x] Replace `assertIsNotNone(ocp.args.Composite(a=1, b=2))` and `assertIsNotNone(ocp.args.PyTreeSave(item=1))` with round-trip save/restore tests through `CompositeCheckpointHandler`, `PyTreeCheckpointHandler`, `StandardCheckpointHandler`, and `ArrayCheckpointHandler` using `args.Composite`, `args.PyTreeSave`, `args.PyTreeRestore`, `args.StandardSave`, `args.StandardRestore`, `args.JsonSave`, `args.JsonRestore`, `args.ArraySave`, and `args.ArrayRestore`.
- [x] **`tests/official/test_checkpointer_official.py`**:
  - [x] Remove `mock.MagicMock()` from `CheckpointerTest.setUp`.
  - [x] Test `Checkpointer`, `AsyncCheckpointer`, `PyTreeCheckpointer`, and `StandardCheckpointer` against real `tempfile.TemporaryDirectory()` directories verifying on-disk file creation, async barrier/finalization callbacks, and PyTree value round-tripping.
- [x] **`tests/official/test_pytree_checkpoint_handler_official.py`**:
  - [x] Instantiate `ocp.PyTreeCheckpointHandler` (instead of base `CheckpointHandler`), save nested PyTrees with arrays/scalars to `TemporaryDirectory`, and restore them with `Transform` rules.
- [x] **`tests/official/test_checkpoint_manager_official.py`**:
  - [x] Replace hardcoded `/tmp/mock_ckpt_dir` with isolated `tempfile.TemporaryDirectory()` directories.
  - [x] Add tests for multi-item `checkpointers={"params": ..., "config": ...}`, `keep_period`, `best_fn` retention (ensuring best step is not evicted by `max_to_keep`), `step_prefix`, `step_format_fixed_length`, and `reload()` restoring steps from disk.
- [x] **`tests/official/checkpoint_manager_options_test.py`**:
  - [x] Replace `MOCK_STEP_NAME_FORMAT = mock.create_autospec(ocp.step.NameFormat)` with a concrete typed `CustomNameFormat(ocp.step.NameFormat)` subclass.

### 5.3 Strengthen `tests/test_missing_handlers.py`, `tests/test_missing_coverage.py`, and `tests/test_checkpoint.py`
- [x] In `tests/test_missing_handlers.py`, test actual disk round-trips for `CompositeCheckpointHandler`, `ProtoCheckpointHandler`, `JaxRandomKeyCheckpointHandler`, `NumpyRandomKeyCheckpointHandler`, `ArrayCheckpointHandler` (verifying `.ts` file restoration), and `JsonCheckpointHandler` (verifying `.json` file restoration when `checkpoint.msgpack` is absent).
- [x] In `tests/test_checkpoint.py`, add tests for `MsgpackEngine().unpackb(MsgpackEngine().packb({})) == {}` and `CheckpointManager.save(1, {})` round-tripping an empty dictionary on disk.

---

## Phase 6: Final Verification & Pre-Commit Gate

- [x] Run `ruff format --check .` and `ruff check .` with zero warnings or errors.
- [x] Run `mypy --strict src tests scripts` with zero errors and zero `# type: ignore` comments.
- [x] Run `python3 scripts/lint_typing.py` with zero typing or reflection violations.
- [x] Run `interrogate -vv -f 100 -M -i -c pyproject.toml src scripts tests` and `python3 scripts/check_docs.py` confirming 100.0% docstring coverage across all modules, classes, functions, and arguments.
- [x] Run `python3 scripts/check_allowed_imports.py src/zero_orbax/__init__.py src/zero_orbax/types.py src/zero_orbax/core.py src/zero_orbax/checkpoint.py` confirming zero disallowed imports.
- [x] Run `python3 scripts/check_compliance.py` confirming 100% API snapshot compliance.
- [x] Run `pytest --cov=src/zero_orbax --cov=scripts --cov-branch --cov-report=term-missing --cov-fail-under=100` confirming 100% line, function, and branch test coverage.
- [x] Run `pre-commit run --all-files` to verify all 10 pre-commit hooks pass cleanly.
- [x] Ensure `git push` is NEVER executed.
