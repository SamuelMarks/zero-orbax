# zero-orbax Semantic Implementation Plan

**Note: All phases of the semantic implementation plan are now complete.**

The structural API definitions for `zero_orbax.checkpoint` are complete, and the semantic logic has been fully implemented. 

`zero-orbax` is responsible for state persistence (checkpoint loading and saving) within the `ml-switcheroo` ecosystem. It acts as the serialization layer for PyTrees, allowing models trained or defined in `zero-jax`, `zero-flax`, or other ecosystem frontends to be saved and restored, specifically supporting standard msgpack/tensorstore/zarr formats (or simplified variants) used by the real `orbax`.

## Phase 1: Core Interfaces & Types
- [x] Implement `Transform` class logic (storing and applying key/value regex mappings).
- [x] Implement `RestoreTransform` class logic (handling fallback and value extraction).
- [x] Implement `apply_transformations` utility to recursively process PyTrees against transformation rules.
- [x] Implement `merge_trees` utility for deep merging of nested dictionary PyTrees.

## Phase 2: Checkpoint Managers
- [x] Implement `CheckpointManagerOptions` property getters and validation logic.
- [x] Implement `AbstractCheckpointManager` base state.
- [x] Implement `CheckpointManager.save()` - Serialize PyTree state to disk using `numpy` and `msgpack` or standard JSON (based on `options`).
- [x] Implement `CheckpointManager.restore()` - Read bytes from disk and reconstitute the Python structures.
- [x] Implement `CheckpointManager.latest_step()` and `CheckpointManager.all_steps()`.
- [x] Implement directory rotation and retention logic (`max_to_keep`, `keep_period`).

## Phase 3: Checkpointers
- [x] Implement `AbstractCheckpointer` base logic.
- [x] Implement `Checkpointer` utilizing the injected `CheckpointHandler`.
- [x] Implement `PyTreeCheckpointer` shorthand for default PyTree serialization.
- [x] Implement `StandardCheckpointer` shorthand for generic standard checkpointing.

## Phase 4: Asynchronous & Futures
- [x] Implement `AsyncOptions` configuration state.
- [x] Implement `Future` wrapper to simulate deferred execution (using standard `concurrent.futures` or dummy synchronous resolution for `zero-orbax`'s trace environment).
- [x] Implement `AsyncCheckpointer` utilizing the `Future` wrapper to return deferred save/restore operations.

## Phase 5: Verification & Testing
- [x] Write integration tests for round-trip saving and loading a complex PyTree using `CheckpointManager`.
- [x] Write tests verifying `max_to_keep` and background delete logic.
- [x] Write tests verifying `apply_transformations` correctly modifies keys and applies `value_fn`.
- [x] Verify no external dependencies (other than `numpy` and standard library) are used.
