# Orbax Official Test Suite Porting Plan

## Phase 1: Setup and Discovery
- [x] Identify the official `google/orbax` repository and navigate to the `v0.6.4` tag.
- [x] Locate the core `orbax.checkpoint` test files (e.g., `checkpoint_manager_test.py`, `checkpointer_test.py`, `pytree_checkpoint_handler_test.py`, etc.).
- [x] Create `requirements-test.txt` in the root directory.
- [x] Add standard testing dependencies (e.g., `pytest`, `pytest-cov`, `numpy`) to `requirements-test.txt`.
- [x] Determine if any mock dependencies (like `absl-py` or dummy `jax` pytrees) are required for the test inputs to match the official shapes and structures.

## Phase 2: Code Acquisition and Adaptation
- [x] Download the target test files from the official `orbax` v0.6.4 source.
- [x] Set up a new testing directory (e.g., `tests/official/`) to house the ported tests.
- [x] Refactor imports within the downloaded tests:
  - Replace `orbax.checkpoint` with `zero_orbax.checkpoint`.
  - Replace `orbax` with `zero_orbax`.
- [x] Adapt test fixtures:
  - Replace heavy hardware-specific fixtures (e.g., TPU sharding, real distributed JAX arrays) with `numpy` arrays or mock objects that evaluate to the same PyTree structures.
  - Stub out `tensorstore` or `msgpack` native operations if they attempt real disk binary IO that conflicts with `zero-orbax`'s trace-based synchronous Future mocks.

## Phase 3: Porting Component Tests
- [x] Port and adapt **CheckpointManagerOptions** tests.
- [x] Port and adapt **CheckpointManager** tests.
  - [x] Verify `save`, `restore`, `wait_until_finished`, and `close` logic.
  - [x] Verify rotation policies (e.g., `max_to_keep`, `keep_time_interval`).
- [x] Port and adapt **Checkpointer** tests.
- [x] Port and adapt **PyTreeCheckpointHandler** tests.
  - [x] Verify dictionary merging and state dictionary persistence logic.
- [x] Port and adapt any relevant **Future/Async** test logic, ensuring our synchronous Future mock behaves compatibly from the caller's perspective.
- [x] Port and adapt **Args/Kwargs** saving structure tests (`args.py`).

## Phase 4: Output Parity and Final Verification
- [x] Execute the test suite using `pytest`.
- [x] Debug and patch `zero_orbax` implementations where behavior diverges from official `orbax` expectations (e.g., exception types, return shapes).
- [x] Ensure outputs are 1-to-1 the same (or `np.allclose` similar for numeric mock arrays).
- [x] Add `requirements-test.txt` installation to the GitHub Actions CI workflow (`.github/workflows/ci.yml`).
- [x] Run standard quality tools (linters, formatters, type-checkers).
- [x] Confirm 100% test and doc coverage are maintained.