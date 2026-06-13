# TODO Plan for 100% Compliance

Currently, the project passes API Compliance, Linting, Typing, Testing (100% test coverage), and allowed 3rd party imports. However, we are failing the **Documentation Coverage** hook due to 66 missing docstrings on the newly mocked methods added for `orbax` parity.

To reach 100% compliance, please add proper docstrings to the following items in `src/zero_orbax/checkpoint.py`:

## Classes Missing Docstrings
- [x] `CheckpointHandler`
- [x] `CheckpointHandlerRegistry`
- [x] `RestoreArgs`
- [x] `AbstractCheckpointManager` (Wait, it has one below the methods, it should be moved directly under the class definition)
- [x] `CheckpointManager` (Docstring needs to be moved under class definition)
- [x] `AbstractCheckpointer` (Docstring needs to be moved under class definition)
- [x] `AsyncCheckpointer` (Docstring needs to be moved under class definition)
- [x] `Checkpointer` (Docstring needs to be moved under class definition)
- [x] `PyTreeCheckpointer` (Docstring needs to be moved under class definition)
- [x] `StandardCheckpointer` (Docstring needs to be moved under class definition)
- [x] `ArrayRestoreArgs` (Docstring needs to be moved under class definition)

## Methods Missing Docstrings in `CheckpointHandler`
- [x] `metadata`
- [x] `finalize`
- [x] `restore`
- [x] `close`
- [x] `save`

## Methods Missing Docstrings in `CheckpointHandlerRegistry`
- [x] `get_all_entries`
- [x] `get`
- [x] `has`
- [x] `add`

## Methods Missing Docstrings in `AbstractCheckpointManager`
- [x] `reached_preemption`
- [x] `check_for_errors`
- [x] `metadata`
- [x] `best_step`
- [x] `close`
- [x] `metrics`
- [x] `reload`
- [x] `wait_until_finished`
- [x] `should_save`
- [x] `delete`
- [x] `item_metadata`

## Methods Missing Docstrings in `CheckpointManager`
- [x] `reached_preemption`
- [x] `check_for_errors`
- [x] `metadata`
- [x] `best_step`
- [x] `close`
- [x] `is_saving_in_progress`
- [x] `metrics`
- [x] `reload`
- [x] `wait_until_finished`
- [x] `should_save`
- [x] `delete`
- [x] `item_metadata`

## Methods Missing Docstrings in `AbstractCheckpointer`
- [x] `close`
- [x] `metadata`
- [x] `structure`

## Methods Missing Docstrings in `AsyncCheckpointer`
- [x] `check_for_errors`
- [x] `metadata`
- [x] `close`
- [x] `create_temporary_path`
- [x] `structure`

## Methods Missing Docstrings in `Checkpointer`
- [x] `close`
- [x] `metadata`
- [x] `structure`
- [x] `create_temporary_path`

## Methods Missing Docstrings in `PyTreeCheckpointer`
- [x] `close`
- [x] `metadata`
- [x] `structure`
- [x] `create_temporary_path`

## Methods Missing Docstrings in `StandardCheckpointer`
- [x] `check_for_errors`
- [x] `metadata`
- [x] `close`
- [x] `create_temporary_path`
- [x] `wait_until_finished`
- [x] `structure`

## Methods Missing Docstrings in `ArrayRestoreArgs`
- [x] `restore_type`

---
Once these docstrings are added (and existing class docstrings are moved to the top of the class definitions), `pre-commit run --all-files` will pass 100%.