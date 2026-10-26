"""Checkpoint discovery, waiting, and test utilities for zero_orbax.checkpoint."""

from __future__ import annotations

import pathlib
import time
from collections.abc import Callable, Iterator

from ml_switcheroo_ir import PartitionSpec

from zero_orbax._checkpoint.step_lib import step_lib
from zero_orbax._checkpoint.type_handlers_lib import ArrayRestoreArgs, RestoreArgs
from zero_orbax.core import BarrierSync, FileSystemIO
from zero_orbax.types import (
    FlaxStateProtocol,
    FlaxVariableProtocol,
    FrozenDictProtocol,
    NamedTupleProtocol,
    PathLike,
    PyTree,
    is_array_protocol,
    is_pytree_dict,
    normalize_path,
)


class checkpoint_utils:
    """Namespace for checkpoint discovery and iteration utilities."""

    @staticmethod
    def checkpoints_iterator(
        checkpoint_dir: PathLike,
        min_interval_secs: int = 0,
        timeout: int | None = None,
        timeout_fn: Callable[[], bool] | None = None,
        *,
        step_prefix: str | None = None,
        step_name_format: step_lib.NameFormat | None = None,
    ) -> Iterator[int]:
        """Yield checkpoint step numbers discovered in ``checkpoint_dir``.

        Args:
            checkpoint_dir: Base directory containing step subdirectories.
            min_interval_secs: Minimum interval in seconds between directory scans. Defaults to 0.
            timeout: Optional timeout in seconds. Defaults to None.
            timeout_fn: Optional callable returning True when iteration should stop. Defaults to None.
            step_prefix: Optional step directory prefix. Defaults to None.
            step_name_format: Optional ``step_lib.NameFormat`` instance. Defaults to None.

        Yields:
            int: Discovered checkpoint step numbers in ascending order.
        """
        _ = timeout
        if min_interval_secs < 0:
            return
        if timeout_fn is not None and timeout_fn():
            return
        formatter = step_name_format or step_lib.standard_name_format(step_prefix=step_prefix)
        discovered_steps: list[int] = []
        for entry in FileSystemIO().listdir(checkpoint_dir):
            parsed = formatter.parse_step(entry)
            if parsed is not None:
                discovered_steps.append(parsed)
        for step_num in sorted(discovered_steps):
            yield step_num

    @staticmethod
    def construct_restore_args(
        target: PyTree,
        sharding_tree: PyTree = None,
        set_global_shape: bool = True,
    ) -> PyTree:
        """Construct a PyTree of ``RestoreArgs`` or ``ArrayRestoreArgs`` from a target template.

        Args:
            target: Target PyTree structure whose leaf types and shapes guide restoration.
            sharding_tree: Optional matching PyTree of shardings / ``PartitionSpec`` objects. Defaults to None.
            set_global_shape: Whether to populate ``global_shape`` on ``ArrayRestoreArgs``. Defaults to True.

        Returns:
            PyTree: Matching PyTree of ``RestoreArgs`` / ``ArrayRestoreArgs``.
        """
        if isinstance(target, FlaxVariableProtocol):
            return checkpoint_utils.construct_restore_args(
                target.value,
                sharding_tree,
                set_global_shape=set_global_shape,
            )
        if isinstance(target, (FlaxStateProtocol, FrozenDictProtocol)):
            sharding_sub = sharding_tree if is_pytree_dict(sharding_tree) else {}
            return {
                str(k): checkpoint_utils.construct_restore_args(
                    v,
                    sharding_sub.get(str(k)),
                    set_global_shape=set_global_shape,
                )
                for k, v in target.items()
            }
        if isinstance(target, NamedTupleProtocol):
            field_map = target._asdict()
            sharding_seq = sharding_tree if isinstance(sharding_tree, (list, tuple)) else ()
            return {
                str(f): checkpoint_utils.construct_restore_args(
                    field_map.get(f),
                    sharding_seq[i] if i < len(sharding_seq) else None,
                    set_global_shape=set_global_shape,
                )
                for i, f in enumerate(target._fields)
            }
        if is_pytree_dict(target):
            sharding_sub = sharding_tree if is_pytree_dict(sharding_tree) else {}
            return {
                k: checkpoint_utils.construct_restore_args(
                    v,
                    sharding_sub.get(k),
                    set_global_shape=set_global_shape,
                )
                for k, v in target.items()
            }
        if isinstance(target, list):
            sharding_seq = sharding_tree if isinstance(sharding_tree, (list, tuple)) else ()
            return [
                checkpoint_utils.construct_restore_args(
                    v,
                    sharding_seq[i] if i < len(sharding_seq) else None,
                    set_global_shape=set_global_shape,
                )
                for i, v in enumerate(target)
            ]
        if isinstance(target, tuple):
            sharding_seq = sharding_tree if isinstance(sharding_tree, (list, tuple)) else ()
            return tuple(
                checkpoint_utils.construct_restore_args(
                    v,
                    sharding_seq[i] if i < len(sharding_seq) else None,
                    set_global_shape=set_global_shape,
                )
                for i, v in enumerate(target)
            )
        if is_array_protocol(target):
            sharding_val: PartitionSpec | str | None = (
                sharding_tree if isinstance(sharding_tree, (PartitionSpec, str)) else None
            )
            g_shape = tuple(int(d) for d in target.shape) if set_global_shape else None
            return ArrayRestoreArgs(
                restore_type=type(target),
                sharding=sharding_val,
                global_shape=g_shape,
                shape=tuple(int(d) for d in target.shape),
                dtype=str(target.dtype),
            )
        return RestoreArgs(restore_type=type(target) if target is not None else None)

    @staticmethod
    def restore_args_from_target(
        target: PyTree,
        sharding_tree: PyTree = None,
    ) -> PyTree:
        """Alias for ``construct_restore_args`` building ``RestoreArgs`` from ``target``.

        Args:
            target: Target PyTree structure.
            sharding_tree: Optional matching sharding PyTree. Defaults to None.

        Returns:
            PyTree: Matching PyTree of ``RestoreArgs`` / ``ArrayRestoreArgs``.
        """
        return checkpoint_utils.construct_restore_args(target, sharding_tree)

    @staticmethod
    def wait_for_new_checkpoint(
        checkpoint_dir: PathLike,
        *,
        until_step: int | None = None,
        seconds_to_sleep: int = 1,
        timeout: int | None = None,
        step_prefix: str | None = None,
        step_format_fixed_length: int | None = None,
        step_name_format: step_lib.NameFormat | None = None,
    ) -> int:
        """Wait for a new checkpoint step directory to appear in ``checkpoint_dir``.

        Args:
            checkpoint_dir: Base checkpoint directory to watch.
            until_step: Optional minimum step number to wait for. Defaults to None.
            seconds_to_sleep: Sleep interval between directory scans. Defaults to 1.
            timeout: Maximum seconds to wait before returning ``-1``. Defaults to None.
            step_prefix: Optional step directory prefix. Defaults to None.
            step_format_fixed_length: Optional fixed digit width for step directories. Defaults to None.
            step_name_format: Optional ``step_lib.NameFormat`` instance. Defaults to None.

        Returns:
            int: Discovered step number, or ``-1`` if no matching checkpoint is found before timeout.
        """
        formatter = step_name_format or step_lib.standard_name_format(
            step_prefix=step_prefix,
            step_format_fixed_length=step_format_fixed_length,
        )
        deadline = time.monotonic() + (float(timeout) if timeout is not None else 0.0)
        io = FileSystemIO()
        while True:
            steps: list[int] = []
            for entry in io.listdir(checkpoint_dir):
                parsed = formatter.parse_step(entry)
                if parsed is not None:
                    steps.append(parsed)
            if steps:
                max_step = max(steps)
                if until_step is None or max_step >= until_step:
                    return max_step
            if timeout is None or time.monotonic() >= deadline:
                return -1
            time.sleep(min(float(seconds_to_sleep), 0.01))


class test_utils:
    """Namespace for synchronization and filesystem test utilities."""

    @staticmethod
    def sync_global_processes(name: str) -> None:
        """Synchronize all processes at a named barrier.

        Args:
            name: Barrier identifier string.
        """
        BarrierSync().wait(name)

    @staticmethod
    def erase_and_create_empty(directory: PathLike) -> pathlib.Path:
        """Remove ``directory`` if it exists and recreate it as an empty directory.

        Args:
            directory: Target directory path to reset.

        Returns:
            pathlib.Path: Normalized path to the newly created empty directory.
        """
        io = FileSystemIO()
        io.remove(directory)
        io.makedirs(directory, exist_ok=True)
        return normalize_path(directory)


__all__ = ["checkpoint_utils", "test_utils"]
