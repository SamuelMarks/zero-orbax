"""Checkpoint argument wrappers and handler catalog for zero_orbax.checkpoint.args."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar
from typing_extensions import TypeAlias

from zero_orbax._checkpoint.type_handlers_lib import ArrayRestoreArgs, RestoreArgs, SaveArgs
from zero_orbax.types import PyTree


class _CheckpointArgsBase:
    """Base class for all structured checkpoint handler argument wrappers."""

    def __init__(self, item: PyTree = None) -> None:
        """Initialize the base CheckpointArgs wrapper.

        Args:
            item: Optional item payload wrapped by these arguments. Defaults to None.
        """
        self.item = item


_ArgsClsT = TypeVar("_ArgsClsT", bound=type[_CheckpointArgsBase])
_RegisteredArgsCls: TypeAlias = type[_CheckpointArgsBase]


class _ArgsHandlerCatalog:
    """Encapsulated catalog mapping ``CheckpointArgs`` classes to ``CheckpointHandler`` classes."""

    def __init__(self) -> None:
        """Initialize the empty catalog mappings."""
        self.save_args_to_handler: dict[_RegisteredArgsCls, type] = {}
        self.restore_args_to_handler: dict[_RegisteredArgsCls, type] = {}
        self.handler_to_save_args: dict[type, _RegisteredArgsCls] = {}
        self.handler_to_restore_args: dict[type, _RegisteredArgsCls] = {}
        self.canonical_grain_handler: type | None = None

    def register(
        self,
        args_cls: _RegisteredArgsCls,
        handler_cls: type,
        *,
        for_save: bool = False,
        for_restore: bool = False,
    ) -> None:
        """Register an association between ``args_cls`` and ``handler_cls``.

        Args:
            args_cls: ``CheckpointArgs`` subclass.
            handler_cls: ``CheckpointHandler`` subclass.
            for_save: Whether ``args_cls`` is used for save operations. Defaults to False.
            for_restore: Whether ``args_cls`` is used for restore operations. Defaults to False.
        """
        if for_save:
            self.save_args_to_handler[args_cls] = handler_cls
            self.handler_to_save_args[handler_cls] = args_cls
        if for_restore:
            self.restore_args_to_handler[args_cls] = handler_cls
            self.handler_to_restore_args[handler_cls] = args_cls

    def get_handler_cls(
        self,
        arg_or_cls: _CheckpointArgsBase | PyTree,
    ) -> type | None:
        """Look up the registered ``CheckpointHandler`` subclass for ``arg_or_cls``.

        Args:
            arg_or_cls: ``CheckpointArgs`` instance or subclass.

        Returns:
            type | None: Registered handler class, or None if not found.
        """
        target_cls: type = arg_or_cls if isinstance(arg_or_cls, type) else type(arg_or_cls)
        if issubclass(target_cls, _CheckpointArgsBase):
            if target_cls in self.save_args_to_handler:
                return self.save_args_to_handler[target_cls]
            if target_cls in self.restore_args_to_handler:
                return self.restore_args_to_handler[target_cls]
        return None

    def _resolve_handler_cls(self, handler_or_cls: PyTree) -> type:
        """Resolve a handler or args instance/class to its canonical registered class.

        Args:
            handler_or_cls: Handler or args instance or class.

        Returns:
            type: Canonical registered class.
        """
        target_cls: type = (
            handler_or_cls if isinstance(handler_or_cls, type) else type(handler_or_cls)
        )
        from zero_orbax._checkpoint.handlers_lib import (
            CheckpointHandler,
            PyGrainCheckpointHandler,
        )

        if target_cls is not PyGrainCheckpointHandler and issubclass(target_cls, CheckpointHandler):
            import zero_grain

            if issubclass(target_cls, zero_grain.PyGrainCheckpointHandler):
                return PyGrainCheckpointHandler
        return target_cls

    def get_args_cls(
        self,
        handler_or_cls: PyTree,
        *,
        for_save: bool = True,
    ) -> _RegisteredArgsCls | None:
        """Look up the registered ``CheckpointArgs`` subclass for ``handler_or_cls``.

        Args:
            handler_or_cls: ``CheckpointHandler`` instance or subclass.
            for_save: Whether to query save args or restore args. Defaults to True.

        Returns:
            _RegisteredArgsCls | None: Registered ``CheckpointArgs`` subclass, or None.
        """
        target_cls = self._resolve_handler_cls(handler_or_cls)
        if for_save and target_cls in self.handler_to_save_args:
            return self.handler_to_save_args[target_cls]
        if not for_save and target_cls in self.handler_to_restore_args:
            return self.handler_to_restore_args[target_cls]
        return None

    def has_registered(self, handler_or_cls: PyTree) -> bool:
        """Return whether ``handler_or_cls`` is registered for save or restore.

        Args:
            handler_or_cls: Handler or args class/instance.

        Returns:
            bool: True if registered.
        """
        target_cls = self._resolve_handler_cls(handler_or_cls)
        return (
            target_cls in self.handler_to_save_args
            or target_cls in self.handler_to_restore_args
            or target_cls in self.save_args_to_handler
            or target_cls in self.restore_args_to_handler
        )


_CATALOG = _ArgsHandlerCatalog()


class args:
    """Namespace for checkpoint argument classes and handler registration utilities.

    Contains structured argument wrappers that specify how different types of
    checkpoint items should be saved or restored, along with handler registration
    lookup helpers.
    """

    class CheckpointArgs(_CheckpointArgsBase):
        """Base class for all structured checkpoint handler argument wrappers."""

    class ArraySave(CheckpointArgs):
        """Arguments for saving a single array or tensor."""

        def __init__(
            self,
            item: PyTree = None,
            save_args: SaveArgs | None = None,
        ) -> None:
            """Initialize the ArraySave arguments.

            Args:
                item: The array or tensor to save. Defaults to None.
                save_args: Optional ``SaveArgs`` configuration. Defaults to None.
            """
            super().__init__(item=item)
            self.save_args = save_args

    class ArrayRestore(CheckpointArgs):
        """Arguments for restoring a single array or tensor."""

        def __init__(
            self,
            item: PyTree = None,
            restore_args: ArrayRestoreArgs | RestoreArgs | None = None,
        ) -> None:
            """Initialize the ArrayRestore arguments.

            Args:
                item: Optional template array or item structure. Defaults to None.
                restore_args: Optional ``ArrayRestoreArgs`` or ``RestoreArgs``. Defaults to None.
            """
            super().__init__(item=item)
            self.restore_args = restore_args

    class Composite(CheckpointArgs):
        """Arguments for saving or restoring multiple named items via ``CompositeCheckpointHandler``."""

        def __init__(
            self,
            **items: _CheckpointArgsBase | PyTree,
        ) -> None:
            """Initialize the Composite arguments mapping.

            Args:
                **items: Named checkpoint items or ``CheckpointArgs`` instances.
            """
            super().__init__(item=None)
            self._items: dict[str, _CheckpointArgsBase | PyTree] = dict(items)

        def __getitem__(self, key: str) -> _CheckpointArgsBase | PyTree:
            """Retrieve a named composite item by key.

            Args:
                key: Item name.

            Returns:
                _CheckpointArgsBase | PyTree: Associated argument wrapper or item.
            """
            return self._items[key]

        def keys(self) -> Sequence[str]:
            """Return the registered item keys in this composite argument set.

            Returns:
                Sequence[str]: List of item keys.
            """
            return list(self._items.keys())

        def values(self) -> Sequence[_CheckpointArgsBase | PyTree]:
            """Return the values stored in this composite argument set.

            Returns:
                Sequence[_CheckpointArgsBase | PyTree]: List of item values.
            """
            return list(self._items.values())

        def items(self) -> Sequence[tuple[str, _CheckpointArgsBase | PyTree]]:
            """Return ``(key, value)`` pairs stored in this composite argument set.

            Returns:
                Sequence[tuple[str, _CheckpointArgsBase | PyTree]]: List of key-value pairs.
            """
            return list(self._items.items())

        def __len__(self) -> int:
            """Return the number of items in the composite argument set.

            Returns:
                int: Number of composite items.
            """
            return len(self._items)

    class JsonSave(CheckpointArgs):
        """Arguments for saving a JSON-serializable dictionary or value."""

        def __init__(self, item: PyTree = None) -> None:
            """Initialize the JsonSave arguments.

            Args:
                item: The JSON-compatible item to save. Defaults to None.
            """
            super().__init__(item=item)

    class JsonRestore(CheckpointArgs):
        """Arguments for restoring from a JSON checkpoint file."""

        def __init__(self, item: PyTree = None) -> None:
            """Initialize the JsonRestore arguments.

            Args:
                item: Optional template structure for JSON restoration. Defaults to None.
            """
            super().__init__(item=item)

    class ProtoSave(CheckpointArgs):
        """Arguments for saving a Protocol Buffer message."""

        def __init__(
            self,
            item: PyTree = None,
            filename: str = "proto.pb",
        ) -> None:
            """Initialize the ProtoSave arguments.

            Args:
                item: Protocol Buffer message or representation to save. Defaults to None.
                filename: Target protobuf filename inside the step directory. Defaults to ``"proto.pb"``.
            """
            super().__init__(item=item)
            self.filename = filename

    class ProtoRestore(CheckpointArgs):
        """Arguments for restoring a Protocol Buffer message."""

        def __init__(
            self,
            item: PyTree = None,
            item_type: type | None = None,
            filename: str = "proto.pb",
        ) -> None:
            """Initialize the ProtoRestore arguments.

            Args:
                item: Optional template message instance. Defaults to None.
                item_type: Optional Protocol Buffer message class. Defaults to None.
                filename: Source protobuf filename inside the step directory. Defaults to ``"proto.pb"``.
            """
            super().__init__(item=item)
            self.item_type = item_type
            self.filename = filename

    class PyTreeSave(CheckpointArgs):
        """Arguments for saving a PyTree structure."""

        def __init__(
            self,
            item: PyTree = None,
            save_args: PyTree = None,
            ocdbt_target_data_file_size: int | None = None,
        ) -> None:
            """Initialize the PyTreeSave arguments.

            Args:
                item: PyTree structure to save. Defaults to None.
                save_args: Optional PyTree of ``SaveArgs``. Defaults to None.
                ocdbt_target_data_file_size: Optional target OCDBT file size in bytes. Defaults to None.
            """
            super().__init__(item=item)
            self.save_args = save_args
            self.ocdbt_target_data_file_size = ocdbt_target_data_file_size

    class PyTreeRestore(CheckpointArgs):
        """Arguments for restoring a PyTree structure."""

        def __init__(
            self,
            item: PyTree = None,
            save_args: PyTree = None,
            restore_args: PyTree = None,
            transforms: PyTree = None,
            transforms_default_to_original: bool = True,
        ) -> None:
            """Initialize the PyTreeRestore arguments.

            Args:
                item: Optional target PyTree structure. Defaults to None.
                save_args: Optional PyTree of ``SaveArgs``. Defaults to None.
                restore_args: Optional PyTree of ``RestoreArgs``. Defaults to None.
                transforms: Optional PyTree of ``Transform`` / ``RestoreTransform`` rules. Defaults to None.
                transforms_default_to_original: Whether untransformed keys default to original values. Defaults to True.
            """
            super().__init__(item=item)
            self.save_args = save_args
            self.restore_args = restore_args
            self.transforms = transforms
            self.transforms_default_to_original = transforms_default_to_original

    class StandardSave(CheckpointArgs):
        """Arguments for standard PyTree saving operations."""

        def __init__(
            self,
            item: PyTree = None,
            save_args: PyTree = None,
        ) -> None:
            """Initialize the StandardSave arguments.

            Args:
                item: PyTree structure to save. Defaults to None.
                save_args: Optional per-leaf ``SaveArgs`` tree. Defaults to None.
            """
            super().__init__(item=item)
            self.save_args = save_args

    class StandardRestore(CheckpointArgs):
        """Arguments for standard PyTree restoration operations."""

        def __init__(
            self,
            item: PyTree = None,
            restore_args: PyTree = None,
            strict: bool = True,
        ) -> None:
            """Initialize the StandardRestore arguments.

            Args:
                item: Optional target PyTree structure to restore into. Defaults to None.
                restore_args: Optional per-leaf ``RestoreArgs`` tree. Defaults to None.
                strict: Whether to enforce strict PyTree structure validation. Defaults to True.
            """
            super().__init__(item=item)
            self.restore_args = restore_args
            self.strict = strict

    class JaxRandomKeySave(CheckpointArgs):
        """Arguments for saving a JAX PRNG random key."""

        def __init__(self, item: PyTree = None) -> None:
            """Initialize the JaxRandomKeySave arguments.

            Args:
                item: JAX PRNG key array or state to save. Defaults to None.
            """
            super().__init__(item=item)

    class JaxRandomKeyRestore(CheckpointArgs):
        """Arguments for restoring a JAX PRNG random key."""

        def __init__(self, item: PyTree = None) -> None:
            """Initialize the JaxRandomKeyRestore arguments.

            Args:
                item: Optional template JAX PRNG key. Defaults to None.
            """
            super().__init__(item=item)

    class NumpyRandomKeySave(CheckpointArgs):
        """Arguments for saving a NumPy random generator state."""

        def __init__(self, item: PyTree = None) -> None:
            """Initialize the NumpyRandomKeySave arguments.

            Args:
                item: NumPy random key or state dictionary to save. Defaults to None.
            """
            super().__init__(item=item)

    class NumpyRandomKeyRestore(CheckpointArgs):
        """Arguments for restoring a NumPy random generator state."""

        def __init__(self, item: PyTree = None) -> None:
            """Initialize the NumpyRandomKeyRestore arguments.

            Args:
                item: Optional template NumPy random state. Defaults to None.
            """
            super().__init__(item=item)

    class PyGrainCheckpointSave(CheckpointArgs):
        """Arguments for saving a ``zero_grain`` dataset or data loader iterator."""

        def __init__(self, item: PyTree = None) -> None:
            """Initialize the PyGrainCheckpointSave arguments.

            Args:
                item: ``GrainIteratorProtocol`` instance to checkpoint. Defaults to None.
            """
            super().__init__(item=item)

    class PyGrainCheckpointRestore(CheckpointArgs):
        """Arguments for restoring a ``zero_grain`` dataset or data loader iterator."""

        def __init__(self, item: PyTree = None) -> None:
            """Initialize the PyGrainCheckpointRestore arguments.

            Args:
                item: ``GrainIteratorProtocol`` instance to restore in-place. Defaults to None.
            """
            super().__init__(item=item)

    @staticmethod
    def get_registered_handler_cls(
        arg_or_cls: _CheckpointArgsBase | PyTree,
    ) -> type | None:
        """Retrieve the registered ``CheckpointHandler`` class for an args class or instance.

        Args:
            arg_or_cls: A ``CheckpointArgs`` instance or subclass.

        Returns:
            type | None: The registered handler class, or ``None`` if not registered.
        """
        return _CATALOG.get_handler_cls(arg_or_cls)

    @staticmethod
    def get_registered_args_cls(
        handler_or_cls: PyTree,
        for_save: bool = True,
    ) -> _RegisteredArgsCls | None:
        """Retrieve the registered ``CheckpointArgs`` class for a handler.

        Args:
            handler_or_cls: A ``CheckpointHandler`` instance or class.
            for_save: If True, look up save args; otherwise look up restore args. Defaults to True.

        Returns:
            _RegisteredArgsCls | None: The registered ``CheckpointArgs`` class, or ``None`` if not registered.
        """
        return _CATALOG.get_args_cls(handler_or_cls, for_save=for_save)

    @staticmethod
    def has_registered_args(
        handler_or_cls: PyTree,
    ) -> bool:
        """Check whether a handler class or args class has registered arguments.

        Args:
            handler_or_cls: Handler class/instance or args class to check.

        Returns:
            bool: True if registered for save or restore, False otherwise.
        """
        return _CATALOG.has_registered(handler_or_cls)

    @staticmethod
    def register_with_handler(
        handler_cls: type,
        for_save: bool = False,
        for_restore: bool = False,
    ) -> Callable[[_ArgsClsT], _ArgsClsT]:
        """Create a class decorator registering a ``CheckpointArgs`` class with a handler.

        Args:
            handler_cls: The ``CheckpointHandler`` class to associate with the decorated args class.
            for_save: Whether to register the decorated class for save operations. Defaults to False.
            for_restore: Whether to register the decorated class for restore operations. Defaults to False.

        Returns:
            Callable[[_ArgsClsT], _ArgsClsT]: Decorator that registers and returns the class unchanged.
        """

        def _decorator(target_cls: _ArgsClsT) -> _ArgsClsT:
            """Register ``target_cls`` in the catalog handler/args registries.

            Args:
                target_cls: The argument class being decorated.

            Returns:
                _ArgsClsT: The unmodified argument class.
            """
            _CATALOG.register(
                target_cls,
                handler_cls,
                for_save=for_save,
                for_restore=for_restore,
            )
            return target_cls

        return _decorator


class checkpoint_args:
    """Namespace exporting ``args`` and ``CheckpointArgs``."""

    CheckpointArgs = args.CheckpointArgs
    register_with_handler = staticmethod(args.register_with_handler)
    get_registered_handler_cls = staticmethod(args.get_registered_handler_cls)
    get_registered_args_cls = staticmethod(args.get_registered_args_cls)
    has_registered_args = staticmethod(args.has_registered_args)


def _unwrap_checkpoint_arg(
    item: PyTree,
    kwargs: Mapping[str, PyTree],
) -> PyTree:
    """Extract the underlying payload from ``item`` or ``kwargs['args']``.

    Args:
        item: Positional item or ``_CheckpointArgsBase`` wrapper.
        kwargs: Keyword arguments mapping potentially containing ``args``.

    Returns:
        PyTree: Extracted payload.
    """
    if isinstance(item, _CheckpointArgsBase):
        return item.item
    if item is None and "args" in kwargs:
        arg_obj = kwargs["args"]
        return arg_obj.item if isinstance(arg_obj, _CheckpointArgsBase) else arg_obj
    return item


_CompositeArgsCls = args.Composite
_PyTreeSaveArgsCls = args.PyTreeSave
_PyTreeRestoreArgsCls = args.PyTreeRestore
_StandardSaveArgsCls = args.StandardSave
_StandardRestoreArgsCls = args.StandardRestore
_ArrayRestoreArgsWrapperCls = args.ArrayRestore
_ProtoSaveArgsCls = args.ProtoSave
_ProtoRestoreArgsCls = args.ProtoRestore

__all__ = [
    "_ArrayRestoreArgsWrapperCls",
    "_CATALOG",
    "_CheckpointArgsBase",
    "_CompositeArgsCls",
    "_ProtoRestoreArgsCls",
    "_ProtoSaveArgsCls",
    "_PyTreeRestoreArgsCls",
    "_PyTreeSaveArgsCls",
    "_RegisteredArgsCls",
    "_StandardRestoreArgsCls",
    "_StandardSaveArgsCls",
    "_unwrap_checkpoint_arg",
    "args",
    "checkpoint_args",
]
