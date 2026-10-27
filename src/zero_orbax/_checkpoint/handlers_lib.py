"""Checkpoint handlers and handler registries for zero_orbax.checkpoint.handlers."""

from __future__ import annotations

import json
from collections.abc import Mapping

from ml_switcheroo_compiler.core.tensor import Tensor as CompilerTensor
from ml_switcheroo_ir import PartitionSpec
from zero_flax.nnx.module import (
    GraphDef as FlaxGraphDef,
    Module as FlaxModule,
    merge as _flax_merge,
    split as _flax_split,
)
from zero_jax import numpy as jnp

from zero_orbax._checkpoint.args_lib import (
    _ArrayRestoreArgsWrapperCls,
    _CATALOG,
    _CheckpointArgsBase,
    _CompositeArgsCls,
    _ProtoRestoreArgsCls,
    _ProtoSaveArgsCls,
    _PyTreeRestoreArgsCls,
    _PyTreeSaveArgsCls,
    _StandardRestoreArgsCls,
    _StandardSaveArgsCls,
    _unwrap_checkpoint_arg,
    args,
)
from zero_orbax._checkpoint.future_lib import Future
from zero_orbax._checkpoint.transform_lib import (
    _apply_restore_args_tree,
    _apply_save_args_tree,
    _build_flax_state,
    _build_frozen_dict,
    _coerce_flax_metadata,
    _coerce_flax_variable_value,
    _extract_save_args_metadata,
    _instantiate_flax_variable,
    _pack_optax_node,
    _reconstruct_array_leaf,
    _resolve_compiler_dtype,
    _unpack_optax_node,
    apply_transformations,
)
from zero_orbax._checkpoint.type_handlers_lib import ArrayRestoreArgs, RestoreArgs, SaveArgs
from zero_orbax.core import (
    AtomicWrite,
    FileSystemIO,
    GlobalArraySharding,
    MsgpackEngine,
    SerializedArray,
    TensorStoreDriver,
)
from zero_orbax.types import (
    ArrayMetadataValidatorProtocol,
    CheckpointStateDict,
    CompositeOptions,
    FlaxStateProtocol,
    FlaxVariableProtocol,
    FrozenDictProtocol,
    GrainIteratorProtocol,
    IsPrioritizedKeyFn,
    JsonValue,
    MaskedNodeProtocol,
    MemoryLimitOptions,
    MultiprocessingOptions,
    NamedTupleProtocol,
    PathLike,
    PyTree,
    PyTreeMetadataOptions,
    TypeHandlerRegistryProtocol,
    array_to_nested_list,
    infer_array_shape as _infer_array_shape,
    is_array_protocol,
    is_pytree_dict,
    is_pytree_mapping,
)


class CheckpointHandler:
    """Base class for handlers that manage checkpoint reading, writing, and metadata."""

    def __init__(self, *args: PyTree, **kwargs: PyTree) -> None:
        """Initialize the base CheckpointHandler.

        Args:
            *args: Optional positional initialization arguments.
            **kwargs: Optional keyword initialization arguments.
        """
        _ = (args, kwargs)
        self._closed: bool = False
        self._last_metadata: dict[str, JsonValue] | None = None

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Save an item synchronously to the specified directory.

        Args:
            directory: Target checkpoint directory path. Defaults to None.
            item: The item or ``CheckpointArgs`` payload to save. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.
        """
        payload = _unwrap_checkpoint_arg(item, kwargs)
        if directory is not None:
            key = str(directory).rstrip("/")
            encoded = MsgpackEngine().packb(payload)
            AtomicWrite().atomic_write_bytes(f"{key}/checkpoint.msgpack", encoded)
            meta: dict[str, JsonValue] = {"directory": str(directory)}
            AtomicWrite().atomic_write_bytes(
                f"{key}/_orbax_metadata.json",
                json.dumps(meta).encode("utf-8"),
            )

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore an item from the specified directory.

        Args:
            directory: Source checkpoint directory path. Defaults to None.
            item: Optional target structure or ``CheckpointArgs`` wrapper. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: The restored PyTree or item structure.
        """
        template = _unwrap_checkpoint_arg(item, kwargs)
        if directory is not None:
            key = str(directory).rstrip("/")
            raw = FileSystemIO().read(f"{key}/checkpoint.msgpack")
            if raw:
                return MsgpackEngine().unpackb(raw)
        return template

    def metadata(self, directory: PathLike | None = None) -> dict[str, JsonValue] | None:
        """Return metadata associated with the checkpoint in ``directory``.

        Args:
            directory: Checkpoint directory path. Defaults to None.

        Returns:
            dict[str, JsonValue] | None: Metadata dictionary if available, otherwise None.
        """
        if directory is not None:
            key = str(directory).rstrip("/")
            raw = FileSystemIO().read(f"{key}/_orbax_metadata.json")
            if raw:
                loaded: dict[str, JsonValue] = json.loads(raw.decode("utf-8"))
                return loaded
        return self._last_metadata

    def structure(self, directory: PathLike | None = None) -> PyTree:
        """Return the structural skeleton of the saved checkpoint.

        Args:
            directory: Checkpoint directory path. Defaults to None.

        Returns:
            PyTree: Structure of the saved item, or None.
        """
        if directory is not None:
            return self.restore(directory)
        return None

    def finalize(self, directory: PathLike | None = None) -> None:
        """Finalize the checkpoint directory after writing completes.

        Args:
            directory: Checkpoint directory path to finalize. Defaults to None.
        """
        if directory is not None:
            key = str(directory).rstrip("/")
            meta: dict[str, JsonValue] = {"finalized": True, "directory": str(directory)}
            self._last_metadata = meta
            AtomicWrite().atomic_write_bytes(
                f"{key}/_orbax_metadata.json",
                json.dumps(meta).encode("utf-8"),
            )

    def close(self) -> None:
        """Release any resources held by the handler."""
        self._closed = True


class AsyncCheckpointHandler(CheckpointHandler):
    """Base class for checkpoint handlers supporting asynchronous save operations."""

    def __init__(self, *args: PyTree, **kwargs: PyTree) -> None:
        """Initialize the base AsyncCheckpointHandler.

        Args:
            *args: Optional positional initialization arguments.
            **kwargs: Optional keyword initialization arguments.
        """
        super().__init__(*args, **kwargs)

    def async_save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> list[Future] | None:
        """Initiate an asynchronous save operation and return commit futures.

        Args:
            directory: Target checkpoint directory path. Defaults to None.
            item: The item or ``CheckpointArgs`` to save. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.

        Returns:
            list[Future] | None: List of futures representing pending commit tasks, or None.
        """
        if directory is None:
            return None
        self.save(directory, item, *pos_args, **kwargs)
        return [Future(result=None, commit_callback=lambda: self.finalize(directory))]


class ArrayCheckpointHandler(AsyncCheckpointHandler):
    """Handler for saving and restoring single arrays or sharded tensors."""

    def __init__(
        self,
        checkpoint_name: str | None = None,
        *,
        multiprocessing_options: MultiprocessingOptions | None = None,
    ) -> None:
        """Initialize the ArrayCheckpointHandler.

        Args:
            checkpoint_name: Optional filename for the serialized array inside the step directory. Defaults to None.
            multiprocessing_options: Optional ``MultiprocessingOptions``. Defaults to None.
        """
        super().__init__()
        self.checkpoint_name = checkpoint_name
        self.multiprocessing_options = multiprocessing_options

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Save an array to ``directory`` using ``TensorStoreDriver``.

        Args:
            directory: Target directory path. Defaults to None.
            item: Array or ``args.ArraySave`` instance to save. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.
        """
        if directory is not None:
            name = self.checkpoint_name or "array"
            driver = TensorStoreDriver()
            driver.open({"driver": "zarr", "path": f"{str(directory).rstrip('/')}/{name}.ts"})
            payload = _unwrap_checkpoint_arg(item, kwargs)
            driver.write(MsgpackEngine().packb(payload))

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        restore_args: ArrayRestoreArgs | RestoreArgs | PyTree = None,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore an array from ``directory`` using ``TensorStoreDriver``.

        Args:
            directory: Source directory path. Defaults to None.
            item: Optional template array or ``args.ArrayRestore`` wrapper. Defaults to None.
            *pos_args: Additional positional restore arguments.
            restore_args: Optional ``ArrayRestoreArgs`` or ``RestoreArgs``. Defaults to None.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: Restored array or template value.

        Raises:
            ValueError: If ``strict=True`` on ``ArrayRestoreArgs`` and the restored shape mismatches.
        """
        active_restore_args = restore_args
        raw_arg = item if isinstance(item, _CheckpointArgsBase) else kwargs.get("args")
        if isinstance(raw_arg, _ArrayRestoreArgsWrapperCls) and active_restore_args is None:
            active_restore_args = raw_arg.restore_args
        template = _unwrap_checkpoint_arg(item, kwargs)
        if directory is not None:
            name = self.checkpoint_name or "array"
            ts_path = f"{str(directory).rstrip('/')}/{name}.ts"
            driver = TensorStoreDriver()
            driver.open({"driver": "zarr", "path": ts_path})
            raw = driver.read()
            if raw:
                restored = MsgpackEngine().unpackb(raw)
                if isinstance(active_restore_args, ArrayRestoreArgs):
                    rtype = active_restore_args.restore_type()
                    if (
                        active_restore_args.sharding is not None
                        or rtype is jnp.ndarray
                        or rtype == "jax"
                    ) and not isinstance(restored, jnp.ndarray):
                        nested_sa = (
                            restored.tolist() if isinstance(restored, SerializedArray) else restored
                        )
                        raw_sa = (
                            nested_sa
                            if isinstance(nested_sa, (int, float, bool, list, tuple))
                            else 0.0
                        )
                        dt_name = (
                            str(restored.dtype)
                            if isinstance(restored, SerializedArray)
                            else str(active_restore_args.dtype or "float32")
                        )
                        restored = jnp.asarray(raw_sa, dtype=_resolve_compiler_dtype(dt_name))
                    expected_shape = active_restore_args.shape or active_restore_args.global_shape
                    if active_restore_args.strict and expected_shape is not None:
                        actual_shape = _infer_array_shape(restored)
                        if actual_shape != expected_shape:
                            raise ValueError(
                                f"Strict shape mismatch in ArrayCheckpointHandler: expected "
                                f"{expected_shape}, got {actual_shape}."
                            )
                    if (
                        active_restore_args.mesh is not None
                        and active_restore_args.mesh_axes is not None
                    ):
                        pspec = (
                            active_restore_args.mesh_axes
                            if not isinstance(active_restore_args.mesh_axes, (list, tuple))
                            else PartitionSpec(axes=tuple(active_restore_args.mesh_axes))
                        )
                        restored = GlobalArraySharding().get_local_shard(
                            restored,
                            host_id=0,
                            num_hosts=1,
                            partition_spec=pspec,
                            mesh=active_restore_args.mesh,
                            sharding=active_restore_args.sharding
                            if not isinstance(active_restore_args.sharding, (PartitionSpec, str))
                            else None,
                        )
                    elif active_restore_args.sharding is not None:
                        restored = GlobalArraySharding().apply_sharding(
                            restored,
                            sharding=active_restore_args.sharding,
                            mesh=active_restore_args.mesh,
                            mesh_axes=active_restore_args.mesh_axes,
                        )
                    if (
                        active_restore_args.strict
                        and (
                            active_restore_args.sharding is not None
                            or active_restore_args.mesh is not None
                        )
                        and isinstance(restored, jnp.ndarray)
                    ):
                        import zero_chex as _chex

                        _chex.assert_tree_is_sharded(restored)
                    if rtype is tuple and isinstance(restored, list):
                        return tuple(restored)
                return restored
            return super().restore(directory, template, *pos_args, **kwargs)
        return template


class BasePyTreeCheckpointHandler(AsyncCheckpointHandler):
    """Base checkpoint handler for structured PyTree serialization."""

    def __init__(
        self,
        *,
        save_concurrent_bytes: int | None = None,
        restore_concurrent_bytes: int | None = None,
        save_device_host_concurrent_bytes: int | None = None,
        memory_limit_options: MemoryLimitOptions | None = None,
        use_ocdbt: bool = True,
        use_zarr3: bool = False,
        use_compression: bool = True,
        multiprocessing_options: MultiprocessingOptions | None = None,
        type_handler_registry: TypeHandlerRegistryProtocol | None = None,
        enable_post_merge_validation: bool = True,
        pytree_metadata_options: PyTreeMetadataOptions | None = None,
        array_metadata_validator: ArrayMetadataValidatorProtocol | None = None,
        enable_pinned_host_transfer: bool | None = None,
        is_prioritized_key_fn: IsPrioritizedKeyFn | None = None,
    ) -> None:
        """Initialize the BasePyTreeCheckpointHandler.

        Args:
            save_concurrent_bytes: Maximum concurrent bytes during save. Defaults to None.
            restore_concurrent_bytes: Maximum concurrent bytes during restore. Defaults to None.
            save_device_host_concurrent_bytes: Max concurrent device-to-host bytes. Defaults to None.
            memory_limit_options: Optional ``MemoryLimitOptions`` configuration. Defaults to None.
            use_ocdbt: Whether to use OCDBT storage driver. Defaults to True.
            use_zarr3: Whether to use Zarr v3 format. Defaults to False.
            use_compression: Whether to compress stored chunks. Defaults to True.
            multiprocessing_options: Optional ``MultiprocessingOptions``. Defaults to None.
            type_handler_registry: Optional leaf ``TypeHandlerRegistryProtocol``. Defaults to None.
            enable_post_merge_validation: Whether to validate shards after OCDBT merge. Defaults to True.
            pytree_metadata_options: Optional ``PyTreeMetadataOptions``. Defaults to None.
            array_metadata_validator: Optional ``ArrayMetadataValidatorProtocol``. Defaults to None.
            enable_pinned_host_transfer: Whether to use pinned host memory transfers. Defaults to None.
            is_prioritized_key_fn: Optional callback prioritizing specific key paths. Defaults to None.
        """
        super().__init__()
        self.save_concurrent_bytes = save_concurrent_bytes
        self.restore_concurrent_bytes = restore_concurrent_bytes
        self.save_device_host_concurrent_bytes = save_device_host_concurrent_bytes
        self.memory_limit_options = memory_limit_options
        self.use_ocdbt = use_ocdbt
        self.use_zarr3 = use_zarr3
        self.use_compression = use_compression
        self.multiprocessing_options = multiprocessing_options
        self.type_handler_registry = type_handler_registry
        self.enable_post_merge_validation = enable_post_merge_validation
        self.pytree_metadata_options = pytree_metadata_options
        self.array_metadata_validator = array_metadata_validator
        self.enable_pinned_host_transfer = enable_pinned_host_transfer
        self.is_prioritized_key_fn = is_prioritized_key_fn

    def _pack_pytree(self, node: PyTree) -> PyTree:
        """Pack rich ecosystem PyTree nodes into tagged serializable dictionaries.

        Args:
            node: Input PyTree node (``Module``, ``Variable``, ``State``, ``FrozenDict``, ``OptState``, array, or container).

        Returns:
            PyTree: Tagged PyTree ready for MessagePack serialization.
        """
        if isinstance(node, FlaxModule):
            _, module_state = _flax_split(node)
            return {
                "__zero_flax_module__": True,
                "module_type": type(node).__name__,
                "state": self._pack_pytree(module_state),
            }
        if isinstance(node, FlaxVariableProtocol):
            return {
                "__zero_flax_variable__": True,
                "var_type": type(node).__name__,
                "value": self._pack_pytree(node.value),
                "metadata": self._pack_pytree(dict(node.metadata)),
            }
        if isinstance(node, FlaxStateProtocol):
            return {
                "__zero_flax_state__": True,
                "entries": {str(k): self._pack_pytree(v) for k, v in node.items()},
            }
        if isinstance(node, FrozenDictProtocol):
            return {
                "__zero_flax_frozendict__": True,
                "entries": {str(k): self._pack_pytree(v) for k, v in node.items()},
            }
        if isinstance(node, (MaskedNodeProtocol, NamedTupleProtocol)):
            return _pack_optax_node(node, self._pack_pytree)
        if is_array_protocol(node):
            array_kind = (
                "tensor"
                if isinstance(node, CompilerTensor)
                else ("serialized_array" if isinstance(node, SerializedArray) else "jax_array")
            )
            return {
                "__zero_orbax_array__": True,
                "array_kind": array_kind,
                "shape": [int(d) for d in node.shape],
                "dtype": str(node.dtype),
                "data": array_to_nested_list(node),
            }
        if isinstance(node, tuple):
            return tuple(self._pack_pytree(item) for item in node)
        if isinstance(node, list):
            return [self._pack_pytree(item) for item in node]
        if is_pytree_mapping(node):
            return {str(k): self._pack_pytree(v) for k, v in node.items()}
        return node

    def _merge_flax_template(
        self,
        entries: Mapping[str, PyTree],
        template: FlaxModule | FlaxGraphDef[FlaxModule],
    ) -> PyTree:
        """Unpack state entries and merge into a ``FlaxModule`` or ``FlaxGraphDef`` template.

        Args:
            entries: Raw or unpacked state entries dictionary.
            template: Target ``FlaxModule`` or ``FlaxGraphDef`` blueprint.

        Returns:
            PyTree: Reconstructed ``FlaxModule`` instance.
        """
        if isinstance(template, FlaxModule):
            gdef, tmpl_state = _flax_split(template)
            tmpl_map = dict(tmpl_state.items())
            ordered = [*tmpl_map.keys(), *(k for k in entries if k not in tmpl_map)]
            unpacked = {
                str(k): self._unpack_pytree(entries[k], tmpl_map.get(str(k)))
                for k in ordered
                if k in entries
            }
            return _flax_merge(gdef, _build_flax_state(unpacked, tmpl_state))
        unpacked_gdef = {str(k): self._unpack_pytree(v, None) for k, v in entries.items()}
        return _flax_merge(template, _build_flax_state(unpacked_gdef, None))

    def _unpack_pytree(self, node: PyTree, template: PyTree = None) -> PyTree:
        """Unpack tagged PyTree dictionaries back into rich ecosystem containers and arrays.

        Args:
            node: Deserialized PyTree node.
            template: Optional target template guiding container and array reconstruction. Defaults to None.

        Returns:
            PyTree: Reconstructed ecosystem PyTree.
        """
        if isinstance(node, FlaxStateProtocol):
            if isinstance(template, (FlaxModule, FlaxGraphDef)):
                return self._merge_flax_template(dict(node.items()), template)
            tmpl_map = (
                dict(template.items())
                if isinstance(template, FlaxStateProtocol)
                else (template if is_pytree_dict(template) else {})
            )
            return _build_flax_state(
                {str(k): self._unpack_pytree(v, tmpl_map.get(str(k))) for k, v in node.items()},
                template,
            )
        if is_pytree_dict(node):
            if node.get("__zero_flax_module__") is True:
                return self._unpack_pytree(node.get("state"), template)
            if (
                node.get("__zero_flax_variable__") is True
                or node.get("__ztype__") == "flax_variable"
            ):
                var_type = str(node.get("var_type", "Variable"))
                tmpl_val = template.value if isinstance(template, FlaxVariableProtocol) else None
                val = self._unpack_pytree(node.get("value"), tmpl_val)
                raw_meta = self._unpack_pytree(node.get("metadata"), None)
                coerced_val = _coerce_flax_variable_value(val)
                coerced_meta = _coerce_flax_metadata(raw_meta)
                if isinstance(template, FlaxVariableProtocol):
                    return template.replace(value=coerced_val, **coerced_meta)
                return _instantiate_flax_variable(var_type, coerced_val, coerced_meta)
            if node.get("__zero_flax_state__") is True or node.get("__ztype__") == "flax_state":
                raw_entries = node.get("entries")
                entries_map = raw_entries if is_pytree_dict(raw_entries) else {}
                if isinstance(template, (FlaxModule, FlaxGraphDef)):
                    return self._merge_flax_template(entries_map, template)
                tmpl_map = (
                    dict(template.items())
                    if isinstance(template, FlaxStateProtocol)
                    else (template if is_pytree_dict(template) else {})
                )
                ordered_keys = [*tmpl_map.keys(), *(k for k in entries_map if k not in tmpl_map)]
                unpacked_entries = {
                    str(k): self._unpack_pytree(entries_map[k], tmpl_map.get(str(k)))
                    for k in ordered_keys
                    if k in entries_map
                }
                return _build_flax_state(unpacked_entries, template)
            if (
                node.get("__zero_flax_frozendict__") is True
                or node.get("__ztype__") == "flax_frozendict"
            ):
                raw_entries = node.get("entries")
                entries_map = raw_entries if is_pytree_dict(raw_entries) else {}
                tmpl_map = (
                    dict(template.items())
                    if isinstance(template, FrozenDictProtocol)
                    else (template if is_pytree_dict(template) else {})
                )
                ordered_keys = [*tmpl_map.keys(), *(k for k in entries_map if k not in tmpl_map)]
                unpacked_entries = {
                    str(k): self._unpack_pytree(entries_map[k], tmpl_map.get(str(k)))
                    for k in ordered_keys
                    if k in entries_map
                }
                return _build_frozen_dict(unpacked_entries)
            if (
                node.get("__optax_masked_node__") is True
                or isinstance(node.get("__optax_state_cls__"), str)
                or node.get("__zero_optax_opt_state__") is True
                or node.get("__ztype__") == "optax_opt_state"
            ):
                return _unpack_optax_node(node, template, self._unpack_pytree)
            if node.get("__zero_orbax_array__") is True or node.get("__ztype__") == "array":
                raw_shape = node.get("shape")
                shape_tuple = (
                    tuple(int(d) for d in raw_shape if isinstance(d, int))
                    if isinstance(raw_shape, (list, tuple))
                    else ()
                )
                dtype_str = str(node.get("dtype", "float32"))
                array_kind = str(node.get("array_kind", "jax_array"))
                return _reconstruct_array_leaf(
                    node.get("data"),
                    shape_tuple,
                    dtype_str,
                    array_kind,
                    template,
                )
            if isinstance(template, (FlaxModule, FlaxGraphDef)):
                return self._merge_flax_template(node, template)
            if isinstance(template, FlaxStateProtocol):
                tmpl_map = dict(template.items())
                unpacked_s = {
                    str(k): self._unpack_pytree(v, tmpl_map.get(str(k))) for k, v in node.items()
                }
                return _build_flax_state(unpacked_s, template)
            if isinstance(template, FrozenDictProtocol):
                tmpl_map = dict(template.items())
                unpacked_fd = {
                    str(k): self._unpack_pytree(v, tmpl_map.get(str(k))) for k, v in node.items()
                }
                return _build_frozen_dict(unpacked_fd)
            if isinstance(template, NamedTupleProtocol):
                tmpl_map = dict(template._asdict())
                vals = [self._unpack_pytree(node.get(f), tmpl_map.get(f)) for f in template._fields]
                return template._make(vals)
            tmpl_dict = template if is_pytree_dict(template) else {}
            return {k: self._unpack_pytree(v, tmpl_dict.get(k)) for k, v in node.items()}
        if isinstance(node, SerializedArray):
            return _reconstruct_array_leaf(
                node.tolist(),
                node.shape,
                str(node.dtype),
                "jax_array",
                template,
            )
        if isinstance(node, tuple):
            tmpl_seq = template if isinstance(template, (list, tuple)) else ()
            return tuple(
                self._unpack_pytree(item, tmpl_seq[idx] if idx < len(tmpl_seq) else None)
                for idx, item in enumerate(node)
            )
        if isinstance(node, list):
            tmpl_seq = template if isinstance(template, (list, tuple)) else ()
            return [
                self._unpack_pytree(item, tmpl_seq[idx] if idx < len(tmpl_seq) else None)
                for idx, item in enumerate(node)
            ]
        return node

    def _extract_structure(self, node: PyTree) -> PyTree:
        """Extract the unpacked structural tree of a saved PyTree payload.

        Args:
            node: Packed or unpacked PyTree node.

        Returns:
            PyTree: Reconstructed structural tree.
        """
        return self._unpack_pytree(node, None)

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Save a PyTree to ``directory`` after packing rich ecosystem nodes.

        Args:
            directory: Target checkpoint directory path. Defaults to None.
            item: PyTree or ``CheckpointArgs`` payload to save. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.
        """
        payload = _unwrap_checkpoint_arg(item, kwargs)
        packed = self._pack_pytree(payload)
        super().save(directory, packed, *pos_args, **kwargs)

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore and unpack a PyTree from ``directory``.

        Args:
            directory: Source checkpoint directory path. Defaults to None.
            item: Optional target template or ``CheckpointArgs`` wrapper. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: Unpacked PyTree structure or fallback template.
        """
        template = _unwrap_checkpoint_arg(item, kwargs)
        sentinel = "__zero_orbax_missing__"
        raw_restored = super().restore(directory, sentinel, *pos_args, **kwargs)
        if raw_restored == sentinel:
            return template
        return self._unpack_pytree(raw_restored, template)

    def structure(self, directory: PathLike | None = None) -> PyTree:
        """Return the structural skeleton of the saved PyTree checkpoint.

        Args:
            directory: Checkpoint directory path. Defaults to None.

        Returns:
            PyTree: Extracted PyTree structure, or None.
        """
        return self._extract_structure(super().structure(directory))

    def get_param_names(self, item: PyTree) -> PyTree:
        """Derive dotted parameter names for each leaf in a PyTree dictionary.

        Args:
            item: PyTree mapping to extract parameter names from.

        Returns:
            PyTree: Nested dictionary of dotted parameter names, or ``None`` if ``item`` is not a mapping.
        """
        if not is_pytree_mapping(item):
            return None

        def _build_names(subtree: Mapping[str, PyTree], prefix: str) -> dict[str, PyTree]:
            """Recursively construct dotted parameter names for a mapping subtree.

            Args:
                subtree: Current dictionary subtree.
                prefix: Accumulated dotted key prefix.

            Returns:
                dict[str, PyTree]: Nested dictionary of parameter name strings.
            """
            result: dict[str, PyTree] = {}
            for k, v in subtree.items():
                full_key = f"{prefix}.{k}" if prefix else str(k)
                if is_pytree_mapping(v):
                    result[str(k)] = _build_names(v, full_key)
                else:
                    result[str(k)] = full_key
            return result

        return _build_names(item, "")


class PyTreeCheckpointHandler(BasePyTreeCheckpointHandler):
    """Handler for saving and restoring PyTree structures with transformations."""

    def __init__(
        self,
        aggregate_filename: str | None = None,
        use_ocdbt: bool = True,
        use_zarr3: bool = False,
        primary_host: int | None = 0,
        type_handler_registry: TypeHandlerRegistryProtocol | None = None,
        handler_impl: BasePyTreeCheckpointHandler | None = None,
        save_concurrent_gb: int | None = None,
        restore_concurrent_gb: int | None = None,
        *,
        save_device_host_concurrent_gb: int | None = None,
        memory_limit_options: MemoryLimitOptions | None = None,
        use_compression: bool = True,
        multiprocessing_options: MultiprocessingOptions | None = None,
        pytree_metadata_options: PyTreeMetadataOptions | None = None,
        array_metadata_validator: ArrayMetadataValidatorProtocol | None = None,
        enable_pinned_host_transfer: bool | None = None,
        is_prioritized_key_fn: IsPrioritizedKeyFn | None = None,
    ) -> None:
        """Initialize the PyTreeCheckpointHandler.

        Args:
            aggregate_filename: Optional filename for aggregated msgpack checkpoints. Defaults to None.
            use_ocdbt: Whether to enable OCDBT storage. Defaults to True.
            use_zarr3: Whether to enable Zarr v3 storage. Defaults to False.
            primary_host: Primary host ID for metadata writes. Defaults to 0.
            type_handler_registry: Optional ``TypeHandlerRegistryProtocol``. Defaults to None.
            handler_impl: Optional custom ``BasePyTreeCheckpointHandler`` delegate. Defaults to None.
            save_concurrent_gb: Max concurrent GB during save. Defaults to None.
            restore_concurrent_gb: Max concurrent GB during restore. Defaults to None.
            save_device_host_concurrent_gb: Max concurrent device-to-host GB. Defaults to None.
            memory_limit_options: Optional ``MemoryLimitOptions``. Defaults to None.
            use_compression: Whether to enable chunk compression. Defaults to True.
            multiprocessing_options: Optional ``MultiprocessingOptions``. Defaults to None.
            pytree_metadata_options: Optional ``PyTreeMetadataOptions``. Defaults to None.
            array_metadata_validator: Optional ``ArrayMetadataValidatorProtocol``. Defaults to None.
            enable_pinned_host_transfer: Whether to enable pinned host memory transfer. Defaults to None.
            is_prioritized_key_fn: Optional key prioritization function. Defaults to None.
        """
        resolved_mp = multiprocessing_options or MultiprocessingOptions(primary_host=primary_host)
        super().__init__(
            memory_limit_options=memory_limit_options,
            use_ocdbt=use_ocdbt,
            use_zarr3=use_zarr3,
            use_compression=use_compression,
            multiprocessing_options=resolved_mp,
            type_handler_registry=type_handler_registry,
            pytree_metadata_options=pytree_metadata_options,
            array_metadata_validator=array_metadata_validator,
            enable_pinned_host_transfer=enable_pinned_host_transfer,
            is_prioritized_key_fn=is_prioritized_key_fn,
        )
        self.aggregate_filename = aggregate_filename
        self.primary_host = primary_host
        self.save_concurrent_gb = save_concurrent_gb
        self.restore_concurrent_gb = restore_concurrent_gb
        self.save_device_host_concurrent_gb = save_device_host_concurrent_gb
        self.handler_impl = handler_impl

    def _order_prioritized_keys(self, node: PyTree) -> PyTree:
        """Order dictionary keys placing prioritized keys first when ``self.is_prioritized_key_fn`` is set.

        Args:
            node: PyTree node to order.

        Returns:
            PyTree: Ordered PyTree structure.
        """
        if self.is_prioritized_key_fn is None:
            return node
        if isinstance(node, list):
            return [self._order_prioritized_keys(v) for v in node]
        if isinstance(node, tuple):
            return tuple(self._order_prioritized_keys(v) for v in node)
        if not is_pytree_dict(node):
            return node
        prio_fn = self.is_prioritized_key_fn
        sorted_keys = sorted(node.keys(), key=lambda k: (0 if prio_fn((str(k),)) else 1, k))
        return {k: self._order_prioritized_keys(node[k]) for k in sorted_keys}

    def _encode_custom_leaves(self, node: PyTree) -> PyTree:
        """Encode leaves registered in ``self.type_handler_registry`` before packing.

        Args:
            node: PyTree node to inspect.

        Returns:
            PyTree: Tree with custom leaves converted to tagged dictionaries.
        """
        if self.type_handler_registry is None or node is None:
            return node
        if is_pytree_dict(node):
            return {k: self._encode_custom_leaves(v) for k, v in node.items()}
        if isinstance(node, list):
            return [self._encode_custom_leaves(v) for v in node]
        if isinstance(node, tuple):
            return tuple(self._encode_custom_leaves(v) for v in node)
        th = self.type_handler_registry.get(type(node))
        if th is not None:
            return {
                "__custom_type__": th.typestr(),
                "__custom_hex__": th.serialize(node).hex(),
            }
        return node

    def _decode_custom_leaves(self, node: PyTree) -> PyTree:
        """Decode leaves tagged with ``__custom_type__`` using ``self.type_handler_registry``.

        Args:
            node: Restored PyTree node to inspect.

        Returns:
            PyTree: Tree with custom leaves deserialized via their registered ``TypeHandlerProtocol``.
        """
        if self.type_handler_registry is None or node is None:
            return node
        if isinstance(node, list):
            return [self._decode_custom_leaves(v) for v in node]
        if isinstance(node, tuple):
            return tuple(self._decode_custom_leaves(v) for v in node)
        if is_pytree_dict(node):
            custom_type = node.get("__custom_type__")
            custom_hex = node.get("__custom_hex__")
            if isinstance(custom_type, str) and isinstance(custom_hex, str):
                th = self.type_handler_registry.get_by_typestr(custom_type)
                if th is not None:
                    return th.deserialize(bytes.fromhex(custom_hex))
            return {k: self._decode_custom_leaves(v) for k, v in node.items()}
        return node

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        save_args: PyTree = None,
        **kwargs: PyTree,
    ) -> None:
        """Save a PyTree to ``directory``, respecting ``handler_impl``, ``save_args``, and ``aggregate_filename``.

        Args:
            directory: Target checkpoint directory. Defaults to None.
            item: PyTree or ``args.PyTreeSave`` / ``args.StandardSave`` wrapper to save. Defaults to None.
            *pos_args: Additional positional save arguments.
            save_args: Optional PyTree of ``SaveArgs``. Defaults to None.
            **kwargs: Additional keyword save arguments (including optional ``check_finite``).
        """
        if self.handler_impl is not None:
            self.handler_impl.save(directory, item, *pos_args, **kwargs)
            return
        active_save_args = save_args
        active_check_finite: bool = kwargs.get("check_finite") is True
        raw_arg = item if isinstance(item, _CheckpointArgsBase) else kwargs.get("args")
        if isinstance(raw_arg, (_PyTreeSaveArgsCls, _StandardSaveArgsCls)):
            if active_save_args is None:
                active_save_args = raw_arg.save_args
            active_check_finite = active_check_finite or raw_arg.check_finite
        payload = _unwrap_checkpoint_arg(item, kwargs)
        if active_check_finite:
            import zero_chex as _chex

            check_target: PyTree = (
                _flax_split(payload)[1] if isinstance(payload, FlaxModule) else payload
            )
            _chex.assert_trees_all_finite(check_target)
        if active_save_args is not None:
            payload = _apply_save_args_tree(payload, active_save_args)
        payload = self._order_prioritized_keys(payload)
        if (
            self.array_metadata_validator is not None
            and self.enable_post_merge_validation
            and is_pytree_mapping(payload)
        ):
            meta_snapshot: dict[str, JsonValue] = {
                str(k): str(type(v).__name__) for k, v in payload.items()
            }
            self.array_metadata_validator.validate(meta_snapshot)
        encoded_payload = self._encode_custom_leaves(payload)
        packed = self._pack_pytree(encoded_payload)
        if directory is not None:
            key = str(directory).rstrip("/")
            target_fname = self.aggregate_filename or "checkpoint.msgpack"
            encoded_bytes = MsgpackEngine().packb(packed)
            AtomicWrite().atomic_write_bytes(f"{key}/{target_fname}", encoded_bytes)
            if (
                self.aggregate_filename is not None
                and self.aggregate_filename != "checkpoint.msgpack"
            ):
                AtomicWrite().atomic_write_bytes(f"{key}/checkpoint.msgpack", encoded_bytes)
            meta: dict[str, JsonValue] = {"directory": str(directory)}
            save_args_meta = _extract_save_args_metadata(active_save_args)
            if save_args_meta is not None:
                meta["save_args"] = save_args_meta
            AtomicWrite().atomic_write_bytes(
                f"{key}/_orbax_metadata.json",
                json.dumps(meta).encode("utf-8"),
            )

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        transforms: PyTree = None,
        transforms_default_to_original: bool | PyTree = True,
        restore_args: PyTree = None,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore a PyTree from ``directory``, optionally applying transformations.

        Args:
            directory: Source checkpoint directory. Defaults to None.
            item: Target structure or ``args.PyTreeRestore`` / ``args.StandardRestore``. Defaults to None.
            *pos_args: Additional positional restore arguments.
            transforms: Optional transformation rules PyTree. Defaults to None.
            transforms_default_to_original: Whether untransformed fields fall back to original values. Defaults to True.
            restore_args: Optional tree of ``RestoreArgs``. Defaults to None.
            **kwargs: Additional keyword restore arguments (including optional ``strict``).

        Returns:
            PyTree: The restored (and optionally transformed) PyTree.
        """
        if self.handler_impl is not None:
            return self.handler_impl.restore(directory, item, *pos_args, **kwargs)
        default_orig = bool(transforms_default_to_original)
        active_restore_args = restore_args
        kw_strict = kwargs.get("strict")
        strict_override: bool | None = kw_strict if isinstance(kw_strict, bool) else None
        explicit_strict: bool = strict_override is True
        active_strict: bool = True if strict_override is None else strict_override
        raw_arg = item if isinstance(item, _CheckpointArgsBase) else kwargs.get("args")
        if isinstance(raw_arg, _PyTreeRestoreArgsCls):
            transforms = raw_arg.transforms if transforms is None else transforms
            default_orig = raw_arg.transforms_default_to_original
            active_restore_args = (
                raw_arg.restore_args if active_restore_args is None else active_restore_args
            )
            if strict_override is None:
                active_strict = raw_arg.strict
            target_item = raw_arg.item
        elif isinstance(raw_arg, _StandardRestoreArgsCls):
            active_restore_args = (
                raw_arg.restore_args if active_restore_args is None else active_restore_args
            )
            if strict_override is None:
                active_strict = raw_arg.strict
            target_item = raw_arg.item
            explicit_strict = active_strict
        else:
            target_item = _unwrap_checkpoint_arg(item, kwargs)

        restored: PyTree = None
        if directory is not None and self.aggregate_filename is not None:
            agg_path = f"{str(directory).rstrip('/')}/{self.aggregate_filename}"
            raw_agg = FileSystemIO().read(agg_path)
            if raw_agg:
                restored = self._unpack_pytree(MsgpackEngine().unpackb(raw_agg), target_item)
        if restored is None:
            restored = super().restore(directory, target_item, *pos_args, **kwargs)
        if restored is not None:
            restored = self._decode_custom_leaves(restored)
        if active_restore_args is not None and restored is not None:
            restored = _apply_restore_args_tree(restored, active_restore_args)
        if transforms is not None and target_item is not None and restored is not None:
            return apply_transformations(
                restored,
                transforms,
                target_item,
                default_to_original=default_orig,
            )
        if (
            transforms is None
            and active_strict
            and isinstance(
                target_item,
                (FlaxStateProtocol, FrozenDictProtocol, NamedTupleProtocol),
            )
            and isinstance(
                restored,
                (FlaxStateProtocol, FrozenDictProtocol, NamedTupleProtocol),
            )
        ):
            import zero_chex as _chex

            _chex.assert_trees_all_equal_structs(restored, target_item)
            if explicit_strict and active_restore_args is None:
                _chex.assert_trees_all_equal_shapes_and_dtypes(restored, target_item)
        elif (
            transforms is None
            and explicit_strict
            and active_restore_args is None
            and target_item is not None
            and restored is not None
            and not isinstance(target_item, (FlaxModule, FlaxGraphDef))
        ):
            import zero_chex as _chex

            _chex.assert_trees_all_equal_shapes_and_dtypes(restored, target_item)
        if isinstance(target_item, FlaxStateProtocol) and isinstance(restored, FlaxStateProtocol):
            target_item.update(restored)
        return restored


class StandardCheckpointHandler(PyTreeCheckpointHandler):
    """Standard handler for saving and restoring typical training state PyTrees."""

    def __init__(
        self,
        save_concurrent_gb: int | None = 96,
        restore_concurrent_gb: int | None = 96,
        *,
        multiprocessing_options: MultiprocessingOptions | None = None,
        pytree_metadata_options: PyTreeMetadataOptions | None = None,
        use_ocdbt: bool = True,
    ) -> None:
        """Initialize the StandardCheckpointHandler.

        Args:
            save_concurrent_gb: Max concurrent GB during save. Defaults to 96.
            restore_concurrent_gb: Max concurrent GB during restore. Defaults to 96.
            multiprocessing_options: Optional ``MultiprocessingOptions``. Defaults to None.
            pytree_metadata_options: Optional ``PyTreeMetadataOptions``. Defaults to None.
            use_ocdbt: Whether to use OCDBT storage format. Defaults to True.
        """
        super().__init__(
            save_concurrent_gb=save_concurrent_gb,
            restore_concurrent_gb=restore_concurrent_gb,
            multiprocessing_options=multiprocessing_options,
            pytree_metadata_options=pytree_metadata_options,
            use_ocdbt=use_ocdbt,
        )


class JsonCheckpointHandler(AsyncCheckpointHandler):
    """Handler for saving and restoring JSON-formatted checkpoint files."""

    def __init__(
        self,
        filename: str | None = None,
        *,
        multiprocessing_options: MultiprocessingOptions | None = None,
    ) -> None:
        """Initialize the JsonCheckpointHandler.

        Args:
            filename: Optional target JSON filename inside the checkpoint directory. Defaults to None.
            multiprocessing_options: Optional ``MultiprocessingOptions``. Defaults to None.
        """
        super().__init__()
        self.filename = filename
        self.multiprocessing_options = multiprocessing_options

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Save a JSON-serializable object to ``directory/<filename>``.

        Args:
            directory: Target checkpoint directory. Defaults to None.
            item: JSON-compatible object or ``args.JsonSave`` wrapper. Defaults to None.
            *pos_args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        if directory is not None:
            payload = _unwrap_checkpoint_arg(item, kwargs)
            fname = self.filename or "metadata.json"
            target_path = f"{str(directory).rstrip('/')}/{fname}"
            AtomicWrite().atomic_write_bytes(
                target_path,
                json.dumps(payload, default=str).encode("utf-8"),
            )

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore a JSON-formatted checkpoint from ``directory/<filename>``.

        Args:
            directory: Source checkpoint directory. Defaults to None.
            item: Optional fallback template or ``args.JsonRestore`` wrapper. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: Parsed JSON object or fallback template.
        """
        template = _unwrap_checkpoint_arg(item, kwargs)
        if directory is not None:
            fname = self.filename or "metadata.json"
            raw = FileSystemIO().read(f"{str(directory).rstrip('/')}/{fname}")
            if raw:
                loaded: PyTree = json.loads(raw.decode("utf-8"))
                return loaded
        return template


class ProtoCheckpointHandler(AsyncCheckpointHandler):
    """Handler for saving and restoring Protocol Buffer checkpoints."""

    def __init__(
        self,
        filename: str = "proto.pb",
        *,
        multiprocessing_options: MultiprocessingOptions | None = None,
    ) -> None:
        """Initialize the ProtoCheckpointHandler.

        Args:
            filename: Target filename for the serialized protobuf artifact. Defaults to ``"proto.pb"``.
            multiprocessing_options: Optional ``MultiprocessingOptions``. Defaults to None.
        """
        super().__init__()
        self.filename = filename
        self.multiprocessing_options = multiprocessing_options

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Save a protobuf payload to ``directory/<self.filename>``.

        Args:
            directory: Target checkpoint directory. Defaults to None.
            item: Protobuf payload or ``args.ProtoSave`` wrapper. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.
        """
        if directory is not None:
            raw_arg = item if isinstance(item, _CheckpointArgsBase) else kwargs.get("args")
            fname = (
                raw_arg.filename
                if isinstance(raw_arg, _ProtoSaveArgsCls) and self.filename == "proto.pb"
                else self.filename
            )
            payload = _unwrap_checkpoint_arg(item, kwargs)
            target_path = f"{str(directory).rstrip('/')}/{fname}"
            AtomicWrite().atomic_write_bytes(target_path, MsgpackEngine().packb(payload))

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore a protobuf payload from ``directory/<self.filename>``.

        Args:
            directory: Source checkpoint directory. Defaults to None.
            item: Optional template or ``args.ProtoRestore`` wrapper. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: Restored protobuf structure or template.
        """
        raw_arg = item if isinstance(item, _CheckpointArgsBase) else kwargs.get("args")
        item_type = raw_arg.item_type if isinstance(raw_arg, _ProtoRestoreArgsCls) else None
        fname = (
            raw_arg.filename
            if isinstance(raw_arg, _ProtoRestoreArgsCls) and self.filename == "proto.pb"
            else self.filename
        )
        template = _unwrap_checkpoint_arg(item, kwargs)
        if directory is not None:
            raw = FileSystemIO().read(f"{str(directory).rstrip('/')}/{fname}")
            if raw:
                unpacked = MsgpackEngine().unpackb(raw)
                if item_type is str and unpacked is not None:
                    return str(unpacked)
                return unpacked
        return template


class RandomKeyCheckpointHandler(AsyncCheckpointHandler):
    """Handler for saving and restoring JAX or NumPy PRNG random keys and states."""

    _default_key_name: str = "random_key"

    def __init__(self, key_name: str | None = None) -> None:
        """Initialize the RandomKeyCheckpointHandler.

        Args:
            key_name: Optional filename/key identifier for the saved random key. Defaults to None.
        """
        super().__init__()
        self.key_name = key_name

    def checkpoint_restore_args(
        self,
        item: PyTree,
        restore_args: RestoreArgs | ArrayRestoreArgs | None = None,
    ) -> RestoreArgs | ArrayRestoreArgs | None:
        """Construct ``RestoreArgs`` for restoring a random key.

        Args:
            item: Target random key item or structure.
            restore_args: Optional explicit ``RestoreArgs``. Defaults to None.

        Returns:
            RestoreArgs | ArrayRestoreArgs | None: The resolved ``RestoreArgs``, or ``None``.
        """
        if restore_args is not None:
            return restore_args
        if is_array_protocol(item):
            return ArrayRestoreArgs(
                restore_type=type(item),
                dtype=str(item.dtype),
                global_shape=tuple(int(d) for d in item.shape),
            )
        if item is not None:
            return RestoreArgs(restore_type=type(item))
        return None

    def checkpoint_save_args(
        self,
        item: PyTree,
    ) -> tuple[PyTree, SaveArgs | None]:
        """Construct ``(item, SaveArgs)`` tuple for saving a random key.

        Args:
            item: Random key item to save.

        Returns:
            tuple[PyTree, SaveArgs | None]: Tuple of ``(None, None)`` or ``(item, SaveArgs)``.
        """
        if item is None:
            return (None, None)
        return (item, SaveArgs())

    def post_restore(self, item: PyTree, metadata: PyTree = None) -> PyTree:
        """Apply post-restoration formatting to a restored random key.

        Args:
            item: Restored raw key data.
            metadata: Optional ``RestoreArgs`` or template metadata guiding type reconstruction. Defaults to None.

        Returns:
            PyTree: Post-processed random key.
        """
        if item is None:
            return None
        if isinstance(metadata, RestoreArgs):
            rtype = metadata.restore_type
            dtype_str = str(metadata.dtype) if metadata.dtype is not None else "int32"
            if rtype is jnp.ndarray or rtype == "jax":
                raw_data = array_to_nested_list(item) if is_array_protocol(item) else item
                if isinstance(raw_data, (int, float, bool, list, tuple)):
                    return jnp.asarray(raw_data, dtype=_resolve_compiler_dtype(dtype_str))
            if rtype is tuple and isinstance(item, list):
                return tuple(item)
            if rtype is list and isinstance(item, tuple):
                return list(item)
        if isinstance(metadata, jnp.ndarray) and isinstance(item, (list, tuple)):
            return jnp.asarray(item, dtype=_resolve_compiler_dtype(str(metadata.dtype)))
        return item

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Save a random key into ``directory/<key_name>.msgpack``.

        Args:
            directory: Target directory path. Defaults to None.
            item: Random key or ``CheckpointArgs`` wrapper. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.
        """
        _ = pos_args
        if directory is not None:
            raw_payload = _unwrap_checkpoint_arg(item, kwargs)
            key_item, _ = self.checkpoint_save_args(raw_payload)
            fname = f"{self.key_name or self._default_key_name}.msgpack"
            AtomicWrite().atomic_write_bytes(
                f"{str(directory).rstrip('/')}/{fname}",
                MsgpackEngine().packb(key_item),
            )

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore a random key from ``directory/<key_name>.msgpack``.

        Args:
            directory: Source directory path. Defaults to None.
            item: Optional template key or ``CheckpointArgs`` wrapper. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: Restored random key.
        """
        _ = pos_args
        template = _unwrap_checkpoint_arg(item, kwargs)
        resolved_rargs = self.checkpoint_restore_args(template)
        if directory is not None:
            fname = f"{self.key_name or self._default_key_name}.msgpack"
            raw = FileSystemIO().read(f"{str(directory).rstrip('/')}/{fname}")
            if raw:
                return self.post_restore(MsgpackEngine().unpackb(raw), metadata=resolved_rargs)
        return self.post_restore(template, metadata=resolved_rargs)


class JaxRandomKeyCheckpointHandler(RandomKeyCheckpointHandler):
    """Handler for saving and restoring JAX PRNG random keys."""

    _default_key_name: str = "jax_random_key"

    def __init__(self, key_name: str | None = None) -> None:
        """Initialize the JaxRandomKeyCheckpointHandler.

        Args:
            key_name: Optional filename/key identifier for the saved JAX random key. Defaults to None.
        """
        super().__init__(key_name=key_name)


class NumpyRandomKeyCheckpointHandler(RandomKeyCheckpointHandler):
    """Handler for saving and restoring NumPy PRNG random keys and states."""

    _default_key_name: str = "numpy_random_key"

    def __init__(self, key_name: str | None = None) -> None:
        """Initialize the NumpyRandomKeyCheckpointHandler.

        Args:
            key_name: Optional filename/key identifier for the saved NumPy random key. Defaults to None.
        """
        super().__init__(key_name=key_name)


class PyGrainCheckpointHandler(AsyncCheckpointHandler):
    """Handler for saving and restoring ``zero_grain`` dataset and data loader iterators."""

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Save a ``GrainIteratorProtocol`` state into ``directory/process_0-of-1.json``.

        Args:
            directory: Target checkpoint directory path. Defaults to None.
            item: ``GrainIteratorProtocol`` or ``args.PyGrainCheckpointSave`` wrapper. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.
        """
        if directory is not None:
            payload = _unwrap_checkpoint_arg(item, kwargs)
            if isinstance(payload, GrainIteratorProtocol):
                state = payload.get_state()
                target_path = f"{str(directory).rstrip('/')}/process_0-of-1.json"
                AtomicWrite().atomic_write_bytes(
                    target_path,
                    json.dumps(state, indent=2).encode("utf-8"),
                )

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore a ``GrainIteratorProtocol`` state from ``directory/process_0-of-1.json``.

        Args:
            directory: Source checkpoint directory path. Defaults to None.
            item: Optional target ``GrainIteratorProtocol`` or ``args.PyGrainCheckpointRestore``. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: The restored iterator (mutated in-place) or raw state dictionary if ``item is None``.
        """
        template = _unwrap_checkpoint_arg(item, kwargs)
        if directory is not None:
            target_path = f"{str(directory).rstrip('/')}/process_0-of-1.json"
            raw = FileSystemIO().read(target_path)
            if raw:
                state: CheckpointStateDict = json.loads(raw.decode("utf-8"))
                if isinstance(template, GrainIteratorProtocol):
                    template.set_state(state)
                    return template
                state_tree: PyTree = dict(state)
                return state_tree
        return template


class CheckpointHandlerRegistry:
    """Base interface for mapping checkpoint items and argument types to handlers."""

    def __init__(self, *args: PyTree, **kwargs: PyTree) -> None:
        """Initialize the CheckpointHandlerRegistry.

        Args:
            *args: Optional positional initialization arguments.
            **kwargs: Optional keyword initialization arguments.
        """
        _ = (args, kwargs)
        self._entries: dict[tuple[str | None, type | None], CheckpointHandler] = {}

    def add(
        self,
        item: str | None = None,
        args_ty: type | _CheckpointArgsBase | None = None,
        handler: CheckpointHandler | type[CheckpointHandler] | None = None,
        override: bool = False,
    ) -> None:
        """Register a handler for a given ``(item, args)`` pair.

        Args:
            item: Optional item name string (e.g. ``"state"``). Defaults to None.
            args_ty: Optional ``CheckpointArgs`` subclass or instance. Defaults to None.
            handler: Handler instance or class to register. Defaults to None.
            override: Whether to overwrite an existing registration. Defaults to False.

        Raises:
            ValueError: If the entry already exists and ``override=False``.
        """
        resolved_args_cls: type | None = (
            args_ty
            if isinstance(args_ty, type)
            else (type(args_ty) if args_ty is not None else None)
        )
        key = (item, resolved_args_cls)
        if key in self._entries and not override:
            raise ValueError(f"Handler already registered for {key}.")
        if handler is None:
            return
        handler_inst = handler() if isinstance(handler, type) else handler
        self._entries[key] = handler_inst

    def get(
        self,
        item: str | None = None,
        args_ty: type | _CheckpointArgsBase | None = None,
    ) -> CheckpointHandler | None:
        """Retrieve a registered handler for the given ``(item, args)`` pair.

        Args:
            item: Optional item name string. Defaults to None.
            args_ty: Optional ``CheckpointArgs`` subclass or instance. Defaults to None.

        Returns:
            CheckpointHandler | None: Registered handler instance, or ``None`` if not found.
        """
        resolved_args_cls: type | None = (
            args_ty
            if isinstance(args_ty, type)
            else (type(args_ty) if args_ty is not None else None)
        )
        if (item, resolved_args_cls) in self._entries:
            return self._entries[(item, resolved_args_cls)]
        if (item, None) in self._entries:
            return self._entries[(item, None)]
        if (None, resolved_args_cls) in self._entries:
            return self._entries[(None, resolved_args_cls)]
        return None

    def has(
        self,
        item: str | None = None,
        args_ty: type | _CheckpointArgsBase | None = None,
    ) -> bool:
        """Check whether a handler is registered for the given ``(item, args)`` pair.

        Args:
            item: Optional item name string. Defaults to None.
            args_ty: Optional ``CheckpointArgs`` subclass or instance. Defaults to None.

        Returns:
            bool: True if a matching handler is registered, False otherwise.
        """
        return self.get(item, args_ty) is not None

    def get_all_entries(
        self,
    ) -> dict[tuple[str | None, type | None], CheckpointHandler]:
        """Return a copy of all registered ``(item, args_cls) -> handler`` entries.

        Returns:
            dict[tuple[str | None, type | None], CheckpointHandler]: Registered handler mapping.
        """
        return dict(self._entries)


class DefaultCheckpointHandlerRegistry(CheckpointHandlerRegistry):
    """Default concrete implementation of ``CheckpointHandlerRegistry``."""

    def __init__(
        self,
        other_registry: CheckpointHandlerRegistry | None = None,
    ) -> None:
        """Initialize the DefaultCheckpointHandlerRegistry, optionally copying entries.

        Args:
            other_registry: Optional existing registry whose entries should be copied. Defaults to None.
        """
        super().__init__()
        self.other_registry = other_registry
        if other_registry is not None:
            self._entries.update(other_registry.get_all_entries())


def create_default_handler_registry(
    **items_and_handlers: CheckpointHandler | None,
) -> CheckpointHandlerRegistry:
    """Create a ``DefaultCheckpointHandlerRegistry`` populated with default and custom item handlers.

    Args:
        **items_and_handlers: Optional keyword mapping from item names to ``CheckpointHandler`` instances.

    Returns:
        CheckpointHandlerRegistry: Initialized registry with standard and custom handlers registered.
    """
    registry = DefaultCheckpointHandlerRegistry()
    for save_cls, handler_cls in _CATALOG.save_args_to_handler.items():
        registry.add(None, save_cls, handler_cls, override=True)
    for restore_cls, handler_cls in _CATALOG.restore_args_to_handler.items():
        registry.add(None, restore_cls, handler_cls, override=True)
    for item_name, handler_inst in items_and_handlers.items():
        if handler_inst is not None:
            registry.add(item_name, None, handler_inst, override=True)
    return registry


class CompositeCheckpointHandler(AsyncCheckpointHandler):
    """Handler coordinating saving and restoring of multiple named sub-items."""

    def __init__(
        self,
        *item_names: str,
        composite_options: CompositeOptions | None = None,
        handler_registry: CheckpointHandlerRegistry | None = None,
        **items_and_handlers: CheckpointHandler | None,
    ) -> None:
        """Initialize the CompositeCheckpointHandler.

        Args:
            *item_names: Positional names of composite sub-items to manage.
            composite_options: Optional ``CompositeOptions`` configuration. Defaults to None.
            handler_registry: Optional ``CheckpointHandlerRegistry`` for resolving item handlers. Defaults to None.
            **items_and_handlers: Keyword mapping of item names to explicit ``CheckpointHandler`` instances.
        """
        super().__init__()
        self.item_names: tuple[str, ...] = item_names + tuple(items_and_handlers.keys())
        self.composite_options = composite_options or CompositeOptions()
        self.handler_registry = handler_registry
        self._item_handlers: dict[str, CheckpointHandler] = {
            name: StandardCheckpointHandler() for name in item_names
        }
        for k, v in items_and_handlers.items():
            if v is not None:
                self._item_handlers[k] = v

    def _resolve_handler_for_item(
        self,
        item_name: str,
        item_arg: _CheckpointArgsBase | PyTree,
    ) -> CheckpointHandler:
        """Resolve the ``CheckpointHandler`` for a composite sub-item.

        Args:
            item_name: Name of the composite sub-item.
            item_arg: Sub-item value or ``CheckpointArgs`` instance.

        Returns:
            CheckpointHandler: Resolved handler for ``item_name``.
        """
        if self.handler_registry is not None:
            arg_hint = item_arg if isinstance(item_arg, _CheckpointArgsBase) else None
            from_reg = self.handler_registry.get(item_name, arg_hint)
            if from_reg is not None:
                self._item_handlers[item_name] = from_reg
                return from_reg
        if isinstance(item_arg, _CheckpointArgsBase):
            reg_cls = args.get_registered_handler_cls(item_arg)
            if reg_cls is not None and issubclass(reg_cls, CheckpointHandler):
                existing = self._item_handlers.get(item_name)
                if existing is not None and isinstance(existing, reg_cls):
                    return existing
                inst: CheckpointHandler = reg_cls()
                self._item_handlers[item_name] = inst
                return inst
        unwrapped_item = item_arg.item if isinstance(item_arg, _CheckpointArgsBase) else item_arg
        if isinstance(unwrapped_item, GrainIteratorProtocol):
            grain_handler = PyGrainCheckpointHandler()
            self._item_handlers[item_name] = grain_handler
            return grain_handler
        if item_name in self._item_handlers:
            return self._item_handlers[item_name]
        default_handler = StandardCheckpointHandler()
        self._item_handlers[item_name] = default_handler
        return default_handler

    def _extract_composite_map(
        self,
        item: PyTree,
        kwargs: Mapping[str, PyTree],
    ) -> dict[str, _CheckpointArgsBase | PyTree]:
        """Extract a mapping of composite item names to payloads from ``item`` or ``kwargs``.

        Args:
            item: Positional ``args.Composite`` or mapping.
            kwargs: Keyword arguments mapping.

        Returns:
            dict[str, _CheckpointArgsBase | PyTree]: Mapping of sub-item names to values/args.
        """
        raw_arg = item if isinstance(item, _CheckpointArgsBase) else kwargs.get("args", item)
        if isinstance(raw_arg, _CompositeArgsCls):
            return dict(raw_arg.items())
        if is_pytree_dict(raw_arg):
            return dict(raw_arg)
        return {k: v for k, v in kwargs.items() if k != "args"}

    def save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> None:
        """Save each named composite item into ``<directory>/<item_name>``.

        Args:
            directory: Target step checkpoint directory. Defaults to None.
            item: ``args.Composite`` or dictionary of named items. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.
        """
        if directory is None:
            return
        perm_mode = (
            self.composite_options.file_options.path_permission_mode
            if self.composite_options.file_options is not None
            else None
        )
        base_dir = str(directory).rstrip("/")
        comp_map = self._extract_composite_map(item, kwargs)
        for name, sub_arg in comp_map.items():
            sub_handler = self._resolve_handler_for_item(name, sub_arg)
            sub_dir = f"{base_dir}/{name}"
            FileSystemIO().makedirs(sub_dir, exist_ok=True, permission_mode=perm_mode)
            sub_handler.save(sub_dir, sub_arg)

    def async_save(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> list[Future] | None:
        """Asynchronously save each named composite item and aggregate futures.

        Args:
            directory: Target step checkpoint directory. Defaults to None.
            item: ``args.Composite`` or dictionary of named items. Defaults to None.
            *pos_args: Additional positional save arguments.
            **kwargs: Additional keyword save arguments.

        Returns:
            list[Future] | None: Aggregated list of futures, or None if ``directory`` is None.
        """
        if directory is None:
            return None
        perm_mode = (
            self.composite_options.file_options.path_permission_mode
            if self.composite_options.file_options is not None
            else None
        )
        base_dir = str(directory).rstrip("/")
        comp_map = self._extract_composite_map(item, kwargs)
        futures: list[Future] = []
        for name, sub_arg in comp_map.items():
            sub_handler = self._resolve_handler_for_item(name, sub_arg)
            sub_dir = f"{base_dir}/{name}"
            FileSystemIO().makedirs(sub_dir, exist_ok=True, permission_mode=perm_mode)
            if isinstance(sub_handler, AsyncCheckpointHandler):
                futures.extend(sub_handler.async_save(sub_dir, sub_arg) or [])
            else:
                sub_handler.save(sub_dir, sub_arg)
        return futures

    def restore(
        self,
        directory: PathLike | None = None,
        item: PyTree = None,
        *pos_args: PyTree,
        **kwargs: PyTree,
    ) -> PyTree:
        """Restore all composite items from ``<directory>/<item_name>``.

        Args:
            directory: Source step checkpoint directory. Defaults to None.
            item: Optional ``args.Composite`` or template dictionary. Defaults to None.
            *pos_args: Additional positional restore arguments.
            **kwargs: Additional keyword restore arguments.

        Returns:
            PyTree: Dictionary mapping each item name to its restored payload.
        """
        if directory is None:
            return None
        base_dir = str(directory).rstrip("/")
        comp_map = self._extract_composite_map(item, kwargs)
        discovered_dirs = [
            entry for entry in FileSystemIO().listdir(base_dir) if not entry.startswith("_")
        ]
        target_names: list[str] = (
            list(comp_map.keys())
            if comp_map
            else list(dict.fromkeys([*self.item_names, *discovered_dirs]))
        )
        restored_dict: dict[str, PyTree] = {}
        for name in target_names:
            sub_arg = comp_map.get(name)
            sub_dir = f"{base_dir}/{name}"
            if (
                sub_arg is None
                and name not in self._item_handlers
                and FileSystemIO().exists(f"{sub_dir}/process_0-of-1.json")
            ):
                self._item_handlers[name] = PyGrainCheckpointHandler()
            sub_handler = self._resolve_handler_for_item(name, sub_arg)
            restored_dict[name] = sub_handler.restore(sub_dir, sub_arg)
        return restored_dict

    def metadata(self, directory: PathLike | None = None) -> dict[str, JsonValue] | None:
        """Return aggregated metadata across all composite sub-items.

        Args:
            directory: Step checkpoint directory. Defaults to None.

        Returns:
            dict[str, JsonValue] | None: Mapping of item name to sub-handler metadata, or None.
        """
        if directory is None:
            return self._last_metadata
        base_dir = str(directory).rstrip("/")
        meta_map: dict[str, JsonValue] = {}
        for name, sub_handler in self._item_handlers.items():
            sub_meta = sub_handler.metadata(f"{base_dir}/{name}")
            if sub_meta is not None:
                meta_map[name] = sub_meta
        return meta_map if meta_map else self._last_metadata

    def structure(self, directory: PathLike | None = None) -> PyTree:
        """Return the composite structure across all sub-handlers.

        Args:
            directory: Step checkpoint directory. Defaults to None.

        Returns:
            PyTree: Composite structure mapping, or None.
        """
        if directory is None:
            return None
        return self.restore(directory)

    def finalize(self, directory: PathLike | None = None) -> None:
        """Finalize all composite sub-item directories.

        Args:
            directory: Step checkpoint directory. Defaults to None.
        """
        super().finalize(directory)
        if directory is not None:
            base_dir = str(directory).rstrip("/")
            for name, sub_handler in self._item_handlers.items():
                sub_handler.finalize(f"{base_dir}/{name}")

    def close(self) -> None:
        """Close all managed sub-handlers."""
        super().close()
        for sub_handler in self._item_handlers.values():
            sub_handler.close()


_CATALOG.canonical_grain_handler = PyGrainCheckpointHandler

for _s_cls, _h_cls in (
    (args.ArraySave, ArrayCheckpointHandler),
    (args.JsonSave, JsonCheckpointHandler),
    (args.ProtoSave, ProtoCheckpointHandler),
    (args.PyTreeSave, PyTreeCheckpointHandler),
    (args.StandardSave, StandardCheckpointHandler),
    (args.JaxRandomKeySave, JaxRandomKeyCheckpointHandler),
    (args.NumpyRandomKeySave, NumpyRandomKeyCheckpointHandler),
    (args.PyGrainCheckpointSave, PyGrainCheckpointHandler),
    (args.Composite, CompositeCheckpointHandler),
):
    _CATALOG.register(_s_cls, _h_cls, for_save=True)

for _r_cls, _h_cls in (
    (args.ArrayRestore, ArrayCheckpointHandler),
    (args.JsonRestore, JsonCheckpointHandler),
    (args.ProtoRestore, ProtoCheckpointHandler),
    (args.PyTreeRestore, PyTreeCheckpointHandler),
    (args.StandardRestore, StandardCheckpointHandler),
    (args.JaxRandomKeyRestore, JaxRandomKeyCheckpointHandler),
    (args.NumpyRandomKeyRestore, NumpyRandomKeyCheckpointHandler),
    (args.PyGrainCheckpointRestore, PyGrainCheckpointHandler),
    (args.Composite, CompositeCheckpointHandler),
):
    _CATALOG.register(_r_cls, _h_cls, for_restore=True)


class async_checkpoint_handler:
    """Namespace for asynchronous checkpoint handler classes."""

    AsyncCheckpointHandler = AsyncCheckpointHandler


class checkpoint_handler:
    """Namespace exporting the base ``CheckpointHandler`` class."""

    CheckpointHandler = CheckpointHandler


class handlers:
    """Namespace exporting all concrete ``CheckpointHandler`` implementations and registries."""

    CheckpointHandler = CheckpointHandler
    AsyncCheckpointHandler = AsyncCheckpointHandler
    ArrayCheckpointHandler = ArrayCheckpointHandler
    BasePyTreeCheckpointHandler = BasePyTreeCheckpointHandler
    PyTreeCheckpointHandler = PyTreeCheckpointHandler
    StandardCheckpointHandler = StandardCheckpointHandler
    JsonCheckpointHandler = JsonCheckpointHandler
    ProtoCheckpointHandler = ProtoCheckpointHandler
    JaxRandomKeyCheckpointHandler = JaxRandomKeyCheckpointHandler
    NumpyRandomKeyCheckpointHandler = NumpyRandomKeyCheckpointHandler
    RandomKeyCheckpointHandler = RandomKeyCheckpointHandler
    PyGrainCheckpointHandler = PyGrainCheckpointHandler
    CompositeCheckpointHandler = CompositeCheckpointHandler
    CheckpointHandlerRegistry = CheckpointHandlerRegistry
    DefaultCheckpointHandlerRegistry = DefaultCheckpointHandlerRegistry
    create_default_handler_registry = staticmethod(create_default_handler_registry)


__all__ = [
    "ArrayCheckpointHandler",
    "AsyncCheckpointHandler",
    "BasePyTreeCheckpointHandler",
    "CheckpointHandler",
    "CheckpointHandlerRegistry",
    "CompositeCheckpointHandler",
    "DefaultCheckpointHandlerRegistry",
    "JaxRandomKeyCheckpointHandler",
    "JsonCheckpointHandler",
    "NumpyRandomKeyCheckpointHandler",
    "ProtoCheckpointHandler",
    "PyGrainCheckpointHandler",
    "PyTreeCheckpointHandler",
    "RandomKeyCheckpointHandler",
    "StandardCheckpointHandler",
    "async_checkpoint_handler",
    "checkpoint_handler",
    "create_default_handler_registry",
    "handlers",
]
