"""PyTree transformation rules and merging utilities for zero_orbax.checkpoint.transform_utils."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence

from ml_switcheroo_compiler import ops as compiler_ops
from ml_switcheroo_compiler.core.dtype import DType as CompilerDType
from ml_switcheroo_compiler.core.tensor import Tensor as CompilerTensor, as_tensor_obj
from ml_switcheroo_ir import DType
from zero_flax.nnx.variables import (
    BatchStat,
    Cache,
    Intermediate,
    LoRAParam,
    Param,
    Rng,
    RngState,
    Variable as FlaxVariable,
    VariableValue,
)
from zero_flax.types import MetadataValue
from zero_jax import numpy as jnp

from zero_orbax._checkpoint.type_handlers_lib import ArrayRestoreArgs, RestoreArgs, SaveArgs
from zero_orbax.core import SerializedArray, resolve_compiler_dtype as _resolve_compiler_dtype
from zero_orbax.types import (
    FlaxStateProtocol,
    FlaxVariableProtocol,
    FrozenDictProtocol,
    JsonValue,
    MultiValueTransformFn,
    NamedTupleProtocol,
    PyTree,
    ValueTransformFn,
    array_to_nested_list,
    is_array_protocol,
    is_pytree_dict,
)

_FLAX_VARIABLE_CLASSES: dict[str, type[FlaxVariable[VariableValue]]] = {
    "Param": Param,
    "BatchStat": BatchStat,
    "Cache": Cache,
    "Intermediate": Intermediate,
    "LoRAParam": LoRAParam,
    "Rng": Rng,
    "RngState": RngState,
    "Variable": FlaxVariable,
}


def _coerce_flax_variable_value(val: PyTree) -> VariableValue:
    """Coerce an unpacked PyTree leaf into a strongly typed ``VariableValue``.

    Args:
        val: Unpacked PyTree value for a Flax ``Variable``.

    Returns:
        VariableValue: Typed JAX array or scalar leaf.
    """
    if val is None or isinstance(val, (jnp.ndarray, int, float, bool, str)):
        return val
    return jnp.asarray(array_to_nested_list(val) if is_array_protocol(val) else val)


def _coerce_flax_metadata(meta: PyTree) -> dict[str, MetadataValue]:
    """Coerce an unpacked metadata mapping into a typed ``dict[str, MetadataValue]``.

    Args:
        meta: Raw unpacked metadata dictionary.

    Returns:
        dict[str, MetadataValue]: Strongly typed variable metadata mapping.
    """
    if not is_pytree_dict(meta):
        return {}
    return {
        str(k): v for k, v in meta.items() if v is None or isinstance(v, (int, float, bool, str))
    }


def _reconstruct_array_leaf(
    data: PyTree,
    shape: tuple[int, ...],
    dtype_str: str,
    array_kind: str,
    template: PyTree = None,
) -> PyTree:
    """Reconstruct a typed array leaf from serialized data, shape, and dtype.

    Args:
        data: Nested list or scalar data payload.
        shape: Target array shape tuple.
        dtype_str: Target data type name.
        array_kind: Recorded array kind (``"tensor"``, ``"serialized_array"``, or ``"jax_array"``).
        template: Optional template leaf guiding array type selection. Defaults to None.

    Returns:
        PyTree: Reconstructed ``zero_jax.numpy.ndarray``, ``CompilerTensor``, or ``SerializedArray``.
    """
    resolved_dt = _resolve_compiler_dtype(dtype_str)
    raw_input = data if isinstance(data, (int, float, bool, list, tuple)) else 0.0
    if isinstance(template, CompilerTensor) or (template is None and array_kind == "tensor"):
        return as_tensor_obj(compiler_ops.asarray(raw_input, dtype=resolved_dt))
    if isinstance(template, SerializedArray) or (
        template is None and array_kind == "serialized_array"
    ):
        return SerializedArray(data=data, shape=shape, dtype=dtype_str)
    return jnp.reshape(jnp.asarray(raw_input, dtype=resolved_dt), shape)


def _build_flax_state(entries: Mapping[str, PyTree], template: PyTree = None) -> PyTree:
    """Construct a ``zero_flax.nnx.State`` from unpacked entries and optionally update ``template``.

    Args:
        entries: Mapping of state keys to unpacked values.
        template: Optional target ``State`` template to update in-place. Defaults to None.

    Returns:
        PyTree: Reconstructed ``zero_flax.nnx.State`` instance.
    """
    from zero_flax.nnx.state import State as FlaxState, StateEntry

    typed_entries: dict[str, StateEntry] = {}
    for k, v in entries.items():
        if isinstance(v, (FlaxState, FlaxVariable)):
            typed_entries[str(k)] = v
        else:
            typed_entries[str(k)] = _coerce_flax_variable_value(v)
    state_obj = FlaxState(typed_entries)
    if isinstance(template, FlaxState):
        template.update(state_obj)
    return state_obj


def _build_frozen_dict(entries: Mapping[str, PyTree]) -> PyTree:
    """Construct a ``zero_flax.core.FrozenDict`` from unpacked entries.

    Args:
        entries: Mapping of keys to unpacked PyTree values.

    Returns:
        PyTree: Reconstructed ``FrozenDict`` instance.
    """
    from zero_flax.core import FrozenDict, FrozenEntry

    typed_entries: dict[str, FrozenEntry] = {}
    for k, v in entries.items():
        if isinstance(v, (FrozenDict, FlaxVariable)):
            typed_entries[str(k)] = v
        else:
            typed_entries[str(k)] = _coerce_flax_variable_value(v)
    return FrozenDict(typed_entries)


def _extract_save_args_metadata(save_args_tree: PyTree) -> dict[str, JsonValue] | None:
    """Extract serializable metadata from a tree of ``SaveArgs``.

    Args:
        save_args_tree: PyTree of ``SaveArgs`` instances.

    Returns:
        dict[str, JsonValue] | None: Extracted save args metadata dictionary, or None.
    """
    if isinstance(save_args_tree, SaveArgs):
        return {
            "aggregate": save_args_tree.aggregate,
            "dtype": str(save_args_tree.dtype) if save_args_tree.dtype is not None else None,
            "chunk_byte_size": save_args_tree.chunk_byte_size,
            "shard_axes": [int(a) for a in save_args_tree.shard_axes],
        }
    if is_pytree_dict(save_args_tree):
        sub_meta: dict[str, JsonValue] = {}
        for k, v in save_args_tree.items():
            extracted = _extract_save_args_metadata(v)
            if extracted is not None:
                sub_meta[str(k)] = extracted
        return sub_meta if sub_meta else None
    return None


def _apply_save_args_tree(item: PyTree, save_args_tree: PyTree) -> PyTree:
    """Recursively apply a tree of ``SaveArgs`` to ``item`` before serialization.

    Args:
        item: PyTree structure to save.
        save_args_tree: Matching tree of ``SaveArgs``.

    Returns:
        PyTree: PyTree with leaf ``dtype`` coercions applied.
    """
    if save_args_tree is None or item is None:
        return item
    if is_pytree_dict(item) and is_pytree_dict(save_args_tree):
        return {k: _apply_save_args_tree(v, save_args_tree.get(k)) for k, v in item.items()}
    if isinstance(item, list) and isinstance(save_args_tree, (list, tuple)):
        return [
            _apply_save_args_tree(v, save_args_tree[i] if i < len(save_args_tree) else None)
            for i, v in enumerate(item)
        ]
    if isinstance(item, tuple) and isinstance(save_args_tree, (list, tuple)):
        return tuple(
            _apply_save_args_tree(v, save_args_tree[i] if i < len(save_args_tree) else None)
            for i, v in enumerate(item)
        )
    if isinstance(save_args_tree, SaveArgs):
        dt = save_args_tree.dtype
        if dt is not None and is_array_protocol(item):
            resolved_dt: CompilerDType = _resolve_compiler_dtype(
                "int32" if dt is int else ("float32" if dt is float else str(dt))
            )
            nested = array_to_nested_list(item)
            raw_in = nested if isinstance(nested, (int, float, bool, list, tuple)) else 0.0
            if isinstance(item, CompilerTensor):
                return as_tensor_obj(compiler_ops.asarray(raw_in, dtype=resolved_dt))
            if isinstance(item, SerializedArray):
                return SerializedArray(data=nested, shape=item.shape, dtype=resolved_dt.name)
            return jnp.reshape(jnp.asarray(raw_in, dtype=resolved_dt), item.shape)
        if (dt is int or dt == "int" or dt == "int32" or dt == "int64") and isinstance(
            item, (int, float, str, bool)
        ):
            return int(item)
        if (dt is float or dt == "float" or dt == "float32" or dt == "float64") and isinstance(
            item, (int, float, str, bool)
        ):
            return float(item)
        if dt is str or dt == "str":
            return str(item)
    return item


def _apply_restore_args_tree(restored: PyTree, restore_args_tree: PyTree) -> PyTree:
    """Recursively apply a tree of ``RestoreArgs`` / ``ArrayRestoreArgs`` to ``restored``.

    Args:
        restored: Restored PyTree structure.
        restore_args_tree: Matching tree of ``RestoreArgs`` or ``ArrayRestoreArgs``.

    Returns:
        PyTree: Coerced and validated PyTree structure.
    """
    if restore_args_tree is None:
        return restored
    if isinstance(restored, FlaxStateProtocol) and is_pytree_dict(restore_args_tree):
        return _build_flax_state(
            {k: _apply_restore_args_tree(v, restore_args_tree.get(k)) for k, v in restored.items()},
            restored,
        )
    if isinstance(restored, FrozenDictProtocol) and is_pytree_dict(restore_args_tree):
        return _build_frozen_dict(
            {k: _apply_restore_args_tree(v, restore_args_tree.get(k)) for k, v in restored.items()}
        )
    if is_pytree_dict(restored) and is_pytree_dict(restore_args_tree):
        return {
            k: _apply_restore_args_tree(v, restore_args_tree.get(k)) for k, v in restored.items()
        }
    if isinstance(restored, list) and isinstance(restore_args_tree, (list, tuple)):
        return [
            _apply_restore_args_tree(
                v, restore_args_tree[i] if i < len(restore_args_tree) else None
            )
            for i, v in enumerate(restored)
        ]
    if isinstance(restored, tuple) and isinstance(restore_args_tree, (list, tuple)):
        return tuple(
            _apply_restore_args_tree(
                v, restore_args_tree[i] if i < len(restore_args_tree) else None
            )
            for i, v in enumerate(restored)
        )
    if isinstance(restored, FlaxVariableProtocol) and isinstance(
        restore_args_tree, ArrayRestoreArgs
    ):
        coerced_inner = _apply_restore_args_tree(restored.value, restore_args_tree)
        return restored.replace(value=_coerce_flax_variable_value(coerced_inner))
    if isinstance(restore_args_tree, ArrayRestoreArgs) and is_array_protocol(restored):
        from zero_chex._src.asserts.tensors import (
            assert_shape as _chex_assert_shape,
            assert_type as _chex_assert_type,
        )

        expected_shape = (
            restore_args_tree.shape
            if restore_args_tree.shape is not None
            else restore_args_tree.global_shape
        )
        if restore_args_tree.strict and expected_shape is not None:
            _chex_assert_shape(restored, expected_shape)
        if restore_args_tree.dtype is not None:
            resolved_dt = _resolve_compiler_dtype(str(restore_args_tree.dtype))
            nested_data = array_to_nested_list(restored)
            raw_in = (
                nested_data if isinstance(nested_data, (int, float, bool, list, tuple)) else 0.0
            )
            restored = jnp.reshape(jnp.asarray(raw_in, dtype=resolved_dt), restored.shape)
            if restore_args_tree.strict:
                _chex_assert_type(restored, resolved_dt)
        return restored
    target_type: type | str | None = (
        restore_args_tree.restore_type
        if isinstance(restore_args_tree, RestoreArgs)
        else (
            restore_args_tree.restore_type()
            if isinstance(restore_args_tree, ArrayRestoreArgs)
            else None
        )
    )
    target_dtype: str | DType | type | None = (
        restore_args_tree.dtype
        if isinstance(restore_args_tree, (RestoreArgs, ArrayRestoreArgs))
        else None
    )
    effective_target = target_type if target_type is not None else target_dtype

    if (effective_target is int or effective_target in ("int", "int32", "int64")) and isinstance(
        restored, (int, float, str, bool)
    ):
        return int(restored)
    if (
        effective_target is float or effective_target in ("float", "float32", "float64")
    ) and isinstance(restored, (int, float, str, bool)):
        return float(restored)
    if (effective_target is str or effective_target == "str") and restored is not None:
        return str(restored)
    if effective_target is list and isinstance(restored, (tuple, list)):
        return list(restored)
    if effective_target is tuple and isinstance(restored, (tuple, list)):
        return tuple(restored)
    return restored


@dataclasses.dataclass
class Transform:
    """Rule defining how to map or transform a PyTree node from an original tree.

    Attributes:
        original_key: Key, slash-delimited path, regex pattern, or key sequence in the original PyTree.
        use_fallback: Whether to fall back to the same key in the original tree if ``original_key`` is missing.
        value_fn: Single-value transformation function ``(val) -> new_val``.
        multi_value_fn: Multi-value transformation function ``(orig_tree) -> new_val``.
        multi_value_fn_input_args: Optional mapping of input keys to ``RestoreArgs``.
    """

    original_key: str | Sequence[str] | None = None
    use_fallback: bool = False
    value_fn: ValueTransformFn | None = None
    multi_value_fn: MultiValueTransformFn | None = None
    multi_value_fn_input_args: Mapping[str, RestoreArgs] | None = None


class RestoreTransform(Transform):
    """Rule defining how to transform data during a checkpoint restore operation."""

    def __init__(
        self,
        original_key: str | Sequence[str] | None = None,
        use_fallback: bool = False,
        value_fn: ValueTransformFn | None = None,
        multi_value_fn: MultiValueTransformFn | None = None,
        multi_value_fn_input_args: Mapping[str, RestoreArgs] | None = None,
    ) -> None:
        """Initialize the RestoreTransform rule.

        Args:
            original_key: Key or path in the original structure to map from. Defaults to None.
            use_fallback: Whether to use fallback resolution logic. Defaults to False.
            value_fn: Function applied to a single restored value. Defaults to None.
            multi_value_fn: Function deriving a restored value from the full original tree. Defaults to None.
            multi_value_fn_input_args: Optional mapping of input keys to ``RestoreArgs``. Defaults to None.
        """
        super().__init__(
            original_key=original_key,
            use_fallback=use_fallback,
            value_fn=value_fn,
            multi_value_fn=multi_value_fn,
            multi_value_fn_input_args=multi_value_fn_input_args,
        )


def _get_by_path(
    tree: dict[str, PyTree],
    path: str | Sequence[str],
    default_val: PyTree,
) -> PyTree:
    """Retrieve a nested dictionary value by string key, slash-delimited path, regex, or key sequence.

    Args:
        tree: The root dictionary to search.
        path: String key, ``"a/b"`` path, regex pattern, or sequence of keys ``("a", "b")``.
        default_val: Fallback value returned if the path cannot be resolved.

    Returns:
        PyTree: The resolved value at ``path``, or ``default_val``.
    """
    if isinstance(path, str):
        if "/" in path:
            parts: Sequence[str] = path.split("/")
        else:
            if path in tree:
                return tree[path]
            for k, v in tree.items():
                if re.fullmatch(path, k):
                    return v
            return default_val
    else:
        parts = path

    curr: PyTree = tree
    for key in parts:
        if not is_pytree_dict(curr):
            return default_val
        if key in curr:
            curr = curr[key]
        else:
            matched_val: PyTree = None
            found = False
            for ck, cv in curr.items():
                if re.fullmatch(key, ck):
                    matched_val = cv
                    found = True
                    break
            if not found:
                return default_val
            curr = matched_val
    return curr


def apply_transformations(
    original_tree: PyTree,
    transformations: PyTree,
    new_tree: PyTree,
    default_to_original: bool | None = True,
) -> PyTree:
    """Apply a tree of ``Transform`` rules to map ``original_tree`` into ``new_tree``.

    Args:
        original_tree: The source PyTree dictionary.
        transformations: The PyTree of ``Transform`` / ``RestoreTransform`` rules.
        new_tree: The target PyTree structure defining expected output keys.
        default_to_original: If True, keys omitted from ``transformations`` fall back
            to ``original_tree`` values. Defaults to True.

    Returns:
        PyTree: The transformed PyTree matching the structure of ``new_tree``.
    """
    if not is_pytree_dict(new_tree):
        return new_tree

    orig_root: dict[str, PyTree] = original_tree if is_pytree_dict(original_tree) else {}

    def _process(
        orig: dict[str, PyTree],
        trans: dict[str, PyTree],
        new: dict[str, PyTree],
        orig_full: dict[str, PyTree],
    ) -> dict[str, PyTree]:
        """Recursively apply transformation rules for each key in ``new``.

        Args:
            orig: Current dictionary node in the original tree.
            trans: Current dictionary node in the transformations tree.
            new: Current dictionary node in the target tree.
            orig_full: Full original root dictionary for global path lookups.

        Returns:
            dict[str, PyTree]: Transformed sub-dictionary.
        """
        res: dict[str, PyTree] = {}
        for k, new_val in new.items():
            t = trans.get(k)
            o = orig.get(k)
            orig_sub: dict[str, PyTree] = o if is_pytree_dict(o) else {}
            if t is not None:
                if is_pytree_dict(t):
                    res[k] = (
                        _process(orig_sub, t, new_val, orig_full)
                        if is_pytree_dict(new_val)
                        else new_val
                    )
                elif isinstance(t, Transform):
                    if t.multi_value_fn is not None:
                        coerced_full = (
                            _apply_restore_args_tree(
                                orig_full,
                                dict(t.multi_value_fn_input_args),
                            )
                            if t.multi_value_fn_input_args is not None
                            else orig_full
                        )
                        res[k] = t.multi_value_fn(coerced_full)
                    else:
                        fallback_val = (
                            o
                            if (t.use_fallback or t.original_key is None)
                            else (new_val if new_val is not None else o)
                        )
                        resolved_o = (
                            _get_by_path(orig_full, t.original_key, fallback_val)
                            if t.original_key is not None
                            else o
                        )
                        if t.value_fn is not None:
                            res[k] = t.value_fn(resolved_o)
                        else:
                            res[k] = resolved_o
                else:
                    res[k] = t
            elif not default_to_original:
                res[k] = new_val
            else:
                res[k] = o if o is not None else new_val
        return res

    trans_root: dict[str, PyTree] = transformations if is_pytree_dict(transformations) else {}
    return _process(orig_root, trans_root, new_tree, orig_root)


def merge_trees(
    *trees: PyTree,
    target: dict[str, PyTree] | None = None,
) -> dict[str, PyTree]:
    """Deeply merge multiple PyTree dictionaries into a single dictionary.

    Args:
        *trees: Variable number of PyTree dictionaries to merge from left to right.
        target: Optional initial target dictionary to seed the merge. Defaults to None.

    Returns:
        dict[str, PyTree]: Deeply merged PyTree dictionary.

    Raises:
        TypeError: If any positional argument in ``trees`` is not a ``dict``.
    """
    if not trees:
        return {}
    res: dict[str, PyTree] = {}
    if target is not None:
        res.update(target)
    for t in trees:
        if is_pytree_dict(t):
            for k, v in t.items():
                existing = res.get(k)
                if is_pytree_dict(existing) and is_pytree_dict(v):
                    res[k] = merge_trees(existing, v)
                else:
                    res[k] = v
        else:
            raise TypeError("Expected PyTree dicts")
    return res


class transform_utils:
    """Namespace for PyTree transformation rules and merging utilities."""

    Transform = Transform
    RestoreTransform = RestoreTransform
    apply_transformations = staticmethod(apply_transformations)
    merge_trees = staticmethod(merge_trees)


__all__ = [
    "FlaxVariable",
    "NamedTupleProtocol",
    "RestoreTransform",
    "Transform",
    "_FLAX_VARIABLE_CLASSES",
    "_apply_restore_args_tree",
    "_apply_save_args_tree",
    "_build_flax_state",
    "_build_frozen_dict",
    "_coerce_flax_metadata",
    "_coerce_flax_variable_value",
    "_extract_save_args_metadata",
    "_get_by_path",
    "_reconstruct_array_leaf",
    "_resolve_compiler_dtype",
    "apply_transformations",
    "merge_trees",
    "transform_utils",
]
