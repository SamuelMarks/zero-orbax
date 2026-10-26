"""Orbax Export model serving and serialization APIs (`orbax.export`).

This module provides strongly typed, zero-dependency classes for wrapping JAX/PyTree
models (`JaxModule`), configuring serving signatures and pre/post-processors
(`ServingConfig`, `ObmExportOptions`), and exporting or loading serialized model
artifacts (`ExportManager`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing_extensions import TypeAlias

from zero_orbax.core import AtomicWrite, FileSystemIO, MsgpackEngine
from zero_orbax.types import PathLike, PyTree

ApplyFn: TypeAlias = Callable[[PyTree, PyTree], PyTree]
MethodFn: TypeAlias = Callable[[PyTree], PyTree]


@dataclasses.dataclass
class ObmExportOptions:
    """Options configuring OpenXLA / Orbax Model (OBM) export behavior.

    Attributes:
        enable_auto_layout: Whether to enable automatic tensor memory layout optimization. Defaults to False.
        checkpoint_path: Optional relative or absolute path for exported checkpoint weights. Defaults to None.
    """

    enable_auto_layout: bool = False
    checkpoint_path: str | None = None


@dataclasses.dataclass
class ServingConfig:
    """Configuration binding a serving signature key to pre/post-processing and model methods.

    Attributes:
        signature_key: Unique identifier key for the serving signature (or sequence of keys).
        input_signature: Optional input tensor/PyTree specification signature. Defaults to None.
        tf_preprocessor: Optional preprocessing callable applied before the model method. Defaults to None.
        tf_postprocessor: Optional postprocessing callable applied after the model method. Defaults to None.
        extra_trackable_resources: Optional additional trackable resources. Defaults to None.
        method_key: Optional key selecting which method in ``JaxModule.methods`` to invoke. Defaults to None.
        obm_export_options: Optional ``ObmExportOptions`` configuration. Defaults to None.
    """

    signature_key: str | Sequence[str]
    input_signature: Sequence[PyTree] | PyTree = None
    tf_preprocessor: MethodFn | None = None
    tf_postprocessor: MethodFn | None = None
    extra_trackable_resources: PyTree = None
    method_key: str | None = None
    obm_export_options: ObmExportOptions | None = None

    def get_signature_keys(self) -> Sequence[str]:
        """Return the normalized sequence of signature keys configured on this ``ServingConfig``.

        Returns:
            Sequence[str]: List of signature key strings.
        """
        if isinstance(self.signature_key, str):
            return [self.signature_key]
        return list(self.signature_key)

    def bind(
        self,
        infer_fn: MethodFn | Mapping[str, MethodFn],
    ) -> Mapping[str, MethodFn]:
        """Bind ``infer_fn`` (or a mapping of method functions) to each configured signature key.

        Args:
            infer_fn: Single inference callable or mapping from ``method_key`` to callable.

        Returns:
            Mapping[str, MethodFn]: Mapping from each signature key to its end-to-end inference pipeline.

        Raises:
            ValueError: If ``infer_fn`` is a mapping and ``method_key`` cannot be resolved.
        """
        resolved_fn: MethodFn
        if callable(infer_fn):
            resolved_fn = infer_fn
        else:
            if self.method_key is not None:
                if self.method_key not in infer_fn:
                    raise ValueError(
                        f"Method key '{self.method_key}' not found in infer_fn mapping."
                    )
                resolved_fn = infer_fn[self.method_key]
            elif len(infer_fn) == 1:
                resolved_fn = next(iter(infer_fn.values()))
            else:
                raise ValueError(
                    "ServingConfig.method_key must be specified when JaxModule has multiple methods."
                )

        pre = self.tf_preprocessor
        post = self.tf_postprocessor

        def _pipeline(inputs: PyTree) -> PyTree:
            """Execute preprocessor, bound model inference function, and postprocessor.

            Args:
                inputs: Input PyTree passed to the serving signature.

            Returns:
                PyTree: Output PyTree after optional preprocessing, model execution, and postprocessing.
            """
            curr = pre(inputs) if pre is not None else inputs
            out = resolved_fn(curr)
            return post(out) if post is not None else out

        return {key: _pipeline for key in self.get_signature_keys()}


class JaxModule:
    """Exportable wrapper encapsulating JAX/PyTree model parameters and application functions."""

    def __init__(
        self,
        params: PyTree,
        apply_fn: ApplyFn | Mapping[str, ApplyFn],
        trainable: bool | PyTree = False,
        input_polymorphic_shape: PyTree = None,
        jax2tf_kwargs: Mapping[str, PyTree] | None = None,
        jit_compile: bool | Mapping[str, bool] = True,
        pspecs: PyTree = None,
        allow_multi_axis_sharding_consolidation: bool | None = None,
        export_version: str = "TF_SAVEDMODEL",
    ) -> None:
        """Initialize the JaxModule instance.

        Args:
            params: Model parameter PyTree.
            apply_fn: Single model application function ``(params, inputs) -> outputs`` or mapping of named functions.
            trainable: Boolean or PyTree of booleans indicating whether parameters are trainable. Defaults to False.
            input_polymorphic_shape: Optional polymorphic shape specification per method. Defaults to None.
            jax2tf_kwargs: Optional keyword arguments forwarded to conversion backends. Defaults to None.
            jit_compile: Whether to JIT-compile methods (bool or per-method mapping). Defaults to True.
            pspecs: Optional ``PartitionSpec`` PyTree for parameter sharding. Defaults to None.
            allow_multi_axis_sharding_consolidation: Optional flag for multi-axis sharding consolidation. Defaults to None.
            export_version: Target export format version identifier. Defaults to ``"TF_SAVEDMODEL"``.
        """
        self._params: PyTree = params
        if callable(apply_fn):
            self._apply_fn_map: dict[str, ApplyFn] = {"__call__": apply_fn}
        else:
            self._apply_fn_map = dict(apply_fn)
        self.trainable = trainable
        self.input_polymorphic_shape = input_polymorphic_shape
        self.jax2tf_kwargs = jax2tf_kwargs
        self.jit_compile = jit_compile
        self.pspecs = pspecs
        self.allow_multi_axis_sharding_consolidation = allow_multi_axis_sharding_consolidation
        self.export_version = export_version

    @property
    def apply_fn_map(self) -> Mapping[str, ApplyFn]:
        """Return the mapping of method names to raw ``(params, inputs)`` apply functions.

        Returns:
            Mapping[str, ApplyFn]: Method name to apply function mapping.
        """
        return dict(self._apply_fn_map)

    @property
    def model_params(self) -> PyTree:
        """Return the current model parameter PyTree.

        Returns:
            PyTree: Model parameters.
        """
        return self._params

    def update_variables(self, params: PyTree) -> None:
        """Update the model parameters held by this ``JaxModule``.

        Args:
            params: New parameter PyTree to store.
        """
        self._params = params

    def _make_bound_method(self, fn: ApplyFn) -> MethodFn:
        """Create a single-argument callable binding ``self._params`` to ``fn``.

        Args:
            fn: Two-argument function ``(params, inputs) -> outputs``.

        Returns:
            MethodFn: Bound function ``(inputs) -> outputs``.
        """

        def _bound(inputs: PyTree) -> PyTree:
            """Invoke ``fn`` with current ``self._params`` and ``inputs``.

            Args:
                inputs: Model input PyTree.

            Returns:
                PyTree: Model output PyTree.
            """
            return fn(self._params, inputs)

        return _bound

    @property
    def methods(self) -> Mapping[str, MethodFn]:
        """Return the mapping of bound export methods taking ``inputs`` and returning outputs.

        Returns:
            Mapping[str, MethodFn]: Mapping of method keys to bound callables.
        """
        return {name: self._make_bound_method(fn) for name, fn in self._apply_fn_map.items()}

    @property
    def jax_methods(self) -> Mapping[str, MethodFn]:
        """Return the mapping of bound JAX methods taking ``inputs`` and returning outputs.

        Returns:
            Mapping[str, MethodFn]: Mapping of method keys to bound JAX callables.
        """
        return self.methods


class ExportManager:
    """Manager orchestrating the export and loading of a ``JaxModule`` with ``ServingConfig``s."""

    def __init__(
        self,
        module: JaxModule,
        serving_configs: Sequence[ServingConfig],
    ) -> None:
        """Initialize the ExportManager.

        Args:
            module: The ``JaxModule`` to export.
            serving_configs: Sequence of ``ServingConfig`` definitions for serving signatures.
        """
        self._module = module
        self._serving_configs = list(serving_configs)

    @property
    def tf_module(self) -> JaxModule:
        """Return the underlying ``JaxModule`` managed by this ``ExportManager``.

        Returns:
            JaxModule: The wrapped module instance.
        """
        return self._module

    @property
    def serving_signatures(self) -> Mapping[str, MethodFn]:
        """Return the compiled mapping of signature keys to bound serving callables.

        Returns:
            Mapping[str, MethodFn]: Mapping of signature keys to serving functions.
        """
        signatures: dict[str, MethodFn] = {}
        methods_map = self._module.methods
        for config in self._serving_configs:
            bound_map = config.bind(methods_map)
            signatures.update(bound_map)
        return signatures

    def save(
        self,
        model_path: PathLike,
        save_options: PyTree = None,
        signature_overrides: Mapping[str, MethodFn] | None = None,
    ) -> None:
        """Persist the exported model parameters and serving signature metadata to ``model_path``.

        Args:
            model_path: Target directory path where the exported model artifact is written.
            save_options: Optional save options configuration. Defaults to None.
            signature_overrides: Optional override callables keyed by signature key. Defaults to None.
        """
        io = FileSystemIO()
        io.makedirs(model_path, exist_ok=True)
        sigs = dict(self.serving_signatures)
        if signature_overrides is not None:
            sigs.update(signature_overrides)
        payload: dict[str, PyTree] = {
            "export_version": self._module.export_version,
            "trainable": self._module.trainable,
            "signature_keys": sorted(sigs.keys()),
            "params": self._module.model_params,
            "save_options": save_options,
        }
        target_file = f"{str(model_path).rstrip('/')}/saved_model.msgpack"
        AtomicWrite().atomic_write_bytes(target_file, MsgpackEngine().packb(payload))

    def load(self, model_path: PathLike, **kwargs: PyTree) -> PyTree:
        """Load a previously saved model artifact from ``model_path``.

        Args:
            model_path: Directory path containing ``saved_model.msgpack``.
            **kwargs: Additional keyword options for loading.

        Returns:
            PyTree: Deserialized model artifact dictionary containing ``params`` and ``signature_keys``.

        Raises:
            FileNotFoundError: If no exported model artifact exists at ``model_path``.
        """
        _ = kwargs
        target_file = f"{str(model_path).rstrip('/')}/saved_model.msgpack"
        raw = FileSystemIO().read(target_file)
        if not raw:
            raise FileNotFoundError(f"Exported model artifact not found at {model_path}")
        return MsgpackEngine().unpackb(raw)


__all__ = [
    "ApplyFn",
    "ExportManager",
    "JaxModule",
    "MethodFn",
    "ObmExportOptions",
    "ServingConfig",
]
