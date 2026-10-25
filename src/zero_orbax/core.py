"""Orbax internal storage, serialization, and networking primitives.

This module provides strongly typed, zero-dependency internal primitives for
Orbax compatibility, including file I/O (local VFS and virtual schemes),
atomic writes, a chunked TensorStore driver, a pure-Python RFC 8949 Msgpack
engine, multi-host barrier synchronization, and global array sharding utilities.
"""

from __future__ import annotations

import contextvars
import dataclasses
import json
import os
import pathlib
import shutil
import struct
import uuid
from collections.abc import Mapping, Sequence
from typing import Union
from typing_extensions import TypeAlias

from ml_switcheroo_ir import LogicalGraph, LogicalMesh, PartitionSpec

from zero_orbax.types import (
    ArrayProtocol,
    BytesSerializableProtocol,
    GraphProtocol,
    JsonValue,
    PathLike,
    PyTree,
    PyTreeLeaf,
    array_to_nested_list,
    is_array_protocol,
    is_pytree_mapping,
    normalize_path,
)

_VIRTUAL_SCHEMES: tuple[str, ...] = ("gs://", "s3://", "memory://")
_VFS_FILES_CTX: contextvars.ContextVar[dict[str, bytes] | None] = contextvars.ContextVar(
    "_VFS_FILES_CTX",
    default=None,
)
_BARRIER_COUNTS_CTX: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "_BARRIER_COUNTS_CTX",
    default=None,
)


def _get_default_vfs_files() -> dict[str, bytes]:
    """Return the active context-local default virtual file system dictionary.

    Returns:
        dict[str, bytes]: Context-scoped virtual file dictionary.
    """
    current = _VFS_FILES_CTX.get()
    if current is None:
        current = {}
        _VFS_FILES_CTX.set(current)
    return current


def _get_default_barrier_counts() -> dict[str, int]:
    """Return the active context-local default barrier counter dictionary.

    Returns:
        dict[str, int]: Context-scoped barrier counter dictionary.
    """
    current = _BARRIER_COUNTS_CTX.get()
    if current is None:
        current = {}
        _BARRIER_COUNTS_CTX.set(current)
    return current


def _is_virtual_uri(path_str: str) -> bool:
    """Check whether a path string uses a virtual URI scheme (e.g. ``gs://``, ``s3://``, ``memory://``).

    Args:
        path_str: Raw path string to inspect.

    Returns:
        bool: True if the path starts with a virtual URI scheme.
    """
    return path_str.startswith(_VIRTUAL_SCHEMES)


def _normalize_virtual_key(key: str) -> str:
    """Normalize a virtual URI or path string by collapsing duplicate slashes and trailing slashes.

    Args:
        key: Raw path or virtual URI string.

    Returns:
        str: Normalized virtual URI or path key.
    """
    for scheme in _VIRTUAL_SCHEMES:
        if key.startswith(scheme):
            rest = key[len(scheme) :]
            parts = [p for p in rest.split("/") if p]
            return scheme + "/".join(parts)
    return key.rstrip("/") if key != "/" else key


class VirtualFileSystem:
    """Encapsulated in-memory virtual file system store for ``gs://``, ``s3://``, and ``memory://`` URIs."""

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        """Initialize the VirtualFileSystem store.

        Args:
            files: Optional isolated dictionary backing this store. Defaults to the context store when None.
        """
        self._files: dict[str, bytes] = files if files is not None else _get_default_vfs_files()

    def get(self, key: str) -> bytes | None:
        """Retrieve stored bytes for ``key`` if present.

        Args:
            key: Virtual URI or path key.

        Returns:
            bytes | None: Stored bytes, or None if not present.
        """
        return self._files.get(_normalize_virtual_key(key))

    def put(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key``.

        Args:
            key: Virtual URI or path key.
            data: Raw byte payload.
        """
        self._files[_normalize_virtual_key(key)] = data

    def has_key(self, key: str) -> bool:
        """Check whether an exact ``key`` exists in the store.

        Args:
            key: Virtual URI or path key.

        Returns:
            bool: True if ``key`` exists.
        """
        return _normalize_virtual_key(key) in self._files

    def keys(self) -> list[str]:
        """Return a snapshot list of all keys in the virtual store.

        Returns:
            list[str]: Stored keys.
        """
        return list(self._files.keys())

    def pop(self, key: str) -> bytes | None:
        """Remove and return the byte payload for ``key`` if present.

        Args:
            key: Virtual URI or path key.

        Returns:
            bytes | None: Removed bytes, or None if missing.
        """
        return self._files.pop(_normalize_virtual_key(key), None)

    def reset(self) -> None:
        """Clear all entries in this virtual file system store."""
        self._files.clear()


class FileSystemIO:
    """File system I/O handler supporting standard local VFS and virtual schemes (GCS, S3, memory)."""

    def __init__(self, vfs: VirtualFileSystem | None = None) -> None:
        """Initialize the FileSystemIO handler.

        Args:
            vfs: Optional ``VirtualFileSystem`` instance. Defaults to the default store when None.
        """
        self.vfs: VirtualFileSystem = vfs if vfs is not None else VirtualFileSystem()

    def read(self, path: PathLike, *, strict: bool = False) -> bytes:
        """Read binary data from a file path or virtual URI.

        Args:
            path: The file path or URI to read from.
            strict: If True, raise ``FileNotFoundError`` when the path does not exist. Defaults to False.

        Returns:
            bytes: The file data as bytes (or ``b""`` when not found and ``strict=False``).

        Raises:
            FileNotFoundError: If ``strict=True`` and the path does not exist.
        """
        raw_str = _normalize_virtual_key(str(path))
        cached = self.vfs.get(raw_str)
        if cached is not None:
            return cached
        if not _is_virtual_uri(raw_str):
            norm = normalize_path(path)
            if norm.is_file():
                return norm.read_bytes()
        if strict:
            raise FileNotFoundError(f"Path not found: {raw_str}")
        return b""

    def write(
        self,
        path: PathLike,
        data: bytes,
        *,
        permission_mode: int | None = None,
    ) -> None:
        """Write binary data to a file path or virtual URI.

        Args:
            path: The destination file path or URI.
            data: The raw bytes to write.
            permission_mode: Optional octal file permission mode to apply. Defaults to None.
        """
        raw_str = _normalize_virtual_key(str(path))
        if _is_virtual_uri(raw_str):
            self.vfs.put(raw_str, data)
            return
        norm = normalize_path(path)
        if norm.parent != pathlib.Path(".") and str(norm.parent):
            norm.parent.mkdir(parents=True, exist_ok=True)
        norm.write_bytes(data)
        if permission_mode is not None:
            os.chmod(norm, permission_mode)

    def exists(self, path: PathLike) -> bool:
        """Check whether a file or directory exists on disk or in the virtual store.

        Args:
            path: The path or URI to check.

        Returns:
            bool: True if the file or directory exists.
        """
        raw_str = _normalize_virtual_key(str(path))
        if self.vfs.has_key(raw_str):
            return True
        prefix = raw_str.rstrip("/") + "/"
        if any(k.startswith(prefix) for k in self.vfs.keys()):
            return True
        if _is_virtual_uri(raw_str):
            return False
        return normalize_path(path).exists()

    def makedirs(
        self,
        path: PathLike,
        exist_ok: bool = True,
        *,
        permission_mode: int | None = None,
    ) -> None:
        """Create a directory and any necessary parent directories.

        Args:
            path: Directory path to create.
            exist_ok: Whether to succeed if the directory already exists. Defaults to True.
            permission_mode: Optional octal directory permission mode. Defaults to None.
        """
        raw_str = _normalize_virtual_key(str(path))
        if not _is_virtual_uri(raw_str):
            norm = normalize_path(path)
            norm.mkdir(parents=True, exist_ok=exist_ok)
            if permission_mode is not None:
                os.chmod(norm, permission_mode)

    def listdir(self, path: PathLike) -> list[str]:
        """List immediate child entry names under a directory path.

        Args:
            path: Directory path or virtual URI prefix.

        Returns:
            list[str]: Sorted list of immediate entry names.
        """
        raw_str = _normalize_virtual_key(str(path))
        entries: set[str] = set()
        prefix = raw_str.rstrip("/") + "/"
        for key in self.vfs.keys():
            if key.startswith(prefix):
                rem = key[len(prefix) :]
                child = rem.split("/", 1)[0]
                entries.add(child)
        if not _is_virtual_uri(raw_str):
            norm = normalize_path(path)
            if norm.is_dir():
                for item in norm.iterdir():
                    entries.add(item.name)
        return sorted(entries)

    def remove(self, path: PathLike) -> None:
        """Delete a file or directory tree from disk and the virtual store.

        Args:
            path: File or directory path to remove.
        """
        raw_str = _normalize_virtual_key(str(path))
        self.vfs.pop(raw_str)
        prefix = raw_str.rstrip("/") + "/"
        for k in self.vfs.keys():
            if k.startswith(prefix):
                self.vfs.pop(k)
        if not _is_virtual_uri(raw_str):
            norm = normalize_path(path)
            if norm.is_dir():
                shutil.rmtree(norm)
            elif norm.exists():
                norm.unlink()


class AtomicWrite:
    """Atomic file write handler using temporary sibling staging and atomic rename."""

    def __init__(self, vfs: VirtualFileSystem | None = None) -> None:
        """Initialize the AtomicWrite handler.

        Args:
            vfs: Optional ``VirtualFileSystem`` instance. Defaults to the default store when None.
        """
        self.vfs: VirtualFileSystem = vfs if vfs is not None else VirtualFileSystem()

    def atomic_write_bytes(
        self,
        path: PathLike,
        data: bytes,
        *,
        permission_mode: int | None = None,
    ) -> None:
        """Atomically write bytes to a destination file path.

        Creates any missing parent directories on local filesystems, writes to a
        temporary sibling file first, and atomically replaces the target path.
        For virtual URIs (``gs://``, ``s3://``, ``memory://``), stores in ``self.vfs``.

        Args:
            path: Target file path or URI.
            data: Raw bytes to write atomically.
            permission_mode: Optional octal permission mode for the written file. Defaults to None.
        """
        raw_str = _normalize_virtual_key(str(path))
        if _is_virtual_uri(raw_str):
            self.vfs.put(raw_str, data)
            return
        norm = normalize_path(path)
        parent = norm.parent
        if parent != pathlib.Path(".") and str(parent):
            parent.mkdir(parents=True, exist_ok=True)
        tmp_name = f".{norm.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        tmp_path = parent / tmp_name
        tmp_path.write_bytes(data)
        if permission_mode is not None:
            os.chmod(tmp_path, permission_mode)
        os.replace(tmp_path, norm)


class SerializedArray:
    """Strongly typed array representation preserving ``shape``, ``dtype``, and ``tobytes()``."""

    def __init__(
        self,
        data: PyTree,
        shape: tuple[int, ...],
        dtype: str = "float32",
        raw_bytes: bytes = b"",
    ) -> None:
        """Initialize a SerializedArray.

        Args:
            data: Nested Python list or scalar data.
            shape: Tuple of integer dimensions.
            dtype: Data type string identifier. Defaults to ``"float32"``.
            raw_bytes: Optional raw byte representation. Defaults to ``b""``.
        """
        self._data: PyTree = data
        self._shape: tuple[int, ...] = shape
        self._dtype: str = dtype
        self._raw_bytes: bytes = raw_bytes

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape dimensions of the array.

        Returns:
            tuple[int, ...]: Array shape tuple.
        """
        return self._shape

    @property
    def dtype(self) -> str:
        """Return the element data type string.

        Returns:
            str: Data type identifier.
        """
        return self._dtype

    def tolist(self) -> PyTree:
        """Return the underlying nested list or scalar representation.

        Returns:
            PyTree: Nested list or scalar data.
        """
        return self._data

    def tobytes(self) -> bytes:
        """Return the raw byte payload of the array.

        Returns:
            bytes: Raw bytes.
        """
        return self._raw_bytes

    def __eq__(self, other: SerializedArray | PyTree) -> bool:
        """Compare equality against another ``SerializedArray`` or Python sequence.

        Args:
            other: Other array or PyTree value to compare.

        Returns:
            bool: True if shape, dtype, and data match.
        """
        if isinstance(other, SerializedArray):
            return (
                self._shape == other._shape
                and self._dtype == other._dtype
                and self._data == other._data
            )
        return bool(self._data == other)


@dataclasses.dataclass(eq=False)
class ReconstructedGraphNode:
    """Concrete implementation of ``GraphProtocol`` reconstructed during MessagePack decoding.

    Attributes:
        name: Graph identifier name.
        inputs: List of input tensor names.
        outputs: List of output tensor names.
    """

    name: str
    inputs: list[str]
    outputs: list[str]

    def __eq__(self, other: ReconstructedGraphNode | PyTree) -> bool:
        """Compare equality against another ``GraphProtocol`` or dictionary representation.

        Args:
            other: GraphProtocol object or dictionary to compare against.

        Returns:
            bool: True if ``name``, ``inputs``, and ``outputs`` match.
        """
        if isinstance(other, GraphProtocol):
            return (
                self.name == other.name
                and self.inputs == other.inputs
                and self.outputs == other.outputs
            )
        if isinstance(other, Mapping):
            return (
                other.get("name") == self.name
                and other.get("inputs") == self.inputs
                and other.get("outputs") == self.outputs
            )
        return False


class TensorStoreDriver:
    """Chunked array and tensor storage driver compatible with TensorStore specifications."""

    def __init__(self, vfs: VirtualFileSystem | None = None) -> None:
        """Initialize the TensorStoreDriver with an empty buffer and specification.

        Args:
            vfs: Optional ``VirtualFileSystem`` instance. Defaults to None.
        """
        self.vfs: VirtualFileSystem = vfs if vfs is not None else VirtualFileSystem()
        self.spec: dict[str, JsonValue] = {}
        self._buffer: bytes = b""
        self._path: str | None = None

    def open(self, spec: Mapping[str, JsonValue]) -> None:
        """Open a TensorStore specification and resolve any backing path.

        Args:
            spec: TensorStore specification dictionary (e.g. ``{"driver": "zarr", "kvstore": ...}``).
        """
        self.spec = dict(spec)
        io_handler = FileSystemIO(vfs=self.vfs)
        path_val = spec.get("path")
        if isinstance(path_val, str):
            self._path = path_val
            if io_handler.exists(path_val):
                self._buffer = io_handler.read(path_val)
            return

        kvstore = spec.get("kvstore")
        if isinstance(kvstore, Mapping):
            kv_path = kvstore.get("path")
            if isinstance(kv_path, str):
                self._path = kv_path
                if io_handler.exists(kv_path):
                    self._buffer = io_handler.read(kv_path)

    def read(self) -> bytes:
        """Read raw bytes from the open TensorStore.

        Returns:
            bytes: Stored bytes from disk/VFS or buffer.
        """
        if self._path is not None:
            io_handler = FileSystemIO(vfs=self.vfs)
            if io_handler.exists(self._path):
                return io_handler.read(self._path)
        return self._buffer

    def write(self, data: bytes) -> None:
        """Write raw bytes to the open TensorStore.

        Args:
            data: Raw byte payload to store.
        """
        self._buffer = data
        if self._path is not None:
            AtomicWrite(vfs=self.vfs).atomic_write_bytes(self._path, data)

    def write_tensor(
        self,
        tensor: PyTreeLeaf,
        *,
        chunks: tuple[int, ...] | None = None,
    ) -> None:
        """Serialize and store a tensor, array, or scalar leaf with optional chunk metadata.

        Args:
            tensor: ArrayProtocol, LogicalGraph, or scalar leaf to serialize.
            chunks: Optional chunk dimensions to record in the specification. Defaults to None.
        """
        if chunks is not None:
            self.spec["chunks"] = list(chunks)
        encoded = MsgpackEngine().packb(tensor)
        self.write(encoded)

    def read_tensor(
        self,
        dtype: str | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> PyTree:
        """Read and deserialize a stored tensor or scalar leaf.

        Args:
            dtype: Optional expected data type name. Defaults to None.
            shape: Optional expected shape tuple. Defaults to None.

        Returns:
            PyTree: Reconstructed tensor data or None if empty.
        """
        raw = self.read()
        if not raw:
            return None
        restored = MsgpackEngine().unpackb(raw)
        if dtype is not None or shape is not None:
            if is_pytree_mapping(restored):
                merged = dict(restored)
                if dtype is not None:
                    merged["dtype"] = dtype
                if shape is not None:
                    merged["shape"] = list(shape)
                return merged
            if isinstance(restored, (list, tuple, SerializedArray)):
                raw_data = (
                    restored.tolist() if isinstance(restored, SerializedArray) else list(restored)
                )
                inferred_shape = (
                    shape
                    if shape is not None
                    else (
                        restored.shape
                        if isinstance(restored, SerializedArray)
                        else (len(restored),)
                    )
                )
                resolved_dtype = (
                    dtype
                    if dtype is not None
                    else (restored.dtype if isinstance(restored, SerializedArray) else "float32")
                )
                return SerializedArray(
                    data=raw_data,
                    shape=inferred_shape,
                    dtype=resolved_dtype,
                )
        return restored


_MSGPACK_MAGIC = b"\x93ZMPK"
_PackedMsgpackValue: TypeAlias = Union[
    None,
    bool,
    int,
    float,
    str,
    bytes,
    list["_PackedMsgpackValue"],
    tuple["_PackedMsgpackValue", ...],
    dict[str, "_PackedMsgpackValue"],
]


def _encode_pytree_to_json_value(node: PyTree) -> _PackedMsgpackValue:
    """Recursively encode a PyTree into a tagged MessagePack-serializable structure.

    Args:
        node: PyTree node to encode.

    Returns:
        _PackedMsgpackValue: Tagged structure preserving tuples, bytes, arrays, and graphs.

    Raises:
        TypeError: If ``node`` is an unsupported type.
    """
    if node is None or isinstance(node, (bool, int, float, str, bytes)):
        return node
    if isinstance(node, complex):
        return {"__ztype__": "complex", "real": node.real, "imag": node.imag}
    if isinstance(node, LogicalGraph):
        return {
            "__ztype__": "logical_graph",
            "payload": node.to_json(),
        }
    if isinstance(node, ArrayProtocol):
        raw_bytes = node.tobytes() if isinstance(node, BytesSerializableProtocol) else b""
        return {
            "__ztype__": "array",
            "shape": [int(d) for d in node.shape],
            "dtype": str(node.dtype),
            "data": _encode_pytree_to_json_value(array_to_nested_list(node)),
            "raw": raw_bytes,
        }
    if isinstance(node, GraphProtocol):
        return {
            "__ztype__": "graph_protocol",
            "name": node.name,
            "inputs": [str(x) for x in node.inputs],
            "outputs": [str(x) for x in node.outputs],
        }
    if isinstance(node, tuple):
        return {
            "__ztype__": "tuple",
            "items": [_encode_pytree_to_json_value(item) for item in node],
        }
    if isinstance(node, Mapping):
        return {str(k): _encode_pytree_to_json_value(v) for k, v in node.items()}
    if isinstance(node, Sequence):
        return [_encode_pytree_to_json_value(item) for item in node]
    raise TypeError(f"Unsupported PyTree node type for MessagePack serialization: {type(node)}")


def _decode_json_value_to_pytree(
    value: _PackedMsgpackValue | JsonValue,
    *,
    preserve_array: bool = False,
) -> PyTree:
    """Recursively decode a tagged MessagePack/JSON structure back into a PyTree.

    Args:
        value: Tagged structure produced by ``_encode_pytree_to_json_value``.
        preserve_array: If True, reconstruct ``SerializedArray`` instances for array nodes. Defaults to False.

    Returns:
        PyTree: Reconstructed PyTree structure.
    """
    if isinstance(value, bytes):
        return value
    if isinstance(value, list):
        return [_decode_json_value_to_pytree(item, preserve_array=preserve_array) for item in value]
    if isinstance(value, dict):
        tag = value.get("__ztype__")
        hex_val = value.get("hex")
        if tag == "bytes" and isinstance(hex_val, str):
            return bytes.fromhex(hex_val)
        real_val = value.get("real")
        imag_val = value.get("imag")
        if (
            tag == "complex"
            and isinstance(real_val, (int, float))
            and isinstance(imag_val, (int, float))
        ):
            return complex(float(real_val), float(imag_val))
        items_val = value.get("items")
        if tag == "tuple" and isinstance(items_val, list):
            return tuple(
                _decode_json_value_to_pytree(item, preserve_array=preserve_array)
                for item in items_val
            )
        payload_val = value.get("payload")
        if tag == "logical_graph" and isinstance(payload_val, str):
            return LogicalGraph.from_json(payload_val)
        if tag == "graph_protocol":
            name_val = value.get("name")
            inputs_val = value.get("inputs")
            outputs_val = value.get("outputs")
            return ReconstructedGraphNode(
                name=str(name_val) if name_val is not None else "",
                inputs=[str(x) for x in inputs_val] if isinstance(inputs_val, list) else [],
                outputs=[str(x) for x in outputs_val] if isinstance(outputs_val, list) else [],
            )
        if tag == "array":
            decoded_data = _decode_json_value_to_pytree(
                value.get("data"),
                preserve_array=preserve_array,
            )
            if preserve_array:
                raw_shape = value.get("shape")
                shape_tuple = (
                    tuple(int(dim) for dim in raw_shape if isinstance(dim, int))
                    if isinstance(raw_shape, list)
                    else ()
                )
                dtype_str = str(value.get("dtype", "float32"))
                raw_field = value.get("raw")
                raw_bytes = (
                    raw_field
                    if isinstance(raw_field, bytes)
                    else (bytes.fromhex(hex_val) if isinstance(hex_val, str) and hex_val else b"")
                )
                return SerializedArray(
                    data=decoded_data,
                    shape=shape_tuple,
                    dtype=dtype_str,
                    raw_bytes=raw_bytes,
                )
            return decoded_data
        return {
            k: _decode_json_value_to_pytree(v, preserve_array=preserve_array)
            for k, v in value.items()
        }
    return value


def _pack_msgpack_bytes(value: _PackedMsgpackValue | JsonValue) -> bytes:
    """Encode a tagged value into RFC 8949 / MessagePack binary bytes.

    Args:
        value: Tagged value to encode.

    Returns:
        bytes: Binary MessagePack representation.
    """
    if value is None:
        return b"\xc0"
    if value is False:
        return b"\xc2"
    if value is True:
        return b"\xc3"
    if isinstance(value, int):
        if 0 <= value <= 0x7F:
            return struct.pack(">B", value)
        if -32 <= value < 0:
            return struct.pack(">b", value)
        if 0 <= value <= 0xFF:
            return b"\xcc" + struct.pack(">B", value)
        if 0 <= value <= 0xFFFF:
            return b"\xcd" + struct.pack(">H", value)
        if 0 <= value <= 0xFFFFFFFF:
            return b"\xce" + struct.pack(">I", value)
        if 0 <= value <= 0xFFFFFFFFFFFFFFFF:
            return b"\xcf" + struct.pack(">Q", value)
        if -128 <= value < 0:
            return b"\xd0" + struct.pack(">b", value)
        if -32768 <= value < 0:
            return b"\xd1" + struct.pack(">h", value)
        if -2147483648 <= value < 0:
            return b"\xd2" + struct.pack(">i", value)
        return b"\xd3" + struct.pack(">q", value)
    if isinstance(value, float):
        return b"\xcb" + struct.pack(">d", value)
    if isinstance(value, bytes):
        n = len(value)
        if n <= 0xFF:
            return b"\xc4" + struct.pack(">B", n) + value
        if n <= 0xFFFF:
            return b"\xc5" + struct.pack(">H", n) + value
        return b"\xc6" + struct.pack(">I", n) + value
    if isinstance(value, str):
        raw = value.encode("utf-8")
        n = len(raw)
        if n <= 31:
            return struct.pack(">B", 0xA0 | n) + raw
        if n <= 0xFF:
            return b"\xd9" + struct.pack(">B", n) + raw
        if n <= 0xFFFF:
            return b"\xda" + struct.pack(">H", n) + raw
        return b"\xdb" + struct.pack(">I", n) + raw
    if isinstance(value, (list, tuple)):
        n = len(value)
        if n <= 15:
            header = struct.pack(">B", 0x90 | n)
        elif n <= 0xFFFF:
            header = b"\xdc" + struct.pack(">H", n)
        else:
            header = b"\xdd" + struct.pack(">I", n)
        return header + b"".join(_pack_msgpack_bytes(item) for item in value)
    n = len(value)
    if n <= 15:
        header = struct.pack(">B", 0x80 | n)
    elif n <= 0xFFFF:
        header = b"\xde" + struct.pack(">H", n)
    else:
        header = b"\xdf" + struct.pack(">I", n)
    parts: list[bytes] = [header]
    for k in sorted(value.keys()):
        parts.append(_pack_msgpack_bytes(k))
        parts.append(_pack_msgpack_bytes(value[k]))
    return b"".join(parts)


def _unpack_msgpack_bytes(data: bytes, offset: int = 0) -> tuple[_PackedMsgpackValue, int]:
    """Decode a binary MessagePack value starting at ``offset``.

    Args:
        data: Full binary MessagePack byte buffer.
        offset: Starting byte index. Defaults to 0.

    Returns:
        tuple[_PackedMsgpackValue, int]: Decoded ``(value, next_offset)`` pair.

    Raises:
        ValueError: If the buffer is truncated or contains an invalid marker.
    """
    if offset >= len(data):
        raise ValueError("Unexpected end of MessagePack buffer.")
    marker = data[offset]
    pos = offset + 1

    if marker <= 0x7F:
        return marker, pos
    if marker >= 0xE0:
        return marker - 256, pos
    if 0x80 <= marker <= 0x8F:
        count = marker & 0x0F
        return _unpack_msgpack_map(data, pos, count)
    if 0x90 <= marker <= 0x9F:
        count = marker & 0x0F
        return _unpack_msgpack_array(data, pos, count)
    if 0xA0 <= marker <= 0xBF:
        length = marker & 0x1F
        end = pos + length
        if end > len(data):
            raise ValueError("Truncated fixstr in MessagePack buffer.")
        return data[pos:end].decode("utf-8"), end
    if marker == 0xC0:
        return None, pos
    if marker == 0xC2:
        return False, pos
    if marker == 0xC3:
        return True, pos
    if marker == 0xC4:
        if pos + 1 > len(data):
            raise ValueError("Truncated bin8 length in MessagePack buffer.")
        length = data[pos]
        start = pos + 1
        end = start + length
        if end > len(data):
            raise ValueError("Truncated bin8 payload in MessagePack buffer.")
        return data[start:end], end
    if marker == 0xC5:
        if pos + 2 > len(data):
            raise ValueError("Truncated bin16 length in MessagePack buffer.")
        length = struct.unpack(">H", data[pos : pos + 2])[0]
        start = pos + 2
        end = start + length
        if end > len(data):
            raise ValueError("Truncated bin16 payload in MessagePack buffer.")
        return data[start:end], end
    if marker == 0xC6:
        if pos + 4 > len(data):
            raise ValueError("Truncated bin32 length in MessagePack buffer.")
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        start = pos + 4
        end = start + length
        if end > len(data):
            raise ValueError("Truncated bin32 payload in MessagePack buffer.")
        return data[start:end], end
    if marker == 0xCB:
        if pos + 8 > len(data):
            raise ValueError("Truncated float64 in MessagePack buffer.")
        val_f: float = struct.unpack(">d", data[pos : pos + 8])[0]
        return val_f, pos + 8
    if marker == 0xCC:
        if pos + 1 > len(data):
            raise ValueError("Truncated uint8 in MessagePack buffer.")
        return data[pos], pos + 1
    if marker == 0xCD:
        if pos + 2 > len(data):
            raise ValueError("Truncated uint16 in MessagePack buffer.")
        val_u16: int = struct.unpack(">H", data[pos : pos + 2])[0]
        return val_u16, pos + 2
    if marker == 0xCE:
        if pos + 4 > len(data):
            raise ValueError("Truncated uint32 in MessagePack buffer.")
        val_u32: int = struct.unpack(">I", data[pos : pos + 4])[0]
        return val_u32, pos + 4
    if marker == 0xCF:
        if pos + 8 > len(data):
            raise ValueError("Truncated uint64 in MessagePack buffer.")
        val_u64: int = struct.unpack(">Q", data[pos : pos + 8])[0]
        return val_u64, pos + 8
    if marker == 0xD0:
        if pos + 1 > len(data):
            raise ValueError("Truncated int8 in MessagePack buffer.")
        val_i8: int = struct.unpack(">b", data[pos : pos + 1])[0]
        return val_i8, pos + 1
    if marker == 0xD1:
        if pos + 2 > len(data):
            raise ValueError("Truncated int16 in MessagePack buffer.")
        val_i16: int = struct.unpack(">h", data[pos : pos + 2])[0]
        return val_i16, pos + 2
    if marker == 0xD2:
        if pos + 4 > len(data):
            raise ValueError("Truncated int32 in MessagePack buffer.")
        val_i32: int = struct.unpack(">i", data[pos : pos + 4])[0]
        return val_i32, pos + 4
    if marker == 0xD3:
        if pos + 8 > len(data):
            raise ValueError("Truncated int64 in MessagePack buffer.")
        val_i64: int = struct.unpack(">q", data[pos : pos + 8])[0]
        return val_i64, pos + 8
    if marker == 0xD9:
        if pos + 1 > len(data):
            raise ValueError("Truncated str8 length in MessagePack buffer.")
        length = data[pos]
        start = pos + 1
        end = start + length
        if end > len(data):
            raise ValueError("Truncated str8 payload in MessagePack buffer.")
        return data[start:end].decode("utf-8"), end
    if marker == 0xDA:
        if pos + 2 > len(data):
            raise ValueError("Truncated str16 length in MessagePack buffer.")
        length = struct.unpack(">H", data[pos : pos + 2])[0]
        start = pos + 2
        end = start + length
        if end > len(data):
            raise ValueError("Truncated str16 payload in MessagePack buffer.")
        return data[start:end].decode("utf-8"), end
    if marker == 0xDB:
        if pos + 4 > len(data):
            raise ValueError("Truncated str32 length in MessagePack buffer.")
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        start = pos + 4
        end = start + length
        if end > len(data):
            raise ValueError("Truncated str32 payload in MessagePack buffer.")
        return data[start:end].decode("utf-8"), end
    if marker == 0xDC:
        if pos + 2 > len(data):
            raise ValueError("Truncated array16 length in MessagePack buffer.")
        count = struct.unpack(">H", data[pos : pos + 2])[0]
        return _unpack_msgpack_array(data, pos + 2, count)
    if marker == 0xDD:
        if pos + 4 > len(data):
            raise ValueError("Truncated array32 length in MessagePack buffer.")
        count = struct.unpack(">I", data[pos : pos + 4])[0]
        return _unpack_msgpack_array(data, pos + 4, count)
    if marker == 0xDE:
        if pos + 2 > len(data):
            raise ValueError("Truncated map16 length in MessagePack buffer.")
        count = struct.unpack(">H", data[pos : pos + 2])[0]
        return _unpack_msgpack_map(data, pos + 2, count)
    if marker == 0xDF:
        if pos + 4 > len(data):
            raise ValueError("Truncated map32 length in MessagePack buffer.")
        count = struct.unpack(">I", data[pos : pos + 4])[0]
        return _unpack_msgpack_map(data, pos + 4, count)
    raise ValueError(f"Unsupported MessagePack marker: 0x{marker:02x}")


def _unpack_msgpack_array(
    data: bytes, pos: int, count: int
) -> tuple[list[_PackedMsgpackValue], int]:
    """Decode ``count`` sequential MessagePack array elements starting at ``pos``.

    Args:
        data: Full binary MessagePack byte buffer.
        pos: Starting offset of the first element.
        count: Number of elements in the array.

    Returns:
        tuple[list[_PackedMsgpackValue], int]: Decoded list and next byte offset.
    """
    items: list[_PackedMsgpackValue] = []
    curr = pos
    for _ in range(count):
        item, curr = _unpack_msgpack_bytes(data, curr)
        items.append(item)
    return items, curr


def _unpack_msgpack_map(
    data: bytes,
    pos: int,
    count: int,
) -> tuple[dict[str, _PackedMsgpackValue], int]:
    """Decode ``count`` key-value pairs from a MessagePack map starting at ``pos``.

    Args:
        data: Full binary MessagePack byte buffer.
        pos: Starting offset of the first key.
        count: Number of key-value entries in the map.

    Returns:
        tuple[dict[str, _PackedMsgpackValue], int]: Decoded dictionary and next byte offset.
    """
    result: dict[str, _PackedMsgpackValue] = {}
    curr = pos
    for _ in range(count):
        key, curr = _unpack_msgpack_bytes(data, curr)
        val, curr = _unpack_msgpack_bytes(data, curr)
        result[str(key)] = val
    return result, curr


class MsgpackEngine:
    """Pure-Python RFC 8949 binary MessagePack PyTree encoder and decoder."""

    def __init__(self, *, preserve_array: bool = False) -> None:
        """Initialize the MsgpackEngine.

        Args:
            preserve_array: If True, decode serialized arrays into ``SerializedArray`` instances.
                Defaults to False.
        """
        self.preserve_array: bool = preserve_array

    def packb(self, tree: PyTree) -> bytes:
        r"""Pack a PyTree into deterministic RFC 8949 binary MessagePack framing.

        ``None`` is encoded as the standard MessagePack ``nil`` byte (``b"\xc0"``)
        and ``{}`` is encoded as the empty fixmap (``b"\x80"``), ensuring
        saved ``None`` and ``{}`` values are never confused with missing/empty files.

        Args:
            tree: The PyTree structure to pack.

        Returns:
            bytes: Packed binary representation.
        """
        encoded = _encode_pytree_to_json_value(tree)
        return _pack_msgpack_bytes(encoded)

    def unpackb(self, data: bytes) -> PyTree:
        """Unpack binary MessagePack bytes (or legacy ``_MSGPACK_MAGIC`` / JSON bytes) into a PyTree.

        Args:
            data: Packed binary bytes produced by ``packb``.

        Returns:
            PyTree: Reconstructed PyTree, or ``None`` if ``data`` is empty or invalid.
        """
        if not data:
            return None
        if data.startswith(_MSGPACK_MAGIC):
            payload = data[len(_MSGPACK_MAGIC) :]
            try:
                decoded_legacy: JsonValue = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None
            return _decode_json_value_to_pytree(
                decoded_legacy,
                preserve_array=self.preserve_array,
            )
        try:
            decoded_mp, end_pos = _unpack_msgpack_bytes(data, 0)
            if end_pos == len(data):
                return _decode_json_value_to_pytree(
                    decoded_mp,
                    preserve_array=self.preserve_array,
                )
        except (ValueError, UnicodeDecodeError):
            pass
        try:
            decoded_json: JsonValue = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return _decode_json_value_to_pytree(
            decoded_json,
            preserve_array=self.preserve_array,
        )


class BarrierSync:
    """Multi-host barrier synchronization primitive with encapsulated counter state."""

    def __init__(self, counts: dict[str, int] | None = None) -> None:
        """Initialize the BarrierSync primitive.

        Args:
            counts: Optional isolated dictionary of barrier counts. Defaults to the context store when None.
        """
        self._counts: dict[str, int] = (
            counts if counts is not None else _get_default_barrier_counts()
        )

    def wait(self, id_str: str, timeout: int = 300) -> None:
        """Wait for a named synchronization barrier across hosts.

        Args:
            id_str: Unique identifier for the barrier synchronization point.
            timeout: Timeout duration in seconds. Must be non-negative. Defaults to 300.

        Raises:
            ValueError: If ``timeout`` is negative.
        """
        if timeout < 0:
            raise ValueError(f"Barrier timeout must be non-negative, got {timeout}.")
        self._counts[id_str] = self._counts.get(id_str, 0) + 1

    def count(self, id_str: str) -> int:
        """Return the number of times a barrier ID has been reached.

        Args:
            id_str: Unique identifier for the barrier synchronization point.

        Returns:
            int: Number of recorded barrier waits for ``id_str``.
        """
        return self._counts.get(id_str, 0)

    def reset(self) -> None:
        """Reset all recorded barrier counts."""
        self._counts.clear()


def _slice_along_axis(
    data: PyTree,
    axis: int,
    host_id: int,
    num_shards: int,
) -> PyTree:
    """Slice a nested Python list or tuple along ``axis`` for ``host_id`` out of ``num_shards``.

    Args:
        data: Nested list or tuple representing an N-dimensional array.
        axis: Target axis index (0-based).
        host_id: Shard index ``0 <= host_id < num_shards``.
        num_shards: Number of shards along ``axis``.

    Returns:
        PyTree: Sliced nested list or tuple.
    """
    if isinstance(data, (list, tuple)) and len(data) > 0:
        total = len(data)
        if axis == 0:
            chunk = max(1, total // num_shards)
            start = host_id * chunk
            end = total if host_id == num_shards - 1 else min(total, start + chunk)
            sliced = data[start:end]
            return tuple(sliced) if isinstance(data, tuple) else list(sliced)
        sub_sliced = [_slice_along_axis(elem, axis - 1, host_id, num_shards) for elem in data]
        return tuple(sub_sliced) if isinstance(data, tuple) else sub_sliced
    return data


def _infer_nested_shape(data: PyTree) -> tuple[int, ...]:
    """Infer the shape tuple of a nested Python list/tuple.

    Args:
        data: Nested list/tuple or scalar.

    Returns:
        tuple[int, ...]: Inferred dimensions.
    """
    dims: list[int] = []
    curr: PyTree = data
    while isinstance(curr, (list, tuple)):
        dims.append(len(curr))
        if len(curr) == 0:
            break
        curr = curr[0]
    return tuple(dims)


class GlobalArraySharding:
    """Global array sharding utility for extracting host-local tensor shards."""

    def get_local_shard(
        self,
        array: PyTree,
        host_id: int,
        num_hosts: int = 1,
        partition_spec: PartitionSpec | None = None,
        mesh: LogicalMesh | None = None,
    ) -> PyTree:
        """Extract the local shard of a global array for a specific host.

        Supports 1D and multi-axis ``PartitionSpec`` + ``LogicalMesh`` sharding across
        lists, tuples, and ``ArrayProtocol`` / ``SerializedArray`` instances.

        Args:
            array: The global array, sequence, or PyTree to shard.
            host_id: The host index to extract the shard for.
            num_hosts: Total number of hosts in the mesh. Defaults to 1.
            partition_spec: Optional ``ml_switcheroo_ir.PartitionSpec`` describing axis partitioning.
            mesh: Optional ``ml_switcheroo_ir.LogicalMesh`` defining mesh axis sizes. Defaults to None.

        Returns:
            PyTree: The local shard corresponding to ``host_id``, or ``None`` if ``array`` is ``None``.

        Raises:
            ValueError: If ``host_id < 0`` or ``num_hosts <= 0`` or ``host_id >= num_hosts``.
        """
        effective_hosts = num_hosts
        target_axis = 0
        if mesh is not None and partition_spec is not None and len(partition_spec.axes) > 0:
            for idx, axis_spec in enumerate(partition_spec.axes):
                if isinstance(axis_spec, str):
                    factor = max(1, mesh.shape.get(axis_spec, 1))
                    target_axis = idx
                    effective_hosts = max(effective_hosts, factor)
                    break
                if axis_spec is not None:
                    axis_factor = 1
                    for axis_name in axis_spec:
                        axis_factor *= max(1, mesh.shape.get(axis_name, 1))
                    target_axis = idx
                    effective_hosts = max(effective_hosts, axis_factor)
                    break

        if effective_hosts <= 0 or host_id < 0 or host_id >= effective_hosts:
            raise ValueError(f"Invalid host_id={host_id} for num_hosts={effective_hosts}.")
        if array is None:
            return None
        if effective_hosts == 1 and (partition_spec is None or len(partition_spec.axes) == 0):
            return array
        if isinstance(array, (list, tuple)):
            return _slice_along_axis(array, target_axis, host_id, effective_hosts)
        if is_array_protocol(array):
            data_list = array_to_nested_list(array)
            if isinstance(data_list, list) and len(data_list) >= (
                effective_hosts if target_axis == 0 else 1
            ):
                sliced_data = _slice_along_axis(data_list, target_axis, host_id, effective_hosts)
                if isinstance(array, SerializedArray):
                    return SerializedArray(
                        data=sliced_data,
                        shape=_infer_nested_shape(sliced_data),
                        dtype=str(array.dtype),
                    )
                return sliced_data
        return array


__all__ = [
    "AtomicWrite",
    "BarrierSync",
    "FileSystemIO",
    "GlobalArraySharding",
    "MsgpackEngine",
    "ReconstructedGraphNode",
    "SerializedArray",
    "TensorStoreDriver",
    "VirtualFileSystem",
]
