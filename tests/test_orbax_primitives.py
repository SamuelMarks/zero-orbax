"""Tests for core storage, serialization, and sharding primitives in zero_orbax.core."""

from __future__ import annotations

import pathlib

import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalMesh, LogicalNode, PartitionSpec

from zero_orbax.core import (
    AtomicWrite,
    BarrierSync,
    FileSystemIO,
    GlobalArraySharding,
    MsgpackEngine,
    ReconstructedGraphNode,
    SerializedArray,
    TensorStoreDriver,
    VirtualFileSystem,
    _encode_pytree_to_json_value,
    _pack_msgpack_bytes,
    _unpack_msgpack_bytes,
)
from zero_orbax.types import (
    GraphProtocol,
    JsonValue,
    PyTree,
    is_array_protocol,
    is_scalar_leaf,
    normalize_path,
)


class SampleArray:
    """Strongly typed test implementation of ``ArrayProtocol`` and ``BytesSerializableProtocol``."""

    def __init__(
        self,
        data: list[int],
        shape: tuple[int, ...] = (3,),
        dtype: str = "int32",
    ) -> None:
        """Initialize the SampleArray.

        Args:
            data: Integer elements of the array.
            shape: Shape tuple. Defaults to ``(3,)``.
            dtype: Data type string. Defaults to ``"int32"``.
        """
        self._data = data
        self.shape = shape
        self.dtype = dtype

    def tolist(self) -> PyTree:
        """Return the underlying list of elements.

        Returns:
            PyTree: List of elements.
        """
        return list(self._data)

    def tobytes(self) -> bytes:
        """Return a byte representation of the elements.

        Returns:
            bytes: Raw bytes.
        """
        return bytes(x % 256 for x in self._data)


class CustomPathObj:
    """Test object implementing ``PathLikeProtocol``."""

    def __init__(self, path_str: str) -> None:
        """Initialize with a string path.

        Args:
            path_str: Underlying path string.
        """
        self._path_str = path_str

    def __fspath__(self) -> str:
        """Return the string path representation.

        Returns:
            str: Path string.
        """
        return self._path_str


class SampleGraphObj:
    """Test object implementing ``GraphProtocol``."""

    def __init__(self) -> None:
        """Initialize the SampleGraphObj."""
        self.name: str = "sample_ir"
        self.inputs: list[str] = ["x"]
        self.outputs: list[str] = ["y"]


class UnsupportedCustomObj:
    """Class not implementing any supported PyTree protocol for negative testing."""


def test_type_helpers_and_normalize_path(tmp_path: pathlib.Path) -> None:
    """Test is_scalar_leaf, is_array_protocol, and normalize_path with various inputs.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    assert is_scalar_leaf(None)
    assert is_scalar_leaf(42)
    assert is_scalar_leaf(3.14)
    assert is_scalar_leaf("hello")
    assert is_scalar_leaf(b"bytes")
    assert is_scalar_leaf(1 + 2j)
    assert not is_scalar_leaf([1, 2])

    arr = SampleArray([1, 2, 3])
    assert is_array_protocol(arr)
    assert not is_array_protocol([1, 2, 3])

    p1 = normalize_path(tmp_path)
    p2 = normalize_path(str(tmp_path))
    p3 = normalize_path(CustomPathObj(str(tmp_path)))
    assert p1 == p2 == p3


def test_filesystem_io_local_and_virtual(tmp_path: pathlib.Path) -> None:
    """Test FileSystemIO read, write, exists, makedirs, listdir, and remove on local and virtual paths.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    vfs = VirtualFileSystem({})
    io = FileSystemIO(vfs=vfs)
    local_dir = tmp_path / "subdir"
    io.makedirs(local_dir, exist_ok=True, permission_mode=0o755)
    assert io.exists(local_dir)

    local_file = local_dir / "data.bin"
    io.write(local_file, b"hello_local", permission_mode=0o644)
    assert io.exists(local_file)
    assert io.read(local_file) == b"hello_local"
    assert io.listdir(local_dir) == ["data.bin"]

    io.write("data_rel.bin", b"rel")
    io.remove("data_rel.bin")

    io.remove(local_file)
    assert not io.exists(local_file)
    assert io.read(local_file) == b""
    with pytest.raises(FileNotFoundError):
        io.read(local_file, strict=True)

    io.write(local_dir / "nested.bin", b"123")
    io.remove(local_dir)
    assert not io.exists(local_dir)

    # Virtual GCS/S3/memory URI paths and trailing/duplicate slash normalization
    gcs_dir = "gs://test-bucket//checkpoints/"
    gcs_file = "gs://test-bucket/checkpoints//step_1/state.bin/"
    io.write("gs://other-bucket/unrelated.bin", b"other")
    io.makedirs(gcs_dir)
    assert not io.exists("gs://test-bucket/nonexistent")
    assert io.read("gs://test-bucket/nonexistent") == b""
    io.write(gcs_file, b"virtual_payload")
    assert io.exists("gs://test-bucket/checkpoints/step_1/state.bin")
    assert io.exists("gs://test-bucket/checkpoints")
    assert io.read("gs://test-bucket/checkpoints/step_1/state.bin") == b"virtual_payload"
    assert io.listdir("gs://test-bucket/checkpoints") == ["step_1"]
    assert io.listdir(tmp_path / "nonexistent_local_dir") == []
    io.remove(gcs_dir)
    assert not io.exists(gcs_file)
    vfs.reset()


def test_atomic_write(tmp_path: pathlib.Path) -> None:
    """Test AtomicWrite on existing directories, auto-created parent directories, and virtual URIs.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    vfs = VirtualFileSystem({})
    aw = AtomicWrite(vfs=vfs)
    target = tmp_path / "atomic_test.bin"
    aw.atomic_write_bytes(target, b"atomic_data", permission_mode=0o644)
    assert target.read_bytes() == b"atomic_data"

    nested = tmp_path / "missing_parent" / "file.bin"
    aw.atomic_write_bytes(nested, b"auto_parent")
    assert nested.read_bytes() == b"auto_parent"

    aw.atomic_write_bytes("rel_atomic.bin", b"rel")
    FileSystemIO().remove("rel_atomic.bin")

    aw.atomic_write_bytes("memory://in_mem/file.bin", b"mem_bytes")
    assert FileSystemIO(vfs=vfs).read("memory://in_mem/file.bin") == b"mem_bytes"


def test_tensorstore_driver(tmp_path: pathlib.Path) -> None:
    """Test TensorStoreDriver open, read, write, write_tensor, and read_tensor.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    vfs = VirtualFileSystem({})
    ts = TensorStoreDriver(vfs=vfs)
    assert ts.read() == b""
    assert ts.read_tensor() is None

    ts.write(b"buffer_only")
    assert ts.read() == b"buffer_only"

    disk_path = tmp_path / "tensor.ts"
    ts.open({"driver": "zarr", "path": str(disk_path)})
    arr = SampleArray([10, 20, 30], shape=(3,), dtype="int32")
    ts.write_tensor(arr, chunks=(3,))
    assert ts.spec["chunks"] == [3]
    assert disk_path.exists()

    ts2 = TensorStoreDriver(vfs=vfs)
    ts2.open({"driver": "zarr", "path": str(disk_path)})
    assert ts2.read_tensor() == [10, 20, 30]
    coerced_arr = ts2.read_tensor(dtype="float64", shape=(3,))
    assert isinstance(coerced_arr, SerializedArray)
    assert coerced_arr.dtype == "float64"
    assert coerced_arr.shape == (3,)
    assert coerced_arr == [10, 20, 30]
    assert coerced_arr == SerializedArray([10, 20, 30], shape=(3,), dtype="float64")
    assert ts2.read_tensor(dtype="int64") == [10, 20, 30]

    ts3 = TensorStoreDriver(vfs=vfs)
    ts3.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(disk_path)}})
    assert ts3.read_tensor() == [10, 20, 30]

    ts4 = TensorStoreDriver(vfs=vfs)
    mem_uri = "memory://ts_kv/array.ts"
    FileSystemIO(vfs=vfs).write(mem_uri, MsgpackEngine().packb({"val": 1}))
    ts4.open({"driver": "zarr", "kvstore": {"path": mem_uri}})
    typed_res = ts4.read_tensor(dtype="float32", shape=(2, 2))
    assert typed_res == {"val": 1, "dtype": "float32", "shape": [2, 2]}
    assert ts4.read_tensor(dtype="float32") == {"val": 1, "dtype": "float32"}
    assert ts4.read_tensor(shape=(2, 2)) == {"val": 1, "shape": [2, 2]}

    ts5 = TensorStoreDriver(vfs=vfs)
    mem_direct = "memory://ts_direct/array.ts"
    FileSystemIO(vfs=vfs).write(mem_direct, MsgpackEngine().packb(123))
    ts5.open({"path": mem_direct})
    assert ts5.read_tensor(dtype="int32") == 123

    ts6 = TensorStoreDriver(vfs=vfs)
    ts6.open({})
    ts6.open({"driver": "zarr", "kvstore": {"driver": "memory"}})
    ts6.open({"driver": "zarr", "path": str(tmp_path / "missing_file.ts")})
    ts6.open({"driver": "zarr", "kvstore": {"path": str(tmp_path / "missing_kv.ts")}})


def test_msgpack_engine_roundtrips_and_binary_framing() -> None:
    """Test MsgpackEngine RFC 8949 binary framing across all integer/string/bin/array/map sizes and types."""
    engine = MsgpackEngine()
    assert engine.packb(None) == b"\xc0"
    assert engine.unpackb(b"\xc0") is None
    assert engine.unpackb(b"") is None
    assert engine.unpackb(b"\xff\xfe\xfd_invalid") is None
    assert engine.unpackb(b"\xc1_invalid_marker") is None
    assert engine.unpackb(b"\x93ZMPK\xff\xfe") is None

    # Empty dictionary round-trips losslessly!
    packed_empty = engine.packb({})
    assert packed_empty == b"\x80"
    assert engine.unpackb(packed_empty) == {}

    # Legacy _MSGPACK_MAGIC and plain JSON backward compatibility (including legacy hex bytes & array)
    assert engine.unpackb(b'\x93ZMPK{"legacy":1}') == {"legacy": 1}
    assert engine.unpackb(b'{"__ztype__": "bytes", "hex": "0102"}') == b"\x01\x02"
    preserving_engine = MsgpackEngine(preserve_array=True)
    legacy_arr = preserving_engine.unpackb(
        b'{"__ztype__": "array", "shape": [2], "dtype": "int32", "data": [1, 2], "hex": "0102"}'
    )
    assert isinstance(legacy_arr, SerializedArray)
    assert legacy_arr.tobytes() == b"\x01\x02"

    with pytest.raises(TypeError, match="Unsupported PyTree node type"):
        bad_dict: dict[str, PyTree] = {}
        object_map: dict[str, UnsupportedCustomObj] = {"bad": UnsupportedCustomObj()}
        bad_dict.update(object_map)  # runtime dict contains UnsupportedCustomObj
        _encode_pytree_to_json_value(bad_dict)

    graph = LogicalGraph(
        name="test_graph",
        nodes={"n1": LogicalNode(id="n1", op_type="Add")},
    )
    arr = SampleArray([4, 5, 6], shape=(3,), dtype="int32")
    sample_graph_proto = SampleGraphObj()
    complex_tree: PyTree = {
        "int_pos": 7,
        "int_neg_fix": -10,
        "uint8": 200,
        "uint16": 40000,
        "uint32": 3000000000,
        "uint64": 9000000000000000000,
        "int8": -100,
        "int16": -20000,
        "int32": -1000000000,
        "int64": -5000000000000000000,
        "float": 2.5,
        "bool_t": True,
        "bool_f": False,
        "none": None,
        "str_short": "orbax",
        "str8": "a" * 100,
        "str16": "b" * 300,
        "bin8": b"\x00\x01\xff",
        "bin16": b"x" * 300,
        "bin32": b"y" * 70000,
        "complex": 3.0 - 4.0j,
        "tuple": (1, "two", b"three"),
        "list": [10, 20],
        "array": arr,
        "graph": graph,
        "graph_proto": sample_graph_proto,
    }
    packed = engine.packb(complex_tree)
    unpacked = engine.unpackb(packed)
    assert isinstance(unpacked, dict)
    assert unpacked["int_pos"] == 7
    assert unpacked["int_neg_fix"] == -10
    assert unpacked["uint8"] == 200
    assert unpacked["uint16"] == 40000
    assert unpacked["uint32"] == 3000000000
    assert unpacked["uint64"] == 9000000000000000000
    assert unpacked["int8"] == -100
    assert unpacked["int16"] == -20000
    assert unpacked["int32"] == -1000000000
    assert unpacked["int64"] == -5000000000000000000
    assert unpacked["float"] == 2.5
    assert unpacked["bool_t"] is True
    assert unpacked["bool_f"] is False
    assert unpacked["none"] is None
    assert unpacked["str_short"] == "orbax"
    assert unpacked["str8"] == "a" * 100
    assert unpacked["str16"] == "b" * 300
    assert unpacked["bin8"] == b"\x00\x01\xff"
    assert unpacked["bin16"] == b"x" * 300
    assert unpacked["bin32"] == b"y" * 70000
    assert unpacked["complex"] == 3.0 - 4.0j
    assert unpacked["tuple"] == (1, "two", b"three")
    assert unpacked["list"] == [10, 20]
    assert unpacked["array"] == [4, 5, 6]
    assert isinstance(unpacked["graph"], LogicalGraph)
    assert unpacked["graph"].name == "test_graph"
    graph_node = unpacked["graph_proto"]
    assert isinstance(graph_node, ReconstructedGraphNode)
    assert isinstance(graph_node, GraphProtocol)
    assert graph_node.name == "sample_ir"
    assert graph_node == sample_graph_proto
    assert graph_node == {"name": "sample_ir", "inputs": ["x"], "outputs": ["y"]}
    assert graph_node != 123

    # Test preserve_array=True reconstructing SerializedArray
    preserved_unpacked = preserving_engine.unpackb(preserving_engine.packb({"arr": arr}))
    assert isinstance(preserved_unpacked, dict)
    restored_arr = preserved_unpacked["arr"]
    assert isinstance(restored_arr, SerializedArray)
    assert restored_arr.shape == (3,)
    assert restored_arr.dtype == "int32"
    assert restored_arr.tolist() == [4, 5, 6]
    assert restored_arr.tobytes() == bytes([4, 5, 6])

    # Test array16 and map16 framing
    list_20 = list(range(20))
    assert engine.unpackb(engine.packb(list_20)) == list_20
    map_20 = {f"k_{i}": i for i in range(20)}
    assert engine.unpackb(engine.packb(map_20)) == map_20

    # Test str32, array32, map32 headers and error branches in _unpack_msgpack_bytes
    str32_bytes = b"\xdb\x00\x00\x00\x03abc"
    assert _unpack_msgpack_bytes(str32_bytes)[0] == "abc"
    arr32_bytes = b"\xdd\x00\x00\x00\x01\x05"
    assert _unpack_msgpack_bytes(arr32_bytes)[0] == [5]
    map32_bytes = b"\xdf\x00\x00\x00\x01\xa1k\x07"
    assert _unpack_msgpack_bytes(map32_bytes)[0] == {"k": 7}

    large_str = "x" * 70000
    assert _unpack_msgpack_bytes(_pack_msgpack_bytes(large_str))[0] == large_str
    large_list: list[JsonValue] = [1] * 65536
    unpacked_large_list, _ = _unpack_msgpack_bytes(_pack_msgpack_bytes(large_list))
    assert isinstance(unpacked_large_list, list)
    assert len(unpacked_large_list) == 65536

    for truncated in (
        b"",
        b"\xa5ab",
        b"\xc4",
        b"\xc4\x05ab",
        b"\xc5\x00",
        b"\xc5\x00\x05ab",
        b"\xc6\x00",
        b"\xc6\x00\x00\x00\x05ab",
        b"\xcb\x00",
        b"\xcc",
        b"\xcd\x00",
        b"\xce\x00",
        b"\xcf\x00",
        b"\xd0",
        b"\xd1\x00",
        b"\xd2\x00",
        b"\xd3\x00",
        b"\xd9",
        b"\xd9\x05ab",
        b"\xda\x00",
        b"\xda\x00\x05ab",
        b"\xdb\x00",
        b"\xdb\x00\x00\x00\x05ab",
        b"\xdc\x00",
        b"\xdd\x00",
        b"\xde\x00",
        b"\xdf\x00",
        b"\xc1",
    ):
        with pytest.raises(ValueError):
            _unpack_msgpack_bytes(truncated)


def test_msgpack_large_map_and_fallback() -> None:
    """Test map32 packing in _pack_msgpack_bytes."""
    large_map: dict[str, JsonValue] = {f"k{i}": 1 for i in range(65536)}
    packed = _pack_msgpack_bytes(large_map)
    assert packed[0] == 0xDF
    unpacked, _ = _unpack_msgpack_bytes(packed)
    assert isinstance(unpacked, dict)
    assert len(unpacked) == 65536


def test_barrier_sync() -> None:
    """Test BarrierSync wait, count, reset, and negative timeout validation."""
    bs = BarrierSync({})
    assert bs.count("step_1") == 0
    bs.wait("step_1", timeout=10)
    bs.wait("step_1", timeout=0)
    assert bs.count("step_1") == 2
    bs.reset()
    assert bs.count("step_1") == 0

    with pytest.raises(ValueError, match="non-negative"):
        bs.wait("step_1", timeout=-1)


def test_global_array_sharding() -> None:
    """Test GlobalArraySharding across 1D/2D lists, tuples, SerializedArray, PartitionSpec, and LogicalMesh."""
    gas = GlobalArraySharding()
    assert gas.get_local_shard(None, 0) is None
    assert gas.get_local_shard([1, 2, 3, 4], 0, num_hosts=1) == [1, 2, 3, 4]
    assert gas.get_local_shard([], 0, num_hosts=2) == []
    assert gas.get_local_shard(
        [[]],
        0,
        num_hosts=2,
        partition_spec=PartitionSpec(axes=(None, "model")),
        mesh=LogicalMesh(shape={"model": 2}),
    ) == [[]]

    spec = PartitionSpec(axes=(("data",),))
    mesh = LogicalMesh(shape={"data": 2, "model": 2})
    assert gas.get_local_shard([1, 2, 3, 4], 0, num_hosts=1, partition_spec=spec, mesh=mesh) == [
        1,
        2,
    ]
    spec_str = PartitionSpec(axes=("data",))
    assert gas.get_local_shard(
        (1, 2, 3, 4), 1, num_hosts=1, partition_spec=spec_str, mesh=mesh
    ) == (
        3,
        4,
    )
    # 2D sharding along axis 1
    spec_axis1 = PartitionSpec(axes=(None, "model"))
    mat_2d = [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert gas.get_local_shard(mat_2d, 0, num_hosts=1, partition_spec=spec_axis1, mesh=mesh) == [
        [1, 2],
        [5, 6],
    ]
    assert gas.get_local_shard(mat_2d, 1, num_hosts=1, partition_spec=spec_axis1, mesh=mesh) == [
        [3, 4],
        [7, 8],
    ]
    spec_axis1_tuple = PartitionSpec(axes=(None, ("model",)))
    assert gas.get_local_shard(
        mat_2d, 1, num_hosts=1, partition_spec=spec_axis1_tuple, mesh=mesh
    ) == [
        [3, 4],
        [7, 8],
    ]
    spec_none = PartitionSpec(axes=(None,))
    assert gas.get_local_shard(
        [1, 2, 3, 4], 0, num_hosts=1, partition_spec=spec_none, mesh=mesh
    ) == [1, 2, 3, 4]

    arr = SampleArray([10, 20, 30, 40], shape=(4,), dtype="int32")
    assert gas.get_local_shard(arr, 0, num_hosts=2) == [10, 20]
    assert gas.get_local_shard(arr, 1, num_hosts=2) == [30, 40]

    ser_arr = SerializedArray([[1, 2], [3, 4]], shape=(2, 2), dtype="float32")
    sharded_ser = gas.get_local_shard(ser_arr, 0, num_hosts=1, partition_spec=spec_axis1, mesh=mesh)
    assert isinstance(sharded_ser, SerializedArray)
    assert sharded_ser.shape == (2, 1)
    assert sharded_ser.tolist() == [[1], [3]]

    short_arr = SampleArray([10], shape=(1,), dtype="int32")
    assert gas.get_local_shard(short_arr, 0, num_hosts=2) is short_arr
    assert gas.get_local_shard(42, 0, num_hosts=2) == 42

    with pytest.raises(ValueError, match="Invalid host_id"):
        gas.get_local_shard([1, 2], -1, num_hosts=2)
    with pytest.raises(ValueError, match="Invalid host_id"):
        gas.get_local_shard([1, 2], 2, num_hosts=2)
    with pytest.raises(ValueError, match="Invalid host_id"):
        gas.get_local_shard([1, 2], 0, num_hosts=0)
