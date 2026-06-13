"""Tests for the orbax_primitives module."""

from zero_orbax.core import (
    AtomicWrite,
    BarrierSync,
    FileSystemIO,
    GlobalArraySharding,
    MsgpackEngine,
    TensorStoreDriver,
)


def test_file_system_io() -> None:
    """Test FileSystemIO."""
    fs = FileSystemIO()
    assert fs.read("path") == b""
    assert fs.write("path", b"") is None


def test_atomic_write() -> None:
    """Test AtomicWrite."""
    aw = AtomicWrite()
    assert aw.atomic_write_bytes("path", b"") is None


def test_tensor_store_driver() -> None:
    """Test TensorStoreDriver."""
    tsd = TensorStoreDriver()
    assert tsd.open({}) is None
    assert tsd.read() == b""
    assert tsd.write(b"") is None


def test_msgpack_engine() -> None:
    """Test MsgpackEngine."""
    me = MsgpackEngine()
    assert me.packb({}) == b""
    assert me.unpackb(b"") is None


def test_barrier_sync() -> None:
    """Test BarrierSync."""
    bs = BarrierSync()
    assert bs.wait("id", 10) is None


def test_global_array_sharding() -> None:
    """Test GlobalArraySharding."""
    gas = GlobalArraySharding()
    assert gas.get_local_shard(None, 0) is None
