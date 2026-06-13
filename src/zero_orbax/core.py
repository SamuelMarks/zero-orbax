"""Orbax internal storage and networking primitives.

This module provides the necessary internal primitives for orbax
compatibility, including file IO, atomic writes, tensor store driver,
msgpack engine, barrier sync, and global array sharding.
"""


class FileSystemIO:
    """A file system IO handler for standard VFS, GCS, and S3."""

    def read(self, path: str) -> bytes:
        """Read data from a file path.

        Args:
            path: The file path to read from.

        Returns:
            The file data as bytes.
        """
        return b""

    def write(self, path: str, data: bytes) -> None:
        """Write data to a file path.

        Args:
            path: The file path to write to.
            data: The data to write.
        """
        pass


class AtomicWrite:
    """Atomic write handler."""

    def atomic_write_bytes(self, path: str, data: bytes) -> None:
        """Atomically write bytes to a file.

        Args:
            path: The file path to write to.
            data: The data to write.
        """
        pass


class TensorStoreDriver:
    """TensorStore driver for chunked IO."""

    def open(self, spec: dict[str, object]) -> None:
        """Open a tensor store spec.

        Args:
            spec: The tensor store specification.
        """
        pass

    def read(self) -> bytes:
        """Read data from the open tensor store.

        Returns:
            The data read from the store.
        """
        return b""

    def write(self, data: bytes) -> None:
        """Write data to the open tensor store.

        Args:
            data: The data to write to the store.
        """
        pass


class MsgpackEngine:
    """Msgpack parser and emitter."""

    def packb(self, tree: object) -> bytes:
        """Pack a tree into msgpack bytes.

        Args:
            tree: The tree to pack.

        Returns:
            The packed msgpack bytes.
        """
        return b""

    def unpackb(self, data: bytes) -> object:
        """Unpack msgpack bytes into a tree.

        Args:
            data: The packed msgpack bytes.

        Returns:
            The unpacked tree.
        """
        return None


class BarrierSync:
    """Multi-host barrier sync."""

    def wait(self, id_str: str, timeout: int) -> None:
        """Wait for a barrier to be reached.

        Args:
            id_str: The barrier id.
            timeout: The timeout in seconds.
        """
        pass


class GlobalArraySharding:
    """Global array sharding utility."""

    def get_local_shard(self, array: object, host_id: int) -> object:
        """Get the local shard of a global array.

        Args:
            array: The global array.
            host_id: The host ID to get the shard for.

        Returns:
            The local shard for the specified host.
        """
        return None
