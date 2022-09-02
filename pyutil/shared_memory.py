import _posixshmem
import mmap
from multiprocessing.shared_memory import SharedMemory
import os
import secrets
import stat
from typing import Optional

_O_CREX = os.O_CREAT | os.O_EXCL


class SharedMem(SharedMemory):
    """The share memory instance in POSIX to replace python's own library

    Parameters
    ==========
    name: str
        The reference name for the shared memory, should be unique, and can't startswith '/'
    create: bool
        Whether create a new memory or just link to an existing one
    size: int
        The size of shared memory block
    prefix: str
        The prefix for the name reference, only valid when creating a new
        instance without specific name, it can't startswith '/'
    durable: bool
        Whether to release the memory when the deconstructor is called

    Examples
    ========
    >>> shm = SharedMem(create=True, name="myshared_mem", size=256)
    >>> shm.buf[:] = bytearray(str.encode("i" * 256))
    >>> shm_linker = SharedMem(name="myshared_mem")
    >>> str(bytes(shm_linker.buf[:]))
    """

    _prefix_max_length = 16
    _name_max_length = 24

    def __init__(
        self,
        name: str = "",
        create: bool = False,
        size: Optional[int] = None,
        prefix: Optional[str] = None,
        durable: bool = False,
    ):
        self._durable = durable
        flags = os.O_RDWR
        if os.name == "nt":
            raise ValueError("This is for POSIX SharedMem")
        if create:
            if size is None or size <= 0:
                raise ValueError("Can't create shared memory without positive size")
            flags |= _O_CREX
            if name:
                if prefix:
                    raise ValueError("parameter prefix is invalid when name is given")
                fd = self._resolve_name(name, flags)
            else:  # create a new instance with prefix
                if not prefix:
                    raise ValueError("Can't create shared memory with empty prefix")
                elif prefix.startswith("/"):
                    raise ValueError(f"prefix can't startswith /, the given prefix is {prefix}")
                if len(prefix) > self._prefix_max_length - 1:
                    raise ValueError(f"prefix length can't exceed {self._prefix_max_length - 1},"
                                     f" the given prefix {prefix} has length {len(prefix)}")
                prefix = "/{}".format(prefix) if not prefix.startswith("/") else prefix
                while True:
                    name = "{}_{}".format(prefix, secrets.token_hex(4))
                    try:
                        fd = self._resolve_name(name, flags)
                        break
                    except FileExistsError:
                        continue
            os.ftruncate(fd, size)
        elif not name:
            raise ValueError("can't link to a shared memory with empty name")
        elif size is not None:
            raise ValueError("parameter size is invalid when connect to a shared memory block")
        else:
            fd = self._resolve_name(name, flags)
        try:
            stats = os.fstat(fd)
            self._size = stats.st_size
            self._mmap = mmap.mmap(fd, self._size)
            self._buf = memoryview(self._mmap)
            os.chmod("/dev/shm/{}".format(self._name), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        except (OSError, PermissionError):
            self.unlink()
            raise

    @property
    def name(self):
        """return the name reference of the shared memory"""
        return self._name[1:]

    def unlink(self):
        """release the shared memory"""
        try:
            _posixshmem.shm_unlink(self._name)
        except Exception:
            pass

    def close(self):
        """Closes access to the shared memory from this instance but does
        not destroy the shared memory block."""
        self._size = None
        if self._buf is not None:
            self._buf.release()
            self._buf = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

    def _resolve_name(self, name: str, flags: int) -> int:
        """resolve the block reference name and get shared memory file descriptor"""
        if name.startswith("/"):
            raise ValueError(f"name can't startswith /, the given name is {name}")
        if len(name) > self._name_max_length - 1:
            raise ValueError(f"name length can't exceed {self._name_max_length - 1},"
                             f" the given name {name} has length {len(name)}")
        self._name = f"/{name}"
        return _posixshmem.shm_open(self._name, flags=flags, mode=self._mode)

    def __enter__(self):
        self._durable = False
        return self

    def __exit__(self, type, value, tb):
        if not self._durable:
            self.unlink()
        self.close()

    def __del__(self):
        if not self._durable:
            self.unlink()
        self.close()


class SharedLockFreeQueue(object):
    """Create a single producer and single consumer lock free queue for IPC

    Parameters
    ==========
    name: str
        The key name of the shared memory for the lock free queue
    size: int
        The size of the queue, should be power of 2

    Examples
    ========
    >>> queue = SharedLockFreeQueue("my_queue", size=2 ** 28)
    >>> queue.delete("my_queue")
    """
    def __init__(self, name: str = "", size: Optional[int] = None):
        if not name:
            name = secrets.token_hex(8)
        if size is not None:
            if size < 65536:
                raise ValueError(f"size must large than 65536 bytes, the given size is {size}")
            elif size & (size - 1):
                raise ValueError(f"size must be power of 2, the given size is {size}")
            elif size > 2 ** 64:
                raise ValueError(f"size should be smaller than 2^64 + 1, the given size is {size}")
            self._shm = SharedMem(name=name, create=True, size=size + 16, durable=True)
        else:
            self._shm = SharedMem(name=name, durable=True)
        self.size = self._shm.size - 16  # Use first 16 bytes to save the head and tail pointers
        self.name = self._shm.name

    def push(self, data: bytes):
        """push back data into the queue

        Parameters
        ==========
        data: bytes
            data to be pushed

        Examples
        ========
        >>> queue = SharedLockFreeQueue("my_queue", size=2 ** 28)
        >>> queue.push(str.encode("1" * test_size))
        >>> queue.delete("my_queue")
        """
        binary = len(data).to_bytes(8, byteorder="big") + data
        if self.tail + len(binary) > self.head + self.size:
            raise BufferError("The lockfree queue doesn't have enough capacity")
        remain = min(self.size - self.tail, len(binary))
        self.buf[self.tail : self.tail + remain] = binary[:remain]
        self.buf[: len(binary[remain:])] = binary[remain:]
        self.tail += len(binary)

    def pop(self) -> bytes:
        """pop out the data from the front of the queue

        Returns
        =======
        bytes
            the data to be poped

        Examples
        ========
        >>> queue = SharedLockFreeQueue("my_queue", size=2 ** 28)
        >>> queue.push(str.encode("1" * test_size))
        >>> queue.pop().decode()
        >>> queue.delete("my_queue")
        """
        if self.empty():
            raise BufferError("The queue is already empty")

        def load(length: int, head: int) -> bytes:
            remain = min(length, self.size - head)
            buffer = bytearray(length)
            buffer[:remain] = self.buf[head : head + remain]
            buffer[remain:] = self.buf[: length - remain]
            return buffer

        size = int.from_bytes(load(8, self.head), byteorder="big")
        data = load(size, self.head + 8)
        self.head += len(data) + 8
        return data

    @property
    def head(self) -> int:
        """Get the head pointer"""
        return int.from_bytes(self._shm.buf[:8], byteorder="big") & (self.size - 1)

    @head.setter
    def head(self, pos: int):
        self._shm.buf[:8] = (pos % 2 ** 64).to_bytes(8, byteorder="big")

    @property
    def tail(self) -> int:
        """Get the tail pointer"""
        return int.from_bytes(self._shm.buf[8:16], byteorder="big") & (self.size - 1)

    @tail.setter
    def tail(self, pos: int):
        self._shm.buf[8:16] = (pos % 2 ** 64).to_bytes(8, byteorder="big")

    def empty(self):
        return self.head == self.tail

    @property
    def buf(self) -> bytearray:
        """Get the buffer of the queue"""
        return self._shm.buf[16:]

    @staticmethod
    def delete(name: str):
        """delete the queue and release the memory"""
        shm = SharedMem(name=name)
        shm.unlink()
