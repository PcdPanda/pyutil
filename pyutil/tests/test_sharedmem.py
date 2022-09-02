from multiprocessing import Pool
import os
from time import time, sleep

import pytest

from pyutil import SharedMem, SharedLockFreeQueue


@pytest.fixture
def valid_name():
    name = "test_sharedmem_name"
    try:
        os.remove(f"/dev/shm/{name}")
    except FileNotFoundError:
        pass
    return name


@pytest.fixture
def shm_size():
    return 256


@pytest.fixture
def queue_size():
    return 2 ** 17


@pytest.fixture
def bytes_data(shm_size):
    return bytearray(str.encode("1" * shm_size))


def process_check(valid_name, shm_size, bytes_data):
    shm = SharedMem(name=valid_name, size=shm_size, create=True)
    shm.buf[:] = bytes_data
    start_time = time()
    while shm.buf[:] != bytearray(str.encode("2" * shm_size)):
        if time() - start_time > 3:
            raise TimeoutError
    return True


def process_change(valid_name, shm_size, bytes_data):
    shm = SharedMem(name=valid_name)
    shm.buf[:] = bytearray(str.encode("2" * shm_size))
    return True


def test_create(valid_name, shm_size):
    shm = SharedMem(name=valid_name, size=shm_size, create=True)
    assert valid_name in os.listdir("/dev/shm")
    del shm
    assert valid_name not in os.listdir("/dev/shm")


def test_durable(valid_name, shm_size, bytes_data):
    shm = SharedMem(name=valid_name, size=shm_size, create=True, durable=True)
    shm.buf[:] = bytes_data
    del shm
    assert valid_name in os.listdir("/dev/shm")
    shm = SharedMem(name=valid_name, durable=True)
    assert shm.buf[:] == bytes_data
    del shm
    assert valid_name in os.listdir("/dev/shm")
    shm = SharedMem(name=valid_name)
    assert shm.buf[:] == bytes_data
    del shm
    assert valid_name not in os.listdir("/dev/shm")


def test_invalid_creation(valid_name, shm_size):
    _ = SharedMem(name=valid_name, size=shm_size, create=True)
    for invalid_kwargs in [dict(create=True, size=shm_size),
                           dict(create=True, name="1" * 24),
                           dict(create=True, name=valid_name),
                           dict(create=True, name=valid_name, size=-1)]:
        with pytest.raises(ValueError):
            _ = SharedMem(**invalid_kwargs)
    with pytest.raises(FileExistsError):
        _ = SharedMem(name=valid_name, size=shm_size, create=True)


def test_close(valid_name, shm_size, bytes_data):
    shm = SharedMem(name=valid_name, size=shm_size, create=True)
    shm.buf[:] = bytes_data
    shm.close()
    assert shm.size is None
    assert shm.buf is None


def test_ipc(valid_name, shm_size, bytes_data):
    result = list()
    with Pool(processes=2) as pool:
        pool.apply_async(process_check, (valid_name, shm_size, bytes_data), callback=result.append)
        sleep(0.01)
        pool.apply_async(process_change, (valid_name, shm_size, bytes_data), callback=result.append)
        pool.close()
        pool.join()
    assert result == [True, True]
    assert valid_name not in os.listdir("/dev/shm")


def test_create_queue(valid_name, queue_size):
    queue = SharedLockFreeQueue(valid_name, queue_size)
    assert valid_name in os.listdir("/dev/shm")
    assert queue.empty()
    queue.delete(valid_name)
    assert valid_name not in os.listdir("/dev/shm")


def test_queue_same_name(valid_name, queue_size):
    queue = SharedLockFreeQueue(valid_name, queue_size)
    with pytest.raises(FileExistsError):
        SharedLockFreeQueue(valid_name, queue_size)
    SharedLockFreeQueue(valid_name)
    queue.delete(valid_name)


def process_push(valid_name, queue_size):
    queue = SharedLockFreeQueue(valid_name, queue_size)
    for i in range(10):
        queue.push(str.encode(str(i)))


def process_pop(valid_name):
    queue = SharedLockFreeQueue(valid_name)
    result = list()
    while not queue.empty():
        result.append(int(queue.pop().decode()))
    return result


def test_queue_ipc(valid_name, queue_size):
    result = list()
    with Pool(processes=2) as pool:
        pool.apply_async(process_push, (valid_name, queue_size))
        sleep(0.01)
        pool.apply_async(process_pop, (valid_name, ), callback=result.append)
        pool.close()
        pool.join()
    SharedLockFreeQueue.delete(valid_name)
    assert valid_name not in os.listdir("/dev/shm")
    assert result[0] == list(range(10))
