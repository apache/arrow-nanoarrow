# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Realistic multi-threaded scenarios.

These tests start their own threads (rather than relying on pytest-run-parallel)
so that many threads operate on the *same* object at the same time. They are most
meaningful on a free-threaded build of Python but are valid on any build.
"""

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

import pytest
from nanoarrow._array import CArray, CArrayBuilder
from nanoarrow._array_stream import CArrayStream

import nanoarrow as na

N_THREADS = 8

_T = TypeVar("_T")


def run_concurrently(
    fn: Callable[[threading.Barrier], _T],
    n_threads: int = N_THREADS,
) -> list[_T]:
    """Call ``fn(barrier)`` on each of ``n_threads`` threads

    ``fn`` should call ``barrier.wait()`` immediately before the operation
    whose concurrency is being tested so that all threads perform it at the
    same time. Exceptions raised in any thread propagate to the caller.
    Returns the list of results.
    """
    barrier = threading.Barrier(n_threads)
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(fn, barrier) for _ in range(n_threads)]
        return [future.result() for future in futures]


def c_array_with_unknown_null_count(n: int) -> CArray:
    """Build an int64 CArray whose null_count is -1 (i.e., not yet computed)

    Arrays from other producers commonly arrive like this; in that case
    the null count is computed lazily by CArrayView.null_count. Every third
    element is null.
    """
    builder = CArrayBuilder.allocate()
    builder.init_from_schema(na.c_schema(na.int64()))
    builder.set_length(n)
    builder.set_buffer(0, na.c_buffer([i % 3 != 0 for i in range(n)], na.bool_()))
    builder.set_buffer(1, na.c_buffer(list(range(n)), na.int64()))
    builder.set_null_count(-1)
    return builder.finish()


@pytest.mark.thread_unsafe(reason="starts its own threads")
def test_shared_array_read_concurrently() -> None:
    """Test concurrent reads to a shared array.

    One Array shared by many threads, each of which only eads it. Every read
    path below creates its own ArrowArrayView, so the lazily computed null
    count is cached on memory private to each thread and never written to the
    shared ArrowArrays.
    """
    n_chunks = 4
    chunk_size = 10_000
    n_rounds = 20

    c_arrays = [c_array_with_unknown_null_count(chunk_size) for _ in range(n_chunks)]
    array = na.Array(CArrayStream.from_c_arrays(c_arrays, na.c_schema(na.int64())))

    chunk_values = [None if i % 3 == 0 else i for i in range(chunk_size)]
    expected_values = chunk_values * n_chunks
    expected_null_count = len(range(0, chunk_size, 3))
    slice_start, slice_stop = 100, 200
    expected_slice_null_count = len(
        [i for i in range(slice_start, slice_stop) if i % 3 == 0]
    )

    def read(barrier: threading.Barrier) -> None:
        barrier.wait()
        for _ in range(n_rounds):
            assert array.to_pylist() == expected_values
            assert list(array.iter_py()) == expected_values
            assert [view.null_count for view in array.iter_chunk_views()] == [
                expected_null_count,
            ] * n_chunks

            for c_array in c_arrays:
                assert c_array.view().null_count == expected_null_count
                c_slice = c_array[slice_start:slice_stop]
                assert c_slice.view().null_count == expected_slice_null_count

    run_concurrently(read)

    # Nothing was written back to the shared arrays
    assert [c_array.null_count for c_array in c_arrays] == [-1] * n_chunks
