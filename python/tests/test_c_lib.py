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

import struct

import pytest
from nanoarrow.c_lib import (
    CBuffer,
    CBufferBuilder,
    c_array_empty,
    c_array_from_pybuffer,
)

import nanoarrow as na


def test_buffer_invalid():
    invalid = CBuffer()

    with pytest.raises(RuntimeError, match="CBuffer is not valid"):
        invalid._addr()
    with pytest.raises(RuntimeError, match="CBuffer is not valid"):
        invalid.size_bytes
    with pytest.raises(RuntimeError, match="CBuffer is not valid"):
        invalid.capacity_bytes
    with pytest.raises(RuntimeError, match="CBuffer is not valid"):
        invalid.data

    assert repr(invalid) == "CBuffer(<invalid>)"


def test_buffer_empty():
    empty = CBuffer().set_empty()

    assert empty._addr() == 0
    assert empty.size_bytes == 0
    assert empty.capacity_bytes == 0
    assert bytes(empty.data) == b""

    assert repr(empty) == "CBuffer(binary[0 b] b'')"


def test_buffer_pybuffer():
    data = bytes(b"abcdefghijklmnopqrstuvwxyz")
    buffer = CBuffer().set_pybuffer(data)

    assert buffer.size_bytes == len(data)
    assert buffer.capacity_bytes == 0
    assert bytes(buffer.data) == b"abcdefghijklmnopqrstuvwxyz"

    assert repr(buffer).startswith("CBuffer(uint8[26 b] 97 98")


def test_buffer_integer():
    formats = ["b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "n", "N"]
    values = [0, 1, 2]

    for format in formats:
        packed = b""
        for value in values:
            packed += struct.pack(format, value)
        buffer = CBuffer().set_pybuffer(packed).set_format(format)
        assert buffer.size_bytes == len(packed)

        view = buffer.data
        assert len(view) == 3
        assert view[0] == 0
        assert view[1] == 1
        assert view[2] == 2
        assert list(view) == [0, 1, 2]


def test_numpy_buffer_numeric():
    np = pytest.importorskip("numpy")

    dtypes = [
        np.int8(),
        np.uint8(),
        np.int16(),
        np.uint16(),
        np.int32(),
        np.uint32(),
        np.int64(),
        np.uint64(),
        np.float16(),
        np.float32(),
        np.float64(),
        "|S1",
    ]

    for dtype in dtypes:
        array = np.array([0, 1, 2], dtype)
        buffer = CBuffer().set_pybuffer(array)
        view = buffer.data
        assert list(view) == list(array)

        array_roundtrip = np.array(view, copy=False)
        np.testing.assert_array_equal(array_roundtrip, array)

        buffer_roundtrip = CBuffer().set_pybuffer(array_roundtrip)
        assert buffer_roundtrip._addr() == buffer._addr()


def test_buffer_float():
    formats = ["e", "f", "d"]
    values = [0.0, 1.0, 2.0]

    for format in formats:
        packed = b""
        for value in values:
            packed += struct.pack(format, value)
        buffer = CBuffer().set_pybuffer(packed).set_format(format)
        assert buffer.size_bytes == len(packed)

        view = buffer.data
        assert len(view) == 3
        assert view[0] == 0.0
        assert view[1] == 1.0
        assert view[2] == 2.0
        assert list(view) == [0.0, 1.0, 2.0]


def test_buffer_string():
    packed = b"abcdefg"
    buffer = CBuffer().set_pybuffer(packed).set_format("c")
    assert buffer.size_bytes == len(packed)

    view = buffer.data
    assert len(view) == len(packed)
    assert list(view) == [c.encode("UTF-8") for c in "abcdefg"]


def test_buffer_fixed_size_binary():
    items = [b"abcd", b"efgh", b"ijkl"]
    packed = b"".join(items)
    buffer = CBuffer().set_pybuffer(packed).set_format("4s")
    assert buffer.size_bytes == len(packed)

    view = buffer.data
    assert len(view) == 3
    assert view[0] == b"abcd"
    assert view[1] == b"efgh"
    assert view[2] == b"ijkl"
    assert list(view) == items


def test_buffer_builder():
    builder = CBufferBuilder().set_empty()
    assert builder.size_bytes == 0
    assert builder.capacity_bytes == 0

    builder.reserve_bytes(123)
    assert builder.size_bytes == 0
    assert builder.capacity_bytes == 123

    builder.write(b"abcde")
    assert builder.size_bytes == 5
    assert builder.capacity_bytes == 123

    builder.write(b"fghij")
    assert builder.size_bytes == 10
    assert builder.capacity_bytes == 123

    assert bytes(builder.data) == b"abcdefghij"


def test_c_array_from_pybuffer_uint8():
    data = b"abcdefg"
    c_array = c_array_from_pybuffer(data)
    assert c_array.length == len(data)
    assert c_array.null_count == 0
    assert c_array.offset == 0
    assert na.c_schema_view(c_array.schema).type == "uint8"

    c_array_view = na.c_array_view(c_array)
    assert list(c_array_view.buffer(1)) == list(data)


def test_c_array_from_pybuffer_string():
    data = b"abcdefg"
    buffer = CBuffer().set_pybuffer(data).set_format("c")
    c_array = c_array_from_pybuffer(buffer.data)
    assert c_array.length == len(data)
    assert c_array.null_count == 0
    assert c_array.offset == 0
    assert na.c_schema_view(c_array.schema).type == "int8"

    c_array_view = na.c_array_view(c_array)
    assert list(c_array_view.buffer(1)) == list(data)


def test_c_array_from_pybuffer_fixed_size_binary():
    items = [b"abcd", b"efgh", b"ijkl"]
    packed = b"".join(items)
    buffer = CBuffer().set_pybuffer(packed).set_format("4s")

    c_array = c_array_from_pybuffer(buffer.data)
    assert c_array.length == len(items)
    assert c_array.null_count == 0
    assert c_array.offset == 0
    assert na.c_schema_view(c_array.schema).type == "fixed_size_binary"
    assert na.c_schema_view(c_array.schema).fixed_size == 4

    c_array_view = na.c_array_view(c_array)
    assert list(c_array_view.buffer(1)) == items


def test_c_array_empty():
    empty_string = c_array_empty(na.c_schema(na.string()))
    assert empty_string.length == 0
    assert empty_string.null_count == 0
    assert empty_string.offset == 0
    assert empty_string.n_buffers == 3

    array_view = na.c_array_view(empty_string)
    assert len(array_view.buffer(0)) == 0
    assert len(array_view.buffer(1)) == 0
    assert len(array_view.buffer(2)) == 0
