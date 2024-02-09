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
import sys

import pytest
from nanoarrow._lib import NanoarrowException
from nanoarrow.c_lib import (
    CArrayBuilder,
    CBuffer,
    CBufferBuilder,
    _c_array_from_pybuffer,
    c_array_empty,
    c_array_from_buffers,
    c_buffer,
    _c_buffer_from_iterable,
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


def test_c_buffer_constructor():
    invalid = CBuffer()
    assert c_buffer(invalid) is invalid

    buffer = c_buffer(b"1234")
    assert isinstance(buffer, CBuffer)
    assert bytes(buffer.data) == b"1234"


def test_buffer_unsupported_format():
    empty = CBuffer().set_empty()

    with pytest.raises(ValueError, match="Can't convert format '>i' to Arrow type"):
        if sys.byteorder == "little":
            empty.set_format(">i")
        else:
            empty.set_format("<i")

    with pytest.raises(ValueError, match=r"Unsupported Arrow type_id"):
        empty.set_data_type(na.Type.SPARSE_UNION.value)


def test_buffer_empty():
    empty = CBuffer().set_empty()

    assert empty._addr() == 0
    assert empty.size_bytes == 0
    assert empty.capacity_bytes == 0
    assert bytes(empty.data) == b""

    assert repr(empty) == "CBuffer(binary[0 b] b'')"

    # Export it via the Python buffer protocol wrapped in a new CBuffer
    empty_roundtrip = c_buffer(empty.data)
    assert empty_roundtrip.size_bytes == 0
    assert empty_roundtrip.capacity_bytes == 0

    view_roundtrip = empty_roundtrip.data
    assert view_roundtrip._addr() == 0
    assert view_roundtrip.size_bytes == 0


def test_buffer_pybuffer():
    data = bytes(b"abcdefghijklmnopqrstuvwxyz")
    buffer = c_buffer(data)

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
        buffer = c_buffer(packed).set_format(format)
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
        buffer = c_buffer(array)
        view = buffer.data
        assert list(view) == list(array)

        array_roundtrip = np.array(view, copy=False)
        np.testing.assert_array_equal(array_roundtrip, array)

        buffer_roundtrip = c_buffer(array_roundtrip)
        assert buffer_roundtrip._addr() == buffer._addr()


def test_buffer_float():
    formats = ["e", "f", "d"]
    values = [0.0, 1.0, 2.0]

    for format in formats:
        packed = b""
        for value in values:
            packed += struct.pack(format, value)
        buffer = c_buffer(packed).set_format(format)
        assert buffer.size_bytes == len(packed)

        view = buffer.data
        assert len(view) == 3
        assert view[0] == 0.0
        assert view[1] == 1.0
        assert view[2] == 2.0
        assert list(view) == [0.0, 1.0, 2.0]


def test_buffer_string():
    packed = b"abcdefg"
    buffer = c_buffer(packed).set_format("c")
    assert buffer.size_bytes == len(packed)

    view = buffer.data
    assert len(view) == len(packed)
    assert list(view) == [c.encode("UTF-8") for c in "abcdefg"]


def test_buffer_fixed_size_binary():
    items = [b"abcd", b"efgh", b"ijkl"]
    packed = b"".join(items)
    buffer = c_buffer(packed).set_format("4s")
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


def test_c_buffer_from_iterable():
    buffer = _c_buffer_from_iterable([1, 2, 3], na.int32())
    assert buffer.size_bytes == 12
    assert buffer.data.data_type == "int32"
    assert buffer.data.element_size_bits == 32
    assert buffer.data.item_size == 4
    assert list(buffer.data) == [1, 2, 3]

    # An Arrow type that does not make sense as a buffer type will error
    with pytest.raises(ValueError, match="Unsupported Arrow type_id"):
        _c_buffer_from_iterable([], na.struct([]))

    # An Arrow type whose storage type is not the same as its top-level
    # type will error.
    with pytest.raises(ValueError, match="Can't create buffer from type"):
        _c_buffer_from_iterable([1, 2, 3], na.date32())


def test_c_buffer_from_fixed_size_binary_iterable():
    items = [b"abcd", b"efgh", b"ijkl"]
    buffer = _c_buffer_from_iterable(items, na.fixed_size_binary(4))
    assert buffer.data.data_type == "binary"
    assert buffer.data.element_size_bits == 32
    assert buffer.data.item_size == 4
    assert bytes(buffer.data) == b"".join(items)
    assert list(buffer.data) == items


def test_c_buffer_from_day_time_iterable():
    buffer = _c_buffer_from_iterable([(1, 2), (3, 4), (5, 6)], na.interval_day_time())
    assert buffer.data.data_type == "interval_day_time"
    assert buffer.data.element_size_bits == 64
    assert buffer.data.item_size == 8
    assert list(buffer.data) == [(1, 2), (3, 4), (5, 6)]


def test_c_buffer_from_month_day_nano_iterable():
    buffer = _c_buffer_from_iterable(
        [(1, 2, 3), (4, 5, 6)], na.interval_month_day_nano()
    )
    assert buffer.data.data_type == "interval_month_day_nano"
    assert buffer.data.element_size_bits == 128
    assert buffer.data.item_size == 16
    assert list(buffer.data) == [(1, 2, 3), (4, 5, 6)]


def test_c_buffer_from_decimal128_iterable():
    bytes64 = bytes(range(64))
    buffer = _c_buffer_from_iterable(
        [bytes64[0:16], bytes64[16:32], bytes64[32:48], bytes64[48:64]],
        na.decimal128(10, 3),
    )
    assert buffer.data.data_type == "decimal128"
    assert buffer.data.element_size_bits == 128
    assert buffer.data.item_size == 16
    assert list(buffer.data) == [
        bytes64[0:16],
        bytes64[16:32],
        bytes64[32:48],
        bytes64[48:64],
    ]


def test_c_buffer_from_decimal256_iterable():
    bytes64 = bytes(range(64))
    buffer = _c_buffer_from_iterable(
        [bytes64[0:32], bytes64[32:64]], na.decimal256(10, 3)
    )
    assert buffer.data.data_type == "decimal256"
    assert buffer.data.element_size_bits == 256
    assert buffer.data.item_size == 32
    assert list(buffer.data) == [bytes64[0:32], bytes64[32:64]]


def test_c_buffer_bitmap_from_iterable():
    # Check something less than one byte
    buffer = _c_buffer_from_iterable([True, False, False, True], na.bool())
    assert "10010000" in repr(buffer)
    assert buffer.size_bytes == 1
    assert buffer.data.data_type == "bool"
    assert buffer.data.item_size == 1
    assert buffer.data.element_size_bits == 1

    # Check something exactly one byte
    buffer = _c_buffer_from_iterable([True, False, False, True] * 2, na.bool())
    assert "10011001" in repr(buffer)
    assert buffer.size_bytes == 1

    # Check something more than one byte
    buffer = _c_buffer_from_iterable([True, False, False, True] * 3, na.bool())
    assert "1001100110010000" in repr(buffer)
    assert buffer.size_bytes == 2

    # Check that appending in more than one batch is an error
    with pytest.raises(NotImplementedError, match="Append to bitmap"):
        buffer.write_values([True])


def test_c_array_builder_init():
    builder = CArrayBuilder.allocate()
    builder.init_from_type(na.Type.INT32.value)

    with pytest.raises(RuntimeError, match="CArrayBuilder is already initialized"):
        builder.init_from_type(na.Type.INT32.value)

    with pytest.raises(RuntimeError, match="CArrayBuilder is already initialized"):
        builder.init_from_schema(na.c_schema(na.int32()))


def test_c_array_from_pybuffer_uint8():
    data = b"abcdefg"
    c_array = _c_array_from_pybuffer(data)
    assert c_array.length == len(data)
    assert c_array.null_count == 0
    assert c_array.offset == 0
    assert na.c_schema_view(c_array.schema).type == "uint8"

    c_array_view = na.c_array_view(c_array)
    assert list(c_array_view.buffer(1)) == list(data)


def test_c_array_from_pybuffer_string():
    data = b"abcdefg"
    buffer = c_buffer(data).set_format("c")
    c_array = _c_array_from_pybuffer(buffer.data)
    assert c_array.length == len(data)
    assert c_array.null_count == 0
    assert c_array.offset == 0
    assert na.c_schema_view(c_array.schema).type == "int8"

    c_array_view = na.c_array_view(c_array)
    assert list(c_array_view.buffer(1)) == list(data)


def test_c_array_from_pybuffer_fixed_size_binary():
    items = [b"abcd", b"efgh", b"ijkl"]
    packed = b"".join(items)
    buffer = c_buffer(packed).set_format("4s")

    c_array = _c_array_from_pybuffer(buffer.data)
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


def test_c_array_from_buffers():
    c_array = c_array_from_buffers(na.uint8(), 5, [None, b"12345"])
    assert c_array.length == 5
    assert c_array.null_count == 0
    assert c_array.offset == 0

    array_view = na.c_array_view(c_array)
    assert array_view.storage_type == "uint8"
    assert bytes(array_view.buffer(0)) == b""
    assert bytes(array_view.buffer(1)) == b"12345"


def test_c_array_from_buffers_null_count():
    # Ensure null_count is not calculated if explicitly set
    c_array = c_array_from_buffers(na.uint8(), 7, [b"\xff", b"12345678"], null_count=1)
    assert c_array.null_count == 1

    # Zero nulls, explicit bitmap
    c_array = c_array_from_buffers(na.uint8(), 8, [b"\xff", b"12345678"])
    assert c_array.null_count == 0

    # All nulls, explicit bitmap
    c_array = c_array_from_buffers(na.uint8(), 8, [b"\x00", b"12345678"])
    assert c_array.null_count == 8

    # Ensure offset is considered in null count calculation
    c_array = c_array_from_buffers(na.uint8(), 7, [b"\xff", b"12345678"], offset=1)
    assert c_array.null_count == 0

    c_array = c_array_from_buffers(na.uint8(), 7, [b"\x00", b"12345678"], offset=1)
    assert c_array.null_count == 7

    # Ensure we don't access out-of-bounds memory to calculate the bitmap
    with pytest.raises(ValueError, match="Expected validity bitmap >= 2 bytes"):
        c_array_from_buffers(na.uint8(), 9, [b"\x00", b"123456789"])


def test_c_array_from_buffers_recursive():
    c_array = c_array_from_buffers(
        na.struct([na.uint8()]), 5, [None], children=[b"12345"]
    )
    assert c_array.length == 5
    assert c_array.n_children == 1

    array_view = na.c_array_view(c_array)
    assert bytes(array_view.child(0).buffer(1)) == b"12345"

    with pytest.raises(ValueError, match="Expected 1 children but got 0"):
        c_array_from_buffers(na.struct([na.uint8()]), 5, [], children=[])


def test_c_array_from_buffers_validation():
    # Should fail with all validation levels except none
    for validation_level in ("full", "default", "minimal"):
        with pytest.raises(
            NanoarrowException,
            match="Expected int32 array buffer 1 to have size >= 4 bytes",
        ):
            c_array_from_buffers(
                na.int32(), 1, [None, b"123"], validation_level=validation_level
            )

    c_array = c_array_from_buffers(
        na.int32(), 1, [None, b"123"], validation_level="none"
    )
    assert c_array.length == 1

    # Should only fail with validation levels of "full", and "default"
    for validation_level in ("full", "default"):
        with pytest.raises(
            NanoarrowException,
            match="Expected string array buffer 2 to have size >= 2 bytes",
        ):
            c_array_from_buffers(
                na.string(),
                2,
                [None, _c_buffer_from_iterable([0, 1, 2], na.int32()), c_buffer(b"a")],
                validation_level=validation_level,
            )

    for validation_level in ("minimal", "none"):
        c_array = c_array_from_buffers(
            na.string(),
            2,
            [None, _c_buffer_from_iterable([0, 1, 2], na.int32()), c_buffer(b"a")],
            validation_level=validation_level,
        )
        assert c_array.length == 2

    # Should only fail with validation level "full"
    with pytest.raises(
        NanoarrowException,
        match="Expected element size >= 0",
    ):
        c_array_from_buffers(
            na.string(),
            2,
            [None, _c_buffer_from_iterable([0, 100, 2], na.int32()), c_buffer(b"ab")],
            validation_level="full",
        )

    for validation_level in ("minimal", "none"):
        c_array = c_array_from_buffers(
            na.string(),
            2,
            [None, _c_buffer_from_iterable([0, 100, 2], na.int32()), c_buffer(b"ab")],
            validation_level=validation_level,
        )
        assert c_array.length == 2
