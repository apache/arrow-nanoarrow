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
from datetime import date, datetime, timezone

import pytest
from nanoarrow._buffer import CBuffer, CBufferBuilder

import nanoarrow as na


def test_buffer_invalid():
    invalid = CBuffer()

    with pytest.raises(RuntimeError, match="CBuffer is not valid"):
        invalid._addr()
    with pytest.raises(RuntimeError, match="CBuffer is not valid"):
        invalid.size_bytes
    with pytest.raises(RuntimeError, match="CBuffer is not valid"):
        memoryview(invalid)

    assert repr(invalid) == "nanoarrow.c_buffer.CBuffer(<invalid>)"


def test_c_buffer_constructor():
    invalid = CBuffer()
    assert na.c_buffer(invalid) is invalid

    buffer = na.c_buffer(b"1234")
    assert isinstance(buffer, CBuffer)
    assert bytes(buffer) == b"1234"


def test_c_buffer_unsupported_format():
    empty = CBuffer.empty()

    if sys.byteorder == "little":
        with pytest.raises(ValueError, match="Can't convert format '>i' to Arrow type"):
            empty._set_format(">i")
    else:
        with pytest.raises(ValueError, match="Can't convert format '<i' to Arrow type"):
            empty._set_format("<i")

    with pytest.raises(ValueError, match=r"Unsupported Arrow type_id"):
        empty._set_data_type(na.Type.SPARSE_UNION.value)


def test_c_buffer_empty():
    empty = CBuffer.empty()

    assert empty._addr() == 0
    assert empty.size_bytes == 0
    assert bytes(empty) == b""

    assert repr(empty) == "nanoarrow.c_buffer.CBuffer(binary[0 b] b'')"

    # Export it via the Python buffer protocol wrapped in a new CBuffer
    empty_roundtrip = na.c_buffer(empty)
    assert empty_roundtrip.size_bytes == 0

    assert empty_roundtrip._addr() == 0
    assert empty_roundtrip.size_bytes == 0


def test_c_buffer_pybuffer():
    data = bytes(b"abcdefghijklmnopqrstuvwxyz")
    buffer = na.c_buffer(data)

    assert buffer.size_bytes == len(data)
    assert bytes(buffer) == b"abcdefghijklmnopqrstuvwxyz"

    assert repr(buffer).startswith("nanoarrow.c_buffer.CBuffer(uint8[26 b] 97 98")


def test_c_buffer_unsupported_type():
    with pytest.raises(TypeError, match="Can't convert object of type NoneType"):
        na.c_buffer(None, na.int32())


def test_c_buffer_missing_requested_schema():
    with pytest.raises(ValueError, match="CBuffer from iterable requires schema"):
        na.c_buffer([1, 2, 3])


def test_c_buffer_pybuffer_with_schema():
    with pytest.raises(
        NotImplementedError, match="schema for pybuffer is not implemented"
    ):
        na.c_buffer(b"1234", na.int32())


def test_c_buffer_integer():
    formats = ["b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "n", "N"]
    values = [0, 1, 2]

    for format in formats:
        packed = b""
        for value in values:
            packed += struct.pack(format, value)
        buffer = na.c_buffer(packed)._set_format(format)
        assert buffer.size_bytes == len(packed)

        assert len(buffer) == 3
        assert buffer[0] == 0
        assert buffer[1] == 1
        assert buffer[2] == 2
        assert list(buffer) == [0, 1, 2]
        assert list(buffer.elements()) == [0, 1, 2]
        assert buffer.n_elements == len(buffer)
        assert [buffer.element(i) for i in range(buffer.n_elements)] == list(buffer)


def test_numpy_c_buffer_numeric():
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
        buffer = na.c_buffer(array)
        assert list(buffer) == list(array)
        assert list(buffer.elements()) == list(array)

        array_roundtrip = np.array(buffer, copy=False)
        np.testing.assert_array_equal(array_roundtrip, array)

        buffer_roundtrip = na.c_buffer(array_roundtrip)
        assert buffer_roundtrip._addr() == buffer._addr()


def test_c_buffer_float():
    formats = ["e", "f", "d"]
    values = [0.0, 1.0, 2.0]

    for format in formats:
        packed = b""
        for value in values:
            packed += struct.pack(format, value)
        buffer = na.c_buffer(packed)._set_format(format)
        assert buffer.size_bytes == len(packed)

        assert len(buffer) == 3
        assert buffer[0] == 0.0
        assert buffer[1] == 1.0
        assert buffer[2] == 2.0
        assert list(buffer) == [0.0, 1.0, 2.0]


def test_c_buffer_string():
    packed = b"abcdefg"
    buffer = na.c_buffer(packed)._set_format("c")
    assert buffer.size_bytes == len(packed)

    assert len(buffer) == len(packed)
    assert list(buffer) == [c.encode("UTF-8") for c in "abcdefg"]


def test_c_buffer_fixed_size_binary():
    items = [b"abcd", b"efgh", b"ijkl"]
    packed = b"".join(items)
    buffer = na.c_buffer(packed)._set_format("4s")
    assert buffer.size_bytes == len(packed)

    assert len(buffer) == 3
    assert buffer[0] == b"abcd"
    assert buffer[1] == b"efgh"
    assert buffer[2] == b"ijkl"
    assert list(buffer) == items


def test_c_buffer_builder():
    builder = CBufferBuilder()
    assert builder.size_bytes == 0
    assert builder.capacity_bytes == 0
    assert repr(builder) == "nanoarrow.c_buffer.CBufferBuilder(0/0)"

    builder.reserve_bytes(123)
    assert builder.size_bytes == 0
    assert builder.capacity_bytes == 123

    builder.write(b"abcde")
    assert builder.size_bytes == 5
    assert builder.capacity_bytes == 123

    builder.write(b"fghij")
    assert builder.size_bytes == 10
    assert builder.capacity_bytes == 123

    with pytest.raises(IndexError):
        builder.advance(-11)

    with pytest.raises(IndexError):
        builder.advance(114)


def test_c_buffer_builder_buffer_protocol():
    import platform

    builder = CBufferBuilder()
    builder.reserve_bytes(1)

    with memoryview(builder) as mv:
        assert len(mv) == 1

        with pytest.raises(BufferError, match="CBufferBuilder is locked"):
            memoryview(builder)

        with pytest.raises(BufferError, match="CBufferBuilder is locked"):
            assert bytes(builder.finish()) == b"abcdefghij"

        mv[builder.size_bytes] = ord("k")

    if platform.python_implementation() == "PyPy":
        pytest.skip("memoryview() release is not guaranteed on PyPy")

    builder.advance(1)
    assert bytes(builder.finish()) == b"k"


def test_c_buffer_from_iterable():
    buffer = na.c_buffer([1, 2, 3], na.int32())
    assert buffer.size_bytes == 12
    assert buffer.data_type == "int32"
    assert buffer.element_size_bits == 32
    assert buffer.itemsize == 4
    assert list(buffer) == [1, 2, 3]

    # An Arrow type that does not make sense as a buffer type will error
    with pytest.raises(ValueError, match="Unsupported Arrow type_id"):
        na.c_buffer([], na.struct([]))

    # An Arrow type whose storage type is not the same as its top-level
    # type will error.
    with pytest.raises(ValueError, match="Can't create buffer"):
        na.c_buffer([1, 2, 3], na.dictionary(na.int32(), na.string()))

    with pytest.raises(ValueError, match="Can't create buffer"):
        na.c_buffer([1, 2, 3], na.extension_type(na.int32(), "arrow.test"))


def test_c_buffer_from_fixed_size_binary_iterable():
    items = [b"abcd", b"efgh", b"ijkl"]
    buffer = na.c_buffer(items, na.fixed_size_binary(4))
    assert buffer.data_type == "binary"
    assert buffer.element_size_bits == 32
    assert buffer.itemsize == 4
    assert bytes(buffer) == b"".join(items)
    assert list(buffer) == items


def test_c_buffer_from_day_time_iterable():
    buffer = na.c_buffer([(1, 2), (3, 4), (5, 6)], na.interval_day_time())
    assert buffer.data_type == "interval_day_time"
    assert buffer.element_size_bits == 64
    assert buffer.itemsize == 8
    assert list(buffer) == [(1, 2), (3, 4), (5, 6)]


def test_c_buffer_from_month_day_nano_iterable():
    buffer = na.c_buffer([(1, 2, 3), (4, 5, 6)], na.interval_month_day_nano())
    assert buffer.data_type == "interval_month_day_nano"
    assert buffer.element_size_bits == 128
    assert buffer.itemsize == 16
    assert list(buffer) == [(1, 2, 3), (4, 5, 6)]


def test_c_buffer_from_decimal128_iterable():
    bytes64 = bytes(range(64))
    buffer = na.c_buffer(
        [bytes64[0:16], bytes64[16:32], bytes64[32:48], bytes64[48:64]],
        na.decimal128(10, 3),
    )
    assert buffer.data_type == "decimal128"
    assert buffer.element_size_bits == 128
    assert buffer.itemsize == 16
    assert list(buffer) == [
        bytes64[0:16],
        bytes64[16:32],
        bytes64[32:48],
        bytes64[48:64],
    ]


def test_c_buffer_from_decimal256_iterable():
    bytes64 = bytes(range(64))
    buffer = na.c_buffer([bytes64[0:32], bytes64[32:64]], na.decimal256(10, 3))
    assert buffer.data_type == "decimal256"
    assert buffer.element_size_bits == 256
    assert buffer.itemsize == 32
    assert list(buffer) == [bytes64[0:32], bytes64[32:64]]


def test_c_buffer_bitmap_from_iterable():
    # Check something less than one byte
    buffer = na.c_buffer([True, False, False, True], na.bool_())
    assert "10010000" in repr(buffer)
    assert buffer.size_bytes == 1
    assert buffer.data_type == "bool"
    assert buffer.itemsize == 1
    assert buffer.element_size_bits == 1
    assert list(buffer.elements()) == [
        True,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert [buffer.element(i) for i in range(buffer.n_elements)] == list(
        buffer.elements()
    )

    # Check something exactly one byte
    buffer = na.c_buffer([True, False, False, True] * 2, na.bool_())
    assert "10011001" in repr(buffer)
    assert buffer.size_bytes == 1
    assert list(buffer.elements()) == [True, False, False, True] * 2

    # Check something more than one byte
    buffer = na.c_buffer([True, False, False, True] * 3, na.bool_())
    assert "1001100110010000" in repr(buffer)
    assert buffer.size_bytes == 2
    assert list(buffer.elements()) == [True, False, False, True] * 3 + [
        False,
        False,
        False,
        False,
    ]

    # Check that appending in more than one batch is an error
    builder = CBufferBuilder().set_data_type(na.Type.BOOL.value)
    builder.write_elements([True, False])
    with pytest.raises(NotImplementedError, match="Append to bitmap"):
        builder.write_elements([True])


def test_c_buffer_from_timestamp_iterable():
    d1 = int(round(datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp() * 1e3))
    d2 = int(round(datetime(1985, 12, 31, tzinfo=timezone.utc).timestamp() * 1e3))
    d3 = int(round(datetime(2005, 3, 4, tzinfo=timezone.utc).timestamp() * 1e3))
    with pytest.raises(ValueError):
        na.c_buffer([d1, d2, d3], na.timestamp("ms"))


def test_c_buffer_from_date64_iterable():
    unix_epoch = date(1970, 1, 1)
    d1 = date(1970, 1, 2)
    diff_in_milliseconds = int(round((d1 - unix_epoch).total_seconds() * 1e3))
    with pytest.raises(ValueError):
        na.c_buffer([diff_in_milliseconds], na.date64())


def test_c_buffer_from_date32_iterable():
    unix_epoch = date(1970, 1, 1)
    d1 = date(1970, 1, 2)
    diff_in_days = (d1 - unix_epoch).days
    with pytest.raises(ValueError):
        na.c_buffer([diff_in_days], na.date32())


def test_c_buffer_from_duration_iterable():
    unix_epoch = date(1970, 1, 1)
    d1 = date(1970, 1, 2)
    diff_in_milliseconds = int(round((d1 - unix_epoch).total_seconds() * 1e3))
    with pytest.raises(ValueError):
        na.c_buffer([diff_in_milliseconds], na.duration("ms"))
