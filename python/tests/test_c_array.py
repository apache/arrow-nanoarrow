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

import pytest
from nanoarrow._lib import NanoarrowException
from nanoarrow.c_lib import CArrayBuilder

import nanoarrow as na


def test_c_array_builder_init():
    builder = CArrayBuilder.allocate()
    builder.init_from_type(na.Type.INT32.value)

    with pytest.raises(RuntimeError, match="CArrayBuilder is already initialized"):
        builder.init_from_type(na.Type.INT32.value)

    with pytest.raises(RuntimeError, match="CArrayBuilder is already initialized"):
        builder.init_from_schema(na.c_schema(na.int32()))


def test_c_array_from_c_array():
    c_array = na.c_array([1, 2, 3], na.int32())
    c_array_from_c_array = na.c_array(c_array)
    assert c_array_from_c_array.length == c_array.length
    assert c_array_from_c_array.buffers == c_array.buffers

    assert list(c_array.view().buffer(1)) == [1, 2, 3]


def test_c_array_from_capsule_protocol():
    class CArrayWrapper:
        def __init__(self, obj):
            self.obj = obj

        def __arrow_c_array__(self, *args, **kwargs):
            return self.obj.__arrow_c_array__(*args, **kwargs)

    c_array = na.c_array([1, 2, 3], na.int32())
    c_array_wrapper = CArrayWrapper(c_array)
    c_array_from_protocol = na.c_array(c_array_wrapper)
    assert c_array_from_protocol.length == c_array.length
    assert c_array_from_protocol.buffers == c_array.buffers

    assert list(c_array_from_protocol.view().buffer(1)) == [1, 2, 3]


def test_c_array_from_old_pyarrow():
    # Simulate a pyarrow Array with no __arrow_c_array__
    class MockLegacyPyarrowArray:
        def __init__(self, obj):
            self.obj = obj

        def _export_to_c(self, *args):
            return self.obj._export_to_c(*args)

    MockLegacyPyarrowArray.__module__ = "pyarrow.lib"

    pa = pytest.importorskip("pyarrow")
    array = MockLegacyPyarrowArray(pa.array([1, 2, 3], pa.int32()))

    c_array = na.c_array(array)
    assert c_array.length == 3
    assert c_array.schema.format == "i"

    assert list(c_array.view().buffer(1)) == [1, 2, 3]

    # Make sure that this heuristic won't result in trying to import
    # something else that has an _export_to_c method
    with pytest.raises(TypeError, match="Can't convert object of type DataType"):
        not_array = pa.int32()
        assert hasattr(not_array, "_export_to_c")
        na.c_array(not_array)


def test_c_array_from_bare_capsule():
    c_array = na.c_array([1, 2, 3], na.int32())

    # Check from bare capsule without supplying a schema
    schema_capsule, array_capsule = c_array.__arrow_c_array__()
    del schema_capsule
    c_array_from_capsule = na.c_array(array_capsule)
    assert c_array_from_capsule.length == c_array.length
    assert c_array_from_capsule.buffers == c_array.buffers

    # Check from bare capsule supplying a schema
    schema_capsule, array_capsule = c_array.__arrow_c_array__()
    c_array_from_capsule = na.c_array(array_capsule, schema_capsule)
    assert c_array_from_capsule.length == c_array.length
    assert c_array_from_capsule.buffers == c_array.buffers

    assert list(c_array_from_capsule.view().buffer(1)) == [1, 2, 3]


def test_c_array_type_not_supported():
    with pytest.raises(TypeError, match="Can't convert object of type NoneType"):
        na.c_array(None)


def test_c_array_slice():
    array = na.c_array([1, 2, 3], na.int32())
    assert array.offset == 0
    assert array.length == 3

    array2 = array[:]
    assert array.offset == 0
    assert array.length == 3
    assert array.buffers == array2.buffers

    array2 = array[:2]
    assert array2.offset == 0
    assert array2.length == 2

    array2 = array[:-1]
    assert array2.offset == 0
    assert array2.length == 2

    array2 = array[1:]
    assert array2.offset == 1
    assert array2.length == 2

    array2 = array[-2:]
    assert array2.offset == 1
    assert array2.length == 2


def test_c_array_slice_errors():
    array = na.c_array([1, 2, 3], na.int32())

    with pytest.raises(TypeError):
        array[None]
    with pytest.raises(IndexError):
        array[4:]
    with pytest.raises(IndexError):
        array[:4]
    with pytest.raises(IndexError):
        array[1:0]


def test_c_array_from_pybuffer_uint8():
    data = b"abcdefg"
    c_array = na.c_array(data)
    assert c_array.length == len(data)
    assert c_array.null_count == 0
    assert c_array.offset == 0
    assert na.c_schema_view(c_array.schema).type == "uint8"

    c_array_view = na.c_array_view(c_array)
    assert list(c_array_view.buffer(1)) == list(data)


def test_c_array_from_pybuffer_string():
    data = b"abcdefg"
    buffer = na.c_buffer(data)._set_format("c")
    c_array = na.c_array(buffer)
    assert c_array.length == len(data)
    assert c_array.null_count == 0
    assert c_array.offset == 0
    assert na.c_schema_view(c_array.schema).type == "int8"

    c_array_view = na.c_array_view(c_array)
    assert list(c_array_view.buffer(1)) == list(data)


def test_c_array_from_pybuffer_fixed_size_binary():
    items = [b"abcd", b"efgh", b"ijkl"]
    packed = b"".join(items)
    buffer = na.c_buffer(packed)._set_format("4s")

    c_array = na.c_array(buffer)
    assert c_array.length == len(items)
    assert c_array.null_count == 0
    assert c_array.offset == 0
    assert na.c_schema_view(c_array.schema).type == "fixed_size_binary"
    assert na.c_schema_view(c_array.schema).fixed_size == 4

    c_array_view = na.c_array_view(c_array)
    assert list(c_array_view.buffer(1)) == items


def test_c_array_from_pybuffer_numpy():
    np = pytest.importorskip("numpy")

    data = np.array([1, 2, 3], dtype=np.int32)
    c_array = na.c_array(data)
    assert c_array.length == len(data)
    assert c_array.null_count == 0
    assert c_array.offset == 0
    assert na.c_schema_view(c_array.schema).type == "int32"

    c_array_view = na.c_array_view(c_array)
    assert list(c_array_view.buffer(1)) == list(data)


def test_c_array_from_iterable_empty():
    empty_string = na.c_array([], na.string())
    assert empty_string.length == 0
    assert empty_string.null_count == 0
    assert empty_string.offset == 0
    assert empty_string.n_buffers == 3

    array_view = na.c_array_view(empty_string)
    assert len(array_view.buffer(0)) == 0
    assert len(array_view.buffer(1)) == 0
    assert len(array_view.buffer(2)) == 0


def test_c_array_from_iterable_string():
    string = na.c_array(["abc", None, "defg"], na.string())
    assert string.length == 3
    assert string.null_count == 1

    array_view = na.c_array_view(string)
    assert len(array_view.buffer(0)) == 1
    assert len(array_view.buffer(1)) == 4
    assert len(array_view.buffer(2)) == 7

    # Check an item that is not a str()
    with pytest.raises(TypeError):
        na.c_array([b"1234"], na.string())


def test_c_array_from_iterable_bytes():
    string = na.c_array([b"abc", None, b"defg"], na.binary())
    assert string.length == 3
    assert string.null_count == 1

    array_view = na.c_array_view(string)
    assert len(array_view.buffer(0)) == 1
    assert len(array_view.buffer(1)) == 4
    assert len(array_view.buffer(2)) == 7

    with pytest.raises(TypeError):
        na.c_array(["1234"], na.binary())

    buf_not_bytes = na.c_buffer([1, 2, 3], na.int32())
    with pytest.raises(ValueError, match="Can't append buffer with itemsize != 1"):
        na.c_array([buf_not_bytes], na.binary())

    np = pytest.importorskip("numpy")
    buf_2d = np.ones((2, 2))
    with pytest.raises(ValueError, match="Can't append buffer with dimensions != 1"):
        na.c_array([buf_2d], na.binary())


def test_c_array_from_iterable_non_empty_nullable_without_nulls():
    c_array = na.c_array([1, 2, 3], na.int32())
    assert c_array.length == 3
    assert c_array.null_count == 0

    view = na.c_array_view(c_array)
    assert list(view.buffer(0)) == []
    assert list(view.buffer(1)) == [1, 2, 3]


def test_c_array_from_iterable_non_empty_non_nullable():
    c_array = na.c_array([1, 2, 3], na.int32(nullable=False))
    assert c_array.length == 3
    assert c_array.null_count == 0

    view = na.c_array_view(c_array)
    assert list(view.buffer(0)) == []
    assert list(view.buffer(1)) == [1, 2, 3]


def test_c_array_from_iterable_int_with_nulls():
    c_array = na.c_array([1, None, 3], na.int32())
    assert c_array.length == 3
    assert c_array.null_count == 1

    view = na.c_array_view(c_array)
    assert list(view.buffer(0).elements()) == [True, False, True] + [False] * 5
    assert list(view.buffer(1)) == [1, 0, 3]


def test_c_array_from_iterable_float_with_nulls():
    c_array = na.c_array([1, None, 3], na.float64())
    assert c_array.length == 3
    assert c_array.null_count == 1

    view = na.c_array_view(c_array)
    assert list(view.buffer(0).elements()) == [True, False, True] + [False] * 5
    assert list(view.buffer(1)) == [1.0, 0.0, 3.0]


def test_c_array_from_iterable_bool_with_nulls():
    c_array = na.c_array([True, None, False], na.bool())
    assert c_array.length == 3
    assert c_array.null_count == 1

    view = na.c_array_view(c_array)
    assert list(view.buffer(0).elements()) == [True, False, True] + [False] * 5
    assert list(view.buffer(1).elements()) == [True, False, False] + [False] * 5


def test_c_array_from_iterable_fixed_size_binary_with_nulls():
    c_array = na.c_array([b"1234", None, b"5678"], na.fixed_size_binary(4))
    assert c_array.length == 3
    assert c_array.null_count == 1

    view = na.c_array_view(c_array)
    assert list(view.buffer(0).elements()) == [True, False, True] + [False] * 5
    assert list(view.buffer(1)) == [b"1234", b"\x00\x00\x00\x00", b"5678"]


def test_c_array_from_iterable_day_time_interval_with_nulls():
    c_array = na.c_array([(1, 2), None, (3, 4)], na.interval_day_time())
    assert c_array.length == 3
    assert c_array.null_count == 1

    view = na.c_array_view(c_array)
    assert list(view.buffer(0).elements()) == [True, False, True] + [False] * 5
    assert list(view.buffer(1)) == [(1, 2), (0, 0), (3, 4)]


def test_c_array_from_iterable_month_day_nano_interval_with_nulls():
    c_array = na.c_array([(1, 2, 3), None, (4, 5, 6)], na.interval_month_day_nano())
    assert c_array.length == 3
    assert c_array.null_count == 1

    view = na.c_array_view(c_array)
    assert list(view.buffer(0).elements()) == [True, False, True] + [False] * 5
    assert list(view.buffer(1)) == [(1, 2, 3), (0, 0, 0), (4, 5, 6)]


def test_c_array_from_iterable_error():
    with pytest.raises(ValueError, match="schema is required"):
        na.c_array([1, 2, 3])


def test_c_array_from_buffers():
    c_array = na.c_array_from_buffers(na.uint8(), 5, [None, b"12345"])
    assert c_array.length == 5
    assert c_array.null_count == 0
    assert c_array.offset == 0

    array_view = na.c_array_view(c_array)
    assert array_view.storage_type == "uint8"
    assert bytes(array_view.buffer(0)) == b""
    assert bytes(array_view.buffer(1)) == b"12345"


def test_c_array_from_buffers_null_count():
    # Ensure null_count is not calculated if explicitly set
    c_array = na.c_array_from_buffers(
        na.uint8(), 7, [b"\xff", b"12345678"], null_count=1
    )
    assert c_array.null_count == 1

    # Zero nulls, explicit bitmap
    c_array = na.c_array_from_buffers(na.uint8(), 8, [b"\xff", b"12345678"])
    assert c_array.null_count == 0

    # All nulls, explicit bitmap
    c_array = na.c_array_from_buffers(na.uint8(), 8, [b"\x00", b"12345678"])
    assert c_array.null_count == 8

    # Ensure offset is considered in null count calculation
    c_array = na.c_array_from_buffers(na.uint8(), 7, [b"\xff", b"12345678"], offset=1)
    assert c_array.null_count == 0

    c_array = na.c_array_from_buffers(na.uint8(), 7, [b"\x00", b"12345678"], offset=1)
    assert c_array.null_count == 7

    # Ensure we don't access out-of-bounds memory to calculate the bitmap
    with pytest.raises(ValueError, match="Expected validity bitmap >= 2 bytes"):
        na.c_array_from_buffers(na.uint8(), 9, [b"\x00", b"123456789"])


def test_c_array_from_buffers_recursive():
    c_array = na.c_array_from_buffers(
        na.struct([na.uint8()]), 5, [None], children=[b"12345"]
    )
    assert c_array.length == 5
    assert c_array.n_children == 1

    array_view = na.c_array_view(c_array)
    assert bytes(array_view.child(0).buffer(1)) == b"12345"

    with pytest.raises(ValueError, match="Expected 1 children but got 0"):
        na.c_array_from_buffers(na.struct([na.uint8()]), 5, [], children=[])

    with pytest.raises(
        NotImplementedError,
        match="Validation for array with children is not implemented",
    ):
        na.c_array_from_buffers(
            na.struct([na.uint8()]),
            5,
            [None],
            children=[b"12345"],
            validation_level="minimal",
        )


def test_c_array_from_buffers_validation():
    # Should fail with all validation levels except none
    for validation_level in ("full", "default", "minimal"):
        with pytest.raises(
            NanoarrowException,
            match="Expected int32 array buffer 1 to have size >= 4 bytes",
        ):
            na.c_array_from_buffers(
                na.int32(), 1, [None, b"123"], validation_level=validation_level
            )

    c_array = na.c_array_from_buffers(
        na.int32(), 1, [None, b"123"], validation_level="none"
    )
    assert c_array.length == 1

    # Should only fail with validation levels of "full", and "default"
    for validation_level in ("full", "default"):
        with pytest.raises(
            NanoarrowException,
            match="Expected string array buffer 2 to have size >= 2 bytes",
        ):
            na.c_array_from_buffers(
                na.string(),
                2,
                [None, na.c_buffer([0, 1, 2], na.int32()), na.c_buffer(b"a")],
                validation_level=validation_level,
            )

    for validation_level in ("minimal", "none"):
        c_array = na.c_array_from_buffers(
            na.string(),
            2,
            [None, na.c_buffer([0, 1, 2], na.int32()), na.c_buffer(b"a")],
            validation_level=validation_level,
        )
        assert c_array.length == 2

    # Should only fail with validation level "full"
    with pytest.raises(
        NanoarrowException,
        match="Expected element size >= 0",
    ):
        na.c_array_from_buffers(
            na.string(),
            2,
            [None, na.c_buffer([0, 100, 2], na.int32()), na.c_buffer(b"ab")],
            validation_level="full",
        )

    for validation_level in ("minimal", "none"):
        c_array = na.c_array_from_buffers(
            na.string(),
            2,
            [None, na.c_buffer([0, 100, 2], na.int32()), na.c_buffer(b"ab")],
            validation_level=validation_level,
        )
        assert c_array.length == 2
