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

import array
from datetime import date, datetime, timezone

import pytest
from nanoarrow._array import CArrayBuilder
from nanoarrow._utils import NanoarrowException
from nanoarrow.c_schema import c_schema_view

import nanoarrow as na
from nanoarrow import device


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
    with pytest.raises(TypeError, match="Can't resolve ArrayBuilder"):
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
    msg = "Can't resolve ArrayBuilder for object of type NoneType"
    with pytest.raises(TypeError, match=msg):
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


def test_c_array_shallow_copy():
    import gc
    import platform

    from nanoarrow._utils import get_pyobject_buffer_count

    if platform.python_implementation() == "PyPy":
        pytest.skip(
            "Reference counting/garbage collection is non-deterministic on PyPy"
        )

    gc.collect()
    initial_ref_count = get_pyobject_buffer_count()

    # Create an array with children
    array = na.c_array_from_buffers(
        na.struct({"col1": na.int32(), "col2": na.int64()}),
        3,
        [None],
        children=[na.c_array([1, 2, 3], na.int32()), na.c_array([4, 5, 6], na.int32())],
        move=True,
    )

    # The move=True should have prevented a shallow copy of the children
    # when constructing the array.
    assert get_pyobject_buffer_count() == initial_ref_count

    # Force a shallow copy via the array protocol and ensure we saved
    # references to two additional buffers.
    _, col1_capsule = array.child(0).__arrow_c_array__()
    assert get_pyobject_buffer_count() == (initial_ref_count + 1)

    _, col2_capsule = array.child(1).__arrow_c_array__()
    assert get_pyobject_buffer_count() == (initial_ref_count + 2)

    # Ensure that the references can be removed
    del col1_capsule
    assert get_pyobject_buffer_count() == (initial_ref_count + 1)

    del col2_capsule
    assert get_pyobject_buffer_count() == initial_ref_count


def test_c_array_builder_init():
    builder = CArrayBuilder.allocate()

    with pytest.raises(RuntimeError, match="CArrayBuilder is not initialized"):
        builder.is_empty()

    builder.init_from_type(na.Type.INT32.value)
    assert builder.is_empty()

    with pytest.raises(RuntimeError, match="CArrayBuilder is already initialized"):
        builder.init_from_type(na.Type.INT32.value)

    with pytest.raises(RuntimeError, match="CArrayBuilder is already initialized"):
        builder.init_from_schema(na.c_schema(na.int32()))


def test_c_array_from_pybuffer_uint8():
    data = b"abcdefg"
    c_array = na.c_array(data)
    assert c_array.length == len(data)
    assert c_array.null_count == 0
    assert c_array.offset == 0
    assert c_schema_view(c_array.schema).type == "uint8"

    c_array_view = c_array.view()
    assert list(c_array_view.buffer(1)) == list(data)


def test_c_array_from_pybuffer_string():
    data = b"abcdefg"
    buffer = na.c_buffer(data)._set_format("c")
    c_array = na.c_array(buffer)
    assert c_array.length == len(data)
    assert c_array.null_count == 0
    assert c_array.offset == 0
    assert c_schema_view(c_array.schema).type == "int8"

    c_array_view = c_array.view()
    assert list(c_array_view.buffer(1)) == list(data)


def test_c_array_from_pybuffer_fixed_size_binary():
    items = [b"abcd", b"efgh", b"ijkl"]
    packed = b"".join(items)
    buffer = na.c_buffer(packed)._set_format("4s")

    c_array = na.c_array(buffer)
    assert c_array.length == len(items)
    assert c_array.null_count == 0
    assert c_array.offset == 0
    assert c_schema_view(c_array.schema).type == "fixed_size_binary"
    assert c_schema_view(c_array.schema).fixed_size == 4

    c_array_view = c_array.view()
    assert list(c_array_view.buffer(1)) == items


def test_c_array_from_pybuffer_numpy():
    np = pytest.importorskip("numpy")

    data = np.array([1, 2, 3], dtype=np.int32)
    c_array = na.c_array(data)
    assert c_array.length == len(data)
    assert c_array.null_count == 0
    assert c_array.offset == 0
    assert c_schema_view(c_array.schema).type == "int32"

    c_array_view = c_array.view()
    assert list(c_array_view.buffer(1)) == list(data)


def test_c_array_from_iterable_empty():
    empty_string = na.c_array([], na.string())
    assert empty_string.length == 0
    assert empty_string.null_count == 0
    assert empty_string.offset == 0
    assert empty_string.n_buffers == 3

    array_view = empty_string.view()
    assert len(array_view.buffer(0)) == 0
    assert len(array_view.buffer(1)) == 0
    assert len(array_view.buffer(2)) == 0


def test_c_array_from_iterable_string():
    string = na.c_array(["abc", None, "defg"], na.string())
    assert string.length == 3
    assert string.null_count == 1

    array_view = string.view()
    assert len(array_view.buffer(0)) == 1
    assert len(array_view.buffer(1)) == 4
    assert len(array_view.buffer(2)) == 7

    # Check an item that is not a str()
    with pytest.raises(ValueError):
        na.c_array([b"1234"], na.string())


def test_c_array_from_iterable_string_view():
    string = na.c_array(
        ["abc", None, "a string longer than 12 bytes"], na.string_view()
    )
    assert string.length == 3
    assert string.null_count == 1
    assert string.n_buffers == 4

    array_view = string.view()
    assert len(array_view.buffer(0)) == 1
    assert bytes(array_view.buffer(2)) == b"a string longer than 12 bytes"
    assert list(array_view.buffer(3)) == [len("a string longer than 12 bytes")]

    # Make sure this also works when all strings are inlined (i.e., no variadic buffers)
    string = na.c_array(["abc", None, "short string"], na.string_view())
    assert string.length == 3
    assert string.null_count == 1
    assert string.n_buffers == 3

    array_view = string.view()
    assert len(array_view.buffer(0)) == 1
    assert len(array_view.buffer(1)) == 3
    assert len(bytes(array_view.buffer(1))) == 3 * 16
    assert list(array_view.buffer(2)) == []


def test_c_array_from_iterable_bytes():
    string = na.c_array([b"abc", None, b"defg"], na.binary())
    assert string.length == 3
    assert string.null_count == 1

    array_view = string.view()
    assert len(array_view.buffer(0)) == 1
    assert len(array_view.buffer(1)) == 4
    assert len(array_view.buffer(2)) == 7

    with pytest.raises(ValueError):
        na.c_array(["1234"], na.binary())

    buf_not_bytes = na.c_buffer([1, 2, 3], na.int32())
    with pytest.raises(ValueError, match="Can't append buffer with itemsize != 1"):
        na.c_array([buf_not_bytes], na.binary())

    np = pytest.importorskip("numpy")
    buf_2d = np.ones((2, 2))
    with pytest.raises(ValueError, match="Can't append buffer with dimensions != 1"):
        na.c_array([buf_2d], na.binary())


def test_c_array_from_iterable_view():
    string = na.c_array(
        [b"abc", None, b"a string longer than 12 bytes"], na.binary_view()
    )
    assert string.length == 3
    assert string.null_count == 1
    assert string.n_buffers == 4

    array_view = string.view()
    assert len(array_view.buffer(0)) == 1
    assert bytes(array_view.buffer(2)) == b"a string longer than 12 bytes"
    assert list(array_view.buffer(3)) == [len("a string longer than 12 bytes")]


def test_c_array_from_iterable_non_empty_nullable_without_nulls():
    c_array = na.c_array([1, 2, 3], na.int32())
    assert c_array.length == 3
    assert c_array.null_count == 0

    view = c_array.view()
    assert list(view.buffer(0)) == []
    assert list(view.buffer(1)) == [1, 2, 3]


def test_c_array_from_iterable_non_empty_non_nullable():
    c_array = na.c_array([1, 2, 3], na.int32(nullable=False))
    assert c_array.length == 3
    assert c_array.null_count == 0

    view = c_array.view()
    assert list(view.buffer(0)) == []
    assert list(view.buffer(1)) == [1, 2, 3]


def test_c_array_from_iterable_int_with_nulls():
    c_array = na.c_array([1, None, 3], na.int32())
    assert c_array.length == 3
    assert c_array.null_count == 1

    view = c_array.view()
    assert list(view.buffer(0).elements()) == [True, False, True] + [False] * 5
    assert list(view.buffer(1)) == [1, 0, 3]


def test_c_array_from_iterable_float_with_nulls():
    c_array = na.c_array([1, None, 3], na.float64())
    assert c_array.length == 3
    assert c_array.null_count == 1

    view = c_array.view()
    assert list(view.buffer(0).elements()) == [True, False, True] + [False] * 5
    assert list(view.buffer(1)) == [1.0, 0.0, 3.0]


def test_c_array_from_iterable_bool_with_nulls():
    c_array = na.c_array([True, None, False], na.bool_())
    assert c_array.length == 3
    assert c_array.null_count == 1

    view = c_array.view()
    assert list(view.buffer(0).elements()) == [True, False, True] + [False] * 5
    assert list(view.buffer(1).elements()) == [True, False, False] + [False] * 5


def test_c_array_from_iterable_fixed_size_binary_with_nulls():
    c_array = na.c_array([b"1234", None, b"5678"], na.fixed_size_binary(4))
    assert c_array.length == 3
    assert c_array.null_count == 1

    view = c_array.view()
    assert list(view.buffer(0).elements()) == [True, False, True] + [False] * 5
    assert list(view.buffer(1)) == [b"1234", b"\x00\x00\x00\x00", b"5678"]


def test_c_array_from_iterable_day_time_interval_with_nulls():
    c_array = na.c_array([(1, 2), None, (3, 4)], na.interval_day_time())
    assert c_array.length == 3
    assert c_array.null_count == 1

    view = c_array.view()
    assert list(view.buffer(0).elements()) == [True, False, True] + [False] * 5
    assert list(view.buffer(1)) == [(1, 2), (0, 0), (3, 4)]


def test_c_array_from_iterable_month_day_nano_interval_with_nulls():
    c_array = na.c_array([(1, 2, 3), None, (4, 5, 6)], na.interval_month_day_nano())
    assert c_array.length == 3
    assert c_array.null_count == 1

    view = c_array.view()
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

    array_view = c_array.view()
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

    array_view = c_array.view()
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


def test_c_array_from_buffers_device():
    c_device_array = na.c_array_from_buffers(
        na.uint8(), 5, [None, b"12345"], device=device.cpu()
    )
    assert isinstance(c_device_array, device.CDeviceArray)

    c_array = na.c_array(c_device_array)
    assert c_array.length == 5
    assert bytes(c_array.view().buffer(1)) == b"12345"


def test_c_array_timestamp_seconds():
    d1 = int(round(datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp()))
    d2 = int(round(datetime(1985, 12, 31, tzinfo=timezone.utc).timestamp()))
    d3 = int(round(datetime(2005, 3, 4, tzinfo=timezone.utc).timestamp()))
    c_array = na.c_array([d1, d2, d3], na.timestamp("s"))
    assert c_array.length == 3
    assert c_array.null_count == 0
    view = c_array.view()
    assert list(view.buffer(0)) == []
    assert list(view.buffer(1)) == [d1, d2, d3]


def test_c_array_timestamp_seconds_from_pybuffer():
    d1 = int(round(datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp()))
    d2 = int(round(datetime(1985, 12, 31, tzinfo=timezone.utc).timestamp()))
    d3 = int(round(datetime(2005, 3, 4, tzinfo=timezone.utc).timestamp()))
    c_array = na.c_array(array.array("q", [d1, d2, d3]), na.timestamp("s"))
    assert c_array.length == 3
    assert c_array.null_count == 0
    view = c_array.view()
    assert list(view.buffer(0)) == []
    assert list(view.buffer(1)) == [d1, d2, d3]


def test_c_array_timestamp_milliseconds():
    d1 = int(round(datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp() * 1e3))
    d2 = int(round(datetime(1985, 12, 31, tzinfo=timezone.utc).timestamp() * 1e3))
    d3 = int(round(datetime(2005, 3, 4, tzinfo=timezone.utc).timestamp() * 1e3))
    c_array = na.c_array([d1, d2, d3], na.timestamp("ms"))
    assert c_array.length == 3
    assert c_array.null_count == 0
    view = c_array.view()
    assert list(view.buffer(0)) == []
    assert list(view.buffer(1)) == [d1, d2, d3]


def test_c_array_timestamp_milliseconds_from_pybuffer():
    d1 = int(round(datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp() * 1e3))
    d2 = int(round(datetime(1985, 12, 31, tzinfo=timezone.utc).timestamp() * 1e3))
    d3 = int(round(datetime(2005, 3, 4, tzinfo=timezone.utc).timestamp() * 1e3))
    c_array = na.c_array(array.array("q", [d1, d2, d3]), na.timestamp("ms"))
    assert c_array.length == 3
    assert c_array.null_count == 0
    view = c_array.view()
    assert list(view.buffer(0)) == []
    assert list(view.buffer(1)) == [d1, d2, d3]


def test_c_array_timestamp_microseconds():
    d1 = int(round(datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp() * 1e6))
    d2 = int(round(datetime(1985, 12, 31, tzinfo=timezone.utc).timestamp() * 1e6))
    d3 = int(round(datetime(2005, 3, 4, tzinfo=timezone.utc).timestamp() * 1e6))
    c_array = na.c_array([d1, d2, d3], na.timestamp("us"))
    assert c_array.length == 3
    assert c_array.null_count == 0
    view = c_array.view()
    assert list(view.buffer(0)) == []
    assert list(view.buffer(1)) == [d1, d2, d3]


def test_c_array_timestamp_nanoseconds():
    d1 = int(round(datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp() * 1e9))
    d2 = int(round(datetime(1985, 12, 31, tzinfo=timezone.utc).timestamp() * 1e9))
    d3 = int(round(datetime(2005, 3, 4).timestamp() * 1e9))
    c_array = na.c_array([d1, d2, d3], na.timestamp("ns"))
    assert c_array.length == 3
    assert c_array.null_count == 0
    view = c_array.view()
    assert list(view.buffer(0)) == []
    assert list(view.buffer(1)) == [d1, d2, d3]


def test_c_array_duration():
    unix_epoch = date(1970, 1, 1)
    d1, d2, d3 = date(1970, 1, 2), date(1970, 1, 3), date(1970, 1, 4)
    d1_duration_in_ms = int(round((d1 - unix_epoch).total_seconds() * 1e3))
    d2_duration_in_ms = int(round((d2 - unix_epoch).total_seconds() * 1e3))
    d3_duration_in_ms = int(round((d3 - unix_epoch).total_seconds() * 1e3))
    c_array = na.c_array(
        [d1_duration_in_ms, d2_duration_in_ms, d3_duration_in_ms], na.duration("ms")
    )
    assert c_array.length == 3
    assert c_array.null_count == 0
    view = c_array.view()
    assert list(view.buffer(0)) == []
    assert list(view.buffer(1)) == [
        d1_duration_in_ms,
        d2_duration_in_ms,
        d3_duration_in_ms,
    ]


def test_device_array_errors(cuda_device):
    if not cuda_device:
        pytest.skip("CUDA device not available")

    # Check that we can't create a CUDA array from CPU buffers
    with pytest.raises(ValueError, match="are not identical"):
        na.c_array_from_buffers(
            na.int64(),
            3,
            [None, na.c_buffer([1, 2, 3], na.int64())],
            device=cuda_device,
        )

    # Check that we can't create a CUDA array from CPU children
    with pytest.raises(ValueError, match="are not identical"):
        na.c_array_from_buffers(
            na.struct([na.int64()]),
            length=0,
            buffers=[None],
            children=[na.c_array([], na.int64())],
            device=cuda_device,
        )


def test_array_from_dlpack_cuda(cuda_device):
    from nanoarrow.device import CDeviceArray, DeviceType

    cp = pytest.importorskip("cupy")
    if not cuda_device:
        pytest.skip("CUDA device not available")

    gpu_validity = cp.array([255], cp.uint8)
    gpu_array = cp.array([1, 2, 3], cp.int64)

    c_array = na.c_array_from_buffers(
        na.int64(),
        3,
        [gpu_validity, gpu_array],
        move=True,
        device=cuda_device,
    )
    assert isinstance(c_array, CDeviceArray)
    assert c_array.device_type == DeviceType.CUDA
    assert c_array.device_id == 0

    c_array_view = c_array.view()
    assert c_array_view.storage_type == "int64"
    assert c_array_view.buffer(0).device == cuda_device
    assert c_array_view.buffer(1).device == cuda_device
    assert len(c_array_view) == 3

    # Make sure we don't attempt accessing a GPU buffer to calculate the null count
    assert c_array_view.null_count == -1

    # Check that buffers made it all the way through
    cp.testing.assert_array_equal(cp.from_dlpack(c_array_view.buffer(1)), gpu_array)

    # Also check a nested array
    c_array_struct = na.c_array_from_buffers(
        na.struct([na.int64()]),
        3,
        buffers=[None],
        children=[c_array],
        move=True,
        device=cuda_device,
    )
    assert c_array_struct.device_type == DeviceType.CUDA
    assert c_array_struct.device_id == 0

    c_array_view_struct = c_array_struct.view()
    assert c_array_view_struct.storage_type == "struct"
    assert c_array_view_struct.buffer(0).device == cuda_device

    c_array_view_child = c_array_view_struct.child(0)
    assert c_array_view_child.buffer(0).device == cuda_device
    assert c_array_view_child.buffer(1).device == cuda_device
    cp.testing.assert_array_equal(
        cp.from_dlpack(c_array_view_child.buffer(1)), gpu_array
    )
