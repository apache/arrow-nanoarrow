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

import re
import sys

import pytest

import nanoarrow as na

np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")


def test_c_version():
    re_version = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(-SNAPSHOT)?$")
    assert re_version.match(na.c_version()) is not None


def test_c_schema_helper():
    schema = na.allocate_c_schema()
    assert na.c_schema(schema) is schema

    schema = na.c_schema(pa.null())
    assert isinstance(schema, na.c_lib.CSchema)

    with pytest.raises(TypeError):
        na.c_schema(1234)


def test_c_array_helper():
    array = na.allocate_c_array()
    assert na.c_array(array) is array

    array = na.c_array(pa.array([], pa.null()))
    assert isinstance(array, na.c_lib.CArray)

    with pytest.raises(TypeError):
        na.c_array(1234)


def test_array_stream_helper():
    array_stream = na.allocate_c_array_stream()
    assert na.c_array_stream(array_stream) is array_stream

    with pytest.raises(TypeError):
        na.c_array_stream(1234)


def test_array_view_helper():
    array = na.c_array(pa.array([1, 2, 3]))
    view = na.c_array_view(array)
    assert isinstance(view, na.c_lib.CArrayView)
    assert na.c_array_view(view) is view


def test_c_schema_basic():
    schema = na.allocate_c_schema()
    assert schema.is_valid() is False
    assert schema._to_string() == "[invalid: schema is released]"
    assert repr(schema) == "<nanoarrow.c_lib.CSchema <released>>"

    schema = na.c_schema(pa.schema([pa.field("some_name", pa.int32())]))

    assert schema.format == "+s"
    assert schema.flags == 0
    assert schema.metadata is None
    assert schema.n_children == 1
    assert len(list(schema.children)) == 1
    assert schema.child(0).format == "i"
    assert schema.child(0).name == "some_name"
    assert schema.child(0)._to_string() == "int32"
    assert "<nanoarrow.c_lib.CSchema int32>" in repr(schema)
    assert schema.dictionary is None

    with pytest.raises(IndexError):
        schema.child(1)


def test_c_schema_dictionary():
    schema = na.c_schema(pa.dictionary(pa.int32(), pa.utf8()))
    assert schema.format == "i"
    assert schema.dictionary.format == "u"
    assert "dictionary: <nanoarrow.c_lib.CSchema string" in repr(schema)


def test_schema_metadata():
    meta = {"key1": "value1", "key2": "value2"}
    schema = na.c_schema(pa.field("", pa.int32(), metadata=meta))

    assert len(schema.metadata) == 2

    meta2 = {k: v for k, v in schema.metadata}
    assert list(meta2.keys()) == ["key1", "key2"]
    assert list(meta2.values()) == [b"value1", b"value2"]
    assert "'key1': b'value1'" in repr(schema)


def test_c_schema_view():
    schema = na.allocate_c_schema()
    with pytest.raises(RuntimeError):
        na.c_schema_view(schema)

    schema = na.c_schema(pa.int32())
    view = na.c_schema_view(schema)
    assert "- type: 'int32'" in repr(view)
    assert view.type == "int32"
    assert view.storage_type == "int32"

    assert view.fixed_size is None
    assert view.decimal_bitwidth is None
    assert view.decimal_scale is None
    assert view.time_unit is None
    assert view.timezone is None
    assert view.union_type_ids is None
    assert view.extension_name is None
    assert view.extension_metadata is None


def test_c_schema_view_extra_params():
    schema = na.c_schema(pa.binary(12))
    view = na.c_schema_view(schema)
    assert view.fixed_size == 12

    schema = na.c_schema(pa.list_(pa.int32(), 12))
    assert view.fixed_size == 12

    schema = na.c_schema(pa.decimal128(10, 3))
    view = na.c_schema_view(schema)
    assert view.decimal_bitwidth == 128
    assert view.decimal_precision == 10
    assert view.decimal_scale == 3

    schema = na.c_schema(pa.decimal256(10, 3))
    view = na.c_schema_view(schema)
    assert view.decimal_bitwidth == 256
    assert view.decimal_precision == 10
    assert view.decimal_scale == 3

    schema = na.c_schema(pa.duration("us"))
    view = na.c_schema_view(schema)
    assert view.time_unit == "us"

    schema = na.c_schema(pa.timestamp("us", tz="America/Halifax"))
    view = na.c_schema_view(schema)
    assert view.type == "timestamp"
    assert view.storage_type == "int64"
    assert view.time_unit == "us"
    assert view.timezone == "America/Halifax"

    meta = {
        "ARROW:extension:name": "some_name",
        "ARROW:extension:metadata": "some_metadata",
    }
    schema = na.c_schema(pa.field("", pa.int32(), metadata=meta))
    view = na.c_schema_view(schema)
    assert view.extension_name == "some_name"
    assert view.extension_metadata == b"some_metadata"


def test_c_array_empty():
    array = na.allocate_c_array()
    assert array.is_valid() is False
    assert repr(array) == "<nanoarrow.c_lib.CArray <released>>"


def test_c_array():
    array = na.c_array(pa.array([1, 2, 3], pa.int32()))
    assert array.is_valid() is True
    assert array.length == 3
    assert array.offset == 0
    assert array.null_count == 0
    assert array.n_buffers == 2
    assert len(array.buffers) == 2
    assert array.buffers[0] == 0
    assert array.n_children == 0
    assert len(list(array.children)) == 0
    assert array.dictionary is None
    assert "<nanoarrow.c_lib.CArray int32" in repr(array)


def test_c_array_recursive():
    array = na.c_array(pa.record_batch([pa.array([1, 2, 3], pa.int32())], ["col"]))
    assert array.n_children == 1
    assert len(list(array.children)) == 1
    assert array.child(0).length == 3
    assert array.child(0).schema._to_string() == "int32"
    assert "'col': <nanoarrow.c_lib.CArray int32" in repr(array)

    with pytest.raises(IndexError):
        array.child(-1)


def test_c_array_dictionary():
    array = na.c_array(pa.array(["a", "b", "b"]).dictionary_encode())
    assert array.length == 3
    assert array.dictionary.length == 2
    assert "dictionary: <nanoarrow.c_lib.CArray string>" in repr(array)


def test_c_array_view():
    array = na.c_array(pa.array([1, 2, 3], pa.int32()))
    view = na.c_array_view(array)

    assert view.storage_type == "int32"
    assert "- storage_type: 'int32'" in repr(view)
    assert "data <int32[12 b] 1 2 3>" in repr(view)

    data_buffer = memoryview(view.buffer(1))
    data_buffer_copy = bytes(data_buffer)
    assert len(data_buffer_copy) == 12

    if sys.byteorder == "little":
        assert data_buffer_copy == b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00"
    else:
        assert data_buffer_copy == b"\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03"

    with pytest.raises(IndexError):
        view.child(0)

    with pytest.raises(IndexError):
        view.child(-1)


def test_c_array_view_recursive():
    pa_array_child = pa.array([1, 2, 3], pa.int32())
    pa_array = pa.record_batch([pa_array_child], names=["some_column"])

    array = na.c_array(pa_array)

    assert array.schema.format == "+s"
    assert array.length == 3
    assert array.n_children == 1
    assert len(list(array.children)) == 1

    assert array.child(0).schema.format == "i"
    assert array.child(0).length == 3
    assert array.child(0).schema._addr() == array.schema.child(0)._addr()

    view = na.c_array_view(array)
    assert view.n_buffers == 1
    assert len(list(view.buffers)) == 1
    assert view.n_children == 1
    assert len(list(view.children)) == 1

    assert view.child(0).n_buffers == 2
    assert len(list(view.child(0).buffers)) == 2
    assert "- children[1]" in repr(view)


def test_c_array_view_dictionary():
    pa_array = pa.array(["a", "b", "b"], pa.dictionary(pa.int32(), pa.utf8()))
    array = na.c_array(pa_array)

    assert array.schema.format == "i"
    assert array.dictionary.schema.format == "u"

    view = na.c_array_view(array)
    assert view.n_buffers == 2
    assert view.dictionary.n_buffers == 3
    assert "- dictionary: <nanoarrow.c_lib.CArrayView>" in repr(view)


def test_buffers_integer():
    data_types = [
        (pa.uint8(), np.uint8()),
        (pa.int8(), np.int8()),
        (pa.uint16(), np.uint16()),
        (pa.int16(), np.int16()),
        (pa.uint32(), np.uint32()),
        (pa.int32(), np.int32()),
        (pa.uint64(), np.uint64()),
        (pa.int64(), np.int64()),
    ]

    for pa_type, np_type in data_types:
        view = na.c_array_view(pa.array([0, 1, 2], pa_type))
        data_buffer = view.buffer(1)

        # Check via buffer interface
        np.testing.assert_array_equal(
            np.array(data_buffer), np.array([0, 1, 2], np_type)
        )

        # Check via iterator interface
        assert list(data_buffer) == [0, 1, 2]

        # Check via buffer get_item interface
        assert [data_buffer[i] for i in range(len(data_buffer))] == list(data_buffer)

        # Check repr
        assert "0 1 2" in repr(data_buffer)


def test_buffers_float():
    data_types = [
        (pa.float32(), np.float32()),
        (pa.float64(), np.float64()),
    ]

    for pa_type, np_type in data_types:
        view = na.c_array_view(pa.array([0, 1, 2], pa_type))
        data_buffer = view.buffer(1)

        # Check via buffer interface
        np.testing.assert_array_equal(
            np.array(data_buffer), np.array([0, 1, 2], np_type)
        )

        # Check via iterator interface
        assert list(data_buffer) == [0.0, 1.0, 2.0]

        # Check via buffer get_item interface
        assert [data_buffer[i] for i in range(len(data_buffer))] == list(data_buffer)

        # Check repr
        assert "0.0 1.0 2.0" in repr(data_buffer)


def test_buffers_half_float():
    # pyarrrow can only create half_float from np.float16()
    np_array = np.array([0, 1, 2], np.float16())
    view = na.c_array_view(pa.array(np_array))
    data_buffer = view.buffer(1)

    # Check via buffer interface
    np.testing.assert_array_equal(
        np.array(data_buffer), np.array([0, 1, 2], np.float16())
    )

    # Check via iterator interface
    assert list(data_buffer) == [0.0, 1.0, 2.0]

    # Check via buffer get_item interface
    assert [data_buffer[i] for i in range(len(data_buffer))] == list(data_buffer)

    # Check repr
    assert "0.0 1.0 2.0" in repr(data_buffer)


def test_buffers_bool():
    view = na.c_array_view(pa.array([True, True, True, False]))
    data_buffer = view.buffer(1)

    assert data_buffer.size_bytes == 1

    # Check via buffer interface
    np.testing.assert_array_equal(
        np.array(data_buffer), np.array([1 + 2 + 4], np.int32())
    )

    # Check via iterator interface
    assert list(data_buffer) == [1 + 2 + 4]

    # Check via buffer get_item interface
    assert [data_buffer[i] for i in range(len(data_buffer))] == list(data_buffer)

    # Check via element interface
    assert data_buffer.n_elements == 8
    assert list(data_buffer.elements()) == [True] * 3 + [False] * 5
    assert [data_buffer.element(i) for i in range(data_buffer.n_elements)] == list(
        data_buffer.elements()
    )

    with pytest.raises(IndexError):
        data_buffer[8]
    with pytest.raises(IndexError):
        data_buffer[-1]
    with pytest.raises(IndexError):
        next(data_buffer.elements(-1, 4))
    with pytest.raises(IndexError):
        next(data_buffer.elements(7, 2))

    # Check repr
    assert "11100000" in repr(data_buffer)


def test_buffers_string():
    view = na.c_array_view(pa.array(["a", "bc", "def"]))

    assert view.buffer(0).size_bytes == 0
    assert view.buffer(1).size_bytes == 16
    assert view.buffer(2).size_bytes == 6

    # Check via buffer interface
    np.testing.assert_array_equal(
        np.array(view.buffer(1)), np.array([0, 1, 3, 6], np.int32())
    )
    np.testing.assert_array_equal(
        np.array(view.buffer(2)), np.array(list("abcdef"), dtype="|S1")
    )

    # Check via iterator interface
    assert list(view.buffer(0)) == []
    assert list(view.buffer(1)) == [0, 1, 3, 6]
    assert list(view.buffer(2)) == [item.encode("UTF-8") for item in "abcdef"]

    # Check repr
    assert "b'abcdef'" in repr(view.buffer(2))


def test_buffers_binary():
    view = na.c_array_view(pa.array([b"a", b"bc", b"def"]))

    assert view.buffer(0).size_bytes == 0
    assert view.buffer(1).size_bytes == 16
    assert view.buffer(2).size_bytes == 6

    # Check via buffer interface
    np.testing.assert_array_equal(
        np.array(view.buffer(1)), np.array([0, 1, 3, 6], np.int32())
    )
    np.testing.assert_array_equal(np.array(view.buffer(2)), np.array(list(b"abcdef")))
    np.testing.assert_array_equal(
        np.array(list(view.buffer(2))), np.array(list(b"abcdef"))
    )

    # Check via iterator interface
    assert list(view.buffer(0)) == []
    assert list(view.buffer(1)) == [0, 1, 3, 6]
    assert list(view.buffer(2)) == [int(item) for item in b"abcdef"]

    # Check repr
    assert "b'abcdef'" in repr(view.buffer(2))


def test_buffers_fixed_size_binary():
    view = na.c_array_view(pa.array([b"abc", b"def", b"ghi"], pa.binary(3)))

    assert view.buffer(1).size_bytes == 9

    # Check via buffer interface
    np.testing.assert_array_equal(
        np.array(list(view.buffer(1))), np.array([b"abc", b"def", b"ghi"])
    )

    # Check via iterator interface
    assert list(view.buffer(1)) == [b"abc", b"def", b"ghi"]


def test_buffers_interval_month_day_nano():
    view = na.c_array_view(
        pa.array([pa.scalar((1, 15, -30), type=pa.month_day_nano_interval())])
    )

    assert view.buffer(1).size_bytes == 16

    # Check via buffer interface
    np.testing.assert_array_equal(
        np.array(list(view.buffer(1))), np.array([(1, 15, -30)])
    )

    # Check via iterator interface
    assert list(view.buffer(1)) == [(1, 15, -30)]


def test_c_array_stream():
    array_stream = na.allocate_c_array_stream()
    assert na.c_array_stream(array_stream) is array_stream
    assert repr(array_stream) == "<nanoarrow.c_lib.CArrayStream <released>>"

    assert array_stream.is_valid() is False
    with pytest.raises(RuntimeError):
        array_stream.get_schema()
    with pytest.raises(RuntimeError):
        array_stream.get_next()

    pa_array_child = pa.array([1, 2, 3], pa.int32())
    pa_array = pa.record_batch([pa_array_child], names=["some_column"])
    reader = pa.RecordBatchReader.from_batches(pa_array.schema, [pa_array])
    array_stream = na.c_array_stream(reader)

    assert array_stream.is_valid() is True
    assert "struct<some_column: int32>" in repr(array_stream)

    array = array_stream.get_next()
    assert array.schema.child(0).name == "some_column"
    with pytest.raises(StopIteration):
        array_stream.get_next()


def test_c_array_stream_iter():
    pa_array_child = pa.array([1, 2, 3], pa.int32())
    pa_array = pa.record_batch([pa_array_child], names=["some_column"])
    reader = pa.RecordBatchReader.from_batches(pa_array.schema, [pa_array])
    array_stream = na.c_array_stream(reader)

    arrays = list(array_stream)
    assert len(arrays) == 1
    assert arrays[0].schema.child(0).name == "some_column"
