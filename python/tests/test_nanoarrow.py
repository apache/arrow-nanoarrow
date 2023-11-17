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

import numpy as np
import pyarrow as pa
import pytest

import nanoarrow as na


def test_c_version():
    re_version = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(-SNAPSHOT)?$")
    assert re_version.match(na.c_version()) is not None


def test_schema_helper():
    schema = na.Schema.allocate()
    assert na.schema(schema) is schema

    schema = na.schema(pa.null())
    assert isinstance(schema, na.Schema)

    with pytest.raises(TypeError):
        na.schema(None)


def test_array_helper():
    array = na.Array.allocate(na.Schema.allocate())
    assert na.array(array) is array

    array = na.array(pa.array([], pa.null()))
    assert isinstance(array, na.Array)

    with pytest.raises(TypeError):
        na.array(None)


def test_array_stream_helper():
    array_stream = na.ArrayStream.allocate()
    assert na.array_stream(array_stream) is array_stream

    with pytest.raises(TypeError):
        na.array_stream(None)


def test_array_view_helper():
    array = na.array(pa.array([1, 2, 3]))
    view = na.array_view(array)
    assert isinstance(view, na.ArrayView)
    assert na.array_view(view) is view


def test_schema_basic():
    schema = na.Schema.allocate()
    assert schema.is_valid() is False
    assert schema._to_string() == "[invalid: schema is released]"
    assert repr(schema) == "<released nanoarrow.Schema>"

    schema = na.schema(pa.schema([pa.field("some_name", pa.int32())]))

    assert schema.format == "+s"
    assert schema.flags == 0
    assert schema.metadata is None
    assert len(schema.children) == 1
    assert schema.children[0].format == "i"
    assert schema.children[0].name == "some_name"
    assert schema.children[0]._to_string() == "int32"
    assert "<nanoarrow.Schema int32>" in repr(schema)
    assert schema.dictionary is None

    with pytest.raises(IndexError):
        schema.children[1]


def test_schema_dictionary():
    schema = na.schema(pa.dictionary(pa.int32(), pa.utf8()))
    assert schema.format == "i"
    assert schema.dictionary.format == "u"
    assert "dictionary: <nanoarrow.Schema string" in repr(schema)


def test_schema_metadata():
    meta = {"key1": "value1", "key2": "value2"}
    schema = na.schema(pa.field("", pa.int32(), metadata=meta))

    assert len(schema.metadata) == 2

    meta2 = {k: v for k, v in schema.metadata}
    assert list(meta2.keys()) == ["key1", "key2"]
    assert list(meta2.values()) == [b"value1", b"value2"]
    assert "'key1': b'value1'" in repr(schema)


def test_schema_view():
    schema = na.Schema.allocate()
    with pytest.raises(RuntimeError):
        schema.view()

    schema = na.schema(pa.int32())
    view = schema.view()
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


def test_schema_view_extra_params():
    schema = na.schema(pa.binary(12))
    view = schema.view()
    assert view.fixed_size == 12

    schema = na.schema(pa.list_(pa.int32(), 12))
    assert view.fixed_size == 12

    schema = na.schema(pa.decimal128(10, 3))
    view = schema.view()
    assert view.decimal_bitwidth == 128
    assert view.decimal_precision == 10
    assert view.decimal_scale == 3

    schema = na.schema(pa.decimal256(10, 3))
    view = schema.view()
    assert view.decimal_bitwidth == 256
    assert view.decimal_precision == 10
    assert view.decimal_scale == 3

    schema = na.schema(pa.duration("us"))
    view = schema.view()
    assert view.time_unit == "us"

    schema = na.schema(pa.timestamp("us", tz="America/Halifax"))
    view = schema.view()
    assert view.type == "timestamp"
    assert view.storage_type == "int64"
    assert view.time_unit == "us"
    assert view.timezone == "America/Halifax"

    meta = {
        "ARROW:extension:name": "some_name",
        "ARROW:extension:metadata": "some_metadata",
    }
    schema = na.schema(pa.field("", pa.int32(), metadata=meta))
    view = schema.view()
    assert view.extension_name == "some_name"
    assert view.extension_metadata == b"some_metadata"


def test_array_empty():
    array = na.Array.allocate(na.Schema.allocate())
    assert array.is_valid() is False
    assert repr(array) == "<released nanoarrow.Array>"


def test_array():
    array = na.array(pa.array([1, 2, 3], pa.int32()))
    assert array.is_valid() is True
    assert array.length == 3
    assert array.offset == 0
    assert array.null_count == 0
    assert len(array.buffers) == 2
    assert array.buffers[0] == 0
    assert len(array.children) == 0
    assert array.dictionary is None
    assert "<nanoarrow.Array int32" in repr(array)


def test_array_recursive():
    array = na.array(pa.record_batch([pa.array([1, 2, 3], pa.int32())], ["col"]))
    assert len(array.children) == 1
    assert array.children[0].length == 3
    assert array.children[0].schema._to_string() == "int32"
    assert "'col': <nanoarrow.Array int32" in repr(array)

    with pytest.raises(IndexError):
        array.children[1]


def test_array_dictionary():
    array = na.array(pa.array(["a", "b", "b"]).dictionary_encode())
    assert array.length == 3
    assert array.dictionary.length == 2
    assert "dictionary: <nanoarrow.Array string>" in repr(array)


def test_array_view():
    array = na.array(pa.array([1, 2, 3], pa.int32()))
    view = na.array_view(array)

    assert view.schema is array.schema

    data_buffer = memoryview(view.buffers[1])
    data_buffer_copy = bytes(data_buffer)
    assert len(data_buffer_copy) == 12

    if sys.byteorder == "little":
        assert data_buffer_copy == b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00"
    else:
        assert data_buffer_copy == b"\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03"

    with pytest.raises(IndexError):
        view.children[1]


def test_array_view_recursive():
    pa_array_child = pa.array([1, 2, 3], pa.int32())
    pa_array = pa.record_batch([pa_array_child], names=["some_column"])

    array = na.array(pa_array)

    assert array.schema.format == "+s"
    assert array.length == 3
    assert len(array.children) == 1

    assert array.children[0].schema.format == "i"
    assert array.children[0].length == 3
    assert array.children[0].schema._addr() == array.schema.children[0]._addr()

    view = na.array_view(array)
    assert len(view.buffers) == 1
    assert len(view.children) == 1
    assert view.schema._addr() == array.schema._addr()

    assert len(view.children[0].buffers) == 2
    assert view.children[0].schema._addr() == array.schema.children[0]._addr()
    assert view.children[0].schema._addr() == array.children[0].schema._addr()


def test_array_view_dictionary():
    pa_array = pa.array(["a", "b", "b"], pa.dictionary(pa.int32(), pa.utf8()))
    array = na.array(pa_array)

    assert array.schema.format == "i"
    assert array.dictionary.schema.format == "u"

    view = na.array_view(array)
    assert len(view.buffers) == 2
    assert len(view.dictionary.buffers) == 3


def test_buffers_data():
    data_types = [
        (pa.uint8(), np.uint8()),
        (pa.int8(), np.int8()),
        (pa.uint16(), np.uint16()),
        (pa.int16(), np.int16()),
        (pa.uint32(), np.uint32()),
        (pa.int32(), np.int32()),
        (pa.uint64(), np.uint64()),
        (pa.int64(), np.int64()),
        (pa.float32(), np.float32()),
        (pa.float64(), np.float64()),
    ]

    for pa_type, np_type in data_types:
        view = na.array_view(pa.array([0, 1, 2], pa_type))
        np.testing.assert_array_equal(
            np.array(view.buffers[1]), np.array([0, 1, 2], np_type)
        )


def test_buffers_string():
    view = na.array_view(pa.array(["a", "bc", "def"]))

    assert view.buffers[0] is None
    np.testing.assert_array_equal(
        np.array(view.buffers[1]), np.array([0, 1, 3, 6], np.int32())
    )
    np.testing.assert_array_equal(
        np.array(view.buffers[2]), np.array(list("abcdef"), dtype="|S1")
    )


def test_buffers_binary():
    view = na.array_view(pa.array([b"a", b"bc", b"def"]))

    assert view.buffers[0] is None
    np.testing.assert_array_equal(
        np.array(view.buffers[1]), np.array([0, 1, 3, 6], np.int32())
    )
    np.testing.assert_array_equal(np.array(view.buffers[2]), np.array(list(b"abcdef")))


def test_array_stream():
    array_stream = na.ArrayStream.allocate()
    assert na.array_stream(array_stream) is array_stream

    assert array_stream.is_valid() is False
    with pytest.raises(RuntimeError):
        array_stream.get_schema()
    with pytest.raises(RuntimeError):
        array_stream.get_next()

    pa_array_child = pa.array([1, 2, 3], pa.int32())
    pa_array = pa.record_batch([pa_array_child], names=["some_column"])
    reader = pa.RecordBatchReader.from_batches(pa_array.schema, [pa_array])
    array_stream = na.array_stream(reader)

    assert array_stream.is_valid() is True
    array = array_stream.get_next()
    assert array.schema.children[0].name == "some_column"
    with pytest.raises(StopIteration):
        array_stream.get_next()


def test_array_stream_iter():
    pa_array_child = pa.array([1, 2, 3], pa.int32())
    pa_array = pa.record_batch([pa_array_child], names=["some_column"])
    reader = pa.RecordBatchReader.from_batches(pa_array.schema, [pa_array])
    array_stream = na.array_stream(reader)

    arrays = list(array_stream)
    assert len(arrays) == 1
    assert arrays[0].schema.children[0].name == "some_column"
