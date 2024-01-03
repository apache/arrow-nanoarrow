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

import sys
import re

import pytest

import nanoarrow as na

np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")

def test_cversion():
    re_version = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(-SNAPSHOT)?$")
    assert re_version.match(na.cversion()) is not None


def test_cschema_helper():
    schema = na.cschema()
    assert na.cschema(schema) is schema

    schema = na.cschema(pa.null())
    assert isinstance(schema, na.clib.CSchema)

    with pytest.raises(TypeError):
        na.cschema(1234)


def test_carray_helper():
    array = na.carray()
    assert na.carray(array) is array

    array = na.carray(pa.array([], pa.null()))
    assert isinstance(array, na.clib.CArray)

    with pytest.raises(TypeError):
        na.carray(1234)


def test_array_stream_helper():
    array_stream = na.carray_stream()
    assert na.carray_stream(array_stream) is array_stream

    with pytest.raises(TypeError):
        na.carray_stream(1234)


def test_array_view_helper():
    array = na.carray(pa.array([1, 2, 3]))
    view = na.carray_view(array)
    assert isinstance(view, na.clib.CArrayView)
    assert na.carray_view(view) is view


def test_cschema_basic():
    schema = na.cschema()
    assert schema.is_valid() is False
    assert schema._to_string() == "[invalid: schema is released]"
    assert repr(schema) == "<released nanoarrow.clib.CSchema>"

    schema = na.cschema(pa.schema([pa.field("some_name", pa.int32())]))

    assert schema.format == "+s"
    assert schema.flags == 0
    assert schema.metadata is None
    assert schema.n_children == 1
    assert len(list(schema.children)) == 1
    assert schema.child(0).format == "i"
    assert schema.child(0).name == "some_name"
    assert schema.child(0)._to_string() == "int32"
    assert "<nanoarrow.clib.CSchema int32>" in repr(schema)
    assert schema.dictionary is None

    with pytest.raises(IndexError):
        schema.child(1)


def test_cschema_dictionary():
    schema = na.cschema(pa.dictionary(pa.int32(), pa.utf8()))
    assert schema.format == "i"
    assert schema.dictionary.format == "u"
    assert "dictionary: <nanoarrow.clib.CSchema string" in repr(schema)


def test_schema_metadata():
    meta = {"key1": "value1", "key2": "value2"}
    schema = na.cschema(pa.field("", pa.int32(), metadata=meta))

    assert len(schema.metadata) == 2

    meta2 = {k: v for k, v in schema.metadata}
    assert list(meta2.keys()) == ["key1", "key2"]
    assert list(meta2.values()) == [b"value1", b"value2"]
    assert "'key1': b'value1'" in repr(schema)


def test_cschema_view():
    schema = na.cschema()
    with pytest.raises(RuntimeError):
        na.cschema_view(schema)

    schema = na.cschema(pa.int32())
    view = na.cschema_view(schema)
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


def test_cschema_view_extra_params():
    schema = na.cschema(pa.binary(12))
    view = na.cschema_view(schema)
    assert view.fixed_size == 12

    schema = na.cschema(pa.list_(pa.int32(), 12))
    assert view.fixed_size == 12

    schema = na.cschema(pa.decimal128(10, 3))
    view = na.cschema_view(schema)
    assert view.decimal_bitwidth == 128
    assert view.decimal_precision == 10
    assert view.decimal_scale == 3

    schema = na.cschema(pa.decimal256(10, 3))
    view = na.cschema_view(schema)
    assert view.decimal_bitwidth == 256
    assert view.decimal_precision == 10
    assert view.decimal_scale == 3

    schema = na.cschema(pa.duration("us"))
    view = na.cschema_view(schema)
    assert view.time_unit == "us"

    schema = na.cschema(pa.timestamp("us", tz="America/Halifax"))
    view = na.cschema_view(schema)
    assert view.type == "timestamp"
    assert view.storage_type == "int64"
    assert view.time_unit == "us"
    assert view.timezone == "America/Halifax"

    meta = {
        "ARROW:extension:name": "some_name",
        "ARROW:extension:metadata": "some_metadata",
    }
    schema = na.cschema(pa.field("", pa.int32(), metadata=meta))
    view = na.cschema_view(schema)
    assert view.extension_name == "some_name"
    assert view.extension_metadata == b"some_metadata"


def test_carray_empty():
    array = na.carray()
    assert array.is_valid() is False
    assert repr(array) == "<released nanoarrow.clib.CArray>"


def test_carray():
    array = na.carray(pa.array([1, 2, 3], pa.int32()))
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
    assert "<nanoarrow.clib.CArray int32" in repr(array)


def test_carray_recursive():
    array = na.carray(pa.record_batch([pa.array([1, 2, 3], pa.int32())], ["col"]))
    assert array.n_children == 1
    assert len(list(array.children)) == 1
    assert array.child(0).length == 3
    assert array.child(0).schema._to_string() == "int32"
    assert "'col': <nanoarrow.clib.CArray int32" in repr(array)

    with pytest.raises(IndexError):
        array.child(-1)


def test_carray_dictionary():
    array = na.carray(pa.array(["a", "b", "b"]).dictionary_encode())
    assert array.length == 3
    assert array.dictionary.length == 2
    assert "dictionary: <nanoarrow.clib.CArray string>" in repr(array)


def test_carray_view():
    array = na.carray(pa.array([1, 2, 3], pa.int32()))
    view = na.carray_view(array)

    assert view.storage_type == "int32"

    data_buffer = memoryview(view.buffer(1))
    data_buffer_copy = bytes(data_buffer)
    assert len(data_buffer_copy) == 12

    if sys.byteorder == "little":
        assert data_buffer_copy == b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00"
    else:
        assert data_buffer_copy == b"\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03"

    with pytest.raises(IndexError):
        view.child(1)


def test_carray_view_recursive():
    pa_array_child = pa.array([1, 2, 3], pa.int32())
    pa_array = pa.record_batch([pa_array_child], names=["some_column"])

    array = na.carray(pa_array)

    assert array.schema.format == "+s"
    assert array.length == 3
    assert array.n_children == 1
    assert len(list(array.children)) == 1

    assert array.child(0).schema.format == "i"
    assert array.child(0).length == 3
    assert array.child(0).schema._addr() == array.schema.child(0)._addr()

    view = na.carray_view(array)
    assert view.n_buffers == 1
    assert len(list(view.buffers)) == 1
    assert view.n_children == 1
    assert len(list(view.children)) == 1

    assert view.child(0).n_buffers == 2
    assert len(list(view.child(0).buffers)) == 2


def test_carray_view_dictionary():
    pa_array = pa.array(["a", "b", "b"], pa.dictionary(pa.int32(), pa.utf8()))
    array = na.carray(pa_array)

    assert array.schema.format == "i"
    assert array.dictionary.schema.format == "u"

    view = na.carray_view(array)
    assert view.n_buffers == 2
    assert view.dictionary.n_buffers == 3


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
        view = na.carray_view(pa.array([0, 1, 2], pa_type))
        # Check via buffer interface
        np.testing.assert_array_equal(
            np.array(view.buffer(1)), np.array([0, 1, 2], np_type)
        )
        # Check via iterator interface
        np.testing.assert_array_equal(
            np.array(list(view.buffer(1))), np.array([0, 1, 2], np_type)
        )


def test_buffers_string():
    view = na.carray_view(pa.array(["a", "bc", "def"]))

    assert view.buffer(0).size_bytes == 0
    np.testing.assert_array_equal(
        np.array(view.buffer(1)), np.array([0, 1, 3, 6], np.int32())
    )
    np.testing.assert_array_equal(
        np.array(view.buffer(2)), np.array(list("abcdef"), dtype="|S1")
    )


def test_buffers_binary():
    view = na.carray_view(pa.array([b"a", b"bc", b"def"]))

    assert view.buffer(0).size_bytes == 0
    np.testing.assert_array_equal(
        np.array(view.buffer(1)), np.array([0, 1, 3, 6], np.int32())
    )
    np.testing.assert_array_equal(np.array(view.buffer(2)), np.array(list(b"abcdef")))
    np.testing.assert_array_equal(
        np.array(list(view.buffer(2))), np.array(list(b"abcdef"))
    )


def test_carray_stream():
    array_stream = na.carray_stream()
    assert na.carray_stream(array_stream) is array_stream
    assert repr(array_stream) == "<released nanoarrow.clib.CArrayStream>"

    assert array_stream.is_valid() is False
    with pytest.raises(RuntimeError):
        array_stream.get_schema()
    with pytest.raises(RuntimeError):
        array_stream.get_next()

    pa_array_child = pa.array([1, 2, 3], pa.int32())
    pa_array = pa.record_batch([pa_array_child], names=["some_column"])
    reader = pa.RecordBatchReader.from_batches(pa_array.schema, [pa_array])
    array_stream = na.carray_stream(reader)

    assert array_stream.is_valid() is True
    assert "<nanoarrow.clib.CSchema struct>" in repr(array_stream)

    array = array_stream.get_next()
    assert array.schema.child(0).name == "some_column"
    with pytest.raises(StopIteration):
        array_stream.get_next()


def test_carray_stream_iter():
    pa_array_child = pa.array([1, 2, 3], pa.int32())
    pa_array = pa.record_batch([pa_array_child], names=["some_column"])
    reader = pa.RecordBatchReader.from_batches(pa_array.schema, [pa_array])
    array_stream = na.carray_stream(reader)

    arrays = list(array_stream)
    assert len(arrays) == 1
    assert arrays[0].schema.child(0).name == "some_column"
