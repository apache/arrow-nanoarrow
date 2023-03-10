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
import pyarrow as pa
import pytest

import nanoarrow as na

def test_version():
    re_version = re.compile(r'^[0-9]+\.[0-9]+\.[0-9]+(-SNAPSHOT)?$')
    assert re_version.match(na.version()) is not None

def test_schema_basic():
    schema = na.Schema.Empty()
    assert schema.is_valid() is False
    assert repr(schema) == "[invalid: schema is released]"

    pa_schema = pa.schema([pa.field("some_name", pa.int32())])
    pa_schema._export_to_c(schema._addr())

    assert schema.format == "+s"
    assert schema.flags == 0
    assert schema.metadata is None
    assert len(schema.children) == 1
    assert schema.children[0].format == "i"
    assert schema.children[0].name == "some_name"
    assert repr(schema.children[0]) == "int32"
    assert schema.dictionary is None

    with pytest.raises(IndexError):
        schema.children[1]

def test_schema_dictionary():
    schema = na.Schema.Empty()
    pa.dictionary(pa.int32(), pa.utf8())._export_to_c(schema._addr())
    assert schema.format == 'i'
    assert schema.dictionary.format == 'u'

def test_schema_metadata():
    schema = na.Schema.Empty()
    meta = {'key1': 'value1', 'key2': 'value2'}
    pa.field('', pa.int32(), metadata=meta)._export_to_c(schema._addr())

    assert len(schema.metadata) == 2

    meta2 = {k: v for k, v in schema.metadata}
    assert list(meta2.keys()) == ['key1', 'key2']
    assert list(meta2.values()) == [b'value1', b'value2']

def test_schema_view():
    schema = na.Schema.Empty()
    with pytest.raises(RuntimeError):
        schema.view()

    pa.int32()._export_to_c(schema._addr())
    view = schema.view()
    assert view.type == 'int32'
    assert view.storage_type == 'int32'

    assert view.fixed_size is None
    assert view.decimal_bitwidth is None
    assert view.decimal_scale is None
    assert view.time_unit is None
    assert view.timezone is None
    assert view.union_type_ids is None
    assert view.extension_name is None
    assert view.extension_metadata is None

def test_schema_view_extra_params():
    schema = na.Schema.Empty()
    pa.binary(12)._export_to_c(schema._addr())
    view = schema.view()
    assert view.fixed_size == 12

    schema = na.Schema.Empty()
    pa.list_(pa.int32(), 12)._export_to_c(schema._addr())
    assert view.fixed_size == 12

    schema = na.Schema.Empty()
    pa.decimal128(10, 3)._export_to_c(schema._addr())
    view = schema.view()
    assert view.decimal_bitwidth == 128
    assert view.decimal_precision == 10
    assert view.decimal_scale == 3

    schema = na.Schema.Empty()
    pa.decimal256(10, 3)._export_to_c(schema._addr())
    view = schema.view()
    assert view.decimal_bitwidth == 256
    assert view.decimal_precision == 10
    assert view.decimal_scale == 3

    schema = na.Schema.Empty()
    pa.duration('us')._export_to_c(schema._addr())
    view = schema.view()
    assert view.time_unit == 'us'

    schema = na.Schema.Empty()
    pa.timestamp('us', tz='America/Halifax')._export_to_c(schema._addr())
    view = schema.view()
    assert view.type == 'timestamp'
    assert view.storage_type == 'int64'
    assert view.time_unit == 'us'
    assert view.timezone == 'America/Halifax'

    schema = na.Schema.Empty()
    meta = {
        'ARROW:extension:name': 'some_name',
        'ARROW:extension:metadata': 'some_metadata'
    }
    pa.field('', pa.int32(), metadata=meta)._export_to_c(schema._addr())
    view = schema.view()
    assert view.extension_name == 'some_name'
    assert view.extension_metadata == b'some_metadata'

def test_array():
    schema = na.Schema.Empty()
    pa.int32()._export_to_c(schema._addr())

    array = na.Array.Empty(schema)
    assert array.is_valid() is False

    pa.array([1, 2, 3], pa.int32())._export_to_c(array._addr())
    assert array.is_valid() is True
    assert array.length == 3
    assert array.offset == 0
    assert array.null_count == 0
    assert len(array.buffers) == 2
    assert array.buffers[0] == 0
    assert len(array.children) == 0
    assert array.dictionary is None

    with pytest.raises(IndexError):
        array.children[1]

def test_array_view():
    array = na.Array.Empty(na.Schema.Empty())
    pa.array([1, 2, 3], pa.int32())._export_to_c(array._addr(), array.schema._addr())
    view = array.view()

    assert view.array is array
    assert view.schema is array.schema

    data_buffer = memoryview(view.buffers[1])
    assert len(data_buffer) == 12
    data_buffer_copy = bytes(data_buffer)

    if sys.byteorder == 'little':
        assert data_buffer_copy == b'\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00'
    else:
        assert data_buffer_copy == b'\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03'

    with pytest.raises(IndexError):
        view.children[1]

def test_array_view_recursive():
    pa_array_child = pa.array([1, 2, 3], pa.int32())
    pa_array = pa.record_batch([pa_array_child], names=["some_column"])

    array = na.Array.Empty(na.Schema.Empty())
    pa_array._export_to_c(array._addr(), array.schema._addr())

    assert array.schema.format == '+s'
    assert array.length == 3
    assert len(array.children) == 1

    assert array.children[0].schema.format == 'i'
    assert array.children[0].length == 3
    assert array.children[0].schema._addr() == array.schema.children[0]._addr()

    view = array.view()
    assert len(view.buffers) == 1
    assert len(view.children) == 1
    assert view.array._addr() == array._addr()
    assert view.schema._addr() == array.schema._addr()

    assert len(view.children[0].buffers) == 2
    assert view.children[0].array._addr() == array.children[0]._addr()
    assert view.children[0].schema._addr() == array.schema.children[0]._addr()
    assert view.children[0].schema._addr() == array.children[0].schema._addr()
