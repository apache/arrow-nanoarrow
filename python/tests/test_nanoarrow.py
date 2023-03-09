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

import numpy as np
import pyarrow as pa

import nanoarrow as na

import pytest

def test_version():
    assert(na.version() == "0.2.0-SNAPSHOT")

def test_as_numpy_array():

    arr = pa.array([1, 2, 3])
    result = na.as_numpy_array(arr)
    expected = arr.to_numpy()
    np.testing.assert_array_equal(result, expected)

    arr = pa.array([1, 2, 3], pa.uint8())
    result = na.as_numpy_array(arr)
    expected = arr.to_numpy()
    np.testing.assert_array_equal(result, expected)

    arr = pa.array([1, 2, None])
    with pytest.raises(ValueError, match="Cannot convert array with nulls"):
        na.as_numpy_array(arr)

    arr = pa.array([[1], [2, 3]])
    with pytest.raises(TypeError, match="Cannot convert a non-primitive array"):
        na.as_numpy_array(arr)

def test_schema_basic():
    # Blank invalid schema
    schema = na.CSchema.Empty()
    assert(schema.is_valid() is False)
    assert(repr(schema) == "[invalid: schema is released]")

    pa_schema = pa.schema([pa.field("some_name", pa.int32())])
    pa_schema._export_to_c(schema._addr())

    assert(schema.format == "+s")
    assert(schema.flags == 0)
    assert(len(schema.children) == 1)
    assert(schema.children[0].format == "i")
    assert(schema.children[0].name == "some_name")
    assert(repr(schema.children[0]) == "int32")

    with pytest.raises(IndexError):
        schema.children[1]

def test_schema_parse():
    schema = na.CSchema.Empty()
    with pytest.raises(RuntimeError):
        schema.parse()

    pa.schema([pa.field("col1", pa.int32())])._export_to_c(schema._addr())

    info = schema.parse()
    assert(info['type'] == 'struct')
    assert(info['storage_type'] == 'struct')
    assert(info['name'] == '')

    # Check on the child
    child = schema.children[0]
    child_info = child.parse()
    assert(child_info['type'] == 'int32')
    assert(child_info['storage_type'] == 'int32')
    assert(child_info['name'] == 'col1')

def test_schema_info_params():
    schema = na.CSchema.Empty()
    pa.binary(12)._export_to_c(schema._addr())
    assert(schema.parse()['fixed_size'] == 12)

    schema = na.CSchema.Empty()
    pa.list_(pa.int32(), 12)._export_to_c(schema._addr())
    assert(schema.parse()['fixed_size'] == 12)

    schema = na.CSchema.Empty()
    pa.decimal128(10, 3)._export_to_c(schema._addr())
    assert(schema.parse()['decimal_bitwidth'] == 128)
    assert(schema.parse()['decimal_precision'] == 10)
    assert(schema.parse()['decimal_scale'] == 3)

def test_array():
    schema = na.CSchema.Empty()
    pa.int32()._export_to_c(schema._addr())

    array = na.CArray.Empty(schema)
    assert(array.is_valid() is False)

    pa.array([1, 2, 3], pa.int32())._export_to_c(array._addr())
    assert(array.is_valid() is True)

    view = array.validate()

    assert(view.array is array)
    assert(view.schema is schema)
    assert(len(view) == 3)

    assert(view.value_int(0) == 1)
    assert(view.value_int(1) == 2)
    assert(view.value_int(2) == 3)

    data_buffer = memoryview(view.buffers[1])
    assert(len(data_buffer) == 12)
    data_buffer_copy = bytes(data_buffer)
    # (needs updating if testing on big endian)
    assert(data_buffer_copy == b'\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00')

def test_array_recursive():
    pa_array = pa.array([1, 2, 3], pa.int32())
    pa_batch = pa.record_batch([pa_array], names=["some_column"])

    schema = na.CSchema.Empty()
    pa_batch.schema._export_to_c(schema._addr())
    assert(len(schema.children) == 1)
    with pytest.raises(IndexError):
        schema.children[1]

    array = na.CArray.Empty(schema)
    assert(array.is_valid() is False)

    pa_batch._export_to_c(array._addr())
    assert(array.is_valid() is True)
    assert(len(array.children) == 1)
    with pytest.raises(IndexError):
        array.children[1]

    view = array.validate()
    assert(len(view.children) == 1)
    with pytest.raises(IndexError):
       view.children[1]

    child = view.children[0]
    assert(len(child) == 3)
    assert(child.value_int(0) == 1)
    assert(child.value_int(1) == 2)
    assert(child.value_int(2) == 3)
