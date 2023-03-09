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

def test_schema_basic():# Blank invalid schema
    schema = na.CSchema.Empty()
    assert(schema.is_valid() is False)
    assert(repr(schema) == "[invalid: schema is released]")

    pa_schema = pa.schema([pa.field("some_name", pa.int32())])
    pa_schema._export_to_c(schema._addr())

    assert(schema.format == "+s")
    assert(schema.flags == 0)
    assert(len(schema.children), 1)
    assert(schema.children[0].format == "i")
    assert(schema.children[0].name == "some_name")
    assert(repr(schema.children[0]) == "int32")

    with pytest.raises(IndexError):
        schema.children[1]

def test_schema_parse():
    schema = na.CSchema.Empty()
    with pytest.raises(ValueError):
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
