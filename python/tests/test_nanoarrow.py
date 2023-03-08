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

def test_schema():
    pa_schema = pa.schema([pa.field("some_name", pa.int32())])
    na_schema = na.Schema.from_pyarrow(pa_schema)
    assert(na_schema.format == "+s")
    assert(na_schema.flags == 0)
    assert(len(na_schema.children), 1)
    assert(na_schema.children[0].format == "i")
