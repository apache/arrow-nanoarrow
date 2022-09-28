import numpy as np
import pyarrow as pa

import nanoarrow

import pytest


def test_as_numpy_array():
    
    arr = pa.array([1, 2, 3])
    result = nanoarrow.as_numpy_array(arr)
    expected = arr.to_numpy()
    np.testing.assert_array_equal(result, expected)

    arr = pa.array([1, 2, 3], pa.uint8())
    result = nanoarrow.as_numpy_array(arr)
    expected = arr.to_numpy()
    np.testing.assert_array_equal(result, expected)

    arr = pa.array([1, 2, None])
    with pytest.raises(ValueError, match="Cannot convert array with nulls"):
        nanoarrow.as_numpy_array(arr)

    arr = pa.array([[1], [2, 3]])
    with pytest.raises(TypeError, match="Cannot convert a non-primitive array"):
        nanoarrow.as_numpy_array(arr)
