import sys

import numpy as np
import pyarrow as pa

import nanoarrow

import pytest


def test_array_from_pyarrow():
    parr = pa.array([1, 2, 3])
    result = nanoarrow.Array.from_pyarrow(parr)
    assert result.format == "l"


def test_array_to_numpy_lifetime():

    parr = pa.array([1, 2, 3])
    arr = nanoarrow.Array.from_pyarrow(parr)
    refcount = sys.getrefcount(arr)
    result = arr.to_numpy()
    assert sys.getrefcount(arr) > refcount
    assert result.base is arr
    del arr
    result
    assert result.base


def test_array_to_numpy():
    parr = pa.array([1, 2, 3])
    arr = nanoarrow.Array.from_pyarrow(parr)
    result = arr.to_numpy()
    expected = parr.to_numpy()
    np.testing.assert_array_equal(result, expected)

    parr = pa.array([1, 2, 3], pa.uint8())
    arr = nanoarrow.Array.from_pyarrow(parr)
    result = arr.to_numpy()
    expected = parr.to_numpy()
    np.testing.assert_array_equal(result, expected)

    arr = nanoarrow.Array.from_pyarrow(pa.array([1, 2, None]))
    with pytest.raises(ValueError, match="Cannot convert array with nulls"):
        arr.to_numpy()

    arr = nanoarrow.Array.from_pyarrow(pa.array([[1], [2, 3]]))
    with pytest.raises(TypeError, match="Cannot convert a non-primitive array"):
       arr.to_numpy()
