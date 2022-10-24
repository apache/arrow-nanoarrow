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


def test_from_external_pointers():
    pytest.importorskip("pyarrow.cffi")

    from pyarrow.cffi import ffi

    c_schema = ffi.new("struct ArrowSchema*")
    ptr_schema = int(ffi.cast("uintptr_t", c_schema))
    c_array = ffi.new("struct ArrowArray*")
    ptr_array = int(ffi.cast("uintptr_t", c_array))

    typ = pa.int32()
    parr = pa.array([1, 2, 3], type=typ)
    parr._export_to_c(ptr_array, ptr_schema)

    arr = nanoarrow.Array.from_pointers(ptr_array, ptr_schema)
    assert arr.to_numpy().tolist() == [1, 2, 3]

    # trying to import second time should not cause a segfault? To enable
    # this we should copy the schema struct and set release to NULL?
    # arr = nanoarrow.Array.from_pointers(ptr_array, ptr_schema)
