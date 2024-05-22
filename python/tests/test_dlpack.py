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

import ctypes
from functools import wraps

import numpy as np
import pytest

import nanoarrow as na

pa = pytest.importorskip("pyarrow")


def PyCapsule_IsValid(capsule, name):
    return ctypes.pythonapi.PyCapsule_IsValid(ctypes.py_object(capsule), name) == 1


def check_dlpack_export(view, expected_arr):
    DLTensor = view.__dlpack__()
    assert PyCapsule_IsValid(DLTensor, b"dltensor") is True

    result = np.from_dlpack(view)
    np.testing.assert_array_equal(result, expected_arr, strict=True)

    assert view.__dlpack_device__() == (1, 0)


def check_bytes_allocated(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        allocated_bytes = pa.total_allocated_bytes()
        try:
            return f(*args, **kwargs)
        finally:
            assert pa.total_allocated_bytes() == allocated_bytes

    return wrapper


@check_bytes_allocated
@pytest.mark.parametrize(
    ("value_type", "np_type"),
    [
        (pa.uint8(), np.uint8),
        (pa.uint16(), np.uint16),
        (pa.uint32(), np.uint32),
        (pa.uint64(), np.uint64),
        (pa.int8(), np.int8),
        (pa.int16(), np.int16),
        (pa.int32(), np.int32),
        (pa.int64(), np.int64),
        (pa.float16(), np.float16),
        (pa.float32(), np.float32),
        (pa.float64(), np.float64),
    ],
)
def test_dlpack(value_type, np_type):
    if np.__version__ < "1.24.0":
        pytest.skip(
            "No dlpack support in numpy versions older than 1.22.0, "
            "strict keyword in assert_array_equal added in numpy version "
            "1.24.0"
        )

    expected = np.array([1, 2, 3], dtype=np_type)
    pa_arr = pa.array(expected, type=value_type)
    # Use the value buffer of the nanoarrow CArray
    view = na.c_array(pa_arr).view().buffer(1)
    check_dlpack_export(view, expected)


def test_dlpack_not_supported():
    # DLPack doesn't support bit-packed boolean values
    view = na.c_array([True, False, True], na.bool_()).view().buffer(1)

    with pytest.raises(
        ValueError, match="Bit-packed boolean data type not supported by DLPack."
    ):
        view.__dlpack__()

    with pytest.raises(
        ValueError, match="Bit-packed boolean data type not supported by DLPack."
    ):
        view.__dlpack_device__()

    with pytest.raises(
        NotImplementedError, match="Only stream=None is supported."
    ):
        view.__dlpack__(stream=3)
