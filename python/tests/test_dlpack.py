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

import pytest
from nanoarrow._buffer import CBuffer
from nanoarrow._utils import obj_is_capsule

import nanoarrow as na

np = pytest.importorskip("numpy")


def check_dlpack_export(view, expected_arr):
    # Check device spec
    assert view.__dlpack_device__() == (1, 0)

    # Check capsule export
    capsule = view.__dlpack__()
    assert obj_is_capsule(capsule, "dltensor") is True

    # Check roundtrip through numpy
    result = np.from_dlpack(view)
    np.testing.assert_array_equal(result, expected_arr, strict=True)

    # Check roundtrip through CBuffer
    buffer = CBuffer.from_dlpack(view)
    np.testing.assert_array_equal(np.array(buffer), expected_arr, strict=True)


@pytest.mark.parametrize(
    ("value_type", "np_type"),
    [
        (na.uint8(), np.uint8),
        (na.uint16(), np.uint16),
        (na.uint32(), np.uint32),
        (na.uint64(), np.uint64),
        (na.int8(), np.int8),
        (na.int16(), np.int16),
        (na.int32(), np.int32),
        (na.int64(), np.int64),
        (na.interval_months(), np.int32),
        (na.float16(), np.float16),
        (na.float32(), np.float32),
        (na.float64(), np.float64),
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
    # Use the value buffer of the nanoarrow CArray
    view = na.c_array([1, 2, 3], value_type).view().buffer(1)
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


def test_dlpack_cuda(cuda_device):
    cp = pytest.importorskip("cupy")
    if not cuda_device:
        pytest.skip("CUDA device not available")

    gpu_array = cp.array([1, 2, 3])
    gpu_buffer = na.c_buffer(gpu_array)
    assert gpu_buffer.device == cuda_device

    gpu_array_roundtrip = cp.from_dlpack(gpu_buffer.view())
    cp.testing.assert_array_equal(gpu_array_roundtrip, gpu_array)
