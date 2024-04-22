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
from nanoarrow.c_lib import CArrayStream

import nanoarrow as na
from nanoarrow import numpy_iterator


def test_numpy_iterator_zero_chunks():
    na_array = na.Array([], na.int32())
    np.testing.assert_array_equal(
        numpy_iterator.to_numpy(na_array), np.array([], np.int32)
    )


def test_numpy_iterator_one_chunk():
    na_array = na.c_array([1, 2, 3], na.int32())

    np.testing.assert_array_equal(
        numpy_iterator.to_numpy(na_array), np.array([1, 2, 3], np.int32)
    )

    # With dtype request
    np.testing.assert_array_equal(
        numpy_iterator.to_numpy(na_array, dtype=np.int64), np.array([1, 2, 3], np.int64)
    )


def test_numpy_iterator_many_chunks():
    src = [
        na.c_array([1, 2, 3], na.int32()),
        na.c_array([4, 5, 6], na.int32()),
        na.c_array([7, 8, 9], na.int32()),
    ]
    na_array = na.Array(CArrayStream.from_array_list(src, na.c_schema(na.int32())))

    # With predetermined length
    np.testing.assert_array_equal(
        numpy_iterator.to_numpy(na_array), np.array(range(1, 10), np.int32)
    )

    # With length not available in advance
    na_stream = CArrayStream.from_array_list(src, na.c_schema(na.int32()))
    np.testing.assert_array_equal(
        numpy_iterator.to_numpy(na_stream), np.array(range(1, 10), np.int32)
    )

    # With dtype request
    np.testing.assert_array_equal(
        numpy_iterator.to_numpy(na_array, dtype=np.int64),
        np.array(range(1, 10), np.int64),
    )


def test_numpy_iterator_dtype_fixed_size_binary():
    na_array = na.Array([b"one", b"two", b"thr"], na.fixed_size_binary(3))

    np.testing.assert_array_equal(
        numpy_iterator.to_numpy(na_array),
        np.array([b"one", b"two", b"thr"], dtype="|S3"),
    )


def test_numpy_iterator_dtype_obj():
    na_array = na.Array(["one", "two", "three"], na.string())

    np.testing.assert_array_equal(
        numpy_iterator.to_numpy(na_array), np.array(["one", "two", "three"])
    )


def test_numpy_iterator_sliced_zero_copy_input():
    na_array = na.c_array([1, 2, 3, 4], na.int32())[-1:]
    assert na_array.offset == 3
    assert na_array.length == 1

    np.testing.assert_array_equal(
        numpy_iterator.to_numpy(na_array), np.array([4], np.int32)
    )
