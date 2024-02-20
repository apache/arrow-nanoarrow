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

import nanoarrow as na
from nanoarrow.iterator import storage


def test_storage_primitive():
    array = na.c_array([1, 2, 3], na.int32())
    assert list(storage(array)) == [1, 2, 3]


def test_storage_nullable_primitive():
    array = na.c_array_from_buffers(
        na.int32(),
        4,
        buffers=[
            na.c_buffer([1, 1, 1, 0], na.bool()),
            na.c_buffer([1, 2, 3, 0], na.int32()),
        ],
    )
    assert list(storage(array)) == [1, 2, 3, None]


def test_storage_string():
    array = na.c_array_from_buffers(
        na.string(), 2, buffers=[None, na.c_buffer([0, 2, 5], na.int32()), b"abcde"]
    )

    assert list(storage(array)) == ["ab", "cde"]


def test_storage_nullable_string():
    array = na.c_array_from_buffers(
        na.string(),
        3,
        buffers=[
            na.c_buffer([1, 1, 0], na.bool()),
            na.c_buffer([0, 2, 5, 5], na.int32()),
            b"abcde",
        ],
    )

    assert list(storage(array)) == ["ab", "cde", None]


def test_storage_binary():
    array = na.c_array_from_buffers(
        na.binary(), 2, buffers=[None, na.c_buffer([0, 2, 5], na.int32()), b"abcde"]
    )

    assert list(storage(array)) == [b"ab", b"cde"]


def test_storage_nullable_binary():
    array = na.c_array_from_buffers(
        na.binary(),
        3,
        buffers=[
            na.c_buffer([1, 1, 0], na.bool()),
            na.c_buffer([0, 2, 5, 5], na.int32()),
            b"abcde",
        ],
    )

    assert list(storage(array)) == [b"ab", b"cde", None]
