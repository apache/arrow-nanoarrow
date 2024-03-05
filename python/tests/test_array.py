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
from nanoarrow.array import Array
from nanoarrow.c_lib import CArrayStream

import nanoarrow as na


def test_array_empty():
    array = Array([], na.int32())
    assert array.schema.type == na.Type.INT32
    assert len(array) == 0
    assert array.n_chunks == 0
    assert list(array.chunks) == []
    with pytest.raises(IndexError, match="Index 0 out of range"):
        array.chunk(0)

    with na.c_array_stream(array) as stream:
        arrays = list(stream)
        assert len(arrays) == 0

    assert list(array) == []
    with pytest.raises(IndexError):
        array[0]

    c_array = na.c_array(array)
    assert c_array.length == 0
    assert c_array.schema.format == "i"


def test_array_contiguous():
    array = Array([1, 2, 3], na.int32())
    assert array.schema.type == na.Type.INT32
    assert len(array) == 3
    assert array.n_chunks == 1
    assert len(list(array.chunks)) == 1
    assert len(array.chunk(0)) == 3

    with na.c_array_stream(array) as stream:
        arrays = list(stream)
        assert len(arrays) == 1

    for py_item, item in zip([1, 2, 3], array):
        assert item.as_py() == py_item

    for py_item, i in zip([1, 2, 3], range(len(array))):
        assert array[i].as_py() == py_item

    c_array = na.c_array(array)
    assert c_array.length == 3
    assert c_array.schema.format == "i"


def test_array_chunked():
    src = [na.c_array([1, 2, 3], na.int32()), na.c_array([4, 5, 6], na.int32())]

    array = Array(CArrayStream.from_array_list(src, na.c_schema(na.int32())))
    assert array.schema.type == na.Type.INT32
    assert len(array) == 6

    with na.c_array_stream(array) as stream:
        arrays = list(stream)
        assert len(arrays) == 2

    for py_item, item in zip([1, 2, 3, 4, 5, 6], array):
        assert item.as_py() == py_item

    for py_item, i in zip([1, 2, 3, 4, 5, 6], range(len(array))):
        assert array[i].as_py() == py_item

    msg = "Can't export Array with 2 chunks to ArrowArray"
    with pytest.raises(ValueError, match=msg):
        na.c_array(array)
