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
from nanoarrow.ipc import Stream
from nanoarrow.iterator import iteritems, itertuples

import nanoarrow as na


def test_iteritems_stream():
    assert list(iteritems(Stream.example())) == [
        {"some_col": 1},
        {"some_col": 2},
        {"some_col": 3},
    ]


def test_itertuples_stream():
    assert list(itertuples(Stream.example())) == [(1,), (2,), (3,)]


def test_iteritems_primitive():
    array = na.c_array([1, 2, 3], na.int32())
    assert list(iteritems(array)) == [1, 2, 3]


def test_iteritems_nullable_primitive():
    array = na.c_array_from_buffers(
        na.int32(),
        4,
        buffers=[
            na.c_buffer([1, 1, 1, 0], na.bool()),
            na.c_buffer([1, 2, 3, 0], na.int32()),
        ],
    )
    assert list(iteritems(array)) == [1, 2, 3, None]


def test_iteritems_string():
    array = na.c_array_from_buffers(
        na.string(), 2, buffers=[None, na.c_buffer([0, 2, 5], na.int32()), b"abcde"]
    )

    assert list(iteritems(array)) == ["ab", "cde"]


def test_iteritems_nullable_string():
    array = na.c_array_from_buffers(
        na.string(),
        3,
        buffers=[
            na.c_buffer([1, 1, 0], na.bool()),
            na.c_buffer([0, 2, 5, 5], na.int32()),
            b"abcde",
        ],
    )

    assert list(iteritems(array)) == ["ab", "cde", None]


def test_iteritems_binary():
    array = na.c_array_from_buffers(
        na.binary(), 2, buffers=[None, na.c_buffer([0, 2, 5], na.int32()), b"abcde"]
    )

    assert list(iteritems(array)) == [b"ab", b"cde"]


def test_iteritems_nullable_binary():
    array = na.c_array_from_buffers(
        na.binary(),
        3,
        buffers=[
            na.c_buffer([1, 1, 0], na.bool()),
            na.c_buffer([0, 2, 5, 5], na.int32()),
            b"abcde",
        ],
    )

    assert list(iteritems(array)) == [b"ab", b"cde", None]


def test_itertuples():
    array = na.c_array_from_buffers(
        na.struct({"col1": na.int32(), "col2": na.bool()}),
        length=3,
        buffers=[None],
        children=[na.c_array([1, 2, 3], na.int32()), na.c_array([1, 0, 1], na.bool())],
    )

    assert list(itertuples(array)) == [(1, True), (2, False), (3, True)]


def test_itertuples_nullable():
    array = na.c_array_from_buffers(
        na.struct({"col1": na.int32(), "col2": na.bool()}),
        length=4,
        buffers=[na.c_buffer([True, True, True, False], na.bool())],
        children=[
            na.c_array([1, 2, 3, 4], na.int32()),
            na.c_array([1, 0, 1, 0], na.bool()),
        ],
    )

    assert list(itertuples(array)) == [(1, True), (2, False), (3, True), None]


def test_itertuples_errors():
    with pytest.raises(TypeError, match="can only iterate over struct arrays"):
        list(itertuples(na.c_array([1, 2, 3], na.int32())))


def test_iteritems_struct():
    array = na.c_array_from_buffers(
        na.struct({"col1": na.int32(), "col2": na.bool()}),
        length=3,
        buffers=[None],
        children=[na.c_array([1, 2, 3], na.int32()), na.c_array([1, 0, 1], na.bool())],
    )

    assert list(iteritems(array)) == [
        {"col1": 1, "col2": True},
        {"col1": 2, "col2": False},
        {"col1": 3, "col2": True},
    ]


def test_iteritems_nullable_struct():
    array = na.c_array_from_buffers(
        na.struct({"col1": na.int32(), "col2": na.bool()}),
        length=4,
        buffers=[na.c_buffer([True, True, True, False], na.bool())],
        children=[
            na.c_array([1, 2, 3, 4], na.int32()),
            na.c_array([1, 0, 1, 0], na.bool()),
        ],
    )

    assert list(iteritems(array)) == [
        {"col1": 1, "col2": True},
        {"col1": 2, "col2": False},
        {"col1": 3, "col2": True},
        None,
    ]


def test_iteritems_list():
    pa = pytest.importorskip("pyarrow")
    items = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0]]
    array = pa.array(items)
    assert list(iteritems(array)) == items


def test_iteritems_nullable_list():
    pa = pytest.importorskip("pyarrow")
    items = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [0], None]
    array = pa.array(items)
    assert list(iteritems(array)) == items


def test_iteritems_fixed_size_list():
    pa = pytest.importorskip("pyarrow")
    items = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    array = pa.array(items, pa.list_(pa.int64(), 3))
    assert list(iteritems(array)) == items


def test_iteritems_nullable_fixed_size_list():
    pa = pytest.importorskip("pyarrow")
    items = [[1, 2, 3], [4, 5, 6], [7, 8, 9], None]
    array = pa.array(items, pa.list_(pa.int64(), 3))
    assert list(iteritems(array)) == items
