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
from nanoarrow.iterator import iter_py, iter_tuples

import nanoarrow as na


def test_iterator_primitive():
    array = na.c_array([1, 2, 3], na.int32())
    assert list(iter_py(array)) == [1, 2, 3]

    sliced = array[1:]
    assert list(iter_py(sliced)) == [2, 3]


def test_iterator_nullable_primitive():
    array = na.c_array_from_buffers(
        na.int32(),
        4,
        buffers=[
            na.c_buffer([1, 1, 1, 0], na.bool()),
            na.c_buffer([1, 2, 3, 0], na.int32()),
        ],
    )
    assert list(iter_py(array)) == [1, 2, 3, None]

    sliced = array[1:]
    assert list(iter_py(sliced)) == [2, 3, None]


def test_iterator_string():
    array = na.c_array_from_buffers(
        na.string(), 2, buffers=[None, na.c_buffer([0, 2, 5], na.int32()), b"abcde"]
    )

    assert list(iter_py(array)) == ["ab", "cde"]

    sliced = array[1:]
    assert list(iter_py(sliced)) == ["cde"]


def test_iterator_nullable_string():
    array = na.c_array_from_buffers(
        na.string(),
        3,
        buffers=[
            na.c_buffer([1, 1, 0], na.bool()),
            na.c_buffer([0, 2, 5, 5], na.int32()),
            b"abcde",
        ],
    )

    assert list(iter_py(array)) == ["ab", "cde", None]

    sliced = array[1:]
    assert list(iter_py(sliced)) == ["cde", None]


def test_iterator_binary():
    array = na.c_array_from_buffers(
        na.binary(), 2, buffers=[None, na.c_buffer([0, 2, 5], na.int32()), b"abcde"]
    )

    assert list(iter_py(array)) == [b"ab", b"cde"]

    sliced = array[1:]
    assert list(iter_py(sliced)) == [b"cde"]


def test_iterator_nullable_binary():
    array = na.c_array_from_buffers(
        na.binary(),
        3,
        buffers=[
            na.c_buffer([1, 1, 0], na.bool()),
            na.c_buffer([0, 2, 5, 5], na.int32()),
            b"abcde",
        ],
    )

    assert list(iter_py(array)) == [b"ab", b"cde", None]

    sliced = array[1:]
    assert list(iter_py(sliced)) == [b"cde", None]


def test_iter_tuples():
    array = na.c_array_from_buffers(
        na.struct({"col1": na.int32(), "col2": na.bool()}),
        length=3,
        buffers=[None],
        children=[na.c_array([1, 2, 3], na.int32()), na.c_array([1, 0, 1], na.bool())],
    )

    assert list(iter_tuples(array)) == [(1, True), (2, False), (3, True)]

    sliced = array[1:]
    assert list(iter_tuples(sliced)) == [(2, False), (3, True)]

    sliced_child = na.c_array_from_buffers(
        array.schema,
        length=2,
        buffers=[None],
        children=[array.child(0)[1:], array.child(1)[1:]],
    )
    assert list(iter_tuples(sliced_child)) == [(2, False), (3, True)]

    nested_sliced = sliced_child[1:]
    assert list(iter_tuples(nested_sliced)) == [(3, True)]


def test_iter_tuples_nullable():
    array = na.c_array_from_buffers(
        na.struct({"col1": na.int32(), "col2": na.bool()}),
        length=4,
        buffers=[na.c_buffer([True, True, True, False], na.bool())],
        children=[
            na.c_array([1, 2, 3, 4], na.int32()),
            na.c_array([1, 0, 1, 0], na.bool()),
        ],
    )

    assert list(iter_tuples(array)) == [(1, True), (2, False), (3, True), None]

    sliced = array[1:]
    assert list(iter_tuples(sliced)) == [(2, False), (3, True), None]

    sliced_child = na.c_array_from_buffers(
        array.schema,
        length=3,
        buffers=[na.c_buffer([True, True, False], na.bool())],
        children=[array.child(0)[1:], array.child(1)[1:]],
    )
    assert list(iter_tuples(sliced_child)) == [(2, False), (3, True), None]

    nested_sliced = sliced_child[1:]
    assert list(iter_tuples(nested_sliced)) == [(3, True), None]


def test_iter_tuples_errors():
    with pytest.raises(TypeError, match="can only iterate over struct arrays"):
        list(iter_tuples(na.c_array([1, 2, 3], na.int32())))


def test_iterator_struct():
    array = na.c_array_from_buffers(
        na.struct({"col1": na.int32(), "col2": na.bool()}),
        length=3,
        buffers=[None],
        children=[na.c_array([1, 2, 3], na.int32()), na.c_array([1, 0, 1], na.bool())],
    )

    assert list(iter_py(array)) == [
        {"col1": 1, "col2": True},
        {"col1": 2, "col2": False},
        {"col1": 3, "col2": True},
    ]

    sliced = array[1:]
    assert list(iter_py(sliced)) == [
        {"col1": 2, "col2": False},
        {"col1": 3, "col2": True},
    ]


def test_iterator_nullable_struct():
    array = na.c_array_from_buffers(
        na.struct({"col1": na.int32(), "col2": na.bool()}),
        length=4,
        buffers=[na.c_buffer([True, True, True, False], na.bool())],
        children=[
            na.c_array([1, 2, 3, 4], na.int32()),
            na.c_array([1, 0, 1, 0], na.bool()),
        ],
    )

    assert list(iter_py(array)) == [
        {"col1": 1, "col2": True},
        {"col1": 2, "col2": False},
        {"col1": 3, "col2": True},
        None,
    ]

    sliced = array[1:]
    assert list(iter_py(sliced)) == [
        {"col1": 2, "col2": False},
        {"col1": 3, "col2": True},
        None,
    ]


def test_iterator_list():
    pa = pytest.importorskip("pyarrow")
    items = [[1, 2, 3], [4, 5, 6], [7, 8, None], [0]]
    array = pa.array(items)
    assert list(iter_py(array)) == items

    sliced = array[1:]
    assert list(iter_py(sliced)) == [[4, 5, 6], [7, 8, None], [0]]

    array_sliced_child = pa.ListArray.from_arrays([0, 2, 5, 8, 9], array.values[1:])
    assert (list(iter_py(array_sliced_child))) == [
        [2, 3],
        [4, 5, 6],
        [7, 8, None],
        [0],
    ]

    nested_sliced = array_sliced_child[1:]
    assert (list(iter_py(nested_sliced))) == [
        [4, 5, 6],
        [7, 8, None],
        [0],
    ]


def test_iterator_nullable_list():
    pa = pytest.importorskip("pyarrow")
    items = [[1, 2, 3], [4, 5, 6], [7, 8, None], [0], None]
    array = pa.array(items)
    assert list(iter_py(array)) == items

    sliced = array[1:]
    assert list(iter_py(sliced)) == [[4, 5, 6], [7, 8, None], [0], None]

    array_sliced_child = pa.ListArray.from_arrays(
        [0, 2, 5, 8, 9, 9],
        array.values[1:],
        mask=pa.array([False, False, False, False, True]),
    )
    assert (list(iter_py(array_sliced_child))) == [
        [2, 3],
        [4, 5, 6],
        [7, 8, None],
        [0],
        None,
    ]

    nested_sliced = array_sliced_child[1:]
    assert (list(iter_py(nested_sliced))) == [[4, 5, 6], [7, 8, None], [0], None]


def test_iterator_fixed_size_list():
    pa = pytest.importorskip("pyarrow")
    items = [[1, 2, 3], [4, 5, 6], [7, 8, None]]
    array = pa.array(items, pa.list_(pa.int64(), 3))
    assert list(iter_py(array)) == items

    sliced = array[1:]
    assert list(iter_py(sliced)) == [[4, 5, 6], [7, 8, None]]

    array_sliced_child = pa.FixedSizeListArray.from_arrays(array.values[3:], 3)
    assert (list(iter_py(array_sliced_child))) == [[4, 5, 6], [7, 8, None]]

    nested_sliced = array_sliced_child[1:]
    assert (list(iter_py(nested_sliced))) == [[7, 8, None]]


def test_iterator_nullable_fixed_size_list():
    pa = pytest.importorskip("pyarrow")
    items = [[1, 2, 3], [4, 5, 6], [7, 8, None], None]
    array = pa.array(items, pa.list_(pa.int64(), 3))
    assert list(iter_py(array)) == items

    sliced = array[1:]
    assert list(iter_py(sliced)) == [[4, 5, 6], [7, 8, None], None]

    # mask argument only available for pyarrow >= 15.0.0
    array_sliced_child = pa.FixedSizeListArray.from_arrays(
        array.values[3:], 3, mask=pa.array([False, False, True])
    )
    assert (list(iter_py(array_sliced_child))) == [[4, 5, 6], [7, 8, None], None]

    nested_sliced = array_sliced_child[1:]
    assert (list(iter_py(nested_sliced))) == [[7, 8, None], None]


def test_iterator_dictionary():
    pa = pytest.importorskip("pyarrow")

    items = ["ab", "cde", "ab", "def", "cde"]
    array = pa.array(items).dictionary_encode()

    assert list(iter_py(array)) == items

    sliced = array[1:]
    assert list(iter_py(sliced)) == ["cde", "ab", "def", "cde"]


def test_iterator_nullable_dictionary():
    pa = pytest.importorskip("pyarrow")

    items = ["ab", "cde", "ab", "def", "cde", None]
    array = pa.array(items).dictionary_encode()

    assert list(iter_py(array)) == items

    sliced = array[1:]
    assert list(iter_py(sliced)) == ["cde", "ab", "def", "cde", None]
