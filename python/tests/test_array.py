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
from nanoarrow.c_lib import CArrayStream

import nanoarrow as na


def test_array_empty():
    array = na.array([], na.int32())
    assert array.schema.type == na.Type.INT32
    assert len(array) == 0
    assert array.n_chunks == 0
    assert list(array.chunks) == []
    with pytest.raises(IndexError):
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
    array = na.array([1, 2, 3], na.int32())
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

    array = na.array(CArrayStream.from_array_list(src, na.c_schema(na.int32())))
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


def test_scalar_to_array():
    array = na.array([123456, 7890], na.int32())
    scalar = scalar = array[1]
    as_array = na.c_array(scalar)
    assert as_array.offset == 1
    assert as_array.length == 1
    assert as_array.buffers == na.c_array(array).buffers

    with pytest.raises(NotImplementedError):
        na.c_array(scalar, na.string())


def test_scalar_repr():
    # Check a scalar repr that does not need truncation
    scalar = na.array([123456], na.int32())[0]
    assert repr(scalar) == "Scalar<int32> 123456"

    # Check a long Scalar repr that needs truncation
    c_array = na.c_array_from_buffers(
        na.struct({f"col{i}": na.int32() for i in range(100)}),
        length=1,
        buffers=[None],
        children=[na.c_array([123456], na.int32())] * 100,
    )
    scalar = na.array(c_array)[0]
    assert repr(scalar) == (
        "Scalar<struct<col0: int3...> {'col0': 123456, "
        "'col1': 123456, 'col2': 123456,..."
    )
    assert len(repr(scalar)) == 80


def test_scalar_repr_long():
    pa = pytest.importorskip("pyarrow")
    scalar = na.array(pa.array(["abcdefg" * 10]))[0]
    assert repr(scalar).endswith("...")
    assert len(repr(scalar)) == 80


def test_array_repr():
    array = na.array(range(10), na.int32())
    one_to_ten = "\n".join(str(i) for i in range(10))

    assert repr(array) == f"nanoarrow.Array<int32>[10]\n{one_to_ten}"

    array = na.array(range(11), na.int32())
    assert (
        repr(array) == f"nanoarrow.Array<int32>[11]\n{one_to_ten}\n...and 1 more item"
    )

    array = na.array(range(12), na.int32())
    assert (
        repr(array) == f"nanoarrow.Array<int32>[12]\n{one_to_ten}\n...and 2 more items"
    )


def test_wide_array_repr():
    c_array = na.c_array_from_buffers(
        na.struct({f"col{i}": na.int32() for i in range(100)}),
        length=1,
        buffers=[None],
        children=[na.c_array([123456], na.int32())] * 100,
    )
    array = na.array(c_array)

    repr_lines = repr(array).splitlines()

    # Check abbreviated schema
    assert repr_lines[0] == (
        "nanoarrow.Array<struct<col0: int32, col1: int32, col2"
        ": int32, col3: int32...>[1]"
    )
    assert len(repr_lines[0]) == 80

    # Check an abbreviated value
    assert len(repr_lines[1]) == 80


def test_array_repr_long():
    pa = pytest.importorskip("pyarrow")

    # Check that exact length is not truncated with a ...
    array = na.array(pa.array(["a" * 78]))
    repr_lines = repr(array).splitlines()
    assert len(repr_lines) == 2
    assert not repr_lines[1].endswith("...")
    assert len(repr_lines[1]) == 80

    # Check that wide output is truncated with a ...
    array = na.array(pa.array(["a" * 79]))
    repr_lines = repr(array).splitlines()
    assert len(repr_lines) == 2
    assert repr_lines[1].endswith("...")
    assert len(repr_lines[1]) == 80
