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


def test_array_construct():
    array = na.Array([], na.int32())
    assert array.schema.type == na.Type.INT32

    array2 = na.Array(array)
    assert array2._data is array._data

    array2 = na.Array(array._data)
    assert array2._data is array._data

    with pytest.raises(TypeError, match="device must be Device"):
        na.Array([], na.int32(), device=1234)

    with pytest.raises(NotImplementedError):
        iter(array)


def test_array_empty():
    array = na.Array([], na.int32())
    assert array.schema.type == na.Type.INT32
    assert len(array) == 0

    assert array.n_buffers == 2
    assert list(array.buffer(0)) == []
    assert list(array.buffer(1)) == []
    assert list(array.iter_buffers()) == []

    assert array.n_children == 0

    assert array.n_chunks == 0
    assert list(array.iter_chunks()) == []
    with pytest.raises(IndexError):
        array.chunk(0)

    assert list(array.iter_py()) == []
    assert list(array.iter_scalar()) == []
    with pytest.raises(IndexError):
        array[0]

    with na.c_array_stream(array) as stream:
        arrays = list(stream)
        assert len(arrays) == 0

    c_array = na.c_array(array)
    assert c_array.length == 0
    assert c_array.schema.format == "i"


def test_array_contiguous():
    array = na.Array([1, 2, 3], na.int32())
    assert array.schema.type == na.Type.INT32
    assert len(array) == 3

    assert array.n_buffers == 2

    validity, data = array.buffers
    assert list(validity) == []
    assert list(data) == [1, 2, 3]
    assert array.buffer(0) is validity
    assert array.buffer(1) is data

    chunk_buffers = list(array.iter_buffers())
    assert len(chunk_buffers) == array.n_chunks
    assert len(chunk_buffers[0]) == array.n_buffers
    assert list(chunk_buffers[0][1]) == [1, 2, 3]

    assert array.n_children == 0
    assert list(array.iter_children()) == []

    assert array.n_chunks == 1
    assert len(list(array.iter_chunks())) == 1
    assert len(array.chunk(0)) == 3

    # Scalars by iterator
    for py_item, item in zip([1, 2, 3], array.iter_scalar()):
        assert item.as_py() == py_item

    # Scalars by __getitem__
    for py_item, i in zip([1, 2, 3], range(len(array))):
        assert array[i].as_py() == py_item

    # Python objects by iter_py()
    for py_item, item in zip([1, 2, 3], array.iter_py()):
        assert item == py_item

    with na.c_array_stream(array) as stream:
        arrays = list(stream)
        assert len(arrays) == 1

    c_array = na.c_array(array)
    assert c_array.length == 3
    assert c_array.schema.format == "i"


def test_array_chunked():
    src = [na.c_array([1, 2, 3], na.int32()), na.c_array([4, 5, 6], na.int32())]

    array = na.Array(CArrayStream.from_array_list(src, na.c_schema(na.int32())))
    assert array.schema.type == na.Type.INT32
    assert len(array) == 6

    assert array.n_buffers == 2
    with pytest.raises(ValueError, match="Can't export ArrowArray"):
        array.buffers

    chunk_buffers = list(array.iter_buffers())
    assert len(chunk_buffers) == array.n_chunks
    assert len(chunk_buffers[0]) == array.n_buffers
    assert list(chunk_buffers[0][1]) == [1, 2, 3]
    assert list(chunk_buffers[1][1]) == [4, 5, 6]

    assert array.n_children == 0
    assert list(array.iter_children()) == []

    assert array.n_children == 0
    assert list(array.iter_children()) == []

    assert array.n_chunks == 2
    assert len(list(array.iter_chunks())) == 2
    assert len(array.chunk(0)) == 3

    for py_item, item in zip([1, 2, 3, 4, 5, 6], array.iter_scalar()):
        assert item.as_py() == py_item

    for py_item, i in zip([1, 2, 3, 4, 5, 6], range(len(array))):
        assert array[i].as_py() == py_item

    # Python objects by iter_py()
    for py_item, item in zip([1, 2, 3], array.iter_py()):
        assert item == py_item

    with na.c_array_stream(array) as stream:
        arrays = list(stream)
        assert len(arrays) == 2

    msg = "Can't export ArrowArray"
    with pytest.raises(ValueError, match=msg):
        na.c_array(array)


def test_array_children():
    c_array = na.c_array_from_buffers(
        na.struct({f"col{i}": na.int32() for i in range(100)}),
        length=1,
        buffers=[None],
        children=[na.c_array([123456], na.int32())] * 100,
    )
    src = [c_array, c_array]
    array = na.Array(CArrayStream.from_array_list(src, c_array.schema))

    assert array.n_children == 100
    assert array.child(0).schema.type == na.Type.INT32
    assert array.child(0).n_chunks == 2
    assert list(array.child(0).iter_py()) == [123456, 123456]

    children = list(array.iter_children())
    assert len(children) == array.n_children

    tuples = list(array.iter_tuples())
    assert len(tuples) == 2
    assert len(tuples[0]) == 100


def test_scalar_to_array():
    array = na.Array([123456, 7890], na.int32())
    scalar = scalar = array[1]
    assert scalar.schema is array.schema
    assert scalar.device is array.device
    as_array = na.c_array(scalar)
    assert as_array.offset == 1
    assert as_array.length == 1
    assert as_array.buffers == na.c_array(array).buffers

    with pytest.raises(NotImplementedError):
        na.c_array(scalar, na.string())


def test_scalar_repr():
    # Check a scalar repr that does not need truncation
    scalar = na.Array([123456], na.int32())[0]
    assert repr(scalar) == "Scalar<int32> 123456"

    # Check a long Scalar repr that needs truncation
    c_array = na.c_array_from_buffers(
        na.struct({f"col{i}": na.int32() for i in range(100)}),
        length=1,
        buffers=[None],
        children=[na.c_array([123456], na.int32())] * 100,
    )
    scalar = na.Array(c_array)[0]
    assert repr(scalar) == (
        "Scalar<struct<col0: int3...> {'col0': 123456, "
        "'col1': 123456, 'col2': 123456,..."
    )
    assert len(repr(scalar)) == 80


def test_scalar_repr_long():
    pa = pytest.importorskip("pyarrow")
    scalar = na.Array(pa.array(["abcdefg" * 10]))[0]
    assert repr(scalar).endswith("...")
    assert len(repr(scalar)) == 80


def test_array_repr():
    array = na.Array(range(10), na.int32())
    one_to_ten = "\n".join(str(i) for i in range(10))

    assert repr(array) == f"nanoarrow.Array<int32>[10]\n{one_to_ten}"

    array = na.Array(range(11), na.int32())
    assert (
        repr(array) == f"nanoarrow.Array<int32>[11]\n{one_to_ten}\n...and 1 more item"
    )

    array = na.Array(range(12), na.int32())
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
    array = na.Array(c_array)

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
    array = na.Array(pa.array(["a" * 78]))
    repr_lines = repr(array).splitlines()
    assert len(repr_lines) == 2
    assert not repr_lines[1].endswith("...")
    assert len(repr_lines[1]) == 80

    # Check that wide output is truncated with a ...
    array = na.Array(pa.array(["a" * 79]))
    repr_lines = repr(array).splitlines()
    assert len(repr_lines) == 2
    assert repr_lines[1].endswith("...")
    assert len(repr_lines[1]) == 80
