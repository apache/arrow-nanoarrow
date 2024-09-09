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
from datetime import date, datetime, timedelta, timezone

import pytest
from nanoarrow.c_array_stream import CArrayStream

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


def test_array_alias_constructor():
    array = na.array([1, 2, 3], na.int32())
    assert array.schema.type == na.Type.INT32


def test_array_from_chunks():
    # Check with explicit schema
    array = na.Array.from_chunks([[1, 2, 3], [4, 5, 6]], na.int32())
    assert array.schema.type == na.Type.INT32
    assert array.n_chunks == 2
    assert array.to_pylist() == [1, 2, 3, 4, 5, 6]

    # Check with schema inferred from first chunk
    array = na.Array.from_chunks(array.iter_chunks())
    assert array.schema.type == na.Type.INT32
    assert array.n_chunks == 2
    assert array.to_pylist() == [1, 2, 3, 4, 5, 6]

    # Check empty
    array = na.Array.from_chunks([], na.int32())
    assert array.schema.type == na.Type.INT32
    assert len(array) == 0
    assert array.n_chunks == 0

    msg = "Can't create empty Array from chunks without schema"
    with pytest.raises(ValueError, match=msg):
        na.Array.from_chunks([])


def test_array_from_chunks_validate():
    chunks = [na.c_array([1, 2, 3], na.uint32()), na.c_array([1, 2, 3], na.int32())]
    # Check that we get validation by default
    with pytest.raises(ValueError, match="Expected schema"):
        na.Array.from_chunks(chunks)

    # ...but that one can opt out
    array = na.Array.from_chunks(chunks, validate=False)
    assert array.to_pylist() == [1, 2, 3, 1, 2, 3]


def test_array_empty():
    array = na.Array([], na.int32())
    assert array.schema.type == na.Type.INT32
    assert len(array) == 0
    assert array.offset == 0

    assert array.n_buffers == 2
    assert list(array.buffer(0)) == []
    assert list(array.buffer(1)) == []
    assert list(array.iter_chunk_views()) == []

    assert array.n_children == 0

    assert array.n_chunks == 0
    assert list(array.iter_chunks()) == []
    with pytest.raises(IndexError):
        array.chunk(0)

    assert array.to_pylist() == []
    assert list(array.iter_scalar()) == []
    with pytest.raises(IndexError):
        array[0]

    with na.c_array_stream(array) as stream:
        arrays = list(stream)
        assert len(arrays) == 0

    c_array = na.c_array(array)
    assert len(c_array) == 0
    assert c_array.schema.format == "i"


def test_array_contiguous():
    array = na.Array([1, 2, 3], na.int32())
    assert array.schema.type == na.Type.INT32
    assert len(array) == 3
    assert array.offset == 0

    assert array.n_buffers == 2

    validity, data = array.buffers
    assert list(validity) == []
    assert list(data) == [1, 2, 3]
    assert array.buffer(0) is validity
    assert array.buffer(1) is data

    chunk_views = list(array.iter_chunk_views())
    assert len(chunk_views) == array.n_chunks
    assert chunk_views[0].n_buffers == array.n_buffers
    assert list(chunk_views[0].buffer(1)) == [1, 2, 3]

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

    # Python objects by to_pylist()
    assert array.to_pylist() == list(array.iter_py())

    with na.c_array_stream(array) as stream:
        arrays = list(stream)
        assert len(arrays) == 1

    c_array = na.c_array(array)
    assert len(c_array) == 3
    assert c_array.schema.format == "i"


def test_array_chunked():
    src = [na.c_array([1, 2, 3], na.int32()), na.c_array([4, 5, 6], na.int32())]

    array = na.Array(CArrayStream.from_c_arrays(src, na.c_schema(na.int32())))
    assert array.schema.type == na.Type.INT32
    assert len(array) == 6

    assert array.n_buffers == 2
    with pytest.raises(ValueError, match="Can't export ArrowArray"):
        array.buffers

    chunk_views = list(array.iter_chunk_views())
    assert len(chunk_views) == array.n_chunks
    assert chunk_views[0].n_buffers == array.n_buffers
    assert list(chunk_views[0].buffer(1)) == [1, 2, 3]
    assert list(chunk_views[1].buffer(1)) == [4, 5, 6]

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

    # Python objects by to_pylist()
    assert array.to_pylist() == list(array.iter_py())

    # Sequence via to_pysequence()
    assert list(array.to_pysequence()) == [1, 2, 3, 4, 5, 6]

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
    array = na.Array(CArrayStream.from_c_arrays(src, c_array.schema))

    assert array.n_children == 100
    assert array.child(0).schema.type == na.Type.INT32
    assert array.child(0).n_chunks == 2
    assert array.child(0).to_pylist() == [123456, 123456]

    children = list(array.iter_children())
    assert len(children) == array.n_children

    tuples = list(array.iter_tuples())
    assert len(tuples) == 2
    assert len(tuples[0]) == 100

    names, columns = array.to_columns_pysequence()
    assert names == [f"col{i}" for i in range(100)]
    assert all(len(col) == len(array) for col in columns)


def test_scalar_to_array():
    array = na.Array([123456, 7890], na.int32())
    scalar = scalar = array[1]
    assert scalar.schema is array.schema
    assert scalar.device is array.device
    as_array = na.c_array(scalar)
    assert as_array.offset == 1
    assert len(as_array) == 1
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


def test_array_inspect(capsys):
    array = na.Array(range(10), na.int32())
    array.inspect()
    captured = capsys.readouterr()
    assert captured.out.startswith("<ArrowArray int32>")

    # with children
    c_array = na.c_array_from_buffers(
        na.struct({f"col{i}": na.int32() for i in range(100)}),
        length=1,
        buffers=[None],
        children=[na.c_array([123456], na.int32())] * 100,
    )
    array = na.Array(c_array)
    array.inspect()
    captured = capsys.readouterr()
    assert captured.out.startswith("<ArrowArray struct<col0: int32")


def test_array_serialize():
    import io

    c_array = na.c_array_from_buffers(
        na.struct({"some_col": na.int32()}, nullable=False),
        length=3,
        buffers=[],
        children=[na.c_array([1, 2, 3], na.int32())],
    )
    array = na.Array(c_array)
    schema_serialized = array.schema.serialize()

    serialized = array.serialize()
    array_roundtrip = na.ArrayStream.from_readable(
        schema_serialized + serialized
    ).read_all()
    assert repr(array_roundtrip) == repr(array)

    out = io.BytesIO()
    array.serialize(out)
    array_roundtrip = na.ArrayStream.from_readable(
        schema_serialized + out.getvalue()
    ).read_all()
    assert repr(array_roundtrip) == repr(array)


def test_timestamp_array():
    d1 = int(round(datetime(1985, 12, 31, 0, 0, tzinfo=timezone.utc).timestamp() * 1e3))
    d2 = int(round(datetime(2005, 3, 4, 0, 0, tzinfo=timezone.utc).timestamp() * 1e3))
    array = na.Array([d1, d2], na.timestamp("ms"))
    assert list(array.to_pysequence()) == [
        datetime(1985, 12, 31, 0, 0),
        datetime(2005, 3, 4, 0, 0),
    ]
    assert array.to_pylist() == [
        datetime(1985, 12, 31, 0, 0),
        datetime(2005, 3, 4, 0, 0),
    ]
    assert repr(array).startswith("nanoarrow.Array<timestamp('ms', '')>")


def test_date64_array():
    unix_epoch = date(1970, 1, 1)
    d1, d2 = date(1970, 1, 2), date(1970, 1, 3)
    d1_date64 = int(round((d1 - unix_epoch).total_seconds() * 1e3))
    d2_date64 = int(round((d2 - unix_epoch).total_seconds() * 1e3))
    array = na.Array([d1_date64, d2_date64], na.date64())
    assert list(array.to_pysequence()) == [d1, d2]
    assert array.to_pylist() == [d1, d2]


def test_duration_array():
    unix_epoch = date(1970, 1, 1)
    d1, d2 = date(1970, 1, 2), date(1970, 1, 3)
    d1_date64 = int(round((d1 - unix_epoch).total_seconds() * 1e3))
    d2_date64 = int(round((d2 - unix_epoch).total_seconds() * 1e3))
    array = na.Array([d1_date64, d2_date64], na.duration("ms"))
    assert list(array.to_pysequence()) == [timedelta(days=1), timedelta(days=2)]
    assert array.to_pylist() == [timedelta(days=1), timedelta(days=2)]


def test_timestamp_array_using_struct():
    schema = na.struct(
        {
            "creation_timestamp": na.timestamp("ms"),
        }
    )

    d1 = int(round(datetime(1985, 12, 31, 0, 0, tzinfo=timezone.utc).timestamp() * 1e3))
    d2 = int(round(datetime(2005, 3, 4, 0, 0, tzinfo=timezone.utc).timestamp() * 1e3))

    columns = [
        na.c_array([d1, d2], na.timestamp("ms")),
    ]

    c_array = na.c_array_from_buffers(
        schema, length=columns[0].length, buffers=[None], children=columns
    )
    array = na.Array(c_array)
    names, columns = array.to_columns_pysequence()
    assert names == ["creation_timestamp"]
    assert list(array.to_pysequence()) == [
        {"creation_timestamp": datetime(1985, 12, 31, 0, 0)},
        {"creation_timestamp": datetime(2005, 3, 4, 0, 0)},
    ]
    assert repr(array).startswith(
        "nanoarrow.Array<struct<creation_timestamp: timestamp('ms', '')>"
    )
