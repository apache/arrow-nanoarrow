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


import os
import pathlib
import tempfile

import pytest

import nanoarrow as na
from nanoarrow.ipc import InputStream


def test_array_stream_iter():
    stream = na.ArrayStream([1, 2, 3], na.int32())
    assert stream.schema.type == na.Type.INT32
    stream_iter = iter(stream)

    assert list(next(stream_iter).iter_py()) == [1, 2, 3]
    with pytest.raises(StopIteration):
        next(stream_iter)


def test_array_stream_iter_chunks():
    stream = na.ArrayStream([1, 2, 3], na.int32())
    chunks = list(stream.iter_chunks())
    assert len(chunks) == 1
    assert chunks[0].to_pylist() == [1, 2, 3]


def test_array_stream_iter_py():
    stream = na.ArrayStream([1, 2, 3], na.int32())
    assert list(stream.iter_py()) == [1, 2, 3]


def test_array_stream_iter_tuples():
    c_array = na.c_array_from_buffers(
        na.struct({"col1": na.int32(), "col2": na.string()}),
        length=3,
        buffers=[None],
        children=[
            na.c_array([1, 2, 3], na.int32()),
            na.c_array(["a", "b", "c"], na.string()),
        ],
    )

    stream = na.ArrayStream(c_array)
    assert list(stream.iter_tuples()) == [(1, "a"), (2, "b"), (3, "c")]


def test_array_stream_to_pylist():
    stream = na.ArrayStream([1, 2, 3], na.int32())
    assert stream.to_pylist() == [1, 2, 3]


def test_array_stream_to_columns():
    c_array = na.c_array_from_buffers(
        na.struct({"col1": na.int32(), "col2": na.string()}),
        length=3,
        buffers=[None],
        children=[
            na.c_array([1, 2, 3], na.int32()),
            na.c_array(["a", "b", "c"], na.string()),
        ],
    )

    stream = na.ArrayStream(c_array)
    names, columns = stream.to_columns_pysequence()
    assert names == ["col1", "col2"]
    assert list(columns[0]) == [1, 2, 3]
    assert list(columns[1]) == ["a", "b", "c"]


def test_array_stream_read_all():
    stream = na.ArrayStream([1, 2, 3], na.int32())
    array = stream.read_all()
    assert array.schema.type == na.Type.INT32
    assert array.to_pylist() == [1, 2, 3]


def test_array_stream_read_next():
    stream = na.ArrayStream([1, 2, 3], na.int32())
    array = stream.read_next()
    assert array.schema.type == na.Type.INT32
    assert array.to_pylist() == [1, 2, 3]

    with pytest.raises(StopIteration):
        stream.read_next()


def test_array_stream_close():
    stream = na.ArrayStream([], na.int32())
    stream.close()
    with pytest.raises(RuntimeError, match="array stream is released"):
        stream.read_all()


def test_array_stream_context_manager():
    stream = na.ArrayStream([], na.int32())
    with stream:
        pass

    with pytest.raises(RuntimeError, match="array stream is released"):
        stream.read_all()


def test_array_stream_from_readable():
    stream = na.ArrayStream.from_readable(InputStream.example_bytes())
    assert stream.schema.type == na.Type.STRUCT
    assert list(stream.read_all().iter_tuples()) == [(1,), (2,), (3,)]


def test_array_stream_from_path():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.arrows")
        with open(path, "wb") as f:
            f.write(InputStream.example_bytes())

        stream = na.ArrayStream.from_path(path)
        assert stream.schema.type == na.Type.STRUCT
        assert list(stream.read_all().iter_tuples()) == [(1,), (2,), (3,)]


def test_array_stream_from_url():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.arrows")
        with open(path, "wb") as f:
            f.write(InputStream.example_bytes())

        uri = pathlib.Path(path).as_uri()
        with na.ArrayStream.from_url(uri) as stream:
            assert stream.schema.type == na.Type.STRUCT
            assert list(stream.read_all().iter_tuples()) == [(1,), (2,), (3,)]
