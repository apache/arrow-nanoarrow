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

import io
import os
import pathlib
import tempfile

import pytest
from nanoarrow._utils import NanoarrowException

import nanoarrow as na
from nanoarrow.ipc import InputStream, StreamWriter


def test_ipc_stream_example():
    with InputStream.example() as input:
        assert input._is_valid() is True
        assert "BytesIO object" in repr(input)

        stream = na.c_array_stream(input)
        assert input._is_valid() is False
        assert stream.is_valid() is True
        assert repr(input) == "<nanoarrow.ipc.InputStream <invalid>>"
        with pytest.raises(RuntimeError, match="no longer valid"):
            stream = na.c_array_stream(input)

        with stream:
            schema = stream.get_schema()
            assert schema.format == "+s"
            assert schema.child(0).format == "i"
            batches = list(stream)
            assert stream.is_valid() is True

        assert stream.is_valid() is False
        assert len(batches) == 1
        batch = na.c_array(batches[0]).view()
        assert list(batch.child(0).buffer(1)) == [1, 2, 3]


def test_ipc_stream_from_readable():
    with io.BytesIO(InputStream.example_bytes()) as f:
        with InputStream.from_readable(f) as input:
            assert input._is_valid() is True
            assert "BytesIO object" in repr(input)

            with na.c_array_stream(input) as stream:
                batches = list(stream)
                assert len(batches) == 1
                assert len(batches[0]) == 3


def test_ipc_stream_from_path():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.arrows")
        with open(path, "wb") as f:
            f.write(InputStream.example_bytes())

        with InputStream.from_path(path) as input:
            assert repr(path) in repr(input)
            with na.c_array_stream(input) as stream:
                batches = list(stream)
                assert len(batches) == 1
                assert len(batches[0]) == 3


def test_ipc_stream_from_url():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.arrows")
        with open(path, "wb") as f:
            f.write(InputStream.example_bytes())

        uri = pathlib.Path(path).as_uri()
        with InputStream.from_url(uri) as input:
            with na.c_array_stream(input) as stream:
                batches = list(stream)
                assert len(batches) == 1
                assert len(batches[0]) == 3


def test_ipc_stream_python_exception_on_read():
    class ExtraordinarilyInconvenientFile:
        def readinto(self, *args, **kwargs):
            raise RuntimeError("I error for all read requests")

    input = InputStream.from_readable(ExtraordinarilyInconvenientFile())
    with pytest.raises(
        NanoarrowException, match="RuntimeError: I error for all read requests"
    ):
        na.c_array_stream(input)


def test_ipc_stream_error_on_read():
    with io.BytesIO(InputStream.example_bytes()[:100]) as f:
        with InputStream.from_readable(f) as input:

            with pytest.raises(
                NanoarrowException,
                match="Expected >= 280 bytes of remaining data",
            ):
                na.c_array_stream(input)


def test_writer_from_writable():
    array = na.c_array_from_buffers(
        na.struct({"some_col": na.int32()}),
        length=3,
        buffers=[],
        children=[na.c_array([1, 2, 3], na.int32())],
    )

    out = io.BytesIO()
    with StreamWriter.from_writable(out) as writer:
        writer.write_array(array)

    with na.ArrayStream.from_readable(out.getvalue()) as stream:
        assert stream.read_all().to_pylist() == na.Array(array).to_pylist()


def test_writer_from_path():
    array = na.c_array_from_buffers(
        na.struct({"some_col": na.int32()}),
        length=3,
        buffers=[],
        children=[na.c_array([1, 2, 3], na.int32())],
    )

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.arrows")

        with StreamWriter.from_path(path) as writer:
            writer.write_array(array)

        with na.ArrayStream.from_path(path) as stream:
            assert stream.read_all().to_pylist() == na.Array(array).to_pylist()


def test_writer_write_stream_schema():
    array = na.c_array_from_buffers(
        na.struct({"some_col": na.int32()}),
        length=3,
        buffers=[],
        children=[na.c_array([1, 2, 3], na.int32())],
    )
    zero_chunk_array = na.Array.from_chunks([], array.schema)

    out = io.BytesIO()
    with StreamWriter.from_writable(out) as writer:
        writer.write_stream(zero_chunk_array)
        schema_bytes = out.getvalue()

    with StreamWriter.from_writable(out) as writer:
        out.truncate(0)
        out.seek(0)
        writer.write_stream(zero_chunk_array)
        writer.write_stream(zero_chunk_array, write_schema=True)
        two_schema_bytes = out.getvalue()

    assert (schema_bytes + schema_bytes) == two_schema_bytes

    with StreamWriter.from_writable(out) as writer:
        out.truncate(0)
        out.seek(0)
        writer.write_array(array, write_schema=False)
        array_bytes = out.getvalue()

    with StreamWriter.from_writable(out) as writer:
        out.truncate(0)
        out.seek(0)
        writer.write_array(array, write_schema=True)
        both_bytes = out.getvalue()

    assert (schema_bytes + array_bytes) == both_bytes


def test_writer_serialize_stream():
    array = na.c_array_from_buffers(
        na.struct({"some_col": na.int32()}),
        length=3,
        buffers=[],
        children=[na.c_array([1, 2, 3], na.int32())],
    )

    out = io.BytesIO()
    with StreamWriter.from_writable(out) as writer:
        writer.write_array(array)

        # Check that we can't serialize after we've already written to stream
        with pytest.raises(ValueError, match="Can't serialize_stream"):
            writer.serialize_stream(array)

        schema_and_array_bytes = out.getvalue()

    end_of_stream_bytes = b"\xff\xff\xff\xff\x00\x00\x00\x00"

    writer = StreamWriter.from_writable(out)
    out.truncate(0)
    out.seek(0)
    writer.serialize_stream(array)
    assert writer._is_valid() is False

    serialized_bytes = out.getvalue()
    assert (schema_and_array_bytes + end_of_stream_bytes) == serialized_bytes


def test_writer_python_exception_on_write():
    class ExtraordinarilyInconvenientFile:
        def write(self, *args, **kwargs):
            raise RuntimeError("I error for all write requests")

    with pytest.raises(NanoarrowException, match="I error for all write requests"):
        with StreamWriter.from_writable(ExtraordinarilyInconvenientFile()) as writer:
            writer.write(na.c_array([], na.struct([na.int32()])))


def test_writer_error_on_write():
    with pytest.raises(NanoarrowException):
        with StreamWriter.from_writable(io.BytesIO()) as writer:
            writer.write_stream(na.c_array([], na.int32()))
