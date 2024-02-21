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
from nanoarrow._lib import NanoarrowException
from nanoarrow.ipc import Stream

import nanoarrow as na


def test_ipc_stream_example():

    with Stream.example() as input:
        assert input._is_valid() is True
        assert "BytesIO object" in repr(input)

        stream = na.c_array_stream(input)
        assert input._is_valid() is False
        assert stream.is_valid() is True
        assert repr(input) == "<nanoarrow.ipc.Stream <invalid>>"
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
        batch = na.c_array_view(batches[0])
        assert list(batch.child(0).buffer(1)) == [1, 2, 3]


def test_ipc_stream_from_path():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.arrows")
        with open(path, "wb") as f:
            f.write(Stream.example_bytes())

        with Stream.from_path(path) as input:
            assert repr(path) in repr(input)
            with na.c_array_stream(input) as stream:
                batches = list(stream)
                assert len(batches) == 1
                assert batches[0].length == 3


def test_ipc_stream_from_url():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.arrows")
        with open(path, "wb") as f:
            f.write(Stream.example_bytes())

        uri = pathlib.Path(path).as_uri()
        with Stream.from_url(uri) as input:
            with na.c_array_stream(input) as stream:
                batches = list(stream)
                assert len(batches) == 1
                assert batches[0].length == 3


def test_ipc_stream_python_exception_on_read():
    class ExtraordinarilyInconvenientFile:
        def readinto(self, obj):
            raise RuntimeError("I error for all read requests")

    input = Stream.from_readable(ExtraordinarilyInconvenientFile())
    with pytest.raises(
        NanoarrowException, match="RuntimeError: I error for all read requests"
    ):
        na.c_array_stream(input)


def test_ipc_stream_error_on_read():
    with io.BytesIO(Stream.example_bytes()[:100]) as f:
        with Stream.from_readable(f) as input:

            with pytest.raises(
                NanoarrowException,
                match="Expected >= 280 bytes of remaining data",
            ):
                na.c_array_stream(input)
