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
from nanoarrow._lib import NanoarrowException
from nanoarrow.c_lib import CArrayStream

import nanoarrow as na


def test_c_array_stream_from_c_array_stream():
    # Wrapping an existing stream is a no-op
    array_stream = CArrayStream.from_array_list([], na.c_schema(na.int32()))
    stream_from_stream = na.c_array_stream(array_stream)
    assert stream_from_stream is array_stream

    # With requested_schema should go through capsule
    array_stream = CArrayStream.from_array_list([], na.c_schema(na.int32()))
    with pytest.raises(NotImplementedError):
        na.c_array_stream(array_stream, na.int64())


def test_c_array_stream_from_capsule_protocol():
    # Use wrapper object to ensure this is the path taken in the constructor
    class CArrayStreamWrapper:
        def __init__(self, obj):
            self.obj = obj

        def __arrow_c_stream__(self, *args, **kwargs):
            return self.obj.__arrow_c_stream__(*args, **kwargs)

    array_stream = CArrayStream.from_array_list([], na.c_schema(na.int32()))
    array_stream_wrapper = CArrayStreamWrapper(array_stream)
    from_protocol = na.c_array_stream(array_stream_wrapper)
    assert array_stream.is_valid() is False
    assert from_protocol.get_schema().format == "i"


def test_c_array_stream_from_old_pyarrow():
    # Simulate a pyarrow RecordBatchReader with no __arrow_c_stream__
    class MockLegacyPyarrowRecordBatchReader:
        def __init__(self, obj):
            self.obj = obj

        def _export_to_c(self, *args):
            return self.obj._export_to_c(*args)

    MockLegacyPyarrowRecordBatchReader.__module__ = "pyarrow.lib"

    pa = pytest.importorskip("pyarrow")
    reader = pa.RecordBatchReader.from_batches(pa.schema([]), [])
    mock_reader = MockLegacyPyarrowRecordBatchReader(reader)

    array_stream = na.c_array_stream(mock_reader)
    assert array_stream.get_schema().format == "+s"


def test_c_array_stream_from_bare_capsule():
    array_stream = CArrayStream.from_array_list([], na.c_schema(na.int32()))

    # Check from bare capsule without supplying a schema
    capsule = array_stream.__arrow_c_stream__()
    from_capsule = na.c_array_stream(capsule)
    assert from_capsule.get_schema().format == "i"

    array_stream = CArrayStream.from_array_list([], na.c_schema(na.int32()))
    capsule = array_stream.__arrow_c_stream__()

    with pytest.raises(TypeError, match="Can't import c_array_stream"):
        na.c_array_stream(capsule, na.int32())


def test_c_array_stream_from_c_array_fallback():
    # Check that arrays are valid input
    c_array = na.c_array([1, 2, 3], na.int32())
    array_stream = na.c_array_stream(c_array)
    assert array_stream.get_schema().format == "i"
    arrays = list(array_stream)
    assert len(arrays) == 1
    assert arrays[0].buffers == c_array.buffers

    # Check fallback with schema
    array_stream = na.c_array_stream([1, 2, 3], na.int32())
    assert array_stream.get_schema().format == "i"
    arrays = list(array_stream)
    assert len(arrays) == 1


def test_c_array_stream_error():
    msg = "Can't convert object of type NoneType"
    with pytest.raises(TypeError, match=msg):
        na.c_array_stream(None)


def test_array_stream_from_arrays_schema():
    schema_in = na.c_schema(na.int32())

    stream = CArrayStream.from_array_list([], schema_in)
    assert schema_in.is_valid()
    assert list(stream) == []
    assert stream.get_schema().format == "i"

    # Check move of schema
    CArrayStream.from_array_list([], schema_in, move=True)
    assert schema_in.is_valid() is False
    assert stream.get_schema().format == "i"


def test_array_stream_from_arrays():
    schema_in = na.c_schema(na.int32())
    array_in = na.c_array([1, 2, 3], schema_in)
    array_in_buffers = array_in.buffers

    stream = CArrayStream.from_array_list([array_in], schema_in)
    assert array_in.is_valid()
    arrays = list(stream)
    assert len(arrays) == 1
    assert arrays[0].buffers == array_in_buffers

    # Check move of array
    stream = CArrayStream.from_array_list([array_in], schema_in, move=True)
    assert array_in.is_valid() is False
    arrays = list(stream)
    assert len(arrays) == 1
    assert arrays[0].buffers == array_in_buffers


def test_array_stream_from_arrays_validate():
    schema_in = na.c_schema(na.null())
    array_in = na.c_array([1, 2, 3], na.int32())

    # Check that we can skip validation and proceed without error
    stream = CArrayStream.from_array_list([array_in], schema_in, validate=False)
    arrays = list(stream)
    assert len(arrays) == 1
    assert arrays[0].n_buffers == 2

    # ...but that validation does happen by default
    msg = "Expected array with 0 buffer"
    with pytest.raises(NanoarrowException, match=msg):
        CArrayStream.from_array_list([array_in], schema_in)
