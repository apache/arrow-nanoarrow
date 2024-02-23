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
