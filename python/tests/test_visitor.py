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

from nanoarrow.c_array_stream import CArrayStream

import nanoarrow as na
from nanoarrow import visitor


def test_to_pylist():
    assert visitor.to_pylist([1, 2, 3], na.int32()) == [1, 2, 3]


def test_to_columms():
    array = na.c_array_from_buffers(
        na.struct({"col1": na.int32(), "col2": na.bool_(), "col3": na.string()}),
        length=3,
        buffers=[None],
        children=[
            na.c_array([1, 2, 3], na.int32()),
            na.c_array([1, 0, 1], na.bool_()),
            na.c_array(["abc", "def", "ghi"], na.string()),
        ],
    )

    columns = visitor.to_columns(array)
    assert list(columns.keys()) == ["col1", "col2", "col3"]
    assert list(columns["col1"]) == [1, 2, 3]
    assert list(columns["col2"]) == [True, False, True]
    assert columns["col3"] == ["abc", "def", "ghi"]


def test_buffer_concatenator():
    src = [na.c_array([1, 2, 3], na.int32()), na.c_array([4, 5, 6], na.int32())]
    stream = CArrayStream.from_array_list(src, na.c_schema(na.int32()))
    buffer = visitor.BufferConcatenator.visit(stream, buffer_index=1)
    assert list(buffer) == [1, 2, 3, 4, 5, 6]


def test_buffer_concatenator_with_offsets():
    src = [na.c_array([1, 2, 3], na.int32())[1:], na.c_array([4, 5, 6], na.int32())[2:]]
    stream = CArrayStream.from_array_list(src, na.c_schema(na.int32()))
    buffer = visitor.BufferConcatenator.visit(stream, buffer_index=1)
    assert list(buffer) == [2, 3, 6]


def test_unpacked_bitmap_concatenator():
    src = [na.c_array([0, 1, 1], na.bool_()), na.c_array([1, 0, 0], na.bool_())]
    stream = CArrayStream.from_array_list(src, na.c_schema(na.bool_()))
    buffer = visitor.UnpackedBitmapConcatenator.visit(stream, buffer_index=1)
    assert list(buffer) == [False, True, True, True, False, False]


def test_unpacked_bitmap_concatenator_with_offsets():
    src = [na.c_array([0, 1, 1], na.bool_())[1:], na.c_array([1, 0, 0], na.bool_())[2:]]
    stream = CArrayStream.from_array_list(src, na.c_schema(na.bool_()))
    buffer = visitor.UnpackedBitmapConcatenator.visit(stream, buffer_index=1)
    assert list(buffer) == [True, True, False]


def test_unpacked_validity_bitmap_concatenator():
    # All valid
    src = [na.c_array([1, 2, 3], na.int32()), na.c_array([4, 5, 6], na.int32())]
    stream = CArrayStream.from_array_list(src, na.c_schema(na.int32()))
    assert visitor.ValidityBufferConcatenator.visit(stream) is None

    # Only nulls in the first chunk
    src = [na.c_array([1, None, 3], na.int32()), na.c_array([4, 5, 6], na.int32())]
    stream = CArrayStream.from_array_list(src, na.c_schema(na.int32()))
    buffer = visitor.ValidityBufferConcatenator.visit(stream)
    assert list(buffer) == [True, False, True, True, True, True]

    # Only nulls in the second chunk
    src = [na.c_array([1, 2, 3], na.int32()), na.c_array([4, None, 6], na.int32())]
    stream = CArrayStream.from_array_list(src, na.c_schema(na.int32()))
    buffer = visitor.ValidityBufferConcatenator.visit(stream)
    assert list(buffer) == [True, True, True, True, False, True]

    # Nulls in both chunks
    src = [na.c_array([1, None, 3], na.int32()), na.c_array([4, None, 6], na.int32())]
    stream = CArrayStream.from_array_list(src, na.c_schema(na.int32()))
    buffer = visitor.ValidityBufferConcatenator.visit(stream)
    assert list(buffer) == [True, False, True, True, False, True]
