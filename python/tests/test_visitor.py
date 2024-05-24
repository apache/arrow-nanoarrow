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
from nanoarrow.c_buffer import CBuffer

import nanoarrow as na
from nanoarrow import visitor


def test_to_pylist():
    array = na.c_array([1, 2, 3], na.int32())
    assert visitor.ToPyListConverter.visit(array) == [1, 2, 3]


def test_convert():
    ints = na.c_array([1, 2, 3], na.int32())
    bools = na.c_array([1, 0, 1], na.bool_())
    strings = na.c_array(["abc", "def", "ghi"], na.string())

    ints_col = visitor.ToPySequenceConverter.visit(ints)
    assert isinstance(ints_col, CBuffer)
    assert ints_col.format == "i"
    assert list(ints_col) == [1, 2, 3]

    bools_col = visitor.ToPySequenceConverter.visit(bools)
    assert isinstance(bools_col, CBuffer)
    assert bools_col.format == "?"
    assert list(bools_col) == [True, False, True]

    strings_col = visitor.ToPySequenceConverter.visit(strings)
    assert isinstance(strings_col, list)
    assert strings_col == ["abc", "def", "ghi"]


def test_convert_non_nullable():
    ints = na.c_array([1, 2, 3], na.int32(nullable=False))
    bools = na.c_array([1, 0, 1], na.bool_(nullable=False))
    strings = na.c_array(["abc", "def", "ghi"], na.string(nullable=False))

    ints_col = visitor.ToPySequenceConverter.visit(ints)
    assert isinstance(ints_col, CBuffer)
    assert ints_col.format == "i"
    assert list(ints_col) == [1, 2, 3]

    bools_col = visitor.ToPySequenceConverter.visit(bools)
    assert isinstance(bools_col, CBuffer)
    assert bools_col.format == "?"
    assert list(bools_col) == [True, False, True]

    strings_col = visitor.ToPySequenceConverter.visit(strings)
    assert isinstance(strings_col, list)
    assert strings_col == ["abc", "def", "ghi"]


def test_convert_columns():
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

    names, columns = visitor.ToColumnsPysequenceConverter.visit(array)
    assert names == ["col1", "col2", "col3"]
    assert list(columns[0]) == [1, 2, 3]
    assert list(columns[1]) == [True, False, True]
    assert columns[2] == ["abc", "def", "ghi"]

    with pytest.raises(ValueError, match="can only be used on a struct array"):
        visitor.ToColumnsPysequenceConverter.visit([], na.int32())

    # Ensure that the columns converter errors for top-level nulls
    array_with_nulls = na.c_array_from_buffers(
        array.schema,
        array.length,
        [na.c_buffer([True, False, True], na.bool_())],
        children=array.children,
    )
    with pytest.raises(ValueError, match="null_count > 0"):
        visitor.ToColumnsPysequenceConverter.visit(array_with_nulls)


def test_contiguous_buffer_converter():
    array = na.Array.from_chunks([[1, 2, 3], [4, 5, 6]], na.int32())
    buffer = visitor.ToPyBufferConverter.visit(array)
    assert list(buffer) == [1, 2, 3, 4, 5, 6]


def test_contiguous_buffer_converter_with_offsets():
    src = [na.c_array([1, 2, 3], na.int32())[1:], na.c_array([4, 5, 6], na.int32())[2:]]
    array = na.Array.from_chunks(src)
    buffer = visitor.ToPyBufferConverter.visit(array)
    assert list(buffer) == [2, 3, 6]


def test_boolean_bytes_converter():
    array = na.Array.from_chunks([[0, 1, 1], [1, 0, 0]], na.bool_())
    buffer = visitor.ToBooleanBufferConverter.visit(array)
    assert list(buffer) == [False, True, True, True, False, False]


def test_boolean_bytes_converter_with_offsets():
    src = [na.c_array([0, 1, 1], na.bool_())[1:], na.c_array([1, 0, 0], na.bool_())[2:]]
    array = na.Array.from_chunks(src)
    buffer = visitor.ToBooleanBufferConverter.visit(array)
    assert list(buffer) == [True, True, False]


def test_nullable_converter():
    # All valid
    array = na.Array.from_chunks([[1, 2, 3], [4, 5, 6]], na.int32())
    is_valid, column = visitor.ToNullableSequenceConverter.visit(
        array, handle_nulls=na.nulls_separate()
    )
    assert is_valid is None
    assert list(column) == [1, 2, 3, 4, 5, 6]

    # Only nulls in the first chunk
    array = na.Array.from_chunks([[1, None, 3], [4, 5, 6]], na.int32())
    is_valid, column = visitor.ToNullableSequenceConverter.visit(
        array, handle_nulls=na.nulls_separate()
    )
    assert list(is_valid) == [True, False, True, True, True, True]
    assert list(column) == [1, 0, 3, 4, 5, 6]

    # Only nulls in the second chunk
    array = na.Array.from_chunks([[1, 2, 3], [4, None, 6]], na.int32())
    is_valid, column = visitor.ToNullableSequenceConverter.visit(
        array, handle_nulls=na.nulls_separate()
    )
    assert list(is_valid) == [True, True, True, True, False, True]
    assert list(column) == [1, 2, 3, 4, 0, 6]

    # Nulls in both chunks
    array = na.Array.from_chunks([[1, None, 3], [4, None, 6]], na.int32())
    is_valid, column = visitor.ToNullableSequenceConverter.visit(
        array, handle_nulls=na.nulls_separate()
    )
    assert list(is_valid) == [True, False, True, True, False, True]
    assert list(column) == [1, 0, 3, 4, 0, 6]


def test_nulls_forbid():
    is_valid_non_empty = na.c_buffer([1, 0, 1], na.uint8())
    data = na.c_buffer([1, 2, 3], na.int32())

    forbid_nulls = visitor.nulls_forbid()
    assert forbid_nulls(None, data) is data
    with pytest.raises(ValueError):
        forbid_nulls(is_valid_non_empty, data)


def test_numpy_null_handling():
    np = pytest.importorskip("numpy")

    is_valid_non_empty = memoryview(na.c_buffer([1, 0, 1], na.uint8())).cast("?")
    data = na.c_buffer([1, 2, 3], na.int32())

    # Check nulls as sentinel
    nulls_as_sentinel = visitor.nulls_as_sentinel()
    np.testing.assert_array_equal(
        nulls_as_sentinel(None, data), np.array([1, 2, 3], np.int32)
    )
    np.testing.assert_array_equal(
        nulls_as_sentinel(is_valid_non_empty, data),
        np.array([1, np.nan, 3], dtype=np.float64),
    )
