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

import numpy as np
from nanoarrow._lib import CArrowType
from nanoarrow.c_lib import c_array_stream, c_schema_view
from nanoarrow.iterator import PyIterator


def to_numpy_dtype(schema):
    schema_view = c_schema_view(schema)
    if schema_view.extension_name is not None:
        return np.dtype("O")

    type_id = schema_view.type_id
    if type_id in _ARROW_TO_NUMPY_FIXED:
        return _ARROW_TO_NUMPY_FIXED[type_id]
    elif type_id in _ARROW_TO_NUMPY_PARAMETERIZED:
        return _ARROW_TO_NUMPY_PARAMETERIZED[type_id](schema_view)
    else:
        return np.dtype("O")


def to_numpy(obj, schema=None, dtype=None):
    if hasattr(obj, "__len__"):
        total_length = len(obj)
    else:
        total_length = None

    dtype, chunks = NumPyIterator.get_dtype_and_iterator(
        obj, schema=schema, dtype=dtype
    )
    first_chunk = next(chunks, None)
    second_chunk = next(chunks, None)

    if first_chunk is None:
        return np.array([], dtype=dtype)

    if second_chunk is None:
        return first_chunk

    if total_length:
        out = np.empty(total_length, dtype)
        cursor = 0

        out[cursor : (cursor + len(first_chunk))] = first_chunk
        cursor += len(first_chunk)

        out[cursor : (cursor + len(second_chunk))] = second_chunk
        cursor += len(second_chunk)

        for chunk in chunks:
            out[cursor : (cursor + len(chunk))] = chunk
            cursor += len(chunk)

        return out
    else:
        chunks = [first_chunk, second_chunk] + list(chunks)
        return np.concatenate(chunks, dtype=dtype)


class NumPyIterator(PyIterator):
    @classmethod
    def get_dtype_and_iterator(cls, obj, schema=None, dtype=None):
        stream = c_array_stream(obj, schema=schema)
        iterator = cls(stream._get_cached_schema(), dtype=dtype)
        return iterator._dtype, iterator._iter_all(stream)

    def __init__(self, schema, *, _array_view=None, dtype=None):
        super().__init__(schema, _array_view=_array_view)

        if dtype is None:
            self._dtype = to_numpy_dtype(self._schema_view)
        else:
            self._dtype = np.dtype(dtype)

    def _iter_all(self, stream):
        with stream:
            for array in stream:
                self._set_array(array)
                yield from self._iter1(0, array.length)

    def _iter1(self, offset, length):
        if self._dtype.str == "|O":
            yield np.array(list(super()._iter1(offset, length)), dtype=self._dtype)
            return

        type_id = self._schema_view.type_id
        if type_id in _ARROW_ZERO_COPY_TYPES:
            yield self._array_from_buffer(offset, length)
        else:
            raise NotImplementedError("Convert to numpy not implemented")

    def _array_from_buffer(self, offset, length):
        offset += self._array_view.offset
        end = offset + length
        array = np.array(self._array_view.buffer(1), self._dtype, copy=False)
        return array[offset:end]


_ARROW_ZERO_COPY_TYPES = {
    CArrowType.INT8,
    CArrowType.UINT8,
    CArrowType.INT16,
    CArrowType.UINT16,
    CArrowType.INT32,
    CArrowType.UINT32,
    CArrowType.INT64,
    CArrowType.UINT64,
    CArrowType.HALF_FLOAT,
    CArrowType.FLOAT,
    CArrowType.DOUBLE,
    CArrowType.INTERVAL_MONTH_DAY_NANO,
    CArrowType.INTERVAL_DAY_TIME,
    CArrowType.INTERVAL_MONTHS,
    CArrowType.FIXED_SIZE_BINARY,
}

_ARROW_TO_NUMPY_FIXED = {
    CArrowType.BOOL: np.dtype(np.bool_),
    CArrowType.INT8: np.dtype(np.int8),
    CArrowType.UINT8: np.dtype(np.uint8),
    CArrowType.INT16: np.dtype(np.int16),
    CArrowType.UINT16: np.dtype(np.uint16),
    CArrowType.INT32: np.dtype(np.int32),
    CArrowType.UINT32: np.dtype(np.uint32),
    CArrowType.INT64: np.dtype(np.int64),
    CArrowType.UINT64: np.dtype(np.uint64),
    CArrowType.HALF_FLOAT: np.dtype(np.float16),
    CArrowType.FLOAT: np.dtype(np.float32),
    CArrowType.DOUBLE: np.dtype(np.float64),
    CArrowType.INTERVAL_MONTH_DAY_NANO: np.dtype(
        {
            "names": ["month", "day", "ns"],
            "formats": [np.int32, np.int32, np.int64],
        }
    ),
    CArrowType.INTERVAL_DAY_TIME: np.dtype(
        {
            "names": ["day", "ms"],
            "formats": [np.int32, np.int32],
        }
    ),
    CArrowType.INTERVAL_MONTHS: np.dtype(
        {
            "names": ["month"],
            "formats": [np.int32],
        }
    ),
}


def _arrow_to_fixed_size_binary(schema_view):
    return np.dtype(f"S{schema_view.fixed_size}")


_ARROW_TO_NUMPY_PARAMETERIZED = {
    CArrowType.FIXED_SIZE_BINARY: _arrow_to_fixed_size_binary
}
