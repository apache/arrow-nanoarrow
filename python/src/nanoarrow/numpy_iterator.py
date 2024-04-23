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
from nanoarrow.c_lib import c_schema_view
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
        return NumPyKnownSizeBuilder.visit(obj, schema, total_length=len(obj))
    else:
        arrays = list(NumPyIterator.iterate(obj, schema))
        return np.concatenate(arrays, dtype=dtype)


class NumPyIterator(PyIterator):
    def __init__(self, schema, *, _array_view=None, dtype=None):
        super().__init__(schema, _array_view=_array_view)
        dtype = self._resolve_dtype(dtype)
        self._array_data_constructor = self._get_array_data_constructor(dtype)
        self._dtype = dtype

    def _resolve_dtype(self, dtype):
        if dtype is None:
            return to_numpy_dtype(self._schema_view)
        else:
            return np.dtype(dtype)

    def _get_array_data_constructor(self, dtype):
        if dtype.str == "O":
            return self._py_object_array

        type_id = self._schema_view.type_id
        if type_id in _ARROW_PY_BUFFER_TYPES:
            return self._py_buffer_array
        elif type_id in _ARROW_TO_ARRAY_DISPATCH:
            return getattr(self, _ARROW_TO_ARRAY_DISPATCH[type_id])
        else:
            raise NotImplementedError("Convert to array not implemented")

    def _get_chunk_array(self, offset, length):
        validity = self._validity_array_bytes(offset, length)
        data = self._array_data_constructor(offset, length)

        if validity and not np.all(validity):
            raise ValueError("null values not supported on convert to numpy")

        return np.array(data, self._dtype)

    def _iter1(self, offset, length):
        yield self._get_chunk_array(offset, length)

    def _validity_array_bytes(self, offset, length):
        if not self._contains_nulls():
            return None

        validity_bits = self._array_view.buffer(0)
        validity_bytes = np.empty(length, np.bool_)
        validity_bits.unpack_bits_into(validity_bytes, offset, length)
        return validity_bytes

    def _py_object_array(self, offset, length):
        return np.array(list(super()._iter1(offset, length)), dtype=self._dtype)

    def _bool_array(self, offset, length):
        offset += self._array_view.offset
        validity_bytes = bytearray(length)
        self._array_view.buffer(0).unpack_bits_into(validity_bytes, offset, length)
        return validity_bytes

    def _variable_binary_array(self, offset, length):
        return self._py_object_array(offset, length)

    def _py_buffer_array(self, offset, length):
        offset += self._array_view.offset
        end = offset + length
        return memoryview(self._array_view.buffer(1))[offset:end]


class NumPyKnownSizeBuilder(NumPyIterator):
    def __init__(self, schema, *, _array_view=None, dtype=None, total_length=None):
        super().__init__(schema, _array_view=_array_view, dtype=dtype)
        self._total_length = total_length

    def begin(self):
        self._out = np.empty(self._total_length, self._dtype)
        self._cursor = 0

    def visit_chunk(self):
        length = self._array_view.length
        array = self._get_chunk_array(0, length)
        self._out[self._cursor : (self._cursor + length)] = array
        self._cursor += length

    def finish(self):
        return self._out


_ARROW_PY_BUFFER_TYPES = {
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

_ARROW_TO_ARRAY_DISPATCH = {
    CArrowType.BOOL: "_array_from_bool",
    CArrowType.STRING: "_variable_binary_array",
    CArrowType.LARGE_STRING: "_variable_binary_array",
    CArrowType.BINARY: "_variable_binary_array",
    CArrowType.LARGE_BINARY: "_variable_binary_array",
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
