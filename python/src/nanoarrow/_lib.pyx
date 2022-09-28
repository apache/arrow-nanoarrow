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

# cython: language_level = 3

"""Low-level nanoarrow Python bindings."""

from libc.stdint cimport int64_t, int8_t, uint8_t, uintptr_t

import numpy as np
cimport numpy as cnp

cnp.import_array()


cdef extern from "nanoarrow.h":
    struct ArrowSchema:
        int64_t n_children
        
    struct ArrowArray:
        int64_t length
        int64_t null_count
        int64_t offset
        const void** buffers

    struct ArrowArrayStream:
        int (*get_schema)(ArrowArrayStream* stream, ArrowSchema* out)

    ctypedef int ArrowErrorCode

    enum ArrowType:
        NANOARROW_TYPE_UNINITIALIZED = 0
        NANOARROW_TYPE_NA = 1
        NANOARROW_TYPE_BOOL
        NANOARROW_TYPE_UINT8
        NANOARROW_TYPE_INT8
        NANOARROW_TYPE_UINT16
        NANOARROW_TYPE_INT16
        NANOARROW_TYPE_UINT32
        NANOARROW_TYPE_INT32
        NANOARROW_TYPE_UINT64
        NANOARROW_TYPE_INT64
        NANOARROW_TYPE_HALF_FLOAT
        NANOARROW_TYPE_FLOAT
        NANOARROW_TYPE_DOUBLE
        NANOARROW_TYPE_STRING
        NANOARROW_TYPE_BINARY
        NANOARROW_TYPE_FIXED_SIZE_BINARY
        NANOARROW_TYPE_DATE32
        NANOARROW_TYPE_DATE64
        NANOARROW_TYPE_TIMESTAMP
        NANOARROW_TYPE_TIME32
        NANOARROW_TYPE_TIME64
        NANOARROW_TYPE_INTERVAL_MONTHS
        NANOARROW_TYPE_INTERVAL_DAY_TIME
        NANOARROW_TYPE_DECIMAL128
        NANOARROW_TYPE_DECIMAL256
        NANOARROW_TYPE_LIST
        NANOARROW_TYPE_STRUCT
        NANOARROW_TYPE_SPARSE_UNION
        NANOARROW_TYPE_DENSE_UNION
        NANOARROW_TYPE_DICTIONARY
        NANOARROW_TYPE_MAP
        NANOARROW_TYPE_EXTENSION
        NANOARROW_TYPE_FIXED_SIZE_LIST
        NANOARROW_TYPE_DURATION
        NANOARROW_TYPE_LARGE_STRING
        NANOARROW_TYPE_LARGE_BINARY
        NANOARROW_TYPE_LARGE_LIST
        NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO

    enum ArrowBufferType:
        NANOARROW_BUFFER_TYPE_NONE
        NANOARROW_BUFFER_TYPE_VALIDITY
        NANOARROW_BUFFER_TYPE_TYPE_ID
        NANOARROW_BUFFER_TYPE_UNION_OFFSET
        NANOARROW_BUFFER_TYPE_DATA_OFFSET
        NANOARROW_BUFFER_TYPE_DATA

    struct ArrowError:
        pass

    struct ArrowLayout:
        ArrowBufferType buffer_type[3]
        int64_t element_size_bits[3]
        int64_t child_size_elements

    cdef union buffer_data:
        const void* data
        const int8_t* as_int8
        const uint8_t* as_uint8

    struct ArrowBufferView:
        buffer_data data
        int64_t n_bytes

    struct ArrowBuffer:
        uint8_t* data
        int64_t size_bytes

    struct ArrowBitmap:
        ArrowBuffer buffer
        int64_t size_bits

    struct ArrowArrayView:
        ArrowArray* array
        ArrowType storage_type
        ArrowLayout layout
        ArrowBufferView buffer_views[3]
        int64_t n_children
        ArrowArrayView** children

    ArrowErrorCode ArrowArrayViewInitFromSchema(ArrowArrayView* array_view, ArrowSchema* schema, ArrowError* error)
    ArrowErrorCode ArrowArrayViewSetArray(ArrowArrayView* array_view, ArrowArray* array, ArrowError* error)
    int64_t ArrowBitCountSet(const uint8_t* bits, int64_t i_from, int64_t i_to)


cdef dict _numpy_type_map = {
    NANOARROW_TYPE_UINT8: cnp.NPY_UINT8,
    NANOARROW_TYPE_INT8: cnp.NPY_INT8,
    NANOARROW_TYPE_UINT16: cnp.NPY_UINT16,
    NANOARROW_TYPE_INT16: cnp.NPY_INT16,
    NANOARROW_TYPE_UINT32: cnp.NPY_UINT32,
    NANOARROW_TYPE_INT32: cnp.NPY_INT32,
    NANOARROW_TYPE_UINT64: cnp.NPY_UINT64,
    NANOARROW_TYPE_INT64: cnp.NPY_INT64,
    NANOARROW_TYPE_HALF_FLOAT: cnp.NPY_FLOAT16,
    NANOARROW_TYPE_FLOAT: cnp.NPY_FLOAT32,
    NANOARROW_TYPE_DOUBLE: cnp.NPY_FLOAT64,
}


def as_numpy_array(arr):
    cdef ArrowSchema schema
    cdef ArrowArray array
    cdef ArrowArrayView array_view
    cdef ArrowError error

    arr._export_to_c(<uintptr_t> &array, <uintptr_t> &schema)
    ArrowArrayViewInitFromSchema(&array_view, &schema, &error)

    # primitive arrays have DATA as the second buffer
    if array_view.layout.buffer_type[1] != NANOARROW_BUFFER_TYPE_DATA:
        raise TypeError("Cannot convert a non-primitive array")

    # disallow nulls for this method
    if array.null_count > 0:
        raise ValueError("Cannot convert array with nulls")
    elif array.null_count < 0:
        # not yet computed
        if array_view.layout.buffer_type[0] == NANOARROW_BUFFER_TYPE_VALIDITY:
            if array.buffers[0] != NULL:
                null_count = ArrowBitCountSet(
                    <const uint8_t *>array.buffers[0], array.offset, array.length
                )
                if null_count > 0:
                    raise ValueError("Cannot convert array with nulls")

    cdef int type_num
    if array_view.storage_type in _numpy_type_map:
        type_num = _numpy_type_map[array_view.storage_type]
    else:
        raise NotImplementedError(array_view.storage_type)

    cdef cnp.npy_intp dims[1]
    dims[0] = array.length
    cdef cnp.ndarray result = cnp.PyArray_New(
        np.ndarray, 1, dims, type_num, NULL, <void *> array.buffers[1], -1, 0, <object>NULL
    )
    # TODO set base

    return result
