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

from libc.stdint cimport int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t

from arrow_c cimport ArrowSchema, ArrowArray, ArrowArrayStream

cdef extern from "nanoarrow.h":
    ctypedef int ArrowErrorCode
    cdef int NANOARROW_OK

    cdef struct ArrowError:
        pass

    enum ArrowType:
        NANOARROW_TYPE_UNINITIALIZED
        NANOARROW_TYPE_NA
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

    enum ArrowTimeUnit:
        NANOARROW_TIME_UNIT_SECOND
        NANOARROW_TIME_UNIT_MILLI
        NANOARROW_TIME_UNIT_MICRO
        NANOARROW_TIME_UNIT_NANO

    cdef struct ArrowStringView:
        const char* data
        int64_t size_bytes

    cdef union buffer_data:
        const void* data
        const int8_t* as_int8
        const uint8_t* as_uint8
        const int16_t* as_int16
        const uint16_t* as_uint16
        const int32_t* as_int32
        const uint32_t* as_uint32
        const int64_t* as_int64
        const uint64_t* as_uint64
        const double* as_double
        const float* as_float
        const char* as_char

    cdef struct ArrowBufferView:
        buffer_data data
        int64_t size_bytes

    cdef struct ArrowBufferAllocator:
        pass

    cdef struct ArrowBuffer:
        uint8_t* data
        int64_t size_bytes
        int64_t capacity_bytes
        ArrowBufferAllocator allocator

    cdef struct ArrowBitmap:
        ArrowBuffer buffer
        int64_t size_bits

    cdef struct ArrowLayout:
        ArrowBufferType buffer_type[3]
        int64_t element_size_bits[3]
        int64_t child_size_elements

    cdef struct ArrowArrayView:
        ArrowArray* array
        ArrowType storage_type
        ArrowLayout layout
        ArrowBufferView buffer_views[3]
        int64_t n_children
        ArrowArrayView** children

    cdef const char* ArrowNanoarrowVersion()
    cdef const char* ArrowErrorMessage(ArrowError* error)

    cdef void ArrowSchemaMove(ArrowSchema* src, ArrowSchema* dst)
    cdef void ArrowArrayMove(ArrowArray* src, ArrowArray* dst)
    cdef void ArrowArrayStreamMove(ArrowArrayStream* src, ArrowArrayStream* dst)

    cdef int64_t ArrowSchemaToString(ArrowSchema* schema, char* out, int64_t n,
                                     char recursive)
    cdef ArrowErrorCode ArrowSchemaDeepCopy(ArrowSchema* schema,
                                            ArrowSchema* schema_out)
    cdef ArrowErrorCode ArrowSchemaSetType(ArrowSchema* schema,ArrowType type_)
    ArrowErrorCode ArrowSchemaSetTypeStruct(ArrowSchema* schema, int64_t n_children)

    cdef struct ArrowMetadataReader:
        pass

    cdef ArrowErrorCode ArrowMetadataReaderInit(ArrowMetadataReader* reader,
                                                const char* metadata)

    cdef ArrowErrorCode ArrowMetadataReaderRead(ArrowMetadataReader* reader,
                                                ArrowStringView* key_out,
                                                ArrowStringView* value_out)

    cdef ArrowErrorCode ArrowArrayViewInitFromSchema(ArrowArrayView* array_view, ArrowSchema* schema, ArrowError* error)
    cdef ArrowErrorCode ArrowArrayViewSetArray(ArrowArrayView* array_view, ArrowArray* array, ArrowError* error)
    cdef int64_t ArrowBitCountSet(const uint8_t* bits, int64_t i_from, int64_t i_to)
