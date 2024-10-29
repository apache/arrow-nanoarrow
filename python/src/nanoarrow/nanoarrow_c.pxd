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


from libc.stdint cimport int8_t, uint8_t, int16_t, uint16_t
from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t

cdef extern from "nanoarrow/nanoarrow.h" nogil:

    cdef int NANOARROW_OK
    cdef int NANOARROW_MAX_FIXED_BUFFERS
    cdef int ARROW_FLAG_DICTIONARY_ORDERED
    cdef int ARROW_FLAG_NULLABLE
    cdef int ARROW_FLAG_MAP_KEYS_SORTED

    struct ArrowSchema:
        const char* format
        const char* name
        const char* metadata
        int64_t flags
        int64_t n_children
        ArrowSchema** children
        ArrowSchema* dictionary
        void (*release)(ArrowSchema*)
        void* private_data

    struct ArrowArray:
        int64_t length
        int64_t null_count
        int64_t offset
        int64_t n_buffers
        int64_t n_children
        const void** buffers
        ArrowArray** children
        ArrowArray* dictionary
        void (*release)(ArrowArray*)
        void* private_data

    struct ArrowArrayStream:
        int (*get_schema)(ArrowArrayStream*, ArrowSchema* out)
        int (*get_next)(ArrowArrayStream*, ArrowArray* out)
        const char* (*get_last_error)(ArrowArrayStream*)
        void (*release)(ArrowArrayStream*)
        void* private_data

    struct ArrowError:
        char message[1024]

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
        NANOARROW_TYPE_RUN_END_ENCODED
        NANOARROW_TYPE_BINARY_VIEW
        NANOARROW_TYPE_STRING_VIEW

    enum ArrowTimeUnit:
        NANOARROW_TIME_UNIT_SECOND = 0
        NANOARROW_TIME_UNIT_MILLI = 1
        NANOARROW_TIME_UNIT_MICRO = 2
        NANOARROW_TIME_UNIT_NANO = 3

    enum ArrowValidationLevel:
        NANOARROW_VALIDATION_LEVEL_NONE = 0
        NANOARROW_VALIDATION_LEVEL_MINIMAL = 1
        NANOARROW_VALIDATION_LEVEL_DEFAULT = 2
        NANOARROW_VALIDATION_LEVEL_FULL = 3

    enum ArrowBufferType:
        NANOARROW_BUFFER_TYPE_NONE
        NANOARROW_BUFFER_TYPE_VALIDITY
        NANOARROW_BUFFER_TYPE_TYPE_ID
        NANOARROW_BUFFER_TYPE_UNION_OFFSET
        NANOARROW_BUFFER_TYPE_DATA_OFFSET
        NANOARROW_BUFFER_TYPE_DATA
        NANOARROW_BUFFER_TYPE_VARIADIC_DATA
        NANOARROW_BUFFER_TYPE_VARIADIC_SIZE

    struct ArrowStringView:
        const char* data
        int64_t size_bytes

    union ArrowBufferViewData:
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
        const ArrowBinaryView* as_binary_view

    struct ArrowBufferView:
        ArrowBufferViewData data
        int64_t size_bytes

    struct ArrowBufferAllocator:
        uint8_t* (*reallocate)(ArrowBufferAllocator* allocator, uint8_t* ptr,
                         int64_t old_size, int64_t new_size)
        void (*free)(ArrowBufferAllocator* allocator, uint8_t* ptr, int64_t size)
        void* private_data

    struct ArrowBuffer:
        uint8_t* data
        int64_t size_bytes
        int64_t capacity_bytes
        ArrowBufferAllocator allocator

    struct ArrowBitmap:
        ArrowBuffer buffer
        int64_t size_bits

    struct ArrowLayout:
        ArrowBufferType buffer_type[3]
        ArrowType buffer_data_type[3]
        int64_t element_size_bits[3]
        int64_t child_size_elements

    struct ArrowArrayView:
        const ArrowArray* array
        int64_t offset
        int64_t length
        int64_t null_count
        ArrowType storage_type
        ArrowLayout layout
        ArrowBufferView buffer_views[3]
        int64_t n_children
        ArrowArrayView** children
        ArrowArrayView* dictionary
        int8_t* union_type_id_map
        int32_t n_variadic_buffers
        const void** variadic_buffers
        int64_t* variadic_buffer_sizes

    struct ArrowMetadataReader:
        const char* metadata
        int64_t offset
        int32_t remaining_keys

    struct ArrowSchemaView:
        const ArrowSchema* schema
        ArrowType type
        ArrowType storage_type
        ArrowLayout layout
        ArrowStringView extension_name
        ArrowStringView extension_metadata
        int32_t fixed_size
        int32_t decimal_bitwidth
        int32_t decimal_precision
        int32_t decimal_scale
        ArrowTimeUnit time_unit
        const char* timezone
        const char* union_type_ids

    struct ArrowBinaryViewInlined:
        int32_t size
        uint8_t data[12]

    struct ArrowBinaryViewRef:
        int32_t size
        uint8_t prefix[4]
        int32_t buffer_index
        int32_t offset

    union ArrowBinaryView:
        ArrowBinaryViewInlined inlined
        ArrowBinaryViewRef ref
        int64_t alignment_dummy

    ctypedef  int ArrowErrorCode
    ctypedef  void (*ArrowBufferDeallocatorCallback)(ArrowBufferAllocator* allocator,
                                               uint8_t* ptr, int64_t size)

    const char* ArrowTypeString(ArrowType type)
    const char* ArrowTimeUnitString(ArrowTimeUnit time_unit)

    void* ArrowMalloc(int64_t size)
    void ArrowFree(void* ptr)
    ArrowBufferAllocator ArrowBufferDeallocator(ArrowBufferDeallocatorCallback, void* private_data)
    void ArrowSchemaMove(ArrowSchema* src, ArrowSchema* dst)
    void ArrowSchemaRelease(ArrowSchema* schema)
    void ArrowArrayMove(ArrowArray* src, ArrowArray* dst)
    void ArrowArrayStreamMove(ArrowArrayStream* src, ArrowArrayStream* dst)
    ArrowErrorCode ArrowArrayStreamGetSchema(ArrowArrayStream* array_stream, ArrowSchema* out, ArrowError* error)
    ArrowErrorCode ArrowArrayStreamGetNext(ArrowArrayStream* array_stream, ArrowArray* out, ArrowError* error)
    void ArrowSchemaRelease(ArrowSchema* schema)
    void ArrowArrayMove(ArrowArray* src, ArrowArray* dst)
    void ArrowArrayRelease(ArrowArray* array)
    void ArrowArrayStreamRelease(ArrowArrayStream* array_stream)
    const char* ArrowNanoarrowVersion()
    int64_t ArrowResolveChunk64(int64_t index, const int64_t* offsets, int64_t lo, int64_t hi)
    void ArrowSchemaInit(ArrowSchema* schema)
    ArrowErrorCode ArrowSchemaInitFromType(ArrowSchema* schema, ArrowType type)
    int64_t ArrowSchemaToString(const ArrowSchema* schema, char* out, int64_t n, char recursive)
    ArrowErrorCode ArrowSchemaSetType(ArrowSchema* schema, ArrowType type)
    ArrowErrorCode ArrowSchemaSetTypeFixedSize(ArrowSchema* schema, ArrowType type, int32_t fixed_size)
    ArrowErrorCode ArrowSchemaSetTypeDecimal(ArrowSchema* schema, ArrowType type, int32_t decimal_precision, int32_t decimal_scale)
    ArrowErrorCode ArrowSchemaSetTypeDateTime(ArrowSchema* schema, ArrowType type, ArrowTimeUnit time_unit, const char* timezone)
    ArrowErrorCode ArrowSchemaSetFormat(ArrowSchema* schema, const char* format)
    ArrowErrorCode ArrowSchemaSetName(ArrowSchema* schema, const char* name)
    ArrowErrorCode ArrowSchemaSetMetadata(ArrowSchema* schema, const char* metadata)
    ArrowErrorCode ArrowSchemaDeepCopy(const ArrowSchema* schema, ArrowSchema* schema_out)
    ArrowErrorCode ArrowSchemaAllocateChildren(ArrowSchema* schema, int64_t n_children)
    ArrowErrorCode ArrowSchemaAllocateDictionary(ArrowSchema* schema)
    ArrowErrorCode ArrowMetadataReaderInit(ArrowMetadataReader* reader, const char* metadata)
    ArrowErrorCode ArrowMetadataReaderInit(ArrowMetadataReader* reader, const char* metadata)
    ArrowErrorCode ArrowMetadataReaderRead(ArrowMetadataReader* reader, ArrowStringView* key_out, ArrowStringView* value_out)
    ArrowErrorCode ArrowMetadataBuilderInit(ArrowBuffer* buffer, const char* metadata)
    ArrowErrorCode ArrowMetadataBuilderAppend(ArrowBuffer* buffer, ArrowStringView key, ArrowStringView value)
    ArrowErrorCode ArrowSchemaViewInit(ArrowSchemaView* schema_view, const ArrowSchema* schema, ArrowError* error)
    void ArrowBufferInit(ArrowBuffer* buffer)
    void ArrowBufferReset(ArrowBuffer* buffer)
    void ArrowBufferMove(ArrowBuffer* src, ArrowBuffer* dst)
    ArrowErrorCode ArrowBufferReserve(ArrowBuffer* buffer, int64_t additional_size_bytes)
    ArrowErrorCode ArrowBufferAppendFill(ArrowBuffer* buffer, uint8_t value, int64_t size_bytes)
    ArrowErrorCode ArrowBufferAppendInt8(ArrowBuffer* buffer, int8_t value)
    ArrowErrorCode ArrowBufferAppendInt64(ArrowBuffer* buffer, int64_t value)
    int8_t ArrowBitGet(const uint8_t* bits, int64_t i)
    int64_t ArrowBitCountSet(const uint8_t* bits, int64_t i_from, int64_t i_to)
    void ArrowBitsUnpackInt8(const uint8_t* bits, int64_t start_offset, int64_t length, int8_t* out)
    void ArrowBitmapInit(ArrowBitmap* bitmap)
    ArrowErrorCode ArrowBitmapReserve(ArrowBitmap* bitmap, int64_t additional_size_bits)
    ArrowErrorCode ArrowBitmapAppend(ArrowBitmap* bitmap, uint8_t bits_are_set, int64_t length)
    void ArrowBitmapAppendUnsafe(ArrowBitmap* bitmap, uint8_t bits_are_set, int64_t length)
    void ArrowBitmapReset(ArrowBitmap* bitmap)
    ArrowErrorCode ArrowArrayInitFromType(ArrowArray* array, ArrowType storage_type)
    ArrowErrorCode ArrowArrayInitFromSchema(ArrowArray* array, const ArrowSchema* schema, ArrowError* error)
    ArrowErrorCode ArrowArrayAllocateChildren(ArrowArray* array, int64_t n_children)
    ArrowErrorCode ArrowArrayAllocateDictionary(ArrowArray* array)
    ArrowBuffer* ArrowArrayBuffer(ArrowArray* array, int64_t i)
    ArrowErrorCode ArrowArrayStartAppending(ArrowArray* array)
    ArrowErrorCode ArrowArrayAppendNull(ArrowArray* array, int64_t n)
    ArrowErrorCode ArrowArrayAppendBytes(ArrowArray* array, ArrowBufferView value)
    ArrowErrorCode ArrowArrayAppendString(ArrowArray* array, ArrowStringView value)
    ArrowErrorCode ArrowArrayFinishBuilding(ArrowArray* array, ArrowValidationLevel validation_level, ArrowError* error)
    void ArrowArrayViewInitFromType(ArrowArrayView* array_view, ArrowType storage_type)
    ArrowErrorCode ArrowArrayViewInitFromSchema(ArrowArrayView* array_view, const ArrowSchema* schema, ArrowError* error)
    ArrowErrorCode ArrowArrayViewInitFromSchema(ArrowArrayView* array_view, const ArrowSchema* schema, ArrowError* error)
    ArrowErrorCode ArrowArrayViewSetArray(ArrowArrayView* array_view, const ArrowArray* array, ArrowError* error)
    ArrowErrorCode ArrowArrayViewSetArrayMinimal(ArrowArrayView* array_view, const ArrowArray* array, ArrowError* error)
    int64_t ArrowArrayViewGetNumBuffers(ArrowArrayView* array_view)
    ArrowBufferView ArrowArrayViewGetBufferView(ArrowArrayView* array_view, int64_t i)
    ArrowBufferType ArrowArrayViewGetBufferType(ArrowArrayView* array_view, int64_t i)
    ArrowType ArrowArrayViewGetBufferDataType(ArrowArrayView* array_view, int64_t i)
    int64_t ArrowArrayViewGetBufferElementSizeBits(ArrowArrayView* array_view, int64_t i)
    void ArrowArrayViewReset(ArrowArrayView* array_view)
    int8_t ArrowArrayViewIsNull(const ArrowArrayView* array_view, int64_t i)
    int64_t ArrowArrayViewComputeNullCount(const ArrowArrayView* array_view)
    ArrowStringView ArrowArrayViewGetStringUnsafe(const ArrowArrayView* array_view, int64_t i)
    ArrowBufferView ArrowArrayViewGetBytesUnsafe(const ArrowArrayView* array_view, int64_t i)
    ArrowErrorCode ArrowBasicArrayStreamInit(ArrowArrayStream* array_stream, ArrowSchema* schema, int64_t n_arrays)
    void ArrowBasicArrayStreamSetArray(ArrowArrayStream* array_stream, int64_t i, ArrowArray* array)
    ArrowErrorCode ArrowBasicArrayStreamValidate(const ArrowArrayStream* array_stream, ArrowError* error)
