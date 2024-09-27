// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef NANOARROW_ARRAY_INLINE_H_INCLUDED
#define NANOARROW_ARRAY_INLINE_H_INCLUDED

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>

#include "nanoarrow/common/inline_buffer.h"
#include "nanoarrow/common/inline_types.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline struct ArrowBitmap* ArrowArrayValidityBitmap(struct ArrowArray* array) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  return &private_data->bitmap;
}

static inline struct ArrowBuffer* ArrowArrayBuffer(struct ArrowArray* array, int64_t i) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  switch (i) {
    case 0:
      return &private_data->bitmap.buffer;
    default:
      return private_data->buffers + i - 1;
  }
}

// We don't currently support the case of unions where type_id != child_index;
// however, these functions are used to keep track of where that assumption
// is made.
static inline int8_t _ArrowArrayUnionChildIndex(struct ArrowArray* array,
                                                int8_t type_id) {
  NANOARROW_UNUSED(array);
  return type_id;
}

static inline int8_t _ArrowArrayUnionTypeId(struct ArrowArray* array,
                                            int8_t child_index) {
  NANOARROW_UNUSED(array);
  return child_index;
}

static inline int32_t _ArrowParseUnionTypeIds(const char* type_ids, int8_t* out) {
  if (*type_ids == '\0') {
    return 0;
  }

  int32_t i = 0;
  long type_id;
  char* end_ptr;
  do {
    type_id = strtol(type_ids, &end_ptr, 10);
    if (end_ptr == type_ids || type_id < 0 || type_id > 127) {
      return -1;
    }

    if (out != NULL) {
      out[i] = (int8_t)type_id;
    }

    i++;

    type_ids = end_ptr;
    if (*type_ids == '\0') {
      return i;
    } else if (*type_ids != ',') {
      return -1;
    } else {
      type_ids++;
    }
  } while (1);

  return -1;
}

static inline int8_t _ArrowParsedUnionTypeIdsWillEqualChildIndices(const int8_t* type_ids,
                                                                   int64_t n_type_ids,
                                                                   int64_t n_children) {
  if (n_type_ids != n_children) {
    return 0;
  }

  for (int8_t i = 0; i < n_type_ids; i++) {
    if (type_ids[i] != i) {
      return 0;
    }
  }

  return 1;
}

static inline int8_t _ArrowUnionTypeIdsWillEqualChildIndices(const char* type_id_str,
                                                             int64_t n_children) {
  int8_t type_ids[128];
  int32_t n_type_ids = _ArrowParseUnionTypeIds(type_id_str, type_ids);
  return _ArrowParsedUnionTypeIdsWillEqualChildIndices(type_ids, n_type_ids, n_children);
}

static inline ArrowErrorCode ArrowArrayStartAppending(struct ArrowArray* array) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_UNINITIALIZED:
      return EINVAL;
    case NANOARROW_TYPE_SPARSE_UNION:
    case NANOARROW_TYPE_DENSE_UNION:
      // Note that this value could be -1 if the type_ids string was invalid
      if (private_data->union_type_id_is_child_index != 1) {
        return EINVAL;
      } else {
        break;
      }
    default:
      break;
  }
  if (private_data->storage_type == NANOARROW_TYPE_UNINITIALIZED) {
    return EINVAL;
  }

  // Initialize any data offset buffer with a single zero
  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    if (private_data->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_DATA_OFFSET &&
        private_data->layout.element_size_bits[i] == 64) {
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt64(ArrowArrayBuffer(array, i), 0));
    } else if (private_data->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_DATA_OFFSET &&
               private_data->layout.element_size_bits[i] == 32) {
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(ArrowArrayBuffer(array, i), 0));
    }
  }

  // Start building any child arrays or dictionaries
  for (int64_t i = 0; i < array->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayStartAppending(array->children[i]));
  }

  if (array->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayStartAppending(array->dictionary));
  }

  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayShrinkToFit(struct ArrowArray* array) {
  for (int64_t i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    struct ArrowBuffer* buffer = ArrowArrayBuffer(array, i);
    NANOARROW_RETURN_NOT_OK(ArrowBufferResize(buffer, buffer->size_bytes, 1));
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayShrinkToFit(array->children[i]));
  }

  if (array->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayShrinkToFit(array->dictionary));
  }

  return NANOARROW_OK;
}

static inline ArrowErrorCode _ArrowArrayAppendBits(struct ArrowArray* array,
                                                   int64_t buffer_i, uint8_t value,
                                                   int64_t n) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  struct ArrowBuffer* buffer = ArrowArrayBuffer(array, buffer_i);
  int64_t bytes_required =
      _ArrowRoundUpToMultipleOf8(private_data->layout.element_size_bits[buffer_i] *
                                 (array->length + 1)) /
      8;
  if (bytes_required > buffer->size_bytes) {
    NANOARROW_RETURN_NOT_OK(
        ArrowBufferAppendFill(buffer, 0, bytes_required - buffer->size_bytes));
  }

  ArrowBitsSetTo(buffer->data, array->length, n, value);
  return NANOARROW_OK;
}

static inline ArrowErrorCode _ArrowArrayAppendEmptyInternal(struct ArrowArray* array,
                                                            int64_t n, uint8_t is_valid) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  if (n == 0) {
    return NANOARROW_OK;
  }

  // Some type-specific handling
  switch (private_data->storage_type) {
    case NANOARROW_TYPE_NA:
      // (An empty value for a null array *is* a null)
      array->null_count += n;
      array->length += n;
      return NANOARROW_OK;

    case NANOARROW_TYPE_DENSE_UNION: {
      // Add one null to the first child and append n references to that child
      int8_t type_id = _ArrowArrayUnionTypeId(array, 0);
      NANOARROW_RETURN_NOT_OK(
          _ArrowArrayAppendEmptyInternal(array->children[0], 1, is_valid));
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppendFill(ArrowArrayBuffer(array, 0), type_id, n));
      for (int64_t i = 0; i < n; i++) {
        NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(
            ArrowArrayBuffer(array, 1), (int32_t)array->children[0]->length - 1));
      }
      // For the purposes of array->null_count, union elements are never considered "null"
      // even if some children contain nulls.
      array->length += n;
      return NANOARROW_OK;
    }

    case NANOARROW_TYPE_SPARSE_UNION: {
      // Add n nulls to the first child and append n references to that child
      int8_t type_id = _ArrowArrayUnionTypeId(array, 0);
      NANOARROW_RETURN_NOT_OK(
          _ArrowArrayAppendEmptyInternal(array->children[0], n, is_valid));
      for (int64_t i = 1; i < array->n_children; i++) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendEmpty(array->children[i], n));
      }

      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppendFill(ArrowArrayBuffer(array, 0), type_id, n));
      // For the purposes of array->null_count, union elements are never considered "null"
      // even if some children contain nulls.
      array->length += n;
      return NANOARROW_OK;
    }

    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      NANOARROW_RETURN_NOT_OK(ArrowArrayAppendEmpty(
          array->children[0], n * private_data->layout.child_size_elements));
      break;
    case NANOARROW_TYPE_STRUCT:
      for (int64_t i = 0; i < array->n_children; i++) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendEmpty(array->children[i], n));
      }
      break;

    default:
      break;
  }

  // Append n is_valid bits to the validity bitmap. If we haven't allocated a bitmap yet
  // and we need to append nulls, do it now.
  if (!is_valid && private_data->bitmap.buffer.data == NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapReserve(&private_data->bitmap, array->length + n));
    ArrowBitmapAppendUnsafe(&private_data->bitmap, 1, array->length);
    ArrowBitmapAppendUnsafe(&private_data->bitmap, is_valid, n);
  } else if (private_data->bitmap.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapReserve(&private_data->bitmap, n));
    ArrowBitmapAppendUnsafe(&private_data->bitmap, is_valid, n);
  }

  // Add appropriate buffer fill
  struct ArrowBuffer* buffer;
  int64_t size_bytes;

  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    buffer = ArrowArrayBuffer(array, i);
    size_bytes = private_data->layout.element_size_bits[i] / 8;

    switch (private_data->layout.buffer_type[i]) {
      case NANOARROW_BUFFER_TYPE_NONE:
      case NANOARROW_BUFFER_TYPE_VALIDITY:
        continue;
      case NANOARROW_BUFFER_TYPE_DATA_OFFSET:
        // Append the current value at the end of the offset buffer for each element
        NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(buffer, size_bytes * n));

        for (int64_t j = 0; j < n; j++) {
          ArrowBufferAppendUnsafe(buffer, buffer->data + size_bytes * (array->length + j),
                                  size_bytes);
        }

        // Skip the data buffer
        i++;
        continue;
      case NANOARROW_BUFFER_TYPE_DATA:
      case NANOARROW_BUFFER_TYPE_DATA_VIEW:
        // Zero out the next bit of memory
        if (private_data->layout.element_size_bits[i] % 8 == 0) {
          NANOARROW_RETURN_NOT_OK(ArrowBufferAppendFill(buffer, 0, size_bytes * n));
        } else {
          NANOARROW_RETURN_NOT_OK(_ArrowArrayAppendBits(array, i, 0, n));
        }
        continue;

      case NANOARROW_BUFFER_TYPE_TYPE_ID:
      case NANOARROW_BUFFER_TYPE_UNION_OFFSET:
        // These cases return above
        return EINVAL;
    }
  }

  array->length += n;
  array->null_count += n * !is_valid;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayAppendNull(struct ArrowArray* array, int64_t n) {
  return _ArrowArrayAppendEmptyInternal(array, n, 0);
}

static inline ArrowErrorCode ArrowArrayAppendEmpty(struct ArrowArray* array, int64_t n) {
  return _ArrowArrayAppendEmptyInternal(array, n, 1);
}

static inline ArrowErrorCode ArrowArrayAppendInt(struct ArrowArray* array,
                                                 int64_t value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  struct ArrowBuffer* data_buffer = ArrowArrayBuffer(array, 1);

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_INT64:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(data_buffer, &value, sizeof(int64_t)));
      break;
    case NANOARROW_TYPE_INT32:
      _NANOARROW_CHECK_RANGE(value, INT32_MIN, INT32_MAX);
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(data_buffer, (int32_t)value));
      break;
    case NANOARROW_TYPE_INT16:
      _NANOARROW_CHECK_RANGE(value, INT16_MIN, INT16_MAX);
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt16(data_buffer, (int16_t)value));
      break;
    case NANOARROW_TYPE_INT8:
      _NANOARROW_CHECK_RANGE(value, INT8_MIN, INT8_MAX);
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt8(data_buffer, (int8_t)value));
      break;
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_UINT8:
      _NANOARROW_CHECK_RANGE(value, 0, INT64_MAX);
      return ArrowArrayAppendUInt(array, value);
    case NANOARROW_TYPE_DOUBLE:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendDouble(data_buffer, (double)value));
      break;
    case NANOARROW_TYPE_FLOAT:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendFloat(data_buffer, (float)value));
      break;
    case NANOARROW_TYPE_HALF_FLOAT:
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppendUInt16(data_buffer, ArrowFloatToHalfFloat((float)value)));
      break;
    case NANOARROW_TYPE_BOOL:
      NANOARROW_RETURN_NOT_OK(_ArrowArrayAppendBits(array, 1, value != 0, 1));
      break;
    default:
      return EINVAL;
  }

  if (private_data->bitmap.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1));
  }

  array->length++;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayAppendUInt(struct ArrowArray* array,
                                                  uint64_t value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  struct ArrowBuffer* data_buffer = ArrowArrayBuffer(array, 1);

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_UINT64:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(data_buffer, &value, sizeof(uint64_t)));
      break;
    case NANOARROW_TYPE_UINT32:
      _NANOARROW_CHECK_UPPER_LIMIT(value, UINT32_MAX);
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt32(data_buffer, (uint32_t)value));
      break;
    case NANOARROW_TYPE_UINT16:
      _NANOARROW_CHECK_UPPER_LIMIT(value, UINT16_MAX);
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt16(data_buffer, (uint16_t)value));
      break;
    case NANOARROW_TYPE_UINT8:
      _NANOARROW_CHECK_UPPER_LIMIT(value, UINT8_MAX);
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt8(data_buffer, (uint8_t)value));
      break;
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_INT8:
      _NANOARROW_CHECK_UPPER_LIMIT(value, INT64_MAX);
      return ArrowArrayAppendInt(array, value);
    case NANOARROW_TYPE_DOUBLE:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendDouble(data_buffer, (double)value));
      break;
    case NANOARROW_TYPE_FLOAT:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendFloat(data_buffer, (float)value));
      break;
    case NANOARROW_TYPE_HALF_FLOAT:
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppendUInt16(data_buffer, ArrowFloatToHalfFloat((float)value)));
      break;
    case NANOARROW_TYPE_BOOL:
      NANOARROW_RETURN_NOT_OK(_ArrowArrayAppendBits(array, 1, value != 0, 1));
      break;
    default:
      return EINVAL;
  }

  if (private_data->bitmap.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1));
  }

  array->length++;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayAppendDouble(struct ArrowArray* array,
                                                    double value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  struct ArrowBuffer* data_buffer = ArrowArrayBuffer(array, 1);

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_DOUBLE:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(data_buffer, &value, sizeof(double)));
      break;
    case NANOARROW_TYPE_FLOAT:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendFloat(data_buffer, (float)value));
      break;
    case NANOARROW_TYPE_HALF_FLOAT:
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppendUInt16(data_buffer, ArrowFloatToHalfFloat((float)value)));
      break;
    default:
      return EINVAL;
  }

  if (private_data->bitmap.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1));
  }

  array->length++;
  return NANOARROW_OK;
}

// Binary views only have two fixed buffers, but be aware that they must also
// always have more 1 buffer to store variadic buffer sizes (even if there are none)
#define NANOARROW_BINARY_VIEW_FIXED_BUFFERS 2
#define NANOARROW_BINARY_VIEW_INLINE_SIZE 12
#define NANOARROW_BINARY_VIEW_PREFIX_SIZE 4
#define NANOARROW_BINARY_VIEW_BLOCK_SIZE (32 << 10)  // 32KB

// The Arrow C++ implementation uses anonymous structs as members
// of the ArrowBinaryView. For Cython support in this library, we define
// those structs outside of the ArrowBinaryView
struct ArrowBinaryViewInlined {
  int32_t size;
  uint8_t data[NANOARROW_BINARY_VIEW_INLINE_SIZE];
};

struct ArrowBinaryViewRef {
  int32_t size;
  uint8_t prefix[NANOARROW_BINARY_VIEW_PREFIX_SIZE];
  int32_t buffer_index;
  int32_t offset;
};

union ArrowBinaryView {
  struct ArrowBinaryViewInlined inlined;
  struct ArrowBinaryViewRef ref;
  int64_t alignment_dummy;
};

static inline int32_t ArrowArrayVariadicBufferCount(struct ArrowArray* array) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  return private_data->n_variadic_buffers;
}

static inline ArrowErrorCode ArrowArrayAddVariadicBuffers(struct ArrowArray* array,
                                                          int32_t nbuffers) {
  const int32_t n_current_bufs = ArrowArrayVariadicBufferCount(array);
  const int32_t nvariadic_bufs_needed = n_current_bufs + nbuffers;

  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  private_data->variadic_buffers = (struct ArrowBuffer*)ArrowRealloc(
      private_data->variadic_buffers, sizeof(struct ArrowBuffer) * nvariadic_bufs_needed);
  if (private_data->variadic_buffers == NULL) {
    return ENOMEM;
  }
  private_data->variadic_buffer_sizes = (int64_t*)ArrowRealloc(
      private_data->variadic_buffer_sizes, sizeof(int64_t) * nvariadic_bufs_needed);
  if (private_data->variadic_buffer_sizes == NULL) {
    return ENOMEM;
  }

  for (int32_t i = n_current_bufs; i < nvariadic_bufs_needed; i++) {
    ArrowBufferInit(&private_data->variadic_buffers[i]);
    private_data->variadic_buffer_sizes[i] = 0;
  }
  private_data->n_variadic_buffers = nvariadic_bufs_needed;
  array->n_buffers = NANOARROW_BINARY_VIEW_FIXED_BUFFERS + 1 + nvariadic_bufs_needed;

  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayAppendBytes(struct ArrowArray* array,
                                                   struct ArrowBufferView value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  if (private_data->storage_type == NANOARROW_TYPE_STRING_VIEW ||
      private_data->storage_type == NANOARROW_TYPE_BINARY_VIEW) {
    struct ArrowBuffer* data_buffer = ArrowArrayBuffer(array, 1);
    union ArrowBinaryView bvt;
    bvt.inlined.size = (int32_t)value.size_bytes;

    if (value.size_bytes <= NANOARROW_BINARY_VIEW_INLINE_SIZE) {
      memcpy(bvt.inlined.data, value.data.as_char, value.size_bytes);
      memset(bvt.inlined.data + bvt.inlined.size, 0,
             NANOARROW_BINARY_VIEW_INLINE_SIZE - bvt.inlined.size);
    } else {
      int32_t current_n_vbufs = ArrowArrayVariadicBufferCount(array);
      if (current_n_vbufs == 0 ||
          private_data->variadic_buffers[current_n_vbufs - 1].size_bytes +
                  value.size_bytes >
              NANOARROW_BINARY_VIEW_BLOCK_SIZE) {
        const int32_t additional_bufs_needed = 1;
        NANOARROW_RETURN_NOT_OK(
            ArrowArrayAddVariadicBuffers(array, additional_bufs_needed));
        current_n_vbufs += additional_bufs_needed;
      }

      const int32_t buf_index = current_n_vbufs - 1;
      struct ArrowBuffer* variadic_buf = &private_data->variadic_buffers[buf_index];
      memcpy(bvt.ref.prefix, value.data.as_char, NANOARROW_BINARY_VIEW_PREFIX_SIZE);
      bvt.ref.buffer_index = (int32_t)buf_index;
      bvt.ref.offset = (int32_t)variadic_buf->size_bytes;
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppend(variadic_buf, value.data.as_char, value.size_bytes));
      private_data->variadic_buffer_sizes[buf_index] = variadic_buf->size_bytes;
    }
    NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(data_buffer, &bvt, sizeof(bvt)));
  } else {
    struct ArrowBuffer* offset_buffer = ArrowArrayBuffer(array, 1);
    struct ArrowBuffer* data_buffer = ArrowArrayBuffer(
        array, 1 + (private_data->storage_type != NANOARROW_TYPE_FIXED_SIZE_BINARY));
    int32_t offset;
    int64_t large_offset;
    int64_t fixed_size_bytes = private_data->layout.element_size_bits[1] / 8;

    switch (private_data->storage_type) {
      case NANOARROW_TYPE_STRING:
      case NANOARROW_TYPE_BINARY:
        offset = ((int32_t*)offset_buffer->data)[array->length];
        if ((((int64_t)offset) + value.size_bytes) > INT32_MAX) {
          return EOVERFLOW;
        }

        offset += (int32_t)value.size_bytes;
        NANOARROW_RETURN_NOT_OK(
            ArrowBufferAppend(offset_buffer, &offset, sizeof(int32_t)));
        NANOARROW_RETURN_NOT_OK(
            ArrowBufferAppend(data_buffer, value.data.data, value.size_bytes));
        break;

      case NANOARROW_TYPE_LARGE_STRING:
      case NANOARROW_TYPE_LARGE_BINARY:
        large_offset = ((int64_t*)offset_buffer->data)[array->length];
        large_offset += value.size_bytes;
        NANOARROW_RETURN_NOT_OK(
            ArrowBufferAppend(offset_buffer, &large_offset, sizeof(int64_t)));
        NANOARROW_RETURN_NOT_OK(
            ArrowBufferAppend(data_buffer, value.data.data, value.size_bytes));
        break;

      case NANOARROW_TYPE_FIXED_SIZE_BINARY:
        if (value.size_bytes != fixed_size_bytes) {
          return EINVAL;
        }

        NANOARROW_RETURN_NOT_OK(
            ArrowBufferAppend(data_buffer, value.data.data, value.size_bytes));
        break;
      default:
        return EINVAL;
    }
  }

  if (private_data->bitmap.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1));
  }

  array->length++;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayAppendString(struct ArrowArray* array,
                                                    struct ArrowStringView value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  struct ArrowBufferView buffer_view;
  buffer_view.data.data = value.data;
  buffer_view.size_bytes = value.size_bytes;

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_STRING_VIEW:
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_BINARY:
    case NANOARROW_TYPE_BINARY_VIEW:
      return ArrowArrayAppendBytes(array, buffer_view);
    default:
      return EINVAL;
  }
}

static inline ArrowErrorCode ArrowArrayAppendInterval(struct ArrowArray* array,
                                                      const struct ArrowInterval* value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  struct ArrowBuffer* data_buffer = ArrowArrayBuffer(array, 1);

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_INTERVAL_MONTHS: {
      if (value->type != NANOARROW_TYPE_INTERVAL_MONTHS) {
        return EINVAL;
      }

      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(data_buffer, value->months));
      break;
    }
    case NANOARROW_TYPE_INTERVAL_DAY_TIME: {
      if (value->type != NANOARROW_TYPE_INTERVAL_DAY_TIME) {
        return EINVAL;
      }

      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(data_buffer, value->days));
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(data_buffer, value->ms));
      break;
    }
    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO: {
      if (value->type != NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO) {
        return EINVAL;
      }

      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(data_buffer, value->months));
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(data_buffer, value->days));
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt64(data_buffer, value->ns));
      break;
    }
    default:
      return EINVAL;
  }

  if (private_data->bitmap.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1));
  }

  array->length++;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayAppendDecimal(struct ArrowArray* array,
                                                     const struct ArrowDecimal* value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  struct ArrowBuffer* data_buffer = ArrowArrayBuffer(array, 1);

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_DECIMAL128:
      if (value->n_words != 2) {
        return EINVAL;
      } else {
        NANOARROW_RETURN_NOT_OK(
            ArrowBufferAppend(data_buffer, value->words, 2 * sizeof(uint64_t)));
        break;
      }
    case NANOARROW_TYPE_DECIMAL256:
      if (value->n_words != 4) {
        return EINVAL;
      } else {
        NANOARROW_RETURN_NOT_OK(
            ArrowBufferAppend(data_buffer, value->words, 4 * sizeof(uint64_t)));
        break;
      }
    default:
      return EINVAL;
  }

  if (private_data->bitmap.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1));
  }

  array->length++;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayFinishElement(struct ArrowArray* array) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  int64_t child_length;

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_MAP:
      child_length = array->children[0]->length;
      if (child_length > INT32_MAX) {
        return EOVERFLOW;
      }
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppendInt32(ArrowArrayBuffer(array, 1), (int32_t)child_length));
      break;
    case NANOARROW_TYPE_LARGE_LIST:
      child_length = array->children[0]->length;
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppendInt64(ArrowArrayBuffer(array, 1), child_length));
      break;
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      child_length = array->children[0]->length;
      if (child_length !=
          ((array->length + 1) * private_data->layout.child_size_elements)) {
        return EINVAL;
      }
      break;
    case NANOARROW_TYPE_STRUCT:
      for (int64_t i = 0; i < array->n_children; i++) {
        child_length = array->children[i]->length;
        if (child_length != (array->length + 1)) {
          return EINVAL;
        }
      }
      break;
      return NANOARROW_OK;
    default:
      return EINVAL;
  }

  if (private_data->bitmap.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1));
  }

  array->length++;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayFinishUnionElement(struct ArrowArray* array,
                                                          int8_t type_id) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  int64_t child_index = _ArrowArrayUnionChildIndex(array, type_id);
  if (child_index < 0 || child_index >= array->n_children) {
    return EINVAL;
  }

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_DENSE_UNION:
      // Append the target child length to the union offsets buffer
      _NANOARROW_CHECK_RANGE(array->children[child_index]->length, 0, INT32_MAX);
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(
          ArrowArrayBuffer(array, 1), (int32_t)array->children[child_index]->length - 1));
      break;
    case NANOARROW_TYPE_SPARSE_UNION:
      // Append one empty to any non-target column that isn't already the right length
      // or abort if appending a null will result in a column with invalid length
      for (int64_t i = 0; i < array->n_children; i++) {
        if (i == child_index || array->children[i]->length == (array->length + 1)) {
          continue;
        }

        if (array->children[i]->length != array->length) {
          return EINVAL;
        }

        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendEmpty(array->children[i], 1));
      }

      break;
    default:
      return EINVAL;
  }

  // Write to the type_ids buffer
  NANOARROW_RETURN_NOT_OK(
      ArrowBufferAppendInt8(ArrowArrayBuffer(array, 0), (int8_t)type_id));
  array->length++;
  return NANOARROW_OK;
}

static inline void ArrowArrayViewMove(struct ArrowArrayView* src,
                                      struct ArrowArrayView* dst) {
  memcpy(dst, src, sizeof(struct ArrowArrayView));
  ArrowArrayViewInitFromType(src, NANOARROW_TYPE_UNINITIALIZED);
}

static inline int8_t ArrowArrayViewIsNull(const struct ArrowArrayView* array_view,
                                          int64_t i) {
  const uint8_t* validity_buffer = array_view->buffer_views[0].data.as_uint8;
  i += array_view->offset;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_NA:
      return 0x01;
    case NANOARROW_TYPE_DENSE_UNION:
    case NANOARROW_TYPE_SPARSE_UNION:
      // Unions are "never null" in Arrow land
      return 0x00;
    default:
      return validity_buffer != NULL && !ArrowBitGet(validity_buffer, i);
  }
}

static inline int64_t ArrowArrayViewComputeNullCount(
    const struct ArrowArrayView* array_view) {
  if (array_view->length == 0) {
    return 0;
  }

  switch (array_view->storage_type) {
    case NANOARROW_TYPE_NA:
      return array_view->length;
    case NANOARROW_TYPE_DENSE_UNION:
    case NANOARROW_TYPE_SPARSE_UNION:
      // Unions are "never null" in Arrow land
      return 0;
    default:
      break;
  }

  const uint8_t* validity_buffer = array_view->buffer_views[0].data.as_uint8;
  if (validity_buffer == NULL) {
    return 0;
  }
  return array_view->length -
         ArrowBitCountSet(validity_buffer, array_view->offset, array_view->length);
}

static inline int8_t ArrowArrayViewUnionTypeId(const struct ArrowArrayView* array_view,
                                               int64_t i) {
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_DENSE_UNION:
    case NANOARROW_TYPE_SPARSE_UNION:
      return array_view->buffer_views[0].data.as_int8[array_view->offset + i];
    default:
      return -1;
  }
}

static inline int8_t ArrowArrayViewUnionChildIndex(
    const struct ArrowArrayView* array_view, int64_t i) {
  int8_t type_id = ArrowArrayViewUnionTypeId(array_view, i);
  if (array_view->union_type_id_map == NULL) {
    return type_id;
  } else {
    return array_view->union_type_id_map[type_id];
  }
}

static inline int64_t ArrowArrayViewUnionChildOffset(
    const struct ArrowArrayView* array_view, int64_t i) {
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_DENSE_UNION:
      return array_view->buffer_views[1].data.as_int32[array_view->offset + i];
    case NANOARROW_TYPE_SPARSE_UNION:
      return array_view->offset + i;
    default:
      return -1;
  }
}

static inline int64_t ArrowArrayViewListChildOffset(
    const struct ArrowArrayView* array_view, int64_t i) {
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_LIST:
      return array_view->buffer_views[1].data.as_int32[i];
    case NANOARROW_TYPE_LARGE_LIST:
      return array_view->buffer_views[1].data.as_int64[i];
    default:
      return -1;
  }
}

static struct ArrowBufferView ArrowArrayViewGetBytesFromViewArrayUnsafe(
    const struct ArrowArrayView* array_view, int64_t i) {
  const union ArrowBinaryView* bv = &array_view->buffer_views[1].data.as_binary_view[i];
  struct ArrowBufferView out = {{NULL}, bv->inlined.size};
  if (bv->inlined.size <= NANOARROW_BINARY_VIEW_INLINE_SIZE) {
    out.data.as_uint8 = bv->inlined.data;
    return out;
  }

  const int32_t buf_index = bv->ref.buffer_index + NANOARROW_BINARY_VIEW_FIXED_BUFFERS;
  out.data.data = array_view->array->buffers[buf_index];
  out.data.as_uint8 += bv->ref.offset;
  return out;
}

static inline int64_t ArrowArrayViewGetIntUnsafe(const struct ArrowArrayView* array_view,
                                                 int64_t i) {
  const struct ArrowBufferView* data_view = &array_view->buffer_views[1];
  i += array_view->offset;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_INT64:
      return data_view->data.as_int64[i];
    case NANOARROW_TYPE_UINT64:
      return data_view->data.as_uint64[i];
    case NANOARROW_TYPE_INTERVAL_MONTHS:
    case NANOARROW_TYPE_INT32:
      return data_view->data.as_int32[i];
    case NANOARROW_TYPE_UINT32:
      return data_view->data.as_uint32[i];
    case NANOARROW_TYPE_INT16:
      return data_view->data.as_int16[i];
    case NANOARROW_TYPE_UINT16:
      return data_view->data.as_uint16[i];
    case NANOARROW_TYPE_INT8:
      return data_view->data.as_int8[i];
    case NANOARROW_TYPE_UINT8:
      return data_view->data.as_uint8[i];
    case NANOARROW_TYPE_DOUBLE:
      return (int64_t)data_view->data.as_double[i];
    case NANOARROW_TYPE_FLOAT:
      return (int64_t)data_view->data.as_float[i];
    case NANOARROW_TYPE_HALF_FLOAT:
      return (int64_t)ArrowHalfFloatToFloat(data_view->data.as_uint16[i]);
    case NANOARROW_TYPE_BOOL:
      return ArrowBitGet(data_view->data.as_uint8, i);
    default:
      return INT64_MAX;
  }
}

static inline uint64_t ArrowArrayViewGetUIntUnsafe(
    const struct ArrowArrayView* array_view, int64_t i) {
  i += array_view->offset;
  const struct ArrowBufferView* data_view = &array_view->buffer_views[1];
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_INT64:
      return data_view->data.as_int64[i];
    case NANOARROW_TYPE_UINT64:
      return data_view->data.as_uint64[i];
    case NANOARROW_TYPE_INTERVAL_MONTHS:
    case NANOARROW_TYPE_INT32:
      return data_view->data.as_int32[i];
    case NANOARROW_TYPE_UINT32:
      return data_view->data.as_uint32[i];
    case NANOARROW_TYPE_INT16:
      return data_view->data.as_int16[i];
    case NANOARROW_TYPE_UINT16:
      return data_view->data.as_uint16[i];
    case NANOARROW_TYPE_INT8:
      return data_view->data.as_int8[i];
    case NANOARROW_TYPE_UINT8:
      return data_view->data.as_uint8[i];
    case NANOARROW_TYPE_DOUBLE:
      return (uint64_t)data_view->data.as_double[i];
    case NANOARROW_TYPE_FLOAT:
      return (uint64_t)data_view->data.as_float[i];
    case NANOARROW_TYPE_HALF_FLOAT:
      return (uint64_t)ArrowHalfFloatToFloat(data_view->data.as_uint16[i]);
    case NANOARROW_TYPE_BOOL:
      return ArrowBitGet(data_view->data.as_uint8, i);
    default:
      return UINT64_MAX;
  }
}

static inline double ArrowArrayViewGetDoubleUnsafe(
    const struct ArrowArrayView* array_view, int64_t i) {
  i += array_view->offset;
  const struct ArrowBufferView* data_view = &array_view->buffer_views[1];
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_INT64:
      return (double)data_view->data.as_int64[i];
    case NANOARROW_TYPE_UINT64:
      return (double)data_view->data.as_uint64[i];
    case NANOARROW_TYPE_INT32:
      return data_view->data.as_int32[i];
    case NANOARROW_TYPE_UINT32:
      return data_view->data.as_uint32[i];
    case NANOARROW_TYPE_INT16:
      return data_view->data.as_int16[i];
    case NANOARROW_TYPE_UINT16:
      return data_view->data.as_uint16[i];
    case NANOARROW_TYPE_INT8:
      return data_view->data.as_int8[i];
    case NANOARROW_TYPE_UINT8:
      return data_view->data.as_uint8[i];
    case NANOARROW_TYPE_DOUBLE:
      return data_view->data.as_double[i];
    case NANOARROW_TYPE_FLOAT:
      return data_view->data.as_float[i];
    case NANOARROW_TYPE_HALF_FLOAT:
      return ArrowHalfFloatToFloat(data_view->data.as_uint16[i]);
    case NANOARROW_TYPE_BOOL:
      return ArrowBitGet(data_view->data.as_uint8, i);
    default:
      return DBL_MAX;
  }
}

static inline struct ArrowStringView ArrowArrayViewGetStringUnsafe(
    const struct ArrowArrayView* array_view, int64_t i) {
  i += array_view->offset;
  const struct ArrowBufferView* offsets_view = &array_view->buffer_views[1];
  const char* data_view = array_view->buffer_views[2].data.as_char;

  struct ArrowStringView view;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      view.data = data_view + offsets_view->data.as_int32[i];
      view.size_bytes =
          offsets_view->data.as_int32[i + 1] - offsets_view->data.as_int32[i];
      break;
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      view.data = data_view + offsets_view->data.as_int64[i];
      view.size_bytes =
          offsets_view->data.as_int64[i + 1] - offsets_view->data.as_int64[i];
      break;
    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
      view.size_bytes = array_view->layout.element_size_bits[1] / 8;
      view.data = array_view->buffer_views[1].data.as_char + (i * view.size_bytes);
      break;
    case NANOARROW_TYPE_STRING_VIEW:
    case NANOARROW_TYPE_BINARY_VIEW: {
      struct ArrowBufferView buf_view =
          ArrowArrayViewGetBytesFromViewArrayUnsafe(array_view, i);
      view.data = buf_view.data.as_char;
      view.size_bytes = buf_view.size_bytes;
      break;
    }
    default:
      view.data = NULL;
      view.size_bytes = 0;
      break;
  }

  return view;
}

static inline struct ArrowBufferView ArrowArrayViewGetBytesUnsafe(
    const struct ArrowArrayView* array_view, int64_t i) {
  i += array_view->offset;
  const struct ArrowBufferView* offsets_view = &array_view->buffer_views[1];
  const uint8_t* data_view = array_view->buffer_views[2].data.as_uint8;

  struct ArrowBufferView view;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      view.size_bytes =
          offsets_view->data.as_int32[i + 1] - offsets_view->data.as_int32[i];
      view.data.as_uint8 = data_view + offsets_view->data.as_int32[i];
      break;
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      view.size_bytes =
          offsets_view->data.as_int64[i + 1] - offsets_view->data.as_int64[i];
      view.data.as_uint8 = data_view + offsets_view->data.as_int64[i];
      break;
    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
      view.size_bytes = array_view->layout.element_size_bits[1] / 8;
      view.data.as_uint8 =
          array_view->buffer_views[1].data.as_uint8 + (i * view.size_bytes);
      break;
    case NANOARROW_TYPE_STRING_VIEW:
    case NANOARROW_TYPE_BINARY_VIEW:
      view = ArrowArrayViewGetBytesFromViewArrayUnsafe(array_view, i);
      break;
    default:
      view.data.data = NULL;
      view.size_bytes = 0;
      break;
  }

  return view;
}

static inline void ArrowArrayViewGetIntervalUnsafe(
    const struct ArrowArrayView* array_view, int64_t i, struct ArrowInterval* out) {
  const uint8_t* data_view = array_view->buffer_views[1].data.as_uint8;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_INTERVAL_MONTHS: {
      const size_t size = sizeof(int32_t);
      memcpy(&out->months, data_view + i * size, sizeof(int32_t));
      break;
    }
    case NANOARROW_TYPE_INTERVAL_DAY_TIME: {
      const size_t size = sizeof(int32_t) + sizeof(int32_t);
      memcpy(&out->days, data_view + i * size, sizeof(int32_t));
      memcpy(&out->ms, data_view + i * size + 4, sizeof(int32_t));
      break;
    }
    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO: {
      const size_t size = sizeof(int32_t) + sizeof(int32_t) + sizeof(int64_t);
      memcpy(&out->months, data_view + i * size, sizeof(int32_t));
      memcpy(&out->days, data_view + i * size + 4, sizeof(int32_t));
      memcpy(&out->ns, data_view + i * size + 8, sizeof(int64_t));
      break;
    }
    default:
      break;
  }
}

static inline void ArrowArrayViewGetDecimalUnsafe(const struct ArrowArrayView* array_view,
                                                  int64_t i, struct ArrowDecimal* out) {
  i += array_view->offset;
  const uint8_t* data_view = array_view->buffer_views[1].data.as_uint8;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_DECIMAL128:
      ArrowDecimalSetBytes(out, data_view + (i * 16));
      break;
    case NANOARROW_TYPE_DECIMAL256:
      ArrowDecimalSetBytes(out, data_view + (i * 32));
      break;
    default:
      memset(out->words, 0, sizeof(out->words));
      break;
  }
}

#ifdef __cplusplus
}
#endif

#endif
