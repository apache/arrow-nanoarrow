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

#include "buffer_inline.h"
#include "nanoarrow_types.h"

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

static inline ArrowErrorCode ArrowArrayStartAppending(struct ArrowArray* array) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  if (private_data->storage_type == NANOARROW_TYPE_UNINITIALIZED) {
    return EINVAL;
  }

  // Initialize any data offset buffer with a single zero
  for (int i = 0; i < 3; i++) {
    if (private_data->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_DATA_OFFSET &&
        private_data->layout.element_size_bits[i] == 64) {
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt64(ArrowArrayBuffer(array, i), 0));
    } else if (private_data->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_DATA_OFFSET &&
               private_data->layout.element_size_bits[i] == 32) {
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(ArrowArrayBuffer(array, i), 0));
    }
  }

  // Start building any child arrays
  for (int64_t i = 0; i < array->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayStartAppending(array->children[i]));
  }

  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayShrinkToFit(struct ArrowArray* array) {
  for (int64_t i = 0; i < 3; i++) {
    struct ArrowBuffer* buffer = ArrowArrayBuffer(array, i);
    NANOARROW_RETURN_NOT_OK(ArrowBufferResize(buffer, buffer->size_bytes, 1));
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayShrinkToFit(array->children[i]));
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

static inline ArrowErrorCode ArrowArrayAppendNull(struct ArrowArray* array, int64_t n) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  if (n == 0) {
    return NANOARROW_OK;
  }

  if (private_data->storage_type == NANOARROW_TYPE_NA) {
    array->null_count += n;
    array->length += n;
    return NANOARROW_OK;
  }

  // Append n 0 bits to the validity bitmap. If we haven't allocated a bitmap yet, do it
  // now
  if (private_data->bitmap.buffer.data == NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapReserve(&private_data->bitmap, array->length + n));
    ArrowBitmapAppendUnsafe(&private_data->bitmap, 1, array->length);
    ArrowBitmapAppendUnsafe(&private_data->bitmap, 0, n);
  } else {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapReserve(&private_data->bitmap, n));
    ArrowBitmapAppendUnsafe(&private_data->bitmap, 0, n);
  }

  // Add appropriate buffer fill
  struct ArrowBuffer* buffer;
  int64_t size_bytes;

  for (int i = 0; i < 3; i++) {
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
        // Zero out the next bit of memory
        if (private_data->layout.element_size_bits[i] % 8 == 0) {
          NANOARROW_RETURN_NOT_OK(ArrowBufferAppendFill(buffer, 0, size_bytes * n));
        } else {
          NANOARROW_RETURN_NOT_OK(_ArrowArrayAppendBits(array, i, 0, n));
        }
        continue;

      case NANOARROW_BUFFER_TYPE_TYPE_ID:
      case NANOARROW_BUFFER_TYPE_UNION_OFFSET:
        // Not supported
        return EINVAL;
    }
  }

  // For fixed-size list and struct we need to append some nulls to
  // children for the lengths to line up properly
  switch (private_data->storage_type) {
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      NANOARROW_RETURN_NOT_OK(ArrowArrayAppendNull(
          array->children[0], n * private_data->layout.child_size_elements));
      break;
    case NANOARROW_TYPE_STRUCT:
      for (int64_t i = 0; i < array->n_children; i++) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendNull(array->children[i], n));
      }
    default:
      break;
  }

  array->length += n;
  array->null_count += n;
  return NANOARROW_OK;
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
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendDouble(data_buffer, value));
      break;
    case NANOARROW_TYPE_FLOAT:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendFloat(data_buffer, value));
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
      _NANOARROW_CHECK_RANGE(value, 0, UINT32_MAX);
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt32(data_buffer, (uint32_t)value));
      break;
    case NANOARROW_TYPE_UINT16:
      _NANOARROW_CHECK_RANGE(value, 0, UINT16_MAX);
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt16(data_buffer, (uint16_t)value));
      break;
    case NANOARROW_TYPE_UINT8:
      _NANOARROW_CHECK_RANGE(value, 0, UINT8_MAX);
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendUInt8(data_buffer, (uint8_t)value));
      break;
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_INT8:
      _NANOARROW_CHECK_RANGE(value, 0, INT64_MAX);
      return ArrowArrayAppendInt(array, value);
    case NANOARROW_TYPE_DOUBLE:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendDouble(data_buffer, value));
      break;
    case NANOARROW_TYPE_FLOAT:
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendFloat(data_buffer, value));
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
    default:
      return EINVAL;
  }

  if (private_data->bitmap.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1));
  }

  array->length++;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayAppendBytes(struct ArrowArray* array,
                                                   struct ArrowBufferView value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

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
      if ((offset + value.n_bytes) > INT32_MAX) {
        return EINVAL;
      }

      offset += value.n_bytes;
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppend(offset_buffer, &offset, sizeof(int32_t)));
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppend(data_buffer, value.data.data, value.n_bytes));
      break;

    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      large_offset = ((int64_t*)offset_buffer->data)[array->length];
      large_offset += value.n_bytes;
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppend(offset_buffer, &large_offset, sizeof(int64_t)));
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppend(data_buffer, value.data.data, value.n_bytes));
      break;

    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
      if (value.n_bytes != fixed_size_bytes) {
        return EINVAL;
      }

      NANOARROW_RETURN_NOT_OK(
          ArrowBufferAppend(data_buffer, value.data.data, value.n_bytes));
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

static inline ArrowErrorCode ArrowArrayAppendString(struct ArrowArray* array,
                                                    struct ArrowStringView value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  struct ArrowBufferView buffer_view;
  buffer_view.data.data = value.data;
  buffer_view.n_bytes = value.n_bytes;

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
      return ArrowArrayAppendBytes(array, buffer_view);
    default:
      return EINVAL;
  }
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
        return EINVAL;
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
    default:
      return EINVAL;
  }

  if (private_data->bitmap.buffer.data != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1));
  }

  array->length++;
  return NANOARROW_OK;
}

static inline int8_t ArrowArrayViewIsNull(struct ArrowArrayView* array_view, int64_t i) {
  const uint8_t* validity_buffer = array_view->buffer_views[0].data.as_uint8;
  i += array_view->array->offset;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_NA:
      return 0x01;
    case NANOARROW_TYPE_DENSE_UNION:
    case NANOARROW_TYPE_SPARSE_UNION:
      // Not supported yet
      return -1;
    default:
      return validity_buffer != NULL && !ArrowBitGet(validity_buffer, i);
  }
}

static inline int64_t ArrowArrayViewGetIntUnsafe(struct ArrowArrayView* array_view,
                                                 int64_t i) {
  struct ArrowBufferView* data_view = &array_view->buffer_views[1];
  i += array_view->array->offset;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_INT64:
      return data_view->data.as_int64[i];
    case NANOARROW_TYPE_UINT64:
      return data_view->data.as_uint64[i];
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
    case NANOARROW_TYPE_BOOL:
      return ArrowBitGet(data_view->data.as_uint8, i);
    default:
      return INT64_MAX;
  }
}

static inline uint64_t ArrowArrayViewGetUIntUnsafe(struct ArrowArrayView* array_view,
                                                   int64_t i) {
  i += array_view->array->offset;
  struct ArrowBufferView* data_view = &array_view->buffer_views[1];
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_INT64:
      return data_view->data.as_int64[i];
    case NANOARROW_TYPE_UINT64:
      return data_view->data.as_uint64[i];
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
    case NANOARROW_TYPE_BOOL:
      return ArrowBitGet(data_view->data.as_uint8, i);
    default:
      return UINT64_MAX;
  }
}

static inline double ArrowArrayViewGetDoubleUnsafe(struct ArrowArrayView* array_view,
                                                   int64_t i) {
  i += array_view->array->offset;
  struct ArrowBufferView* data_view = &array_view->buffer_views[1];
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_INT64:
      return data_view->data.as_int64[i];
    case NANOARROW_TYPE_UINT64:
      return data_view->data.as_uint64[i];
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
    case NANOARROW_TYPE_BOOL:
      return ArrowBitGet(data_view->data.as_uint8, i);
    default:
      return DBL_MAX;
  }
}

static inline struct ArrowStringView ArrowArrayViewGetStringUnsafe(
    struct ArrowArrayView* array_view, int64_t i) {
  i += array_view->array->offset;
  struct ArrowBufferView* offsets_view = &array_view->buffer_views[1];
  const char* data_view = array_view->buffer_views[2].data.as_char;

  struct ArrowStringView view;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      view.data = data_view + offsets_view->data.as_int32[i];
      view.n_bytes = offsets_view->data.as_int32[i + 1] - offsets_view->data.as_int32[i];
      break;
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      view.data = data_view + offsets_view->data.as_int64[i];
      view.n_bytes = offsets_view->data.as_int64[i + 1] - offsets_view->data.as_int64[i];
      break;
    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
      view.n_bytes = array_view->layout.element_size_bits[1] / 8;
      view.data = array_view->buffer_views[1].data.as_char + (i * view.n_bytes);
      break;
    default:
      view.data = NULL;
      view.n_bytes = 0;
      break;
  }

  return view;
}

static inline struct ArrowBufferView ArrowArrayViewGetBytesUnsafe(
    struct ArrowArrayView* array_view, int64_t i) {
  i += array_view->array->offset;
  struct ArrowBufferView* offsets_view = &array_view->buffer_views[1];
  const uint8_t* data_view = array_view->buffer_views[2].data.as_uint8;

  struct ArrowBufferView view;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      view.n_bytes = offsets_view->data.as_int32[i + 1] - offsets_view->data.as_int32[i];
      view.data.as_uint8 = data_view + offsets_view->data.as_int32[i];
      break;
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      view.n_bytes = offsets_view->data.as_int64[i + 1] - offsets_view->data.as_int64[i];
      view.data.as_uint8 = data_view + offsets_view->data.as_int64[i];
      break;
    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
      view.n_bytes = array_view->layout.element_size_bits[1] / 8;
      view.data.as_uint8 = array_view->buffer_views[1].data.as_uint8 + (i * view.n_bytes);
      break;
    default:
      view.data.data = NULL;
      view.n_bytes = 0;
      break;
  }

  return view;
}

#ifdef __cplusplus
}
#endif

#endif
