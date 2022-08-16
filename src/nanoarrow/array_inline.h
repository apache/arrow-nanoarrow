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
#include <limits.h>
#include <stdint.h>
#include <string.h>

#include "bitmap_inline.h"
#include "buffer_inline.h"
#include "typedefs_inline.h"
#include "utils_inline.h"

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
  int result;

  for (int i = 0; i < 3; i++) {
    if (private_data->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_DATA_OFFSET &&
        private_data->layout.element_size_bits[i] == 64) {
      result = ArrowBufferAppendInt64(ArrowArrayBuffer(array, i), 0);
      if (result != NANOARROW_OK) {
        return result;
      }
    } else if (private_data->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_DATA_OFFSET &&
               private_data->layout.element_size_bits[i] == 32) {
      result = ArrowBufferAppendInt32(ArrowArrayBuffer(array, i), 0);
      if (result != NANOARROW_OK) {
        return result;
      }
    }
  }

  // Start building any child arrays
  for (int64_t i = 0; i < array->n_children; i++) {
    result = ArrowArrayStartAppending(array->children[i]);
    if (result != NANOARROW_OK) {
      return result;
    }
  }

  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayFinishBuilding(struct ArrowArray* array,
                                                      char shrink_to_fit) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  // Make sure the value we get with array->buffers[i] is set to the actual
  // pointer (which may have changed from the original due to reallocation)
  int result;
  for (int64_t i = 0; i < 3; i++) {
    struct ArrowBuffer* buffer = ArrowArrayBuffer(array, i);
    if (shrink_to_fit) {
      result = ArrowBufferResize(buffer, buffer->size_bytes, shrink_to_fit);
      if (result != NANOARROW_OK) {
        return result;
      }
    }

    private_data->buffer_data[i] = ArrowArrayBuffer(array, i)->data;
  }

  // Finish building any child arrays
  for (int64_t i = 0; i < array->n_children; i++) {
    result = ArrowArrayFinishBuilding(array->children[i], shrink_to_fit);
    if (result != NANOARROW_OK) {
      return result;
    }
  }

  // Check buffer sizes to make sure we are not sending an ArrowArray
  // into the wild that is going to segfault
  struct ArrowArrayView array_view;
  ArrowArrayViewInit(&array_view, private_data->storage_type);
  array_view.layout = private_data->layout;
  ArrowArrayViewSetLength(&array_view, array->length);

  for (int64_t i = 0; i < array->n_buffers; i++) {
    if (array_view.layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_VALIDITY &&
        array->null_count == 0 && array->buffers[i] == NULL) {
      continue;
    }

    if (array_view.buffer_views[i].n_bytes > ArrowArrayBuffer(array, i)->size_bytes) {
      return EINVAL;
    }
  }

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

  int result;

  // Append n 0 bits to the validity bitmap. If we haven't allocated a bitmap yet, do it
  // now
  if (private_data->bitmap.buffer.data == NULL) {
    result = ArrowBitmapReserve(&private_data->bitmap, array->length + n);
    if (result != NANOARROW_OK) {
      return result;
    }

    ArrowBitmapAppendUnsafe(&private_data->bitmap, 1, array->length);
    ArrowBitmapAppendUnsafe(&private_data->bitmap, 0, n);
  } else {
    result = ArrowBitmapReserve(&private_data->bitmap, n);
    if (result != NANOARROW_OK) {
      return result;
    }

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
        result = ArrowBufferReserve(buffer, size_bytes * n);
        if (result != NANOARROW_OK) {
          return result;
        }
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
          result = ArrowBufferAppendFill(buffer, 0, size_bytes * n);
          if (result != NANOARROW_OK) {
            return result;
          }
        } else {
          // TODO: handle booleans
          return EINVAL;
        }
        continue;

      case NANOARROW_BUFFER_TYPE_TYPE_ID:
      case NANOARROW_BUFFER_TYPE_UNION_OFFSET:
        // Not supported
        return EINVAL;
    }

    if (result != NANOARROW_OK) {
      return result;
    }
  }

  // For fixed-size list and struct we need to append some nulls to
  // children for the lengths to line up properly
  switch (private_data->storage_type) {
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      result = ArrowArrayAppendNull(array->children[0],
                                    n * private_data->layout.child_size_elements);
      if (result != NANOARROW_OK) {
        return result;
      }
      break;
    case NANOARROW_TYPE_STRUCT:
      for (int64_t i = 0; i < array->n_children; i++) {
        result = ArrowArrayAppendNull(array->children[i], n);
        if (result != NANOARROW_OK) {
          return result;
        }
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

  int result;
  struct ArrowBuffer* data_buffer = ArrowArrayBuffer(array, 1);

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_INT64:
      result = ArrowBufferAppend(data_buffer, &value, sizeof(int64_t));
      if (result != NANOARROW_OK) {
        return result;
      }
      break;
    case NANOARROW_TYPE_INT32:
      _NANOARROW_CHECK_RANGE(value, INT32_MIN, INT32_MAX);
      result = ArrowBufferAppendInt32(data_buffer, value);
      if (result != NANOARROW_OK) {
        return result;
      }
      break;
    case NANOARROW_TYPE_INT16:
      _NANOARROW_CHECK_RANGE(value, INT16_MIN, INT16_MAX);
      result = ArrowBufferAppendInt16(data_buffer, value);
      if (result != NANOARROW_OK) {
        return result;
      }
      break;
    case NANOARROW_TYPE_INT8:
      _NANOARROW_CHECK_RANGE(value, INT8_MIN, INT8_MAX);
      result = ArrowBufferAppendInt8(data_buffer, value);
      if (result != NANOARROW_OK) {
        return result;
      }
      break;
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_UINT8:
      _NANOARROW_CHECK_RANGE(value, 0, INT64_MAX);
      return ArrowArrayAppendUInt(array, value);
    default:
      return EINVAL;
  }

  if (private_data->bitmap.buffer.data != NULL) {
    result = ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1);
    if (result != NANOARROW_OK) {
      return result;
    }
  }

  array->length++;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayAppendUInt(struct ArrowArray* array,
                                                  uint64_t value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  int result;
  struct ArrowBuffer* data_buffer = ArrowArrayBuffer(array, 1);

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_UINT64:
      result = ArrowBufferAppend(data_buffer, &value, sizeof(uint64_t));
      if (result != NANOARROW_OK) {
        return result;
      }
      break;
    case NANOARROW_TYPE_UINT32:
      _NANOARROW_CHECK_RANGE(value, 0, UINT32_MAX);
      result = ArrowBufferAppendUInt32(data_buffer, value);
      if (result != NANOARROW_OK) {
        return result;
      }
      break;
    case NANOARROW_TYPE_UINT16:
      _NANOARROW_CHECK_RANGE(value, 0, UINT16_MAX);
      result = ArrowBufferAppendUInt16(data_buffer, value);
      if (result != NANOARROW_OK) {
        return result;
      }
      break;
    case NANOARROW_TYPE_UINT8:
      _NANOARROW_CHECK_RANGE(value, 0, UINT8_MAX);
      result = ArrowBufferAppendUInt8(data_buffer, value);
      if (result != NANOARROW_OK) {
        return result;
      }
      break;
    case NANOARROW_TYPE_INT64:
      _NANOARROW_CHECK_RANGE(value, 0, INT64_MAX);
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_INT8:
      return ArrowArrayAppendInt(array, value);
    default:
      return EINVAL;
  }

  if (private_data->bitmap.buffer.data != NULL) {
    result = ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1);
    if (result != NANOARROW_OK) {
      return result;
    }
  }

  array->length++;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayAppendDouble(struct ArrowArray* array,
                                                    double value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  int result;
  struct ArrowBuffer* data_buffer = ArrowArrayBuffer(array, 1);

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_DOUBLE:
      result = ArrowBufferAppend(data_buffer, &value, sizeof(double));
      if (result != NANOARROW_OK) {
        return result;
      }
      break;
    default:
      return EINVAL;
  }

  if (private_data->bitmap.buffer.data != NULL) {
    result = ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1);
    if (result != NANOARROW_OK) {
      return result;
    }
  }

  array->length++;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowArrayAppendBytes(struct ArrowArray* array,
                                                   struct ArrowBufferView value) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  int result;
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
      result = ArrowBufferAppend(offset_buffer, &offset, sizeof(int32_t));
      if (result != NANOARROW_OK) {
        return result;
      }
      result = ArrowBufferAppend(data_buffer, value.data.data, value.n_bytes);
      if (result != NANOARROW_OK) {
        return result;
      }
      break;

    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      large_offset = ((int64_t*)offset_buffer->data)[array->length];
      large_offset += value.n_bytes;
      result = ArrowBufferAppend(offset_buffer, &large_offset, sizeof(int64_t));
      if (result != NANOARROW_OK) {
        return result;
      }
      result = ArrowBufferAppend(data_buffer, value.data.data, value.n_bytes);
      if (result != NANOARROW_OK) {
        return result;
      }
      break;

    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
      if (value.n_bytes != fixed_size_bytes) {
        return EINVAL;
      }

      result = ArrowBufferAppend(data_buffer, value.data.data, value.n_bytes);
      if (result != NANOARROW_OK) {
        return result;
      }

      break;
    default:
      return EINVAL;
  }

  if (private_data->bitmap.buffer.data != NULL) {
    result = ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1);
    if (result != NANOARROW_OK) {
      return result;
    }
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
  int result;

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_LIST:
      child_length = array->children[0]->length;
      if (child_length > INT32_MAX) {
        return EINVAL;
      }
      result = ArrowBufferAppendInt32(ArrowArrayBuffer(array, 1), child_length);
      if (result != NANOARROW_OK) {
        return result;
      }
      break;
    case NANOARROW_TYPE_LARGE_LIST:
      child_length = array->children[0]->length;
      result = ArrowBufferAppendInt64(ArrowArrayBuffer(array, 1), child_length);
      if (result != NANOARROW_OK) {
        return result;
      }
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
    result = ArrowBitmapAppend(ArrowArrayValidityBitmap(array), 1, 1);
    if (result != NANOARROW_OK) {
      return result;
    }
  }

  array->length++;
  return NANOARROW_OK;
}

#ifdef __cplusplus
}
#endif

#endif
