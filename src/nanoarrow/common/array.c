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

#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanoarrow/nanoarrow.h"

static void ArrowArrayReleaseInternal(struct ArrowArray* array) {
  // Release buffers held by this array
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  if (private_data != NULL) {
    ArrowBitmapReset(&private_data->bitmap);
    ArrowBufferReset(&private_data->buffers[0]);
    ArrowBufferReset(&private_data->buffers[1]);
    ArrowFree(private_data->buffer_data);
    for (int32_t i = 0; i < private_data->n_variadic_buffers; ++i) {
      ArrowBufferReset(&private_data->variadic_buffers[i]);
    }
    ArrowFree(private_data->variadic_buffers);
    ArrowFree(private_data);
  }

  // This object owns the memory for all the children, but those
  // children may have been generated elsewhere and might have
  // their own release() callback.
  if (array->children != NULL) {
    for (int64_t i = 0; i < array->n_children; i++) {
      if (array->children[i] != NULL) {
        if (array->children[i]->release != NULL) {
          ArrowArrayRelease(array->children[i]);
        }

        ArrowFree(array->children[i]);
      }
    }

    ArrowFree(array->children);
  }

  // This object owns the memory for the dictionary but it
  // may have been generated somewhere else and have its own
  // release() callback.
  if (array->dictionary != NULL) {
    if (array->dictionary->release != NULL) {
      ArrowArrayRelease(array->dictionary);
    }

    ArrowFree(array->dictionary);
  }

  // Mark released
  array->release = NULL;
}

int ArrowArrayIsInternal(struct ArrowArray* array) {
  return array->release == &ArrowArrayReleaseInternal;
}

static int ArrowTypeIsSignedInteger(enum ArrowType type) {
  switch (type) {
    case NANOARROW_TYPE_BOOL:
    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_DATE32:
    case NANOARROW_TYPE_DATE64:
    case NANOARROW_TYPE_TIMESTAMP:
    case NANOARROW_TYPE_TIME32:
    case NANOARROW_TYPE_TIME64:
    case NANOARROW_TYPE_DURATION:
      return 1;
    default:
      return 0;
  }
}

static int ArrowTypeIsUnsignedInteger(enum ArrowType type) {
  switch (type) {
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_UINT64:
      return 1;
    default:
      return 0;
  }
}

static int ArrowTypeIsFloatingPoint(enum ArrowType type) {
  return type == NANOARROW_TYPE_HALF_FLOAT || type == NANOARROW_TYPE_FLOAT ||
         type == NANOARROW_TYPE_DOUBLE;
}

static int ArrowTypeIsBinaryLike(enum ArrowType type) {
  switch (type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
    case NANOARROW_TYPE_BINARY_VIEW:
    case NANOARROW_TYPE_STRING_VIEW:
      return 1;
    default:
      return 0;
  }
}

static int ArrowTypeIsListLike(enum ArrowType type) {
  switch (type) {
    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_LARGE_LIST:
    case NANOARROW_TYPE_MAP:
    case NANOARROW_TYPE_LIST_VIEW:
    case NANOARROW_TYPE_LARGE_LIST_VIEW:
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      return 1;
    default:
      return 0;
  }
}

static int ArrowArrayCanAppendStorageType(enum ArrowType dst_type,
                                          enum ArrowType src_type) {
  if (src_type == NANOARROW_TYPE_NA) {
    return 1;
  }

  if (ArrowTypeIsSignedInteger(src_type) || ArrowTypeIsUnsignedInteger(src_type)) {
    return ArrowTypeIsSignedInteger(dst_type) || ArrowTypeIsUnsignedInteger(dst_type) ||
           ArrowTypeIsFloatingPoint(dst_type);
  }

  if (ArrowTypeIsFloatingPoint(src_type)) {
    return ArrowTypeIsFloatingPoint(dst_type);
  }

  if (ArrowTypeIsBinaryLike(src_type)) {
    return ArrowTypeIsBinaryLike(dst_type);
  }

  if (ArrowTypeIsListLike(src_type)) {
    return ArrowTypeIsListLike(dst_type);
  }

  return dst_type == src_type;
}

static int ArrowArrayAppendNullMayReachRunEndEncoded(struct ArrowArray* dst) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)dst->private_data;

  switch (private_data->storage_type) {
    case NANOARROW_TYPE_RUN_END_ENCODED:
      return 1;
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      return private_data->layout.child_size_elements > 0 &&
             ArrowArrayAppendNullMayReachRunEndEncoded(dst->children[0]);
    case NANOARROW_TYPE_STRUCT:
    case NANOARROW_TYPE_SPARSE_UNION:
      for (int64_t i = 0; i < dst->n_children; i++) {
        if (ArrowArrayAppendNullMayReachRunEndEncoded(dst->children[i])) {
          return 1;
        }
      }
      return 0;
    case NANOARROW_TYPE_DENSE_UNION:
      return ArrowArrayAppendNullMayReachRunEndEncoded(dst->children[0]);
    default:
      return 0;
  }
}

static int ArrowArrayViewHasNulls(const struct ArrowArrayView* src) {
  if (src->length == 0) {
    return 0;
  }

  if (src->storage_type == NANOARROW_TYPE_NA) {
    return 1;
  }

  if (src->layout.buffer_type[0] != NANOARROW_BUFFER_TYPE_VALIDITY) {
    return 0;
  }

  const uint8_t* validity = src->buffer_views[0].data.as_uint8;
  return validity != NULL &&
         ArrowBitCountSet(validity, src->offset, src->length) != src->length;
}

static ArrowErrorCode ArrowArrayCheckCanAppendStorageFromArrayView(
    struct ArrowArray* dst, const struct ArrowArrayView* src, struct ArrowError* error) {
  if (src->offset < 0 || src->length < 0 || src->offset > INT64_MAX - src->length) {
    ArrowErrorSet(error,
                  "Expected source offset and length to describe a valid int64 "
                  "range but found %" PRId64 " and %" PRId64,
                  src->offset, src->length);
    return EINVAL;
  }

  if (!ArrowArrayIsInternal(dst)) {
    ArrowErrorSet(error, "Expected destination to be an internal ArrowArray");
    return EINVAL;
  }

  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)dst->private_data;
  enum ArrowType dst_type = private_data->storage_type;

  if (src->storage_type == NANOARROW_TYPE_DENSE_UNION ||
      src->storage_type == NANOARROW_TYPE_SPARSE_UNION) {
    ArrowErrorSet(error, "Appending array views is not supported for %s",
                  ArrowTypeString(src->storage_type));
    return ENOTSUP;
  }

  if (!ArrowArrayCanAppendStorageType(dst_type, src->storage_type)) {
    ArrowErrorSet(error, "Can't append %s storage to an array with %s storage",
                  ArrowTypeString(src->storage_type), ArrowTypeString(dst_type));
    return EINVAL;
  }

  if ((src->dictionary == NULL) != (dst->dictionary == NULL)) {
    ArrowErrorSet(error,
                  "Can't append storage when exactly one of source and destination "
                  "is dictionary-encoded");
    return EINVAL;
  }

  if (src->storage_type == NANOARROW_TYPE_NA) {
    return NANOARROW_OK;
  }

  if (src->n_children != dst->n_children) {
    ArrowErrorSet(error,
                  "Expected source and destination to have the same number of "
                  "children but found %" PRId64 " and %" PRId64,
                  src->n_children, dst->n_children);
    return EINVAL;
  }

  if (src->storage_type == NANOARROW_TYPE_FIXED_SIZE_LIST &&
      dst_type == NANOARROW_TYPE_FIXED_SIZE_LIST &&
      src->layout.child_size_elements != private_data->layout.child_size_elements) {
    ArrowErrorSet(error,
                  "Expected source and destination fixed-size list child sizes to "
                  "match but found %" PRId64 " and %" PRId64,
                  src->layout.child_size_elements,
                  private_data->layout.child_size_elements);
    return EINVAL;
  }

  if (src->storage_type == NANOARROW_TYPE_FIXED_SIZE_BINARY &&
      dst_type == NANOARROW_TYPE_FIXED_SIZE_BINARY &&
      src->layout.element_size_bits[1] != private_data->layout.element_size_bits[1]) {
    ArrowErrorSet(error,
                  "Expected source and destination fixed-size binary widths to "
                  "match but found %" PRId64 " and %" PRId64 " bits",
                  src->layout.element_size_bits[1],
                  private_data->layout.element_size_bits[1]);
    return EINVAL;
  }

  for (int64_t i = 0; i < src->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayCheckCanAppendStorageFromArrayView(
        dst->children[i], src->children[i], error));
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowArrayAppendStorageFromArrayViewRange(
    struct ArrowArray* dst, const struct ArrowArrayView* src, int64_t offset,
    int64_t length, struct ArrowError* error);

static ArrowErrorCode ArrowArrayAppendStorageFromArrayViewElement(
    struct ArrowArray* dst, const struct ArrowArrayView* src, int64_t i,
    struct ArrowError* error) {
  if (ArrowArrayViewIsNull(src, i)) {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowArrayAppendNull(dst, 1), error);
    return NANOARROW_OK;
  }

  switch (src->storage_type) {
    case NANOARROW_TYPE_NA:
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowArrayAppendNull(dst, 1), error);
      return NANOARROW_OK;
    case NANOARROW_TYPE_BOOL:
    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_DATE32:
    case NANOARROW_TYPE_DATE64:
    case NANOARROW_TYPE_TIMESTAMP:
    case NANOARROW_TYPE_TIME32:
    case NANOARROW_TYPE_TIME64:
    case NANOARROW_TYPE_DURATION:
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(
          ArrowArrayAppendInt(dst, ArrowArrayViewGetIntUnsafe(src, i)), error);
      return NANOARROW_OK;
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_UINT64:
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(
          ArrowArrayAppendUInt(dst, ArrowArrayViewGetUIntUnsafe(src, i)), error);
      return NANOARROW_OK;
    case NANOARROW_TYPE_HALF_FLOAT:
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_DOUBLE:
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(
          ArrowArrayAppendDouble(dst, ArrowArrayViewGetDoubleUnsafe(src, i)), error);
      return NANOARROW_OK;
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
    case NANOARROW_TYPE_BINARY_VIEW:
    case NANOARROW_TYPE_STRING_VIEW:
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(
          ArrowArrayAppendBytes(dst, ArrowArrayViewGetBytesUnsafe(src, i)), error);
      return NANOARROW_OK;
    case NANOARROW_TYPE_INTERVAL_MONTHS:
    case NANOARROW_TYPE_INTERVAL_DAY_TIME:
    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO: {
      struct ArrowInterval interval;
      ArrowIntervalInit(&interval, src->storage_type);
      ArrowArrayViewGetIntervalUnsafe(src, i, &interval);
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowArrayAppendInterval(dst, &interval), error);
      return NANOARROW_OK;
    }
    case NANOARROW_TYPE_STRUCT:
      for (int64_t child_i = 0; child_i < src->n_children; child_i++) {
        NANOARROW_RETURN_NOT_OK(ArrowArrayAppendStorageFromArrayViewRange(
            dst->children[child_i], src->children[child_i], src->offset + i, 1, error));
      }
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowArrayFinishElement(dst), error);
      return NANOARROW_OK;
    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_LARGE_LIST:
    case NANOARROW_TYPE_MAP:
    case NANOARROW_TYPE_LIST_VIEW:
    case NANOARROW_TYPE_LARGE_LIST_VIEW: {
      int64_t logical_i = src->offset + i;
      int64_t child_offset = ArrowArrayViewListChildOffset(src, logical_i);
      int64_t child_length;
      if (src->storage_type == NANOARROW_TYPE_LIST_VIEW) {
        child_length = src->buffer_views[2].data.as_int32[logical_i];
      } else if (src->storage_type == NANOARROW_TYPE_LARGE_LIST_VIEW) {
        child_length = src->buffer_views[2].data.as_int64[logical_i];
      } else {
        child_length = ArrowArrayViewListChildOffset(src, logical_i + 1) - child_offset;
      }

      NANOARROW_RETURN_NOT_OK(ArrowArrayAppendStorageFromArrayViewRange(
          dst->children[0], src->children[0], child_offset, child_length, error));
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowArrayFinishElement(dst), error);
      return NANOARROW_OK;
    }
    case NANOARROW_TYPE_FIXED_SIZE_LIST: {
      int64_t child_offset = (src->offset + i) * src->layout.child_size_elements;
      NANOARROW_RETURN_NOT_OK(ArrowArrayAppendStorageFromArrayViewRange(
          dst->children[0], src->children[0], child_offset,
          src->layout.child_size_elements, error));
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowArrayFinishElement(dst), error);
      return NANOARROW_OK;
    }
    case NANOARROW_TYPE_DECIMAL32:
    case NANOARROW_TYPE_DECIMAL64:
    case NANOARROW_TYPE_DECIMAL128:
    case NANOARROW_TYPE_DECIMAL256:
    case NANOARROW_TYPE_DENSE_UNION:
    case NANOARROW_TYPE_SPARSE_UNION:
    case NANOARROW_TYPE_RUN_END_ENCODED:
    case NANOARROW_TYPE_UNINITIALIZED:
    default:
      ArrowErrorSet(error, "Appending array views is not supported for %s",
                    ArrowTypeString(src->storage_type));
      return ENOTSUP;
  }
}

static int ArrowArrayCanAppendFixedWidthStorage(struct ArrowArray* dst,
                                                const struct ArrowArrayView* src) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)dst->private_data;
  return private_data->storage_type == src->storage_type && dst->n_buffers == 2 &&
         src->n_children == 0 &&
         src->layout.buffer_type[1] == NANOARROW_BUFFER_TYPE_DATA &&
         src->layout.element_size_bits[1] > 0 &&
         src->layout.element_size_bits[1] % 8 == 0;
}

static int ArrowArrayCanAppendStructStorage(struct ArrowArray* dst,
                                            const struct ArrowArrayView* src) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)dst->private_data;
  return private_data->storage_type == NANOARROW_TYPE_STRUCT &&
         src->storage_type == NANOARROW_TYPE_STRUCT;
}

static ArrowErrorCode ArrowArrayAppendValidityFromArrayView(
    struct ArrowArray* dst, const struct ArrowArrayView* src, struct ArrowError* error) {
  struct ArrowBitmap* dst_validity = ArrowArrayValidityBitmap(dst);
  const uint8_t* src_validity = src->buffer_views[0].data.as_uint8;
  if (src_validity == NULL && dst_validity->buffer.data == NULL) {
    return NANOARROW_OK;
  }

  if (dst_validity->buffer.data == NULL) {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBitmapAppend(dst_validity, 1, dst->length),
                                       error);
  }

  if (src_validity == NULL) {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBitmapAppend(dst_validity, 1, src->length),
                                       error);
    return NANOARROW_OK;
  }

  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBitmapReserve(dst_validity, src->length),
                                     error);
  int8_t validity[1024] = {0};
  for (int64_t offset = 0; offset < src->length;) {
    int64_t remaining = src->length - offset;
    int64_t chunk_size =
        remaining < (int64_t)sizeof(validity) ? remaining : (int64_t)sizeof(validity);
    ArrowBitsUnpackInt8(src_validity, src->offset + offset, chunk_size, validity);
    ArrowBitmapAppendInt8Unsafe(dst_validity, validity, chunk_size);
    offset += chunk_size;
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowArrayAppendFixedWidthStorageFromArrayView(
    struct ArrowArray* dst, const struct ArrowArrayView* src, struct ArrowError* error) {
  if (src->length == 0) {
    return NANOARROW_OK;
  }

  int64_t element_size_bytes = src->layout.element_size_bits[1] / 8;
  if (src->offset > INT64_MAX / element_size_bytes ||
      src->length > INT64_MAX / element_size_bytes) {
    ArrowErrorSet(error,
                  "Expected fixed-width append size to fit in int64 but found "
                  "element size %" PRId64 ", offset %" PRId64 ", and length %" PRId64,
                  element_size_bytes, src->offset, src->length);
    return EOVERFLOW;
  }

  int64_t src_offset_bytes = src->offset * element_size_bytes;
  int64_t src_size_bytes = src->length * element_size_bytes;
  if (src->buffer_views[1].data.as_uint8 == NULL ||
      src_offset_bytes > src->buffer_views[1].size_bytes ||
      src_size_bytes > src->buffer_views[1].size_bytes - src_offset_bytes) {
    ArrowErrorSet(error,
                  "Expected fixed-width source buffer to contain %" PRId64
                  " bytes at offset %" PRId64 " but its size is %" PRId64,
                  src_size_bytes, src_offset_bytes, src->buffer_views[1].size_bytes);
    return EINVAL;
  }

  NANOARROW_RETURN_NOT_OK(ArrowArrayAppendValidityFromArrayView(dst, src, error));

  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowBufferAppend(ArrowArrayBuffer(dst, 1),
                        src->buffer_views[1].data.as_uint8 + src_offset_bytes,
                        src_size_bytes),
      error);

  if (src->buffer_views[0].data.as_uint8 != NULL) {
    dst->null_count += src->length - ArrowBitCountSet(src->buffer_views[0].data.as_uint8,
                                                      src->offset, src->length);
  }
  dst->length += src->length;
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowArrayAppendStructStorageFromArrayView(
    struct ArrowArray* dst, const struct ArrowArrayView* src, struct ArrowError* error) {
  for (int64_t i = 0; i < src->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayAppendStorageFromArrayViewRange(
        dst->children[i], src->children[i], src->offset, src->length, error));
  }

  NANOARROW_RETURN_NOT_OK(ArrowArrayAppendValidityFromArrayView(dst, src, error));
  if (src->buffer_views[0].data.as_uint8 != NULL) {
    dst->null_count += src->length - ArrowBitCountSet(src->buffer_views[0].data.as_uint8,
                                                      src->offset, src->length);
  }
  dst->length += src->length;
  return NANOARROW_OK;
}

static int64_t ArrowArrayViewResolveRun(const struct ArrowArrayView* run_ends,
                                        int64_t logical_offset) {
  if (run_ends->length <= 1) {
    return 0;
  }

  switch (run_ends->storage_type) {
    case NANOARROW_TYPE_INT32:
      return ArrowResolveChunk32(
          (int32_t)(logical_offset + 1),
          run_ends->buffer_views[1].data.as_int32 + run_ends->offset, 0,
          (int32_t)run_ends->length);
    case NANOARROW_TYPE_INT64:
      return ArrowResolveChunk64(
          logical_offset + 1, run_ends->buffer_views[1].data.as_int64 + run_ends->offset,
          0, run_ends->length);
    case NANOARROW_TYPE_INT16:
      return ArrowResolveChunk16(
          (int16_t)(logical_offset + 1),
          run_ends->buffer_views[1].data.as_int16 + run_ends->offset, 0,
          (int16_t)run_ends->length);
    default:
      return 0;
  }
}

static ArrowErrorCode ArrowArrayAppendStorageFromArrayViewRange(
    struct ArrowArray* dst, const struct ArrowArrayView* src, int64_t offset,
    int64_t length, struct ArrowError* error) {
  // Note: src_slice must not be freed; it is a convenience to avoid repeatedly
  // composing offset and length on top of src in internal calls.
  struct ArrowArrayView src_slice = *src;
  if (offset < 0 || src->offset < 0 || offset > INT64_MAX - src->offset) {
    ArrowErrorSet(error,
                  "Expected source offset and range offset to describe a valid "
                  "int64 offset but found %" PRId64 " and %" PRId64,
                  src->offset, offset);
    return EINVAL;
  }

  src_slice.offset = src->offset + offset;
  src_slice.length = length;
  src = &src_slice;

  NANOARROW_RETURN_NOT_OK(ArrowArrayCheckCanAppendStorageFromArrayView(dst, src, error));

  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)dst->private_data;
  // Exact struct-to-struct appends copy child storage in bulk, including storage
  // beneath null parent slots. Other null appends use ArrowArrayAppendNull(), whose
  // empty child values cannot represent run-end encoded storage.
  if (!(src->storage_type == NANOARROW_TYPE_STRUCT &&
        private_data->storage_type == NANOARROW_TYPE_STRUCT) &&
      ArrowArrayAppendNullMayReachRunEndEncoded(dst) && ArrowArrayViewHasNulls(src)) {
    ArrowErrorSet(error,
                  "Can't append null storage to an array whose null representation "
                  "requires appending to run-end encoded storage");
    return EINVAL;
  }

  if (ArrowArrayCanAppendFixedWidthStorage(dst, src)) {
    return ArrowArrayAppendFixedWidthStorageFromArrayView(dst, src, error);
  }

  if (ArrowArrayCanAppendStructStorage(dst, src)) {
    return ArrowArrayAppendStructStorageFromArrayView(dst, src, error);
  }

  if (src->storage_type == NANOARROW_TYPE_RUN_END_ENCODED) {
    if (src->length == 0) {
      return NANOARROW_OK;
    }

    const struct ArrowArrayView* run_ends = src->children[0];
    if (run_ends->length == 0) {
      ArrowErrorSet(error,
                    "Expected a non-empty run ends array for a non-empty "
                    "run-end encoded array");
      return EINVAL;
    }

    if (dst->length > INT64_MAX - src->length) {
      ArrowErrorSet(error,
                    "Expected run-end encoded destination length plus source length "
                    "to fit in int64 but found %" PRId64 " and %" PRId64,
                    dst->length, src->length);
      return EOVERFLOW;
    }

    int64_t first_run = ArrowArrayViewResolveRun(run_ends, src->offset);
    int64_t slice_end = src->offset + src->length;
    int64_t run_end_offset = dst->length;
    for (int64_t i = first_run; i < run_ends->length; i++) {
      int64_t src_run_end = ArrowArrayViewGetIntUnsafe(run_ends, i);
      int64_t clipped_run_end = src_run_end < slice_end ? src_run_end : slice_end;
      int64_t dst_run_end = run_end_offset + clipped_run_end - src->offset;
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(
          ArrowArrayAppendInt(dst->children[0], dst_run_end), error);
      NANOARROW_RETURN_NOT_OK(ArrowArrayAppendStorageFromArrayViewRange(
          dst->children[1], src->children[1], i, 1, error));
      if (src_run_end >= slice_end) {
        break;
      }
    }
    dst->length += src->length;
    return NANOARROW_OK;
  }

  for (int64_t i = 0; i < src->length; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayAppendStorageFromArrayViewElement(dst, src, i, error));
  }
  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayAppendStorageFromArrayView(struct ArrowArray* dst,
                                                    const struct ArrowArrayView* src,
                                                    struct ArrowError* error) {
  return ArrowArrayAppendStorageFromArrayViewRange(dst, src, 0, src->length, error);
}

static ArrowErrorCode ArrowArraySetStorageType(struct ArrowArray* array,
                                               enum ArrowType storage_type) {
  switch (storage_type) {
    case NANOARROW_TYPE_UNINITIALIZED:
    case NANOARROW_TYPE_NA:
    case NANOARROW_TYPE_RUN_END_ENCODED:
      array->n_buffers = 0;
      break;

    case NANOARROW_TYPE_FIXED_SIZE_LIST:
    case NANOARROW_TYPE_STRUCT:
    case NANOARROW_TYPE_SPARSE_UNION:
      array->n_buffers = 1;
      break;

    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_LARGE_LIST:
    case NANOARROW_TYPE_MAP:
    case NANOARROW_TYPE_BOOL:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_HALF_FLOAT:
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_DOUBLE:
    case NANOARROW_TYPE_DECIMAL32:
    case NANOARROW_TYPE_DECIMAL64:
    case NANOARROW_TYPE_DECIMAL128:
    case NANOARROW_TYPE_DECIMAL256:
    case NANOARROW_TYPE_INTERVAL_MONTHS:
    case NANOARROW_TYPE_INTERVAL_DAY_TIME:
    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO:
    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
    case NANOARROW_TYPE_DENSE_UNION:
      array->n_buffers = 2;
      break;
    case NANOARROW_TYPE_BINARY_VIEW:
    case NANOARROW_TYPE_STRING_VIEW:
      array->n_buffers = NANOARROW_BINARY_VIEW_FIXED_BUFFERS + 1;
      break;
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_BINARY:
    case NANOARROW_TYPE_LIST_VIEW:
    case NANOARROW_TYPE_LARGE_LIST_VIEW:
      array->n_buffers = 3;
      break;

    default:
      return EINVAL;

      return NANOARROW_OK;
  }

  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  private_data->storage_type = storage_type;
  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayInitFromType(struct ArrowArray* array,
                                      enum ArrowType storage_type) {
  array->length = 0;
  array->null_count = 0;
  array->offset = 0;
  array->n_buffers = 0;
  array->n_children = 0;
  array->buffers = NULL;
  array->children = NULL;
  array->dictionary = NULL;
  array->release = &ArrowArrayReleaseInternal;
  array->private_data = NULL;

  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)ArrowMalloc(sizeof(struct ArrowArrayPrivateData));
  if (private_data == NULL) {
    array->release = NULL;
    return ENOMEM;
  }

  ArrowBitmapInit(&private_data->bitmap);
  ArrowBufferInit(&private_data->buffers[0]);
  ArrowBufferInit(&private_data->buffers[1]);
  private_data->buffer_data =
      (const void**)ArrowMalloc(sizeof(void*) * NANOARROW_MAX_FIXED_BUFFERS);
  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; ++i) {
    private_data->buffer_data[i] = NULL;
  }
  private_data->n_variadic_buffers = 0;
  private_data->variadic_buffers = NULL;
  private_data->list_view_offset = 0;

  array->private_data = private_data;
  array->buffers = (const void**)(private_data->buffer_data);

  // These are not technically "storage" in the sense that they do not appear
  // in the ArrowSchemaView's storage_type member; however, allowing them here
  // is helpful to maximize the number of types that can avoid going through
  // ArrowArrayInitFromSchema().
  switch (storage_type) {
    case NANOARROW_TYPE_DURATION:
    case NANOARROW_TYPE_TIMESTAMP:
    case NANOARROW_TYPE_TIME64:
    case NANOARROW_TYPE_DATE64:
      storage_type = NANOARROW_TYPE_INT64;
      break;
    case NANOARROW_TYPE_TIME32:
    case NANOARROW_TYPE_DATE32:
      storage_type = NANOARROW_TYPE_INT32;
      break;
    default:
      break;
  }

  int result = ArrowArraySetStorageType(array, storage_type);
  if (result != NANOARROW_OK) {
    ArrowArrayRelease(array);
    return result;
  }

  ArrowLayoutInit(&private_data->layout, storage_type);
  // We can only know this not to be true when initializing based on a schema
  // so assume this to be true.
  private_data->union_type_id_is_child_index = 1;
  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayInitFromArrayView(struct ArrowArray* array,
                                           const struct ArrowArrayView* array_view,
                                           struct ArrowError* error) {
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowArrayInitFromType(array, array_view->storage_type), error);
  int result;

  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  private_data->layout = array_view->layout;

  if (array_view->n_children > 0) {
    result = ArrowArrayAllocateChildren(array, array_view->n_children);
    if (result != NANOARROW_OK) {
      ArrowArrayRelease(array);
      return result;
    }

    for (int64_t i = 0; i < array_view->n_children; i++) {
      result =
          ArrowArrayInitFromArrayView(array->children[i], array_view->children[i], error);
      if (result != NANOARROW_OK) {
        ArrowArrayRelease(array);
        return result;
      }
    }
  }

  if (array_view->dictionary != NULL) {
    result = ArrowArrayAllocateDictionary(array);
    if (result != NANOARROW_OK) {
      ArrowArrayRelease(array);
      return result;
    }

    result =
        ArrowArrayInitFromArrayView(array->dictionary, array_view->dictionary, error);
    if (result != NANOARROW_OK) {
      ArrowArrayRelease(array);
      return result;
    }
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayInitFromSchema(struct ArrowArray* array,
                                        const struct ArrowSchema* schema,
                                        struct ArrowError* error) {
  struct ArrowArrayView array_view;
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewInitFromSchema(&array_view, schema, error));
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromArrayView(array, &array_view, error));
  if (array_view.storage_type == NANOARROW_TYPE_DENSE_UNION ||
      array_view.storage_type == NANOARROW_TYPE_SPARSE_UNION) {
    struct ArrowArrayPrivateData* private_data =
        (struct ArrowArrayPrivateData*)array->private_data;
    // We can still build arrays if this isn't true; however, the append
    // functions won't work. Instead, we store this value and error only
    // when StartAppending is called.
    private_data->union_type_id_is_child_index =
        _ArrowUnionTypeIdsWillEqualChildIndices(schema->format + 4, schema->n_children);
  }

  ArrowArrayViewReset(&array_view);
  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayAllocateChildren(struct ArrowArray* array, int64_t n_children) {
  if (array->children != NULL) {
    return EINVAL;
  }

  if (n_children == 0) {
    return NANOARROW_OK;
  }

  array->children =
      (struct ArrowArray**)ArrowMalloc(n_children * sizeof(struct ArrowArray*));
  if (array->children == NULL) {
    return ENOMEM;
  }

  memset(array->children, 0, n_children * sizeof(struct ArrowArray*));

  for (int64_t i = 0; i < n_children; i++) {
    array->children[i] = (struct ArrowArray*)ArrowMalloc(sizeof(struct ArrowArray));
    if (array->children[i] == NULL) {
      return ENOMEM;
    }
    array->children[i]->release = NULL;
  }

  array->n_children = n_children;
  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayAllocateDictionary(struct ArrowArray* array) {
  if (array->dictionary != NULL) {
    return EINVAL;
  }

  array->dictionary = (struct ArrowArray*)ArrowMalloc(sizeof(struct ArrowArray));
  if (array->dictionary == NULL) {
    return ENOMEM;
  }

  array->dictionary->release = NULL;
  return NANOARROW_OK;
}

void ArrowArraySetValidityBitmap(struct ArrowArray* array, struct ArrowBitmap* bitmap) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;
  ArrowBufferMove(&bitmap->buffer, &private_data->bitmap.buffer);
  private_data->bitmap.size_bits = bitmap->size_bits;
  bitmap->size_bits = 0;
  private_data->buffer_data[0] = private_data->bitmap.buffer.data;
  array->null_count = -1;
}

ArrowErrorCode ArrowArraySetBuffer(struct ArrowArray* array, int64_t i,
                                   struct ArrowBuffer* buffer) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  if (i >= array->n_buffers || i < 0) {
    return EINVAL;
  }

  // Find the `i`th buffer, release what is currently there, and move the
  // supplied buffer into that slot.
  struct ArrowBuffer* dst = ArrowArrayBuffer(array, i);
  ArrowBufferReset(dst);
  ArrowBufferMove(buffer, dst);

  // Flush the pointer into array->buffers. In theory clients should call
  // ArrowArrayFinishBuilding() to flush the pointer values before passing
  // this array elsewhere; however, in early nanoarrow versions this was not
  // needed and some code may depend on this being true.
  private_data->buffer_data[i] = dst->data;
  array->buffers = private_data->buffer_data;

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowArrayViewInitFromArray(struct ArrowArrayView* array_view,
                                                  struct ArrowArray* array,
                                                  struct ArrowError* error) {
  if (!ArrowArrayIsInternal(array)) {
    ArrowErrorSet(error,
                  "Can't initialize internal ArrowArrayView from external ArrowArray");
    return EINVAL;
  }

  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  ArrowArrayViewInitFromType(array_view, private_data->storage_type);
  array_view->layout = private_data->layout;
  array_view->array = array;
  array_view->length = array->length;
  array_view->offset = array->offset;
  array_view->null_count = array->null_count;

  array_view->buffer_views[0].data.as_uint8 = private_data->bitmap.buffer.data;
  array_view->buffer_views[0].size_bytes = private_data->bitmap.buffer.size_bytes;
  array_view->buffer_views[1].data.as_uint8 = private_data->buffers[0].data;
  array_view->buffer_views[1].size_bytes = private_data->buffers[0].size_bytes;
  array_view->buffer_views[2].data.as_uint8 = private_data->buffers[1].data;
  array_view->buffer_views[2].size_bytes = private_data->buffers[1].size_bytes;

  int result = ArrowArrayViewAllocateChildren(array_view, array->n_children);
  if (result != NANOARROW_OK) {
    ArrowArrayViewReset(array_view);
    return result;
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    result =
        ArrowArrayViewInitFromArray(array_view->children[i], array->children[i], error);
    if (result != NANOARROW_OK) {
      ArrowArrayViewReset(array_view);
      return result;
    }
  }

  if (array->dictionary != NULL) {
    result = ArrowArrayViewAllocateDictionary(array_view);
    if (result != NANOARROW_OK) {
      ArrowArrayViewReset(array_view);
      return result;
    }

    result =
        ArrowArrayViewInitFromArray(array_view->dictionary, array->dictionary, error);
    if (result != NANOARROW_OK) {
      ArrowArrayViewReset(array_view);
      return result;
    }
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowArrayReserveInternal(struct ArrowArray* array,
                                                struct ArrowArrayView* array_view) {
  // Loop through buffers and reserve the extra space that we know about
  for (int64_t i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    // Don't reserve on a validity buffer that hasn't been allocated yet
    if (array_view->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_VALIDITY &&
        ArrowArrayBuffer(array, i)->data == NULL) {
      continue;
    }

    int64_t additional_size_bytes =
        array_view->buffer_views[i].size_bytes - ArrowArrayBuffer(array, i)->size_bytes;

    if (additional_size_bytes > 0) {
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferReserve(ArrowArrayBuffer(array, i), additional_size_bytes));
    }
  }

  // Recursively reserve children
  for (int64_t i = 0; i < array->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayReserveInternal(array->children[i], array_view->children[i]));
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayReserve(struct ArrowArray* array,
                                 int64_t additional_size_elements) {
  struct ArrowArrayView array_view;
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewInitFromArray(&array_view, array, NULL));

  // Calculate theoretical buffer sizes (recursively)
  ArrowArrayViewSetLength(&array_view, array->length + additional_size_elements);

  // Walk the structure (recursively)
  int result = ArrowArrayReserveInternal(array, &array_view);
  ArrowArrayViewReset(&array_view);
  if (result != NANOARROW_OK) {
    return result;
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowArrayFinalizeBuffers(struct ArrowArray* array) {
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    if (private_data->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_VALIDITY ||
        private_data->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_NONE) {
      continue;
    }

    struct ArrowBuffer* buffer = ArrowArrayBuffer(array, i);
    if (buffer->data == NULL) {
      NANOARROW_RETURN_NOT_OK((ArrowBufferReserve(buffer, 1)));
    }
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    if (ArrowArrayIsInternal(array->children[i])) {
      NANOARROW_RETURN_NOT_OK(ArrowArrayFinalizeBuffers(array->children[i]));
    }
  }

  if (array->dictionary != NULL && ArrowArrayIsInternal(array->dictionary)) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayFinalizeBuffers(array->dictionary));
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowArrayFlushInternalPointers(struct ArrowArray* array) {
  NANOARROW_DCHECK(ArrowArrayIsInternal(array));
  struct ArrowArrayPrivateData* private_data =
      (struct ArrowArrayPrivateData*)array->private_data;

  if (array->n_buffers > NANOARROW_MAX_FIXED_BUFFERS) {
    // If the variadic sizes buffer was not set and there is at least one variadic
    // buffer, populate it now (if there are no variadic buffers there will be exactly
    // three total buffers and we don't need to do anything special here). Notably, this
    // will occur when building a BinaryView/StringView array by element using the
    // appender.
    struct ArrowBuffer* sizes_buffer = ArrowArrayBuffer(array, array->n_buffers - 1);
    if (sizes_buffer->data == NULL && sizes_buffer->size_bytes == 0) {
      NANOARROW_RETURN_NOT_OK(
          ArrowBufferReserve(sizes_buffer, private_data->n_variadic_buffers));
      for (int64_t i = 0; i < private_data->n_variadic_buffers; i++) {
        struct ArrowBuffer* variadic_buffer =
            ArrowArrayBuffer(array, i + NANOARROW_BINARY_VIEW_FIXED_BUFFERS);
        NANOARROW_RETURN_NOT_OK(
            ArrowBufferAppendInt64(sizes_buffer, variadic_buffer->size_bytes));
      }
    }
  }

  for (int32_t i = 0; i < array->n_buffers; i++) {
    private_data->buffer_data[i] = ArrowArrayBuffer(array, i)->data;
  }

  array->buffers = (const void**)(private_data->buffer_data);

  // Flush internal pointers for child/dictionary arrays if we allocated them. Clients
  // building arrays by buffer might have moved arrays from some other source (e.g.,
  // to create a record batch) and calling this function in that case will cause a crash.
  for (int64_t i = 0; i < array->n_children; i++) {
    if (ArrowArrayIsInternal(array->children[i])) {
      NANOARROW_RETURN_NOT_OK(ArrowArrayFlushInternalPointers(array->children[i]));
    }
  }

  if (array->dictionary != NULL && ArrowArrayIsInternal(array->dictionary)) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayFlushInternalPointers(array->dictionary));
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayFinishBuilding(struct ArrowArray* array,
                                        enum ArrowValidationLevel validation_level,
                                        struct ArrowError* error) {
  // Even if the data buffer is size zero, the pointer value needed to be non-null
  // in some implementations (at least one version of Arrow C++ at the time this
  // was added and C# as later discovered). Only do this fix if we can assume
  // CPU data access.
  if (validation_level >= NANOARROW_VALIDATION_LEVEL_DEFAULT) {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowArrayFinalizeBuffers(array), error);
  }

  // Make sure the value we get with array->buffers[i] is set to the actual
  // pointer (which may have changed from the original due to reallocation)
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowArrayFlushInternalPointers(array), error);

  if (validation_level == NANOARROW_VALIDATION_LEVEL_NONE) {
    return NANOARROW_OK;
  }

  // For validation, initialize an ArrowArrayView with our known buffer sizes
  struct ArrowArrayView array_view;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowArrayViewInitFromArray(&array_view, array, error), error);
  int result = ArrowArrayViewValidate(&array_view, validation_level, error);
  ArrowArrayViewReset(&array_view);
  return result;
}

ArrowErrorCode ArrowArrayFinishBuildingDefault(struct ArrowArray* array,
                                               struct ArrowError* error) {
  return ArrowArrayFinishBuilding(array, NANOARROW_VALIDATION_LEVEL_DEFAULT, error);
}

static int ArrowArrayIsShared(struct ArrowArray* array) {
  if (!ArrowArrayIsInternal(array)) {
    return 0;
  }

  for (int64_t i = 0; i < array->n_buffers; i++) {
    struct ArrowBuffer* buffer = ArrowArrayBuffer(array, i);
    if (buffer->data != NULL && !ArrowIsSharedBuffer(buffer)) {
      return 0;
    }
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    if (!ArrowArrayIsShared(array->children[i])) {
      return 0;
    }
  }

  if (array->dictionary != NULL && !ArrowArrayIsShared(array->dictionary)) {
    return 0;
  }

  return 1;
}

static ArrowErrorCode ArrowArrayMoveSharedInternal(struct ArrowArray* src,
                                                   struct ArrowArray* dst) {
  if (ArrowArrayIsShared(src)) {
    ArrowArrayMove(src, dst);
    return NANOARROW_OK;
  }

  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(dst, NANOARROW_TYPE_UNINITIALIZED));

  // Allocate children and move source children to dst children
  NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateChildren(dst, src->n_children));
  for (int64_t i = 0; i < src->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayMoveSharedInternal(src->children[i], dst->children[i]));
  }

  // Allocate dictionary if needed and move source dictionary to dst dictionary
  if (src->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateDictionary(dst));
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayMoveSharedInternal(src->dictionary, dst->dictionary));
  }

  // We might need some more buffers if we are shallowly copying a string/binary view
  if (src->n_buffers > 3) {
    if (src->n_buffers > INT_MAX) {
      return EINVAL;
    }

    NANOARROW_RETURN_NOT_OK(
        ArrowArrayAddVariadicBuffers(dst, (int32_t)src->n_buffers - 3));
  }

  // Move src into a shared array and set dst's buffers using the ref-counted version
  struct ArrowSharedArray shared_array;
  NANOARROW_RETURN_NOT_OK(ArrowSharedArrayInit(&shared_array, src));

  for (int64_t i = 0; i < src->n_buffers; i++) {
    struct ArrowBuffer* dst_buffer = ArrowArrayBuffer(dst, i);
    ArrowErrorCode result = ArrowSharedArrayBuffer(&shared_array, i, dst_buffer);
    if (result != NANOARROW_OK) {
      ArrowSharedArrayRelease(&shared_array);
      return result;
    }
  }

  ArrowSharedArrayRelease(&shared_array);

  dst->n_buffers = src->n_buffers;
  dst->length = src->length;
  dst->null_count = src->null_count;
  dst->offset = src->offset;

  // Flush internal buffer pointers to array->buffers
  NANOARROW_RETURN_NOT_OK(ArrowArrayFlushInternalPointers(dst));

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowArrayCloneSharedInternal(struct ArrowArray* src,
                                                    struct ArrowArray* dst) {
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(dst, NANOARROW_TYPE_UNINITIALIZED));

  // Allocate children and clone source children to dst children
  NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateChildren(dst, src->n_children));
  for (int64_t i = 0; i < src->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayCloneSharedInternal(src->children[i], dst->children[i]));
  }

  // Allocate dictionary if needed and clone source dictionary to dst dictionary
  if (src->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateDictionary(dst));
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayCloneSharedInternal(src->dictionary, dst->dictionary));
  }

  // We might need some more buffers if we are shallowly copying a string/binary view
  if (src->n_buffers > 3) {
    if (src->n_buffers > INT_MAX) {
      return EINVAL;
    }

    NANOARROW_RETURN_NOT_OK(
        ArrowArrayAddVariadicBuffers(dst, (int32_t)src->n_buffers - 3));
  }

  for (int64_t i = 0; i < src->n_buffers; i++) {
    struct ArrowBuffer* src_buffer = ArrowArrayBuffer(src, i);
    struct ArrowBuffer* dst_buffer = ArrowArrayBuffer(dst, i);
    NANOARROW_RETURN_NOT_OK(ArrowSharedBufferClone(src_buffer, dst_buffer));
  }

  dst->n_buffers = src->n_buffers;
  dst->length = src->length;
  dst->null_count = src->null_count;
  dst->offset = src->offset;

  // Flush internal buffer pointers to array->buffers
  NANOARROW_RETURN_NOT_OK(ArrowArrayFlushInternalPointers(dst));

  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayMoveShared(struct ArrowArray* array, struct ArrowArray* shared) {
  struct ArrowArray tmp;
  tmp.release = NULL;
  ArrowErrorCode result = ArrowArrayMoveSharedInternal(array, shared);
  if (result != NANOARROW_OK) {
    // On failure, release the temporary output
    if (tmp.release != NULL) {
      ArrowArrayRelease(&tmp);
    }

    // Because this operation may have partially moved the input array at this point
    // we also have to release it on failure to be predictable. These failures are
    // usually failed heap allocations and are difficult to trigger in practice.
    if (array->release != NULL) {
      ArrowArrayRelease(array);
    }
  }

  return result;
}

ArrowErrorCode ArrowArrayCloneShared(struct ArrowArray* shared,
                                     struct ArrowArray* array) {
  if (!ArrowArrayIsShared(shared)) {
    return EINVAL;
  }

  struct ArrowArray tmp;
  tmp.release = NULL;
  ArrowErrorCode result = ArrowArrayCloneSharedInternal(shared, array);
  if (result != NANOARROW_OK && tmp.release != NULL) {
    ArrowArrayRelease(&tmp);
  }

  return result;
}

void ArrowArrayViewInitFromType(struct ArrowArrayView* array_view,
                                enum ArrowType storage_type) {
  memset(array_view, 0, sizeof(struct ArrowArrayView));
  array_view->storage_type = storage_type;
  ArrowLayoutInit(&array_view->layout, storage_type);
}

ArrowErrorCode ArrowArrayViewAllocateChildren(struct ArrowArrayView* array_view,
                                              int64_t n_children) {
  if (array_view->children != NULL) {
    return EINVAL;
  }

  if (n_children == 0) {
    array_view->n_children = 0;
    return NANOARROW_OK;
  }

  array_view->children =
      (struct ArrowArrayView**)ArrowMalloc(n_children * sizeof(struct ArrowArrayView*));
  if (array_view->children == NULL) {
    return ENOMEM;
  }

  for (int64_t i = 0; i < n_children; i++) {
    array_view->children[i] = NULL;
  }

  array_view->n_children = n_children;

  for (int64_t i = 0; i < n_children; i++) {
    array_view->children[i] =
        (struct ArrowArrayView*)ArrowMalloc(sizeof(struct ArrowArrayView));
    if (array_view->children[i] == NULL) {
      return ENOMEM;
    }
    ArrowArrayViewInitFromType(array_view->children[i], NANOARROW_TYPE_UNINITIALIZED);
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayViewAllocateDictionary(struct ArrowArrayView* array_view) {
  if (array_view->dictionary != NULL) {
    return EINVAL;
  }

  array_view->dictionary =
      (struct ArrowArrayView*)ArrowMalloc(sizeof(struct ArrowArrayView));
  if (array_view->dictionary == NULL) {
    return ENOMEM;
  }

  ArrowArrayViewInitFromType(array_view->dictionary, NANOARROW_TYPE_UNINITIALIZED);
  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayViewInitFromSchema(struct ArrowArrayView* array_view,
                                            const struct ArrowSchema* schema,
                                            struct ArrowError* error) {
  struct ArrowSchemaView schema_view;
  int result = ArrowSchemaViewInit(&schema_view, schema, error);
  if (result != NANOARROW_OK) {
    return result;
  }

  ArrowArrayViewInitFromType(array_view, schema_view.storage_type);
  array_view->layout = schema_view.layout;

  result = ArrowArrayViewAllocateChildren(array_view, schema->n_children);
  if (result != NANOARROW_OK) {
    ArrowErrorSet(error, "ArrowArrayViewAllocateChildren() failed");
    ArrowArrayViewReset(array_view);
    return result;
  }

  for (int64_t i = 0; i < schema->n_children; i++) {
    result =
        ArrowArrayViewInitFromSchema(array_view->children[i], schema->children[i], error);
    if (result != NANOARROW_OK) {
      ArrowArrayViewReset(array_view);
      return result;
    }
  }

  if (schema->dictionary != NULL) {
    result = ArrowArrayViewAllocateDictionary(array_view);
    if (result != NANOARROW_OK) {
      ArrowArrayViewReset(array_view);
      return result;
    }

    result =
        ArrowArrayViewInitFromSchema(array_view->dictionary, schema->dictionary, error);
    if (result != NANOARROW_OK) {
      ArrowArrayViewReset(array_view);
      return result;
    }
  }

  if (array_view->storage_type == NANOARROW_TYPE_SPARSE_UNION ||
      array_view->storage_type == NANOARROW_TYPE_DENSE_UNION) {
    array_view->union_type_id_map = (int8_t*)ArrowMalloc(256 * sizeof(int8_t));
    if (array_view->union_type_id_map == NULL) {
      return ENOMEM;
    }

    memset(array_view->union_type_id_map, -1, 256);
    int32_t n_type_ids = _ArrowParseUnionTypeIds(schema_view.union_type_ids,
                                                 array_view->union_type_id_map + 128);
    for (int8_t child_index = 0; child_index < n_type_ids; child_index++) {
      int8_t type_id = array_view->union_type_id_map[128 + child_index];
      array_view->union_type_id_map[type_id] = child_index;
    }
  }

  return NANOARROW_OK;
}

void ArrowArrayViewReset(struct ArrowArrayView* array_view) {
  if (array_view->children != NULL) {
    for (int64_t i = 0; i < array_view->n_children; i++) {
      if (array_view->children[i] != NULL) {
        ArrowArrayViewReset(array_view->children[i]);
        ArrowFree(array_view->children[i]);
      }
    }

    ArrowFree(array_view->children);
  }

  if (array_view->dictionary != NULL) {
    ArrowArrayViewReset(array_view->dictionary);
    ArrowFree(array_view->dictionary);
  }

  if (array_view->union_type_id_map != NULL) {
    ArrowFree(array_view->union_type_id_map);
  }

  ArrowArrayViewInitFromType(array_view, NANOARROW_TYPE_UNINITIALIZED);
}

void ArrowArrayViewSetLength(struct ArrowArrayView* array_view, int64_t length) {
  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    int64_t element_size_bytes = array_view->layout.element_size_bits[i] / 8;

    switch (array_view->layout.buffer_type[i]) {
      case NANOARROW_BUFFER_TYPE_VALIDITY:
        array_view->buffer_views[i].size_bytes = _ArrowBytesForBits(length);
        continue;
      case NANOARROW_BUFFER_TYPE_DATA_OFFSET:
        // Probably don't want/need to rely on the producer to have allocated an
        // offsets buffer of length 1 for a zero-size array
        array_view->buffer_views[i].size_bytes =
            (length != 0) * element_size_bytes * (length + 1);
        continue;
      case NANOARROW_BUFFER_TYPE_DATA:
        array_view->buffer_views[i].size_bytes =
            _ArrowRoundUpToMultipleOf8(array_view->layout.element_size_bits[i] * length) /
            8;
        continue;
      case NANOARROW_BUFFER_TYPE_TYPE_ID:
      case NANOARROW_BUFFER_TYPE_UNION_OFFSET:
      case NANOARROW_BUFFER_TYPE_VIEW_OFFSET:
      case NANOARROW_BUFFER_TYPE_SIZE:
        array_view->buffer_views[i].size_bytes = element_size_bytes * length;
        continue;
      case NANOARROW_BUFFER_TYPE_VARIADIC_DATA:
      case NANOARROW_BUFFER_TYPE_VARIADIC_SIZE:
      case NANOARROW_BUFFER_TYPE_NONE:
        array_view->buffer_views[i].size_bytes = 0;
        continue;
    }
  }

  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRUCT:
    case NANOARROW_TYPE_SPARSE_UNION:
      for (int64_t i = 0; i < array_view->n_children; i++) {
        ArrowArrayViewSetLength(array_view->children[i], length);
      }
      break;
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      if (array_view->n_children >= 1) {
        ArrowArrayViewSetLength(array_view->children[0],
                                length * array_view->layout.child_size_elements);
      }
    default:
      break;
  }
}

// This version recursively extracts information from the array and stores it
// in the array view, performing any checks that require the original array.
static int ArrowArrayViewSetArrayInternal(struct ArrowArrayView* array_view,
                                          const struct ArrowArray* array,
                                          struct ArrowError* error) {
  array_view->array = array;
  array_view->offset = array->offset;
  array_view->length = array->length;
  array_view->null_count = array->null_count;
  array_view->variadic_buffer_sizes = NULL;
  array_view->variadic_buffers = NULL;
  array_view->n_variadic_buffers = 0;

  int64_t buffers_required = 0;
  const int nfixed_buf = array_view->storage_type == NANOARROW_TYPE_STRING_VIEW ||
                                 array_view->storage_type == NANOARROW_TYPE_BINARY_VIEW
                             ? NANOARROW_BINARY_VIEW_FIXED_BUFFERS
                             : NANOARROW_MAX_FIXED_BUFFERS;
  for (int i = 0; i < nfixed_buf; i++) {
    if (array_view->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_NONE) {
      break;
    }

    buffers_required++;

    // Set buffer pointer
    array_view->buffer_views[i].data.data = array->buffers[i];

    // If non-null, set buffer size to unknown.
    if (array->buffers[i] == NULL) {
      array_view->buffer_views[i].size_bytes = 0;
    } else {
      array_view->buffer_views[i].size_bytes = -1;
    }
  }

  if (array_view->storage_type == NANOARROW_TYPE_STRING_VIEW ||
      array_view->storage_type == NANOARROW_TYPE_BINARY_VIEW) {
    const int64_t n_buffers = array->n_buffers;
    const int32_t nfixed_buf = NANOARROW_BINARY_VIEW_FIXED_BUFFERS;

    const int32_t nvariadic_buf = (int32_t)(n_buffers - nfixed_buf - 1);
    array_view->n_variadic_buffers = nvariadic_buf;
    buffers_required += nvariadic_buf + 1;
    array_view->variadic_buffers = array->buffers + NANOARROW_BINARY_VIEW_FIXED_BUFFERS;
    array_view->variadic_buffer_sizes = (int64_t*)array->buffers[n_buffers - 1];
  }

  if (buffers_required != array->n_buffers) {
    ArrowErrorSet(error,
                  "Expected array with %" PRId64 " buffer(s) but found %" PRId64
                  " buffer(s)",
                  buffers_required, array->n_buffers);
    return EINVAL;
  }

  // Check number of children
  if (array_view->n_children != array->n_children) {
    ArrowErrorSet(error, "Expected %" PRId64 " children but found %" PRId64 " children",
                  array_view->n_children, array->n_children);
    return EINVAL;
  }

  // Recurse for children
  for (int64_t i = 0; i < array_view->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayViewSetArrayInternal(array_view->children[i],
                                                           array->children[i], error));
  }

  // Check dictionary
  if (array->dictionary == NULL && array_view->dictionary != NULL) {
    ArrowErrorSet(error, "Expected dictionary but found NULL");
    return EINVAL;
  }

  if (array->dictionary != NULL && array_view->dictionary == NULL) {
    ArrowErrorSet(error, "Expected NULL dictionary but found dictionary member");
    return EINVAL;
  }

  if (array->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewSetArrayInternal(array_view->dictionary, array->dictionary, error));
  }

  return NANOARROW_OK;
}

static int ArrowArrayViewValidateMinimal(struct ArrowArrayView* array_view,
                                         struct ArrowError* error) {
  if (array_view->length < 0) {
    ArrowErrorSet(error, "Expected length >= 0 but found length %" PRId64,
                  array_view->length);
    return EINVAL;
  }

  if (array_view->offset < 0) {
    ArrowErrorSet(error, "Expected offset >= 0 but found offset %" PRId64,
                  array_view->offset);
    return EINVAL;
  }

  // Ensure that offset + length fits within an int64 before a possible overflow
  if ((uint64_t)array_view->offset + (uint64_t)array_view->length > (uint64_t)INT64_MAX) {
    ArrowErrorSet(error, "Offset + length is > INT64_MAX");
    return EINVAL;
  }

  // Calculate buffer sizes that do not require buffer access. If marked as
  // unknown, assign the buffer size; otherwise, validate it.
  int64_t offset_plus_length = array_view->offset + array_view->length;

  // Only loop over the first two buffers because the size of the third buffer
  // is always data dependent for all current Arrow types.
  for (int i = 0; i < 2; i++) {
    int64_t element_size_bytes = array_view->layout.element_size_bits[i] / 8;
    // Initialize with a value that will cause an error if accidentally used uninitialized
    // Need to suppress the clang-tidy warning because gcc warns for possible use
    int64_t min_buffer_size_bytes =  // NOLINT(clang-analyzer-deadcode.DeadStores)
        array_view->buffer_views[i].size_bytes + 1;

    switch (array_view->layout.buffer_type[i]) {
      case NANOARROW_BUFFER_TYPE_VALIDITY:
        if (array_view->null_count == 0 && array_view->buffer_views[i].size_bytes == 0) {
          continue;
        }

        min_buffer_size_bytes = _ArrowBytesForBits(offset_plus_length);
        break;
      case NANOARROW_BUFFER_TYPE_SIZE:
        min_buffer_size_bytes = element_size_bytes * offset_plus_length;
        break;
      case NANOARROW_BUFFER_TYPE_DATA_OFFSET:
        // Probably don't want/need to rely on the producer to have allocated an
        // offsets buffer of length 1 for a zero-size array
        min_buffer_size_bytes =
            (offset_plus_length != 0) * element_size_bytes * (offset_plus_length + 1);
        break;
      case NANOARROW_BUFFER_TYPE_VIEW_OFFSET:
        min_buffer_size_bytes =
            (offset_plus_length != 0) * element_size_bytes * offset_plus_length;
        break;
      case NANOARROW_BUFFER_TYPE_DATA:
        min_buffer_size_bytes =
            _ArrowRoundUpToMultipleOf8(array_view->layout.element_size_bits[i] *
                                       offset_plus_length) /
            8;
        break;
      case NANOARROW_BUFFER_TYPE_TYPE_ID:
      case NANOARROW_BUFFER_TYPE_UNION_OFFSET:
        min_buffer_size_bytes = element_size_bytes * offset_plus_length;
        break;
      case NANOARROW_BUFFER_TYPE_VARIADIC_DATA:
      case NANOARROW_BUFFER_TYPE_VARIADIC_SIZE:
      case NANOARROW_BUFFER_TYPE_NONE:
        continue;
    }

    // Assign or validate buffer size
    if (array_view->buffer_views[i].size_bytes == -1) {
      array_view->buffer_views[i].size_bytes = min_buffer_size_bytes;
    } else if (array_view->buffer_views[i].size_bytes < min_buffer_size_bytes) {
      ArrowErrorSet(error,
                    "Expected %s array buffer %d to have size >= %" PRId64
                    " bytes but found "
                    "buffer with %" PRId64 " bytes",
                    ArrowTypeString(array_view->storage_type), i, min_buffer_size_bytes,
                    array_view->buffer_views[i].size_bytes);
      return EINVAL;
    }
  }

  // For list, fixed-size list and map views, we can validate the number of children
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_LARGE_LIST:
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
    case NANOARROW_TYPE_MAP:
    case NANOARROW_TYPE_LIST_VIEW:
    case NANOARROW_TYPE_LARGE_LIST_VIEW:
      if (array_view->n_children != 1) {
        ArrowErrorSet(error,
                      "Expected 1 child of %s array but found %" PRId64 " child arrays",
                      ArrowTypeString(array_view->storage_type), array_view->n_children);
        return EINVAL;
      }
      break;
    case NANOARROW_TYPE_RUN_END_ENCODED:
      if (array_view->n_children != 2) {
        ArrowErrorSet(
            error, "Expected 2 children for %s array but found %" PRId64 " child arrays",
            ArrowTypeString(array_view->storage_type), array_view->n_children);
        return EINVAL;
      }
      break;
    default:
      break;
  }

  // For struct, the sparse union, and the fixed-size list views, we can validate child
  // lengths.
  int64_t child_min_length;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_SPARSE_UNION:
    case NANOARROW_TYPE_STRUCT:
      child_min_length = (array_view->offset + array_view->length);
      for (int64_t i = 0; i < array_view->n_children; i++) {
        if (array_view->children[i]->length < child_min_length) {
          ArrowErrorSet(error,
                        "Expected struct child %" PRId64 " to have length >= %" PRId64
                        " but found child with "
                        "length %" PRId64,
                        i + 1, child_min_length, array_view->children[i]->length);
          return EINVAL;
        }
      }
      break;

    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      child_min_length = (array_view->offset + array_view->length) *
                         array_view->layout.child_size_elements;
      if (array_view->children[0]->length < child_min_length) {
        ArrowErrorSet(error,
                      "Expected child of fixed_size_list array to have length >= %" PRId64
                      " but "
                      "found array with length %" PRId64,
                      child_min_length, array_view->children[0]->length);
        return EINVAL;
      }
      break;

    case NANOARROW_TYPE_RUN_END_ENCODED: {
      if (array_view->n_children != 2) {
        ArrowErrorSet(error,
                      "Expected 2 children for run-end encoded array but found %" PRId64,
                      array_view->n_children);
        return EINVAL;
      }
      struct ArrowArrayView* run_ends_view = array_view->children[0];
      struct ArrowArrayView* values_view = array_view->children[1];
      int64_t max_length;
      switch (run_ends_view->storage_type) {
        case NANOARROW_TYPE_INT16:
          max_length = INT16_MAX;
          break;
        case NANOARROW_TYPE_INT32:
          max_length = INT32_MAX;
          break;
        case NANOARROW_TYPE_INT64:
          max_length = INT64_MAX;
          break;
        default:
          ArrowErrorSet(
              error,
              "Run-end encoded array only supports INT16, INT32 or INT64 run-ends "
              "but found run-ends type %s",
              ArrowTypeString(run_ends_view->storage_type));
          return EINVAL;
      }

      // There is already a check above that offset_plus_length < INT64_MAX
      if (offset_plus_length > max_length) {
        ArrowErrorSet(error,
                      "Offset + length of a run-end encoded array must fit in a value"
                      " of the run end type %s but is %" PRId64 " + %" PRId64,
                      ArrowTypeString(run_ends_view->storage_type), array_view->offset,
                      array_view->length);
        return EINVAL;
      }

      if (run_ends_view->length > values_view->length) {
        ArrowErrorSet(error,
                      "Length of run_ends is greater than the length of values: %" PRId64
                      " > %" PRId64,
                      run_ends_view->length, values_view->length);
        return EINVAL;
      }

      if (run_ends_view->length == 0 && values_view->length != 0) {
        ArrowErrorSet(error,
                      "Run-end encoded array has zero length %" PRId64
                      ", but values array has "
                      "non-zero length",
                      values_view->length);
        return EINVAL;
      }

      if (run_ends_view->null_count != 0) {
        ArrowErrorSet(error, "Null count must be 0 for run ends array, but is %" PRId64,
                      run_ends_view->null_count);
        return EINVAL;
      }
      break;
    }

    default:
      break;
  }

  // Recurse for children
  for (int64_t i = 0; i < array_view->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewValidateMinimal(array_view->children[i], error));
  }

  // Recurse for dictionary
  if (array_view->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayViewValidateMinimal(array_view->dictionary, error));
  }

  return NANOARROW_OK;
}

static int ArrowArrayViewValidateDefault(struct ArrowArrayView* array_view,
                                         struct ArrowError* error) {
  // Perform minimal validation. This will validate or assign
  // buffer sizes as long as buffer access is not required.
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewValidateMinimal(array_view, error));

  // Calculate buffer sizes or child lengths that require accessing the offsets
  // buffer. Where appropriate, validate that the first offset is >= 0.
  // If a buffer size is marked as unknown, assign it; otherwise, validate it.
  int64_t offset_plus_length = array_view->offset + array_view->length;

  int64_t first_offset;
  int64_t last_offset;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      if (array_view->buffer_views[1].size_bytes != 0) {
        first_offset = array_view->buffer_views[1].data.as_int32[array_view->offset];
        if (first_offset < 0) {
          ArrowErrorSet(error, "Expected first offset >= 0 but found %" PRId64,
                        first_offset);
          return EINVAL;
        }

        last_offset = array_view->buffer_views[1].data.as_int32[offset_plus_length];
        if (last_offset < 0) {
          ArrowErrorSet(error, "Expected last offset >= 0 but found %" PRId64,
                        last_offset);
          return EINVAL;
        }

        // If the data buffer size is unknown, assign it; otherwise, check it
        if (array_view->buffer_views[2].size_bytes == -1) {
          array_view->buffer_views[2].size_bytes = last_offset;
        } else if (array_view->buffer_views[2].size_bytes < last_offset) {
          ArrowErrorSet(error,
                        "Expected %s array buffer 2 to have size >= %" PRId64
                        " bytes but found "
                        "buffer with %" PRId64 " bytes",
                        ArrowTypeString(array_view->storage_type), last_offset,
                        array_view->buffer_views[2].size_bytes);
          return EINVAL;
        }
      } else if (array_view->buffer_views[2].size_bytes == -1) {
        // If the data buffer size is unknown and there are no bytes in the offset buffer,
        // set the data buffer size to 0.
        array_view->buffer_views[2].size_bytes = 0;
      }
      break;

    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      if (array_view->buffer_views[1].size_bytes != 0) {
        first_offset = array_view->buffer_views[1].data.as_int64[array_view->offset];
        if (first_offset < 0) {
          ArrowErrorSet(error, "Expected first offset >= 0 but found %" PRId64,
                        first_offset);
          return EINVAL;
        }

        last_offset = array_view->buffer_views[1].data.as_int64[offset_plus_length];
        if (last_offset < 0) {
          ArrowErrorSet(error, "Expected last offset >= 0 but found %" PRId64,
                        last_offset);
          return EINVAL;
        }

        // If the data buffer size is unknown, assign it; otherwise, check it
        if (array_view->buffer_views[2].size_bytes == -1) {
          array_view->buffer_views[2].size_bytes = last_offset;
        } else if (array_view->buffer_views[2].size_bytes < last_offset) {
          ArrowErrorSet(error,
                        "Expected %s array buffer 2 to have size >= %" PRId64
                        " bytes but found "
                        "buffer with %" PRId64 " bytes",
                        ArrowTypeString(array_view->storage_type), last_offset,
                        array_view->buffer_views[2].size_bytes);
          return EINVAL;
        }
      } else if (array_view->buffer_views[2].size_bytes == -1) {
        // If the data buffer size is unknown and there are no bytes in the offset
        // buffer, set the data buffer size to 0.
        array_view->buffer_views[2].size_bytes = 0;
      }
      break;

    case NANOARROW_TYPE_STRUCT:
      for (int64_t i = 0; i < array_view->n_children; i++) {
        if (array_view->children[i]->length < offset_plus_length) {
          ArrowErrorSet(error,
                        "Expected struct child %" PRId64 " to have length >= %" PRId64
                        " but found child with "
                        "length %" PRId64,
                        i + 1, offset_plus_length, array_view->children[i]->length);
          return EINVAL;
        }
      }
      break;

    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_MAP:
      if (array_view->buffer_views[1].size_bytes != 0) {
        first_offset = array_view->buffer_views[1].data.as_int32[array_view->offset];
        if (first_offset < 0) {
          ArrowErrorSet(error, "Expected first offset >= 0 but found %" PRId64,
                        first_offset);
          return EINVAL;
        }

        last_offset = array_view->buffer_views[1].data.as_int32[offset_plus_length];
        if (last_offset < 0) {
          ArrowErrorSet(error, "Expected last offset >= 0 but found %" PRId64,
                        last_offset);
          return EINVAL;
        }

        if (array_view->children[0]->length < last_offset) {
          ArrowErrorSet(error,
                        "Expected child of %s array to have length >= %" PRId64
                        " but found array with "
                        "length %" PRId64,
                        ArrowTypeString(array_view->storage_type), last_offset,
                        array_view->children[0]->length);
          return EINVAL;
        }
      }
      break;

    case NANOARROW_TYPE_LARGE_LIST:
      if (array_view->buffer_views[1].size_bytes != 0) {
        first_offset = array_view->buffer_views[1].data.as_int64[array_view->offset];
        if (first_offset < 0) {
          ArrowErrorSet(error, "Expected first offset >= 0 but found %" PRId64,
                        first_offset);
          return EINVAL;
        }

        last_offset = array_view->buffer_views[1].data.as_int64[offset_plus_length];
        if (last_offset < 0) {
          ArrowErrorSet(error, "Expected last offset >= 0 but found %" PRId64,
                        last_offset);
          return EINVAL;
        }

        if (array_view->children[0]->length < last_offset) {
          ArrowErrorSet(error,
                        "Expected child of %s array to have length >= %" PRId64
                        " but found array "
                        "with length %" PRId64,
                        ArrowTypeString(array_view->storage_type), last_offset,
                        array_view->children[0]->length);
          return EINVAL;
        }
      }
      break;

    case NANOARROW_TYPE_RUN_END_ENCODED: {
      struct ArrowArrayView* run_ends_view = array_view->children[0];
      if (run_ends_view->length == 0) {
        break;
      }

      int64_t first_run_end = ArrowArrayViewGetIntUnsafe(run_ends_view, 0);
      if (first_run_end < 1) {
        ArrowErrorSet(
            error,
            "All run ends must be greater than 0 but the first run end is %" PRId64,
            first_run_end);
        return EINVAL;
      }

      // offset + length < INT64_MAX is checked in ArrowArrayViewValidateMinimal()
      int64_t last_run_end =
          ArrowArrayViewGetIntUnsafe(run_ends_view, run_ends_view->length - 1);
      if (last_run_end < offset_plus_length) {
        ArrowErrorSet(error,
                      "Last run end is %" PRId64 " but it should be >= (%" PRId64
                      " + %" PRId64 ")",
                      last_run_end, array_view->offset, array_view->length);
        return EINVAL;
      }
      break;
    }
    default:
      break;
  }

  // Recurse for children
  for (int64_t i = 0; i < array_view->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewValidateDefault(array_view->children[i], error));
  }

  // Recurse for dictionary
  if (array_view->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayViewValidateDefault(array_view->dictionary, error));
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayViewSetArray(struct ArrowArrayView* array_view,
                                      const struct ArrowArray* array,
                                      struct ArrowError* error) {
  // Extract information from the array into the array view
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewSetArrayInternal(array_view, array, error));

  // Run default validation. Because we've marked all non-NULL buffers as having unknown
  // size, validation will also update the buffer sizes as it goes.
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewValidateDefault(array_view, error));

  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayViewSetArrayMinimal(struct ArrowArrayView* array_view,
                                             const struct ArrowArray* array,
                                             struct ArrowError* error) {
  // Extract information from the array into the array view
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewSetArrayInternal(array_view, array, error));

  // Run default validation. Because we've marked all non-NULL buffers as having unknown
  // size, validation will also update the buffer sizes as it goes.
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewValidateMinimal(array_view, error));

  return NANOARROW_OK;
}

static int ArrowAssertIncreasingInt32(struct ArrowBufferView view,
                                      struct ArrowError* error) {
  if (view.size_bytes <= (int64_t)sizeof(int32_t)) {
    return NANOARROW_OK;
  }

  for (int64_t i = 1; i < view.size_bytes / (int64_t)sizeof(int32_t); i++) {
    if (view.data.as_int32[i] < view.data.as_int32[i - 1]) {
      ArrowErrorSet(error, "[%" PRId64 "] Expected element size >= 0", i);
      return EINVAL;
    }
  }

  return NANOARROW_OK;
}

static int ArrowAssertIncreasingInt64(struct ArrowBufferView view,
                                      struct ArrowError* error) {
  if (view.size_bytes <= (int64_t)sizeof(int64_t)) {
    return NANOARROW_OK;
  }

  for (int64_t i = 1; i < view.size_bytes / (int64_t)sizeof(int64_t); i++) {
    if (view.data.as_int64[i] < view.data.as_int64[i - 1]) {
      ArrowErrorSet(error, "[%" PRId64 "] Expected element size >= 0", i);
      return EINVAL;
    }
  }

  return NANOARROW_OK;
}

static int ArrowAssertRangeInt8(struct ArrowBufferView view, int8_t min_value,
                                int8_t max_value, struct ArrowError* error) {
  for (int64_t i = 0; i < view.size_bytes; i++) {
    if (view.data.as_int8[i] < min_value || view.data.as_int8[i] > max_value) {
      ArrowErrorSet(error,
                    "[%" PRId64 "] Expected buffer value between %" PRId8 " and %" PRId8
                    " but found value %" PRId8,
                    i, min_value, max_value, view.data.as_int8[i]);
      return EINVAL;
    }
  }

  return NANOARROW_OK;
}

static int ArrowAssertInt8In(struct ArrowBufferView view, const int8_t* values,
                             int64_t n_values, struct ArrowError* error) {
  for (int64_t i = 0; i < view.size_bytes; i++) {
    int item_found = 0;
    for (int64_t j = 0; j < n_values; j++) {
      if (view.data.as_int8[i] == values[j]) {
        item_found = 1;
        break;
      }
    }

    if (!item_found) {
      ArrowErrorSet(error, "[%" PRId64 "] Unexpected buffer value %" PRId8, i,
                    view.data.as_int8[i]);
      return EINVAL;
    }
  }

  return NANOARROW_OK;
}

static int ArrowArrayViewValidateFull(struct ArrowArrayView* array_view,
                                      struct ArrowError* error) {
  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    switch (array_view->layout.buffer_type[i]) {
      // Only validate the portion of the buffer that is strictly required,
      // which includes not validating the offset buffer of a zero-length array.
      case NANOARROW_BUFFER_TYPE_DATA_OFFSET:
        if (array_view->length == 0) {
          continue;
        }
        if (array_view->layout.element_size_bits[i] == 32) {
          struct ArrowBufferView sliced_offsets;
          sliced_offsets.data.as_int32 =
              array_view->buffer_views[i].data.as_int32 + array_view->offset;
          sliced_offsets.size_bytes = (array_view->length + 1) * sizeof(int32_t);
          NANOARROW_RETURN_NOT_OK(ArrowAssertIncreasingInt32(sliced_offsets, error));
        } else {
          struct ArrowBufferView sliced_offsets;
          sliced_offsets.data.as_int64 =
              array_view->buffer_views[i].data.as_int64 + array_view->offset;
          sliced_offsets.size_bytes = (array_view->length + 1) * sizeof(int64_t);
          NANOARROW_RETURN_NOT_OK(ArrowAssertIncreasingInt64(sliced_offsets, error));
        }
        break;
      default:
        break;
    }
  }

  if (array_view->storage_type == NANOARROW_TYPE_DENSE_UNION ||
      array_view->storage_type == NANOARROW_TYPE_SPARSE_UNION) {
    struct ArrowBufferView sliced_type_ids;
    sliced_type_ids.size_bytes = array_view->length * sizeof(int8_t);
    if (array_view->length > 0) {
      sliced_type_ids.data.as_int8 =
          array_view->buffer_views[0].data.as_int8 + array_view->offset;
    } else {
      sliced_type_ids.data.as_int8 = NULL;
    }

    if (array_view->union_type_id_map == NULL) {
      // If the union_type_id map is NULL (e.g., when using ArrowArrayInitFromType() +
      // ArrowArrayAllocateChildren() + ArrowArrayFinishBuilding()), we don't have enough
      // information to validate this buffer.
      ArrowErrorSet(error,
                    "Insufficient information provided for validation of union array");
      return EINVAL;
    } else if (_ArrowParsedUnionTypeIdsWillEqualChildIndices(
                   array_view->union_type_id_map, array_view->n_children,
                   array_view->n_children)) {
      NANOARROW_RETURN_NOT_OK(ArrowAssertRangeInt8(
          sliced_type_ids, 0, (int8_t)(array_view->n_children - 1), error));
    } else {
      NANOARROW_RETURN_NOT_OK(ArrowAssertInt8In(sliced_type_ids,
                                                array_view->union_type_id_map + 128,
                                                array_view->n_children, error));
    }
  }

  if (array_view->storage_type == NANOARROW_TYPE_DENSE_UNION &&
      array_view->union_type_id_map != NULL) {
    // Check that offsets refer to child elements that actually exist
    for (int64_t i = 0; i < array_view->length; i++) {
      int8_t child_id = ArrowArrayViewUnionChildIndex(array_view, i);
      int64_t offset = ArrowArrayViewUnionChildOffset(array_view, i);
      int64_t child_length = array_view->children[child_id]->length;
      if (offset < 0 || offset > child_length) {
        ArrowErrorSet(error,
                      "[%" PRId64 "] Expected union offset for child id %" PRId8
                      " to be between 0 and %" PRId64
                      " but "
                      "found offset value %" PRId64,
                      i, child_id, child_length, offset);
        return EINVAL;
      }
    }
  }

  if (array_view->storage_type == NANOARROW_TYPE_RUN_END_ENCODED) {
    struct ArrowArrayView* run_ends_view = array_view->children[0];
    if (run_ends_view->length > 0) {
      int64_t last_run_end = ArrowArrayViewGetIntUnsafe(run_ends_view, 0);
      for (int64_t i = 1; i < run_ends_view->length; i++) {
        const int64_t run_end = ArrowArrayViewGetIntUnsafe(run_ends_view, i);
        if (run_end <= last_run_end) {
          ArrowErrorSet(
              error,
              "Every run end must be strictly greater than the previous run end, "
              "but run_ends[%" PRId64 " is %" PRId64 " and run_ends[%" PRId64
              "] is %" PRId64,
              i, run_end, i - 1, last_run_end);
          return EINVAL;
        }
        last_run_end = run_end;
      }
    }
  }

  if (array_view->storage_type == NANOARROW_TYPE_LIST_VIEW ||
      array_view->storage_type == NANOARROW_TYPE_LARGE_LIST_VIEW) {
    int64_t child_len = array_view->children[0]->length;

    struct ArrowBufferView offsets, sizes;
    offsets.data.data = array_view->buffer_views[1].data.data;
    sizes.data.data = array_view->buffer_views[2].data.data;

    for (int64_t i = array_view->offset; i < array_view->length + array_view->offset;
         i++) {
      int64_t offset, size;
      if (array_view->storage_type == NANOARROW_TYPE_LIST_VIEW) {
        offset = offsets.data.as_int32[i];
        size = sizes.data.as_int32[i];
      } else {
        offset = offsets.data.as_int64[i];
        size = sizes.data.as_int64[i];
      }

      if (offset < 0) {
        ArrowErrorSet(error, "Invalid negative offset %" PRId64 " at index %" PRId64,
                      offset, i);
        return EINVAL;
      }

      if (size < 0) {
        ArrowErrorSet(error, "Invalid negative size %" PRId64 " at index %" PRId64, size,
                      i);
        return EINVAL;
      }

      if ((offset + size) > child_len) {
        ArrowErrorSet(error,
                      "Offset: %" PRId64 " + size: %" PRId64 " at index: %" PRId64
                      " exceeds length of child view: %" PRId64,
                      offset, size, i, child_len);
        return EINVAL;
      }
    }
  }

  // Recurse for children
  for (int64_t i = 0; i < array_view->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayViewValidateFull(array_view->children[i], error));
  }

  // Dictionary index validation
  if (array_view->dictionary != NULL) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayViewValidateFull(array_view->dictionary, error));

    // Validate that all non-null indices are within the dictionary bounds
    int64_t dictionary_length = array_view->dictionary->length;
    for (int64_t i = 0; i < array_view->length; i++) {
      if (ArrowArrayViewIsNull(array_view, i)) {
        continue;
      }

      int64_t index = ArrowArrayViewGetIntUnsafe(array_view, i);
      if (index < 0 || index >= dictionary_length) {
        ArrowErrorSet(error,
                      "[%" PRId64 "] Expected dictionary index >= 0 and < %" PRId64
                      " but found value %" PRId64,
                      i, dictionary_length, index);
        return EINVAL;
      }
    }
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowArrayViewValidate(struct ArrowArrayView* array_view,
                                      enum ArrowValidationLevel validation_level,
                                      struct ArrowError* error) {
  switch (validation_level) {
    case NANOARROW_VALIDATION_LEVEL_NONE:
      return NANOARROW_OK;
    case NANOARROW_VALIDATION_LEVEL_MINIMAL:
      return ArrowArrayViewValidateMinimal(array_view, error);
    case NANOARROW_VALIDATION_LEVEL_DEFAULT:
      return ArrowArrayViewValidateDefault(array_view, error);
    case NANOARROW_VALIDATION_LEVEL_FULL:
      NANOARROW_RETURN_NOT_OK(ArrowArrayViewValidateDefault(array_view, error));
      return ArrowArrayViewValidateFull(array_view, error);
  }

  ArrowErrorSet(error, "validation_level not recognized");
  return EINVAL;
}

struct ArrowComparisonInternalState {
  enum ArrowCompareLevel level;
  int is_equal;
  struct ArrowError* reason;
};

NANOARROW_CHECK_PRINTF_ATTRIBUTE static void ArrowComparePrependPath(
    struct ArrowError* out, const char* fmt, ...) {
  if (out == NULL) {
    return;
  }

  char prefix[128];
  prefix[0] = '\0';
  va_list args;
  va_start(args, fmt);
  int prefix_len = vsnprintf(prefix, sizeof(prefix), fmt, args);
  va_end(args);

  if (prefix_len <= 0) {
    return;
  }

  size_t out_len = strlen(out->message);
  size_t out_len_to_move = sizeof(struct ArrowError) - prefix_len - 1;
  if (out_len_to_move > out_len) {
    out_len_to_move = out_len;
  }

  memmove(out->message + prefix_len, out->message, out_len_to_move);
  memcpy(out->message, prefix, prefix_len);
  out->message[out_len + prefix_len] = '\0';
}

#define SET_NOT_EQUAL_AND_RETURN_IF_IMPL(cond_, state_, reason_) \
  do {                                                           \
    if (cond_) {                                                 \
      ArrowErrorSet(state_->reason, ": %s", reason_);            \
      state_->is_equal = 0;                                      \
      return;                                                    \
    }                                                            \
  } while (0)

#define SET_NOT_EQUAL_AND_RETURN_IF(condition_, state_) \
  SET_NOT_EQUAL_AND_RETURN_IF_IMPL(condition_, state_, #condition_)

static void ArrowArrayViewCompareBuffer(const struct ArrowArrayView* actual,
                                        const struct ArrowArrayView* expected, int i,
                                        struct ArrowComparisonInternalState* state) {
  SET_NOT_EQUAL_AND_RETURN_IF(
      actual->buffer_views[i].size_bytes != expected->buffer_views[i].size_bytes, state);

  int64_t buffer_size = actual->buffer_views[i].size_bytes;
  if (buffer_size > 0) {
    SET_NOT_EQUAL_AND_RETURN_IF(
        memcmp(actual->buffer_views[i].data.data, expected->buffer_views[i].data.data,
               buffer_size) != 0,
        state);
  }
}

static void ArrowArrayViewCompareIdentical(const struct ArrowArrayView* actual,
                                           const struct ArrowArrayView* expected,
                                           struct ArrowComparisonInternalState* state) {
  SET_NOT_EQUAL_AND_RETURN_IF(actual->storage_type != expected->storage_type, state);
  SET_NOT_EQUAL_AND_RETURN_IF(actual->n_children != expected->n_children, state);
  SET_NOT_EQUAL_AND_RETURN_IF(actual->dictionary == NULL && expected->dictionary != NULL,
                              state);
  SET_NOT_EQUAL_AND_RETURN_IF(actual->dictionary != NULL && expected->dictionary == NULL,
                              state);

  SET_NOT_EQUAL_AND_RETURN_IF(actual->length != expected->length, state);
  SET_NOT_EQUAL_AND_RETURN_IF(actual->offset != expected->offset, state);
  SET_NOT_EQUAL_AND_RETURN_IF(actual->null_count != expected->null_count, state);

  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    ArrowArrayViewCompareBuffer(actual, expected, i, state);
    if (!state->is_equal) {
      ArrowComparePrependPath(state->reason, ".buffers[%d]", i);
      return;
    }
  }

  for (int64_t i = 0; i < actual->n_children; i++) {
    ArrowArrayViewCompareIdentical(actual->children[i], expected->children[i], state);
    if (!state->is_equal) {
      ArrowComparePrependPath(state->reason, ".children[%" PRId64 "]", i);
      return;
    }
  }

  if (actual->dictionary != NULL) {
    ArrowArrayViewCompareIdentical(actual->dictionary, expected->dictionary, state);
    if (!state->is_equal) {
      ArrowComparePrependPath(state->reason, ".dictionary");
      return;
    }
  }
}

// Top-level entry point to take care of creating, cleaning up, and
// propagating the ArrowComparisonInternalState to the caller
ArrowErrorCode ArrowArrayViewCompare(const struct ArrowArrayView* actual,
                                     const struct ArrowArrayView* expected,
                                     enum ArrowCompareLevel level, int* out,
                                     struct ArrowError* reason) {
  struct ArrowComparisonInternalState state;
  state.level = level;
  state.is_equal = 1;
  state.reason = reason;

  switch (level) {
    case NANOARROW_COMPARE_IDENTICAL:
      ArrowArrayViewCompareIdentical(actual, expected, &state);
      break;
    default:
      return EINVAL;
  }

  *out = state.is_equal;
  if (!state.is_equal) {
    ArrowComparePrependPath(state.reason, "root");
  }

  return NANOARROW_OK;
}

#undef SET_NOT_EQUAL_AND_RETURN_IF
#undef SET_NOT_EQUAL_AND_RETURN_IF_IMPL
