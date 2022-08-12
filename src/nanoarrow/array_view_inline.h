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

#ifndef NANOARROW_ARRAY_VIEW_INLINE_H_INCLUDED
#define NANOARROW_ARRAY_VIEW_INLINE_H_INCLUDED

#include <errno.h>
#include <stdint.h>
#include <string.h>

#include "buffer_inline.h"
#include "typedefs_inline.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline void ArrowArrayViewSetLength(struct ArrowArrayView* array_view,
                                           int64_t length) {
  for (int i = 0; i < 3; i++) {
    int64_t element_size_bytes = array_view->layout.element_size_bits[i] / 8;

    switch (array_view->layout.buffer_type[i]) {
      array_view->buffer_views[i].data.data = NULL;

      case NANOARROW_BUFFER_TYPE_VALIDITY:
        array_view->buffer_views[i].n_bytes = _ArrowBytesForBits(length);
        continue;
      case NANOARROW_BUFFER_TYPE_DATA_OFFSET:
        // Probably don't want/need to rely on the producer to have allocated an
        // offsets buffer of length 1 for a zero-size array
        array_view->buffer_views[i].n_bytes =
            (length != 0) * element_size_bytes * (length + 1);
        continue;
      case NANOARROW_BUFFER_TYPE_DATA:
        array_view->buffer_views[i].n_bytes =
            _ArrowRoundUpToMultipleOf8(array_view->layout.element_size_bits[i] * length) /
            8;
        continue;
      case NANOARROW_BUFFER_TYPE_TYPE_ID:
      case NANOARROW_BUFFER_TYPE_UNION_OFFSET:
        array_view->buffer_views[i].n_bytes = element_size_bytes * length;
        continue;
      case NANOARROW_BUFFER_TYPE_NONE:
        array_view->buffer_views[i].n_bytes = 0;
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
    default:
      break;
  }
}

static inline ArrowErrorCode ArrowArrayViewSetArray(struct ArrowArrayView* array_view,
                                                    struct ArrowArray* array) {
  array_view->array = array;
  ArrowArrayViewSetLength(array_view, array->offset + array->length);

  int64_t buffers_required = 0;
  for (int i = 0; i < 3; i++) {
    if (array_view->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_NONE) {
      break;
    }

    buffers_required++;

    // If the null_count is 0, the validity buffer can be NULL
    if (array_view->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_VALIDITY &&
        array->null_count == 0 && array->buffers[i] == NULL) {
      array_view->buffer_views[i].n_bytes = 0;
    }

    array_view->buffer_views[i].data.data = array->buffers[i];
  }

  if (buffers_required != array->n_buffers) {
    return EINVAL;
  }

  if (array_view->n_children != array->n_children) {
    return EINVAL;
  }

  // Calculate child sizes and sizes that depend on data in the array buffers
  int result;
  switch (array_view->storage_type) {
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      if (array_view->buffer_views[1].n_bytes != 0) {
        array_view->buffer_views[2].n_bytes =
            array_view->buffer_views[1].data.int32[array->offset + array->length];
      }
      break;
    case NANOARROW_TYPE_LARGE_STRING:
    case NANOARROW_TYPE_LARGE_BINARY:
      if (array_view->buffer_views[1].n_bytes != 0) {
        array_view->buffer_views[2].n_bytes =
            array_view->buffer_views[1].data.int64[array->offset + array->length];
      }
      break;
    case NANOARROW_TYPE_STRUCT:
    case NANOARROW_TYPE_SPARSE_UNION:
      for (int64_t i = 0; i < array_view->n_children; i++) {
        result = ArrowArrayViewSetArray(array_view->children[i], array->children[i]);
        if (result != NANOARROW_OK) {
          return result;
        }
      }
      break;
    default:
      break;
  }

  return NANOARROW_OK;
}

#ifdef __cplusplus
}
#endif

#endif
