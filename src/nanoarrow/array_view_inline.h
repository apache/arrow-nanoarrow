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

static inline void ArrayViewInit(struct ArrowArrayView* array_view,
                                 enum ArrowType storage_type) {
  memset(array_view, 0, sizeof(struct ArrowArrayView));
  array_view->storage_type = storage_type;
  ArrowLayoutInit(&array_view->layout, storage_type);
}

static inline void ArrayViewSetLength(struct ArrowArrayView* array_view, int64_t length) {
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
        if (length > 0) {
          array_view->buffer_views[i].n_bytes = element_size_bytes * (length + 1);
          array_view->buffer_views[i + 1].n_bytes =
              array_view->buffer_views[i].data.int32[length];
        } else {
          array_view->buffer_views[i].n_bytes = 0;
          array_view->buffer_views[i + 1].n_bytes = 0;
        }
        
        // Skip the data buffer
        i++;
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
}

static inline ArrowErrorCode ArrayViewSetArray(struct ArrowArrayView* array_view,
                                               struct ArrowArray* array) {
  array_view->array = array;
  ArrayViewSetLength(array_view, array->offset + array->length);

  int64_t buffers_required = 0;
  for (int i = 0; i < 3; i++) {
    if (array_view->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_NONE) {
      break;
    } else {
      array_view->buffer_views[i].data.data = array->buffers[i];
    }
  }

  if (buffers_required != array->n_buffers) {
    return EINVAL;
  }

  return NANOARROW_OK;
}

#ifdef __cplusplus
}
#endif

#endif
