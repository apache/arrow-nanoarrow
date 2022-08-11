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
#include <stdint.h>
#include <string.h>

#include "buffer_inline.h"
#include "typedefs_inline.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline ArrowErrorCode ArrayViewInit(struct ArrowArrayView* array_view,
                                           struct ArrowArray* array,
                                           enum ArrowType storage_type) {
  array_view->array = array;
  array_view->storage_type = storage_type;
  ArrowLayoutInit(&array_view->layout, storage_type);

  int64_t buffers_required = 0;
  for (int i = 0; i < 3; i++) {
    if (array_view->layout.buffer_type == NANOARROW_BUFFER_TYPE_NONE) {
      break;
    }

    switch (array_view->layout.buffer_type) {
      case NANOARROW_BUFFER_TYPE_VALIDITY:
        array_view->buffer_views[i].n_bytes = _ArrowBytesForBits(array->length);
        break;
      case NANOARROW_BUFFER_TYPE_ID:
        array_view->buffer_views[i].n_bytes =
            array_view->layout.element_size_bits[i] / 8 * array->length;
        break;
      case NANOARROW_BUFFER_TYPE_UNION_OFFSET:
        array_view->buffer_views[i].n_bytes =
            array_view->layout.element_size_bits[i] / 8 * array->length;
        break;
      case NANOARROW_BUFFER_TYPE_DATA_OFFSET:
        array_view->buffer_views[i].n_bytes =
            array_view->layout.element_size_bits[i] / 8 * (array->length + 1);
        break;
      case NANOARROW_BUFFER_TYPE_DATA:
        array_view->buffer_views[i].n_bytes =
            _ArrowRoundUpToMultipleOf8(array_view->layout.element_size_bits[i] *
                                       (1 + array->length)) /
            8;
        break;
      default:
        break;
    }

    array_view->buffer_views[i].data.data = array->buffers[i];
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
