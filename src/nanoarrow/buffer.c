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
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "nanoarrow.h"

static int64_t ArrowGrowByFactor(int64_t current_capacity, int64_t new_capacity) {
  int64_t doubled_capacity = current_capacity * 2;
  if (doubled_capacity > new_capacity) {
    return doubled_capacity;
  } else {
    return new_capacity;
  }
}

void ArrowBufferInit(struct ArrowBuffer* buffer) {
  buffer->data = NULL;
  buffer->size_bytes = 0;
  buffer->capacity_bytes = 0;
  buffer->allocator = ArrowBufferAllocatorDefault();
}

ArrowErrorCode ArrowBufferSetAllocator(struct ArrowBuffer* buffer,
                                       struct ArrowBufferAllocator* allocator) {
  if (buffer->data == NULL) {
    buffer->allocator = allocator;
    return NANOARROW_OK;
  } else {
    return EINVAL;
  }
}

void ArrowBufferReset(struct ArrowBuffer* buffer) {
  if (buffer->data != NULL) {
    buffer->allocator->free(buffer->allocator, (uint8_t*)buffer->data,
                            buffer->capacity_bytes);
    buffer->data = NULL;
  }

  buffer->capacity_bytes = 0;
  buffer->size_bytes = 0;
}

void ArrowBufferMove(struct ArrowBuffer* buffer, struct ArrowBuffer* buffer_out) {
  memcpy(buffer_out, buffer, sizeof(struct ArrowBuffer));
  buffer->data = NULL;
  ArrowBufferReset(buffer);
}

ArrowErrorCode ArrowBufferResize(struct ArrowBuffer* buffer, int64_t new_capacity_bytes,
                                 char shrink_to_fit) {
  if (new_capacity_bytes < 0) {
    return EINVAL;
  }

  if (new_capacity_bytes > buffer->capacity_bytes || shrink_to_fit) {
    buffer->data = buffer->allocator->reallocate(
        buffer->allocator, buffer->data, buffer->capacity_bytes, new_capacity_bytes);
    if (buffer->data == NULL && new_capacity_bytes > 0) {
      buffer->capacity_bytes = 0;
      buffer->size_bytes = 0;
      return ENOMEM;
    }

    buffer->capacity_bytes = new_capacity_bytes;
  }

  // Ensures that when shrinking that size <= capacity
  if (new_capacity_bytes < buffer->size_bytes) {
    buffer->size_bytes = new_capacity_bytes;
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowBufferReserve(struct ArrowBuffer* buffer,
                                  int64_t additional_size_bytes) {
  int64_t min_capacity_bytes = buffer->size_bytes + additional_size_bytes;
  if (min_capacity_bytes <= buffer->capacity_bytes) {
    return NANOARROW_OK;
  }

  return ArrowBufferResize(
      buffer, ArrowGrowByFactor(buffer->capacity_bytes, min_capacity_bytes), 0);
}

void ArrowBufferAppendUnsafe(struct ArrowBuffer* buffer, const void* data,
                             int64_t size_bytes) {
  if (size_bytes > 0) {
    memcpy(buffer->data + buffer->size_bytes, data, size_bytes);
    buffer->size_bytes += size_bytes;
  }
}

ArrowErrorCode ArrowBufferAppend(struct ArrowBuffer* buffer, const void* data,
                                 int64_t size_bytes) {
  int result = ArrowBufferReserve(buffer, size_bytes);
  if (result != NANOARROW_OK) {
    return result;
  }

  ArrowBufferAppendUnsafe(buffer, data, size_bytes);
  return NANOARROW_OK;
}
