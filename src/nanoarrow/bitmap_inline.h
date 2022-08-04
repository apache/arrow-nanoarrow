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

#ifndef NANOARROW_BITMAP_INLINE_H_INCLUDED
#define NANOARROW_BITMAP_INLINE_H_INCLUDED

#include <stdlib.h>
#include <string.h>

#include "buffer_inline.h"
#include "typedefs_inline.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline int8_t ArrowBitmapGetBit(const uint8_t* bits, int64_t i) {
  return 0 != (bits[i / 8] & ((int8_t)0x01) << (i % 8));
}

static inline void ArrowBitmapSetElement(uint8_t* bits, int64_t i, int8_t value) {
  int8_t mask = 0x01 << (i % 8);
  if (value) {
    bits[i / 8] |= mask;
  } else {
    bits[i / 8] &= ~mask;
  }
}

static inline int64_t ArrowBitmapCountTrue(const uint8_t* bits, int64_t i_from,
                                           int64_t i_to) {
  int64_t count = 0;
  for (int64_t i = i_from; i < i_to; i++) {
    count += ArrowBitmapGetBit(bits, i);
  }
  return count;
}

static inline int64_t ArrowBitmapCountFalse(const uint8_t* bits, int64_t i_from,
                                            int64_t i_to) {
  int64_t count = 0;
  for (int64_t i = i_from; i < i_to; i++) {
    count += !ArrowBitmapGetBit(bits, i);
  }
  return count;
}

static inline void ArrowBitmapBuilderInit(struct ArrowBitmapBuilder* bitmap_builder) {
  ArrowBufferInit(&bitmap_builder->buffer);
  bitmap_builder->size_bits = 0;
  bitmap_builder->n_pending_values = 0;
}

static inline ArrowErrorCode ArrowBitmapBuilderAppend(
    struct ArrowBitmapBuilder* bitmap_builder, int8_t value) {
  if (bitmap_builder->n_pending_values == 64) {
    int result = ArrowBitmapBuilderFlush(bitmap_builder);
    if (result != NANOARROW_OK) {
      return result;
    }
  }

  bitmap_builder->pending_values[bitmap_builder->n_pending_values] = value != 0;
  bitmap_builder->size_bits++;
  bitmap_builder->n_pending_values++;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowBitmapBuilderAppendInt8(
    struct ArrowBitmapBuilder* bitmap_builder, const int8_t* values, int64_t n_values) {
  int result;
  for (int64_t i = 0; i < n_values; i++) {
    result = ArrowBitmapBuilderAppend(bitmap_builder, values[i]);
    if (result != NANOARROW_OK) {
      return result;
    }
  }
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowBitmapBuilderAppendInt32(
    struct ArrowBitmapBuilder* bitmap_builder, const int32_t* values, int64_t n_values) {
  int result;
  for (int64_t i = 0; i < n_values; i++) {
    result = ArrowBitmapBuilderAppend(bitmap_builder, values[i] != 0);
    if (result != NANOARROW_OK) {
      return result;
    }
  }
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowBitmapBuilderFlush(
    struct ArrowBitmapBuilder* bitmap_builder) {
  int8_t pending_bitmap[8];
  memset(pending_bitmap, 0, sizeof(pending_bitmap));

  for (int i = 0; i < bitmap_builder->n_pending_values; i++) {
    pending_bitmap[i / 8] |= bitmap_builder->pending_values[i] << (i % 8);
  }

  int result =
      ArrowBufferAppend(&bitmap_builder->buffer, pending_bitmap, sizeof(pending_bitmap));
  if (result != NANOARROW_OK) {
    return result;
  }

  bitmap_builder->n_pending_values = 0;

  return NANOARROW_OK;
}

static inline void ArrowBitmapBuilderReset(struct ArrowBitmapBuilder* bitmap_builder) {
  ArrowBufferReset(&bitmap_builder->buffer);
  bitmap_builder->size_bits = 0;
  bitmap_builder->n_pending_values = 0;
}

#ifdef __cplusplus
}
#endif

#endif
