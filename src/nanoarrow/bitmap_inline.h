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

static const uint8_t _ArrowkBitmask[] = {1, 2, 4, 8, 16, 32, 64, 128};
static const uint8_t _ArrowkFlippedBitmask[] = {254, 253, 251, 247, 239, 223, 191, 127};
static const uint8_t _ArrowkPrecedingBitmask[] = {0, 1, 3, 7, 15, 31, 63, 127};
static const uint8_t _ArrowkTrailingBitmask[] = {255, 254, 252, 248, 240, 224, 192, 128};

static const uint8_t _ArrowkBytePopcount[] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3,
    4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
    4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4,
    5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5,
    4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2,
    3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
    5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4,
    5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6,
    4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

static inline int64_t _ArrowRoundUpToMultipleOf8(int64_t value) {
  return (value + 7) & ~((int64_t)7);
}

static inline int64_t _ArrowRoundDownToMultipleOf8(int64_t value) {
  return (value / 8) * 8;
}

static inline int64_t _ArrowBytesForBits(int64_t bits) {
  return (bits >> 3) + ((bits & 7) != 0);
}

static inline void _ArrowBitmapPackInt8(const int8_t* values, uint8_t* out) {
  *out = (values[0] | values[1] << 1 | values[2] << 2 | values[3] << 3 | values[4] << 4 |
          values[5] << 5 | values[6] << 6 | values[7] << 7);
}

static inline void _ArrowBitmapPackInt32(const int32_t* values, uint8_t* out) {
  *out = (values[0] | values[1] << 1 | values[2] << 2 | values[3] << 3 | values[4] << 4 |
          values[5] << 5 | values[6] << 6 | values[7] << 7);
}

static inline int8_t ArrowBitmapGetBit(const uint8_t* bits, int64_t i) {
  return (bits[i >> 3] >> (i & 0x07)) & 1;
}

static inline void ArrowBitmapSetBit(uint8_t* bits, int64_t i) {
  bits[i / 8] |= _ArrowkBitmask[i % 8];
}

static inline void ArrowBitmapClearBit(uint8_t* bits, int64_t i) {
  bits[i / 8] &= _ArrowkFlippedBitmask[i % 8];
}

static inline void ArrowBitmapSetBitTo(uint8_t* bits, int64_t i, uint8_t bit_is_set) {
  bits[i / 8] ^=
      ((uint8_t)(-((uint8_t)(bit_is_set != 0)) ^ bits[i / 8])) & _ArrowkBitmask[i % 8];
}

static inline void ArrowBitmapSetBitsTo(uint8_t* bits, int64_t start_offset,
                                        int64_t length, uint8_t bits_are_set) {
  const int64_t i_begin = start_offset;
  const int64_t i_end = start_offset + length;
  const uint8_t fill_byte = (uint8_t)(-bits_are_set);

  const int64_t bytes_begin = i_begin / 8;
  const int64_t bytes_end = i_end / 8 + 1;

  const uint8_t first_byte_mask = _ArrowkPrecedingBitmask[i_begin % 8];
  const uint8_t last_byte_mask = _ArrowkTrailingBitmask[i_end % 8];

  if (bytes_end == bytes_begin + 1) {
    // set bits within a single byte
    const uint8_t only_byte_mask =
        i_end % 8 == 0 ? first_byte_mask : (uint8_t)(first_byte_mask | last_byte_mask);
    bits[bytes_begin] &= only_byte_mask;
    bits[bytes_begin] |= (uint8_t)(fill_byte & ~only_byte_mask);
    return;
  }

  // set/clear trailing bits of first byte
  bits[bytes_begin] &= first_byte_mask;
  bits[bytes_begin] |= (uint8_t)(fill_byte & ~first_byte_mask);

  if (bytes_end - bytes_begin > 2) {
    // set/clear whole bytes
    memset(bits + bytes_begin + 1, fill_byte, (size_t)(bytes_end - bytes_begin - 2));
  }

  if (i_end % 8 == 0) {
    return;
  }

  // set/clear leading bits of last byte
  bits[bytes_end - 1] &= last_byte_mask;
  bits[bytes_end - 1] |= (uint8_t)(fill_byte & ~last_byte_mask);
}

static inline int64_t ArrowBitmapCountSet(const uint8_t* bits, int64_t start_offset,
                                          int64_t length) {
  if (length == 0) {
    return 0;
  }

  const int64_t i_begin = start_offset;
  const int64_t i_end = start_offset + length;

  const int64_t bytes_begin = i_begin / 8;
  const int64_t bytes_end = i_end / 8 + 1;

  const uint8_t first_byte_mask = _ArrowkPrecedingBitmask[i_begin % 8];
  const uint8_t last_byte_mask = _ArrowkTrailingBitmask[i_end % 8];

  if (bytes_end == bytes_begin + 1) {
    // count bits within a single byte
    const uint8_t only_byte_mask =
        i_end % 8 == 0 ? first_byte_mask : (uint8_t)(first_byte_mask | last_byte_mask);
    const uint8_t byte_masked = bits[bytes_begin] & only_byte_mask;
    return _ArrowkBytePopcount[byte_masked];
  }

  int64_t count = 0;

  // first byte
  count += _ArrowkBytePopcount[bits[bytes_begin] & ~first_byte_mask];

  // middle bytes
  for (int64_t i = bytes_begin + 1; i < (bytes_end - 1); i++) {
    count += _ArrowkBytePopcount[bits[i]];
  }

  // last byte
  count += _ArrowkBytePopcount[bits[bytes_end - 1] & ~last_byte_mask];

  return count;
}

static inline void ArrowBitmapBuilderInit(struct ArrowBitmapBuilder* bitmap_builder) {
  ArrowBufferInit(&bitmap_builder->buffer);
  bitmap_builder->size_bits = 0;
}

static inline ArrowErrorCode ArrowBitmapBuilderReserve(
    struct ArrowBitmapBuilder* bitmap_builder, int64_t additional_size_bits) {
  int64_t min_capacity_bits = bitmap_builder->size_bits + additional_size_bits;
  if (min_capacity_bits <= (bitmap_builder->buffer.capacity_bytes * 8)) {
    return NANOARROW_OK;
  }

  int result = ArrowBufferReserve(&bitmap_builder->buffer,
                                  _ArrowBytesForBits(additional_size_bits));
  if (result != NANOARROW_OK) {
    return result;
  }

  bitmap_builder->buffer.data[bitmap_builder->buffer.capacity_bytes - 1] = 0;
  return NANOARROW_OK;
}

static inline ArrowErrorCode ArrowBitmapBuilderAppend(
    struct ArrowBitmapBuilder* bitmap_builder, uint8_t bits_are_set, int64_t length) {
  int result = ArrowBitmapBuilderReserve(bitmap_builder, length);
  if (result != NANOARROW_OK) {
    return result;
  }

  ArrowBitmapSetBitsTo(bitmap_builder->buffer.data, bitmap_builder->size_bits, length,
                       bits_are_set);
  bitmap_builder->size_bits += length;
  bitmap_builder->buffer.size_bytes = _ArrowBytesForBits(bitmap_builder->size_bits);
  return NANOARROW_OK;
}

static inline void ArrowBitmapBuilderAppendInt8Unsafe(
    struct ArrowBitmapBuilder* bitmap_builder, const int8_t* values, int64_t n_values) {
  if (n_values == 0) {
    return;
  }

  const int8_t* values_cursor = values;
  int64_t n_remaining = n_values;
  int64_t out_i_cursor = bitmap_builder->size_bits;
  uint8_t* out_cursor = bitmap_builder->buffer.data + bitmap_builder->size_bits;

  // First byte
  if ((out_i_cursor % 8) != 0) {
    int64_t n_partial_bits = _ArrowRoundUpToMultipleOf8(out_i_cursor) - out_i_cursor;
    for (int i = 0; i < n_partial_bits; i++) {
      ArrowBitmapSetBitTo(bitmap_builder->buffer.data, out_i_cursor++, values[i]);
    }

    out_cursor++;
    values_cursor += n_partial_bits;
    n_remaining -= n_partial_bits;
  }

  // Middle bytes
  int64_t n_full_bytes = n_remaining / 8;
  for (int64_t i = 0; i < n_full_bytes; i++) {
    _ArrowBitmapPackInt8(values_cursor, out_cursor);
    values_cursor += 8;
    out_cursor++;
  }

  // Last byte
  out_i_cursor += n_full_bytes * 8;
  n_remaining -= n_full_bytes * 8;
  if (n_remaining > 0) {
    for (int i = 0; i < n_remaining; i++) {
      ArrowBitmapSetBitTo(bitmap_builder->buffer.data, out_i_cursor++, values_cursor[i]);
    }
  }

  bitmap_builder->size_bits += n_values;
  bitmap_builder->buffer.size_bytes = _ArrowBytesForBits(bitmap_builder->size_bits);
}

static inline void ArrowBitmapBuilderAppendInt32Unsafe(
    struct ArrowBitmapBuilder* bitmap_builder, const int32_t* values, int64_t n_values) {
  for (int64_t i = 0; i < n_values; i++) {
    ArrowBitmapBuilderAppend(bitmap_builder, values[i] != 0, 1);
  }
}

static inline void ArrowBitmapBuilderReset(struct ArrowBitmapBuilder* bitmap_builder) {
  ArrowBufferReset(&bitmap_builder->buffer);
  bitmap_builder->size_bits = 0;
}

#ifdef __cplusplus
}
#endif

#endif
