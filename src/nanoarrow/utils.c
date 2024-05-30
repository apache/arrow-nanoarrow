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
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanoarrow.h"

const char* ArrowNanoarrowVersion(void) { return NANOARROW_VERSION; }

int ArrowNanoarrowVersionInt(void) { return NANOARROW_VERSION_INT; }

ArrowErrorCode ArrowErrorSet(struct ArrowError* error, const char* fmt, ...) {
  if (error == NULL) {
    return NANOARROW_OK;
  }

  memset(error->message, 0, sizeof(error->message));

  va_list args;
  va_start(args, fmt);
  int chars_needed = vsnprintf(error->message, sizeof(error->message), fmt, args);
  va_end(args);

  if (chars_needed < 0) {
    return EINVAL;
  } else if (((size_t)chars_needed) >= sizeof(error->message)) {
    return ERANGE;
  } else {
    return NANOARROW_OK;
  }
}

void ArrowLayoutInit(struct ArrowLayout* layout, enum ArrowType storage_type) {
  layout->buffer_type[0] = NANOARROW_BUFFER_TYPE_VALIDITY;
  layout->buffer_data_type[0] = NANOARROW_TYPE_BOOL;
  layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_DATA;
  layout->buffer_data_type[1] = storage_type;
  layout->buffer_type[2] = NANOARROW_BUFFER_TYPE_NONE;
  layout->buffer_data_type[2] = NANOARROW_TYPE_UNINITIALIZED;

  layout->element_size_bits[0] = 1;
  layout->element_size_bits[1] = 0;
  layout->element_size_bits[2] = 0;

  layout->child_size_elements = 0;

  switch (storage_type) {
    case NANOARROW_TYPE_UNINITIALIZED:
    case NANOARROW_TYPE_NA:
      layout->buffer_type[0] = NANOARROW_BUFFER_TYPE_NONE;
      layout->buffer_data_type[0] = NANOARROW_TYPE_UNINITIALIZED;
      layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_NONE;
      layout->buffer_data_type[1] = NANOARROW_TYPE_UNINITIALIZED;
      layout->element_size_bits[0] = 0;
      break;

    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_MAP:
      layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_DATA_OFFSET;
      layout->buffer_data_type[1] = NANOARROW_TYPE_INT32;
      layout->element_size_bits[1] = 32;
      break;

    case NANOARROW_TYPE_LARGE_LIST:
      layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_DATA_OFFSET;
      layout->buffer_data_type[1] = NANOARROW_TYPE_INT64;
      layout->element_size_bits[1] = 64;
      break;

    case NANOARROW_TYPE_STRUCT:
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_NONE;
      layout->buffer_data_type[1] = NANOARROW_TYPE_UNINITIALIZED;
      break;

    case NANOARROW_TYPE_BOOL:
      layout->element_size_bits[1] = 1;
      break;

    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT8:
      layout->element_size_bits[1] = 8;
      break;

    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_HALF_FLOAT:
      layout->element_size_bits[1] = 16;
      break;

    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_FLOAT:
      layout->element_size_bits[1] = 32;
      break;
    case NANOARROW_TYPE_INTERVAL_MONTHS:
      layout->buffer_data_type[1] = NANOARROW_TYPE_INT32;
      layout->element_size_bits[1] = 32;
      break;

    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_DOUBLE:
    case NANOARROW_TYPE_INTERVAL_DAY_TIME:
      layout->element_size_bits[1] = 64;
      break;

    case NANOARROW_TYPE_DECIMAL128:
    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO:
      layout->element_size_bits[1] = 128;
      break;

    case NANOARROW_TYPE_DECIMAL256:
      layout->element_size_bits[1] = 256;
      break;

    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
      layout->buffer_data_type[1] = NANOARROW_TYPE_BINARY;
      break;

    case NANOARROW_TYPE_DENSE_UNION:
      layout->buffer_type[0] = NANOARROW_BUFFER_TYPE_TYPE_ID;
      layout->buffer_data_type[0] = NANOARROW_TYPE_INT8;
      layout->element_size_bits[0] = 8;
      layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_UNION_OFFSET;
      layout->buffer_data_type[1] = NANOARROW_TYPE_INT32;
      layout->element_size_bits[1] = 32;
      break;

    case NANOARROW_TYPE_SPARSE_UNION:
      layout->buffer_type[0] = NANOARROW_BUFFER_TYPE_TYPE_ID;
      layout->buffer_data_type[0] = NANOARROW_TYPE_INT8;
      layout->element_size_bits[0] = 8;
      layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_NONE;
      layout->buffer_data_type[1] = NANOARROW_TYPE_UNINITIALIZED;
      break;

    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_DATA_OFFSET;
      layout->buffer_data_type[1] = NANOARROW_TYPE_INT32;
      layout->element_size_bits[1] = 32;
      layout->buffer_type[2] = NANOARROW_BUFFER_TYPE_DATA;
      layout->buffer_data_type[2] = storage_type;
      break;

    case NANOARROW_TYPE_LARGE_STRING:
      layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_DATA_OFFSET;
      layout->buffer_data_type[1] = NANOARROW_TYPE_INT64;
      layout->element_size_bits[1] = 64;
      layout->buffer_type[2] = NANOARROW_BUFFER_TYPE_DATA;
      layout->buffer_data_type[2] = NANOARROW_TYPE_STRING;
      break;
    case NANOARROW_TYPE_LARGE_BINARY:
      layout->buffer_type[1] = NANOARROW_BUFFER_TYPE_DATA_OFFSET;
      layout->buffer_data_type[1] = NANOARROW_TYPE_INT64;
      layout->element_size_bits[1] = 64;
      layout->buffer_type[2] = NANOARROW_BUFFER_TYPE_DATA;
      layout->buffer_data_type[2] = NANOARROW_TYPE_BINARY;
      break;

    default:
      break;
  }
}

void* ArrowMalloc(int64_t size) { return malloc(size); }

void* ArrowRealloc(void* ptr, int64_t size) { return realloc(ptr, size); }

void ArrowFree(void* ptr) { free(ptr); }

static uint8_t* ArrowBufferAllocatorMallocReallocate(
    struct ArrowBufferAllocator* allocator, uint8_t* ptr, int64_t old_size,
    int64_t new_size) {
  NANOARROW_UNUSED(allocator);
  NANOARROW_UNUSED(old_size);
  return (uint8_t*)ArrowRealloc(ptr, new_size);
}

static void ArrowBufferAllocatorMallocFree(struct ArrowBufferAllocator* allocator,
                                           uint8_t* ptr, int64_t size) {
  NANOARROW_UNUSED(allocator);
  NANOARROW_UNUSED(size);
  if (ptr != NULL) {
    ArrowFree(ptr);
  }
}

static struct ArrowBufferAllocator ArrowBufferAllocatorMalloc = {
    &ArrowBufferAllocatorMallocReallocate, &ArrowBufferAllocatorMallocFree, NULL};

struct ArrowBufferAllocator ArrowBufferAllocatorDefault(void) {
  return ArrowBufferAllocatorMalloc;
}

static uint8_t* ArrowBufferDeallocatorReallocate(struct ArrowBufferAllocator* allocator,
                                                 uint8_t* ptr, int64_t old_size,
                                                 int64_t new_size) {
  NANOARROW_UNUSED(new_size);

  // Attempting to reallocate a buffer with a custom deallocator is
  // a programming error. In debug mode, crash here.
#if defined(NANOARROW_DEBUG)
  NANOARROW_PRINT_AND_DIE(ENOMEM,
                          "It is an error to reallocate a buffer whose allocator is "
                          "ArrowBufferDeallocator()");
#endif

  // In release mode, ensure the the deallocator is called exactly
  // once using the pointer it was given and return NULL, which
  // will trigger the caller to return ENOMEM.
  allocator->free(allocator, ptr, old_size);
  *allocator = ArrowBufferAllocatorDefault();
  return NULL;
}

struct ArrowBufferAllocator ArrowBufferDeallocator(
    void (*custom_free)(struct ArrowBufferAllocator* allocator, uint8_t* ptr,
                        int64_t size),
    void* private_data) {
  struct ArrowBufferAllocator allocator;
  allocator.reallocate = &ArrowBufferDeallocatorReallocate;
  allocator.free = custom_free;
  allocator.private_data = private_data;
  return allocator;
}

static const int kInt32DecimalDigits = 9;

static const uint64_t kUInt32PowersOfTen[] = {
    1ULL,      10ULL,      100ULL,      1000ULL,      10000ULL,
    100000ULL, 1000000ULL, 10000000ULL, 100000000ULL, 1000000000ULL};

// Adapted from Arrow C++ to use 32-bit words for better C portability
// https://github.com/apache/arrow/blob/cd3321b28b0c9703e5d7105d6146c1270bbadd7f/cpp/src/arrow/util/decimal.cc#L524-L544
static void ShiftAndAdd(struct ArrowStringView value, uint32_t* out, int64_t out_size) {
  // We use strtoll for parsing, which needs input that is null-terminated
  char chunk_string[16];

  for (int64_t posn = 0; posn < value.size_bytes;) {
    int64_t remaining = value.size_bytes - posn;

    int64_t group_size;
    if (remaining > kInt32DecimalDigits) {
      group_size = kInt32DecimalDigits;
    } else {
      group_size = remaining;
    }

    const uint64_t multiple = kUInt32PowersOfTen[group_size];

    memcpy(chunk_string, value.data + posn, group_size);
    chunk_string[group_size] = '\0';
    uint32_t chunk = (uint32_t)strtoll(chunk_string, NULL, 10);

    for (int64_t i = 0; i < out_size; i++) {
      uint64_t tmp = out[i];
      tmp *= multiple;
      tmp += chunk;
      out[i] = (uint32_t)(tmp & 0xFFFFFFFFULL);
      chunk = (uint32_t)(tmp >> 32);
    }
    posn += group_size;
  }
}

ArrowErrorCode ArrowDecimalSetDigits(struct ArrowDecimal* decimal,
                                     struct ArrowStringView value) {
  // Check for sign
  int is_negative = value.data[0] == '-';
  int has_sign = is_negative || value.data[0] == '+';
  value.data += has_sign;
  value.size_bytes -= has_sign;

  // Check all characters are digits that are not the negative sign
  for (int64_t i = 0; i < value.size_bytes; i++) {
    char c = value.data[i];
    if (c < '0' || c > '9') {
      return EINVAL;
    }
  }

  // Skip over leading 0s
  int64_t n_leading_zeroes = 0;
  for (int64_t i = 0; i < value.size_bytes; i++) {
    if (value.data[i] == '0') {
      n_leading_zeroes++;
    } else {
      break;
    }
  }

  value.data += n_leading_zeroes;
  value.size_bytes -= n_leading_zeroes;

  // Use 32-bit words for portability
  uint32_t words32[8];
  int n_words32 = decimal->n_words * 2;
  NANOARROW_DCHECK(n_words32 <= 8);
  memset(words32, 0, sizeof(words32));

  ShiftAndAdd(value, words32, n_words32);

  if (decimal->low_word_index == 0) {
    memcpy(decimal->words, words32, sizeof(uint32_t) * n_words32);
  } else {
    uint64_t lo;
    uint64_t hi;

    for (int i = 0; i < decimal->n_words; i++) {
      lo = (uint64_t)words32[i * 2];
      hi = (uint64_t)words32[i * 2 + 1] << 32;
      decimal->words[decimal->n_words - i - 1] = lo | hi;
    }
  }

  if (is_negative) {
    ArrowDecimalNegate(decimal);
  }

  return NANOARROW_OK;
}

// Adapted from Arrow C++ for C
// https://github.com/apache/arrow/blob/cd3321b28b0c9703e5d7105d6146c1270bbadd7f/cpp/src/arrow/util/decimal.cc#L365
ArrowErrorCode ArrowDecimalAppendDigitsToBuffer(const struct ArrowDecimal* decimal,
                                                struct ArrowBuffer* buffer) {
  int is_negative = ArrowDecimalSign(decimal) < 0;

  uint64_t words_little_endian[4];
  if (decimal->low_word_index == 0) {
    memcpy(words_little_endian, decimal->words, decimal->n_words * sizeof(uint64_t));
  } else {
    for (int i = 0; i < decimal->n_words; i++) {
      words_little_endian[i] = decimal->words[decimal->n_words - i - 1];
    }
  }

  // We've already made a copy, so negate that if needed
  if (is_negative) {
    uint64_t carry = 1;
    for (int i = 0; i < decimal->n_words; i++) {
      uint64_t elem = words_little_endian[i];
      elem = ~elem + carry;
      carry &= (elem == 0);
      words_little_endian[i] = elem;
    }
  }

  // Find the most significant word that is non-zero
  int most_significant_elem_idx = -1;
  for (int i = decimal->n_words - 1; i >= 0; i--) {
    if (words_little_endian[i] != 0) {
      most_significant_elem_idx = i;
      break;
    }
  }

  // If they are all zero, the output is just '0'
  if (most_significant_elem_idx == -1) {
    NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt8(buffer, '0'));
    return NANOARROW_OK;
  }

  // Define segments such that each segment represents 9 digits with the
  // least significant group of 9 digits first. For example, if the input represents
  // 9876543210123456789, then segments will be [123456789, 876543210, 9].
  // We handle at most a signed 256 bit integer, whose maximum value occupies 77
  // characters. Thus, we need at most 9 segments.
  const uint32_t k1e9 = 1000000000U;
  int num_segments = 0;
  uint32_t segments[9];
  memset(segments, 0, sizeof(segments));
  uint64_t* most_significant_elem = words_little_endian + most_significant_elem_idx;

  do {
    // Compute remainder = words_little_endian % 1e9 and words_little_endian =
    // words_little_endian / 1e9.
    uint32_t remainder = 0;
    uint64_t* elem = most_significant_elem;

    do {
      // Compute dividend = (remainder << 32) | *elem  (a virtual 96-bit integer);
      // *elem = dividend / 1e9;
      // remainder = dividend % 1e9.
      uint32_t hi = (uint32_t)(*elem >> 32);
      uint32_t lo = (uint32_t)(*elem & 0xFFFFFFFFULL);
      uint64_t dividend_hi = ((uint64_t)(remainder) << 32) | hi;
      uint64_t quotient_hi = dividend_hi / k1e9;
      remainder = (uint32_t)(dividend_hi % k1e9);
      uint64_t dividend_lo = ((uint64_t)(remainder) << 32) | lo;
      uint64_t quotient_lo = dividend_lo / k1e9;
      remainder = (uint32_t)(dividend_lo % k1e9);

      *elem = (quotient_hi << 32) | quotient_lo;
    } while (elem-- != words_little_endian);

    segments[num_segments++] = remainder;
  } while (*most_significant_elem != 0 || most_significant_elem-- != words_little_endian);

  // We know our output has no more than 9 digits per segment, plus a negative sign,
  // plus any further digits between our output of 9 digits plus enough
  // extra characters to ensure that snprintf() with n = 21 (maximum length of %lu
  // including a the null terminator) is bounded properly.
  NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(buffer, num_segments * 9 + 1 + 21 - 9));
  if (is_negative) {
    buffer->data[buffer->size_bytes++] = '-';
  }

  // The most significant segment should have no leading zeroes
  int n_chars = snprintf((char*)buffer->data + buffer->size_bytes, 21, "%lu",
                         (unsigned long)segments[num_segments - 1]);

  // Ensure that an encoding error from snprintf() does not result
  // in an out-of-bounds access.
  if (n_chars < 0) {
    return ERANGE;
  }

  buffer->size_bytes += n_chars;

  // Subsequent output needs to be left-padded with zeroes such that each segment
  // takes up exactly 9 digits.
  for (int i = num_segments - 2; i >= 0; i--) {
    int n_chars = snprintf((char*)buffer->data + buffer->size_bytes, 21, "%09lu",
                           (unsigned long)segments[i]);
    buffer->size_bytes += n_chars;
    NANOARROW_DCHECK(buffer->size_bytes <= buffer->capacity_bytes);
  }

  return NANOARROW_OK;
}
