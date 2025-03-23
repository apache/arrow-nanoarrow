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

#include <cstring>
#include <string>

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW)
#include <arrow/util/decimal.h>
#endif
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "nanoarrow/nanoarrow.hpp"

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW)
using namespace arrow;
#endif

TEST(BuildIdTest, VersionTest) {
  EXPECT_STREQ(ArrowNanoarrowVersion(), NANOARROW_VERSION);
  EXPECT_EQ(ArrowNanoarrowVersionInt(), NANOARROW_VERSION_INT);
}

TEST(ErrorTest, ErrorTestInit) {
  struct ArrowError error;
  memset(&error.message, 0xff, sizeof(ArrowError));
  ArrowErrorInit(&error);
  EXPECT_STREQ(ArrowErrorMessage(&error), "");
  ArrowErrorInit(nullptr);
  EXPECT_STREQ(ArrowErrorMessage(nullptr), "");
}

TEST(ErrorTest, ErrorTestSet) {
  struct ArrowError error;
  EXPECT_EQ(ArrowErrorSet(&error, "there were %d foxes", 4), NANOARROW_OK);
  EXPECT_STREQ(ArrowErrorMessage(&error), "there were 4 foxes");
}

TEST(ErrorTest, ErrorTestSetOverrun) {
  struct ArrowError error;
  char big_error[2048];
  const char* a_few_chars = "abcdefg";
  for (int i = 0; i < 2047; i++) {
    big_error[i] = a_few_chars[i % strlen(a_few_chars)];
  }
  big_error[2047] = '\0';

  EXPECT_EQ(ArrowErrorSet(&error, "%s", big_error), ERANGE);
  EXPECT_EQ(std::string(ArrowErrorMessage(&error)), std::string(big_error, 1023));

  wchar_t bad_string[] = {0xFFFF, 0};
  EXPECT_EQ(ArrowErrorSet(&error, "%ls", bad_string), EINVAL);
}

TEST(ErrorTest, ErrorTestSetString) {
  struct ArrowError error;
  ArrowErrorSetString(&error, "a pretty short error string");
  EXPECT_STREQ(ArrowErrorMessage(&error), "a pretty short error string");
}

TEST(ErrorTest, ErrorTestSetStringOverrun) {
  struct ArrowError error;
  char big_error[2048];
  const char* a_few_chars = "abcdefg";
  for (int i = 0; i < 2047; i++) {
    big_error[i] = a_few_chars[i % strlen(a_few_chars)];
  }
  big_error[2047] = '\0';

  ArrowErrorSetString(&error, big_error);
  EXPECT_EQ(std::string(ArrowErrorMessage(&error)), std::string(big_error, 1023));
}

#if defined(NANOARROW_DEBUG)
#undef NANOARROW_PRINT_AND_DIE
#define NANOARROW_PRINT_AND_DIE(VALUE, EXPR_STR)             \
  do {                                                       \
    ArrowErrorSet(&error, "%s failed with errno", EXPR_STR); \
  } while (0)

TEST(ErrorTest, ErrorTestAssertNotOkDebug) {
  struct ArrowError error;
  NANOARROW_ASSERT_OK(EINVAL);
  EXPECT_STREQ(ArrowErrorMessage(&error), "EINVAL failed with errno");
}
#else
TEST(ErrorTest, ErrorTestAssertNotOkRelease) { NANOARROW_ASSERT_OK(EINVAL); }
#endif

class CustomMemoryPool {
 public:
  CustomMemoryPool() : bytes_allocated(0), num_reallocations(0), num_frees(0) {}
  int64_t bytes_allocated;
  int64_t num_reallocations;
  int64_t num_frees;

  static CustomMemoryPool* GetInstance() {
    static CustomMemoryPool pool;
    return &pool;
  }
};

static uint8_t* MemoryPoolReallocate(struct ArrowBufferAllocator* allocator, uint8_t* ptr,
                                     int64_t old_size, int64_t new_size) {
  auto pool = reinterpret_cast<CustomMemoryPool*>(allocator->private_data);
  pool->bytes_allocated -= old_size;
  pool->num_reallocations += 1;

  void* out = ArrowRealloc(ptr, new_size);
  if (out != nullptr) {
    pool->bytes_allocated += new_size;
  }

  return reinterpret_cast<uint8_t*>(out);
}

static void MemoryPoolFree(struct ArrowBufferAllocator* allocator, uint8_t* ptr,
                           int64_t size) {
  auto pool = reinterpret_cast<CustomMemoryPool*>(allocator->private_data);
  pool->bytes_allocated -= size;
  ArrowFree(ptr);
}

static void MemoryPoolAllocatorInit(struct ArrowBufferAllocator* allocator) {
  allocator->reallocate = &MemoryPoolReallocate;
  allocator->free = &MemoryPoolFree;
  allocator->private_data = CustomMemoryPool::GetInstance();
}

TEST(AllocatorTest, AllocatorTestDefault) {
  struct ArrowBufferAllocator allocator = ArrowBufferAllocatorDefault();

  uint8_t* buffer = allocator.reallocate(&allocator, nullptr, 0, 10);
  const char* test_str = "abcdefg";
  memcpy(buffer, test_str, strlen(test_str) + 1);

  buffer = allocator.reallocate(&allocator, buffer, 10, 100);
  EXPECT_STREQ(reinterpret_cast<const char*>(buffer), test_str);

  allocator.free(&allocator, buffer, 100);

#if !defined(__SANITIZE_ADDRESS__)
  buffer =
      allocator.reallocate(&allocator, nullptr, 0, std::numeric_limits<int64_t>::max());
  EXPECT_EQ(buffer, nullptr);

  buffer =
      allocator.reallocate(&allocator, buffer, 0, std::numeric_limits<int64_t>::max());
  EXPECT_EQ(buffer, nullptr);
#endif
}

// In a non-trivial test this struct could hold a reference to an object
// that keeps the buffer from being garbage collected (e.g., an SEXP in R)
struct CustomFreeData {
  void* pointer_proxy;
  int64_t num_frees;
};

static void CustomFree(struct ArrowBufferAllocator* allocator, uint8_t* ptr,
                       int64_t size) {
  NANOARROW_UNUSED(ptr);
  NANOARROW_UNUSED(size);
  auto data = reinterpret_cast<struct CustomFreeData*>(allocator->private_data);
  ArrowFree(data->pointer_proxy);
  data->pointer_proxy = nullptr;
  data->num_frees++;
}

TEST(AllocatorTest, AllocatorTestDeallocator) {
  struct CustomFreeData data;
  data.pointer_proxy = nullptr;
  data.num_frees = 0;

  struct ArrowBuffer buffer;
  ArrowBufferInit(&buffer);

  // Check that a zero-size, nullptr buffer is freed exactly once
  ASSERT_EQ(ArrowBufferSetAllocator(&buffer, ArrowBufferDeallocator(&CustomFree, &data)),
            NANOARROW_OK);
  ASSERT_EQ(buffer.allocator.private_data, &data);

  ArrowBufferReset(&buffer);
  EXPECT_EQ(data.num_frees, 1);
  EXPECT_NE(buffer.allocator.private_data, &data);

  // Check reallocate behaviour
  ASSERT_EQ(ArrowBufferSetAllocator(&buffer, ArrowBufferDeallocator(&CustomFree, &data)),
            NANOARROW_OK);
  data.num_frees = 0;

  // In debug mode, an attempt to reallocate will crash with a message;
  // in release mode it should result in ENOMEM.
#if !defined(NANOARROW_DEBUG)
  ASSERT_EQ(ArrowBufferAppendUInt8(&buffer, 0), ENOMEM);
#endif
  //  Either way we should get exactly one free
  ArrowBufferReset(&buffer);
  EXPECT_EQ(data.num_frees, 1);

  // Check wrapping an actual buffer
  ASSERT_EQ(ArrowBufferSetAllocator(&buffer, ArrowBufferDeallocator(&CustomFree, &data)),
            NANOARROW_OK);
  buffer.data = reinterpret_cast<uint8_t*>(ArrowMalloc(12));
  buffer.size_bytes = 12;
  data.pointer_proxy = buffer.data;
  data.num_frees = 0;

  ArrowBufferReset(&buffer);
  EXPECT_EQ(data.num_frees, 1);
}

TEST(AllocatorTest, AllocatorTestMemoryPool) {
  struct ArrowBufferAllocator arrow_allocator;
  MemoryPoolAllocatorInit(&arrow_allocator);

  int64_t allocated0 = CustomMemoryPool::GetInstance()->bytes_allocated;

  uint8_t* buffer = arrow_allocator.reallocate(&arrow_allocator, nullptr, 0, 10);
  EXPECT_EQ(CustomMemoryPool::GetInstance()->bytes_allocated - allocated0, 10);
  memset(buffer, 0, 10);

  const char* test_str = "abcdefg";
  memcpy(buffer, test_str, strlen(test_str) + 1);

  buffer = arrow_allocator.reallocate(&arrow_allocator, buffer, 10, 100);
  EXPECT_EQ(CustomMemoryPool::GetInstance()->bytes_allocated - allocated0, 100);
  EXPECT_STREQ(reinterpret_cast<const char*>(buffer), test_str);

  arrow_allocator.free(&arrow_allocator, buffer, 100);
  EXPECT_EQ(CustomMemoryPool::GetInstance()->bytes_allocated, allocated0);

#if !defined(__SANITIZE_ADDRESS__)
  buffer = arrow_allocator.reallocate(&arrow_allocator, nullptr, 0,
                                      std::numeric_limits<int64_t>::max());
  EXPECT_EQ(buffer, nullptr);

  buffer = arrow_allocator.reallocate(&arrow_allocator, buffer, 0,
                                      std::numeric_limits<int64_t>::max());
  EXPECT_EQ(buffer, nullptr);
#endif
}

TEST(DecimalTest, Decimal32Test) {
  struct ArrowDecimal decimal;
  ArrowDecimalInit(&decimal, 32, 8, 3);

  EXPECT_EQ(decimal.n_words, 0);
  EXPECT_EQ(decimal.precision, 8);
  EXPECT_EQ(decimal.scale, 3);

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW) && ARROW_VERSION_MAJOR >= 18
  auto dec_pos = *Decimal32::FromString("12.345");
  uint8_t bytes_pos[4];
  dec_pos.ToBytes(bytes_pos);

  auto dec_neg = *Decimal32::FromString("-34.567");
  uint8_t bytes_neg[4];
  dec_neg.ToBytes(bytes_neg);
#endif

  ArrowDecimalSetInt(&decimal, 12345);
  EXPECT_EQ(ArrowDecimalGetIntUnsafe(&decimal), 12345);
  EXPECT_EQ(ArrowDecimalSign(&decimal), 1);
#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW) && ARROW_VERSION_MAJOR >= 18
  EXPECT_EQ(memcmp(decimal.words, bytes_pos, sizeof(bytes_pos)), 0);
  ArrowDecimalSetBytes(&decimal, bytes_pos);
  EXPECT_EQ(memcmp(decimal.words, bytes_pos, sizeof(bytes_pos)), 0);
#endif

  ArrowDecimalSetInt(&decimal, -34567);
  EXPECT_EQ(ArrowDecimalGetIntUnsafe(&decimal), -34567);
  EXPECT_EQ(ArrowDecimalSign(&decimal), -1);
#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW) && ARROW_VERSION_MAJOR >= 18
  EXPECT_EQ(memcmp(decimal.words, bytes_neg, sizeof(bytes_neg)), 0);
  ArrowDecimalSetBytes(&decimal, bytes_neg);
  EXPECT_EQ(memcmp(decimal.words, bytes_neg, sizeof(bytes_neg)), 0);
#endif
}

TEST(DecimalTest, Decimal64Test) {
  struct ArrowDecimal decimal;
  ArrowDecimalInit(&decimal, 64, 10, 3);

  EXPECT_EQ(decimal.n_words, 1);
  EXPECT_EQ(decimal.precision, 10);
  EXPECT_EQ(decimal.scale, 3);

  if (_ArrowIsLittleEndian()) {
    EXPECT_EQ(decimal.high_word_index - decimal.low_word_index + 1, decimal.n_words);
  } else {
    EXPECT_EQ(decimal.low_word_index - decimal.high_word_index + 1, decimal.n_words);
  }

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW) && ARROW_VERSION_MAJOR >= 18
  auto dec_pos = *Decimal64::FromString("12.345");
  uint8_t bytes_pos[8];
  dec_pos.ToBytes(bytes_pos);

  auto dec_neg = *Decimal64::FromString("-34.567");
  uint8_t bytes_neg[8];
  dec_neg.ToBytes(bytes_neg);
#endif

  ArrowDecimalSetInt(&decimal, 12345);
  EXPECT_EQ(ArrowDecimalGetIntUnsafe(&decimal), 12345);
  EXPECT_EQ(ArrowDecimalSign(&decimal), 1);
#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW) && ARROW_VERSION_MAJOR >= 18
  EXPECT_EQ(memcmp(decimal.words, bytes_pos, sizeof(bytes_pos)), 0);
  ArrowDecimalSetBytes(&decimal, bytes_pos);
  EXPECT_EQ(memcmp(decimal.words, bytes_pos, sizeof(bytes_pos)), 0);
#endif

  ArrowDecimalSetInt(&decimal, -34567);
  EXPECT_EQ(ArrowDecimalGetIntUnsafe(&decimal), -34567);
  EXPECT_EQ(ArrowDecimalSign(&decimal), -1);
#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW) && ARROW_VERSION_MAJOR >= 18
  EXPECT_EQ(memcmp(decimal.words, bytes_neg, sizeof(bytes_neg)), 0);
  ArrowDecimalSetBytes(&decimal, bytes_neg);
  EXPECT_EQ(memcmp(decimal.words, bytes_neg, sizeof(bytes_neg)), 0);
#endif
}

TEST(DecimalTest, Decimal128Test) {
  struct ArrowDecimal decimal;
  ArrowDecimalInit(&decimal, 128, 10, 3);

  EXPECT_EQ(decimal.n_words, 2);
  EXPECT_EQ(decimal.precision, 10);
  EXPECT_EQ(decimal.scale, 3);

  if (_ArrowIsLittleEndian()) {
    EXPECT_EQ(decimal.high_word_index - decimal.low_word_index + 1, decimal.n_words);
  } else {
    EXPECT_EQ(decimal.low_word_index - decimal.high_word_index + 1, decimal.n_words);
  }

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW)
  auto dec_pos = *Decimal128::FromString("12.345");
  uint8_t bytes_pos[16];
  dec_pos.ToBytes(bytes_pos);

  auto dec_neg = *Decimal128::FromString("-34.567");
  uint8_t bytes_neg[16];
  dec_neg.ToBytes(bytes_neg);

  ArrowDecimalSetInt(&decimal, 12345);
  EXPECT_EQ(ArrowDecimalGetIntUnsafe(&decimal), 12345);
  EXPECT_EQ(ArrowDecimalSign(&decimal), 1);
  EXPECT_EQ(memcmp(decimal.words, bytes_pos, sizeof(bytes_pos)), 0);
  ArrowDecimalSetBytes(&decimal, bytes_pos);
  EXPECT_EQ(memcmp(decimal.words, bytes_pos, sizeof(bytes_pos)), 0);

  ArrowDecimalSetInt(&decimal, -34567);
  EXPECT_EQ(ArrowDecimalGetIntUnsafe(&decimal), -34567);
  EXPECT_EQ(ArrowDecimalSign(&decimal), -1);
  EXPECT_EQ(memcmp(decimal.words, bytes_neg, sizeof(bytes_neg)), 0);
  ArrowDecimalSetBytes(&decimal, bytes_neg);
  EXPECT_EQ(memcmp(decimal.words, bytes_neg, sizeof(bytes_neg)), 0);
#endif
}

TEST(DecimalTest, DecimalNegateTest) {
  using namespace nanoarrow::literals;

  struct ArrowDecimal decimal;
  struct ArrowBuffer buffer;
  ArrowBufferInit(&buffer);

  for (auto bitwidth : {32, 64, 128, 256}) {
    if (bitwidth > 64) {
      ArrowDecimalInit(&decimal, bitwidth, 39, 0);
    } else {
      ArrowDecimalInit(&decimal, bitwidth, 8, 3);
    }

    // Check with a value whose value is contained entirely in the least significant digit
    ArrowDecimalSetInt(&decimal, 12345);
    ArrowDecimalNegate(&decimal);
    EXPECT_EQ(ArrowDecimalGetIntUnsafe(&decimal), -12345);
    ArrowDecimalNegate(&decimal);
    EXPECT_EQ(ArrowDecimalGetIntUnsafe(&decimal), 12345);

    // Check with a value whose negative value will carry into a more significant digit
    memset(decimal.words, 0, sizeof(decimal.words));
    if (bitwidth > 64) {
      decimal.words[decimal.low_word_index] = std::numeric_limits<uint64_t>::max();
    } else if (bitwidth == 64) {
      decimal.words[decimal.low_word_index] = std::numeric_limits<int64_t>::max();
    } else {
      ArrowDecimalSetInt(&decimal, std::numeric_limits<int32_t>::max());
    }
    ASSERT_EQ(ArrowDecimalSign(&decimal), 1);
    ArrowDecimalNegate(&decimal);
    ASSERT_EQ(ArrowDecimalSign(&decimal), -1);
    ArrowDecimalNegate(&decimal);
    ASSERT_EQ(ArrowDecimalSign(&decimal), 1);
    if (bitwidth > 64) {
      EXPECT_EQ(decimal.words[decimal.low_word_index],
                std::numeric_limits<uint64_t>::max());
    } else if (bitwidth == 64) {
      EXPECT_EQ(decimal.words[decimal.low_word_index],
                std::numeric_limits<int64_t>::max());
    } else {
      EXPECT_EQ(ArrowDecimalGetIntUnsafe(&decimal), std::numeric_limits<int32_t>::max());
    }

    if (bitwidth > 64) {
      // Check with a large value that fits in the 128 bit size
      ASSERT_EQ(
          ArrowDecimalSetDigits(&decimal, "123456789012345678901234567890123456789"_asv),
          NANOARROW_OK);
      ArrowDecimalNegate(&decimal);

      buffer.size_bytes = 0;
      ASSERT_EQ(ArrowDecimalAppendDigitsToBuffer(&decimal, &buffer), NANOARROW_OK);
      EXPECT_EQ(std::string(reinterpret_cast<char*>(buffer.data), buffer.size_bytes),
                "-123456789012345678901234567890123456789");
    }
  }

  // Check with a large value that only fits in the 256 bit range
  ArrowDecimalInit(&decimal, 256, 76, 0);
  ASSERT_EQ(ArrowDecimalSetDigits(&decimal,
                                  ArrowCharView("1234567890123456789012345678901234567890"
                                                "12345678901234567890123456789012345")),
            NANOARROW_OK);
  ArrowDecimalNegate(&decimal);

  buffer.size_bytes = 0;
  ASSERT_EQ(ArrowDecimalAppendDigitsToBuffer(&decimal, &buffer), NANOARROW_OK);
  EXPECT_EQ(
      std::string(reinterpret_cast<char*>(buffer.data), buffer.size_bytes),
      "-123456789012345678901234567890123456789012345678901234567890123456789012345");

  ArrowBufferReset(&buffer);
}

TEST(DecimalTest, Decimal256Test) {
  using namespace nanoarrow::literals;

  struct ArrowDecimal decimal;
  ArrowDecimalInit(&decimal, 256, 10, 3);

  EXPECT_EQ(decimal.n_words, 4);
  EXPECT_EQ(decimal.precision, 10);
  EXPECT_EQ(decimal.scale, 3);

  if (_ArrowIsLittleEndian()) {
    EXPECT_EQ(decimal.high_word_index - decimal.low_word_index + 1, decimal.n_words);
  } else {
    EXPECT_EQ(decimal.low_word_index - decimal.high_word_index + 1, decimal.n_words);
  }

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW)
  auto dec_pos = *Decimal256::FromString("12.345");
  uint8_t bytes_pos[32];
  dec_pos.ToBytes(bytes_pos);

  ArrowDecimalSetInt(&decimal, 12345);
  EXPECT_EQ(ArrowDecimalGetIntUnsafe(&decimal), 12345);
  EXPECT_EQ(ArrowDecimalSign(&decimal), 1);
  EXPECT_EQ(memcmp(decimal.words, bytes_pos, sizeof(bytes_pos)), 0);
  ArrowDecimalSetBytes(&decimal, bytes_pos);
  EXPECT_EQ(memcmp(decimal.words, bytes_pos, sizeof(bytes_pos)), 0);

  auto dec_neg = *Decimal256::FromString("-34.567");
  uint8_t bytes_neg[32];
  dec_neg.ToBytes(bytes_neg);

  ArrowDecimalSetInt(&decimal, -34567);
  EXPECT_EQ(ArrowDecimalGetIntUnsafe(&decimal), -34567);
  EXPECT_EQ(ArrowDecimalSign(&decimal), -1);
  EXPECT_EQ(memcmp(decimal.words, bytes_neg, sizeof(bytes_neg)), 0);
  ArrowDecimalSetBytes(&decimal, bytes_neg);
  EXPECT_EQ(memcmp(decimal.words, bytes_neg, sizeof(bytes_neg)), 0);
#endif
}

TEST(DecimalTest, DecimalDigitsTestBasic) {
  using namespace nanoarrow::literals;

  struct ArrowDecimal decimal;
  ArrowDecimalInit(&decimal, 128, 39, 0);

  struct ArrowBuffer buffer;
  ArrowBufferInit(&buffer);

  // Only spans one 32-bit word
  ASSERT_EQ(ArrowDecimalSetDigits(&decimal, "123456"_asv), NANOARROW_OK);
  EXPECT_EQ(ArrowDecimalGetIntUnsafe(&decimal), 123456);

  // Check roundtrip to string
  buffer.size_bytes = 0;
  ASSERT_EQ(ArrowDecimalAppendDigitsToBuffer(&decimal, &buffer), NANOARROW_OK);
  EXPECT_EQ(std::string(reinterpret_cast<char*>(buffer.data), buffer.size_bytes),
            "123456");

  // Negative value
  ASSERT_EQ(ArrowDecimalSetDigits(&decimal, "-123456"_asv), NANOARROW_OK);
  EXPECT_EQ(ArrowDecimalGetIntUnsafe(&decimal), -123456);

  // Check roundtrip to string
  buffer.size_bytes = 0;
  ASSERT_EQ(ArrowDecimalAppendDigitsToBuffer(&decimal, &buffer), NANOARROW_OK);
  EXPECT_EQ(std::string(reinterpret_cast<char*>(buffer.data), buffer.size_bytes),
            "-123456");

  // Spans >1 32-bit word
  ASSERT_EQ(ArrowDecimalSetDigits(&decimal, "1234567899"_asv), NANOARROW_OK);
  EXPECT_EQ(ArrowDecimalGetIntUnsafe(&decimal), 1234567899L);

  // Check roundtrip to string
  buffer.size_bytes = 0;
  ASSERT_EQ(ArrowDecimalAppendDigitsToBuffer(&decimal, &buffer), NANOARROW_OK);
  EXPECT_EQ(std::string(reinterpret_cast<char*>(buffer.data), buffer.size_bytes),
            "1234567899");

  // Check maximum value of a 64-bit integer
  ASSERT_EQ(ArrowDecimalSetDigits(&decimal, "18446744073709551615"_asv), NANOARROW_OK);
  EXPECT_EQ(decimal.words[decimal.low_word_index], std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(decimal.words[decimal.high_word_index], 0);

  // Check roundtrip to string
  buffer.size_bytes = 0;
  ASSERT_EQ(ArrowDecimalAppendDigitsToBuffer(&decimal, &buffer), NANOARROW_OK);
  EXPECT_EQ(std::string(reinterpret_cast<char*>(buffer.data), buffer.size_bytes),
            "18446744073709551615");

  // Check with the maximum value of a signed 128-bit integer
  ASSERT_EQ(
      ArrowDecimalSetDigits(&decimal, "170141183460469231731687303715884105727"_asv),
      NANOARROW_OK);
  EXPECT_EQ(decimal.words[decimal.low_word_index], std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(decimal.words[decimal.high_word_index], std::numeric_limits<int64_t>::max());

  buffer.size_bytes = 0;
  ASSERT_EQ(ArrowDecimalAppendDigitsToBuffer(&decimal, &buffer), NANOARROW_OK);
  EXPECT_EQ(std::string(reinterpret_cast<char*>(buffer.data), buffer.size_bytes),
            "170141183460469231731687303715884105727");

  // Check with the maximum value of a signed 256-bit integer
  ArrowDecimalInit(&decimal, 256, 76, 0);
  ASSERT_EQ(ArrowDecimalSetDigits(&decimal,
                                  ArrowCharView("5789604461865809771178549250434395392663"
                                                "4992332820282019728792003956564819967")),
            NANOARROW_OK);
  EXPECT_EQ(decimal.words[decimal.low_word_index], std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(decimal.words[decimal.high_word_index], std::numeric_limits<int64_t>::max());

  buffer.size_bytes = 0;
  ASSERT_EQ(ArrowDecimalAppendDigitsToBuffer(&decimal, &buffer), NANOARROW_OK);
  EXPECT_EQ(
      std::string(reinterpret_cast<char*>(buffer.data), buffer.size_bytes),
      "57896044618658097711785492504343953926634992332820282019728792003956564819967");

  ArrowBufferReset(&buffer);
}

TEST(DecimalTest, DecimalDigitsTestInvalid) {
  using namespace nanoarrow::literals;

  struct ArrowDecimal decimal;
  ArrowDecimalInit(&decimal, 128, 39, 0);
  EXPECT_EQ(ArrowDecimalSetDigits(&decimal, "this is not an integer"_asv), EINVAL);
}

TEST(DecimalTest, DecimalRoundtripPowerOfTenTest) {
  std::vector<std::pair<int, int>> bitwidth_and_max_precision = {
      {32, 9}, {64, 18}, {128, 38}, {256, 76}};

  for (const auto item : bitwidth_and_max_precision) {
    SCOPED_TRACE(item.first);

    struct ArrowDecimal decimal;
    ArrowDecimalInit(&decimal, item.first, item.second, 0);

    struct ArrowBuffer buffer;
    ArrowBufferInit(&buffer);

    // Generate test strings with positive and negative powers of 10 and check
    // roundtrip back to string.
    std::stringstream ss;

    for (const auto& sign : {"", "-"}) {
      for (int i = 0; i < item.second; i++) {
        ss.str("");
        ss << sign;
        ss << "1";
        for (int j = 0; j < i; j++) {
          ss << "0";
        }

        SCOPED_TRACE(ss.str());
        ASSERT_EQ(ArrowDecimalSetDigits(&decimal, ArrowCharView(ss.str().c_str())),
                  NANOARROW_OK);

        buffer.size_bytes = 0;
        ASSERT_EQ(ArrowDecimalAppendDigitsToBuffer(&decimal, &buffer), NANOARROW_OK);
        ASSERT_EQ(std::string(reinterpret_cast<char*>(buffer.data), buffer.size_bytes),
                  ss.str());
      }
    }

    ArrowBufferReset(&buffer);
  }
}

TEST(DecimalTest, DecimalRoundtripBitshiftTest) {
  struct ArrowDecimal decimal;
  ArrowDecimalInit(&decimal, 256, 76, 0);

  struct ArrowDecimal decimal2;
  ArrowDecimalInit(&decimal2, 256, 76, 0);

  struct ArrowBuffer buffer;
  ArrowBufferInit(&buffer);

  struct ArrowStringView str;

  // Generate test decimals by bitshifting powers of two and check roundtrip
  // through string back to decimal.
  for (const auto is_negative : {false, true}) {
    for (int i = 0; i < 255; i++) {
      SCOPED_TRACE("1 << " + std::to_string(i));

      memset(decimal.words, 0, sizeof(decimal.words));
      int word = i / (8 * sizeof(uint64_t));
      int shift = i % (8 * sizeof(uint64_t));
      if (decimal.low_word_index == 0) {
        decimal.words[word] = 1ULL << shift;
      } else {
        decimal.words[decimal.low_word_index - word] = 1ULL << shift;
      }

      if (is_negative) {
        ArrowDecimalNegate(&decimal);
      }

      buffer.size_bytes = 0;
      ASSERT_EQ(ArrowDecimalAppendDigitsToBuffer(&decimal, &buffer), NANOARROW_OK);
      str.data = reinterpret_cast<char*>(buffer.data);
      str.size_bytes = buffer.size_bytes;

      ASSERT_EQ(ArrowDecimalSetDigits(&decimal2, str), NANOARROW_OK);

      ASSERT_EQ(memcmp(decimal2.words, decimal.words, decimal.n_words * sizeof(uint64_t)),
                0);
    }
  }

  ArrowBufferReset(&buffer);
}

TEST(DecimalTest, DecimalTestStringPositiveScale) {
  using namespace nanoarrow::literals;

  struct ArrowDecimal decimal;
  ArrowDecimalInit(&decimal, 128, 9, 3);

  nanoarrow::UniqueBuffer buffer;

  std::vector<std::pair<struct ArrowStringView, std::string>> items = {
      {"0"_asv, "0.000"},      {"12"_asv, "0.012"},    {"-12"_asv, "-0.012"},
      {"123"_asv, "0.123"},    {"-123"_asv, "-0.123"}, {"1234"_asv, "1.234"},
      {"-1234"_asv, "-1.234"},
  };

  for (const auto& int_and_string : items) {
    buffer->size_bytes = 0;
    ASSERT_EQ(ArrowDecimalSetDigits(&decimal, int_and_string.first), NANOARROW_OK);
    ASSERT_EQ(ArrowDecimalAppendStringToBuffer(&decimal, buffer.get()), NANOARROW_OK);
    EXPECT_EQ(std::string(reinterpret_cast<char*>(buffer->data), buffer->size_bytes),
              int_and_string.second);
  }
}

TEST(DecimalTest, DecimalTestStringZeroScale) {
  using namespace nanoarrow::literals;

  struct ArrowDecimal decimal;
  ArrowDecimalInit(&decimal, 128, 9, 0);

  nanoarrow::UniqueBuffer buffer;

  std::vector<std::pair<struct ArrowStringView, std::string>> items = {
      {"0"_asv, "0"},     {"12"_asv, "12"},     {"-12"_asv, "-12"},
      {"123"_asv, "123"}, {"-123"_asv, "-123"},
  };

  for (const auto& int_and_string : items) {
    buffer->size_bytes = 0;
    ASSERT_EQ(ArrowDecimalSetDigits(&decimal, int_and_string.first), NANOARROW_OK);
    ASSERT_EQ(ArrowDecimalAppendStringToBuffer(&decimal, buffer.get()), NANOARROW_OK);
    EXPECT_EQ(std::string(reinterpret_cast<char*>(buffer->data), buffer->size_bytes),
              int_and_string.second);
  }
}

TEST(DecimalTest, DecimalTestStringNegativeScale) {
  using namespace nanoarrow::literals;

  struct ArrowDecimal decimal;
  ArrowDecimalInit(&decimal, 128, 9, -3);

  nanoarrow::UniqueBuffer buffer;

  std::vector<std::pair<struct ArrowStringView, std::string>> items = {
      {"0"_asv, "0000"},     {"12"_asv, "12000"},     {"-12"_asv, "-12000"},
      {"123"_asv, "123000"}, {"-123"_asv, "-123000"},
  };

  for (const auto& int_and_string : items) {
    buffer->size_bytes = 0;
    ASSERT_EQ(ArrowDecimalSetDigits(&decimal, int_and_string.first), NANOARROW_OK);
    ASSERT_EQ(ArrowDecimalAppendStringToBuffer(&decimal, buffer.get()), NANOARROW_OK);
    EXPECT_EQ(std::string(reinterpret_cast<char*>(buffer->data), buffer->size_bytes),
              int_and_string.second);
  }
}

// test case adapted from
// https://github.com/apache/arrow/blob/main/go/arrow/float16/float16_test.go
TEST(HalfFloatTest, FloatAndHalfFloatRoundTrip) {
  uint16_t cases_bits[] = {
      0x8000, 0x7c00, 0xfc00, 0x3c00, 0x4000, 0xc000,
      0x0000, 0x5b8f, 0xdb8f, 0x48c8, 0xc8c8,
  };
  float cases_float[] = {
      -0.0, INFINITY, -INFINITY, 1, 2, -2, 0, 241.875, -241.875, 9.5625, -9.5625,
  };

  for (size_t i = 0; i < sizeof(cases_float) / sizeof(float); i++) {
    uint16_t bits = ArrowFloatToHalfFloat(cases_float[i]);
    EXPECT_EQ(bits, cases_bits[i]);
    float floats = ArrowHalfFloatToFloat(bits);
    EXPECT_FLOAT_EQ(floats, cases_float[i]);
  }
}

TEST(UtilsTest, ArrowResolveChunk64Test) {
  int64_t offsets[] = {0, 2, 3, 6};
  int64_t n_offsets = 4;

  EXPECT_EQ(ArrowResolveChunk64(0, offsets, 0, n_offsets), 0);
  EXPECT_EQ(ArrowResolveChunk64(1, offsets, 0, n_offsets), 0);
  EXPECT_EQ(ArrowResolveChunk64(2, offsets, 0, n_offsets), 1);
  EXPECT_EQ(ArrowResolveChunk64(3, offsets, 0, n_offsets), 2);
  EXPECT_EQ(ArrowResolveChunk64(4, offsets, 0, n_offsets), 2);
  EXPECT_EQ(ArrowResolveChunk64(5, offsets, 0, n_offsets), 2);
}

TEST(UtilsTest, ArrowResolveChunk32Test) {
  int32_t offsets[] = {0, 2, 3, 6};
  int32_t n_offsets = 4;

  EXPECT_EQ(ArrowResolveChunk32(0, offsets, 0, n_offsets), 0);
  EXPECT_EQ(ArrowResolveChunk32(1, offsets, 0, n_offsets), 0);
  EXPECT_EQ(ArrowResolveChunk32(2, offsets, 0, n_offsets), 1);
  EXPECT_EQ(ArrowResolveChunk32(3, offsets, 0, n_offsets), 2);
  EXPECT_EQ(ArrowResolveChunk32(4, offsets, 0, n_offsets), 2);
  EXPECT_EQ(ArrowResolveChunk32(5, offsets, 0, n_offsets), 2);
}

TEST(MaybeTest, ConstructionAndConversion) {
  using nanoarrow::NA;
  using nanoarrow::internal::Maybe;

  Maybe<int> na, three{3}, five{5};
  EXPECT_FALSE(na);
  EXPECT_TRUE(three);
  EXPECT_TRUE(five);

  EXPECT_EQ(na, NA);
  EXPECT_NE(na, 0);
  EXPECT_EQ(na.value_or(0), 0);

  EXPECT_EQ(three, 3);
  EXPECT_EQ(five, 5);
  EXPECT_NE(five, NA);

  for (auto* l : {&na, &three, &five}) {
    for (auto* r : {&na, &three, &five}) {
      if (l == r) {
        EXPECT_EQ(*l, *r);
      } else {
        EXPECT_NE(*l, *r);
      }
    }
  }

  EXPECT_EQ(*three, 3);
  EXPECT_EQ(*five, 5);
}

TEST(RandomAccessRangeTest, ConstructionAndPrinting) {
  auto square = [](int64_t i) { return i * i; };

  // the range is usable as a constant
  const nanoarrow::internal::RandomAccessRange<decltype(square)> squares{square, 0, 4};

  // gtest recognizes the range as a container and can print it
  EXPECT_THAT(squares, testing::ElementsAre(0, 1, 4, 9));

  // since the range is usable as a constant, we can iterate through it multiple times and
  // it will work
  int64_t sum = 0;
  for (int64_t i : squares) {
    sum += i;
  }
  EXPECT_EQ(sum, 1 + 4 + 9);
}

TEST(InputRangeTest, ConstructionAndPrinting) {
  auto factorial = [i = 1, f = 1]() mutable {
    f *= i++;
    return f < 1000 ? &f : nullptr;
  };
  nanoarrow::internal::InputRange<decltype(factorial)> factorials{factorial};

  // GTest's range utils require a constant range, which this is not
  // EXPECT_THAT(factorials, testing::ElementsAre(1, 2, 6, 24, 120, 720));

  std::vector<int> factorials_vector;
  for (int i : factorials) {
    factorials_vector.push_back(i);
  }
  EXPECT_THAT(factorials_vector, testing::ElementsAre(1, 2, 6, 24, 120, 720));
}
