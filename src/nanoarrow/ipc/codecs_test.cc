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
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "nanoarrow/nanoarrow_ipc.hpp"

// ZSTD compressed little endian int32s [0, 1, 2]
const uint8_t kZstdCompressed012[] = {0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x0c, 0x61,
                                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
                                      0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00};
const uint8_t kUncompressed012[] = {0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
                                    0x00, 0x00, 0x02, 0x00, 0x00, 0x00};

TEST(NanoarrowIpcTest, NanoarrowIpcZstdBuildMatchesRuntime) {
#if defined(NANOARROW_IPC_WITH_ZSTD)
  ASSERT_NE(ArrowIpcGetZstdDecompressionFunction(), nullptr);
#else
  ASSERT_EQ(ArrowIpcGetZstdDecompressionFunction(), nullptr);
#endif
}

TEST(NanoarrowIpcTest, ZstdDecodeValidInput) {
  auto decompress = ArrowIpcGetZstdDecompressionFunction();
  if (!decompress) {
    GTEST_SKIP() << "nanoarrow_ipc not built with NANOARROW_IPC_WITH_ZSTD";
  }

  // Empty->empty seems to work
  struct ArrowError error {};
  EXPECT_EQ(decompress({{nullptr}, 0}, nullptr, 0, &error), NANOARROW_OK);

  // Check a decompress of a valid compressed buffer
  uint8_t out[16];
  std::memset(out, 0, sizeof(out));
  ASSERT_EQ(decompress({{&kZstdCompressed012}, sizeof(kZstdCompressed012)}, out,
                       sizeof(kUncompressed012), &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_TRUE(std::memcmp(out, kUncompressed012, sizeof(kUncompressed012)) == 0);

  ASSERT_EQ(decompress({{kZstdCompressed012}, sizeof(kZstdCompressed012)}, out,
                       sizeof(kUncompressed012) + 1, &error),
            EIO);
  EXPECT_STREQ(error.message, "Expected decompressed size of 13 bytes but got 12 bytes");
}

TEST(NanoarrowIpcTest, ZstdDecodeInvalidInput) {
  auto decompress = ArrowIpcGetZstdDecompressionFunction();
  if (!decompress) {
    GTEST_SKIP() << "nanoarrow_ipc not built with NANOARROW_IPC_WITH_ZSTD";
  }

  struct ArrowError error {};
  const char* bad_data = "abcde";
  EXPECT_EQ(decompress({{bad_data}, 5}, nullptr, 0, &error), EIO);
  EXPECT_THAT(error.message,
              ::testing::StartsWith("ZSTD_decompress([buffer with 5 bytes] -> [buffer "
                                    "with 0 bytes]) failed with error"));
}

// LZ4 compressed little endian int32s [0, 1, 2]
const uint8_t kLZ4Compressed012[] = {
    0x04, 0x22, 0x4d, 0x18, 0x60, 0x40, 0x82, 0x0c, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

TEST(NanoarrowIpcTest, NanoarrowIpcLZ4BuildMatchesRuntime) {
#if defined(NANOARROW_IPC_WITH_LZ4)
  ASSERT_NE(ArrowIpcGetLZ4DecompressionFunction(), nullptr);
#else
  ASSERT_EQ(ArrowIpcGetLZ4DecompressionFunction(), nullptr);
#endif
}

TEST(NanoarrowIpcTest, LZ4DecodeValidInput) {
  auto decompress = ArrowIpcGetLZ4DecompressionFunction();
  if (!decompress) {
    GTEST_SKIP() << "nanoarrow_ipc not built with NANOARROW_IPC_WITH_LZ4";
  }

  struct ArrowError error {};

  // Check a decompress of a valid compressed buffer
  uint8_t out[16];
  std::memset(out, 0, sizeof(out));
  ASSERT_EQ(decompress({{&kLZ4Compressed012}, sizeof(kLZ4Compressed012)}, out,
                       sizeof(kUncompressed012), &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_TRUE(std::memcmp(out, kUncompressed012, sizeof(kUncompressed012)) == 0);

  ASSERT_EQ(decompress({{kLZ4Compressed012}, sizeof(kLZ4Compressed012)}, out,
                       sizeof(kUncompressed012) + 1, &error),
            EIO);
  EXPECT_STREQ(error.message, "Expected decompressed size of 13 bytes but got 12 bytes");
}

TEST(NanoarrowIpcTest, LZ4DecodeInvalidInput) {
  auto decompress = ArrowIpcGetLZ4DecompressionFunction();
  if (!decompress) {
    GTEST_SKIP() << "nanoarrow_ipc not built with NANOARROW_IPC_WITH_LZ4";
  }

  struct ArrowError error {};
  uint8_t out[16];
  std::memset(out, 0, sizeof(out));

  // LZ4_decompress() needs almost correct data to trigger this failure branch
  uint8_t src[16];
  memcpy(src, kLZ4Compressed012, sizeof(src));
  src[5] = 0xff;
  ASSERT_EQ(decompress({{&src}, sizeof(src)}, out, sizeof(kUncompressed012), &error),
            EIO);
  EXPECT_THAT(
      error.message,
      ::testing::StartsWith(
          "LZ4F_decompress([buffer with 16 bytes] -> [buffer with 12 bytes]) failed"));

  // Nonsensical data triggers a different failure branch
  const char* bad_data = "abcde";
  EXPECT_EQ(decompress({{bad_data}, 5}, nullptr, 0, &error), EIO);
  EXPECT_THAT(error.message,
              ::testing::StartsWith(
                  "Expected complete LZ4 frame but found frame with 6 bytes remaining"));
}

TEST(NanoarrowIpcTest, SerialDecompressor) {
  struct ArrowError error {};
  nanoarrow::ipc::UniqueDecompressor decompressor;

  ASSERT_EQ(ArrowIpcSerialDecompressor(decompressor.get()), NANOARROW_OK);

  // Check the function setter error
  ASSERT_EQ(ArrowIpcSerialDecompressorSetFunction(
                decompressor.get(), NANOARROW_IPC_COMPRESSION_TYPE_NONE, nullptr),
            EINVAL);

  // NONE is not a codec that can be used to decompress
  EXPECT_EQ(decompressor->decompress_add(decompressor.get(),
                                         NANOARROW_IPC_COMPRESSION_TYPE_NONE,
                                         {{nullptr}, 0}, nullptr, 0, &error),
            EINVAL);
  EXPECT_STREQ(error.message, "Unknown decompression type with value 0");

  // The serial decompressor never waits and always succeeds when requested to
  EXPECT_EQ(decompressor->decompress_wait(decompressor.get(), 0, &error), NANOARROW_OK);

  // Check a decompress for a supported codec if we have one (or for an error if we don't)
  uint8_t out[12];
  std::memset(out, 0, sizeof(out));
  if (ArrowIpcGetZstdDecompressionFunction() != nullptr) {
    EXPECT_EQ(decompressor->decompress_add(
                  decompressor.get(), NANOARROW_IPC_COMPRESSION_TYPE_ZSTD,
                  {{&kZstdCompressed012}, sizeof(kZstdCompressed012)}, out, sizeof(out),
                  &error),
              NANOARROW_OK);
  } else {
    EXPECT_EQ(decompressor->decompress_add(decompressor.get(),
                                           NANOARROW_IPC_COMPRESSION_TYPE_ZSTD,
                                           {{nullptr}, 0}, nullptr, 0, &error),
              ENOTSUP);
    EXPECT_STREQ(
        error.message,
        "Compression type with value 2 not supported by this build of nanoarrow");
  }

  // Either way, if we explicitly remove support for a codec, we should get an error
  ASSERT_EQ(ArrowIpcSerialDecompressorSetFunction(
                decompressor.get(), NANOARROW_IPC_COMPRESSION_TYPE_ZSTD, nullptr),
            NANOARROW_OK);
  EXPECT_EQ(decompressor->decompress_add(decompressor.get(),
                                         NANOARROW_IPC_COMPRESSION_TYPE_ZSTD,
                                         {{nullptr}, 0}, nullptr, 0, &error),
            ENOTSUP);
  EXPECT_STREQ(error.message,
               "Compression type with value 2 not supported by this build of nanoarrow");
}

TEST(NanoarrowIpcTest, CompressionTypeStrings) {
  EXPECT_STREQ(ArrowIpcCompressionTypeToString(NANOARROW_IPC_COMPRESSION_TYPE_NONE),
               "none");
  EXPECT_STREQ(ArrowIpcCompressionTypeToString(NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME),
               "lz4");
  EXPECT_STREQ(ArrowIpcCompressionTypeToString(NANOARROW_IPC_COMPRESSION_TYPE_ZSTD),
               "zstd");
  // 3 is not an enumerator but is within the enum's value range (unlike, e.g., 99,
  // which C++ can't represent in this enum)
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  auto unknown_type = static_cast<enum ArrowIpcCompressionType>(3);
  EXPECT_STREQ(ArrowIpcCompressionTypeToString(unknown_type),
               "<unknown compression type>");

  struct ArrowError error {};
  for (auto type :
       {NANOARROW_IPC_COMPRESSION_TYPE_NONE, NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME,
        NANOARROW_IPC_COMPRESSION_TYPE_ZSTD}) {
    enum ArrowIpcCompressionType parsed = unknown_type;
    ASSERT_EQ(ArrowIpcCompressionTypeFromString(ArrowIpcCompressionTypeToString(type),
                                                &parsed, &error),
              NANOARROW_OK)
        << error.message;
    EXPECT_EQ(parsed, type);
  }

  enum ArrowIpcCompressionType parsed = NANOARROW_IPC_COMPRESSION_TYPE_NONE;
  EXPECT_EQ(ArrowIpcCompressionTypeFromString("LZ4", &parsed, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Unknown compression type name 'LZ4' (expected 'none', 'lz4', or 'zstd')");
  EXPECT_EQ(ArrowIpcCompressionTypeFromString("", &parsed, &error), EINVAL);
  EXPECT_EQ(
      ArrowIpcCompressionTypeFromString("<unknown compression type>", &parsed, &error),
      EINVAL);
  EXPECT_EQ(ArrowIpcCompressionTypeFromString(nullptr, &parsed, &error), EINVAL);
  EXPECT_STREQ(error.message,
               "Unknown compression type name '' (expected 'none', 'lz4', or 'zstd')");
  // A failed lookup leaves the output untouched
  EXPECT_EQ(parsed, NANOARROW_IPC_COMPRESSION_TYPE_NONE);
}

TEST(NanoarrowIpcTest, CompressionLevelRange) {
  int min_level = 1;
  int max_level = -1;
  EXPECT_EQ(ArrowIpcGetCompressionLevelRange(NANOARROW_IPC_COMPRESSION_TYPE_NONE,
                                             &min_level, &max_level),
            EINVAL);
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  auto unknown_type = static_cast<enum ArrowIpcCompressionType>(3);
  EXPECT_EQ(ArrowIpcGetCompressionLevelRange(unknown_type, &min_level, &max_level),
            EINVAL);

  if (ArrowIpcGetLZ4CompressionFunction() != nullptr) {
    ASSERT_EQ(ArrowIpcGetCompressionLevelRange(NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME,
                                               &min_level, &max_level),
              NANOARROW_OK);
    EXPECT_EQ(min_level, -65536);
    EXPECT_EQ(max_level, 12);
  } else {
    EXPECT_EQ(ArrowIpcGetCompressionLevelRange(NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME,
                                               &min_level, &max_level),
              ENOTSUP);
  }

  if (ArrowIpcGetZstdCompressionFunction() != nullptr) {
    ASSERT_EQ(ArrowIpcGetCompressionLevelRange(NANOARROW_IPC_COMPRESSION_TYPE_ZSTD,
                                               &min_level, &max_level),
              NANOARROW_OK);
    // The levels used by the roundtrip tests below must be in range
    EXPECT_LE(min_level, -5);
    EXPECT_LE(min_level, NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT);
    EXPECT_GE(max_level, 19);
  } else {
    EXPECT_EQ(ArrowIpcGetCompressionLevelRange(NANOARROW_IPC_COMPRESSION_TYPE_ZSTD,
                                               &min_level, &max_level),
              ENOTSUP);
  }
}

// Compress input at compression_level (appending to a buffer that already has content),
// decompress the appended bytes, and check that the result matches the input. Returns
// the number of compressed bytes that were appended.
static int64_t TestCompressRoundtrip(ArrowIpcCompressFunction compress,
                                     ArrowIpcDecompressFunction decompress,
                                     int compression_level,
                                     const std::vector<uint8_t>& input) {
  struct ArrowError error {};
  nanoarrow::UniqueBuffer compressed;

  // Content already in dst must be preserved (compress functions only append)
  const char* existing = "existing";
  const int64_t existing_size = 8;
  EXPECT_EQ(ArrowBufferAppend(compressed.get(), existing, existing_size), NANOARROW_OK);

  EXPECT_EQ(compress({{input.data()}, static_cast<int64_t>(input.size())},
                     compression_level, compressed.get(), &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_GT(compressed->size_bytes, existing_size);
  EXPECT_EQ(std::memcmp(compressed->data, existing, existing_size), 0);

  std::vector<uint8_t> output(input.size());
  struct ArrowBufferView compressed_view = {{compressed->data + existing_size},
                                            compressed->size_bytes - existing_size};
  EXPECT_EQ(decompress(compressed_view, output.data(),
                       static_cast<int64_t>(output.size()), &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_EQ(output, input);

  return compressed->size_bytes - existing_size;
}

static std::vector<uint8_t> CompressibleInput(size_t n) {
  std::vector<uint8_t> out(n);
  for (size_t i = 0; i < n; i++) {
    out[i] = static_cast<uint8_t>(i % 7);
  }
  return out;
}

// Check compress/decompress on empty, small, and multi-block inputs at each level
static void TestCompressionFunctions(ArrowIpcCompressFunction compress,
                                     ArrowIpcDecompressFunction decompress,
                                     const std::vector<int>& compression_levels) {
  ASSERT_NE(compress, nullptr);
  ASSERT_NE(decompress, nullptr);

  auto input = CompressibleInput(1 << 20);
  int64_t default_size = 0;
  for (int level : compression_levels) {
    SCOPED_TRACE("compression level " + std::to_string(level));
    TestCompressRoundtrip(compress, decompress, level, {});
    TestCompressRoundtrip(
        compress, decompress, level,
        std::vector<uint8_t>(kUncompressed012,
                             kUncompressed012 + sizeof(kUncompressed012)));

    // Large enough to span several blocks; a repetitive input must actually shrink
    int64_t compressed_size = TestCompressRoundtrip(compress, decompress, level, input);
    EXPECT_LT(compressed_size, static_cast<int64_t>(input.size() / 10));
    if (level == NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT) {
      default_size = compressed_size;
    }
  }

  // High acceleration must trade compression ratio for speed on this input. Merely
  // roundtripping at several levels would also pass if the level were ignored.
  ASSERT_GT(default_size, 0);
  int64_t accelerated_size = TestCompressRoundtrip(compress, decompress, -65536, input);
  EXPECT_GT(accelerated_size, default_size * 2);
}

TEST(NanoarrowIpcTest, NanoarrowIpcZstdCompressBuildMatchesRuntime) {
#if defined(NANOARROW_IPC_WITH_ZSTD)
  ASSERT_NE(ArrowIpcGetZstdCompressionFunction(), nullptr);
#else
  ASSERT_EQ(ArrowIpcGetZstdCompressionFunction(), nullptr);
#endif
}

TEST(NanoarrowIpcTest, ZstdCompressRoundtrip) {
  if (ArrowIpcGetZstdCompressionFunction() == nullptr) {
    GTEST_SKIP() << "nanoarrow_ipc not built with NANOARROW_IPC_WITH_ZSTD";
  }
  // Default, a negative (fast) level, the lowest regular level, and a high level
  TestCompressionFunctions(ArrowIpcGetZstdCompressionFunction(),
                           ArrowIpcGetZstdDecompressionFunction(),
                           {NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT, -5, 1, 19});
}

TEST(NanoarrowIpcTest, NanoarrowIpcLZ4CompressBuildMatchesRuntime) {
#if defined(NANOARROW_IPC_WITH_LZ4)
  ASSERT_NE(ArrowIpcGetLZ4CompressionFunction(), nullptr);
#else
  ASSERT_EQ(ArrowIpcGetLZ4CompressionFunction(), nullptr);
#endif
}

TEST(NanoarrowIpcTest, LZ4CompressRoundtrip) {
  if (ArrowIpcGetLZ4CompressionFunction() == nullptr) {
    GTEST_SKIP() << "nanoarrow_ipc not built with NANOARROW_IPC_WITH_LZ4";
  }
  // Default (fast), acceleration, the last fast level, and LZ4HC levels
  TestCompressionFunctions(ArrowIpcGetLZ4CompressionFunction(),
                           ArrowIpcGetLZ4DecompressionFunction(),
                           {NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT, -1, 2, 9, 12});
}

TEST(NanoarrowIpcTest, LZ4CompressMinimumLevels) {
  auto compress = ArrowIpcGetLZ4CompressionFunction();
  if (compress == nullptr) {
    GTEST_SKIP() << "nanoarrow_ipc not built with NANOARROW_IPC_WITH_LZ4";
  }
  auto decompress = ArrowIpcGetLZ4DecompressionFunction();
  ASSERT_NE(decompress, nullptr);

  auto input = CompressibleInput(1 << 20);
  int64_t accelerated_size = TestCompressRoundtrip(compress, decompress, -65536, input);
  for (int level : {std::numeric_limits<int>::min(), std::numeric_limits<int>::min() + 1,
                    std::numeric_limits<int>::min() + 2}) {
    SCOPED_TRACE("compression level " + std::to_string(level));
    // The most negative levels must saturate at maximum acceleration rather than
    // overflow and fall back to the default compression level.
    EXPECT_EQ(TestCompressRoundtrip(compress, decompress, level, input),
              accelerated_size);
  }
}

// An allocator whose reallocate() fails on the fail_on-th call (1-based) and otherwise
// delegates to the default allocator
struct FailingAllocatorState {
  int calls;
  int fail_on;
};

static uint8_t* FailingReallocate(struct ArrowBufferAllocator* allocator, uint8_t* ptr,
                                  int64_t old_size, int64_t new_size) {
  auto* state = static_cast<FailingAllocatorState*>(allocator->private_data);
  auto default_allocator = ArrowBufferAllocatorDefault();
  if (++state->calls == state->fail_on) {
    // nanoarrow discards the buffer on failure, so the old allocation is freed here
    default_allocator.free(&default_allocator, ptr, old_size);
    return nullptr;
  }
  return default_allocator.reallocate(&default_allocator, ptr, old_size, new_size);
}

static void FailingFree(struct ArrowBufferAllocator* allocator, uint8_t* ptr,
                        int64_t size) {
  NANOARROW_UNUSED(allocator);
  auto default_allocator = ArrowBufferAllocatorDefault();
  default_allocator.free(&default_allocator, ptr, size);
}

static struct ArrowBufferAllocator FailingAllocator(FailingAllocatorState* state) {
  struct ArrowBufferAllocator allocator = ArrowBufferAllocatorDefault();
  allocator.reallocate = &FailingReallocate;
  allocator.free = &FailingFree;
  allocator.private_data = state;
  return allocator;
}

TEST(NanoarrowIpcTest, CompressAllocationFailure) {
  struct ArrowError error {};
  for (auto compress :
       {ArrowIpcGetLZ4CompressionFunction(), ArrowIpcGetZstdCompressionFunction()}) {
    if (compress == nullptr) {
      continue;
    }

    FailingAllocatorState state{0, 1};
    nanoarrow::UniqueBuffer dst;
    ASSERT_EQ(ArrowBufferSetAllocator(dst.get(), FailingAllocator(&state)), NANOARROW_OK);
    EXPECT_EQ(compress({{kUncompressed012}, sizeof(kUncompressed012)},
                       NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT, dst.get(), &error),
              ENOMEM);
    EXPECT_THAT(error.message, ::testing::HasSubstr("ArrowBufferReserve"));
    EXPECT_EQ(state.calls, 1);
  }
}

// A stand-in compression function that always fails
static ArrowErrorCode FailCompress(struct ArrowBufferView src, int compression_level,
                                   struct ArrowBuffer* dst, struct ArrowError* error) {
  NANOARROW_UNUSED(src);
  NANOARROW_UNUSED(compression_level);
  NANOARROW_UNUSED(dst);
  ArrowErrorSet(error, "FailCompress() failed");
  return EIO;
}

// A stand-in compression function that records the level it was called with and
// "compresses" by copying
static int last_compression_level = 0;

static ArrowErrorCode RecordLevelAndCopy(struct ArrowBufferView src,
                                         int compression_level, struct ArrowBuffer* dst,
                                         struct ArrowError* error) {
  NANOARROW_UNUSED(error);
  last_compression_level = compression_level;
  return ArrowBufferAppend(dst, src.data.data, src.size_bytes);
}

TEST(NanoarrowIpcTest, SerialCompressor) {
  struct ArrowError error {};
  nanoarrow::ipc::UniqueCompressor compressor;

  // An invalid compression type is rejected at construction
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  auto unknown_type = static_cast<enum ArrowIpcCompressionType>(3);
  EXPECT_EQ(ArrowIpcSerialCompressor(compressor.get(), unknown_type,
                                     NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT),
            EINVAL);
  EXPECT_EQ(compressor->release, nullptr);

  ASSERT_EQ(
      ArrowIpcSerialCompressor(compressor.get(), NANOARROW_IPC_COMPRESSION_TYPE_NONE,
                               NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT),
      NANOARROW_OK);
  EXPECT_EQ(compressor->compression_type, NANOARROW_IPC_COMPRESSION_TYPE_NONE);

  // Check the function setter error
  ASSERT_EQ(ArrowIpcSerialCompressorSetFunction(
                compressor.get(), NANOARROW_IPC_COMPRESSION_TYPE_NONE, nullptr),
            EINVAL);

  // The serial compressor never waits and always succeeds when requested to
  EXPECT_EQ(compressor->compress_wait(compressor.get(), 0, &error), NANOARROW_OK);

  // NONE is not a codec that can be used to compress
  nanoarrow::UniqueBuffer dst;
  EXPECT_EQ(compressor->compress_add(compressor.get(), {{nullptr}, 0}, dst.get(), &error),
            EINVAL);
  EXPECT_STREQ(error.message, "Unknown compression type with value 0");

  // Check a compress for a supported codec if we have one (or for an error if we don't)
  compressor->compression_type = NANOARROW_IPC_COMPRESSION_TYPE_ZSTD;
  if (ArrowIpcGetZstdCompressionFunction() != nullptr) {
    ASSERT_EQ(compressor->compress_add(compressor.get(),
                                       {{kUncompressed012}, sizeof(kUncompressed012)},
                                       dst.get(), &error),
              NANOARROW_OK)
        << error.message;
    ASSERT_EQ(compressor->compress_wait(compressor.get(), -1, &error), NANOARROW_OK);
    ASSERT_GT(dst->size_bytes, 0);

    uint8_t out[sizeof(kUncompressed012)];
    std::memset(out, 0, sizeof(out));
    ASSERT_EQ(ArrowIpcGetZstdDecompressionFunction()({{dst->data}, dst->size_bytes}, out,
                                                     sizeof(out), &error),
              NANOARROW_OK)
        << error.message;
    EXPECT_TRUE(std::memcmp(out, kUncompressed012, sizeof(kUncompressed012)) == 0);
  } else {
    EXPECT_EQ(
        compressor->compress_add(compressor.get(), {{nullptr}, 0}, dst.get(), &error),
        ENOTSUP);
    EXPECT_STREQ(
        error.message,
        "Compression type with value 2 not supported by this build of nanoarrow");
  }

  // Either way, if we explicitly remove support for a codec, we should get an error
  ASSERT_EQ(ArrowIpcSerialCompressorSetFunction(
                compressor.get(), NANOARROW_IPC_COMPRESSION_TYPE_ZSTD, nullptr),
            NANOARROW_OK);
  EXPECT_EQ(compressor->compress_add(compressor.get(), {{nullptr}, 0}, dst.get(), &error),
            ENOTSUP);
  EXPECT_STREQ(error.message,
               "Compression type with value 2 not supported by this build of nanoarrow");

  // The compression level given at construction is passed to the function for the codec
  for (int level : {NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT, 7}) {
    nanoarrow::ipc::UniqueCompressor leveled;
    ASSERT_EQ(ArrowIpcSerialCompressor(leveled.get(),
                                       NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME, level),
              NANOARROW_OK);
    ASSERT_EQ(
        ArrowIpcSerialCompressorSetFunction(
            leveled.get(), NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME, &RecordLevelAndCopy),
        NANOARROW_OK);
    dst->size_bytes = 0;
    last_compression_level = -1;
    ASSERT_EQ(leveled->compress_add(leveled.get(),
                                    {{kUncompressed012}, sizeof(kUncompressed012)},
                                    dst.get(), &error),
              NANOARROW_OK)
        << error.message;
    EXPECT_EQ(last_compression_level, level);
    ASSERT_EQ(dst->size_bytes, static_cast<int64_t>(sizeof(kUncompressed012)));
    EXPECT_EQ(std::memcmp(dst->data, kUncompressed012, sizeof(kUncompressed012)), 0);
  }

  compressor->compression_type = NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME;
  // Errors from the function for the codec are propagated
  ASSERT_EQ(
      ArrowIpcSerialCompressorSetFunction(
          compressor.get(), NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME, &FailCompress),
      NANOARROW_OK);
  EXPECT_EQ(compressor->compress_add(compressor.get(),
                                     {{kUncompressed012}, sizeof(kUncompressed012)},
                                     dst.get(), &error),
            EIO);
  EXPECT_STREQ(error.message, "FailCompress() failed");
}
