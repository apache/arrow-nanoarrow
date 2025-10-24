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
