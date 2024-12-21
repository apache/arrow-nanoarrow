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

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "nanoarrow/nanoarrow_ipc.hpp"

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
  struct ArrowError error{};
  EXPECT_EQ(decompress({{nullptr}, 0}, nullptr, 0, &error), NANOARROW_OK);

  // Check a decompress of little endian int32s [0, 1, 2]
  const uint8_t compressed012[] = {0x28, 0xb5, 0x2f, 0xfd, 0x20, 0x0c, 0x61,
                                   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
                                   0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00};
  const uint8_t uncompressed012[] = {0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
                                     0x00, 0x00, 0x02, 0x00, 0x00, 0x00};
  uint8_t out[16];
  std::memset(out, 0, sizeof(out));

  ASSERT_EQ(decompress({{&compressed012}, sizeof(compressed012)}, out,
                       sizeof(uncompressed012), &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_TRUE(std::memcmp(out, uncompressed012, sizeof(uncompressed012)) == 0);

  ASSERT_EQ(decompress({{compressed012}, sizeof(compressed012)}, out,
                       sizeof(uncompressed012) + 1, &error),
            EIO);
  EXPECT_STREQ(error.message, "Expected decompressed size of 13 bytes but got 12 bytes");
}

TEST(NanoarrowIpcTest, ZstdDecodeInvalidInput) {
  auto decompress = ArrowIpcGetZstdDecompressionFunction();
  if (!decompress) {
    GTEST_SKIP() << "nanoarrow_ipc not built with NANOARROW_IPC_WITH_ZSTD";
  }

  struct ArrowError error{};
  const char* bad_data = "abcde";
  EXPECT_EQ(decompress({{bad_data}, 5}, nullptr, 0, &error), EIO);
  EXPECT_THAT(error.message,
              ::testing::StartsWith("ZSTD_decompress([buffer with 5 bytes] -> [buffer "
                                    "with 0 bytes]) failed with error"));
}
