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

#include <gtest/gtest.h>

#include "nanoarrow/nanoarrow.h"

TEST(BitmapTest, BitmapTestElement) {
  uint8_t bitmap[10];

  memset(bitmap, 0xff, sizeof(bitmap));
  for (int i = 0; i < sizeof(bitmap) * 8; i++) {
    EXPECT_EQ(ArrowBitmapGetBit(bitmap, i), 1);
  }

  bitmap[2] = 0xfd;
  EXPECT_EQ(ArrowBitmapGetBit(bitmap, 16 + 0), 1);
  EXPECT_EQ(ArrowBitmapGetBit(bitmap, 16 + 1), 0);
  EXPECT_EQ(ArrowBitmapGetBit(bitmap, 16 + 2), 1);
  EXPECT_EQ(ArrowBitmapGetBit(bitmap, 16 + 3), 1);
  EXPECT_EQ(ArrowBitmapGetBit(bitmap, 16 + 4), 1);
  EXPECT_EQ(ArrowBitmapGetBit(bitmap, 16 + 5), 1);
  EXPECT_EQ(ArrowBitmapGetBit(bitmap, 16 + 6), 1);
  EXPECT_EQ(ArrowBitmapGetBit(bitmap, 16 + 7), 1);

  memset(bitmap, 0x00, sizeof(bitmap));
  for (int i = 0; i < sizeof(bitmap) * 8; i++) {
    EXPECT_EQ(ArrowBitmapGetBit(bitmap, i), 0);
  }

  bitmap[2] = 0x02;
  EXPECT_EQ(ArrowBitmapGetBit(bitmap, 16 + 0), 0);
  EXPECT_EQ(ArrowBitmapGetBit(bitmap, 16 + 1), 1);
  EXPECT_EQ(ArrowBitmapGetBit(bitmap, 16 + 2), 0);
  EXPECT_EQ(ArrowBitmapGetBit(bitmap, 16 + 3), 0);
  EXPECT_EQ(ArrowBitmapGetBit(bitmap, 16 + 4), 0);
  EXPECT_EQ(ArrowBitmapGetBit(bitmap, 16 + 5), 0);
  EXPECT_EQ(ArrowBitmapGetBit(bitmap, 16 + 6), 0);
  EXPECT_EQ(ArrowBitmapGetBit(bitmap, 16 + 7), 0);
}

TEST(BitmapTest, BitmapTestSetTo) {
  uint8_t bitmap[10];

  memset(bitmap, 0xff, sizeof(bitmap));
  ArrowBitmapSetBitTo(bitmap, 16 + 1, 0);
  EXPECT_EQ(bitmap[2], 0xfd);
  ArrowBitmapSetBitTo(bitmap, 16 + 1, 1);
  EXPECT_EQ(bitmap[2], 0xff);

  memset(bitmap, 0xff, sizeof(bitmap));
  ArrowBitmapClearBit(bitmap, 16 + 1);
  EXPECT_EQ(bitmap[2], 0xfd);
  ArrowBitmapSetBit(bitmap, 16 + 1);
  EXPECT_EQ(bitmap[2], 0xff);

  memset(bitmap, 0x00, sizeof(bitmap));
  ArrowBitmapSetBitTo(bitmap, 16 + 1, 1);
  EXPECT_EQ(bitmap[2], 0x02);
  ArrowBitmapSetBitTo(bitmap, 16 + 1, 0);
  EXPECT_EQ(bitmap[2], 0x00);

  memset(bitmap, 0x00, sizeof(bitmap));
  ArrowBitmapSetBit(bitmap, 16 + 1);
  EXPECT_EQ(bitmap[2], 0x02);
  ArrowBitmapClearBit(bitmap, 16 + 1);
  EXPECT_EQ(bitmap[2], 0x00);
}

TEST(BitmapTest, BitmapTestCountSet) {
  uint8_t bitmap[10];
  memset(bitmap, 0x00, sizeof(bitmap));
  ArrowBitmapSetBit(bitmap, 18);
  ArrowBitmapSetBit(bitmap, 23);
  ArrowBitmapSetBit(bitmap, 74);

  EXPECT_EQ(ArrowBitmapCountSet(bitmap, 0, 80), 3);
  EXPECT_EQ(ArrowBitmapCountSet(bitmap, 18, 57), 3);

  EXPECT_EQ(ArrowBitmapCountSet(bitmap, 18, 0), 0);
  EXPECT_EQ(ArrowBitmapCountSet(bitmap, 18, 1), 1);
  EXPECT_EQ(ArrowBitmapCountSet(bitmap, 18, 2), 1);
  EXPECT_EQ(ArrowBitmapCountSet(bitmap, 18, 3), 1);
  EXPECT_EQ(ArrowBitmapCountSet(bitmap, 18, 4), 1);
  EXPECT_EQ(ArrowBitmapCountSet(bitmap, 18, 5), 1);
  EXPECT_EQ(ArrowBitmapCountSet(bitmap, 18, 6), 2);

  EXPECT_EQ(ArrowBitmapCountSet(bitmap, 23, 1), 1);
}

TEST(BitmapTest, BitmapTestBuilderAppend) {
  int8_t test_values[65];
  memset(test_values, 0, sizeof(test_values));
  test_values[4] = 1;
  test_values[63] = 1;
  test_values[64] = 1;

  struct ArrowBitmapBuilder bitmap_builder;
  ArrowBitmapBuilderInit(&bitmap_builder);

  for (int64_t i = 0; i < 65; i++) {
    ASSERT_EQ(ArrowBitmapBuilderAppend(&bitmap_builder, test_values[i], 1), NANOARROW_OK);
  }

  EXPECT_EQ(bitmap_builder.size_bits, 65);
  EXPECT_EQ(ArrowBitmapGetBit(bitmap_builder.buffer.data, 4), test_values[4]);
  for (int i = 0; i < 65; i++) {
    EXPECT_EQ(ArrowBitmapGetBit(bitmap_builder.buffer.data, i), test_values[i]);
  }

  ArrowBitmapBuilderReset(&bitmap_builder);
}

TEST(BitmapTest, BitmapTestBuilderAppendUnsafe) {
  int8_t test_values[65];
  memset(test_values, 0, sizeof(test_values));
  test_values[4] = 1;
  test_values[63] = 1;
  test_values[64] = 1;

  struct ArrowBitmapBuilder bitmap_builder;
  ArrowBitmapBuilderInit(&bitmap_builder);

  ASSERT_EQ(ArrowBitmapBuilderReserve(&bitmap_builder, 65), NANOARROW_OK);
  ArrowBitmapBuilderAppendInt8Unsafe(&bitmap_builder, test_values, 65);

  EXPECT_EQ(bitmap_builder.size_bits, 65);
  EXPECT_EQ(ArrowBitmapGetBit(bitmap_builder.buffer.data, 4), test_values[4]);
  for (int i = 0; i < 65; i++) {
    EXPECT_EQ(ArrowBitmapGetBit(bitmap_builder.buffer.data, i), test_values[i]);
  }

  ArrowBitmapBuilderReset(&bitmap_builder);

  int32_t test_values_int32[65];
  memset(test_values_int32, 0, sizeof(test_values_int32));
  test_values[4] = 1;
  test_values[63] = 1;
  test_values[64] = 1;

  ASSERT_EQ(ArrowBitmapBuilderReserve(&bitmap_builder, 65), NANOARROW_OK);
  ArrowBitmapBuilderAppendInt32Unsafe(&bitmap_builder, test_values_int32, 65);

  EXPECT_EQ(bitmap_builder.size_bits, 65);
  for (int i = 0; i < 65; i++) {
    EXPECT_EQ(ArrowBitmapGetBit(bitmap_builder.buffer.data, i), test_values_int32[i]);
  }

  ArrowBitmapBuilderReset(&bitmap_builder);
}
