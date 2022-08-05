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

  struct ArrowBitmap bitmap;
  ArrowBitmapInit(&bitmap);

  for (int64_t i = 0; i < 65; i++) {
    ASSERT_EQ(ArrowBitmapAppend(&bitmap, test_values[i], 1), NANOARROW_OK);
  }

  EXPECT_EQ(bitmap.size_bits, 65);
  EXPECT_EQ(ArrowBitmapGetBit(bitmap.buffer.data, 4), test_values[4]);
  for (int i = 0; i < 65; i++) {
    EXPECT_EQ(ArrowBitmapGetBit(bitmap.buffer.data, i), test_values[i]);
  }

  ArrowBitmapReset(&bitmap);
}

TEST(BitmapTest, BitmapTestBuilderAppendInt8Unsafe) {
  struct ArrowBitmap bitmap;
  ArrowBitmapInit(&bitmap);

  // 68 because this will end in the middle of a byte, and appending twice
  // will end exactly on the end of a byte
  int8_t test_values[68];
  memset(test_values, 0, sizeof(test_values));
  // Make it easy to check the answer without repeating sequential packed byte values
  for (int i = 0; i < 68; i++) {
    test_values[i] = (i % 5) == 0;
  }

  // Append starting at 0
  ASSERT_EQ(ArrowBitmapReserve(&bitmap, 68), NANOARROW_OK);
  ArrowBitmapAppendInt8Unsafe(&bitmap, test_values, 68);

  EXPECT_EQ(bitmap.size_bits, 68);
  EXPECT_EQ(bitmap.buffer.size_bytes, 9);
  for (int i = 0; i < 68; i++) {
    EXPECT_EQ(ArrowBitmapGetBit(bitmap.buffer.data, i), test_values[i]);
  }

  // Append starting at a non-byte aligned value
  ASSERT_EQ(ArrowBitmapReserve(&bitmap, 68), NANOARROW_OK);
  ArrowBitmapAppendInt8Unsafe(&bitmap, test_values, 68);

  EXPECT_EQ(bitmap.size_bits, 68 * 2);
  EXPECT_EQ(bitmap.buffer.size_bytes, 17);
  for (int i = 0; i < 68; i++) {
    EXPECT_EQ(ArrowBitmapGetBit(bitmap.buffer.data, i), test_values[i]);
  }
  for (int i = 69; i < (68 * 2); i++) {
    EXPECT_EQ(ArrowBitmapGetBit(bitmap.buffer.data, i), test_values[i - 68]);
  }

  // Append starting at a byte aligned but non-zero value
  ASSERT_EQ(ArrowBitmapReserve(&bitmap, 68), NANOARROW_OK);
  ArrowBitmapAppendInt8Unsafe(&bitmap, test_values, 68);

  EXPECT_EQ(bitmap.size_bits, 204);
  EXPECT_EQ(bitmap.buffer.size_bytes, 26);
  for (int i = 0; i < 68; i++) {
    EXPECT_EQ(ArrowBitmapGetBit(bitmap.buffer.data, i), test_values[i]);
  }
  for (int i = 69; i < 136; i++) {
    EXPECT_EQ(ArrowBitmapGetBit(bitmap.buffer.data, i), test_values[i - 68]);
  }
  for (int i = 136; i < 204; i++) {
    EXPECT_EQ(ArrowBitmapGetBit(bitmap.buffer.data, i), test_values[i - 136]);
  }

  ArrowBitmapReset(&bitmap);
}

TEST(BitmapTest, BitmapTestBuilderAppendInt32Unsafe) {
  struct ArrowBitmap bitmap;
  ArrowBitmapInit(&bitmap);

  // 68 because this will end in the middle of a byte, and appending twice
  // will end exactly on the end of a byte
  int32_t test_values[68];
  memset(test_values, 0, sizeof(test_values));
  // Make it easy to check the answer without repeating sequential packed byte values
  for (int i = 0; i < 68; i++) {
    test_values[i] = (i % 5) == 0;
  }

  // Append starting at 0
  ASSERT_EQ(ArrowBitmapReserve(&bitmap, 68), NANOARROW_OK);
  ArrowBitmapAppendInt32Unsafe(&bitmap, test_values, 68);

  EXPECT_EQ(bitmap.size_bits, 68);
  EXPECT_EQ(bitmap.buffer.size_bytes, 9);
  for (int i = 0; i < 68; i++) {
    EXPECT_EQ(ArrowBitmapGetBit(bitmap.buffer.data, i), test_values[i]);
  }

  // Append starting at a non-byte aligned value
  ASSERT_EQ(ArrowBitmapReserve(&bitmap, 68), NANOARROW_OK);
  ArrowBitmapAppendInt32Unsafe(&bitmap, test_values, 68);

  EXPECT_EQ(bitmap.size_bits, 68 * 2);
  EXPECT_EQ(bitmap.buffer.size_bytes, 17);
  for (int i = 0; i < 68; i++) {
    EXPECT_EQ(ArrowBitmapGetBit(bitmap.buffer.data, i), test_values[i]);
  }
  for (int i = 69; i < (68 * 2); i++) {
    EXPECT_EQ(ArrowBitmapGetBit(bitmap.buffer.data, i), test_values[i - 68]);
  }

  // Append starting at a byte aligned but non-zero value
  ASSERT_EQ(ArrowBitmapReserve(&bitmap, 68), NANOARROW_OK);
  ArrowBitmapAppendInt32Unsafe(&bitmap, test_values, 68);

  EXPECT_EQ(bitmap.size_bits, 204);
  EXPECT_EQ(bitmap.buffer.size_bytes, 26);
  for (int i = 0; i < 68; i++) {
    EXPECT_EQ(ArrowBitmapGetBit(bitmap.buffer.data, i), test_values[i]);
  }
  for (int i = 69; i < 136; i++) {
    EXPECT_EQ(ArrowBitmapGetBit(bitmap.buffer.data, i), test_values[i - 68]);
  }
  for (int i = 136; i < 204; i++) {
    EXPECT_EQ(ArrowBitmapGetBit(bitmap.buffer.data, i), test_values[i - 136]);
  }

  ArrowBitmapReset(&bitmap);
}
