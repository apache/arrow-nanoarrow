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
  int8_t bitmap[10];

  memset(bitmap, 0xff, sizeof(bitmap));
  for (int i = 0; i < sizeof(bitmap) * 8; i++) {
    EXPECT_EQ(ArrowBitmapElement(bitmap, i), 1);
  }

  bitmap[2] = 0xfd;
  EXPECT_EQ(ArrowBitmapElement(bitmap, 16 + 0), 1);
  EXPECT_EQ(ArrowBitmapElement(bitmap, 16 + 1), 0);
  EXPECT_EQ(ArrowBitmapElement(bitmap, 16 + 2), 1);
  EXPECT_EQ(ArrowBitmapElement(bitmap, 16 + 3), 1);
  EXPECT_EQ(ArrowBitmapElement(bitmap, 16 + 4), 1);
  EXPECT_EQ(ArrowBitmapElement(bitmap, 16 + 5), 1);
  EXPECT_EQ(ArrowBitmapElement(bitmap, 16 + 6), 1);
  EXPECT_EQ(ArrowBitmapElement(bitmap, 16 + 7), 1);

  memset(bitmap, 0x00, sizeof(bitmap));
  for (int i = 0; i < sizeof(bitmap) * 8; i++) {
    EXPECT_EQ(ArrowBitmapElement(bitmap, i), 0);
  }

  bitmap[2] = 0x02;
  EXPECT_EQ(ArrowBitmapElement(bitmap, 16 + 0), 0);
  EXPECT_EQ(ArrowBitmapElement(bitmap, 16 + 1), 1);
  EXPECT_EQ(ArrowBitmapElement(bitmap, 16 + 2), 0);
  EXPECT_EQ(ArrowBitmapElement(bitmap, 16 + 3), 0);
  EXPECT_EQ(ArrowBitmapElement(bitmap, 16 + 4), 0);
  EXPECT_EQ(ArrowBitmapElement(bitmap, 16 + 5), 0);
  EXPECT_EQ(ArrowBitmapElement(bitmap, 16 + 6), 0);
  EXPECT_EQ(ArrowBitmapElement(bitmap, 16 + 7), 0);
}

TEST(BitmapTest, BitmapTestSetElement) {
  int8_t bitmap[10];

  memset(bitmap, 0xff, sizeof(bitmap));
  ArrowBitmapSetElement(bitmap, 16 + 1, 0);
  EXPECT_EQ(bitmap[2], (int8_t)0xfd);
  ArrowBitmapSetElement(bitmap, 16 + 1, 1);
  EXPECT_EQ(bitmap[2], (int8_t)0xff);

  memset(bitmap, 0x00, sizeof(bitmap));
  ArrowBitmapSetElement(bitmap, 16 + 1, 1);
  EXPECT_EQ(bitmap[2], 0x02);
  ArrowBitmapSetElement(bitmap, 16 + 1, 0);
  EXPECT_EQ(bitmap[2], 0x00);
}
