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

#include <gtest/gtest.h>

#include "nanoarrow_ipc.h"

TEST(NanoarrowIpcReader, InputStreamLiteral) {
  uint8_t input_data[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  struct ArrowBuffer input;
  ArrowBufferInit(&input);
  ASSERT_EQ(ArrowBufferAppend(&input, input_data, sizeof(input_data)), NANOARROW_OK);

  struct ArrowIpcInputStream stream;
  uint8_t output_data[] = {0xff, 0xff, 0xff, 0xff, 0xff};
  int64_t size_read_bytes;

  ASSERT_EQ(ArrowIpcInputStreamInitLiteral(&stream, &input), NANOARROW_OK);
  EXPECT_EQ(input.data, nullptr);

  EXPECT_EQ(stream.read(&stream, output_data, 2, &size_read_bytes, nullptr), NANOARROW_OK);
  EXPECT_EQ(size_read_bytes, 2);
  uint8_t output_data1[] = {0x01, 0x02, 0xff, 0xff, 0xff};
  EXPECT_EQ(memcmp(output_data, output_data1, sizeof(output_data)), 0);

  EXPECT_EQ(stream.read(&stream, output_data + 2, 2, &size_read_bytes, nullptr), NANOARROW_OK);
  EXPECT_EQ(size_read_bytes, 2);
  uint8_t output_data2[] = {0x01, 0x02, 0x03, 0x04, 0xff};
  EXPECT_EQ(memcmp(output_data, output_data2, sizeof(output_data)), 0);

  EXPECT_EQ(stream.read(&stream, output_data + 4, 2, &size_read_bytes, nullptr), NANOARROW_OK);
  EXPECT_EQ(size_read_bytes, 1);
  uint8_t output_data3[] = {0x01, 0x02, 0x03, 0x04, 0x05};
  EXPECT_EQ(memcmp(output_data, output_data3, sizeof(output_data)), 0);

  EXPECT_EQ(stream.read(&stream, nullptr, 2, &size_read_bytes, nullptr), NANOARROW_OK);
  EXPECT_EQ(size_read_bytes, 0);

  EXPECT_EQ(stream.read(&stream, nullptr, 0, &size_read_bytes, nullptr), NANOARROW_OK);
  EXPECT_EQ(size_read_bytes, 0);

  stream.release(&stream);
}
