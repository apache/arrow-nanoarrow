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

#include <array>

#include <gtest/gtest.h>

#include "nanoarrow/nanoarrow.hpp"

struct TestWrappedObj {
  int64_t* num_frees;

  TestWrappedObj(int64_t* addr) { num_frees = addr; }

  TestWrappedObj(TestWrappedObj&& obj) {
    num_frees = obj.num_frees;
    obj.num_frees = nullptr;
  }

  ~TestWrappedObj() {
    if (num_frees != nullptr) {
      *num_frees = *num_frees + 1;
    }
  }
};

TEST(HppBuffer, BufferInitWrapped) {
  nanoarrow::UniqueBuffer buffer;
  int64_t num_frees = 0;

  TestWrappedObj obj(&num_frees);
  nanoarrow::BufferInitWrapped(buffer.get(), std::move(obj), nullptr, 0);
  EXPECT_EQ(obj.num_frees, nullptr);
  EXPECT_EQ(num_frees, 0);
  buffer.reset();
  EXPECT_EQ(num_frees, 1);

  // Ensure the destructor won't get called again when ArrowBufferReset is
  // called on the empty buffer.
  buffer.reset();
  EXPECT_EQ(num_frees, 1);
}

TEST(HppBuffer, BufferInitSequence) {
  nanoarrow::UniqueBuffer buffer;

  // Check templating magic with std::string
  nanoarrow::BufferInitSequence(buffer.get(), std::string("1234"));
  EXPECT_EQ(buffer->size_bytes, 4);
  EXPECT_EQ(buffer->capacity_bytes, 0);
  EXPECT_EQ(memcmp(buffer->data, "1234", 4), 0);

  // Check templating magic with std::vector
  buffer.reset();
  nanoarrow::BufferInitSequence(buffer.get(), std::vector<uint8_t>({1, 2, 3, 4}));
  EXPECT_EQ(buffer->size_bytes, 4);
  EXPECT_EQ(buffer->capacity_bytes, 0);
  EXPECT_EQ(buffer->data[0], 1);
  EXPECT_EQ(buffer->data[1], 2);
  EXPECT_EQ(buffer->data[2], 3);
  EXPECT_EQ(buffer->data[3], 4);

  // Check templating magic with std::array
  buffer.reset();
  nanoarrow::BufferInitSequence(buffer.get(), std::array<uint8_t, 4>({1, 2, 3, 4}));
  EXPECT_EQ(buffer->size_bytes, 4);
  EXPECT_EQ(buffer->capacity_bytes, 0);
  EXPECT_EQ(buffer->data[0], 1);
  EXPECT_EQ(buffer->data[1], 2);
  EXPECT_EQ(buffer->data[2], 3);
  EXPECT_EQ(buffer->data[3], 4);
}
