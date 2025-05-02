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

#include "nanoarrow/nanoarrow_ipc.hpp"

TEST(NanoarrowIpcHppTest, NanoarrowIpcHppTestUniqueSharedBuffer) {
  nanoarrow::ipc::UniqueSharedBuffer shared_buffer;

  nanoarrow::UniqueBuffer buffer;
  ASSERT_EQ(ArrowBufferAppend(buffer.get(), "1234", 4), NANOARROW_OK);

  EXPECT_EQ(shared_buffer->private_src.data, nullptr);
  ASSERT_EQ(ArrowIpcSharedBufferInit(shared_buffer.get(), buffer.get()), NANOARROW_OK);
  EXPECT_NE(shared_buffer->private_src.data, nullptr);

  nanoarrow::ipc::UniqueSharedBuffer shared_buffer2 = std::move(shared_buffer);
  EXPECT_NE(shared_buffer2->private_src.data, nullptr);
  EXPECT_EQ(shared_buffer->private_src.data,
            nullptr);  // NOLINT(clang-analyzer-cplusplus.Move)
}

TEST(NanoarrowIpcHppTest, NanoarrowIpcHppTestUniqueDecoder) {
  nanoarrow::ipc::UniqueDecoder decoder;

  EXPECT_EQ(decoder->private_data, nullptr);
  ASSERT_EQ(ArrowIpcDecoderInit(decoder.get()), NANOARROW_OK);
  EXPECT_NE(decoder->private_data, nullptr);

  nanoarrow::ipc::UniqueDecoder decoder2 = std::move(decoder);
  EXPECT_NE(decoder2->private_data, nullptr);
  EXPECT_EQ(decoder->private_data, nullptr);  // NOLINT(clang-analyzer-cplusplus.Move)
}

TEST(NanoarrowIpcHppTest, NanoarrowIpcHppTestUniqueEncoder) {
  nanoarrow::ipc::UniqueEncoder encoder;

  EXPECT_EQ(encoder->private_data, nullptr);
  ASSERT_EQ(ArrowIpcEncoderInit(encoder.get()), NANOARROW_OK);
  EXPECT_NE(encoder->private_data, nullptr);

  nanoarrow::ipc::UniqueEncoder encoder2 = std::move(encoder);
  EXPECT_NE(encoder2->private_data, nullptr);
  EXPECT_EQ(encoder->private_data, nullptr);  // NOLINT(clang-analyzer-cplusplus.Move)
}

TEST(NanoarrowIpcHppTest, NanoarrowIpcHppTestUniqueInputStream) {
  nanoarrow::ipc::UniqueInputStream input;
  nanoarrow::UniqueBuffer buf;
  ASSERT_EQ(ArrowBufferAppend(buf.get(), "abcdefg", 7), NANOARROW_OK);

  EXPECT_EQ(input->release, nullptr);
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(input.get(), buf.get()), NANOARROW_OK);
  EXPECT_NE(input->release, nullptr);

  nanoarrow::ipc::UniqueInputStream input2 = std::move(input);
  EXPECT_NE(input2->release, nullptr);
  EXPECT_EQ(input->release, nullptr);  // NOLINT(clang-analyzer-cplusplus.Move)
}

TEST(NanoarrowIpcHppTest, NanoarrowIpcHppTestUniqueOutputStream) {
  nanoarrow::ipc::UniqueOutputStream output;
  nanoarrow::UniqueBuffer buf;
  ASSERT_EQ(ArrowBufferAppend(buf.get(), "abcdefg", 7), NANOARROW_OK);

  EXPECT_EQ(output->release, nullptr);
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(output.get(), buf.get()), NANOARROW_OK);
  EXPECT_NE(output->release, nullptr);

  nanoarrow::ipc::UniqueOutputStream output2 = std::move(output);
  EXPECT_NE(output2->release, nullptr);
  EXPECT_EQ(output->release, nullptr);  // NOLINT(clang-analyzer-cplusplus.Move)
}
