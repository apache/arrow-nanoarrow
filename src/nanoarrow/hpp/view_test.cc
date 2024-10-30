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

#include "nanoarrow/nanoarrow.hpp"
#include "nanoarrow/nanoarrow_gtest_util.hpp"

using testing::ElementsAre;

TEST(NanoarrowHppTest, NanoarrowHppViewArrayAsTest) {
  nanoarrow::UniqueBuffer is_valid, floats;
  nanoarrow::BufferInitSequence(is_valid.get(), std::vector<uint8_t>{0xFF});
  ArrowBitClear(is_valid->data, 2);
  ArrowBitClear(is_valid->data, 5);
  nanoarrow::BufferInitSequence(floats.get(),
                                std::vector<float>{8, 4, 2, 1, .5, .25, .125});

  const void* buffers[] = {is_valid->data, floats->data};
  struct ArrowArray array{};
  array.length = 7;
  array.null_count = 2;
  array.n_buffers = 2;
  array.buffers = buffers;

  int i = 0;
  float f = 8;
  for (auto slot : nanoarrow::ViewArrayAs<float>(&array)) {
    if (i == 2 || i == 5) {
      EXPECT_EQ(slot, nanoarrow::NA);
    } else {
      EXPECT_EQ(slot, f);
    }
    ++i;
    f /= 2;
  }
}

TEST(NanoarrowHppTest, NanoarrowHppViewArrayAsBytesTest) {
  using namespace nanoarrow::literals;

  nanoarrow::UniqueBuffer is_valid, offsets, data;
  nanoarrow::BufferInitSequence(is_valid.get(), std::vector<uint8_t>{0xFF});
  ArrowBitClear(is_valid->data, 2);
  ArrowBitClear(is_valid->data, 5);
  nanoarrow::BufferInitSequence(offsets.get(),
                                std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7});
  nanoarrow::BufferInitSequence(data.get(), std::string{"abcdefghi"});

  const void* buffers[] = {is_valid->data, offsets->data, data->data};
  struct ArrowArray array{};
  array.length = 7;
  array.null_count = 2;
  array.n_buffers = 2;
  array.buffers = buffers;

  int i = 0;
  ArrowStringView expected[] = {"a"_asv, "b"_asv, "c"_asv, "d"_asv,
                                "e"_asv, "f"_asv, "g"_asv};
  for (auto slot : nanoarrow::ViewArrayAsBytes<32>(&array)) {
    if (i == 2 || i == 5) {
      EXPECT_EQ(slot, nanoarrow::NA);
    } else {
      EXPECT_EQ(slot, expected[i]);
    }
    ++i;
  }
}

TEST(NanoarrowHppTest, NanoarrowHppViewArrayAsFixedSizeBytesTest) {
  using namespace nanoarrow::literals;

  nanoarrow::UniqueBuffer is_valid, data;
  nanoarrow::BufferInitSequence(is_valid.get(), std::vector<uint8_t>{0xFF});
  ArrowBitClear(is_valid->data, 2);
  ArrowBitClear(is_valid->data, 5);
  nanoarrow::BufferInitSequence(
      data.get(), std::string{"foo"} + "bar" + "foo" + "bar" + "foo" + "bar" + "foo");

  const void* buffers[] = {is_valid->data, data->data};
  struct ArrowArray array{};
  array.length = 7;
  array.null_count = 2;
  array.n_buffers = 2;
  array.buffers = buffers;

  int i = 0;
  for (auto slot : nanoarrow::ViewArrayAsFixedSizeBytes(&array, 3)) {
    if (i == 2 || i == 5) {
      EXPECT_EQ(slot, nanoarrow::NA);
    } else {
      EXPECT_EQ(slot, i % 2 == 0 ? "foo"_asv : "bar"_asv);
    }
    ++i;
  }
}

TEST(NanoarrowHppTest, NanoarrowHppViewArrayStreamTest) {
  static int32_t slot = 1;

  struct ArrowArrayStream stream{};
  stream.get_schema = [](struct ArrowArrayStream*, struct ArrowSchema* out) {
    return ArrowSchemaInitFromType(out, NANOARROW_TYPE_INT32);
  };
  stream.get_next = [](struct ArrowArrayStream*, struct ArrowArray* out) {
    if (slot >= 16) return ENOMEM;
    NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(out, NANOARROW_TYPE_INT32));
    NANOARROW_RETURN_NOT_OK(ArrowArrayStartAppending(out));
    NANOARROW_RETURN_NOT_OK(ArrowArrayAppendInt(out, slot *= 2));
    return ArrowArrayFinishBuildingDefault(out, nullptr);
  };
  stream.get_last_error = [](struct ArrowArrayStream*) { return "foo bar"; };
  stream.release = [](struct ArrowArrayStream*) {};

  nanoarrow::ViewArrayStream stream_view(&stream);
  for (ArrowArray& array : stream_view) {
    EXPECT_THAT(nanoarrow::ViewArrayAs<int32_t>(&array), ElementsAre(slot));
  }
  EXPECT_EQ(stream_view.count(), 4);
  EXPECT_EQ(stream_view.code(), ENOMEM);
  EXPECT_STREQ(stream_view.error()->message, "foo bar");
}
