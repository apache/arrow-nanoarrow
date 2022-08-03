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

#include <arrow/c/bridge.h>
#include <arrow/testing/gtest_util.h>
#include <arrow/util/key_value_metadata.h>

#include "nanoarrow/nanoarrow.h"

using namespace arrow;

TEST(MetadataTest, Metadata) {
  // (test will only work on little endian)
  char simple_metadata[] = {'\1', '\0', '\0', '\0', '\3', '\0', '\0', '\0', 'k', 'e',
                            'y',  '\5', '\0', '\0', '\0', 'v',  'a',  'l',  'u', 'e'};

  EXPECT_EQ(ArrowMetadataSizeOf(nullptr), 0);
  EXPECT_EQ(ArrowMetadataSizeOf(simple_metadata), sizeof(simple_metadata));

  EXPECT_EQ(ArrowMetadataHasKey(simple_metadata, "key"), 1);
  EXPECT_EQ(ArrowMetadataHasKey(simple_metadata, "not_a_key"), 0);

  struct ArrowStringView value;
  EXPECT_EQ(ArrowMetadataGetValue(simple_metadata, "key", "default_val", &value),
            NANOARROW_OK);
  EXPECT_EQ(std::string(value.data, value.n_bytes), "value");
  EXPECT_EQ(ArrowMetadataGetValue(simple_metadata, "not_a_key", "default_val", &value),
            NANOARROW_OK);
  EXPECT_EQ(std::string(value.data, value.n_bytes), "default_val");
}

TEST(MetadataTest, MetadataBuild) {
  // (test will only work on little endian)
  char simple_metadata[] = {'\1', '\0', '\0', '\0', '\3', '\0', '\0', '\0', 'k', 'e',
                            'y',  '\5', '\0', '\0', '\0', 'v',  'a',  'l',  'u', 'e'};

  struct ArrowBuffer metadata_builder;
  ASSERT_EQ(ArrowMetadataBuilderInit(&metadata_builder, nullptr), NANOARROW_OK);
  EXPECT_EQ(metadata_builder.size_bytes, 0);
  EXPECT_EQ(metadata_builder.data, nullptr);

  ASSERT_EQ(ArrowMetadataBuilderAppend(&metadata_builder, "key", "value"), NANOARROW_OK);
  ASSERT_EQ(metadata_builder.size_bytes, ArrowMetadataSizeOf(simple_metadata));
  EXPECT_EQ(memcmp(metadata_builder.data, simple_metadata, metadata_builder.size_bytes),
            0);

  ASSERT_EQ(ArrowMetadataBuilderAppend(&metadata_builder, "key2", "value2"),
            NANOARROW_OK);
  EXPECT_EQ(metadata_builder.size_bytes, ArrowMetadataSizeOf(simple_metadata) +
                                             sizeof(int32_t) + 4 + sizeof(int32_t) + 6);

  struct ArrowStringView value;
  ASSERT_EQ(
      ArrowMetadataGetValue((const char*)metadata_builder.data, "key2", nullptr, &value),
      NANOARROW_OK);
  EXPECT_EQ(std::string(value.data, value.n_bytes), "value2");

  ASSERT_EQ(ArrowMetadataBuilderSet(&metadata_builder, "key", "value3"), NANOARROW_OK);
  ASSERT_EQ(
      ArrowMetadataGetValue((const char*)metadata_builder.data, "key", nullptr, &value),
      NANOARROW_OK);
  EXPECT_EQ(std::string(value.data, value.n_bytes), "value3");
  ASSERT_EQ(
      ArrowMetadataGetValue((const char*)metadata_builder.data, "key2", nullptr, &value),
      NANOARROW_OK);
  EXPECT_EQ(std::string(value.data, value.n_bytes), "value2");

  ASSERT_EQ(ArrowMetadataBuilderSet(&metadata_builder, "key", NULL), NANOARROW_OK);
  EXPECT_EQ(ArrowMetadataHasKey((const char*)metadata_builder.data, "key"), false);
  ASSERT_EQ(
      ArrowMetadataGetValue((const char*)metadata_builder.data, "key2", nullptr, &value),
      NANOARROW_OK);
  EXPECT_EQ(std::string(value.data, value.n_bytes), "value2");

  ArrowBufferReset(&metadata_builder);
}
