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

TEST(SchemaTest, Metadata) {
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
