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

#include <cstdio>
#include <fstream>

#include <gtest/gtest.h>

#include <nanoarrow/nanoarrow.hpp>

#include "c_data_integration.h"

// Not a true tempfile (writes to working directory), but is possibly more
// portable than mkstemp()
class TempFile {
 public:
  const char* name() { return name_; }

  ~TempFile() {
    if (std::remove(name_) != 0) {
      std::cerr << "Failed to remove '" << name_ << "'\n";
    }
  }

 private:
  const char* name_ = "c_data_integration.tmp.json";
};

ArrowErrorCode WriteFileString(const std::string& content, const std::string& path) {
  std::ofstream outfile(path, std::ios::out | std::ios::binary);
  outfile.write(content.data(), content.size());
  outfile.close();
  return NANOARROW_OK;
}

TEST(NanoarrowIntegrationTest, NanoarrowIntegrationTestSchema) {
  TempFile temp;
  nanoarrow::UniqueSchema schema;

  // Check error on export
  ASSERT_EQ(WriteFileString("this is not valid JSON", temp.name()), NANOARROW_OK);
  const char* err =
      nanoarrow_CDataIntegration_ExportSchemaFromJson(temp.name(), schema.get());
  ASSERT_NE(err, nullptr);
  ASSERT_EQ(std::string(err).substr(0, 9), "Exception");

  // Check valid roundtrip
  ASSERT_EQ(WriteFileString(R"({"schema": {"fields": []}, "batches": []})", temp.name()),
            NANOARROW_OK);

  err = nanoarrow_CDataIntegration_ExportSchemaFromJson(temp.name(), schema.get());
  ASSERT_EQ(err, nullptr) << err;
  ASSERT_NE(schema->release, nullptr);

  err =
      nanoarrow_CDataIntegration_ImportSchemaAndCompareToJson(temp.name(), schema.get());
  ASSERT_EQ(err, nullptr) << err;
  ASSERT_EQ(schema->release, nullptr);

  // Check roundtrip with differences
  ASSERT_EQ(WriteFileString(R"({"schema": {"fields": []}, "batches": []})", temp.name()),
            NANOARROW_OK);

  err = nanoarrow_CDataIntegration_ExportSchemaFromJson(temp.name(), schema.get());
  ASSERT_EQ(err, nullptr) << err;
  ASSERT_NE(schema->release, nullptr);

  // Change underlying JSON so we get differences
  ASSERT_EQ(
      WriteFileString(
          R"({"schema": {"fields": [{"name": "col1", "nullable": true, "type": {"name": "null"}, "children": []}]}, "batches": []})",
          temp.name()),
      NANOARROW_OK);
  err =
      nanoarrow_CDataIntegration_ImportSchemaAndCompareToJson(temp.name(), schema.get());
  ASSERT_NE(err, nullptr);
  ASSERT_EQ(std::string(err).substr(0, 19), "Found 1 differences") << err;
  ASSERT_EQ(schema->release, nullptr);
}

TEST(NanoarrowIntegrationTest, NanoarrowIntegrationTestBatch) {
  TempFile temp;
  nanoarrow::UniqueArray array;
  int64_t bytes_allocated_start = nanoarrow_BytesAllocated();

  // Check error on export
  ASSERT_EQ(WriteFileString("this is not valid JSON", temp.name()), NANOARROW_OK);
  const char* err =
      nanoarrow_CDataIntegration_ExportBatchFromJson(temp.name(), 0, array.get());
  ASSERT_NE(err, nullptr);
  ASSERT_EQ(std::string(err).substr(0, 9), "Exception") << err;

  // Check error for invalid batch id
  ASSERT_EQ(
      WriteFileString(
          R"({"schema": {)"
          R"("fields": [{"name": "col1", "nullable": true, "type": {"name": "utf8"}, "children": []}]}, )"
          R"("batches": [{"count": 1, "columns": [{"name": "col1", "count": 1, "VALIDITY": [1], "OFFSET": [0, 3], "DATA": ["abc"]}]}]})",
          temp.name()),
      NANOARROW_OK);
  err = nanoarrow_CDataIntegration_ExportBatchFromJson(temp.name(), 1, array.get());
  ASSERT_EQ(array->release, nullptr);
  ASSERT_NE(err, nullptr);
  ASSERT_STREQ(err, "Expected num_batch between 0 and 0 but got 1") << err;

  err = nanoarrow_CDataIntegration_ExportBatchFromJson(temp.name(), 0, array.get());
  ASSERT_EQ(err, nullptr) << err;
  ASSERT_NE(array->release, nullptr);
  err =
      nanoarrow_CDataIntegration_ImportBatchAndCompareToJson(temp.name(), 1, array.get());
  ASSERT_EQ(array->release, nullptr);
  ASSERT_STREQ(err, "Expected num_batch between 0 and 0 but got 1") << err;

  // Check valid roundtrip
  err = nanoarrow_CDataIntegration_ExportBatchFromJson(temp.name(), 0, array.get());
  ASSERT_NE(array->release, nullptr);
  ASSERT_EQ(err, nullptr);
  ASSERT_GT(nanoarrow_BytesAllocated(), bytes_allocated_start);

  err =
      nanoarrow_CDataIntegration_ImportBatchAndCompareToJson(temp.name(), 0, array.get());
  ASSERT_EQ(array->release, nullptr);
  ASSERT_EQ(err, nullptr) << err;
  ASSERT_EQ(nanoarrow_BytesAllocated(), bytes_allocated_start);

  // Check roundtrip with differences
  err = nanoarrow_CDataIntegration_ExportBatchFromJson(temp.name(), 0, array.get());
  ASSERT_NE(array->release, nullptr);
  ASSERT_EQ(err, nullptr) << err;
  ASSERT_GT(nanoarrow_BytesAllocated(), bytes_allocated_start);

  ASSERT_EQ(
      WriteFileString(
          R"({"schema": {)"
          R"("fields": [{"name": "col1", "nullable": true, "type": {"name": "utf8"}, "children": []}]}, )"
          R"("batches": [{"count": 0, "columns": [{"name": "col1", "count": 0, "VALIDITY": [], "OFFSET": [0], "DATA": []}]}]})",
          temp.name()),
      NANOARROW_OK);
  err =
      nanoarrow_CDataIntegration_ImportBatchAndCompareToJson(temp.name(), 0, array.get());
  ASSERT_NE(err, nullptr);
  ASSERT_EQ(std::string(err).substr(0, 19), "Found 2 differences") << err;
  ASSERT_EQ(array->release, nullptr);

  ASSERT_EQ(nanoarrow_BytesAllocated(), bytes_allocated_start);
}
