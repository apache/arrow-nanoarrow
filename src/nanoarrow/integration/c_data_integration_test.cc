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
  std::ofstream outfile(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!outfile.is_open()) {
    return EINVAL;
  }
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
  ASSERT_EQ(err, nullptr) << err;
  ASSERT_EQ(schema->release, nullptr);
}
