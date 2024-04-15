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

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <nanoarrow/nanoarrow_testing.hpp>
#include "c_data_integration.h"

static int64_t kBytesAllocated = 0;

static uint8_t* IntegrationTestReallocate(ArrowBufferAllocator* allocator, uint8_t* ptr,
                                          int64_t old_size, int64_t new_size) {
  NANOARROW_UNUSED(allocator);
  ArrowBufferAllocator default_allocator = ArrowBufferAllocatorDefault();
  kBytesAllocated -= old_size;
  uint8_t* out =
      default_allocator.reallocate(&default_allocator, ptr, old_size, new_size);
  if (out != nullptr) {
    kBytesAllocated += new_size;
  }

  return out;
}

static void IntegrationTestFree(struct ArrowBufferAllocator* allocator, uint8_t* ptr,
                                int64_t size) {
  NANOARROW_UNUSED(allocator);
  ArrowBufferAllocator default_allocator = ArrowBufferAllocatorDefault();
  kBytesAllocated -= size;
  default_allocator.free(&default_allocator, ptr, size);
}

static ArrowBufferAllocator IntegrationTestAllocator() {
  ArrowBufferAllocator allocator;
  allocator.reallocate = &IntegrationTestReallocate;
  allocator.free = &IntegrationTestFree;
  allocator.private_data = nullptr;
  return allocator;
}

static void SetComparisonOptions(nanoarrow::testing::TestingJSONComparison* comparison) {
  comparison->set_compare_batch_flags(false);
  comparison->set_compare_float_precision(3);
  comparison->set_compare_metadata_order(false);
}

static ArrowErrorCode ReadFileString(std::ostream& out, const std::string& file_path) {
  std::ifstream infile(file_path, std::ios::in | std::ios::binary);
  char buf[8096];
  do {
    infile.read(buf, sizeof(buf));
    out << std::string(buf, infile.gcount());
  } while (infile.gcount() > 0);

  infile.close();
  return NANOARROW_OK;
}

static ArrowErrorCode ArrayStreamFromJsonFilePath(const std::string& json_path,
                                                  ArrowArrayStream* out, int num_batch,
                                                  ArrowError* error) {
  std::stringstream ss;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ReadFileString(ss, json_path), error);

  nanoarrow::testing::TestingJSONReader reader(IntegrationTestAllocator());
  NANOARROW_RETURN_NOT_OK(reader.ReadDataFile(ss.str(), out, num_batch, error));
  return NANOARROW_OK;
}

struct MaterializedArrayStream {
  nanoarrow::UniqueSchema schema;
  std::vector<nanoarrow::UniqueArray> arrays;
};

static ArrowErrorCode MaterializeJsonFilePath(const std::string& json_path,
                                              MaterializedArrayStream* out, int num_batch,
                                              ArrowError* error) {
  nanoarrow::UniqueArrayStream stream;
  NANOARROW_RETURN_NOT_OK(
      ArrayStreamFromJsonFilePath(json_path, stream.get(), num_batch, error));

  int result = stream->get_schema(stream.get(), out->schema.get());
  if (result != NANOARROW_OK) {
    const char* err = stream->get_last_error(stream.get());
    if (err != nullptr) {
      ArrowErrorSet(error, "%s", err);
    }
  }

  nanoarrow::UniqueArray tmp;
  do {
    tmp.reset();
    int result = stream->get_next(stream.get(), tmp.get());
    if (result != NANOARROW_OK) {
      const char* err = stream->get_last_error(stream.get());
      if (err != nullptr) {
        ArrowErrorSet(error, "%s", err);
      }

      return result;
    }

    if (tmp->release == nullptr) {
      break;
    }

    out->arrays.emplace_back(tmp.get());
  } while (true);

  return NANOARROW_OK;
}

static ArrowErrorCode ExportSchemaFromJson(const char* json_path, ArrowSchema* out,
                                           ArrowError* error) {
  MaterializedArrayStream data;
  NANOARROW_RETURN_NOT_OK(MaterializeJsonFilePath(
      json_path, &data, nanoarrow::testing::TestingJSONReader::kNumBatchOnlySchema,
      error));
  ArrowSchemaMove(data.schema.get(), out);
  return NANOARROW_OK;
}

static ArrowErrorCode ImportSchemaAndCompareToJson(const char* json_path,
                                                   ArrowSchema* schema,
                                                   ArrowError* error) {
  nanoarrow::UniqueSchema actual(schema);

  MaterializedArrayStream data;
  NANOARROW_RETURN_NOT_OK(MaterializeJsonFilePath(
      json_path, &data, nanoarrow::testing::TestingJSONReader::kNumBatchOnlySchema,
      error));

  nanoarrow::testing::TestingJSONComparison comparison;
  SetComparisonOptions(&comparison);

  NANOARROW_RETURN_NOT_OK(
      comparison.CompareSchema(actual.get(), data.schema.get(), error));
  if (comparison.num_differences() > 0) {
    std::stringstream ss;
    comparison.WriteDifferences(ss);
    ArrowErrorSet(error, "Found %d differences:\n%s",
                  static_cast<int>(comparison.num_differences()), ss.str().c_str());
    return EINVAL;
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ExportBatchFromJson(const char* json_path, int num_batch,
                                          ArrowArray* out, ArrowError* error) {
  MaterializedArrayStream data;
  NANOARROW_RETURN_NOT_OK(MaterializeJsonFilePath(json_path, &data, num_batch, error));

  ArrowArrayMove(data.arrays[0].get(), out);
  return NANOARROW_OK;
}

static ArrowErrorCode ImportBatchAndCompareToJson(const char* json_path, int num_batch,
                                                  ArrowArray* batch, ArrowError* error) {
  nanoarrow::UniqueArray actual(batch);

  MaterializedArrayStream data;
  NANOARROW_RETURN_NOT_OK(MaterializeJsonFilePath(json_path, &data, num_batch, error));

  nanoarrow::testing::TestingJSONComparison comparison;
  SetComparisonOptions(&comparison);

  NANOARROW_RETURN_NOT_OK(comparison.SetSchema(data.schema.get(), error));
  NANOARROW_RETURN_NOT_OK(
      comparison.CompareBatch(actual.get(), data.arrays[0].get(), error));
  if (comparison.num_differences() > 0) {
    std::stringstream ss;
    comparison.WriteDifferences(ss);
    ArrowErrorSet(error, "Found %d differences:\n%s",
                  static_cast<int>(comparison.num_differences()), ss.str().c_str());
    return EINVAL;
  }

  return NANOARROW_OK;
}

static ArrowError global_error;

static const char* ConvertError(ArrowErrorCode errno_code) {
  if (errno_code == NANOARROW_OK) {
    return nullptr;
  } else {
    return global_error.message;
  }
}

int64_t nanoarrow_BytesAllocated() { return kBytesAllocated; }

DLL_EXPORT const char* nanoarrow_CDataIntegration_ExportSchemaFromJson(
    const char* json_path, ArrowSchema* out) {
  ArrowErrorInit(&global_error);
  return ConvertError(ExportSchemaFromJson(json_path, out, &global_error));
}

DLL_EXPORT const char* nanoarrow_CDataIntegration_ImportSchemaAndCompareToJson(
    const char* json_path, ArrowSchema* schema) {
  ArrowErrorInit(&global_error);
  return ConvertError(ImportSchemaAndCompareToJson(json_path, schema, &global_error));
}

DLL_EXPORT const char* nanoarrow_CDataIntegration_ExportBatchFromJson(
    const char* json_path, int num_batch, ArrowArray* out) {
  ArrowErrorInit(&global_error);
  return ConvertError(ExportBatchFromJson(json_path, num_batch, out, &global_error));
}

DLL_EXPORT const char* nanoarrow_CDataIntegration_ImportBatchAndCompareToJson(
    const char* json_path, int num_batch, ArrowArray* batch) {
  ArrowErrorInit(&global_error);
  return ConvertError(
      ImportBatchAndCompareToJson(json_path, num_batch, batch, &global_error));
}
