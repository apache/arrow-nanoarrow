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

#include <fstream>

#include <nanoarrow/nanoarrow_testing.hpp>

extern "C" {

const char* nanoarrow_CDataIntegration_ExportSchemaFromJson(const char* json_path,
                                                            ArrowSchema* out);

const char* nanoarrow_CDataIntegration_ImportSchemaAndCompareToJson(const char* json_path,
                                                                    ArrowSchema* schema);

const char* nanoarrow_CDataIntegration_ExportBatchFromJson(const char* json_path,
                                                           int num_batch,
                                                           ArrowArray* out);

const char* nanoarrow_CDataIntegration_ImportBatchAndCompareToJson(const char* json_path,
                                                                   int num_batch,
                                                                   ArrowArray* batch);

int64_t nanoarrow_BytesAllocated();

}  // extern "C"

static ArrowErrorCode ReadFileString(std::ostream& out, const std::string& file_path,
                                     ArrowError* error) {
  std::ifstream infile(file_path, std::ios::in | std::ios::binary);
  char buf[8096];
  do {
    out << std::string(buf, infile.gcount());
  } while (infile.read(buf, sizeof(buf)));

  return NANOARROW_OK;
}

static ArrowErrorCode ExportSchemaFromJson(const char* json_path, ArrowSchema* out,
                                           ArrowError* error) {
  return ENOTSUP;
}

static ArrowErrorCode ImportSchemaAndCompareToJson(const char* json_path,
                                                   ArrowSchema* schema,
                                                   ArrowError* error) {
  return ENOTSUP;
}

static ArrowErrorCode ExportBatchFromJson(const char* json_path, int num_batch,
                                          ArrowArray* out, ArrowError* error) {
  return ENOTSUP;
}

static ArrowErrorCode ImportBatchAndCompareToJson(const char* json_path, int num_batch,
                                                  ArrowArray* batch, ArrowError* error) {
  return ENOTSUP;
}

static ArrowError global_error;

static const char* ConvertError(ArrowErrorCode errno_code) {
  if (errno_code == NANOARROW_OK) {
    return nullptr;
  } else {
    return global_error.message;
  }
}

// TODO
int64_t nanoarrow_BytesAllocated() { return 0; }

const char* nanoarrow_CDataIntegration_ExportSchemaFromJson(const char* json_path,
                                                            ArrowSchema* out) {
  ArrowErrorInit(&global_error);
  return ConvertError(ExportSchemaFromJson(json_path, out, &global_error));
}

const char* nanoarrow_CDataIntegration_ImportSchemaAndCompareToJson(const char* json_path,
                                                                    ArrowSchema* schema) {
  ArrowErrorInit(&global_error);
  return ConvertError(ImportSchemaAndCompareToJson(json_path, schema, &global_error));
}

const char* nanoarrow_CDataIntegration_ExportBatchFromJson(const char* json_path,
                                                           int num_batch,
                                                           ArrowArray* out) {
  ArrowErrorInit(&global_error);
  return ConvertError(ExportBatchFromJson(json_path, num_batch, out, &global_error));
}

const char* nanoarrow_CDataIntegration_ImportBatchAndCompareToJson(const char* json_path,
                                                                   int num_batch,
                                                                   ArrowArray* batch) {
  ArrowErrorInit(&global_error);
  return ConvertError(
      ImportBatchAndCompareToJson(json_path, num_batch, batch, &global_error));
}
