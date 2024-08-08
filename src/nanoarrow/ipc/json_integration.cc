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

#include <cstdlib>
#include <fstream>

#include <nanoarrow/nanoarrow_ipc.hpp>
#include <nanoarrow/nanoarrow_testing.hpp>

#include "flatcc/flatcc_builder.h"
#include "nanoarrow/ipc/flatcc_generated.h"

struct File {
  File() = default;

  ~File() {
    if (file_ != nullptr) {
      fclose(file_);
    }
  }

  ArrowErrorCode open(std::string path, std::string mode,
                      struct ArrowError* error) {
    file_ = fopen(path.c_str(), mode.c_str());
    if (file_ != nullptr) {
      return NANOARROW_OK;
    }
    ArrowErrorSet(error, "Opening file '%s' failed with errno=%d", path.c_str(), errno);
    return EINVAL;
  }

  operator FILE*() const { return file_; }

  FILE* file_;
};

struct Table {
  nanoarrow::UniqueSchema schema;
  std::vector<nanoarrow::UniqueArray> batches;
};

std::string GetEnv(char const* name) {
  if (char const* val = std::getenv(name)) {
    return val;
  }
  return "";
}

constexpr char kPaddedMagic[8] = "ARROW1\0";

std::string ReadFileIntoString(std::string const& path) {
  std::ifstream stream{path};
  std::string contents(stream.seekg(0, std::ios_base::end).tellg(), '\0');
  stream.seekg(0).read(contents.data(), contents.size());
  return contents;
}

ArrowErrorCode ReadTableFromArrayStream(struct ArrowArrayStream* stream, Table* table,
                                        struct ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(ArrowArrayStreamGetSchema(stream, table->schema.get(), error));

  while (true) {
    nanoarrow::UniqueArray batch;
    NANOARROW_RETURN_NOT_OK(ArrowArrayStreamGetNext(stream, batch.get(), error));
    if (batch->release == nullptr) {
      break;
    }
    table->batches.push_back(std::move(batch));
  }

  return NANOARROW_OK;
}

ArrowErrorCode ReadTableFromJson(std::string const& json, Table* table,
                                 struct ArrowError* error) {
  nanoarrow::testing::TestingJSONReader reader;
  nanoarrow::UniqueArrayStream array_stream;
  NANOARROW_RETURN_NOT_OK(
      reader.ReadDataFile(json, array_stream.get(), reader.kNumBatchReadAll, error));
  return ReadTableFromArrayStream(array_stream.get(), table, error);
}

ArrowErrorCode ReadTableFromIpcFile(std::string const& path, Table* table,
                                    struct ArrowError* error) {
  // FIXME this API needs to be public; it's a bit smelly to pretend that we support
  // reading files when this bespoke program is the only one which can do it
  //
  // For now: just check the first 8 bytes of the file and read a stream (ignoring the
  // Footer).
  File ipc_file;
  NANOARROW_RETURN_NOT_OK(ipc_file.open(path, "rb", error));

  char prefix[sizeof(kPaddedMagic)] = {};
  if (fread(&prefix, 1, sizeof(prefix), ipc_file) < sizeof(prefix)) {
    ArrowErrorSet(error, "Expected file of more than %lu bytes, got %ld", sizeof(prefix),
                  ftell(ipc_file));
    return EINVAL;
  }

  if (memcmp(&prefix, kPaddedMagic, sizeof(prefix)) != 0) {
    ArrowErrorSet(error, "File did not begin with 'ARROW1\\0\\0'");
    return EINVAL;
  }

  nanoarrow::ipc::UniqueInputStream input_stream;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowIpcInputStreamInitFile(input_stream.get(), ipc_file,
                                  /*close_on_release=*/false),
      error);

  nanoarrow::UniqueArrayStream array_stream;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowIpcArrayStreamReaderInit(array_stream.get(), input_stream.get(),
                                    /*options=*/nullptr),
      error);

  return ReadTableFromArrayStream(array_stream.get(), table, error);
}

ArrowErrorCode WriteTableAsStream(Table const& table,
                                  struct ArrowIpcOutputStream* output_stream,
                                  struct ArrowError* error,
                                  struct ArrowBuffer* blocks = nullptr) {
  nanoarrow::ipc::UniqueWriter writer;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowIpcWriterInit(writer.get(), output_stream),
                                     error);

  NANOARROW_RETURN_NOT_OK(
      ArrowIpcWriterWriteSchema(writer.get(), table.schema.get(), error));

  nanoarrow::UniqueArrayView array_view;
  NANOARROW_RETURN_NOT_OK(
      ArrowArrayViewInitFromSchema(array_view.get(), table.schema.get(), error));

  for (const auto& batch : table.batches) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayViewSetArray(array_view.get(), batch.get(), error));
    NANOARROW_RETURN_NOT_OK(
        ArrowIpcWriterWriteArrayView(writer.get(), array_view.get(), error));
  }

  if (blocks != nullptr) {
    struct ArrowIpcWriterPrivate {
      struct ArrowIpcEncoder encoder;
      struct ArrowIpcOutputStream output_stream;
      struct ArrowBuffer buffer;
      struct ArrowBuffer body_buffer;

      int64_t offset;
      struct ArrowBuffer blocks;
    };
    ArrowBufferMove(&static_cast<ArrowIpcWriterPrivate*>(writer->private_data)->blocks,
                    blocks);
  }

  return ArrowIpcWriterWriteArrayView(writer.get(), nullptr, error);
}

ArrowErrorCode WriteTableToIpcFile(std::string const& path, Table const& table,
                                   struct ArrowError* error) {
  // FIXME this API needs to be public; it's a bit smelly to pretend that we support
  // writing files when this bespoke program is the only one which can do it
  //
  // For now: just write the leading magic, the stream + EOS, and a manual Footer.
  File ipc_file;
  NANOARROW_RETURN_NOT_OK(ipc_file.open(path, "wb", error));

  nanoarrow::ipc::UniqueOutputStream output_stream;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowIpcOutputStreamInitFile(output_stream.get(), ipc_file,
                                   /*close_on_release=*/false),
      error);

  struct ArrowBufferView magic = {{kPaddedMagic}, sizeof(kPaddedMagic)};
  NANOARROW_RETURN_NOT_OK(ArrowIpcOutputStreamWrite(output_stream.get(), magic, error));

  nanoarrow::UniqueBuffer blocks;
  NANOARROW_RETURN_NOT_OK(
      WriteTableAsStream(table, output_stream.get(), error, blocks.get()));

  nanoarrow::ipc::UniqueEncoder encoder;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowIpcEncoderInit(encoder.get()), error);

#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(org_apache_arrow_flatbuf, x)

#define FLATCC_RETURN_UNLESS_0_NO_NS(x, error)                        \
  if ((x) != 0) {                                                     \
    ArrowErrorSet(error, "%s:%d: %s failed", __FILE__, __LINE__, #x); \
    return ENOMEM;                                                    \
  }

#define FLATCC_RETURN_UNLESS_0(x, error) FLATCC_RETURN_UNLESS_0_NO_NS(ns(x), error)

#define FLATCC_RETURN_IF_NULL(x, error)                                 \
  if (!(x)) {                                                           \
    ArrowErrorSet(error, "%s:%d: %s was null", __FILE__, __LINE__, #x); \
    return ENOMEM;                                                      \
  }

  struct ArrowIpcEncoderPrivate {
    flatcc_builder_t builder;
    struct ArrowBuffer buffers;
    struct ArrowBuffer nodes;
    int encoding_footer;
  };

  auto* builder = &static_cast<ArrowIpcEncoderPrivate*>(encoder->private_data)->builder;

  FLATCC_RETURN_UNLESS_0(Footer_start_as_root(builder), error);

  FLATCC_RETURN_UNLESS_0(Footer_version_add(builder, ns(MetadataVersion_V5)), error);

  static_cast<ArrowIpcEncoderPrivate*>(encoder->private_data)->encoding_footer = 1;
  FLATCC_RETURN_UNLESS_0(Footer_schema_start(builder), error);
  NANOARROW_RETURN_NOT_OK(
      ArrowIpcEncoderEncodeSchema(encoder.get(), table.schema.get(), error));
  FLATCC_RETURN_UNLESS_0(Footer_schema_end(builder), error);

  auto* blocks_ptr = reinterpret_cast<struct ns(Block)*>(blocks->data);
  int64_t n = blocks->size_bytes / sizeof(struct ns(Block));
  for (int i = 0; i < n; i++) {
    // Offsets were written relative to the stream, so we need to adjust them to account
    // for the leading padded magic
    blocks_ptr[i].offset += sizeof(kPaddedMagic);
  }
  FLATCC_RETURN_UNLESS_0(Footer_recordBatches_create(builder, blocks_ptr, n), error);

  FLATCC_RETURN_IF_NULL(ns(Footer_end_as_root(builder)), error);

  nanoarrow::UniqueBuffer footer_buffer;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/false,
                                    footer_buffer.get()),
      error);

  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowIpcOutputStreamInitFile(output_stream.get(), ipc_file,
                                   /*close_on_release=*/false),
      error);

  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowBufferAppendInt32(footer_buffer.get(),
                             static_cast<int32_t>(footer_buffer->size_bytes)),
      error);
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowBufferAppend(footer_buffer.get(), kPaddedMagic, strlen(kPaddedMagic)), error);

  struct ArrowBufferView footer = {{footer_buffer->data}, footer_buffer->size_bytes};
  NANOARROW_RETURN_NOT_OK(ArrowIpcOutputStreamWrite(output_stream.get(), footer, error));

  return NANOARROW_OK;
}

ArrowErrorCode Validate(struct ArrowError* error) {
  auto json_path = GetEnv("JSON_PATH");
  Table json_table;
  NANOARROW_RETURN_NOT_OK(
      ReadTableFromJson(ReadFileIntoString(json_path), &json_table, error));

  auto arrow_path = GetEnv("ARROW_PATH");
  Table arrow_table;
  NANOARROW_RETURN_NOT_OK(ReadTableFromIpcFile(arrow_path, &arrow_table, error));

  nanoarrow::testing::TestingJSONComparison comparison;
  NANOARROW_RETURN_NOT_OK(
      comparison.CompareSchema(arrow_table.schema.get(), json_table.schema.get(), error));
  if (comparison.num_differences() != 0) {
    std::stringstream differences;
    comparison.WriteDifferences(differences);
    ArrowErrorSet(error, "Found %d differences between schemas:\n%s\n",
                  (int)comparison.num_differences(), differences.str().c_str());
    return EINVAL;
  }

  if (arrow_table.batches.size() != json_table.batches.size()) {
    ArrowErrorSet(error, "%s had %d batches but\n%s had %d batches\n",  //
                  json_path.c_str(), (int)json_table.batches.size(),    //
                  arrow_path.c_str(), (int)arrow_table.batches.size());
    return EINVAL;
  }

  NANOARROW_RETURN_NOT_OK(comparison.SetSchema(arrow_table.schema.get(), error));
  for (size_t i = 0; i < arrow_table.batches.size(); i++) {
    const auto& json_batch = json_table.batches[i];
    const auto& arrow_batch = arrow_table.batches[i];
    NANOARROW_RETURN_NOT_OK(comparison.CompareBatch(arrow_batch.get(), json_batch.get(),
                                                    error, "Batch " + std::to_string(i)));
  }
  if (comparison.num_differences() != 0) {
    std::stringstream differences;
    comparison.WriteDifferences(differences);
    ArrowErrorSet(error, "Found %d differences between batches:\n%s\n",
                  (int)comparison.num_differences(), differences.str().c_str());
    return EINVAL;
  }

  return NANOARROW_OK;
}

ArrowErrorCode JsonToArrow(struct ArrowError* error) {
  Table table;
  NANOARROW_RETURN_NOT_OK(
      ReadTableFromJson(ReadFileIntoString(GetEnv("JSON_PATH")), &table, error));
  return WriteTableToIpcFile(GetEnv("ARROW_PATH"), table, error);
}

ArrowErrorCode StreamToFile(struct ArrowError* error) {
  // wrap stdin into ArrowIpcInputStream
  nanoarrow::ipc::UniqueInputStream input_stream;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowIpcInputStreamInitFile(input_stream.get(), stdin, /*close_on_release=*/true),
      error);

  nanoarrow::UniqueArrayStream array_stream;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowIpcArrayStreamReaderInit(array_stream.get(), input_stream.get(),
                                    /*options=*/nullptr),
      error);

  Table table;
  NANOARROW_RETURN_NOT_OK(ReadTableFromArrayStream(array_stream.get(), &table, error));
  return WriteTableToIpcFile(GetEnv("ARROW_PATH"), table, error);
}

ArrowErrorCode FileToStream(struct ArrowError* error) {
  Table table;
  NANOARROW_RETURN_NOT_OK(ReadTableFromIpcFile(GetEnv("ARROW_PATH"), &table, error));

  // wrap stdout into ArrowIpcOutputStream
  nanoarrow::ipc::UniqueOutputStream output_stream;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowIpcOutputStreamInitFile(output_stream.get(), stdout,
                                   /*close_on_release=*/true),
      error);

  return WriteTableAsStream(table, output_stream.get(), error);
}

int main() try {
  std::string command = GetEnv("COMMAND");

  ArrowErrorCode error_code;
  struct ArrowError error;
  if (command == "VALIDATE") {
    std::cout << "Validating that " << GetEnv("ARROW_PATH") << " reads identical to "
              << GetEnv("JSON_PATH") << std::endl;

    error_code = Validate(&error);
  } else if (command == "JSON_TO_ARROW") {
    std::cout << "Producing " << GetEnv("ARROW_PATH") << " from " << GetEnv("JSON_PATH")
              << std::endl;

    error_code = JsonToArrow(&error);
  } else if (command == "STREAM_TO_FILE") {
    error_code = StreamToFile(&error);
  } else if (command == "FILE_TO_STREAM") {
    error_code = FileToStream(&error);
  } else {
    std::cerr << R"(USAGE:
  # assert that f.arrow reads identical to f.json
  env COMMAND=VALIDATE    \
      ARROW_PATH=f.arrow  \
      JSON_PATH=f.json    \
      nanoarrow_ipc_json_integration

  # produce f.arrow from f.json
  env COMMAND=JSON_TO_ARROW  \
      ARROW_PATH=f.arrow     \
      JSON_PATH=f.json       \
      nanoarrow_ipc_json_integration

  # copy f.stream into f.arrow
  env COMMAND=STREAM_TO_FILE  \
      ARROW_PATH=f.arrow      \
      nanoarrow_ipc_json_integration < f.stream

  # copy f.arrow into f.stream
  env COMMAND=FILE_TO_STREAM  \
      ARROW_PATH=f.arrow      \
      nanoarrow_ipc_json_integration > f.stream
)";
    return EINVAL;
  }

  if (error_code != NANOARROW_OK) {
    std::cerr << "Command " << command << " failed (" << error_code << "="
              << strerror(error_code) << "): " << error.message << std::endl;
  }
  return error_code;
} catch (std::exception const& e) {
  std::cerr << "Uncaught exception: " << e.what() << std::endl;
  return 1;
}
