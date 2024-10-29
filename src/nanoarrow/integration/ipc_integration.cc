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
#include <sstream>

#include <nanoarrow/nanoarrow_ipc.hpp>
#include <nanoarrow/nanoarrow_testing.hpp>

#define NANOARROW_IPC_FILE_PADDED_MAGIC "ARROW1\0"

std::string GetEnv(char const* name) {
  char const* val = std::getenv(name);
  return val ? val : "";
}

constexpr auto kUsage = R"(USAGE:
  # assert that f.arrow reads identical to f.json
  env COMMAND=VALIDATE    \
      ARROW_PATH=f.arrow  \
      JSON_PATH=f.json    \
      nanoarrow_ipc_integration

  # produce f.arrow from f.json
  env COMMAND=JSON_TO_ARROW  \
      ARROW_PATH=f.arrow     \
      JSON_PATH=f.json       \
      nanoarrow_ipc_integration

  # copy f.stream into f.arrow
  env COMMAND=STREAM_TO_FILE  \
      ARROW_PATH=f.arrow      \
      nanoarrow_ipc_integration < f.stream

  # copy f.arrow into f.stream
  env COMMAND=FILE_TO_STREAM  \
      ARROW_PATH=f.arrow      \
      nanoarrow_ipc_integration > f.stream

  # run all internal test cases
  nanoarrow_ipc_integration
)";

ArrowErrorCode Validate(struct ArrowError*);
ArrowErrorCode JsonToArrow(struct ArrowError*);
ArrowErrorCode StreamToFile(struct ArrowError*);
ArrowErrorCode FileToStream(struct ArrowError*);

int main(int argc, char** argv) try {
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
    std::cerr << kUsage;
    return 1;
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

struct File {
  File(FILE* file) : file_{file} {}
  File() = default;

  ~File() {
    if (file_ != nullptr) {
      fclose(file_);
    }
  }

  ArrowErrorCode open(std::string path, std::string mode, struct ArrowError* error) {
    file_ = fopen(path.c_str(), mode.c_str());
    if (file_ != nullptr) {
      return NANOARROW_OK;
    }
    ArrowErrorSet(error, "Opening file '%s' failed with errno=%d", path.c_str(), errno);
    return EINVAL;
  }

  std::string read() const {
    fseek(file_, 0, SEEK_END);
    std::string contents(ftell(file_), '\0');
    rewind(file_);

    size_t bytes_read = 0;
    while (bytes_read < contents.size()) {
      bytes_read += fread(&contents[bytes_read], 1, contents.size() - bytes_read, file_);
    }
    return contents;
  }

  operator FILE*() const { return file_; }

  FILE* file_ = nullptr;
};

struct MaterializedArrayStream {
  nanoarrow::UniqueSchema schema;
  std::vector<nanoarrow::UniqueArray> batches;

  ArrowErrorCode From(struct ArrowArrayStream* stream, struct ArrowError* error) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayStreamGetSchema(stream, schema.get(), error));

    while (true) {
      nanoarrow::UniqueArray batch;
      NANOARROW_RETURN_NOT_OK(ArrowArrayStreamGetNext(stream, batch.get(), error));
      if (batch->release == nullptr) {
        break;
      }
      batches.push_back(std::move(batch));
    }

    return NANOARROW_OK;
  }

  ArrowErrorCode FromJsonFile(std::string const& path, struct ArrowError* error) {
    File json_file;
    NANOARROW_RETURN_NOT_OK(json_file.open(path, "r", error));
    auto json = json_file.read();

    nanoarrow::testing::TestingJSONReader reader;
    nanoarrow::UniqueArrayStream array_stream;
    NANOARROW_RETURN_NOT_OK(
        reader.ReadDataFile(json, array_stream.get(), reader.kNumBatchReadAll, error));
    return From(array_stream.get(), error);
  }

  ArrowErrorCode FromIpcFile(std::string const& path, struct ArrowError* error) {
    // FIXME this API needs to be public; it's a bit smelly to pretend that we support
    // reading files when this bespoke program is the only one which can do it
    //
    // For now: just check the first 8 bytes of the file and read a stream (ignoring the
    // Footer).
    File ipc_file;
    NANOARROW_RETURN_NOT_OK(ipc_file.open(path, "rb", error));
    auto bytes = ipc_file.read();

    auto min_size = sizeof(NANOARROW_IPC_FILE_PADDED_MAGIC) + sizeof(int32_t) +
                    strlen(NANOARROW_IPC_FILE_PADDED_MAGIC);
    if (bytes.size() < min_size) {
      ArrowErrorSet(error, "Expected file of more than %lu bytes, got %ld", min_size,
                    bytes.size());
      return EINVAL;
    }

    if (memcmp(bytes.data(), NANOARROW_IPC_FILE_PADDED_MAGIC,
               sizeof(NANOARROW_IPC_FILE_PADDED_MAGIC)) != 0) {
      ArrowErrorSet(error, "File did not begin with 'ARROW1\\0\\0'");
      return EINVAL;
    }

    nanoarrow::ipc::UniqueDecoder decoder;
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowIpcDecoderInit(decoder.get()), error);
    NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderVerifyFooter(
        decoder.get(), {{bytes.data()}, static_cast<int64_t>(bytes.size())}, error));
    NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderDecodeFooter(
        decoder.get(), {{bytes.data()}, static_cast<int64_t>(bytes.size())}, error));

    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowSchemaDeepCopy(&decoder->footer->schema, schema.get()), error);
    NANOARROW_RETURN_NOT_OK(
        ArrowIpcDecoderSetSchema(decoder.get(), &decoder->footer->schema, error));
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowIpcDecoderSetEndianness(decoder.get(), decoder->endianness), error);

    nanoarrow::UniqueBuffer record_batch_blocks;
    ArrowBufferMove(&decoder->footer->record_batch_blocks, record_batch_blocks.get());

    for (int i = 0;
         i < record_batch_blocks->size_bytes / sizeof(struct ArrowIpcFileBlock); i++) {
      const auto& block =
          reinterpret_cast<struct ArrowIpcFileBlock*>(record_batch_blocks->data)[i];
      struct ArrowBufferView metadata_view = {
          {bytes.data() + block.offset},
          block.metadata_length,
      };
      NANOARROW_RETURN_NOT_OK(
          ArrowIpcDecoderDecodeHeader(decoder.get(), metadata_view, error));

      struct ArrowBufferView body_view = {
          {metadata_view.data.as_uint8 + metadata_view.size_bytes},
          block.body_length,
      };
      nanoarrow::UniqueArray batch;
      NANOARROW_RETURN_NOT_OK(
          ArrowIpcDecoderDecodeArray(decoder.get(), body_view, -1, batch.get(),
                                     NANOARROW_VALIDATION_LEVEL_FULL, error));
      batches.push_back(std::move(batch));
    }

    return NANOARROW_OK;
  }

  ArrowErrorCode Write(struct ArrowIpcOutputStream* output_stream, bool write_file,
                       struct ArrowError* error) {
    nanoarrow::ipc::UniqueWriter writer;
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowIpcWriterInit(writer.get(), output_stream),
                                       error);

    if (write_file) {
      NANOARROW_RETURN_NOT_OK(ArrowIpcWriterStartFile(writer.get(), error));
    }

    NANOARROW_RETURN_NOT_OK(ArrowIpcWriterWriteSchema(writer.get(), schema.get(), error));

    nanoarrow::UniqueArrayView array_view;
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), error));

    for (const auto& batch : batches) {
      NANOARROW_RETURN_NOT_OK(
          ArrowArrayViewSetArray(array_view.get(), batch.get(), error));
      NANOARROW_RETURN_NOT_OK(
          ArrowIpcWriterWriteArrayView(writer.get(), array_view.get(), error));
    }

    NANOARROW_RETURN_NOT_OK(ArrowIpcWriterWriteArrayView(writer.get(), nullptr, error));

    if (write_file) {
      NANOARROW_RETURN_NOT_OK(ArrowIpcWriterFinalizeFile(writer.get(), error));
    }

    return NANOARROW_OK;
  }

  ArrowErrorCode WriteIpcFile(std::string const& path, struct ArrowError* error) {
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

    return Write(output_stream.get(), /*write_file=*/true, error);
  }
};

ArrowErrorCode Validate(struct ArrowError* error) {
  auto json_path = GetEnv("JSON_PATH");
  MaterializedArrayStream json_table;
  NANOARROW_RETURN_NOT_OK(json_table.FromJsonFile(json_path, error));

  auto arrow_path = GetEnv("ARROW_PATH");
  MaterializedArrayStream arrow_table;
  NANOARROW_RETURN_NOT_OK(arrow_table.FromIpcFile(arrow_path, error));

  nanoarrow::testing::TestingJSONComparison comparison;
  comparison.set_compare_metadata_order(false);
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
  MaterializedArrayStream table;
  NANOARROW_RETURN_NOT_OK(table.FromJsonFile(GetEnv("JSON_PATH"), error));
  return table.WriteIpcFile(GetEnv("ARROW_PATH"), error);
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

  MaterializedArrayStream table;
  NANOARROW_RETURN_NOT_OK(table.From(array_stream.get(), error));
  return table.WriteIpcFile(GetEnv("ARROW_PATH"), error);
}

ArrowErrorCode FileToStream(struct ArrowError* error) {
  MaterializedArrayStream table;
  NANOARROW_RETURN_NOT_OK(table.FromIpcFile(GetEnv("ARROW_PATH"), error));

  // wrap stdout into ArrowIpcOutputStream
  nanoarrow::ipc::UniqueOutputStream output_stream;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowIpcOutputStreamInitFile(output_stream.get(), stdout,
                                   /*close_on_release=*/true),
      error);

  return table.Write(output_stream.get(), /*write_file=*/false, error);
}
