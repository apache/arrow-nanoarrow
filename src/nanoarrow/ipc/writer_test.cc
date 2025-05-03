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

#include <stdio.h>

#include "nanoarrow/nanoarrow_ipc.hpp"

TEST(NanoarrowIpcWriter, OutputStreamBuffer) {
  struct ArrowError error;

  // The output buffer starts with some header
  std::string header = "HELLO WORLD";
  nanoarrow::UniqueBuffer output;
  ASSERT_EQ(ArrowBufferAppend(output.get(), header.data(), header.size()), NANOARROW_OK);

  // Then the stream starts appending to it
  nanoarrow::ipc::UniqueOutputStream stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(stream.get(), output.get()), NANOARROW_OK);

  std::string message = "\n-_-_";
  for (int i = 0; i < 4; ++i) {
    int64_t actually_written;
    ASSERT_EQ(stream->write(stream.get(), message.data(), message.size(),
                            &actually_written, &error),
              NANOARROW_OK)
        << error.message;
    EXPECT_EQ(actually_written, message.size());
  }

  EXPECT_EQ(output->size_bytes, header.size() + 4 * message.size());

  std::vector<char> output_str(output->size_bytes, '\0');
  memcpy(output_str.data(), output->data, output->size_bytes);
  EXPECT_EQ(std::string(output_str.data(), output_str.size()),
            header + message + message + message + message);
}

// clang-tidy helpfully reminds us that file_ptr might not be released
// if an assertion fails
struct FileCloser {
  FileCloser(FILE* file) : file_(file) {}
  ~FileCloser() {
    if (file_) fclose(file_);
  }
  FILE* file_{};
};

TEST(NanoarrowIpcWriter, OutputStreamFile) {
  FILE* file_ptr = tmpfile();
  FileCloser closer{file_ptr};
  ASSERT_NE(file_ptr, nullptr);

  // Start by writing some header
  std::string header = "HELLO WORLD";
  ASSERT_EQ(fwrite(header.data(), 1, header.size(), file_ptr), header.size());

  // Then seek to test that we overwrite WORLD but not HELLO
  fseek(file_ptr, 6, SEEK_SET);

  nanoarrow::ipc::UniqueOutputStream stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitFile(stream.get(), file_ptr, /*close_on_release=*/1),
            NANOARROW_OK);
  closer.file_ = nullptr;

  struct ArrowError error;

  // Start appending using the stream
  std::string message = "\n-_-_";
  for (int i = 0; i < 4; ++i) {
    int64_t actually_written;
    ASSERT_EQ(stream->write(stream.get(), message.data(), message.size(),
                            &actually_written, &error),
              NANOARROW_OK)
        << error.message;
    EXPECT_EQ(actually_written, message.size());
  }

  // Read back the whole file
  fseek(file_ptr, 0, SEEK_END);
  std::vector<char> buffer(static_cast<size_t>(ftell(file_ptr)), '\0');
  rewind(file_ptr);
  ASSERT_EQ(fread(buffer.data(), 1, buffer.size(), file_ptr), buffer.size());

  EXPECT_EQ(buffer.size(), 6 + 4 * message.size());
  EXPECT_EQ(std::string(buffer.data(), buffer.size()),
            "HELLO " + message + message + message + message);
}

TEST(NanoarrowIpcWriter, OutputStreamFileError) {
  nanoarrow::ipc::UniqueOutputStream stream;
  errno = EINVAL;
  EXPECT_EQ(ArrowIpcOutputStreamInitFile(stream.get(), nullptr, /*close_on_release=*/1),
            EINVAL);

  auto phony_path = __FILE__ + std::string(".phony");
  FILE* file_ptr = fopen(phony_path.c_str(), "rb");
  FileCloser closer{file_ptr};
  ASSERT_EQ(file_ptr, nullptr);
  EXPECT_EQ(ArrowIpcOutputStreamInitFile(stream.get(), file_ptr, /*close_on_release=*/1),
            ENOENT);
  closer.file_ = nullptr;
}

struct ArrowIpcWriterPrivate {
  struct ArrowIpcEncoder encoder;
  struct ArrowIpcOutputStream output_stream;
  struct ArrowBuffer buffer;
  struct ArrowBuffer body_buffer;

  int writing_file;
  int64_t bytes_written;
  struct ArrowIpcFooter footer;
};

#define NANOARROW_IPC_FILE_PADDED_MAGIC "ARROW1\0"

TEST(NanoarrowIpcWriter, FileWriting) {
  struct ArrowError error;

  nanoarrow::UniqueBuffer output;
  nanoarrow::ipc::UniqueOutputStream stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(stream.get(), output.get()), NANOARROW_OK);

  nanoarrow::ipc::UniqueWriter writer;
  ASSERT_EQ(ArrowIpcWriterInit(writer.get(), stream.get()), NANOARROW_OK);

  // the writer starts out in stream mode
  auto* p = static_cast<struct ArrowIpcWriterPrivate*>(writer->private_data);
  EXPECT_FALSE(p->writing_file);
  EXPECT_EQ(p->bytes_written, 0);
  EXPECT_EQ(p->footer.schema.release, nullptr);
  EXPECT_EQ(p->footer.record_batch_blocks.size_bytes, 0);

  // now it switches to file mode
  EXPECT_EQ(ArrowIpcWriterStartFile(writer.get(), &error), NANOARROW_OK) << error.message;
  EXPECT_TRUE(p->writing_file);
  // and has written the leading magic
  EXPECT_EQ(p->bytes_written, sizeof(NANOARROW_IPC_FILE_PADDED_MAGIC));
  // but not a schema or any record batches
  EXPECT_EQ(p->footer.schema.release, nullptr);
  EXPECT_EQ(p->footer.record_batch_blocks.size_bytes, 0);

  // write a schema
  nanoarrow::UniqueSchema schema;
  ASSERT_EQ(ArrowSchemaInitFromType(schema.get(), NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  EXPECT_EQ(ArrowIpcWriterWriteSchema(writer.get(), schema.get(), &error), NANOARROW_OK)
      << error.message;
  // more has been written
  auto after_schema = p->bytes_written;
  EXPECT_GT(after_schema, sizeof(NANOARROW_IPC_FILE_PADDED_MAGIC));
  // the schema is cached in the writer's footer for later finalization
  EXPECT_NE(p->footer.schema.release, nullptr);
  // still no record batches
  EXPECT_EQ(p->footer.record_batch_blocks.size_bytes, 0);

  // write a batch
  nanoarrow::UniqueArray array;
  nanoarrow::UniqueArrayView array_view;
  ASSERT_EQ(ArrowArrayInitFromSchema(array.get(), schema.get(), &error), NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayViewSetArray(array_view.get(), array.get(), &error), NANOARROW_OK)
      << error.message;
  EXPECT_EQ(ArrowIpcWriterWriteArrayView(writer.get(), array_view.get(), &error),
            NANOARROW_OK)
      << error.message;
  // more has been written
  auto after_batch = p->bytes_written;
  EXPECT_GT(after_batch, after_schema);
  // one record batch's block is stored
  EXPECT_EQ(p->footer.record_batch_blocks.size_bytes, sizeof(struct ArrowIpcFileBlock));

  // end the stream
  EXPECT_EQ(ArrowIpcWriterWriteArrayView(writer.get(), nullptr, &error), NANOARROW_OK)
      << error.message;
  // more has been written
  auto after_eos = p->bytes_written;
  EXPECT_GT(after_eos, after_batch);
  // EOS isn't stored in the blocks
  EXPECT_EQ(p->footer.record_batch_blocks.size_bytes, sizeof(struct ArrowIpcFileBlock));

  // finalize the file
  EXPECT_EQ(ArrowIpcWriterFinalizeFile(writer.get(), &error), NANOARROW_OK)
      << error.message;
  // more has been written
  auto after_footer = p->bytes_written;
  EXPECT_GT(after_footer, after_eos);
}
