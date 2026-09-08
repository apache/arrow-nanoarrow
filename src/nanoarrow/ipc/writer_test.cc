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

TEST(NanoarrowIpcWriter, WriteDictionaryBatch) {
  struct ArrowError error;

  nanoarrow::UniqueBuffer output;
  nanoarrow::ipc::UniqueOutputStream stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(stream.get(), output.get()), NANOARROW_OK);

  nanoarrow::ipc::UniqueWriter writer;
  ASSERT_EQ(ArrowIpcWriterInit(writer.get(), stream.get()), NANOARROW_OK);

  auto* p = static_cast<struct ArrowIpcWriterPrivate*>(writer->private_data);

  // Build a simple Utf8 values array
  nanoarrow::UniqueSchema values_schema;
  ASSERT_EQ(ArrowSchemaInitFromType(values_schema.get(), NANOARROW_TYPE_STRING),
            NANOARROW_OK);

  nanoarrow::UniqueArray values_array;
  ASSERT_EQ(ArrowArrayInitFromSchema(values_array.get(), values_schema.get(), nullptr),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(values_array.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(values_array.get(), ArrowCharView("foo")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(values_array.get(), ArrowCharView("bar")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(values_array.get(), &error), NANOARROW_OK)
      << error.message;

  nanoarrow::UniqueArrayView values_view;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(values_view.get(), values_schema.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayViewSetArray(values_view.get(), values_array.get(), &error),
            NANOARROW_OK)
      << error.message;

  // stream mode: write a DictionaryBatch — bytes are emitted but no block is tracked
  EXPECT_EQ(p->bytes_written, 0);
  EXPECT_EQ(p->footer.dictionary_blocks.size_bytes, 0);

  EXPECT_EQ(ArrowIpcWriterWriteDictionaryBatch(writer.get(), /*dictionary_id=*/0,
                                               /*is_delta=*/0, values_view.get(), &error),
            NANOARROW_OK)
      << error.message;

  auto after_dict_stream = p->bytes_written;
  EXPECT_GT(after_dict_stream, 0);
  // no block tracked in stream mode
  EXPECT_EQ(p->footer.dictionary_blocks.size_bytes, 0);

  // file mode: the block is tracked in the footer
  nanoarrow::ipc::UniqueOutputStream stream2;
  nanoarrow::UniqueBuffer output2;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(stream2.get(), output2.get()), NANOARROW_OK);

  nanoarrow::ipc::UniqueWriter writer2;
  ASSERT_EQ(ArrowIpcWriterInit(writer2.get(), stream2.get()), NANOARROW_OK);

  auto* p2 = static_cast<struct ArrowIpcWriterPrivate*>(writer2->private_data);

  ASSERT_EQ(ArrowIpcWriterStartFile(writer2.get(), &error), NANOARROW_OK)
      << error.message;
  EXPECT_EQ(p2->footer.dictionary_blocks.size_bytes, 0);

  EXPECT_EQ(ArrowIpcWriterWriteDictionaryBatch(writer2.get(), /*dictionary_id=*/0,
                                               /*is_delta=*/0, values_view.get(), &error),
            NANOARROW_OK)
      << error.message;

  // one block tracked in file mode
  EXPECT_EQ(p2->footer.dictionary_blocks.size_bytes, sizeof(struct ArrowIpcFileBlock));
}

// Build a struct array with a single dictionary-encoded (int32 -> utf8) child.
static void MakeDictionaryStructArray(struct ArrowArray* array,
                                      struct ArrowSchema* schema) {
  ASSERT_EQ(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInitFromType(schema->children[0], NANOARROW_TYPE_INT32),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema->children[0], "dict_col"), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateDictionary(schema->children[0]), NANOARROW_OK);
  ASSERT_EQ(
      ArrowSchemaInitFromType(schema->children[0]->dictionary, NANOARROW_TYPE_STRING),
      NANOARROW_OK);

  ASSERT_EQ(ArrowArrayInitFromSchema(array, schema, nullptr), NANOARROW_OK);
  struct ArrowArray* indices = array->children[0];
  struct ArrowArray* values = indices->dictionary;

  ASSERT_EQ(ArrowArrayStartAppending(array), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(values, ArrowCharView("foo")), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(values, ArrowCharView("bar")), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayAppendInt(indices, 0), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(indices, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(indices, 0), NANOARROW_OK);
  array->length = 3;

  ASSERT_EQ(ArrowArrayFinishBuildingDefault(array, nullptr), NANOARROW_OK);
}

// Write a dictionary-encoded stream through the high-level WriteArrayStream path
// and read it back through the IPC reader, confirming the DictionaryBatch is
// emitted automatically and the decoded values match.
TEST(NanoarrowIpcWriter, RoundtripDictionaryStream) {
  struct ArrowError error;

  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array;
  MakeDictionaryStructArray(array.get(), schema.get());

  nanoarrow::UniqueArrayStream array_stream;
  ASSERT_EQ(ArrowBasicArrayStreamInit(array_stream.get(), schema.get(), 1), NANOARROW_OK);
  ArrowBasicArrayStreamSetArray(array_stream.get(), 0, array.get());

  nanoarrow::UniqueBuffer output;
  nanoarrow::ipc::UniqueOutputStream out_stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(out_stream.get(), output.get()), NANOARROW_OK);

  nanoarrow::ipc::UniqueWriter writer;
  ASSERT_EQ(ArrowIpcWriterInit(writer.get(), out_stream.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteArrayStream(writer.get(), array_stream.get(), &error),
            NANOARROW_OK)
      << error.message;

  // Read the encoded bytes back
  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, output.get()), NANOARROW_OK);

  nanoarrow::UniqueArrayStream reader;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(reader.get(), &input, nullptr), NANOARROW_OK);

  nanoarrow::UniqueSchema roundtrip_schema;
  ASSERT_EQ(ArrowArrayStreamGetSchema(reader.get(), roundtrip_schema.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(roundtrip_schema->n_children, 1);
  ASSERT_NE(roundtrip_schema->children[0]->dictionary, nullptr);
  EXPECT_STREQ(roundtrip_schema->children[0]->dictionary->format, "u");

  nanoarrow::UniqueArray roundtrip_array;
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), roundtrip_array.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(roundtrip_array->length, 3);
  ASSERT_EQ(roundtrip_array->n_children, 1);
  ASSERT_NE(roundtrip_array->children[0]->dictionary, nullptr);
  EXPECT_EQ(roundtrip_array->children[0]->dictionary->length, 2);

  // Validate the decoded indices resolve to the original values
  nanoarrow::UniqueArrayView view;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(view.get(), roundtrip_schema.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayViewSetArray(view.get(), roundtrip_array.get(), &error),
            NANOARROW_OK)
      << error.message;

  struct ArrowArrayView* indices_view = view->children[0];
  struct ArrowArrayView* values_view = indices_view->dictionary;
  ASSERT_NE(values_view, nullptr);
  EXPECT_EQ(ArrowArrayViewGetIntUnsafe(indices_view, 0), 0);
  EXPECT_EQ(ArrowArrayViewGetIntUnsafe(indices_view, 1), 1);
  EXPECT_EQ(ArrowArrayViewGetIntUnsafe(indices_view, 2), 0);

  struct ArrowStringView v0 = ArrowArrayViewGetStringUnsafe(values_view, 0);
  struct ArrowStringView v1 = ArrowArrayViewGetStringUnsafe(values_view, 1);
  EXPECT_EQ(std::string(v0.data, v0.size_bytes), "foo");
  EXPECT_EQ(std::string(v1.data, v1.size_bytes), "bar");

  roundtrip_array.reset();
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), roundtrip_array.get(), &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_EQ(roundtrip_array->release, nullptr);
}
