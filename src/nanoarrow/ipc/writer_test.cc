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

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW)
#include <arrow/array.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>
#endif

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

  int64_t bytes_written = p2->bytes_written;
  EXPECT_EQ(ArrowIpcWriterWriteDictionaryBatch(writer2.get(), /*dictionary_id=*/0,
                                               /*is_delta=*/0, values_view.get(), &error),
            ENOTSUP);
  EXPECT_STREQ(error.message,
               "IPC file writing supports exactly one non-delta dictionary batch");
  EXPECT_EQ(p2->bytes_written, bytes_written);
  EXPECT_EQ(p2->footer.dictionary_blocks.size_bytes, sizeof(struct ArrowIpcFileBlock));

  nanoarrow::ipc::UniqueOutputStream stream3;
  nanoarrow::UniqueBuffer output3;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(stream3.get(), output3.get()), NANOARROW_OK);

  nanoarrow::ipc::UniqueWriter writer3;
  ASSERT_EQ(ArrowIpcWriterInit(writer3.get(), stream3.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterStartFile(writer3.get(), &error), NANOARROW_OK)
      << error.message;
  EXPECT_EQ(ArrowIpcWriterWriteDictionaryBatch(writer3.get(), /*dictionary_id=*/0,
                                               /*is_delta=*/1, values_view.get(), &error),
            ENOTSUP);
  EXPECT_STREQ(error.message,
               "IPC file writing supports exactly one non-delta dictionary batch");
}

// Build a struct array with a single dictionary-encoded (int32 -> utf8) child.
static void MakeDictionaryStructArray(struct ArrowArray* array,
                                      struct ArrowSchema* schema,
                                      const char* value0 = "foo",
                                      const char* value1 = "bar") {
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
  ASSERT_EQ(ArrowArrayAppendString(values, ArrowCharView(value0)), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(values, ArrowCharView(value1)), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayAppendInt(indices, 0), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(indices, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(indices, 0), NANOARROW_OK);
  array->length = 3;

  ASSERT_EQ(ArrowArrayFinishBuildingDefault(array, nullptr), NANOARROW_OK);
}

static std::vector<int32_t> DecodeMessageTypes(const struct ArrowBuffer* buffer) {
  std::vector<int32_t> message_types;
  struct ArrowBufferView remaining;
  remaining.data.as_uint8 = buffer->data;
  remaining.size_bytes = buffer->size_bytes;
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;
  ArrowIpcDecoderInit(&decoder);

  while (remaining.size_bytes > 0) {
    int result = ArrowIpcDecoderVerifyHeader(&decoder, remaining, &error);
    if (result == ENODATA) {
      break;
    }

    EXPECT_EQ(result, NANOARROW_OK) << error.message;
    if (result != NANOARROW_OK) {
      break;
    }

    message_types.push_back(decoder.message_type);
    int64_t message_size = ((decoder.header_size_bytes + 7) / 8) * 8 +
                           ((decoder.body_size_bytes + 7) / 8) * 8;
    EXPECT_LE(message_size, remaining.size_bytes);
    if (message_size > remaining.size_bytes) {
      break;
    }

    remaining.data.as_uint8 += message_size;
    remaining.size_bytes -= message_size;
  }

  ArrowIpcDecoderReset(&decoder);
  return message_types;
}

static std::vector<int64_t> DecodeDictionaryIds(const struct ArrowBuffer* buffer) {
  std::vector<int64_t> dictionary_ids;
  struct ArrowBufferView remaining;
  remaining.data.as_uint8 = buffer->data;
  remaining.size_bytes = buffer->size_bytes;
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;
  ArrowIpcDecoderInit(&decoder);

  while (remaining.size_bytes > 0) {
    int result = ArrowIpcDecoderVerifyHeader(&decoder, remaining, &error);
    if (result == ENODATA) {
      break;
    }

    EXPECT_EQ(result, NANOARROW_OK) << error.message;
    if (result != NANOARROW_OK) {
      break;
    }

    if (decoder.message_type == NANOARROW_IPC_MESSAGE_TYPE_DICTIONARY_BATCH) {
      result = ArrowIpcDecoderDecodeHeader(&decoder, remaining, &error);
      EXPECT_EQ(result, NANOARROW_OK) << error.message;
      if (result != NANOARROW_OK) {
        break;
      }
      dictionary_ids.push_back(decoder.dictionary->id);
    }

    int64_t message_size = ((decoder.header_size_bytes + 7) / 8) * 8 +
                           ((decoder.body_size_bytes + 7) / 8) * 8;
    remaining.data.as_uint8 += message_size;
    remaining.size_bytes -= message_size;
  }

  ArrowIpcDecoderReset(&decoder);
  return dictionary_ids;
}

static void MakeNestedDictionaryStructArray(struct ArrowArray* array,
                                            struct ArrowSchema* schema) {
  ASSERT_EQ(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(schema, 1), NANOARROW_OK);

  struct ArrowSchema* outer_field = schema->children[0];
  ASSERT_EQ(ArrowSchemaInitFromType(outer_field, NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(outer_field, "outer"), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateDictionary(outer_field), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInitFromType(outer_field->dictionary, NANOARROW_TYPE_STRUCT),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(outer_field->dictionary, 1), NANOARROW_OK);

  struct ArrowSchema* inner_field = outer_field->dictionary->children[0];
  ASSERT_EQ(ArrowSchemaInitFromType(inner_field, NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(inner_field, "inner"), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateDictionary(inner_field), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInitFromType(inner_field->dictionary, NANOARROW_TYPE_STRING),
            NANOARROW_OK);

  ASSERT_EQ(ArrowArrayInitFromSchema(array, schema, nullptr), NANOARROW_OK);
  struct ArrowArray* outer_indices = array->children[0];
  struct ArrowArray* outer_values = outer_indices->dictionary;
  struct ArrowArray* inner_indices = outer_values->children[0];
  struct ArrowArray* inner_values = inner_indices->dictionary;

  ASSERT_EQ(ArrowArrayStartAppending(array), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(inner_values, ArrowCharView("foo")), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(inner_values, ArrowCharView("bar")), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(inner_indices, 0), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(inner_indices, 1), NANOARROW_OK);
  outer_values->length = 2;
  ASSERT_EQ(ArrowArrayAppendInt(outer_indices, 0), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(outer_indices, 1), NANOARROW_OK);
  array->length = 2;
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(array, nullptr), NANOARROW_OK);
}

TEST(NanoarrowIpcWriter, WritesNestedDictionariesDependencyFirst) {
  struct ArrowError error;
  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array;
  MakeNestedDictionaryStructArray(array.get(), schema.get());

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

  EXPECT_EQ(DecodeDictionaryIds(output.get()), (std::vector<int64_t>{1, 0}));

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, output.get()), NANOARROW_OK);
  nanoarrow::UniqueArrayStream reader;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(reader.get(), &input, nullptr), NANOARROW_OK);
  nanoarrow::UniqueSchema roundtrip_schema;
  ASSERT_EQ(ArrowArrayStreamGetSchema(reader.get(), roundtrip_schema.get(), &error),
            NANOARROW_OK)
      << error.message;
  nanoarrow::UniqueArray roundtrip_array;
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), roundtrip_array.get(), &error),
            NANOARROW_OK)
      << error.message;

  struct ArrowArray* outer_dictionary = roundtrip_array->children[0]->dictionary;
  ASSERT_NE(outer_dictionary, nullptr);
  ASSERT_EQ(outer_dictionary->n_children, 1);
  struct ArrowArray* inner_dictionary = outer_dictionary->children[0]->dictionary;
  ASSERT_NE(inner_dictionary, nullptr);
  EXPECT_EQ(inner_dictionary->length, 2);
}

TEST(NanoarrowIpcWriter, DoesNotRepeatUnchangedDictionary) {
  struct ArrowError error;

  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array1;
  MakeDictionaryStructArray(array1.get(), schema.get());

  nanoarrow::UniqueSchema unused_schema;
  nanoarrow::UniqueArray array2;
  MakeDictionaryStructArray(array2.get(), unused_schema.get());

  nanoarrow::UniqueArrayStream array_stream;
  ASSERT_EQ(ArrowBasicArrayStreamInit(array_stream.get(), schema.get(), 2), NANOARROW_OK);
  ArrowBasicArrayStreamSetArray(array_stream.get(), 0, array1.get());
  ArrowBasicArrayStreamSetArray(array_stream.get(), 1, array2.get());

  nanoarrow::UniqueBuffer output;
  nanoarrow::ipc::UniqueOutputStream out_stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(out_stream.get(), output.get()), NANOARROW_OK);

  nanoarrow::ipc::UniqueWriter writer;
  ASSERT_EQ(ArrowIpcWriterInit(writer.get(), out_stream.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteArrayStream(writer.get(), array_stream.get(), &error),
            NANOARROW_OK)
      << error.message;

  std::vector<int32_t> message_types = DecodeMessageTypes(output.get());
  EXPECT_EQ(message_types,
            (std::vector<int32_t>{NANOARROW_IPC_MESSAGE_TYPE_SCHEMA,
                                  NANOARROW_IPC_MESSAGE_TYPE_DICTIONARY_BATCH,
                                  NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH,
                                  NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH}));
}

TEST(NanoarrowIpcWriter, EmitsChangedDictionary) {
  struct ArrowError error;

  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array1;
  MakeDictionaryStructArray(array1.get(), schema.get());

  nanoarrow::UniqueSchema unused_schema;
  nanoarrow::UniqueArray array2;
  MakeDictionaryStructArray(array2.get(), unused_schema.get(), "foo", "baz");

  nanoarrow::UniqueArrayStream array_stream;
  ASSERT_EQ(ArrowBasicArrayStreamInit(array_stream.get(), schema.get(), 2), NANOARROW_OK);
  ArrowBasicArrayStreamSetArray(array_stream.get(), 0, array1.get());
  ArrowBasicArrayStreamSetArray(array_stream.get(), 1, array2.get());

  nanoarrow::UniqueBuffer output;
  nanoarrow::ipc::UniqueOutputStream out_stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(out_stream.get(), output.get()), NANOARROW_OK);

  nanoarrow::ipc::UniqueWriter writer;
  ASSERT_EQ(ArrowIpcWriterInit(writer.get(), out_stream.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteArrayStream(writer.get(), array_stream.get(), &error),
            NANOARROW_OK)
      << error.message;

  std::vector<int32_t> message_types = DecodeMessageTypes(output.get());
  EXPECT_EQ(message_types,
            (std::vector<int32_t>{NANOARROW_IPC_MESSAGE_TYPE_SCHEMA,
                                  NANOARROW_IPC_MESSAGE_TYPE_DICTIONARY_BATCH,
                                  NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH,
                                  NANOARROW_IPC_MESSAGE_TYPE_DICTIONARY_BATCH,
                                  NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH}));

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW)
  auto arrow_input = std::make_shared<arrow::io::BufferReader>(
      arrow::Buffer::Wrap(output->data, output->size_bytes));
  auto maybe_arrow_reader = arrow::ipc::RecordBatchStreamReader::Open(arrow_input);
  ASSERT_TRUE(maybe_arrow_reader.ok()) << maybe_arrow_reader.status();
  auto arrow_reader = maybe_arrow_reader.ValueUnsafe();

  std::shared_ptr<arrow::RecordBatch> arrow_batch1;
  std::shared_ptr<arrow::RecordBatch> arrow_batch2;
  ASSERT_TRUE(arrow_reader->ReadNext(&arrow_batch1).ok());
  ASSERT_TRUE(arrow_reader->ReadNext(&arrow_batch2).ok());
  auto arrow_dictionary1 =
      std::static_pointer_cast<arrow::DictionaryArray>(arrow_batch1->column(0));
  auto arrow_dictionary2 =
      std::static_pointer_cast<arrow::DictionaryArray>(arrow_batch2->column(0));
  auto arrow_values1 =
      std::static_pointer_cast<arrow::StringArray>(arrow_dictionary1->dictionary());
  auto arrow_values2 =
      std::static_pointer_cast<arrow::StringArray>(arrow_dictionary2->dictionary());
  EXPECT_EQ(arrow_values1->GetString(1), "bar");
  EXPECT_EQ(arrow_values2->GetString(1), "baz");
#endif

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, output.get()), NANOARROW_OK);
  nanoarrow::UniqueArrayStream reader;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(reader.get(), &input, nullptr), NANOARROW_OK);

  nanoarrow::UniqueSchema roundtrip_schema;
  ASSERT_EQ(ArrowArrayStreamGetSchema(reader.get(), roundtrip_schema.get(), &error),
            NANOARROW_OK)
      << error.message;

  nanoarrow::UniqueArray roundtrip_array1;
  nanoarrow::UniqueArray roundtrip_array2;
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), roundtrip_array1.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), roundtrip_array2.get(), &error),
            NANOARROW_OK)
      << error.message;

  nanoarrow::UniqueArrayView roundtrip_view1;
  nanoarrow::UniqueArrayView roundtrip_view2;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(roundtrip_view1.get(), roundtrip_schema.get(),
                                        &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(roundtrip_view2.get(), roundtrip_schema.get(),
                                        &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayViewSetArray(roundtrip_view1.get(), roundtrip_array1.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayViewSetArray(roundtrip_view2.get(), roundtrip_array2.get(), &error),
            NANOARROW_OK)
      << error.message;

  struct ArrowStringView first_dictionary_value =
      ArrowArrayViewGetStringUnsafe(roundtrip_view1->children[0]->dictionary, 1);
  struct ArrowStringView replacement_dictionary_value =
      ArrowArrayViewGetStringUnsafe(roundtrip_view2->children[0]->dictionary, 1);
  EXPECT_EQ(std::string(first_dictionary_value.data,
                        first_dictionary_value.size_bytes),
            "bar");
  EXPECT_EQ(std::string(replacement_dictionary_value.data,
                        replacement_dictionary_value.size_bytes),
            "baz");

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

// A struct array with a single int32 column of repeating values (i.e., compressible)
static constexpr int64_t kCompressibleBatchLength = 1024;

static void InitCompressibleBatch(struct ArrowSchema* schema, struct ArrowArray* array) {
  ASSERT_EQ(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInitFromType(schema->children[0], NANOARROW_TYPE_INT32),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema->children[0], "col"), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayInitFromSchema(array, schema, nullptr), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(array), NANOARROW_OK);
  for (int64_t i = 0; i < kCompressibleBatchLength; i++) {
    ASSERT_EQ(ArrowArrayAppendInt(array->children[0], i % 8), NANOARROW_OK);
    ASSERT_EQ(ArrowArrayFinishElement(array), NANOARROW_OK);
  }
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(array, nullptr), NANOARROW_OK);
}

// Write schema + batch + EOS (optionally as an IPC file) using codec and
// compression_level
static void WriteCompressibleBatch(enum ArrowIpcCompressionType codec,
                                   int compression_level, bool as_file,
                                   struct ArrowBuffer* output) {
  struct ArrowError error;

  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array;
  ASSERT_NO_FATAL_FAILURE(InitCompressibleBatch(schema.get(), array.get()));
  nanoarrow::UniqueArrayView array_view;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayViewSetArray(array_view.get(), array.get(), &error), NANOARROW_OK)
      << error.message;

  nanoarrow::ipc::UniqueOutputStream stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(stream.get(), output), NANOARROW_OK);
  nanoarrow::ipc::UniqueWriter writer;
  ASSERT_EQ(ArrowIpcWriterInit(writer.get(), stream.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterSetCompression(writer.get(), codec, compression_level, &error),
            NANOARROW_OK)
      << error.message;

  if (as_file) {
    ASSERT_EQ(ArrowIpcWriterStartFile(writer.get(), &error), NANOARROW_OK)
        << error.message;
  }
  ASSERT_EQ(ArrowIpcWriterWriteSchema(writer.get(), schema.get(), &error), NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowIpcWriterWriteArrayView(writer.get(), array_view.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowIpcWriterWriteArrayView(writer.get(), nullptr, &error), NANOARROW_OK)
      << error.message;

  if (as_file) {
    // The block for the record batch records the (compressed) body length
    auto* p = static_cast<struct ArrowIpcWriterPrivate*>(writer->private_data);
    ASSERT_EQ(p->footer.record_batch_blocks.size_bytes, sizeof(struct ArrowIpcFileBlock));
    auto* block =
        reinterpret_cast<struct ArrowIpcFileBlock*>(p->footer.record_batch_blocks.data);
    EXPECT_EQ(block->body_length, p->body_buffer.size_bytes);
    ASSERT_EQ(ArrowIpcWriterFinalizeFile(writer.get(), &error), NANOARROW_OK)
        << error.message;
  }
}

// Read the stream starting at offset back with the array stream reader and check that
// the values match what InitCompressibleBatch() produced
static void CheckCompressibleBatch(const struct ArrowBuffer* output, int64_t offset) {
  struct ArrowError error;

  nanoarrow::UniqueBuffer input_buffer;
  ASSERT_EQ(ArrowBufferAppend(input_buffer.get(), output->data + offset,
                              output->size_bytes - offset),
            NANOARROW_OK);
  nanoarrow::ipc::UniqueInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(input.get(), input_buffer.get()), NANOARROW_OK);
  nanoarrow::UniqueArrayStream stream;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(stream.get(), input.get(), nullptr),
            NANOARROW_OK);

  nanoarrow::UniqueSchema schema;
  ASSERT_EQ(ArrowArrayStreamGetSchema(stream.get(), schema.get(), &error), NANOARROW_OK)
      << error.message;
  EXPECT_STREQ(schema->format, "+s");

  nanoarrow::UniqueArray array;
  ASSERT_EQ(ArrowArrayStreamGetNext(stream.get(), array.get(), &error), NANOARROW_OK)
      << error.message;
  ASSERT_EQ(array->length, kCompressibleBatchLength);

  nanoarrow::UniqueArrayView array_view;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayViewSetArray(array_view.get(), array.get(), &error), NANOARROW_OK)
      << error.message;
  for (int64_t i = 0; i < kCompressibleBatchLength; i++) {
    ASSERT_EQ(ArrowArrayViewGetIntUnsafe(array_view->children[0], i), i % 8);
  }

  nanoarrow::UniqueArray eos;
  ASSERT_EQ(ArrowArrayStreamGetNext(stream.get(), eos.get(), &error), NANOARROW_OK)
      << error.message;
  EXPECT_EQ(eos->release, nullptr);
}

// Check whether the schema message (of a stream) or the footer (of a file) declares
// the COMPRESSED_BODY feature
static void CheckDeclaresCompression(const struct ArrowBuffer* output, bool as_file,
                                     bool expected) {
  struct ArrowError error;
  nanoarrow::ipc::UniqueDecoder decoder;
  ASSERT_EQ(ArrowIpcDecoderInit(decoder.get()), NANOARROW_OK);
  struct ArrowBufferView view = {{output->data}, output->size_bytes};
  if (as_file) {
    ASSERT_EQ(ArrowIpcDecoderVerifyFooter(decoder.get(), view, &error), NANOARROW_OK)
        << error.message;
    ASSERT_EQ(ArrowIpcDecoderDecodeFooter(decoder.get(), view, &error), NANOARROW_OK)
        << error.message;
  } else {
    ASSERT_EQ(ArrowIpcDecoderVerifyHeader(decoder.get(), view, &error), NANOARROW_OK)
        << error.message;
    ASSERT_EQ(ArrowIpcDecoderDecodeHeader(decoder.get(), view, &error), NANOARROW_OK)
        << error.message;
    ASSERT_EQ(decoder->message_type, NANOARROW_IPC_MESSAGE_TYPE_SCHEMA);
  }
  EXPECT_EQ((decoder->feature_flags & NANOARROW_IPC_FEATURE_COMPRESSED_BODY) != 0,
            expected);
}

static void TestCompressedWriting(enum ArrowIpcCompressionType codec) {
  for (bool as_file : {false, true}) {
    SCOPED_TRACE(as_file ? "file" : "stream");

    nanoarrow::UniqueBuffer uncompressed, compressed, accelerated;
    ASSERT_NO_FATAL_FAILURE(WriteCompressibleBatch(
        NANOARROW_IPC_COMPRESSION_TYPE_NONE, NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT,
        as_file, uncompressed.get()));
    ASSERT_NO_FATAL_FAILURE(WriteCompressibleBatch(
        codec, NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT, as_file, compressed.get()));
    ASSERT_NO_FATAL_FAILURE(
        WriteCompressibleBatch(codec, -65536, as_file, accelerated.get()));
    EXPECT_LT(compressed->size_bytes, uncompressed->size_bytes);
    // A nondefault level must reach the codec through the writer and encoder.
    EXPECT_GT(accelerated->size_bytes, compressed->size_bytes);

    // The stream portion of a file follows the padded magic
    int64_t offset = as_file ? sizeof(NANOARROW_IPC_FILE_PADDED_MAGIC) : 0;
    ASSERT_NO_FATAL_FAILURE(CheckCompressibleBatch(compressed.get(), offset));
    ASSERT_NO_FATAL_FAILURE(CheckCompressibleBatch(accelerated.get(), offset));

    // The schema message (or the file footer) declares that bodies are compressed
    ASSERT_NO_FATAL_FAILURE(CheckDeclaresCompression(compressed.get(), as_file, true));
    ASSERT_NO_FATAL_FAILURE(CheckDeclaresCompression(uncompressed.get(), as_file, false));
  }
}

TEST(NanoarrowIpcWriter, FileRetainsCompressionDeclaration) {
  if (ArrowIpcGetLZ4CompressionFunction() == nullptr &&
      ArrowIpcGetZstdCompressionFunction() == nullptr) {
    GTEST_SKIP() << "nanoarrow_ipc not built with NANOARROW_IPC_WITH_LZ4 or "
                    "NANOARROW_IPC_WITH_ZSTD";
  }

  for (auto codec :
       {NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME, NANOARROW_IPC_COMPRESSION_TYPE_ZSTD}) {
    int min_level, max_level;
    if (ArrowIpcGetCompressionLevelRange(codec, &min_level, &max_level) == ENOTSUP) {
      continue;
    }
    SCOPED_TRACE(ArrowIpcCompressionTypeToString(codec));
    for (bool write_compressed_batch : {false, true}) {
      SCOPED_TRACE(write_compressed_batch ? "compressed batch" : "schema only");
      struct ArrowError error;
      nanoarrow::UniqueSchema schema;
      nanoarrow::UniqueArray array;
      ASSERT_NO_FATAL_FAILURE(InitCompressibleBatch(schema.get(), array.get()));
      nanoarrow::UniqueArrayView view;
      ASSERT_EQ(ArrowArrayViewInitFromSchema(view.get(), schema.get(), &error),
                NANOARROW_OK);
      ASSERT_EQ(ArrowArrayViewSetArray(view.get(), array.get(), &error), NANOARROW_OK);
      nanoarrow::UniqueBuffer output;
      nanoarrow::ipc::UniqueOutputStream stream;
      ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(stream.get(), output.get()), NANOARROW_OK);
      nanoarrow::ipc::UniqueWriter writer;
      ASSERT_EQ(ArrowIpcWriterInit(writer.get(), stream.get()), NANOARROW_OK);
      ASSERT_EQ(ArrowIpcWriterSetCompression(
                    writer.get(), codec, NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT, &error),
                NANOARROW_OK);
      ASSERT_EQ(ArrowIpcWriterStartFile(writer.get(), &error), NANOARROW_OK);
      ASSERT_EQ(ArrowIpcWriterWriteSchema(writer.get(), schema.get(), &error),
                NANOARROW_OK);
      if (write_compressed_batch) {
        ASSERT_EQ(ArrowIpcWriterWriteArrayView(writer.get(), view.get(), &error),
                  NANOARROW_OK);
      }
      ASSERT_EQ(
          ArrowIpcWriterSetCompression(writer.get(), NANOARROW_IPC_COMPRESSION_TYPE_NONE,
                                       NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT, &error),
          NANOARROW_OK);
      ASSERT_EQ(ArrowIpcWriterWriteArrayView(writer.get(), view.get(), &error),
                NANOARROW_OK);
      ASSERT_EQ(ArrowIpcWriterWriteArrayView(writer.get(), nullptr, &error),
                NANOARROW_OK);
      ASSERT_EQ(ArrowIpcWriterFinalizeFile(writer.get(), &error), NANOARROW_OK);
      ASSERT_NO_FATAL_FAILURE(CheckDeclaresCompression(output.get(), true, true));
    }
  }
}

TEST(NanoarrowIpcWriter, CompressedWritingLZ4) {
  if (ArrowIpcGetLZ4CompressionFunction() == nullptr) {
    GTEST_SKIP() << "nanoarrow_ipc not built with NANOARROW_IPC_WITH_LZ4";
  }
  TestCompressedWriting(NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME);
}

TEST(NanoarrowIpcWriter, CompressedWritingZstd) {
  if (ArrowIpcGetZstdCompressionFunction() == nullptr) {
    GTEST_SKIP() << "nanoarrow_ipc not built with NANOARROW_IPC_WITH_ZSTD";
  }
  TestCompressedWriting(NANOARROW_IPC_COMPRESSION_TYPE_ZSTD);
}

TEST(NanoarrowIpcWriter, SetCompressionErrors) {
  nanoarrow::UniqueBuffer output;
  nanoarrow::ipc::UniqueOutputStream stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(stream.get(), output.get()), NANOARROW_OK);
  nanoarrow::ipc::UniqueWriter writer;
  ASSERT_EQ(ArrowIpcWriterInit(writer.get(), stream.get()), NANOARROW_OK);

  struct ArrowError error;
  // 3 is not an enumerator but is within the enum's value range (unlike, e.g., 99,
  // which C++ can't represent in this enum); it exercises the EINVAL path
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  auto unknown_type = static_cast<enum ArrowIpcCompressionType>(3);
  EXPECT_EQ(ArrowIpcWriterSetCompression(writer.get(), unknown_type,
                                         NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT, &error),
            EINVAL);
  EXPECT_STREQ(error.message, "Unknown compression type with value 3");

#if defined(NANOARROW_IPC_WITH_LZ4)
  EXPECT_EQ(ArrowIpcWriterSetCompression(
                writer.get(), NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME, 1000000, &error),
            EINVAL);
  EXPECT_STREQ(
      error.message,
      "Compression level 1000000 is out of range for lz4 (expected -65536 to 12)");
#endif
}

// A dictionary-encoded column written with compression: the DictionaryBatch is
// compressed too and the reader decodes it
TEST(NanoarrowIpcWriter, CompressedDictionaryStream) {
  if (ArrowIpcGetLZ4CompressionFunction() == nullptr) {
    GTEST_SKIP() << "nanoarrow_ipc not built with NANOARROW_IPC_WITH_LZ4";
  }
  struct ArrowError error;

  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array;
  ASSERT_NO_FATAL_FAILURE(MakeDictionaryStructArray(array.get(), schema.get()));
  nanoarrow::UniqueArrayStream array_stream;
  ASSERT_EQ(ArrowBasicArrayStreamInit(array_stream.get(), schema.get(), 1), NANOARROW_OK);
  ArrowBasicArrayStreamSetArray(array_stream.get(), 0, array.get());

  nanoarrow::UniqueBuffer output;
  nanoarrow::ipc::UniqueOutputStream out_stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(out_stream.get(), output.get()), NANOARROW_OK);
  nanoarrow::ipc::UniqueWriter writer;
  ASSERT_EQ(ArrowIpcWriterInit(writer.get(), out_stream.get()), NANOARROW_OK);
  ASSERT_EQ(
      ArrowIpcWriterSetCompression(writer.get(), NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME,
                                   NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT, &error),
      NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowIpcWriterWriteArrayStream(writer.get(), array_stream.get(), &error),
            NANOARROW_OK)
      << error.message;

  nanoarrow::ipc::UniqueInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(input.get(), output.get()), NANOARROW_OK);
  nanoarrow::UniqueArrayStream reader;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(reader.get(), input.get(), nullptr),
            NANOARROW_OK);

  nanoarrow::UniqueSchema roundtrip_schema;
  ASSERT_EQ(ArrowArrayStreamGetSchema(reader.get(), roundtrip_schema.get(), &error),
            NANOARROW_OK)
      << error.message;
  nanoarrow::UniqueArray roundtrip_array;
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), roundtrip_array.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(roundtrip_array->length, 3);

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
}
