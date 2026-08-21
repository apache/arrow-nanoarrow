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

#include <string>
#include <utility>
#include <vector>

#include "flatcc/flatcc_builder.h"
#include "nanoarrow/nanoarrow.hpp"
#include "nanoarrow/nanoarrow_ipc.hpp"

// Copied from encoder.c so we can test the internal state
extern "C" {
struct ArrowIpcEncoderPrivate {
  flatcc_builder_t builder;
  struct ArrowBuffer buffers;
  struct ArrowBuffer nodes;
};
}

#define NANOARROW_IPC_FILE_PADDED_MAGIC "ARROW1\0"
static_assert(sizeof(NANOARROW_IPC_FILE_PADDED_MAGIC) == 8, "");

TEST(NanoarrowIpcTest, NanoarrowIpcEncoderConstruction) {
  nanoarrow::ipc::UniqueEncoder encoder;

  EXPECT_EQ(ArrowIpcEncoderInit(encoder.get()), NANOARROW_OK);

  auto* p = static_cast<struct ArrowIpcEncoderPrivate*>(encoder->private_data);
  ASSERT_NE(p, nullptr);
  for (auto* b : {&p->buffers, &p->nodes}) {
    // Buffers are empty but initialized with the default allocator
    EXPECT_EQ(b->size_bytes, 0);

    auto default_allocator = ArrowBufferAllocatorDefault();
    EXPECT_EQ(memcmp(&b->allocator, &default_allocator, sizeof(b->allocator)), 0);
  }

  // Empty buffer works
  nanoarrow::UniqueBuffer buffer;
  EXPECT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/false, buffer.get()),
      NANOARROW_OK);
  EXPECT_EQ(buffer->size_bytes, 0);
  EXPECT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/true, buffer.get()),
      NANOARROW_OK);
  EXPECT_EQ(buffer->size_bytes, 8);

  // Append a string (finalizing an empty buffer is an error for flatcc_builder_t)
  EXPECT_NE(flatcc_builder_create_string_str(&p->builder, "hello world"), 0);
  EXPECT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/false, buffer.get()),
      NANOARROW_OK);
  EXPECT_GT(buffer->size_bytes, sizeof("hello world"));

  EXPECT_NE(flatcc_builder_create_string_str(&p->builder, "hello world"), 0);
  EXPECT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/true, buffer.get()),
      NANOARROW_OK);
  EXPECT_GT(buffer->size_bytes, 8 + sizeof("hello world"));
  EXPECT_EQ(buffer->size_bytes % 8, 0);
}

TEST(NanoarrowIpcTest, NanoarrowIpcFooterEncoding) {
  nanoarrow::ipc::UniqueEncoder encoder;
  ASSERT_EQ(ArrowIpcEncoderInit(encoder.get()), NANOARROW_OK);

  nanoarrow::ipc::UniqueFooter footer;
  ASSERT_EQ(ArrowSchemaInitFromType(&footer->schema, NANOARROW_TYPE_STRUCT),
            NANOARROW_OK);

  nanoarrow::UniqueBuffer footer_buffer, raw_schema_buffer;
  struct ArrowError error;

  EXPECT_EQ(ArrowIpcEncoderEncodeFooter(encoder.get(), footer.get(), &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_EQ(ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/false,
                                          footer_buffer.get()),
            NANOARROW_OK);

  EXPECT_EQ(ArrowIpcEncoderEncodeSchema(encoder.get(), &footer->schema, &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_EQ(ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/false,
                                          raw_schema_buffer.get()),
            NANOARROW_OK);

  EXPECT_GT(footer_buffer->size_bytes, raw_schema_buffer->size_bytes);
}

using KeyValues = std::vector<std::pair<std::string, std::string>>;

// Unpack nanoarrow's metadata representation into something comparable
static KeyValues UnpackMetadata(const char* metadata) {
  struct ArrowMetadataReader reader;
  NANOARROW_THROW_NOT_OK(ArrowMetadataReaderInit(&reader, metadata));

  KeyValues out;
  while (reader.remaining_keys > 0) {
    struct ArrowStringView key, value;
    NANOARROW_THROW_NOT_OK(ArrowMetadataReaderRead(&reader, &key, &value));
    out.emplace_back(std::string(key.data, key.size_bytes),
                     std::string(value.data, value.size_bytes));
  }
  return out;
}

static nanoarrow::UniqueBuffer PackMetadata(const KeyValues& key_values) {
  nanoarrow::UniqueBuffer metadata;
  NANOARROW_THROW_NOT_OK(ArrowMetadataBuilderInit(metadata.get(), nullptr));
  for (const auto& kv : key_values) {
    NANOARROW_THROW_NOT_OK(ArrowMetadataBuilderAppend(metadata.get(),
                                                      ArrowCharView(kv.first.c_str()),
                                                      ArrowCharView(kv.second.c_str())));
  }
  return metadata;
}

static ArrowErrorCode CollectKeyValue(struct ArrowStringView key,
                                      struct ArrowStringView value, void* private_data,
                                      struct ArrowError* error) {
  NANOARROW_UNUSED(error);
  static_cast<KeyValues*>(private_data)
      ->emplace_back(std::string(key.data, key.size_bytes),
                     std::string(value.data, value.size_bytes));
  return NANOARROW_OK;
}

// Decodes the header of an encapsulated message and returns its Message.custom_metadata
static KeyValues DecodeMessageMetadata(struct ArrowBuffer* message,
                                       struct ArrowIpcDecoder* decoder) {
  struct ArrowBufferView view;
  view.data.data = message->data;
  view.size_bytes = message->size_bytes;

  struct ArrowError error;
  NANOARROW_THROW_NOT_OK(ArrowIpcDecoderVerifyHeader(decoder, view, &error));
  NANOARROW_THROW_NOT_OK(ArrowIpcDecoderDecodeHeader(decoder, view, &error));

  nanoarrow::UniqueBuffer metadata;
  NANOARROW_THROW_NOT_OK(
      ArrowIpcDecoderGetMessageMetadata(decoder, metadata.get(), &error));
  KeyValues out = UnpackMetadata(reinterpret_cast<const char*>(metadata->data));

  // The visitor should see exactly the same pairs, in the same order
  KeyValues visited;
  NANOARROW_THROW_NOT_OK(
      ArrowIpcDecoderVisitMessageMetadata(decoder, &CollectKeyValue, &visited, &error));
  EXPECT_EQ(visited, out);

  return out;
}

// A struct array view with no columns and no rows: the smallest valid RecordBatch
class SimpleRecordBatch {
 public:
  SimpleRecordBatch() {
    NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(schema_.get(), NANOARROW_TYPE_STRUCT));
    NANOARROW_THROW_NOT_OK(
        ArrowArrayInitFromSchema(array_.get(), schema_.get(), nullptr));
    NANOARROW_THROW_NOT_OK(
        ArrowArrayViewInitFromSchema(array_view_.get(), schema_.get(), nullptr));
    NANOARROW_THROW_NOT_OK(
        ArrowArrayViewSetArray(array_view_.get(), array_.get(), nullptr));
  }

  struct ArrowSchema* schema() { return schema_.get(); }
  const struct ArrowArrayView* array_view() { return array_view_.get(); }

 private:
  nanoarrow::UniqueSchema schema_;
  nanoarrow::UniqueArray array_;
  nanoarrow::UniqueArrayView array_view_;
};

TEST(NanoarrowIpcTest, NanoarrowIpcEncoderMessageMetadataRoundtrip) {
  nanoarrow::ipc::UniqueEncoder encoder;
  ASSERT_EQ(ArrowIpcEncoderInit(encoder.get()), NANOARROW_OK);
  nanoarrow::ipc::UniqueDecoder decoder;
  ASSERT_EQ(ArrowIpcDecoderInit(decoder.get()), NANOARROW_OK);

  SimpleRecordBatch batch;
  struct ArrowError error;
  ASSERT_EQ(ArrowIpcDecoderSetSchema(decoder.get(), batch.schema(), &error), NANOARROW_OK)
      << error.message;

  KeyValues key_values{{"message_type", "data"}, {"cache-control", "no-store"}};
  auto metadata = PackMetadata(key_values);
  ASSERT_EQ(ArrowIpcEncoderSetMessageMetadata(encoder.get(), metadata.get(), &error),
            NANOARROW_OK)
      << error.message;

  // The encoder took ownership of the metadata
  EXPECT_EQ(metadata->data, nullptr);
  EXPECT_EQ(metadata->size_bytes, 0);

  nanoarrow::UniqueBuffer message, body;
  ASSERT_EQ(ArrowIpcEncoderEncodeSimpleRecordBatch(encoder.get(), batch.array_view(),
                                                   body.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/true, message.get()),
      NANOARROW_OK);

  EXPECT_EQ(DecodeMessageMetadata(message.get(), decoder.get()), key_values);

  // Values can also be read in place, without copying
  struct ArrowStringView value = ArrowCharView(nullptr);
  ASSERT_EQ(ArrowIpcDecoderGetMessageMetadataValue(
                decoder.get(), ArrowCharView("cache-control"), &value, &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_EQ(std::string(value.data, value.size_bytes), "no-store");

  // A key that isn't present leaves value_out untouched
  value = ArrowCharView(nullptr);
  ASSERT_EQ(ArrowIpcDecoderGetMessageMetadataValue(
                decoder.get(), ArrowCharView("not-a-key"), &value, &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_EQ(value.data, nullptr);

  // A key which is a prefix of a present key is not a match
  value = ArrowCharView(nullptr);
  ASSERT_EQ(ArrowIpcDecoderGetMessageMetadataValue(decoder.get(), ArrowCharView("cache"),
                                                   &value, &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_EQ(value.data, nullptr);

  // The metadata applied to exactly one message: the next one has none
  message->size_bytes = 0;
  body->size_bytes = 0;
  ASSERT_EQ(ArrowIpcEncoderEncodeSimpleRecordBatch(encoder.get(), batch.array_view(),
                                                   body.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/true, message.get()),
      NANOARROW_OK);
  EXPECT_EQ(DecodeMessageMetadata(message.get(), decoder.get()), KeyValues{});
}

TEST(NanoarrowIpcTest, NanoarrowIpcEncoderSchemaMessageMetadata) {
  nanoarrow::ipc::UniqueEncoder encoder;
  ASSERT_EQ(ArrowIpcEncoderInit(encoder.get()), NANOARROW_OK);
  nanoarrow::ipc::UniqueDecoder decoder;
  ASSERT_EQ(ArrowIpcDecoderInit(decoder.get()), NANOARROW_OK);

  SimpleRecordBatch batch;
  struct ArrowError error;

  // Message metadata is distinct from the metadata of the Schema it contains
  auto schema_metadata = PackMetadata({{"schema_key", "schema_value"}});
  ASSERT_EQ(ArrowSchemaSetMetadata(batch.schema(),
                                   reinterpret_cast<const char*>(schema_metadata->data)),
            NANOARROW_OK);

  KeyValues message_key_values{{"message_key", "message_value"}};
  auto message_metadata = PackMetadata(message_key_values);
  ASSERT_EQ(
      ArrowIpcEncoderSetMessageMetadata(encoder.get(), message_metadata.get(), &error),
      NANOARROW_OK)
      << error.message;

  nanoarrow::UniqueBuffer message;
  ASSERT_EQ(ArrowIpcEncoderEncodeSchema(encoder.get(), batch.schema(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/true, message.get()),
      NANOARROW_OK);

  EXPECT_EQ(DecodeMessageMetadata(message.get(), decoder.get()), message_key_values);

  nanoarrow::UniqueSchema roundtripped;
  ASSERT_EQ(ArrowIpcDecoderDecodeSchema(decoder.get(), roundtripped.get(), &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_EQ(UnpackMetadata(roundtripped->metadata),
            (KeyValues{{"schema_key", "schema_value"}}));
}

TEST(NanoarrowIpcTest, NanoarrowIpcEncoderMessageMetadataEmpty) {
  nanoarrow::ipc::UniqueEncoder encoder;
  ASSERT_EQ(ArrowIpcEncoderInit(encoder.get()), NANOARROW_OK);
  nanoarrow::ipc::UniqueDecoder decoder;
  ASSERT_EQ(ArrowIpcDecoderInit(decoder.get()), NANOARROW_OK);

  SimpleRecordBatch batch;
  struct ArrowError error;
  ASSERT_EQ(ArrowIpcDecoderSetSchema(decoder.get(), batch.schema(), &error), NANOARROW_OK)
      << error.message;

  // Metadata with no keys, NULL metadata, and no call at all are all equivalent
  auto empty_metadata = PackMetadata({});
  auto keyless_metadata = PackMetadata({{"key", "value"}});
  ASSERT_EQ(ArrowMetadataBuilderRemove(keyless_metadata.get(), ArrowCharView("key")),
            NANOARROW_OK);

  nanoarrow::UniqueBuffer baseline, body;
  ASSERT_EQ(ArrowIpcEncoderEncodeSimpleRecordBatch(encoder.get(), batch.array_view(),
                                                   body.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/true, baseline.get()),
      NANOARROW_OK);

  for (struct ArrowBuffer* metadata :
       {empty_metadata.get(), keyless_metadata.get(), (struct ArrowBuffer*)nullptr}) {
    ASSERT_EQ(ArrowIpcEncoderSetMessageMetadata(encoder.get(), metadata, &error),
              NANOARROW_OK)
        << error.message;

    nanoarrow::UniqueBuffer message;
    body->size_bytes = 0;
    ASSERT_EQ(ArrowIpcEncoderEncodeSimpleRecordBatch(encoder.get(), batch.array_view(),
                                                     body.get(), &error),
              NANOARROW_OK)
        << error.message;
    ASSERT_EQ(
        ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/true, message.get()),
        NANOARROW_OK);

    // No custom_metadata field is written at all
    ASSERT_EQ(message->size_bytes, baseline->size_bytes);
    EXPECT_EQ(memcmp(message->data, baseline->data, message->size_bytes), 0);

    EXPECT_EQ(DecodeMessageMetadata(message.get(), decoder.get()), KeyValues{});
  }

  // Setting metadata and then clearing it encodes nothing
  auto metadata = PackMetadata({{"key", "value"}});
  ASSERT_EQ(ArrowIpcEncoderSetMessageMetadata(encoder.get(), metadata.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowIpcEncoderSetMessageMetadata(encoder.get(), nullptr, &error),
            NANOARROW_OK)
      << error.message;

  nanoarrow::UniqueBuffer message;
  body->size_bytes = 0;
  ASSERT_EQ(ArrowIpcEncoderEncodeSimpleRecordBatch(encoder.get(), batch.array_view(),
                                                   body.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/true, message.get()),
      NANOARROW_OK);
  ASSERT_EQ(message->size_bytes, baseline->size_bytes);
  EXPECT_EQ(memcmp(message->data, baseline->data, message->size_bytes), 0);

  // Reading in place from a message without metadata finds nothing
  struct ArrowStringView value = ArrowCharView(nullptr);
  ASSERT_EQ(ArrowIpcDecoderGetMessageMetadataValue(decoder.get(), ArrowCharView("key"),
                                                   &value, &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_EQ(value.data, nullptr);
}

TEST(NanoarrowIpcTest, NanoarrowIpcVisitMessageMetadataError) {
  nanoarrow::ipc::UniqueEncoder encoder;
  ASSERT_EQ(ArrowIpcEncoderInit(encoder.get()), NANOARROW_OK);
  nanoarrow::ipc::UniqueDecoder decoder;
  ASSERT_EQ(ArrowIpcDecoderInit(decoder.get()), NANOARROW_OK);

  SimpleRecordBatch batch;
  struct ArrowError error;
  ASSERT_EQ(ArrowIpcDecoderSetSchema(decoder.get(), batch.schema(), &error), NANOARROW_OK)
      << error.message;

  auto metadata = PackMetadata({{"key1", "value1"}, {"key2", "value2"}});
  ASSERT_EQ(ArrowIpcEncoderSetMessageMetadata(encoder.get(), metadata.get(), &error),
            NANOARROW_OK)
      << error.message;

  nanoarrow::UniqueBuffer message, body;
  ASSERT_EQ(ArrowIpcEncoderEncodeSimpleRecordBatch(encoder.get(), batch.array_view(),
                                                   body.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/true, message.get()),
      NANOARROW_OK);

  struct ArrowBufferView view;
  view.data.data = message->data;
  view.size_bytes = message->size_bytes;
  ASSERT_EQ(ArrowIpcDecoderDecodeHeader(decoder.get(), view, &error), NANOARROW_OK)
      << error.message;

  // A visitor which errors stops the visit and its error code is returned
  struct Visitor {
    static ArrowErrorCode Visit(struct ArrowStringView key, struct ArrowStringView value,
                                void* private_data, struct ArrowError* error) {
      NANOARROW_UNUSED(value);
      auto* visited = static_cast<KeyValues*>(private_data);
      ArrowErrorSet(error, "visitor stopped at %.*s", static_cast<int>(key.size_bytes),
                    key.data);
      NANOARROW_RETURN_NOT_OK(CollectKeyValue(key, value, private_data, nullptr));
      return visited->size() == 1 ? ENOTSUP : NANOARROW_OK;
    }
  };

  KeyValues visited;
  EXPECT_EQ(ArrowIpcDecoderVisitMessageMetadata(decoder.get(), &Visitor::Visit, &visited,
                                                &error),
            ENOTSUP);
  EXPECT_EQ(visited, (KeyValues{{"key1", "value1"}}));
  EXPECT_STREQ(error.message, "visitor stopped at key1");
}
