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

#include <cstring>
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

// A record batch whose columns exercise each path of the compressed body builder:
// - "compressible": int32s with a repeating pattern
// - "with_nulls": int32s with a validity buffer
// - "incompressible": pseudo-random bytes, which are stored uncompressed (prefix -1)
// Columns without nulls have a zero-length validity buffer, which is never compressed.
class CompressibleRecordBatch {
 public:
  static constexpr int64_t kLength = 4096;
  static constexpr int64_t kBytesPerValue = 16;

  CompressibleRecordBatch() {
    NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(schema_.get(), NANOARROW_TYPE_STRUCT));
    NANOARROW_THROW_NOT_OK(ArrowSchemaAllocateChildren(schema_.get(), 3));
    NANOARROW_THROW_NOT_OK(
        ArrowSchemaInitFromType(schema_->children[0], NANOARROW_TYPE_INT32));
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema_->children[0], "compressible"));
    NANOARROW_THROW_NOT_OK(
        ArrowSchemaInitFromType(schema_->children[1], NANOARROW_TYPE_INT32));
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema_->children[1], "with_nulls"));
    NANOARROW_THROW_NOT_OK(
        ArrowSchemaInitFromType(schema_->children[2], NANOARROW_TYPE_BINARY));
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema_->children[2], "incompressible"));

    NANOARROW_THROW_NOT_OK(
        ArrowArrayInitFromSchema(array_.get(), schema_.get(), nullptr));
    NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(array_.get()));

    uint32_t state = 2463534242u;
    uint8_t random_bytes[kBytesPerValue];
    for (int64_t i = 0; i < kLength; i++) {
      NANOARROW_THROW_NOT_OK(ArrowArrayAppendInt(array_->children[0], i % 4));

      if (i % 3 == 0) {
        NANOARROW_THROW_NOT_OK(ArrowArrayAppendNull(array_->children[1], 1));
      } else {
        NANOARROW_THROW_NOT_OK(ArrowArrayAppendInt(array_->children[1], i));
      }

      // xorshift32 so that the bytes are deterministic but not compressible
      for (int64_t j = 0; j < kBytesPerValue; j += 4) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        std::memcpy(random_bytes + j, &state, sizeof(state));
      }
      struct ArrowBufferView bytes = {{random_bytes}, kBytesPerValue};
      NANOARROW_THROW_NOT_OK(ArrowArrayAppendBytes(array_->children[2], bytes));

      NANOARROW_THROW_NOT_OK(ArrowArrayFinishElement(array_.get()));
    }

    NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuildingDefault(array_.get(), nullptr));
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

static int64_t ReadLittleEndianInt64(const uint8_t* data) {
  int64_t value;
  std::memcpy(&value, data, sizeof(value));
  if (ArrowIpcSystemEndianness() == NANOARROW_IPC_ENDIANNESS_BIG) {
    value = static_cast<int64_t>(bswap64(static_cast<uint64_t>(value)));
  }
  return value;
}

static void TestCompressedRecordBatchRoundtrip(enum ArrowIpcCompressionType codec) {
  nanoarrow::ipc::UniqueEncoder encoder;
  ASSERT_EQ(ArrowIpcEncoderInit(encoder.get()), NANOARROW_OK);
  nanoarrow::ipc::UniqueDecoder decoder;
  ASSERT_EQ(ArrowIpcDecoderInit(decoder.get()), NANOARROW_OK);

  CompressibleRecordBatch batch;
  struct ArrowError error;
  ASSERT_EQ(ArrowIpcDecoderSetSchema(decoder.get(), batch.schema(), &error), NANOARROW_OK)
      << error.message;

  // Encode without compression for reference
  nanoarrow::UniqueBuffer uncompressed_message, uncompressed_body;
  ASSERT_EQ(ArrowIpcEncoderEncodeSimpleRecordBatch(encoder.get(), batch.array_view(),
                                                   uncompressed_body.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/true,
                                          uncompressed_message.get()),
            NANOARROW_OK);

  ASSERT_EQ(ArrowIpcEncoderSetCompression(
                encoder.get(), codec, NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT, &error),
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

  // The compressible column should have made the body smaller, and the body must
  // still be padded to a multiple of 8 bytes
  EXPECT_LT(body->size_bytes, uncompressed_body->size_bytes);
  EXPECT_EQ(body->size_bytes % 8, 0);

  // The first buffer in the body is the data buffer of "compressible" (its validity
  // buffer is empty and takes no space). It should be prefixed with its uncompressed
  // length.
  const int64_t int32_data_size = CompressibleRecordBatch::kLength * sizeof(int32_t);
  EXPECT_EQ(ReadLittleEndianInt64(body->data), int32_data_size);

  // The last buffer in the body is the data buffer of "incompressible", which should
  // have been stored uncompressed with a prefix of -1 (and is a multiple of 8 bytes,
  // so ends exactly at the end of the body).
  const int64_t binary_data_size =
      CompressibleRecordBatch::kLength * CompressibleRecordBatch::kBytesPerValue;
  const uint8_t* last_buffer = body->data + body->size_bytes - binary_data_size - 8;
  EXPECT_EQ(ReadLittleEndianInt64(last_buffer), -1);
  EXPECT_EQ(std::memcmp(last_buffer + 8,
                        batch.array_view()->children[2]->buffer_views[2].data.data,
                        binary_data_size),
            0);

  // Decode the header: the codec is recorded and the body length is correct
  struct ArrowBufferView message_view = {{message->data}, message->size_bytes};
  ASSERT_EQ(ArrowIpcDecoderVerifyHeader(decoder.get(), message_view, &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowIpcDecoderDecodeHeader(decoder.get(), message_view, &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_EQ(decoder->message_type, NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH);
  EXPECT_EQ(decoder->codec, codec);
  EXPECT_EQ(decoder->body_size_bytes, body->size_bytes);

  // Decode the body and compare with the original
  nanoarrow::UniqueArray decoded;
  struct ArrowBufferView body_view = {{body->data}, body->size_bytes};
  ASSERT_EQ(ArrowIpcDecoderDecodeArray(decoder.get(), body_view, -1, decoded.get(),
                                       NANOARROW_VALIDATION_LEVEL_FULL, &error),
            NANOARROW_OK)
      << error.message;

  nanoarrow::UniqueArrayView decoded_view;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(decoded_view.get(), batch.schema(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayViewSetArray(decoded_view.get(), decoded.get(), &error),
            NANOARROW_OK)
      << error.message;
  int is_equal = 0;
  ASSERT_EQ(ArrowArrayViewCompare(decoded_view.get(), batch.array_view(),
                                  NANOARROW_COMPARE_IDENTICAL, &is_equal, &error),
            NANOARROW_OK);
  EXPECT_EQ(is_equal, 1) << error.message;

  // Compression can be turned off again
  ASSERT_EQ(
      ArrowIpcEncoderSetCompression(encoder.get(), NANOARROW_IPC_COMPRESSION_TYPE_NONE,
                                    NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT, &error),
      NANOARROW_OK)
      << error.message;
  message->size_bytes = 0;
  body->size_bytes = 0;
  ASSERT_EQ(ArrowIpcEncoderEncodeSimpleRecordBatch(encoder.get(), batch.array_view(),
                                                   body.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/true, message.get()),
      NANOARROW_OK);
  EXPECT_EQ(body->size_bytes, uncompressed_body->size_bytes);
  EXPECT_EQ(std::memcmp(body->data, uncompressed_body->data, body->size_bytes), 0);

  message_view = {{message->data}, message->size_bytes};
  ASSERT_EQ(ArrowIpcDecoderDecodeHeader(decoder.get(), message_view, &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_EQ(decoder->codec, NANOARROW_IPC_COMPRESSION_TYPE_NONE);
}

TEST(NanoarrowIpcTest, NanoarrowIpcEncoderUncompressedRecordBatchAllocation) {
  nanoarrow::ipc::UniqueEncoder encoder;
  ASSERT_EQ(ArrowIpcEncoderInit(encoder.get()), NANOARROW_OK);

  // An odd number of int32 values requires four bytes of trailing padding.
  std::vector<int32_t> values(1025, 42);
  struct ArrowError error;
  nanoarrow::UniqueSchema schema;
  ASSERT_EQ(ArrowSchemaInitFromType(schema.get(), NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(schema.get(), 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInitFromType(schema->children[0], NANOARROW_TYPE_INT32),
            NANOARROW_OK);
  nanoarrow::UniqueArray array;
  ASSERT_EQ(ArrowArrayInitFromSchema(array.get(), schema.get(), &error), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(array.get()), NANOARROW_OK);
  for (int32_t value : values) {
    ASSERT_EQ(ArrowArrayAppendInt(array->children[0], value), NANOARROW_OK);
    ASSERT_EQ(ArrowArrayFinishElement(array.get()), NANOARROW_OK);
  }
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(array.get(), &error), NANOARROW_OK);
  nanoarrow::UniqueArrayView array_view;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(array_view.get(), array.get(), &error), NANOARROW_OK);

  int allocations = 0;
  auto allocator = ArrowBufferAllocatorDefault();
  allocator.private_data = &allocations;
  allocator.reallocate = [](struct ArrowBufferAllocator* allocator, uint8_t* ptr,
                            int64_t old_size, int64_t new_size) {
    ++*static_cast<int*>(allocator->private_data);
    auto default_allocator = ArrowBufferAllocatorDefault();
    return default_allocator.reallocate(&default_allocator, ptr, old_size, new_size);
  };
  nanoarrow::UniqueBuffer body;
  ASSERT_EQ(ArrowBufferSetAllocator(body.get(), allocator), NANOARROW_OK);

  ASSERT_EQ(ArrowIpcEncoderEncodeSimpleRecordBatch(encoder.get(), array_view.get(),
                                                   body.get(), &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_EQ(allocations, 1);
  ASSERT_EQ(body->size_bytes, 4104);
  EXPECT_EQ(body->capacity_bytes, 4104);
  EXPECT_EQ(std::memcmp(body->data, values.data(), 4100), 0);
  EXPECT_EQ(std::memcmp(body->data + 4100, "\0\0\0\0", 4), 0);
}

TEST(NanoarrowIpcTest, NanoarrowIpcEncoderCompressedRecordBatchLZ4) {
  if (ArrowIpcGetLZ4CompressionFunction() == nullptr) {
    GTEST_SKIP() << "nanoarrow_ipc not built with NANOARROW_IPC_WITH_LZ4";
  }
  TestCompressedRecordBatchRoundtrip(NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME);
}

TEST(NanoarrowIpcTest, NanoarrowIpcEncoderCompressedRecordBatchZstd) {
  if (ArrowIpcGetZstdCompressionFunction() == nullptr) {
    GTEST_SKIP() << "nanoarrow_ipc not built with NANOARROW_IPC_WITH_ZSTD";
  }
  TestCompressedRecordBatchRoundtrip(NANOARROW_IPC_COMPRESSION_TYPE_ZSTD);
}

TEST(NanoarrowIpcTest, NanoarrowIpcEncoderSetCompressionErrors) {
  nanoarrow::ipc::UniqueEncoder encoder;
  ASSERT_EQ(ArrowIpcEncoderInit(encoder.get()), NANOARROW_OK);
  struct ArrowError error;

  // 99 is not an enumerator; it exercises the EINVAL path
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  auto unknown_type = static_cast<enum ArrowIpcCompressionType>(99);
  EXPECT_EQ(
      ArrowIpcEncoderSetCompression(encoder.get(), unknown_type,
                                    NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT, &error),
      EINVAL);
  EXPECT_STREQ(error.message, "Unknown compression type with value 99");

  // NONE is always supported
  EXPECT_EQ(
      ArrowIpcEncoderSetCompression(encoder.get(), NANOARROW_IPC_COMPRESSION_TYPE_NONE,
                                    NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT, &error),
      NANOARROW_OK)
      << error.message;

  // Codecs that were not built in are rejected when they are set rather than when
  // the first batch is encoded
#if defined(NANOARROW_IPC_WITH_LZ4)
  EXPECT_EQ(ArrowIpcEncoderSetCompression(
                encoder.get(), NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME,
                NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT, &error),
            NANOARROW_OK)
      << error.message;
#else
  EXPECT_EQ(ArrowIpcEncoderSetCompression(
                encoder.get(), NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME,
                NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT, &error),
            ENOTSUP);
  EXPECT_STREQ(error.message,
               "Compression type with value 1 not supported by this build of nanoarrow");
#endif

#if defined(NANOARROW_IPC_WITH_ZSTD)
  EXPECT_EQ(
      ArrowIpcEncoderSetCompression(encoder.get(), NANOARROW_IPC_COMPRESSION_TYPE_ZSTD,
                                    NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT, &error),
      NANOARROW_OK)
      << error.message;
#else
  EXPECT_EQ(
      ArrowIpcEncoderSetCompression(encoder.get(), NANOARROW_IPC_COMPRESSION_TYPE_ZSTD,
                                    NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT, &error),
      ENOTSUP);
  EXPECT_STREQ(error.message,
               "Compression type with value 2 not supported by this build of nanoarrow");
#endif
}

TEST(NanoarrowIpcTest, NanoarrowIpcEncoderSetCompressor) {
  nanoarrow::ipc::UniqueEncoder encoder;
  ASSERT_EQ(ArrowIpcEncoderInit(encoder.get()), NANOARROW_OK);
  struct ArrowError error;

  // A custom compressor that explicitly does not support LZ4
  nanoarrow::ipc::UniqueCompressor compressor;
  ASSERT_EQ(ArrowIpcSerialCompressor(compressor.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcSerialCompressorSetFunction(
                compressor.get(), NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME, nullptr),
            NANOARROW_OK);

  ASSERT_EQ(ArrowIpcEncoderSetCompressor(encoder.get(), compressor.get()), NANOARROW_OK);
  // The encoder took ownership of the compressor
  EXPECT_EQ(compressor->release, nullptr);

  // With a custom compressor, support is not checked until a batch is encoded
  ASSERT_EQ(ArrowIpcEncoderSetCompression(
                encoder.get(), NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME,
                NANOARROW_IPC_COMPRESSION_LEVEL_DEFAULT, &error),
            NANOARROW_OK)
      << error.message;

  CompressibleRecordBatch batch;
  nanoarrow::UniqueBuffer body;
  EXPECT_EQ(ArrowIpcEncoderEncodeSimpleRecordBatch(encoder.get(), batch.array_view(),
                                                   body.get(), &error),
            ENOTSUP);
  EXPECT_STREQ(error.message,
               "Compression type with value 1 not supported by this build of nanoarrow");
}

// A stand-in compression function that records the level it was called with and
// "compresses" by copying (so that every buffer takes the uncompressed fallback path)
static int last_compression_level = 0;

static ArrowErrorCode RecordLevelAndCopy(struct ArrowBufferView src,
                                         int compression_level, struct ArrowBuffer* dst,
                                         struct ArrowError* error) {
  NANOARROW_UNUSED(error);
  last_compression_level = compression_level;
  return ArrowBufferAppend(dst, src.data.data, src.size_bytes);
}

TEST(NanoarrowIpcTest, NanoarrowIpcEncoderCompressionLevel) {
  nanoarrow::ipc::UniqueEncoder encoder;
  ASSERT_EQ(ArrowIpcEncoderInit(encoder.get()), NANOARROW_OK);
  nanoarrow::ipc::UniqueDecoder decoder;
  ASSERT_EQ(ArrowIpcDecoderInit(decoder.get()), NANOARROW_OK);
  struct ArrowError error;

  nanoarrow::ipc::UniqueCompressor compressor;
  ASSERT_EQ(ArrowIpcSerialCompressor(compressor.get()), NANOARROW_OK);
  ASSERT_EQ(
      ArrowIpcSerialCompressorSetFunction(
          compressor.get(), NANOARROW_IPC_COMPRESSION_TYPE_ZSTD, &RecordLevelAndCopy),
      NANOARROW_OK);
  ASSERT_EQ(ArrowIpcEncoderSetCompressor(encoder.get(), compressor.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcEncoderSetCompression(
                encoder.get(), NANOARROW_IPC_COMPRESSION_TYPE_ZSTD, 11, &error),
            NANOARROW_OK)
      << error.message;

  CompressibleRecordBatch batch;
  ASSERT_EQ(ArrowIpcDecoderSetSchema(decoder.get(), batch.schema(), &error), NANOARROW_OK)
      << error.message;

  // The level reaches the codec function
  last_compression_level = 0;
  nanoarrow::UniqueBuffer message, body;
  ASSERT_EQ(ArrowIpcEncoderEncodeSimpleRecordBatch(encoder.get(), batch.array_view(),
                                                   body.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(
      ArrowIpcEncoderFinalizeBuffer(encoder.get(), /*encapsulate=*/true, message.get()),
      NANOARROW_OK);
  EXPECT_EQ(last_compression_level, 11);

  // Copying never shrinks a buffer, so every buffer took the uncompressed (-1) path;
  // the message still declares the codec and must decode
  struct ArrowBufferView message_view = {{message->data}, message->size_bytes};
  ASSERT_EQ(ArrowIpcDecoderVerifyHeader(decoder.get(), message_view, &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowIpcDecoderDecodeHeader(decoder.get(), message_view, &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_EQ(decoder->codec, NANOARROW_IPC_COMPRESSION_TYPE_ZSTD);

  nanoarrow::UniqueArray decoded;
  struct ArrowBufferView body_view = {{body->data}, body->size_bytes};
  ASSERT_EQ(ArrowIpcDecoderDecodeArray(decoder.get(), body_view, -1, decoded.get(),
                                       NANOARROW_VALIDATION_LEVEL_FULL, &error),
            NANOARROW_OK)
      << error.message;

  nanoarrow::UniqueArrayView decoded_view;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(decoded_view.get(), batch.schema(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayViewSetArray(decoded_view.get(), decoded.get(), &error),
            NANOARROW_OK)
      << error.message;
  int is_equal = 0;
  ASSERT_EQ(ArrowArrayViewCompare(decoded_view.get(), batch.array_view(),
                                  NANOARROW_COMPARE_IDENTICAL, &is_equal, &error),
            NANOARROW_OK);
  EXPECT_EQ(is_equal, 1) << error.message;
}
