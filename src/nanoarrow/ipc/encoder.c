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

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "flatcc/flatcc_builder.h"
#include "nanoarrow/ipc/flatcc_generated.h"
#include "nanoarrow/nanoarrow.h"
#include "nanoarrow/nanoarrow_ipc.h"

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
  struct ArrowIpcDictionaryEncodings dictionary_encodings;
  // Metadata to attach to the next encoded Message (in nanoarrow's packed
  // representation), or an empty buffer if the next Message has no metadata.
  struct ArrowBuffer message_metadata;
  // Compressor for the body buffers of subsequently encoded messages (release is
  // NULL when they are not compressed)
  struct ArrowIpcCompressor compressor;
};

ArrowErrorCode ArrowIpcEncoderInit(struct ArrowIpcEncoder* encoder) {
  NANOARROW_DCHECK(encoder != NULL);
  memset(encoder, 0, sizeof(struct ArrowIpcEncoder));
  encoder->private_data = ArrowMalloc(sizeof(struct ArrowIpcEncoderPrivate));
  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;
  if (private == NULL) {
    return ENOMEM;
  }
  if (flatcc_builder_init(&private->builder) == -1) {
    ArrowFree(private);
    return ESPIPE;
  }
  private->encoding_footer = 0;
  ArrowBufferInit(&private->buffers);
  ArrowBufferInit(&private->nodes);
  ArrowIpcDictionaryEncodingsInit(&private->dictionary_encodings);
  ArrowBufferInit(&private->message_metadata);
  private->compressor.release = NULL;
  return NANOARROW_OK;
}

void ArrowIpcEncoderReset(struct ArrowIpcEncoder* encoder) {
  NANOARROW_DCHECK(encoder != NULL);
  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;
  if (private != NULL) {
    flatcc_builder_clear(&private->builder);
    ArrowBufferReset(&private->nodes);
    ArrowBufferReset(&private->buffers);
    ArrowIpcDictionaryEncodingsReset(&private->dictionary_encodings);
    ArrowBufferReset(&private->message_metadata);
    if (private->compressor.release != NULL) {
      private->compressor.release(&private->compressor);
    }
    ArrowFree(private);
  }
  memset(encoder, 0, sizeof(struct ArrowIpcEncoder));
}

ArrowErrorCode ArrowIpcEncoderSetMessageMetadata(struct ArrowIpcEncoder* encoder,
                                                 struct ArrowBuffer* metadata,
                                                 struct ArrowError* error) {
  NANOARROW_DCHECK(encoder != NULL && encoder->private_data != NULL);
  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;

  // Any previously set metadata that was not yet encoded is discarded
  ArrowBufferReset(&private->message_metadata);

  if (metadata != NULL) {
    ArrowBufferMove(metadata, &private->message_metadata);
  }

  // Metadata that can't contain a key count is empty; metadata with no keys is
  // equivalent to no metadata at all. In both cases no custom_metadata is encoded.
  if (private->message_metadata.size_bytes < (int64_t)sizeof(int32_t)) {
    ArrowBufferReset(&private->message_metadata);
    return NANOARROW_OK;
  }

  struct ArrowMetadataReader reader;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowMetadataReaderInit(&reader, (const char*)private->message_metadata.data),
      error);
  if (reader.remaining_keys <= 0) {
    ArrowBufferReset(&private->message_metadata);
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcEncoderSetCompressor(struct ArrowIpcEncoder* encoder,
                                            struct ArrowIpcCompressor* compressor) {
  NANOARROW_DCHECK(encoder != NULL && encoder->private_data != NULL &&
                   compressor != NULL && compressor->release != NULL);
  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;

  if (private->compressor.release != NULL) {
    private->compressor.release(&private->compressor);
  }

  memcpy(&private->compressor, compressor, sizeof(struct ArrowIpcCompressor));
  compressor->release = NULL;
  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcEncoderSetCompression(
    struct ArrowIpcEncoder* encoder, enum ArrowIpcCompressionType compression_type,
    int compression_level, struct ArrowError* error) {
  NANOARROW_DCHECK(encoder != NULL && encoder->private_data != NULL);
  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;

  switch (compression_type) {
    case NANOARROW_IPC_COMPRESSION_TYPE_NONE:
      if (private->compressor.release != NULL) {
        private->compressor.release(&private->compressor);
      }
      return NANOARROW_OK;
    case NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME:
    case NANOARROW_IPC_COMPRESSION_TYPE_ZSTD:
      break;
    default:
      ArrowErrorSet(error, "Unknown compression type with value %d",
                    (int)compression_type);
      return EINVAL;
  }

  // Fail now rather than when the first message is encoded if this build does not
  // support the codec or the level is out of range
  int min_level;
  int max_level;
  if (ArrowIpcGetCompressionLevelRange(compression_type, &min_level, &max_level) !=
      NANOARROW_OK) {
    ArrowErrorSet(
        error, "Compression type with value %d not supported by this build of nanoarrow",
        (int)compression_type);
    return ENOTSUP;
  }

  if (compression_level < min_level || compression_level > max_level) {
    ArrowErrorSet(error,
                  "Compression level %d is out of range for %s (expected %d to %d)",
                  compression_level, ArrowIpcCompressionTypeToString(compression_type),
                  min_level, max_level);
    return EINVAL;
  }

  struct ArrowIpcCompressor compressor;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowIpcSerialCompressor(&compressor), error);
  compressor.compression_type = compression_type;
  compressor.compression_level = compression_level;
  return ArrowIpcEncoderSetCompressor(encoder, &compressor);
}

static ArrowErrorCode ArrowIpcEncoderWriteContinuationAndSize(struct ArrowBuffer* out,
                                                              size_t size) {
  _NANOARROW_CHECK_UPPER_LIMIT(size, INT32_MAX);
  NANOARROW_RETURN_NOT_OK(ArrowBufferAppendInt32(out, -1));

  if (ArrowIpcSystemEndianness() == NANOARROW_IPC_ENDIANNESS_BIG) {
    return ArrowBufferAppendInt32(out, (int32_t)bswap32((uint32_t)size));
  } else {
    return ArrowBufferAppendInt32(out, (int32_t)size);
  }
}

ArrowErrorCode ArrowIpcEncoderFinalizeBuffer(struct ArrowIpcEncoder* encoder,
                                             char encapsulate, struct ArrowBuffer* out) {
  NANOARROW_DCHECK(encoder != NULL && encoder->private_data != NULL && out != NULL);
  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;

  size_t size = flatcc_builder_get_buffer_size(&private->builder);

  if (encapsulate) {
    int64_t padded_size = _ArrowRoundUpToMultipleOf8(size);
    NANOARROW_RETURN_NOT_OK(
        ArrowBufferReserve(out, sizeof(int32_t) + sizeof(int32_t) + padded_size));
    NANOARROW_ASSERT_OK(ArrowIpcEncoderWriteContinuationAndSize(out, padded_size));
  } else {
    NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(out, size));
  }

  if (size == 0) {
    // Finalizing an empty flatcc_builder_t triggers an assertion
    return NANOARROW_OK;
  }

  void* data =
      flatcc_builder_copy_buffer(&private->builder, out->data + out->size_bytes, size);
  NANOARROW_DCHECK(data != NULL);
  NANOARROW_UNUSED(data);
  out->size_bytes += size;

  while (encapsulate && out->size_bytes % 8 != 0) {
    // zero padding bytes, if any
    out->data[out->size_bytes++] = 0;
  }

  // don't deallocate yet, just wipe the builder's current Message
  flatcc_builder_reset(&private->builder);
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowIpcEncodeFieldType(flatcc_builder_t* builder,
                                              const struct ArrowSchemaView* schema_view,
                                              struct ArrowError* error) {
  switch (schema_view->type) {
    case NANOARROW_TYPE_NA:
      FLATCC_RETURN_UNLESS_0(Field_type_Null_create(builder), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_BOOL:
      FLATCC_RETURN_UNLESS_0(Field_type_Bool_create(builder), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT8:
      FLATCC_RETURN_UNLESS_0(
          Field_type_Int_create(builder, 8, schema_view->type == NANOARROW_TYPE_INT8),
          error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT16:
      FLATCC_RETURN_UNLESS_0(
          Field_type_Int_create(builder, 16, schema_view->type == NANOARROW_TYPE_INT16),
          error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT32:
      FLATCC_RETURN_UNLESS_0(
          Field_type_Int_create(builder, 32, schema_view->type == NANOARROW_TYPE_INT32),
          error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_INT64:
      FLATCC_RETURN_UNLESS_0(
          Field_type_Int_create(builder, 64, schema_view->type == NANOARROW_TYPE_INT64),
          error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_HALF_FLOAT:
      FLATCC_RETURN_UNLESS_0(Field_type_FloatingPoint_create(builder, ns(Precision_HALF)),
                             error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_FLOAT:
      FLATCC_RETURN_UNLESS_0(
          Field_type_FloatingPoint_create(builder, ns(Precision_SINGLE)), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_DOUBLE:
      FLATCC_RETURN_UNLESS_0(
          Field_type_FloatingPoint_create(builder, ns(Precision_DOUBLE)), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_DECIMAL32:
    case NANOARROW_TYPE_DECIMAL64:
    case NANOARROW_TYPE_DECIMAL128:
    case NANOARROW_TYPE_DECIMAL256:
      FLATCC_RETURN_UNLESS_0(
          Field_type_Decimal_create(builder, schema_view->decimal_precision,
                                    schema_view->decimal_scale,
                                    schema_view->decimal_bitwidth),
          error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_STRING:
      FLATCC_RETURN_UNLESS_0(Field_type_Utf8_create(builder), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_LARGE_STRING:
      FLATCC_RETURN_UNLESS_0(Field_type_LargeUtf8_create(builder), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_BINARY:
      FLATCC_RETURN_UNLESS_0(Field_type_Binary_create(builder), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_LARGE_BINARY:
      FLATCC_RETURN_UNLESS_0(Field_type_LargeBinary_create(builder), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_DATE32:
      FLATCC_RETURN_UNLESS_0(Field_type_Date_create(builder, ns(DateUnit_DAY)), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_DATE64:
      FLATCC_RETURN_UNLESS_0(Field_type_Date_create(builder, ns(DateUnit_MILLISECOND)),
                             error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_INTERVAL_MONTHS:
      FLATCC_RETURN_UNLESS_0(
          Field_type_Interval_create(builder, ns(IntervalUnit_YEAR_MONTH)), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_INTERVAL_DAY_TIME:
      FLATCC_RETURN_UNLESS_0(
          Field_type_Interval_create(builder, ns(IntervalUnit_DAY_TIME)), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO:
      FLATCC_RETURN_UNLESS_0(
          Field_type_Interval_create(builder, ns(IntervalUnit_MONTH_DAY_NANO)), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_TIMESTAMP:
      FLATCC_RETURN_UNLESS_0(Field_type_Timestamp_start(builder), error);
      FLATCC_RETURN_UNLESS_0(
          Timestamp_unit_add(builder, (ns(TimeUnit_enum_t))schema_view->time_unit),
          error);
      if (schema_view->timezone && schema_view->timezone[0] != 0) {
        FLATCC_RETURN_UNLESS_0(
            Timestamp_timezone_create_str(builder, schema_view->timezone), error);
      }
      FLATCC_RETURN_UNLESS_0(Field_type_Timestamp_end(builder), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_TIME32:
      FLATCC_RETURN_UNLESS_0(
          Field_type_Time_create(builder, (ns(TimeUnit_enum_t))schema_view->time_unit,
                                 32),
          error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_TIME64:
      FLATCC_RETURN_UNLESS_0(
          Field_type_Time_create(builder, (ns(TimeUnit_enum_t))schema_view->time_unit,
                                 64),
          error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_DURATION:
      FLATCC_RETURN_UNLESS_0(Field_type_Duration_create(
                                 builder, (ns(TimeUnit_enum_t))schema_view->time_unit),
                             error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
      FLATCC_RETURN_UNLESS_0(
          Field_type_FixedSizeBinary_create(builder, schema_view->fixed_size), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_LIST:
      FLATCC_RETURN_UNLESS_0(Field_type_List_create(builder), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_LARGE_LIST:
      FLATCC_RETURN_UNLESS_0(Field_type_LargeList_create(builder), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      FLATCC_RETURN_UNLESS_0(
          Field_type_FixedSizeList_create(builder, schema_view->fixed_size), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_RUN_END_ENCODED:
      FLATCC_RETURN_UNLESS_0(Field_type_RunEndEncoded_create(builder), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_STRUCT:
      FLATCC_RETURN_UNLESS_0(Field_type_Struct__create(builder), error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_SPARSE_UNION:
    case NANOARROW_TYPE_DENSE_UNION: {
      FLATCC_RETURN_UNLESS_0(Field_type_Union_start(builder), error);

      FLATCC_RETURN_UNLESS_0(
          Union_mode_add(builder, schema_view->type == NANOARROW_TYPE_DENSE_UNION),
          error);
      if (schema_view->union_type_ids) {
        int8_t type_ids[128];
        int n = _ArrowParseUnionTypeIds(schema_view->union_type_ids, type_ids);
        if (n != 0) {
          FLATCC_RETURN_UNLESS_0(Union_typeIds_start(builder), error);
          int32_t* type_ids_32 = (int32_t*)ns(Union_typeIds_extend(builder, n));
          FLATCC_RETURN_IF_NULL(type_ids_32, error);

          for (int i = 0; i < n; i++) {
            type_ids_32[i] = type_ids[i];
          }
          FLATCC_RETURN_UNLESS_0(Union_typeIds_end(builder), error);
        }
      }

      FLATCC_RETURN_UNLESS_0(Field_type_Union_end(builder), error);
      return NANOARROW_OK;
    }

    case NANOARROW_TYPE_MAP:
      FLATCC_RETURN_UNLESS_0(
          Field_type_Map_create(builder,
                                schema_view->schema->flags & ARROW_FLAG_MAP_KEYS_SORTED),
          error);
      return NANOARROW_OK;

    case NANOARROW_TYPE_DICTIONARY:
      ArrowErrorSet(error, "IPC encoding of dictionary types unsupported");
      return ENOTSUP;

    default:
      ArrowErrorSet(error, "Expected a valid enum ArrowType value but found %d",
                    schema_view->type);
      return EINVAL;
  }
}

static ArrowErrorCode ArrowIpcEncodeField(
    flatcc_builder_t* builder, const struct ArrowSchema* schema,
    const struct ArrowIpcDictionaryEncodings* dictionary_encodings,
    struct ArrowError* error);

static ArrowErrorCode ArrowIpcEncodeMetadata(flatcc_builder_t* builder,
                                             const char* packed_metadata,
                                             int (*push_start)(flatcc_builder_t*),
                                             ns(KeyValue_ref_t) *
                                                 (*push_end)(flatcc_builder_t*),
                                             struct ArrowError* error) {
  struct ArrowMetadataReader metadata;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowMetadataReaderInit(&metadata, packed_metadata),
                                     error);
  while (metadata.remaining_keys > 0) {
    struct ArrowStringView key, value;
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowMetadataReaderRead(&metadata, &key, &value),
                                       error);
    FLATCC_RETURN_UNLESS_0_NO_NS(push_start(builder), error);
    FLATCC_RETURN_UNLESS_0(KeyValue_key_create_strn(builder, key.data, key.size_bytes),
                           error);
    FLATCC_RETURN_UNLESS_0(
        KeyValue_value_create_strn(builder, value.data, value.size_bytes), error);
    FLATCC_RETURN_IF_NULL(push_end(builder), error);
  }
  return NANOARROW_OK;
}

// Encodes any metadata set by ArrowIpcEncoderSetMessageMetadata() into the Message
// currently under construction and clears it, such that it applies to exactly one
// Message.
static ArrowErrorCode ArrowIpcEncodeMessageMetadata(
    struct ArrowIpcEncoderPrivate* private, struct ArrowError* error) {
  if (private->message_metadata.size_bytes == 0) {
    return NANOARROW_OK;
  }

  flatcc_builder_t* builder = &private->builder;
  FLATCC_RETURN_UNLESS_0(Message_custom_metadata_start(builder), error);
  NANOARROW_RETURN_NOT_OK(
      ArrowIpcEncodeMetadata(builder, (const char*)private->message_metadata.data,
                             &ns(Message_custom_metadata_push_start),
                             &ns(Message_custom_metadata_push_end), error));
  FLATCC_RETURN_UNLESS_0(Message_custom_metadata_end(builder), error);

  ArrowBufferReset(&private->message_metadata);
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowIpcEncodeFields(
    flatcc_builder_t* builder, const struct ArrowSchema* schema,
    int (*push_start)(flatcc_builder_t*),
    ns(Field_ref_t) * (*push_end)(flatcc_builder_t*),
    const struct ArrowIpcDictionaryEncodings* dictionary_encodings,
    struct ArrowError* error) {
  for (int i = 0; i < schema->n_children; i++) {
    FLATCC_RETURN_UNLESS_0_NO_NS(push_start(builder), error);
    NANOARROW_RETURN_NOT_OK(
        ArrowIpcEncodeField(builder, schema->children[i], dictionary_encodings, error));
    FLATCC_RETURN_IF_NULL(push_end(builder), error);
  }
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowIpcEncodeField(
    flatcc_builder_t* builder, const struct ArrowSchema* schema,
    const struct ArrowIpcDictionaryEncodings* dictionary_encodings,
    struct ArrowError* error) {
  // Check before ArrowSchemaViewInit(), which assumes dictionary values are not
  // themselves dictionary-encoded.
  if (schema->dictionary != NULL && schema->dictionary->dictionary != NULL) {
    ArrowErrorSet(error, "IPC encoding of nested dictionary values unsupported");
    return ENOTSUP;
  }

  FLATCC_RETURN_UNLESS_0(Field_name_create_str(builder, schema->name), error);
  FLATCC_RETURN_UNLESS_0(
      Field_nullable_add(builder, (schema->flags & ARROW_FLAG_NULLABLE) != 0), error);

  struct ArrowSchemaView schema_view;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&schema_view, schema, error));

  if (schema_view.type == NANOARROW_TYPE_DICTIONARY) {
    const struct ArrowIpcDictionaryEncoding* encoding =
        ArrowIpcDictionaryEncodingsFind(dictionary_encodings, schema);

    // We just computed these dictionary ids, so we should be able to resolve them
    if (encoding == NULL) {
      ArrowErrorSet(error, "Unexpected missing dictionary encoding for field");
      return EINVAL;
    }

    // Determine the index type's bitWidth and is_signed from the storage_type
    int32_t index_bitwidth;
    flatbuffers_bool_t index_is_signed;
    switch (schema_view.storage_type) {
      case NANOARROW_TYPE_INT8:
        index_bitwidth = 8;
        index_is_signed = 1;
        break;
      case NANOARROW_TYPE_UINT8:
        index_bitwidth = 8;
        index_is_signed = 0;
        break;
      case NANOARROW_TYPE_INT16:
        index_bitwidth = 16;
        index_is_signed = 1;
        break;
      case NANOARROW_TYPE_UINT16:
        index_bitwidth = 16;
        index_is_signed = 0;
        break;
      case NANOARROW_TYPE_INT32:
        index_bitwidth = 32;
        index_is_signed = 1;
        break;
      case NANOARROW_TYPE_UINT32:
        index_bitwidth = 32;
        index_is_signed = 0;
        break;
      case NANOARROW_TYPE_INT64:
        index_bitwidth = 64;
        index_is_signed = 1;
        break;
      case NANOARROW_TYPE_UINT64:
        index_bitwidth = 64;
        index_is_signed = 0;
        break;
      default:
        ArrowErrorSet(error, "Invalid dictionary index type: %s",
                      ArrowTypeString(schema_view.storage_type));
        return EINVAL;
    }

    // Create the Int type for the index type
    ns(Int_ref_t) index_type_ref =
        ns(Int_create(builder, index_bitwidth, index_is_signed));
    FLATCC_RETURN_IF_NULL(index_type_ref, error);

    // Create the DictionaryEncoding with id, indexType, isOrdered, and dictionaryKind
    flatbuffers_bool_t is_ordered = (schema->flags & ARROW_FLAG_DICTIONARY_ORDERED) != 0;
    ns(DictionaryEncoding_ref_t) dict_encoding_ref =
        ns(DictionaryEncoding_create(builder, encoding->id, index_type_ref, is_ordered,
                                     ns(DictionaryKind_DenseArray)));
    FLATCC_RETURN_IF_NULL(dict_encoding_ref, error);

    // Add the dictionary encoding to the field
    FLATCC_RETURN_UNLESS_0(Field_dictionary_add(builder, dict_encoding_ref), error);

    // Support dictionary values with children by encoding children from
    // schema->dictionary (and add a roundtrip test for a nested value type).
    // Using schema below would encode the index type's children instead and
    // produce a Field whose type and children do not agree.
    if (schema->dictionary->n_children != 0) {
      ArrowErrorSet(error, "IPC encoding of dictionary values with children unsupported");
      return ENOTSUP;
    }

    NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&schema_view, schema->dictionary, error));
  }

  NANOARROW_RETURN_NOT_OK(ArrowIpcEncodeFieldType(builder, &schema_view, error));

  if (schema->n_children != 0) {
    FLATCC_RETURN_UNLESS_0(Field_children_start(builder), error);
    NANOARROW_RETURN_NOT_OK(
        ArrowIpcEncodeFields(builder, schema, &ns(Field_children_push_start),
                             &ns(Field_children_push_end), dictionary_encodings, error));
    FLATCC_RETURN_UNLESS_0(Field_children_end(builder), error);
  }

  if (schema->metadata) {
    FLATCC_RETURN_UNLESS_0(Field_custom_metadata_start(builder), error);
    NANOARROW_RETURN_NOT_OK(ArrowIpcEncodeMetadata(
        builder, schema->metadata, &ns(Field_custom_metadata_push_start),
        &ns(Field_custom_metadata_push_end), error));
    FLATCC_RETURN_UNLESS_0(Field_custom_metadata_end(builder), error);
  }
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowIpcEncodeSchema(
    flatcc_builder_t* builder, const struct ArrowSchema* schema,
    const struct ArrowIpcDictionaryEncodings* dictionary_encodings,
    struct ArrowError* error) {
  NANOARROW_DCHECK(schema->release != NULL);

  if (strcmp(schema->format, "+s") != 0) {
    ArrowErrorSet(
        error,
        "Cannot encode schema with format '%s'; top level schema must have struct type",
        schema->format);
    return EINVAL;
  }

  if (ArrowIpcSystemEndianness() == NANOARROW_IPC_ENDIANNESS_LITTLE) {
    FLATCC_RETURN_UNLESS_0(Schema_endianness_add(builder, ns(Endianness_Little)), error);
  } else {
    FLATCC_RETURN_UNLESS_0(Schema_endianness_add(builder, ns(Endianness_Big)), error);
  }

  FLATCC_RETURN_UNLESS_0(Schema_fields_start(builder), error);
  NANOARROW_RETURN_NOT_OK(
      ArrowIpcEncodeFields(builder, schema, &ns(Schema_fields_push_start),
                           &ns(Schema_fields_push_end), dictionary_encodings, error));
  FLATCC_RETURN_UNLESS_0(Schema_fields_end(builder), error);

  FLATCC_RETURN_UNLESS_0(Schema_custom_metadata_start(builder), error);
  if (schema->metadata) {
    NANOARROW_RETURN_NOT_OK(ArrowIpcEncodeMetadata(
        builder, schema->metadata, &ns(Schema_custom_metadata_push_start),
        &ns(Schema_custom_metadata_push_end), error));
  }
  FLATCC_RETURN_UNLESS_0(Schema_custom_metadata_end(builder), error);

  FLATCC_RETURN_UNLESS_0(Schema_features_start(builder), error);
  FLATCC_RETURN_UNLESS_0(Schema_features_end(builder), error);

  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcEncoderEncodeSchema(struct ArrowIpcEncoder* encoder,
                                           const struct ArrowSchema* schema,
                                           struct ArrowError* error) {
  NANOARROW_DCHECK(encoder != NULL && encoder->private_data != NULL && schema != NULL);

  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;

  flatcc_builder_t* builder = &private->builder;

  FLATCC_RETURN_UNLESS_0(Message_start_as_root(builder), error);

  FLATCC_RETURN_UNLESS_0(Message_version_add(builder, ns(MetadataVersion_V5)), error);

  FLATCC_RETURN_UNLESS_0(Message_header_Schema_start(builder), error);

  // Look for any fields of the schema that should be dictionary encoded
  if (private->dictionary_encodings.encodings.size_bytes > 0) {
    ArrowIpcDictionaryEncodingsReset(&private->dictionary_encodings);
    ArrowIpcDictionaryEncodingsInit(&private->dictionary_encodings);
  }
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowIpcDictionaryEncodingsAppendSchema(&private->dictionary_encodings, schema),
      error);

  NANOARROW_RETURN_NOT_OK(
      ArrowIpcEncodeSchema(builder, schema, &private->dictionary_encodings, error));

  FLATCC_RETURN_UNLESS_0(Message_header_Schema_end(builder), error);

  NANOARROW_RETURN_NOT_OK(ArrowIpcEncodeMessageMetadata(private, error));

  FLATCC_RETURN_UNLESS_0(Message_bodyLength_add(builder, 0), error);

  FLATCC_RETURN_IF_NULL(ns(Message_end_as_root(builder)), error);
  return NANOARROW_OK;
}

struct ArrowIpcBufferEncoder {
  /// \brief Callback invoked against each buffer to be encoded
  ///
  /// Encoding of buffers is left as a callback to accommodate dissociated data storage.
  /// One implementation of this callback might copy all buffers into a contiguous body
  /// for use in an arrow IPC stream, another implementation might store offsets and
  /// lengths relative to a known arena.
  ArrowErrorCode (*encode_buffer)(struct ArrowBufferView buffer_view,
                                  struct ArrowIpcEncoder* encoder,
                                  struct ArrowIpcBufferEncoder* buffer_encoder,
                                  int64_t* offset, int64_t* length,
                                  struct ArrowError* error);

  /// \brief Pointer to arbitrary data used by encode_buffer()
  void* encode_buffer_state;

  /// \brief Finalized body length of the most recently encoded RecordBatch message
  ///
  /// encode_buffer() is expected to update this while encoding each buffer. After all
  /// buffers are encoded, this will be written to the RecordBatch's .bodyLength
  int64_t body_length;
};

// The codec applied to message bodies (NONE when no compressor is set)
static enum ArrowIpcCompressionType ArrowIpcEncoderCodec(
    struct ArrowIpcEncoderPrivate* private) {
  if (private->compressor.release == NULL) {
    return NANOARROW_IPC_COMPRESSION_TYPE_NONE;
  }
  return private->compressor.compression_type;
}

// Append buffer_view to body_buffer as a compressed IPC buffer: the uncompressed length
// as a little-endian int64 followed by the compressed bytes. If compression does not
// reduce the size, the buffer is stored uncompressed with a length prefix of -1 instead.
static ArrowErrorCode ArrowIpcEncoderAppendCompressedBuffer(
    struct ArrowIpcEncoderPrivate* private, struct ArrowBufferView buffer_view,
    struct ArrowBuffer* body_buffer, struct ArrowError* error) {
  NANOARROW_DCHECK(ArrowIpcEncoderCodec(private) != NANOARROW_IPC_COMPRESSION_TYPE_NONE);

  // placeholder for the prefix, then compress directly into the body
  int64_t prefix_offset = body_buffer->size_bytes;
  int64_t payload_offset = prefix_offset + (int64_t)sizeof(int64_t);
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBufferAppendInt64(body_buffer, 0), error);
  NANOARROW_RETURN_NOT_OK(private->compressor.compress(&private->compressor, buffer_view,
                                                       body_buffer, error));

  int64_t prefix = buffer_view.size_bytes;
  if (body_buffer->size_bytes - payload_offset >= buffer_view.size_bytes) {
    body_buffer->size_bytes = payload_offset;
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowBufferAppend(body_buffer, buffer_view.data.data, buffer_view.size_bytes),
        error);
    prefix = -1;
  }

  // the prefix is always little endian
  if (ArrowIpcSystemEndianness() == NANOARROW_IPC_ENDIANNESS_BIG) {
    prefix = (int64_t)bswap64((uint64_t)prefix);
  }
  memcpy(body_buffer->data + prefix_offset, &prefix, sizeof(int64_t));
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowIpcEncoderBuildContiguousBodyBufferCallback(
    struct ArrowBufferView buffer_view, struct ArrowIpcEncoder* encoder,
    struct ArrowIpcBufferEncoder* buffer_encoder, int64_t* offset, int64_t* length,
    struct ArrowError* error) {
  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;
  struct ArrowBuffer* body_buffer =
      (struct ArrowBuffer*)buffer_encoder->encode_buffer_state;

  int64_t buffer_begin = _ArrowRoundUpToMultipleOf8(body_buffer->size_bytes);
  // Empty buffers are never compressed (nor length-prefixed), matching Arrow C++.
  int needs_compression =
      ArrowIpcEncoderCodec(private) != NANOARROW_IPC_COMPRESSION_TYPE_NONE &&
      buffer_view.size_bytes > 0;
  if (!needs_compression) {
    // Reserve the data and padding together to avoid growing the buffer twice.
    int64_t new_size = _ArrowRoundUpToMultipleOf8(buffer_begin + buffer_view.size_bytes);
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowBufferReserve(body_buffer, new_size - body_buffer->size_bytes), error);
  }

  // zero padding up to the start of the buffer
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowBufferAppendFill(body_buffer, 0, buffer_begin - body_buffer->size_bytes),
      error);

  if (needs_compression) {
    NANOARROW_RETURN_NOT_OK(
        ArrowIpcEncoderAppendCompressedBuffer(private, buffer_view, body_buffer, error));
  } else {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowBufferAppend(body_buffer, buffer_view.data.data, buffer_view.size_bytes),
        error);
  }

  // store offset and length (including any prefix) of the buffer
  *offset = buffer_begin;
  *length = body_buffer->size_bytes - buffer_begin;

  // zero padding after writing the buffer
  int64_t buffer_end = body_buffer->size_bytes;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowBufferAppendFill(body_buffer, 0,
                            _ArrowRoundUpToMultipleOf8(buffer_end) - buffer_end),
      error);

  buffer_encoder->body_length = body_buffer->size_bytes;
  return NANOARROW_OK;
}

// Add the BodyCompression table to the RecordBatch currently being built, if any
// compression is enabled. Bodies of RecordBatch and DictionaryBatch messages are both
// built by the same buffer encoder, so both need this.
static ArrowErrorCode ArrowIpcEncoderEncodeBodyCompression(
    struct ArrowIpcEncoderPrivate* private, struct ArrowError* error) {
  ns(CompressionType_enum_t) codec;
  switch (ArrowIpcEncoderCodec(private)) {
    case NANOARROW_IPC_COMPRESSION_TYPE_NONE:
      return NANOARROW_OK;
    case NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME:
      codec = ns(CompressionType_LZ4_FRAME);
      break;
    case NANOARROW_IPC_COMPRESSION_TYPE_ZSTD:
      codec = ns(CompressionType_ZSTD);
      break;
    default:
      ArrowErrorSet(error, "Unknown compression type with value %d",
                    (int)private->compressor.compression_type);
      return EINVAL;
  }

  flatcc_builder_t* builder = &private->builder;
  FLATCC_RETURN_UNLESS_0(RecordBatch_compression_start(builder), error);
  FLATCC_RETURN_UNLESS_0(BodyCompression_codec_add(builder, codec), error);
  FLATCC_RETURN_UNLESS_0(
      BodyCompression_method_add(builder, ns(BodyCompressionMethod_BUFFER)), error);
  FLATCC_RETURN_UNLESS_0(RecordBatch_compression_end(builder), error);
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowIpcEncoderEncodeRecordBatchImpl(
    struct ArrowIpcEncoder* encoder, struct ArrowIpcBufferEncoder* buffer_encoder,
    const struct ArrowArrayView* array_view, struct ArrowBuffer* buffers,
    struct ArrowBuffer* nodes, struct ArrowError* error) {
  if (array_view->offset != 0) {
    ArrowErrorSet(error, "Cannot encode arrays with nonzero offset");
    return ENOTSUP;
  }

  if (array_view->dictionary != NULL) {
    // Values live in a separate DictionaryBatch message per the Arrow IPC spec;
    // the parent's index node + buffers were already emitted by the caller loop,
    // so stop recursing here.
    return NANOARROW_OK;
  }

  for (int64_t c = 0; c < array_view->n_children; ++c) {
    const struct ArrowArrayView* child = array_view->children[c];

    struct ns(FieldNode) node = {child->length, child->null_count};
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBufferAppend(nodes, &node, sizeof(node)),
                                       error);

    for (int64_t b = 0; b < child->array->n_buffers; ++b) {
      struct ns(Buffer) buffer;
      NANOARROW_RETURN_NOT_OK(
          buffer_encoder->encode_buffer(child->buffer_views[b], encoder, buffer_encoder,
                                        &buffer.offset, &buffer.length, error));
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(
          ArrowBufferAppend(buffers, &buffer, sizeof(buffer)), error);
    }

    NANOARROW_RETURN_NOT_OK(ArrowIpcEncoderEncodeRecordBatchImpl(
        encoder, buffer_encoder, child, buffers, nodes, error));
  }
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowIpcEncoderEncodeRecordBatch(
    struct ArrowIpcEncoder* encoder, struct ArrowIpcBufferEncoder* buffer_encoder,
    const struct ArrowArrayView* array_view, struct ArrowError* error) {
  NANOARROW_DCHECK(encoder != NULL && encoder->private_data != NULL &&
                   buffer_encoder != NULL && buffer_encoder->encode_buffer != NULL);
  if (array_view->null_count != 0 && ArrowArrayViewComputeNullCount(array_view) != 0) {
    ArrowErrorSet(error,
                  "RecordBatches cannot be constructed from arrays with top level nulls");
    return EINVAL;
  }

  if (array_view->storage_type != NANOARROW_TYPE_STRUCT) {
    ArrowErrorSet(
        error,
        "RecordBatches cannot be constructed from arrays of type other than struct");
    return EINVAL;
  }

  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;

  flatcc_builder_t* builder = &private->builder;

  FLATCC_RETURN_UNLESS_0(Message_start_as_root(builder), error);
  FLATCC_RETURN_UNLESS_0(Message_version_add(builder, ns(MetadataVersion_V5)), error);

  FLATCC_RETURN_UNLESS_0(Message_header_RecordBatch_start(builder), error);
  FLATCC_RETURN_UNLESS_0(RecordBatch_length_add(builder, array_view->length), error);

  NANOARROW_RETURN_NOT_OK(ArrowIpcEncoderEncodeBodyCompression(private, error));

  NANOARROW_ASSERT_OK(ArrowBufferResize(&private->buffers, 0, 0));
  NANOARROW_ASSERT_OK(ArrowBufferResize(&private->nodes, 0, 0));
  NANOARROW_RETURN_NOT_OK(ArrowIpcEncoderEncodeRecordBatchImpl(
      encoder, buffer_encoder, array_view, &private->buffers, &private->nodes, error));

  FLATCC_RETURN_UNLESS_0(RecordBatch_nodes_create(  //
                             builder, (struct ns(FieldNode)*)private->nodes.data,
                             private->nodes.size_bytes / sizeof(struct ns(FieldNode))),
                         error);
  FLATCC_RETURN_UNLESS_0(RecordBatch_buffers_create(  //
                             builder, (struct ns(Buffer)*)private->buffers.data,
                             private->buffers.size_bytes / sizeof(struct ns(Buffer))),
                         error);

  FLATCC_RETURN_UNLESS_0(Message_header_RecordBatch_end(builder), error);

  NANOARROW_RETURN_NOT_OK(ArrowIpcEncodeMessageMetadata(private, error));

  FLATCC_RETURN_UNLESS_0(Message_bodyLength_add(builder, buffer_encoder->body_length),
                         error);
  FLATCC_RETURN_IF_NULL(ns(Message_end_as_root(builder)), error);
  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcEncoderEncodeSimpleRecordBatch(
    struct ArrowIpcEncoder* encoder, const struct ArrowArrayView* array_view,
    struct ArrowBuffer* body_buffer, struct ArrowError* error) {
  NANOARROW_DCHECK(encoder != NULL && encoder->private_data != NULL &&
                   body_buffer != NULL);

  struct ArrowIpcBufferEncoder buffer_encoder = {
      .encode_buffer = &ArrowIpcEncoderBuildContiguousBodyBufferCallback,
      .encode_buffer_state = body_buffer,
      .body_length = 0,
  };

  return ArrowIpcEncoderEncodeRecordBatch(encoder, &buffer_encoder, array_view, error);
}

static ArrowErrorCode ArrowIpcEncoderEncodeDictionaryBatch(
    struct ArrowIpcEncoder* encoder, struct ArrowIpcBufferEncoder* buffer_encoder,
    int64_t dictionary_id, char is_delta, const struct ArrowArrayView* values_view,
    struct ArrowError* error) {
  NANOARROW_DCHECK(encoder != NULL && encoder->private_data != NULL &&
                   buffer_encoder != NULL && buffer_encoder->encode_buffer != NULL);
  if (values_view->dictionary != NULL) {
    ArrowErrorSet(error,
                  "DictionaryBatch values array must not itself be dictionary-encoded");
    return EINVAL;
  }

  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;
  flatcc_builder_t* builder = &private->builder;

  FLATCC_RETURN_UNLESS_0(Message_start_as_root(builder), error);
  FLATCC_RETURN_UNLESS_0(Message_version_add(builder, ns(MetadataVersion_V5)), error);

  FLATCC_RETURN_UNLESS_0(Message_header_DictionaryBatch_start(builder), error);
  FLATCC_RETURN_UNLESS_0(DictionaryBatch_id_add(builder, dictionary_id), error);
  FLATCC_RETURN_UNLESS_0(DictionaryBatch_data_start(builder), error);
  FLATCC_RETURN_UNLESS_0(RecordBatch_length_add(builder, values_view->length), error);
  NANOARROW_RETURN_NOT_OK(ArrowIpcEncoderEncodeBodyCompression(private, error));

  NANOARROW_ASSERT_OK(ArrowBufferResize(&private->buffers, 0, 0));
  NANOARROW_ASSERT_OK(ArrowBufferResize(&private->nodes, 0, 0));

  // The values array is a single top-level column. Emit the top-level node +
  // buffers here, then descend into any nested children.
  struct ns(FieldNode) top_node = {values_view->length, values_view->null_count};
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowBufferAppend(&private->nodes, &top_node, sizeof(top_node)), error);
  for (int64_t b = 0; b < values_view->array->n_buffers; ++b) {
    struct ns(Buffer) buffer;
    NANOARROW_RETURN_NOT_OK(buffer_encoder->encode_buffer(
        values_view->buffer_views[b], encoder, buffer_encoder, &buffer.offset,
        &buffer.length, error));
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowBufferAppend(&private->buffers, &buffer, sizeof(buffer)), error);
  }
  NANOARROW_RETURN_NOT_OK(ArrowIpcEncoderEncodeRecordBatchImpl(
      encoder, buffer_encoder, values_view, &private->buffers, &private->nodes, error));

  FLATCC_RETURN_UNLESS_0(
      RecordBatch_nodes_create(builder, (struct ns(FieldNode)*)private->nodes.data,
                               private->nodes.size_bytes / sizeof(struct ns(FieldNode))),
      error);
  FLATCC_RETURN_UNLESS_0(
      RecordBatch_buffers_create(builder, (struct ns(Buffer)*)private->buffers.data,
                                 private->buffers.size_bytes / sizeof(struct ns(Buffer))),
      error);
  FLATCC_RETURN_UNLESS_0(DictionaryBatch_data_end(builder), error);
  FLATCC_RETURN_UNLESS_0(DictionaryBatch_isDelta_add(builder, is_delta ? 1 : 0), error);
  FLATCC_RETURN_UNLESS_0(Message_header_DictionaryBatch_end(builder), error);
  FLATCC_RETURN_UNLESS_0(Message_bodyLength_add(builder, buffer_encoder->body_length),
                         error);
  FLATCC_RETURN_IF_NULL(ns(Message_end_as_root(builder)), error);
  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcEncoderEncodeSimpleDictionaryBatch(
    struct ArrowIpcEncoder* encoder, int64_t dictionary_id, char is_delta,
    const struct ArrowArrayView* values_view, struct ArrowBuffer* body_buffer,
    struct ArrowError* error) {
  NANOARROW_DCHECK(encoder != NULL && encoder->private_data != NULL &&
                   body_buffer != NULL);
  struct ArrowIpcBufferEncoder buffer_encoder = {
      .encode_buffer = &ArrowIpcEncoderBuildContiguousBodyBufferCallback,
      .encode_buffer_state = body_buffer,
      .body_length = 0,
  };
  return ArrowIpcEncoderEncodeDictionaryBatch(encoder, &buffer_encoder, dictionary_id,
                                              is_delta, values_view, error);
}

void ArrowIpcFooterInit(struct ArrowIpcFooter* footer) {
  footer->schema.release = NULL;
  ArrowBufferInit(&footer->record_batch_blocks);
  ArrowBufferInit(&footer->dictionary_blocks);
  ArrowIpcDictionaryEncodingsInit(&footer->dictionaries);
}

void ArrowIpcFooterReset(struct ArrowIpcFooter* footer) {
  if (footer->schema.release != NULL) {
    ArrowSchemaRelease(&footer->schema);
  }
  ArrowBufferReset(&footer->record_batch_blocks);
  ArrowBufferReset(&footer->dictionary_blocks);
  ArrowIpcDictionaryEncodingsReset(&footer->dictionaries);
}

ArrowErrorCode ArrowIpcEncoderEncodeFooter(struct ArrowIpcEncoder* encoder,
                                           const struct ArrowIpcFooter* footer,
                                           struct ArrowError* error) {
  NANOARROW_DCHECK(encoder != NULL && encoder->private_data != NULL && footer != NULL);

  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;

  flatcc_builder_t* builder = &private->builder;

  FLATCC_RETURN_UNLESS_0(Footer_start_as_root(builder), error);

  FLATCC_RETURN_UNLESS_0(Footer_version_add(builder, ns(MetadataVersion_V5)), error);

  FLATCC_RETURN_UNLESS_0(Footer_schema_start(builder), error);
  NANOARROW_RETURN_NOT_OK(
      ArrowIpcEncodeSchema(builder, &footer->schema, &footer->dictionaries, error));
  FLATCC_RETURN_UNLESS_0(Footer_schema_end(builder), error);

  const struct ArrowIpcFileBlock* blocks =
      (struct ArrowIpcFileBlock*)footer->record_batch_blocks.data;
  int64_t n_blocks =
      footer->record_batch_blocks.size_bytes / sizeof(struct ArrowIpcFileBlock);

  FLATCC_RETURN_UNLESS_0(Footer_recordBatches_start(builder), error);
  struct ns(Block)* flatcc_RecordBatch_blocks =
      ns(Footer_recordBatches_extend(builder, n_blocks));
  FLATCC_RETURN_IF_NULL(flatcc_RecordBatch_blocks, error);
  for (int64_t i = 0; i < n_blocks; i++) {
    struct ns(Block) block = {
        blocks[i].offset,
        blocks[i].metadata_length,
        blocks[i].body_length,
    };
    flatcc_RecordBatch_blocks[i] = block;
  }
  FLATCC_RETURN_UNLESS_0(Footer_recordBatches_end(builder), error);

  const struct ArrowIpcFileBlock* dict_blocks =
      (struct ArrowIpcFileBlock*)footer->dictionary_blocks.data;
  int64_t n_dict_blocks =
      footer->dictionary_blocks.size_bytes / sizeof(struct ArrowIpcFileBlock);

  FLATCC_RETURN_UNLESS_0(Footer_dictionaries_start(builder), error);
  struct ns(Block)* flatcc_dict_blocks =
      ns(Footer_dictionaries_extend(builder, n_dict_blocks));
  FLATCC_RETURN_IF_NULL(flatcc_dict_blocks, error);
  for (int64_t i = 0; i < n_dict_blocks; i++) {
    struct ns(Block) block = {
        dict_blocks[i].offset,
        dict_blocks[i].metadata_length,
        dict_blocks[i].body_length,
    };
    flatcc_dict_blocks[i] = block;
  }
  FLATCC_RETURN_UNLESS_0(Footer_dictionaries_end(builder), error);

  FLATCC_RETURN_IF_NULL(ns(Footer_end_as_root(builder)), error);
  return NANOARROW_OK;
}
