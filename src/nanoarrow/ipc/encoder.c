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
    ArrowFree(private);
  }
  memset(encoder, 0, sizeof(struct ArrowIpcEncoder));
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

static ArrowErrorCode ArrowIpcEncodeField(flatcc_builder_t* builder,
                                          const struct ArrowSchema* schema,
                                          struct ArrowError* error);

static ArrowErrorCode ArrowIpcEncodeMetadata(flatcc_builder_t* builder,
                                             const struct ArrowSchema* schema,
                                             int (*push_start)(flatcc_builder_t*),
                                             ns(KeyValue_ref_t) *
                                                 (*push_end)(flatcc_builder_t*),
                                             struct ArrowError* error) {
  struct ArrowMetadataReader metadata;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowMetadataReaderInit(&metadata, schema->metadata),
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

static ArrowErrorCode ArrowIpcEncodeFields(flatcc_builder_t* builder,
                                           const struct ArrowSchema* schema,
                                           int (*push_start)(flatcc_builder_t*),
                                           ns(Field_ref_t) *
                                               (*push_end)(flatcc_builder_t*),
                                           struct ArrowError* error) {
  for (int i = 0; i < schema->n_children; i++) {
    FLATCC_RETURN_UNLESS_0_NO_NS(push_start(builder), error);
    NANOARROW_RETURN_NOT_OK(ArrowIpcEncodeField(builder, schema->children[i], error));
    FLATCC_RETURN_IF_NULL(push_end(builder), error);
  }
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowIpcEncodeField(flatcc_builder_t* builder,
                                          const struct ArrowSchema* schema,
                                          struct ArrowError* error) {
  FLATCC_RETURN_UNLESS_0(Field_name_create_str(builder, schema->name), error);
  FLATCC_RETURN_UNLESS_0(
      Field_nullable_add(builder, (schema->flags & ARROW_FLAG_NULLABLE) != 0), error);

  struct ArrowSchemaView schema_view;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&schema_view, schema, error));
  NANOARROW_RETURN_NOT_OK(ArrowIpcEncodeFieldType(builder, &schema_view, error));

  if (schema->n_children != 0) {
    FLATCC_RETURN_UNLESS_0(Field_children_start(builder), error);
    NANOARROW_RETURN_NOT_OK(ArrowIpcEncodeFields(builder, schema,
                                                 &ns(Field_children_push_start),
                                                 &ns(Field_children_push_end), error));
    FLATCC_RETURN_UNLESS_0(Field_children_end(builder), error);
  }

  if (schema->metadata) {
    FLATCC_RETURN_UNLESS_0(Field_custom_metadata_start(builder), error);
    NANOARROW_RETURN_NOT_OK(
        ArrowIpcEncodeMetadata(builder, schema, &ns(Field_custom_metadata_push_start),
                               &ns(Field_custom_metadata_push_end), error));
    FLATCC_RETURN_UNLESS_0(Field_custom_metadata_end(builder), error);
  }
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowIpcEncodeSchema(flatcc_builder_t* builder,
                                           const struct ArrowSchema* schema,
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
  NANOARROW_RETURN_NOT_OK(ArrowIpcEncodeFields(builder, schema,
                                               &ns(Schema_fields_push_start),
                                               &ns(Schema_fields_push_end), error));
  FLATCC_RETURN_UNLESS_0(Schema_fields_end(builder), error);

  FLATCC_RETURN_UNLESS_0(Schema_custom_metadata_start(builder), error);
  if (schema->metadata) {
    NANOARROW_RETURN_NOT_OK(
        ArrowIpcEncodeMetadata(builder, schema, &ns(Schema_custom_metadata_push_start),
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
  NANOARROW_RETURN_NOT_OK(ArrowIpcEncodeSchema(builder, schema, error));
  FLATCC_RETURN_UNLESS_0(Message_header_Schema_end(builder), error);

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

static ArrowErrorCode ArrowIpcEncoderBuildContiguousBodyBufferCallback(
    struct ArrowBufferView buffer_view, struct ArrowIpcEncoder* encoder,
    struct ArrowIpcBufferEncoder* buffer_encoder, int64_t* offset, int64_t* length,
    struct ArrowError* error) {
  NANOARROW_UNUSED(encoder);

  struct ArrowBuffer* body_buffer =
      (struct ArrowBuffer*)buffer_encoder->encode_buffer_state;

  int64_t old_size = body_buffer->size_bytes;
  int64_t buffer_begin = _ArrowRoundUpToMultipleOf8(old_size);
  int64_t buffer_end = buffer_begin + buffer_view.size_bytes;
  int64_t new_size = _ArrowRoundUpToMultipleOf8(buffer_end);

  // reserve all the memory we'll need now
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBufferReserve(body_buffer, new_size - old_size),
                                     error);

  // zero padding up to the start of the buffer
  NANOARROW_ASSERT_OK(ArrowBufferAppendFill(body_buffer, 0, buffer_begin - old_size));

  // store offset and length of the buffer
  *offset = buffer_begin;
  *length = buffer_view.size_bytes;

  NANOARROW_ASSERT_OK(
      ArrowBufferAppend(body_buffer, buffer_view.data.data, buffer_view.size_bytes));

  // zero padding after writing the buffer
  NANOARROW_DCHECK(body_buffer->size_bytes == buffer_end);
  NANOARROW_ASSERT_OK(ArrowBufferAppendFill(body_buffer, 0, new_size - buffer_end));

  buffer_encoder->body_length = body_buffer->size_bytes;
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

void ArrowIpcFooterInit(struct ArrowIpcFooter* footer) {
  footer->schema.release = NULL;
  ArrowBufferInit(&footer->record_batch_blocks);
}

void ArrowIpcFooterReset(struct ArrowIpcFooter* footer) {
  if (footer->schema.release != NULL) {
    ArrowSchemaRelease(&footer->schema);
  }
  ArrowBufferReset(&footer->record_batch_blocks);
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
  NANOARROW_RETURN_NOT_OK(ArrowIpcEncodeSchema(builder, &footer->schema, error));
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

  FLATCC_RETURN_IF_NULL(ns(Footer_end_as_root(builder)), error);
  return NANOARROW_OK;
}
