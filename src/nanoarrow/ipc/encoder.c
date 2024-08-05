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
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "flatcc/flatcc_builder.h"
#include "nanoarrow/ipc/flatcc_generated.h"
#include "nanoarrow/nanoarrow.h"
#include "nanoarrow/nanoarrow_ipc.h"

#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(org_apache_arrow_flatbuf, x)

#define FLATCC_RETURN_UNLESS_0(x) \
  if (ns(x) != 0) return ENOMEM;

struct ArrowIpcEncoderPrivate {
  flatcc_builder_t builder;
  struct ArrowBuffer buffers;
  struct ArrowBuffer nodes;
};

static int32_t ArrowInt32ToLe(int32_t i) {
  if (ArrowIpcSystemEndianness() == NANOARROW_IPC_ENDIANNESS_BIG) {
    return bswap32(i);
  }
  return i;
}

ArrowErrorCode ArrowIpcEncoderInit(struct ArrowIpcEncoder* encoder) {
  NANOARROW_DCHECK(encoder != NULL);
  memset(encoder, 0, sizeof(struct ArrowIpcEncoder));
  encoder->encode_buffer = NULL;
  encoder->encode_buffer_state = NULL;
  encoder->codec = NANOARROW_IPC_COMPRESSION_TYPE_NONE;
  encoder->private_data = ArrowMalloc(sizeof(struct ArrowIpcEncoderPrivate));
  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;
  if (flatcc_builder_init(&private->builder) == -1) {
    ArrowFree(private);
    return ESPIPE;
  }
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
    memset(encoder, 0, sizeof(struct ArrowIpcEncoder));
  }
}

ArrowErrorCode ArrowIpcEncoderFinalizeBuffer(struct ArrowIpcEncoder* encoder,
                                             char encapsulate, struct ArrowBuffer* out) {
  NANOARROW_DCHECK(encoder != NULL && encoder->private_data != NULL && out != NULL);
  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;

  int64_t size = (int64_t)flatcc_builder_get_buffer_size(&private->builder);
  int32_t header[] = {-1, ArrowInt32ToLe((int32_t)size)};

  if (size == 0) {
    // Finalizing an empty flatcc_builder_t triggers an assertion
    return encapsulate ? ArrowBufferAppend(out, &header, sizeof(header)) : NANOARROW_OK;
  }

  const void* data = flatcc_builder_get_direct_buffer(&private->builder, NULL);
  if (data == NULL) {
    return ENOMEM;
  }

  int64_t i = out->size_bytes;
  if (encapsulate) {
    int64_t encapsulated_size =
        _ArrowRoundUpToMultipleOf8(sizeof(int32_t) + sizeof(int32_t) + size);
    NANOARROW_RETURN_NOT_OK(
        ArrowBufferResize(out, out->size_bytes + encapsulated_size, 0));
  } else {
    NANOARROW_RETURN_NOT_OK(ArrowBufferResize(out, out->size_bytes + size, 0));
  }

  if (encapsulate) {
    memcpy(out->data + i, &header, sizeof(header));
    i += sizeof(header);
  }

  memcpy(out->data + i, data, size);
  i += size;

  // zero padding bytes, if any
  memset(out->data + i, 0, out->size_bytes - i);

  // don't deallocate yet, just wipe the builder's current Message
  flatcc_builder_reset(&private->builder);
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowIpcEncodeFieldType(flatcc_builder_t* builder,
                                              const struct ArrowSchemaView* schema_view,
                                              struct ArrowError* error) {
  switch (schema_view->type) {
    case NANOARROW_TYPE_NA:
      FLATCC_RETURN_UNLESS_0(Field_type_Null_create(builder));
      return NANOARROW_OK;

    case NANOARROW_TYPE_BOOL:
      FLATCC_RETURN_UNLESS_0(Field_type_Bool_create(builder));
      return NANOARROW_OK;

    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT8:
      FLATCC_RETURN_UNLESS_0(
          Field_type_Int_create(builder, 8, schema_view->type == NANOARROW_TYPE_INT8));
      return NANOARROW_OK;

    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT16:
      FLATCC_RETURN_UNLESS_0(
          Field_type_Int_create(builder, 16, schema_view->type == NANOARROW_TYPE_INT16));
      return NANOARROW_OK;

    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT32:
      FLATCC_RETURN_UNLESS_0(
          Field_type_Int_create(builder, 32, schema_view->type == NANOARROW_TYPE_INT32));
      return NANOARROW_OK;

    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_INT64:
      FLATCC_RETURN_UNLESS_0(
          Field_type_Int_create(builder, 64, schema_view->type == NANOARROW_TYPE_INT64));
      return NANOARROW_OK;

    case NANOARROW_TYPE_HALF_FLOAT:
      FLATCC_RETURN_UNLESS_0(
          Field_type_FloatingPoint_create(builder, ns(Precision_HALF)));
      return NANOARROW_OK;

    case NANOARROW_TYPE_FLOAT:
      FLATCC_RETURN_UNLESS_0(
          Field_type_FloatingPoint_create(builder, ns(Precision_SINGLE)));
      return NANOARROW_OK;

    case NANOARROW_TYPE_DOUBLE:
      FLATCC_RETURN_UNLESS_0(
          Field_type_FloatingPoint_create(builder, ns(Precision_DOUBLE)));
      return NANOARROW_OK;

    case NANOARROW_TYPE_DECIMAL128:
    case NANOARROW_TYPE_DECIMAL256:
      FLATCC_RETURN_UNLESS_0(Field_type_Decimal_create(
          builder, schema_view->decimal_precision, schema_view->decimal_scale,
          schema_view->decimal_bitwidth));
      return NANOARROW_OK;

    case NANOARROW_TYPE_STRING:
      FLATCC_RETURN_UNLESS_0(Field_type_Utf8_create(builder));
      return NANOARROW_OK;

    case NANOARROW_TYPE_LARGE_STRING:
      FLATCC_RETURN_UNLESS_0(Field_type_LargeUtf8_create(builder));
      return NANOARROW_OK;

    case NANOARROW_TYPE_BINARY:
      FLATCC_RETURN_UNLESS_0(Field_type_Binary_create(builder));
      return NANOARROW_OK;

    case NANOARROW_TYPE_LARGE_BINARY:
      FLATCC_RETURN_UNLESS_0(Field_type_LargeBinary_create(builder));
      return NANOARROW_OK;

    case NANOARROW_TYPE_DATE32:
      FLATCC_RETURN_UNLESS_0(Field_type_Date_create(builder, ns(DateUnit_DAY)));
      return NANOARROW_OK;

    case NANOARROW_TYPE_DATE64:
      FLATCC_RETURN_UNLESS_0(Field_type_Date_create(builder, ns(DateUnit_MILLISECOND)));
      return NANOARROW_OK;

    case NANOARROW_TYPE_INTERVAL_MONTHS:
      FLATCC_RETURN_UNLESS_0(
          Field_type_Interval_create(builder, ns(IntervalUnit_YEAR_MONTH)));
      return NANOARROW_OK;

    case NANOARROW_TYPE_INTERVAL_DAY_TIME:
      FLATCC_RETURN_UNLESS_0(
          Field_type_Interval_create(builder, ns(IntervalUnit_DAY_TIME)));
      return NANOARROW_OK;

    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO:
      FLATCC_RETURN_UNLESS_0(
          Field_type_Interval_create(builder, ns(IntervalUnit_MONTH_DAY_NANO)));
      return NANOARROW_OK;

    case NANOARROW_TYPE_TIMESTAMP:
      FLATCC_RETURN_UNLESS_0(Field_type_Timestamp_start(builder));
      FLATCC_RETURN_UNLESS_0(Timestamp_unit_add(builder, schema_view->time_unit));
      FLATCC_RETURN_UNLESS_0(
          Timestamp_timezone_create_str(builder, schema_view->timezone));
      FLATCC_RETURN_UNLESS_0(Field_type_Timestamp_end(builder));
      return NANOARROW_OK;

    case NANOARROW_TYPE_TIME32:
      FLATCC_RETURN_UNLESS_0(Field_type_Time_create(builder, schema_view->time_unit, 32));
      return NANOARROW_OK;

    case NANOARROW_TYPE_TIME64:
      FLATCC_RETURN_UNLESS_0(Field_type_Time_create(builder, schema_view->time_unit, 64));
      return NANOARROW_OK;

    case NANOARROW_TYPE_DURATION:
      FLATCC_RETURN_UNLESS_0(Field_type_Duration_create(builder, schema_view->time_unit));
      return NANOARROW_OK;

    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
      FLATCC_RETURN_UNLESS_0(
          Field_type_FixedSizeBinary_create(builder, schema_view->fixed_size));
      return NANOARROW_OK;

    case NANOARROW_TYPE_LIST:
      FLATCC_RETURN_UNLESS_0(Field_type_List_create(builder));
      return NANOARROW_OK;

    case NANOARROW_TYPE_LARGE_LIST:
      FLATCC_RETURN_UNLESS_0(Field_type_LargeList_create(builder));
      return NANOARROW_OK;

    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      FLATCC_RETURN_UNLESS_0(
          Field_type_FixedSizeList_create(builder, schema_view->fixed_size));
      return NANOARROW_OK;

    case NANOARROW_TYPE_RUN_END_ENCODED:
      FLATCC_RETURN_UNLESS_0(Field_type_RunEndEncoded_create(builder));
      return NANOARROW_OK;

    case NANOARROW_TYPE_STRUCT:
      FLATCC_RETURN_UNLESS_0(Field_type_Struct__create(builder));
      return NANOARROW_OK;

    case NANOARROW_TYPE_SPARSE_UNION:
    case NANOARROW_TYPE_DENSE_UNION: {
      FLATCC_RETURN_UNLESS_0(Field_type_Union_start(builder));

      FLATCC_RETURN_UNLESS_0(
          Union_mode_add(builder, schema_view->type == NANOARROW_TYPE_DENSE_UNION));
      if (schema_view->union_type_ids) {
        int8_t type_ids[128];
        int n = _ArrowParseUnionTypeIds(schema_view->union_type_ids, type_ids);
        if (n != 0) {
          FLATCC_RETURN_UNLESS_0(Union_typeIds_start(builder));
          int32_t* type_ids_32 = (int32_t*)ns(Union_typeIds_extend(builder, n));
          if (!type_ids_32) {
            return ENOMEM;
          }

          for (int i = 0; i < n; ++i) {
            type_ids_32[i] = type_ids[i];
          }
          FLATCC_RETURN_UNLESS_0(Union_typeIds_end(builder));
        }
      }

      FLATCC_RETURN_UNLESS_0(Field_type_Union_end(builder));
      return NANOARROW_OK;
    }

    case NANOARROW_TYPE_MAP:
      FLATCC_RETURN_UNLESS_0(Field_type_Map_create(
          builder, schema_view->schema->flags & ARROW_FLAG_MAP_KEYS_SORTED));
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
  NANOARROW_RETURN_NOT_OK(ArrowMetadataReaderInit(&metadata, schema->metadata));
  while (metadata.remaining_keys > 0) {
    struct ArrowStringView key, value;
    NANOARROW_RETURN_NOT_OK(ArrowMetadataReaderRead(&metadata, &key, &value));
    if (push_start(builder) != 0) {
      return ENOMEM;
    }
    FLATCC_RETURN_UNLESS_0(KeyValue_key_create_strn(builder, key.data, key.size_bytes));
    FLATCC_RETURN_UNLESS_0(
        KeyValue_value_create_strn(builder, value.data, value.size_bytes));
    if (!push_end(builder)) {
      return ENOMEM;
    }
  }
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowIpcEncodeFields(flatcc_builder_t* builder,
                                           const struct ArrowSchema* schema,
                                           int (*push_start)(flatcc_builder_t*),
                                           ns(Field_ref_t) *
                                               (*push_end)(flatcc_builder_t*),
                                           struct ArrowError* error) {
  for (int i = 0; i < schema->n_children; ++i) {
    if (push_start(builder) != 0) {
      return ENOMEM;
    }
    NANOARROW_RETURN_NOT_OK(ArrowIpcEncodeField(builder, schema->children[i], error));
    if (!push_end(builder)) {
      return ENOMEM;
    }
  }
  return NANOARROW_OK;
}

static ArrowErrorCode ArrowIpcEncodeField(flatcc_builder_t* builder,
                                          const struct ArrowSchema* schema,
                                          struct ArrowError* error) {
  FLATCC_RETURN_UNLESS_0(Field_name_create_str(builder, schema->name));
  FLATCC_RETURN_UNLESS_0(
      Field_nullable_add(builder, schema->flags & ARROW_FLAG_NULLABLE));

  struct ArrowSchemaView schema_view;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&schema_view, schema, error));
  NANOARROW_RETURN_NOT_OK(ArrowIpcEncodeFieldType(builder, &schema_view, error));

  if (schema->n_children != 0) {
    FLATCC_RETURN_UNLESS_0(Field_children_start(builder));
    NANOARROW_RETURN_NOT_OK(ArrowIpcEncodeFields(builder, schema,
                                                 &ns(Field_children_push_start),
                                                 &ns(Field_children_push_end), error));
    FLATCC_RETURN_UNLESS_0(Field_children_end(builder));
  }

  if (schema->metadata) {
    FLATCC_RETURN_UNLESS_0(Field_custom_metadata_start(builder));
    NANOARROW_RETURN_NOT_OK(
        ArrowIpcEncodeMetadata(builder, schema, &ns(Field_custom_metadata_push_start),
                               &ns(Field_custom_metadata_push_end), error));
    FLATCC_RETURN_UNLESS_0(Field_custom_metadata_end(builder));
  }
  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcEncoderEncodeSchema(struct ArrowIpcEncoder* encoder,
                                           const struct ArrowSchema* schema,
                                           struct ArrowError* error) {
  NANOARROW_DCHECK(encoder != NULL && encoder->private_data != NULL && schema != NULL);

  struct ArrowIpcEncoderPrivate* private =
      (struct ArrowIpcEncoderPrivate*)encoder->private_data;

  flatcc_builder_t* builder = &private->builder;

  FLATCC_RETURN_UNLESS_0(Message_start_as_root(builder));
  FLATCC_RETURN_UNLESS_0(Message_version_add(builder, ns(MetadataVersion_V5)));

  FLATCC_RETURN_UNLESS_0(Message_header_Schema_start(builder));

  if (ArrowIpcSystemEndianness() == NANOARROW_IPC_ENDIANNESS_LITTLE) {
    FLATCC_RETURN_UNLESS_0(Schema_endianness_add(builder, ns(Endianness_Little)));
  } else {
    FLATCC_RETURN_UNLESS_0(Schema_endianness_add(builder, ns(Endianness_Big)));
  }

  FLATCC_RETURN_UNLESS_0(Schema_fields_start(builder));
  NANOARROW_RETURN_NOT_OK(ArrowIpcEncodeFields(builder, schema,
                                               &ns(Schema_fields_push_start),
                                               &ns(Schema_fields_push_end), error));
  FLATCC_RETURN_UNLESS_0(Schema_fields_end(builder));

  if (schema->metadata) {
    FLATCC_RETURN_UNLESS_0(Schema_custom_metadata_start(builder));
    NANOARROW_RETURN_NOT_OK(
        ArrowIpcEncodeMetadata(builder, schema, &ns(Schema_custom_metadata_push_start),
                               &ns(Schema_custom_metadata_push_end), error));
    FLATCC_RETURN_UNLESS_0(Schema_custom_metadata_end(builder));
  }

  FLATCC_RETURN_UNLESS_0(Schema_features_start(builder));
  ns(Feature_enum_t)* features = ns(Schema_features_extend(builder, 1));
  if (!features) {
    return ENOMEM;
  }
  features[0] = ns(Feature_COMPRESSED_BODY);
  FLATCC_RETURN_UNLESS_0(Schema_features_end(builder));

  FLATCC_RETURN_UNLESS_0(Message_header_Schema_end(builder));

  FLATCC_RETURN_UNLESS_0(Message_bodyLength_add(builder, 0));
  return ns(Message_end_as_root(builder)) ? NANOARROW_OK : ENOMEM;
}
