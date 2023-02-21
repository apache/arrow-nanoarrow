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
#include <string.h>

#include "nanoarrow.h"
#include "nanoarrow_ipc.h"
#include "nanoarrow_ipc_flatcc_generated.h"

ArrowErrorCode ArrowIpcCheckRuntime(struct ArrowError* error) {
  const char* nanoarrow_runtime_version = ArrowNanoarrowVersion();
  const char* nanoarrow_ipc_build_time_version = NANOARROW_VERSION;

  if (strcmp(nanoarrow_runtime_version, nanoarrow_ipc_build_time_version) != 0) {
    ArrowErrorSet(error, "Expected nanoarrow runtime version '%s' but found version '%s'",
                  nanoarrow_ipc_build_time_version, nanoarrow_runtime_version);
    return EINVAL;
  }

  return NANOARROW_OK;
}

void ArrowIpcReaderInit(struct ArrowIpcReader* reader) {
  memset(reader, 0, sizeof(struct ArrowIpcReader));
}

void ArrowIpcReaderReset(struct ArrowIpcReader* reader) {
  if (reader->schema.release != NULL) {
    reader->schema.release(&reader->schema);
  }

  ArrowIpcReaderInit(reader);
}

static inline uint32_t ArrowIpcReadUint32LE(struct ArrowBufferView* data) {
  uint32_t value;
  memcpy(&value, data->data.as_uint8, sizeof(uint32_t));
  // TODO: bswap32() if big endian
  data->data.as_uint8 += sizeof(uint32_t);
  data->size_bytes -= sizeof(uint32_t);
  return value;
}

static inline int32_t ArrowIpcReadInt32LE(struct ArrowBufferView* data) {
  int32_t value;
  memcpy(&value, data->data.as_uint8, sizeof(int32_t));
  // TODO: bswap32() if big endian
  data->data.as_uint8 += sizeof(int32_t);
  data->size_bytes -= sizeof(int32_t);
  return value;
}

#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(org_apache_arrow_flatbuf, x)

static int ArrowIpcReaderSetTypeSimple(struct ArrowSchema* schema, int nanoarrow_type,
                                       struct ArrowError* error) {
  int result = ArrowSchemaSetType(schema, nanoarrow_type);
  if (result != NANOARROW_OK) {
    ArrowErrorSet(error, "ArrowSchemaSetType() failed for type %s",
                  ArrowTypeString(nanoarrow_type));
    return result;
  }

  return NANOARROW_OK;
}

static int ArrowIpcReaderSetTypeInt(struct ArrowSchema* schema,
                                    flatbuffers_generic_t type_generic,
                                    struct ArrowError* error) {
  ns(Int_table_t) type = (ns(Int_table_t))type_generic;

  int is_signed = ns(Int_is_signed_get(type));
  int bitwidth = ns(Int_bitWidth_get(type));
  int nanoarrow_type = NANOARROW_TYPE_UNINITIALIZED;

  if (is_signed) {
    switch (bitwidth) {
      case 8:
        nanoarrow_type = NANOARROW_TYPE_INT8;
        break;
      case 16:
        nanoarrow_type = NANOARROW_TYPE_INT16;
        break;
      case 32:
        nanoarrow_type = NANOARROW_TYPE_INT32;
        break;
      case 64:
        nanoarrow_type = NANOARROW_TYPE_INT64;
        break;
      default:
        ArrowErrorSet(error,
                      "Expected signed int bitwidth of 8, 16, 32, or 64 but got %d",
                      (int)bitwidth);
        return EINVAL;
    }
  } else {
    switch (bitwidth) {
      case 8:
        nanoarrow_type = NANOARROW_TYPE_UINT8;
        break;
      case 16:
        nanoarrow_type = NANOARROW_TYPE_UINT16;
        break;
      case 32:
        nanoarrow_type = NANOARROW_TYPE_UINT32;
        break;
      case 64:
        nanoarrow_type = NANOARROW_TYPE_UINT64;
        break;
      default:
        ArrowErrorSet(error,
                      "Expected unsigned int bitwidth of 8, 16, 32, or 64 but got %d",
                      (int)bitwidth);
        return EINVAL;
    }
  }

  return ArrowIpcReaderSetTypeSimple(schema, nanoarrow_type, error);
}

static int ArrowIpcReaderSetTypeFloatingPoint(struct ArrowSchema* schema,
                                              flatbuffers_generic_t type_generic,
                                              struct ArrowError* error) {
  ns(FloatingPoint_table_t) type = (ns(FloatingPoint_table_t))type_generic;
  return ENOTSUP;
}

static int ArrowIpcReaderSetTypeDecimal(struct ArrowSchema* schema,
                                        flatbuffers_generic_t type_generic,
                                        struct ArrowError* error) {
  ns(Decimal_table_t) type = (ns(Decimal_table_t))type_generic;
  return ENOTSUP;
}

static int ArrowIpcReaderSetTypeFixedSizeBinary(struct ArrowSchema* schema,
                                                flatbuffers_generic_t type_generic,
                                                struct ArrowError* error) {
  ns(FixedSizeBinary_table_t) type = (ns(FixedSizeBinary_table_t))type_generic;
  return ENOTSUP;
}

static int ArrowIpcReaderSetTypeDate(struct ArrowSchema* schema,
                                     flatbuffers_generic_t type_generic,
                                     struct ArrowError* error) {
  ns(Date_table_t) type = (ns(Date_table_t))type_generic;
  return ENOTSUP;
}

static int ArrowIpcReaderSetTypeTime(struct ArrowSchema* schema,
                                     flatbuffers_generic_t type_generic,
                                     struct ArrowError* error) {
  ns(Time_table_t) type = (ns(Time_table_t))type_generic;
  return ENOTSUP;
}

static int ArrowIpcReaderSetTypeTimestamp(struct ArrowSchema* schema,
                                          flatbuffers_generic_t type_generic,
                                          struct ArrowError* error) {
  ns(Timestamp_table_t) type = (ns(Timestamp_table_t))type_generic;
  return ENOTSUP;
}

static int ArrowIpcReaderSetTypeDuration(struct ArrowSchema* schema,
                                         flatbuffers_generic_t type_generic,
                                         struct ArrowError* error) {
  ns(Duration_table_t) type = (ns(Duration_table_t))type_generic;
  return ENOTSUP;
}

static int ArrowIpcReaderSetTypeInterval(struct ArrowSchema* schema,
                                         flatbuffers_generic_t type_generic,
                                         struct ArrowError* error) {
  ns(Interval_table_t) type = (ns(Interval_table_t))type_generic;
  return ENOTSUP;
}

static int ArrowIpcReaderSetField(struct ArrowSchema* schema, ns(Field_table_t) field,
                                  struct ArrowError* error) {
  int result;
  if (ns(Field_name_is_present(field))) {
    result = ArrowSchemaSetName(schema, ns(Field_name_get(field)));
  } else {
    result = ArrowSchemaSetName(schema, "");
  }

  if (result != NANOARROW_OK) {
    ArrowErrorSet(error, "ArrowSchemaSetName() failed");
    return result;
  }

  if (ns(Field_nullable_get(field))) {
    schema->flags |= ARROW_FLAG_NULLABLE;
  }

  int type_type = ns(Field_type_type(field));
  switch (type_type) {
    case ns(Type_Null):
      return ArrowIpcReaderSetTypeSimple(schema, NANOARROW_TYPE_NA, error);
    case ns(Type_Bool):
      return ArrowIpcReaderSetTypeSimple(schema, NANOARROW_TYPE_BOOL, error);
    case ns(Type_Int):
      return ArrowIpcReaderSetTypeInt(schema, ns(Field_type_get(field)), error);
    case ns(Type_FloatingPoint):
      return ArrowIpcReaderSetTypeFloatingPoint(schema, ns(Field_type_get(field)), error);
    case ns(Type_Decimal):
      return ArrowIpcReaderSetTypeDecimal(schema, ns(Field_type_get(field)), error);
    case ns(Type_Binary):
      return ArrowIpcReaderSetTypeSimple(schema, NANOARROW_TYPE_BINARY, error);
    case ns(Type_LargeBinary):
      return ArrowIpcReaderSetTypeSimple(schema, NANOARROW_TYPE_LARGE_BINARY, error);
    case ns(Type_FixedSizeBinary):
      return ArrowIpcReaderSetTypeFixedSizeBinary(schema, ns(Field_type_get(field)),
                                                  error);
    case ns(Type_Utf8):
      return ArrowIpcReaderSetTypeSimple(schema, NANOARROW_TYPE_STRING, error);
    case ns(Type_LargeUtf8):
      return ArrowIpcReaderSetTypeSimple(schema, NANOARROW_TYPE_LARGE_STRING, error);
    case ns(Type_Date):
      return ArrowIpcReaderSetTypeDate(schema, ns(Field_type_get(field)), error);
    case ns(Type_Time):
      return ArrowIpcReaderSetTypeTime(schema, ns(Field_type_get(field)), error);
    case ns(Type_Timestamp):
      return ArrowIpcReaderSetTypeTimestamp(schema, ns(Field_type_get(field)), error);
    case ns(Type_Duration):
      return ArrowIpcReaderSetTypeDuration(schema, ns(Field_type_get(field)), error);
    case ns(Type_Interval):
      return ArrowIpcReaderSetTypeInterval(schema, ns(Field_type_get(field)), error);
    case ns(Type_Struct_):
      return ENOTSUP;
    case ns(Type_List):
      return ENOTSUP;
    case ns(Type_LargeList):
      return ENOTSUP;
    case ns(Type_FixedSizeList):
      return ENOTSUP;
    case ns(Type_Union):
      return ENOTSUP;
    default:
      ArrowErrorSet(error, "Unrecognized Field type with value %d", (int)type_type);
      return EINVAL;
  }

  return NANOARROW_OK;
}

static int ArrowIpcReaderSetChildren(struct ArrowSchema* schema, ns(Field_vec_t) fields,
                                     struct ArrowError* error) {
  int64_t n_fields = ns(Schema_vec_len(fields));

  for (int64_t i = 0; i < n_fields; i++) {
    ns(Field_table_t) field = ns(Field_vec_at(fields, i));
    NANOARROW_RETURN_NOT_OK(ArrowIpcReaderSetField(schema->children[i], field, error));
  }

  return NANOARROW_OK;
}

static int ArrowIpcReaderDecodeSchema(struct ArrowIpcReader* reader,
                                      flatbuffers_generic_t message_header,
                                      struct ArrowError* error) {
  ns(Schema_table_t) schema = (ns(Schema_table_t))message_header;
  int endianness = ns(Schema_endianness(schema));
  switch (endianness) {
    case ns(Endianness_Little):
      reader->endianness = NANOARROW_IPC_ENDIANNESS_LITTLE;
      break;
    case ns(Endianness_Big):
      reader->endianness = NANOARROW_IPC_ENDIANNESS_BIG;
      break;
    default:
      ArrowErrorSet(error,
                    "Expected Schema endianness of 0 (little) or 1 (big) but got %d",
                    (int)endianness);
      return EINVAL;
  }

  ns(Feature_vec_t) features = ns(Schema_features(schema));
  int64_t n_features = ns(Feature_vec_len(features));
  reader->features = 0;

  for (int64_t i = 0; i < n_features; i++) {
    ns(Feature_enum_t) feature = ns(Feature_vec_at(features, i));
    switch (feature) {
      case ns(Feature_COMPRESSED_BODY):
        reader->features |= NANOARROW_IPC_FEATURE_COMPRESSED_BODY;
        break;
      case ns(Feature_DICTIONARY_REPLACEMENT):
        reader->features |= NANOARROW_IPC_FEATURE_DICTIONARY_REPLACEMENT;
        break;
      default:
        ArrowErrorSet(error, "Unrecognized Schema feature with value %d", (int)feature);
        return EINVAL;
    }
  }

  ns(Field_vec_t) fields = ns(Schema_fields(schema));
  int64_t n_fields = ns(Schema_vec_len(fields));
  if (reader->schema.release != NULL) {
    reader->schema.release(&reader->schema);
  }

  ArrowSchemaInit(&reader->schema);
  int result = ArrowSchemaSetTypeStruct(&reader->schema, n_fields);
  if (result != NANOARROW_OK) {
    ArrowErrorSet(error, "Failed to allocate struct schema with %ld children",
                  (long)n_fields);
    return result;
  }

  return ArrowIpcReaderSetChildren(&reader->schema, fields, error);
}

static inline int ArrowIpcReaderCheckHeader(struct ArrowIpcReader* reader,
                                            struct ArrowBufferView* data_mut,
                                            int32_t* message_size_bytes,
                                            struct ArrowError* error) {
  if (data_mut->size_bytes < 8) {
    ArrowErrorSet(error, "Expected data of at least 8 bytes but only %ld bytes remain",
                  (long)data_mut->size_bytes);
    return EINVAL;
  }

  uint32_t continuation = ArrowIpcReadUint32LE(data_mut);
  if (continuation != 0xFFFFFFFF) {
    ArrowErrorSet(error, "Expected 0xFFFFFFFF at start of message but found 0x%08X",
                  (unsigned int)continuation);
    return EINVAL;
  }

  *message_size_bytes = ArrowIpcReadInt32LE(data_mut);
  if ((*message_size_bytes) > data_mut->size_bytes || (*message_size_bytes) < 0) {
    ArrowErrorSet(error,
                  "Expected 0 <= message body size <= %ld bytes but found message "
                  "body size of %ld bytes",
                  (long)data_mut->size_bytes, (long)(*message_size_bytes));
    return EINVAL;
  }

  if (*message_size_bytes == 0) {
    ArrowErrorSet(error, "End of Arrow stream");
    return ENODATA;
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcReaderPeek(struct ArrowIpcReader* reader,
                                  struct ArrowBufferView data, struct ArrowError* error) {
  reader->message_type = NANOARROW_IPC_MESSAGE_TYPE_UNINITIALIZED;
  reader->body_size_bytes = 0;
  NANOARROW_RETURN_NOT_OK(
      ArrowIpcReaderCheckHeader(reader, &data, &reader->header_size_bytes, error));
  reader->header_size_bytes += 2 * sizeof(int32_t);
  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcReaderVerify(struct ArrowIpcReader* reader,
                                    struct ArrowBufferView data,
                                    struct ArrowError* error) {
  reader->message_type = NANOARROW_IPC_MESSAGE_TYPE_UNINITIALIZED;
  reader->body_size_bytes = 0;
  NANOARROW_RETURN_NOT_OK(
      ArrowIpcReaderCheckHeader(reader, &data, &reader->header_size_bytes, error));

  // Run flatbuffers verification
  if (ns(Message_verify_as_root(data.data.as_uint8, reader->header_size_bytes)) !=
      flatcc_verify_ok) {
    ArrowErrorSet(error, "Message flatbuffer verification failed");
    return EINVAL;
  }

  // Read some basic information from the message
  reader->header_size_bytes += 2 * sizeof(int32_t);
  ns(Message_table_t) message = ns(Message_as_root(data.data.as_uint8));
  reader->metadata_version = ns(Message_version(message));
  reader->message_type = ns(Message_header_type(message));
  reader->body_size_bytes = ns(Message_bodyLength(message));

  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcReaderDecode(struct ArrowIpcReader* reader,
                                    struct ArrowBufferView data,
                                    struct ArrowError* error) {
  reader->message_type = NANOARROW_IPC_MESSAGE_TYPE_UNINITIALIZED;
  reader->body_size_bytes = 0;
  NANOARROW_RETURN_NOT_OK(
      ArrowIpcReaderCheckHeader(reader, &data, &reader->header_size_bytes, error));
  reader->header_size_bytes += 2 * sizeof(int32_t);

  ns(Message_table_t) message = ns(Message_as_root(data.data.as_uint8));
  if (!message) {
    return EINVAL;
  }

  // Read some basic information from the message
  reader->metadata_version = ns(Message_version(message));
  reader->message_type = ns(Message_header_type(message));
  reader->body_size_bytes = ns(Message_bodyLength(message));

  switch (reader->metadata_version) {
    case ns(MetadataVersion_V4):
    case ns(MetadataVersion_V5):
      break;
    case ns(MetadataVersion_V1):
    case ns(MetadataVersion_V2):
    case ns(MetadataVersion_V3):
      ArrowErrorSet(error, "Expected metadata version V4 or V5 but found %s",
                    ns(MetadataVersion_name(reader->metadata_version)));
      break;
    default:
      ArrowErrorSet(error, "Unexpected value for Message metadata version (%d)",
                    reader->metadata_version);
      return EINVAL;
  }

  flatbuffers_generic_t message_header = ns(Message_header_get(message));
  switch (reader->message_type) {
    case ns(MessageHeader_Schema):
      NANOARROW_RETURN_NOT_OK(ArrowIpcReaderDecodeSchema(reader, message_header, error));
      break;
    case ns(MessageHeader_DictionaryBatch):
    case ns(MessageHeader_RecordBatch):
    case ns(MessageHeader_Tensor):
    case ns(MessageHeader_SparseTensor):
      ArrowErrorSet(error, "Unsupported message type: '%s'",
                    ns(MessageHeader_type_name(reader->message_type)));
      return ENOTSUP;
    default:
      ArrowErrorSet(error, "Unnown message type: %d", (int)(reader->message_type));
      return EINVAL;
  }

  return NANOARROW_OK;
}
