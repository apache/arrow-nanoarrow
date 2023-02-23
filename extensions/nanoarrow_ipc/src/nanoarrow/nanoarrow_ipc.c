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

static int ArrowIpcReaderSetMetadata(struct ArrowSchema* schema,
                                     ns(KeyValue_vec_t) kv_vec,
                                     struct ArrowError* error) {
  int64_t n_pairs = ns(KeyValue_vec_len(kv_vec));
  if (n_pairs == 0) {
    return NANOARROW_OK;
  }

  if (n_pairs > 2147483647) {
    ArrowErrorSet(error,
                  "Expected between 0 and 2147483647 key/value pairs but found %ld",
                  (long)n_pairs);
    return EINVAL;
  }

  struct ArrowBuffer buf;
  struct ArrowStringView key;
  struct ArrowStringView value;
  ns(KeyValue_table_t) kv;

  int result = ArrowMetadataBuilderInit(&buf, NULL);
  if (result != NANOARROW_OK) {
    ArrowBufferReset(&buf);
    ArrowErrorSet(error, "ArrowMetadataBuilderInit() failed");
    return result;
  }

  for (int64_t i = 0; i < n_pairs; i++) {
    kv = ns(KeyValue_vec_at(kv_vec, i));

    key.data = ns(KeyValue_key(kv));
    key.size_bytes = strlen(key.data);
    value.data = ns(KeyValue_value(kv));
    value.size_bytes = strlen(value.data);

    result = ArrowMetadataBuilderAppend(&buf, key, value);
    if (result != NANOARROW_OK) {
      ArrowBufferReset(&buf);
      ArrowErrorSet(error, "ArrowMetadataBuilderAppend() failed");
      return result;
    }
  }

  result = ArrowSchemaSetMetadata(schema, (const char*)buf.data);
  ArrowBufferReset(&buf);
  if (result != NANOARROW_OK) {
    ArrowErrorSet(error, "ArrowSchemaSetMetadata() failed");
    return result;
  }

  return NANOARROW_OK;
}

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
  int precision = ns(FloatingPoint_precision(type));
  switch (precision) {
    case ns(Precision_HALF):
      return ArrowIpcReaderSetTypeSimple(schema, NANOARROW_TYPE_HALF_FLOAT, error);
    case ns(Precision_SINGLE):
      return ArrowIpcReaderSetTypeSimple(schema, NANOARROW_TYPE_FLOAT, error);
    case ns(Precision_DOUBLE):
      return ArrowIpcReaderSetTypeSimple(schema, NANOARROW_TYPE_DOUBLE, error);
    default:
      ArrowErrorSet(error, "Unexpected FloatingPoint Precision value: %d",
                    (int)precision);
      return EINVAL;
  }
}

static int ArrowIpcReaderSetTypeDecimal(struct ArrowSchema* schema,
                                        flatbuffers_generic_t type_generic,
                                        struct ArrowError* error) {
  ns(Decimal_table_t) type = (ns(Decimal_table_t))type_generic;
  int scale = ns(Decimal_scale(type));
  int precision = ns(Decimal_precision(type));
  int bitwidth = ns(Decimal_bitWidth(type));

  int result;
  switch (bitwidth) {
    case 128:
      result =
          ArrowSchemaSetTypeDecimal(schema, NANOARROW_TYPE_DECIMAL128, precision, scale);
      break;
    case 256:
      result =
          ArrowSchemaSetTypeDecimal(schema, NANOARROW_TYPE_DECIMAL256, precision, scale);
      break;
    default:
      ArrowErrorSet(error, "Unexpected Decimal bitwidth value: %d", (int)bitwidth);
      return EINVAL;
  }

  if (result != NANOARROW_OK) {
    ArrowErrorSet(error, "ArrowSchemaSetTypeDecimal() failed");
    return result;
  }

  return NANOARROW_OK;
}

static int ArrowIpcReaderSetTypeFixedSizeBinary(struct ArrowSchema* schema,
                                                flatbuffers_generic_t type_generic,
                                                struct ArrowError* error) {
  ns(FixedSizeBinary_table_t) type = (ns(FixedSizeBinary_table_t))type_generic;
  int fixed_size = ns(FixedSizeBinary_byteWidth(type));
  return ArrowSchemaSetTypeFixedSize(schema, NANOARROW_TYPE_FIXED_SIZE_BINARY,
                                     fixed_size);
}

static int ArrowIpcReaderSetTypeDate(struct ArrowSchema* schema,
                                     flatbuffers_generic_t type_generic,
                                     struct ArrowError* error) {
  ns(Date_table_t) type = (ns(Date_table_t))type_generic;
  int date_unit = ns(Date_unit(type));
  switch (date_unit) {
    case ns(DateUnit_DAY):
      return ArrowIpcReaderSetTypeSimple(schema, NANOARROW_TYPE_DATE32, error);
    case ns(DateUnit_MILLISECOND):
      return ArrowIpcReaderSetTypeSimple(schema, NANOARROW_TYPE_DATE64, error);
    default:
      ArrowErrorSet(error, "Unexpected Date DateUnit value: %d", (int)date_unit);
      return EINVAL;
  }
}

static int ArrowIpcReaderSetTypeTime(struct ArrowSchema* schema,
                                     flatbuffers_generic_t type_generic,
                                     struct ArrowError* error) {
  ns(Time_table_t) type = (ns(Time_table_t))type_generic;
  int time_unit = ns(Time_unit(type));
  int bitwidth = ns(Time_bitWidth(type));
  int nanoarrow_type;

  switch (time_unit) {
    case ns(TimeUnit_SECOND):
    case ns(TimeUnit_MILLISECOND):
      if (bitwidth != 32) {
        ArrowErrorSet(error, "Expected bitwidth of 32 for Time TimeUnit %s but found %d",
                      ns(TimeUnit_name(time_unit)), bitwidth);
        return EINVAL;
      }

      nanoarrow_type = NANOARROW_TYPE_TIME32;
      break;

    case ns(TimeUnit_MICROSECOND):
    case ns(TimeUnit_NANOSECOND):
      if (bitwidth != 64) {
        ArrowErrorSet(error, "Expected bitwidth of 64 for Time TimeUnit %s but found %d",
                      ns(TimeUnit_name(time_unit)), bitwidth);
        return EINVAL;
      }

      nanoarrow_type = NANOARROW_TYPE_TIME64;
      break;

    default:
      ArrowErrorSet(error, "Unexpected Time TimeUnit value: %d", (int)time_unit);
      return EINVAL;
  }

  int result = ArrowSchemaSetTypeDateTime(schema, nanoarrow_type, time_unit, NULL);
  if (result != NANOARROW_OK) {
    ArrowErrorSet(error, "ArrowSchemaSetTypeDateTime() failed");
    return result;
  }

  return NANOARROW_OK;
}

static int ArrowIpcReaderSetTypeTimestamp(struct ArrowSchema* schema,
                                          flatbuffers_generic_t type_generic,
                                          struct ArrowError* error) {
  ns(Timestamp_table_t) type = (ns(Timestamp_table_t))type_generic;
  int time_unit = ns(Timestamp_unit(type));

  const char* timezone = "";
  if (ns(Timestamp_timezone_is_present(type))) {
    timezone = ns(Timestamp_timezone_get(type));
  }

  int result =
      ArrowSchemaSetTypeDateTime(schema, NANOARROW_TYPE_TIMESTAMP, time_unit, timezone);
  if (result != NANOARROW_OK) {
    ArrowErrorSet(error, "ArrowSchemaSetTypeDateTime() failed");
    return result;
  }

  return NANOARROW_OK;
}

static int ArrowIpcReaderSetTypeDuration(struct ArrowSchema* schema,
                                         flatbuffers_generic_t type_generic,
                                         struct ArrowError* error) {
  ns(Duration_table_t) type = (ns(Duration_table_t))type_generic;
  int time_unit = ns(Duration_unit(type));

  int result =
      ArrowSchemaSetTypeDateTime(schema, NANOARROW_TYPE_DURATION, time_unit, NULL);
  if (result != NANOARROW_OK) {
    ArrowErrorSet(error, "ArrowSchemaSetTypeDateTime() failed");
    return result;
  }

  return NANOARROW_OK;
}

static int ArrowIpcReaderSetTypeInterval(struct ArrowSchema* schema,
                                         flatbuffers_generic_t type_generic,
                                         struct ArrowError* error) {
  ns(Interval_table_t) type = (ns(Interval_table_t))type_generic;
  int interval_unit = ns(Interval_unit(type));

  switch (interval_unit) {
    case ns(IntervalUnit_YEAR_MONTH):
      return ArrowIpcReaderSetTypeSimple(schema, NANOARROW_TYPE_INTERVAL_MONTHS, error);
    case ns(IntervalUnit_DAY_TIME):
      return ArrowIpcReaderSetTypeSimple(schema, NANOARROW_TYPE_INTERVAL_DAY_TIME, error);
    case ns(IntervalUnit_MONTH_DAY_NANO):
      return ArrowIpcReaderSetTypeSimple(schema, NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO,
                                         error);
    default:
      ArrowErrorSet(error, "Unexpected Interval unit value: %d", (int)interval_unit);
      return EINVAL;
  }
}

// We can't quite use nanoarrow's built-in SchemaSet functions for nested types
// because the IPC format allows modifying some of the defaults those functions assume.
// In particular, the allocate + initialize children step is handled outside these
// setters.
static int ArrowIpcReaderSetTypeSimpleNested(struct ArrowSchema* schema,
                                             const char* format,
                                             struct ArrowError* error) {
  int result = ArrowSchemaSetFormat(schema, format);
  if (result != NANOARROW_OK) {
    ArrowErrorSet(error, "ArrowSchemaSetFormat('%s') failed", format);
    return result;
  }

  return NANOARROW_OK;
}

static int ArrowIpcReaderSetTypeFixedSizeList(struct ArrowSchema* schema,
                                              flatbuffers_generic_t type_generic,
                                              struct ArrowError* error) {
  ns(FixedSizeList_table_t) type = (ns(FixedSizeList_table_t))type_generic;
  int32_t fixed_size = ns(FixedSizeList_listSize(type));

  char fixed_size_str[128];
  int n_chars = snprintf(fixed_size_str, 128, "+w:%d", fixed_size);
  fixed_size_str[n_chars] = '\0';
  return ArrowIpcReaderSetTypeSimpleNested(schema, fixed_size_str, error);
}

static int ArrowIpcReaderSetTypeMap(struct ArrowSchema* schema,
                                    flatbuffers_generic_t type_generic,
                                    struct ArrowError* error) {
  ns(Map_table_t) type = (ns(Map_table_t))type_generic;
  NANOARROW_RETURN_NOT_OK(ArrowIpcReaderSetTypeSimpleNested(schema, "+m", error));

  if (ns(Map_keysSorted(type))) {
    schema->flags |= ARROW_FLAG_MAP_KEYS_SORTED;
  } else {
    schema->flags &= ~ARROW_FLAG_MAP_KEYS_SORTED;
  }

  return NANOARROW_OK;
}

static int ArrowIpcReaderSetTypeUnion(struct ArrowSchema* schema,
                                      flatbuffers_generic_t type_generic,
                                      int64_t n_children, struct ArrowError* error) {
  ns(Union_table_t) type = (ns(Union_table_t))type_generic;
  int union_mode = ns(Union_mode(type));

  if (n_children < 0 || n_children > 127) {
    ArrowErrorSet(error,
                  "Expected between 0 and 127 children for Union type but found %ld",
                  (long)n_children);
    return EINVAL;
  }

  // Max valid typeIds size is 127; the longest single ID that could be present here
  // is -INT_MIN (11 chars). With commas and the prefix the max size would be
  // 1527 characters. (Any ids outside the range 0...127 are unlikely to be valid
  // elsewhere but they could in theory be present here).
  char union_types_str[2048];
  memset(union_types_str, 0, sizeof(union_types_str));
  char* format_cursor = union_types_str;
  int format_out_size = sizeof(union_types_str);
  int n_chars = 0;

  const char* format_prefix;
  switch (union_mode) {
    case ns(UnionMode_Sparse):
      n_chars = snprintf(format_cursor, format_out_size, "+us:");
      format_cursor += n_chars;
      format_out_size -= n_chars;
      break;
    case ns(UnionMode_Dense):
      n_chars = snprintf(format_cursor, format_out_size, "+ud:");
      format_cursor += n_chars;
      format_out_size -= n_chars;
      break;
    default:
      ArrowErrorSet(error, "Unexpected Union UnionMode value: %d", (int)union_mode);
      return EINVAL;
  }

  if (ns(Union_typeIds_is_present(type))) {
    flatbuffers_int32_vec_t type_ids = ns(Union_typeIds(type));
    int64_t n_type_ids = flatbuffers_int32_vec_len(type_ids);

    if (n_type_ids != n_children) {
      ArrowErrorSet(
          error,
          "Expected between %ld children for Union type with %ld typeIds but found %ld",
          (long)n_type_ids, (long)n_type_ids, (long)n_children);
      return EINVAL;
    }

    if (n_type_ids > 0) {
      n_chars = snprintf(format_cursor, format_out_size, "%d",
                         flatbuffers_int32_vec_at(type_ids, 0));
      format_cursor += n_chars;
      format_out_size -= n_chars;

      for (int64_t i = 1; i < n_type_ids; i++) {
        n_chars = snprintf(format_cursor, format_out_size, ",%d",
                           (int)flatbuffers_int32_vec_at(type_ids, i));
        format_cursor += n_chars;
        format_out_size -= n_chars;
      }
    }
  } else if (n_children > 0) {
    n_chars = snprintf(format_cursor, format_out_size, "0");
    format_cursor += n_chars;
    format_out_size -= n_chars;

    for (int64_t i = 1; i < n_children; i++) {
      n_chars = snprintf(format_cursor, format_out_size, ",%d", (int)i);
      format_cursor += n_chars;
      format_out_size -= n_chars;
    }
  }

  return ArrowIpcReaderSetTypeSimpleNested(schema, union_types_str, error);
}

static int ArrowIpcReaderSetType(struct ArrowSchema* schema, ns(Field_table_t) field,
                                 int64_t n_children, struct ArrowError* error) {
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
      return ArrowIpcReaderSetTypeSimpleNested(schema, "+s", error);
    case ns(Type_List):
      return ArrowIpcReaderSetTypeSimpleNested(schema, "+l", error);
    case ns(Type_LargeList):
      return ArrowIpcReaderSetTypeSimpleNested(schema, "+L", error);
    case ns(Type_FixedSizeList):
      return ArrowIpcReaderSetTypeFixedSizeList(schema, ns(Field_type_get(field)), error);
    case ns(Type_Map):
      return ArrowIpcReaderSetTypeMap(schema, ns(Field_type_get(field)), error);
    case ns(Type_Union):
      return ArrowIpcReaderSetTypeUnion(schema, ns(Field_type_get(field)), n_children,
                                        error);
    default:
      ArrowErrorSet(error, "Unrecognized Field type with value %d", (int)type_type);
      return EINVAL;
  }
}

static int ArrowIpcReaderSetChildren(struct ArrowSchema* schema, ns(Field_vec_t) fields,
                                     struct ArrowError* error);

static int ArrowIpcReaderSetField(struct ArrowSchema* schema, ns(Field_table_t) field,
                                  struct ArrowError* error) {
  // No dictionary support yet
  if (ns(Field_dictionary_is_present(field))) {
    ArrowErrorSet(error, "Field DictionaryEncoding not supported");
    return ENOTSUP;
  }

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

  // Sets the schema->format and validates type-related inconsistencies
  // that might exist in the flatbuffer
  ns(Field_vec_t) children = ns(Field_children(field));
  int64_t n_children = ns(Field_vec_len(children));

  NANOARROW_RETURN_NOT_OK(ArrowIpcReaderSetType(schema, field, n_children, error));

  // nanoarrow's type setters set the nullable flag by default, so we might
  // have to unset it here.
  if (ns(Field_nullable_get(field))) {
    schema->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->flags &= ~ARROW_FLAG_NULLABLE;
  }

  // Children are defined separately in the flatbuffer, so we allocate, initialize
  // and set them separately as well.
  result = ArrowSchemaAllocateChildren(schema, n_children);
  if (result != NANOARROW_OK) {
    ArrowErrorSet(error, "ArrowSchemaAllocateChildren() failed");
    return result;
  }

  for (int64_t i = 0; i < n_children; i++) {
    ArrowSchemaInit(schema->children[i]);
  }

  NANOARROW_RETURN_NOT_OK(ArrowIpcReaderSetChildren(schema, children, error));
  return ArrowIpcReaderSetMetadata(schema, ns(Field_custom_metadata(field)), error);
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

  NANOARROW_RETURN_NOT_OK(ArrowIpcReaderSetChildren(&reader->schema, fields, error));
  return ArrowIpcReaderSetMetadata(&reader->schema, ns(Schema_custom_metadata(schema)),
                                   error);
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
