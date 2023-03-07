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

ArrowErrorCode ArrowIpcReaderInit(struct ArrowIpcReader* reader) {
  memset(reader, 0, sizeof(struct ArrowIpcReader));
  struct ArrowIpcReaderPrivate* private_data =
      (struct ArrowIpcReaderPrivate*)ArrowMalloc(sizeof(struct ArrowIpcReaderPrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  memset(private_data, 0, sizeof(struct ArrowIpcReaderPrivate));
  reader->private_data = private_data;
  return NANOARROW_OK;
}

void ArrowIpcReaderReset(struct ArrowIpcReader* reader) {
  struct ArrowIpcReaderPrivate* private_data =
      (struct ArrowIpcReaderPrivate*)reader->private_data;
  if (private_data->schema.release != NULL) {
    private_data->schema.release(&private_data->schema);
  }

  ArrowArrayViewReset(&private_data->array_view);

  if (private_data->fields != NULL) {
    ArrowFree(private_data->fields);
    private_data->n_fields = 0;
  }

  ArrowFree(private_data);
  memset(reader, 0, sizeof(struct ArrowIpcReader));
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
  struct ArrowIpcReaderPrivate* private_data =
      (struct ArrowIpcReaderPrivate*)reader->private_data;

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
  reader->feature_flags = 0;

  for (int64_t i = 0; i < n_features; i++) {
    ns(Feature_enum_t) feature = ns(Feature_vec_at(features, i));
    switch (feature) {
      case ns(Feature_COMPRESSED_BODY):
        reader->feature_flags |= NANOARROW_IPC_FEATURE_COMPRESSED_BODY;
        break;
      case ns(Feature_DICTIONARY_REPLACEMENT):
        reader->feature_flags |= NANOARROW_IPC_FEATURE_DICTIONARY_REPLACEMENT;
        break;
      default:
        ArrowErrorSet(error, "Unrecognized Schema feature with value %d", (int)feature);
        return EINVAL;
    }
  }

  ns(Field_vec_t) fields = ns(Schema_fields(schema));
  int64_t n_fields = ns(Schema_vec_len(fields));
  if (private_data->schema.release != NULL) {
    private_data->schema.release(&private_data->schema);
  }

  ArrowSchemaInit(&private_data->schema);
  int result = ArrowSchemaSetTypeStruct(&private_data->schema, n_fields);
  if (result != NANOARROW_OK) {
    ArrowErrorSet(error, "Failed to allocate struct schema with %ld children",
                  (long)n_fields);
    return result;
  }

  NANOARROW_RETURN_NOT_OK(
      ArrowIpcReaderSetChildren(&private_data->schema, fields, error));
  return ArrowIpcReaderSetMetadata(&private_data->schema,
                                   ns(Schema_custom_metadata(schema)), error);
}

static int ArrowIpcReaderDecodeRecordBatch(struct ArrowIpcReader* reader,
                                           flatbuffers_generic_t message_header,
                                           struct ArrowError* error) {
  struct ArrowIpcReaderPrivate* private_data =
      (struct ArrowIpcReaderPrivate*)reader->private_data;

  ns(RecordBatch_table_t) batch = (ns(RecordBatch_table_t))message_header;

  ns(FieldNode_vec_t) fields = ns(RecordBatch_nodes(batch));
  ns(Buffer_vec_t) buffers = ns(RecordBatch_buffers(batch));
  int64_t n_fields = ns(FieldNode_vec_len(fields));
  int64_t n_buffers = ns(Buffer_vec_len(buffers));

  // Check field node and buffer count. We have one more field and buffer
  // because we count the root struct and the flatbuffer message does not.
  if ((n_fields + 1) != private_data->n_fields) {
    ArrowErrorSet(error, "Expected %ld field nodes in message but found %ld",
                  (long)private_data->n_fields - 1, (long)n_fields);
    return EINVAL;
  }

  if ((n_buffers + 1) != private_data->n_buffers) {
    ArrowErrorSet(error, "Expected %ld buffers in message but found %ld",
                  (long)private_data->n_buffers - 1, (long)n_buffers);
    return EINVAL;
  }

  if (ns(RecordBatch_compression_is_present(batch))) {
    ns(BodyCompression_table_t) compression = ns(RecordBatch_compression(batch));
    ns(CompressionType_enum_t) codec = ns(BodyCompression_codec(compression));
    switch (codec) {
      case ns(CompressionType_LZ4_FRAME):
        reader->codec = NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME;
        break;
      case ns(CompressionType_ZSTD):
        reader->codec = NANOARROW_IPC_COMPRESSION_TYPE_ZSTD;
        break;
      default:
        ArrowErrorSet(error, "Unrecognized RecordBatch BodyCompression codec value: %d",
                      (int)codec);
        return EINVAL;
    }
  } else {
    reader->codec = NANOARROW_IPC_COMPRESSION_TYPE_NONE;
  }

  // Copying field node and buffer information is separate so as only to pay for the
  // nodes that are actually accessed.
  return NANOARROW_OK;
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
  struct ArrowIpcReaderPrivate* private_data =
      (struct ArrowIpcReaderPrivate*)reader->private_data;

  reader->message_type = NANOARROW_IPC_MESSAGE_TYPE_UNINITIALIZED;
  reader->body_size_bytes = 0;
  private_data->last_message = NULL;
  NANOARROW_RETURN_NOT_OK(
      ArrowIpcReaderCheckHeader(reader, &data, &reader->header_size_bytes, error));
  reader->header_size_bytes += 2 * sizeof(int32_t);
  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcReaderVerify(struct ArrowIpcReader* reader,
                                    struct ArrowBufferView data,
                                    struct ArrowError* error) {
  struct ArrowIpcReaderPrivate* private_data =
      (struct ArrowIpcReaderPrivate*)reader->private_data;

  reader->message_type = NANOARROW_IPC_MESSAGE_TYPE_UNINITIALIZED;
  reader->body_size_bytes = 0;
  private_data->last_message = NULL;
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

  private_data->last_message = ns(Message_header_get(message));
  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcReaderDecode(struct ArrowIpcReader* reader,
                                    struct ArrowBufferView data,
                                    struct ArrowError* error) {
  struct ArrowIpcReaderPrivate* private_data =
      (struct ArrowIpcReaderPrivate*)reader->private_data;

  reader->message_type = NANOARROW_IPC_MESSAGE_TYPE_UNINITIALIZED;
  reader->body_size_bytes = 0;
  private_data->last_message = NULL;

  NANOARROW_RETURN_NOT_OK(
      ArrowIpcReaderCheckHeader(reader, &data, &reader->header_size_bytes, error));
  reader->header_size_bytes += 2 * sizeof(int32_t);

  ns(Message_table_t) message = ns(Message_as_root(data.data.as_uint8));
  if (!message) {
    return EINVAL;
  }

  // Read some basic information from the message
  int32_t metadata_version = ns(Message_version(message));
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
    case ns(MessageHeader_RecordBatch):
      NANOARROW_RETURN_NOT_OK(
          ArrowIpcReaderDecodeRecordBatch(reader, message_header, error));
      break;
    case ns(MessageHeader_DictionaryBatch):
    case ns(MessageHeader_Tensor):
    case ns(MessageHeader_SparseTensor):
      ArrowErrorSet(error, "Unsupported message type: '%s'",
                    ns(MessageHeader_type_name(reader->message_type)));
      return ENOTSUP;
    default:
      ArrowErrorSet(error, "Unnown message type: %d", (int)(reader->message_type));
      return EINVAL;
  }

  private_data->last_message = message_header;
  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcReaderGetSchema(struct ArrowIpcReader* reader,
                                       struct ArrowSchema* out,
                                       struct ArrowError* error) {
  struct ArrowIpcReaderPrivate* private_data =
      (struct ArrowIpcReaderPrivate*)reader->private_data;

  if (private_data->schema.release == NULL) {
    ArrowErrorSet(error, "reader does not contain a valid schema");
    return EINVAL;
  }

  ArrowSchemaMove(&private_data->schema, out);
  return NANOARROW_OK;
}

static void ArrowIpcReaderCountFields(struct ArrowSchema* schema, int64_t* n_fields) {
  *n_fields += 1;
  for (int64_t i = 0; i < schema->n_children; i++) {
    ArrowIpcReaderCountFields(schema->children[i], n_fields);
  }
}

static void ArrowIpcReaderInitFields(struct ArrowIpcField* fields,
                                     struct ArrowArrayView* view, int64_t* n_fields,
                                     int64_t* n_buffers) {
  struct ArrowIpcField* field = fields + (*n_fields);
  field->array_view = view;
  field->buffer_offset = *n_buffers;

  for (int i = 0; i < 3; i++) {
    *n_buffers += view->layout.buffer_type[i] != NANOARROW_BUFFER_TYPE_NONE;
  }

  *n_fields += 1;

  for (int64_t i = 0; i < view->n_children; i++) {
    ArrowIpcReaderInitFields(fields, view->children[i], n_fields, n_buffers);
  }
}

ArrowErrorCode ArrowIpcReaderSetSchema(struct ArrowIpcReader* reader,
                                       struct ArrowSchema* schema,
                                       struct ArrowError* error) {
  struct ArrowIpcReaderPrivate* private_data =
      (struct ArrowIpcReaderPrivate*)reader->private_data;

  ArrowArrayViewReset(&private_data->array_view);

  if (private_data->fields != NULL) {
    ArrowFree(private_data->fields);
  }

  // Allocate Array and ArrayView based on schema without moving the schema
  // this will fail if the schema is not valid.
  NANOARROW_RETURN_NOT_OK(
      ArrowArrayViewInitFromSchema(&private_data->array_view, schema, error));

  // Root must be a struct
  if (private_data->array_view.storage_type != NANOARROW_TYPE_STRUCT) {
    ArrowErrorSet(error, "schema must be a struct type");
    return EINVAL;
  }

  // Walk tree and calculate how many fields we need to allocate
  private_data->n_fields = 0;
  ArrowIpcReaderCountFields(schema, &private_data->n_fields);
  private_data->fields = (struct ArrowIpcField*)ArrowMalloc(private_data->n_fields *
                                                            sizeof(struct ArrowIpcField));
  if (private_data->fields == NULL) {
    ArrowErrorSet(error, "Failed to allocate reader->fields");
    return ENOMEM;
  }
  memset(private_data->fields, 0, private_data->n_fields * sizeof(struct ArrowIpcField));

  // Init field information and calculate starting buffer offset for each
  int64_t field_i = 0;
  ArrowIpcReaderInitFields(private_data->fields, &private_data->array_view, &field_i,
                           &private_data->n_buffers);

  return NANOARROW_OK;
}

struct ArrowIpcArraySetter {
  ns(FieldNode_vec_t) fields;
  int64_t field_i;
  ns(Buffer_vec_t) buffers;
  int64_t buffer_i;
  struct ArrowBufferView body;
  enum ArrowIpcCompressionType codec;
  enum ArrowIpcEndianness endianness;
  enum ArrowIpcEndianness system_endianness;
};

static int ArrowIpcReaderMakeBuffer(struct ArrowIpcArraySetter* setter, int64_t offset,
                                    int64_t length, struct ArrowBuffer* out,
                                    struct ArrowError* error) {
  if (length == 0) {
    return NANOARROW_OK;
  }

  // Check that this buffer fits within the body
  if (offset < 0 || (offset + length) > setter->body.size_bytes) {
    ArrowErrorSet(error,
                  "Buffer %ld requires body offsets [%ld..%ld) but body has size %ld",
                  (long)setter->buffer_i - 1, (long)offset, (long)offset + (long)length,
                  setter->body.size_bytes);
    return EINVAL;
  }

  struct ArrowBufferView view;
  view.data.as_uint8 = setter->body.data.as_uint8 + offset;
  view.size_bytes = length;

  if (setter->codec != NANOARROW_IPC_COMPRESSION_TYPE_NONE) {
    ArrowErrorSet(error, "The nanoarrow_ipc extension does not support compression");
    return ENOTSUP;
  }

  if (setter->endianness != NANOARROW_IPC_ENDIANNESS_UNINITIALIZED &&
      setter->endianness != setter->system_endianness) {
    ArrowErrorSet(error,
                  "The nanoarrow_ipc extension does not support non-system endianness");
    return ENOTSUP;
  }

  int result = ArrowBufferAppendBufferView(out, view);
  if (result != NANOARROW_OK) {
    ArrowErrorSet(error, "Failed to copy buffer");
    return result;
  }

  return NANOARROW_OK;
}

static int ArrowIpcReaderWalkGetArray(struct ArrowIpcArraySetter* setter,
                                      struct ArrowArray* array,
                                      struct ArrowError* error) {
  ns(FieldNode_struct_t) field =
      ns(FieldNode_vec_at(setter->fields, (size_t)setter->field_i));
  array->length = ns(FieldNode_length(field));
  array->null_count = ns(FieldNode_null_count(field));
  setter->field_i += 1;

  for (int64_t i = 0; i < array->n_buffers; i++) {
    ns(Buffer_struct_t) buffer =
        ns(Buffer_vec_at(setter->buffers, (size_t)setter->buffer_i));
    int64_t buffer_offset = ns(Buffer_offset(buffer));
    int64_t buffer_length = ns(Buffer_length(buffer));
    setter->buffer_i += 1;

    struct ArrowBuffer* buffer_dst = ArrowArrayBuffer(array, i);
    NANOARROW_RETURN_NOT_OK(ArrowIpcReaderMakeBuffer(setter, buffer_offset, buffer_length,
                                                     buffer_dst, error));
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowIpcReaderWalkGetArray(setter, array->children[i], error));
  }

  return NANOARROW_OK;
}

static int ArrowIpcArrayInitFromArrayView(struct ArrowArray* array,
                                          struct ArrowArrayView* array_view) {
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(array, array_view->storage_type));
  NANOARROW_RETURN_NOT_OK(ArrowArrayAllocateChildren(array, array_view->n_children));
  for (int64_t i = 0; i < array_view->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        ArrowIpcArrayInitFromArrayView(array->children[i], array_view->children[i]));
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcReaderGetArray(struct ArrowIpcReader* reader,
                                      struct ArrowBufferView body, int64_t field_i,
                                      struct ArrowArray* out, struct ArrowError* error) {
  struct ArrowIpcReaderPrivate* private_data =
      (struct ArrowIpcReaderPrivate*)reader->private_data;

  if (private_data->last_message == NULL ||
      reader->message_type != NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH) {
    ArrowErrorSet(error, "reader did not just decode a RecordBatch message");
    return EINVAL;
  }

  ns(RecordBatch_table_t) batch = (ns(RecordBatch_table_t))private_data->last_message;

  // RecordBatch messages don't count the root node but reader->fields does
  struct ArrowIpcField* root = private_data->fields + field_i + 1;

  struct ArrowArray temp;
  temp.release = NULL;
  int result = ArrowIpcArrayInitFromArrayView(&temp, root->array_view);
  if (result != NANOARROW_OK) {
    ArrowErrorSet(error, "Failed to initialize output array");
    return result;
  }

  struct ArrowIpcArraySetter setter;
  setter.fields = ns(RecordBatch_nodes(batch));
  setter.field_i = field_i;
  setter.buffers = ns(RecordBatch_buffers(batch));
  setter.buffer_i = root->buffer_offset - 1;
  setter.body = body;
  setter.codec = reader->codec;
  setter.endianness = reader->endianness;

  // This should probably be done at compile time
  uint32_t check = 1;
  char first_byte;
  memcpy(&first_byte, &check, sizeof(char));
  if (first_byte) {
    setter.system_endianness = NANOARROW_IPC_ENDIANNESS_LITTLE;
  } else {
    setter.system_endianness = NANOARROW_IPC_ENDIANNESS_BIG;
  }

  // The flatbuffers FieldNode doesn't count the root struct so we have to loop over the
  // children ourselves
  if (field_i == -1) {
    temp.length = ns(RecordBatch_length(batch));
    temp.null_count = 0;
    setter.field_i++;
    setter.buffer_i++;

    for (int64_t i = 0; i < temp.n_children; i++) {
      result = ArrowIpcReaderWalkGetArray(&setter, temp.children[i], error);
      if (result != NANOARROW_OK) {
        temp.release(&temp);
        return result;
      }
    }
  } else {
    result = ArrowIpcReaderWalkGetArray(&setter, &temp, error);
    if (result != NANOARROW_OK) {
      temp.release(&temp);
      return result;
    }
  }

  // TODO: this performs some validation but doesn't do everything we need it to do
  // notably it doesn't loop over offset buffers to look for values that will cause
  // out-of-bounds buffer access on the data buffer or child arrays.
  result = ArrowArrayFinishBuilding(&temp, error);
  if (result != NANOARROW_OK) {
    temp.release(&temp);
    return result;
  }

  ArrowArrayMove(&temp, out);
  return NANOARROW_OK;
}
