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

// For thread safe shared buffers we need C11 + stdatomic.h
// Can compile with -DNANOARROW_IPC_USE_STDATOMIC=0 or 1 to override
// automatic detection
#if !defined(NANOARROW_IPC_USE_STDATOMIC)
#define NANOARROW_IPC_USE_STDATOMIC 0

// Check for C11
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L

// Check for GCC 4.8, which doesn't include stdatomic.h but does
// not define __STDC_NO_ATOMICS__
#if defined(__clang__) || !defined(__GNUC__) || __GNUC__ >= 5

#if !defined(__STDC_NO_ATOMICS__)
#include <stdatomic.h>
#undef NANOARROW_IPC_USE_STDATOMIC
#define NANOARROW_IPC_USE_STDATOMIC 1
#endif
#endif
#endif

#endif

#include "nanoarrow.h"
#include "nanoarrow_ipc.h"
#include "nanoarrow_ipc_flatcc_generated.h"

// R 3.6 / Windows builds on a very old toolchain that does not define ENODATA
#if defined(_WIN32) && !defined(_MSC_VER) && !defined(ENODATA)
#define ENODATA 120
#endif

// A more readable expression way to refer to the fact that there are 8 bytes
// at the beginning of every message header.
static const int32_t kMessageHeaderPrefixSize = 8;

// Internal representation of a parsed "Field" from flatbuffers. This
// represents a field in a depth-first walk of column arrays and their
// children.
struct ArrowIpcField {
  // Pointer to the ArrowIpcDecoderPrivate::array_view or child for this node
  struct ArrowArrayView* array_view;
  // Pointer to the ArrowIpcDecoderPrivate::array or child for this node. This
  // array is scratch space for any intermediary allocations (i.e., it is never moved
  // to the user).
  struct ArrowArray* array;
  // The cumulative number of buffers preceding this node.
  int64_t buffer_offset;
};

// Internal data specific to the read/decode process
struct ArrowIpcDecoderPrivate {
  // The endianness that will be assumed for decoding future RecordBatch messages
  enum ArrowIpcEndianness endianness;
  // A cached system endianness value
  enum ArrowIpcEndianness system_endianness;
  // An ArrowArrayView whose length/null_count/buffers are set directly from the
  // deserialized flatbuffer message (i.e., no fully underlying ArrowArray exists,
  // although some buffers may be temporarily owned by ArrowIpcDecoderPrivate::array).
  struct ArrowArrayView array_view;
  // An ArrowArray with the same structure as the ArrowArrayView whose ArrowArrayBuffer()
  // values are used to allocate or store memory when this is required. This ArrowArray
  // is never moved to the caller; however, its buffers may be moved to the final output
  // ArrowArray if the caller requests one.
  struct ArrowArray array;
  // The number of fields in the flattened depth-first walk of columns and their children
  int64_t n_fields;
  // Array of cached information such that given a field index it is possible to locate
  // the ArrowArrayView/ArrowArray where the depth-first buffer/field walk should start.
  struct ArrowIpcField* fields;
  // The number of buffers that future RecordBatch messages must have to match the schema
  // that has been set.
  int64_t n_buffers;
  // A pointer to the last flatbuffers message.
  const void* last_message;
};

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

static enum ArrowIpcEndianness ArrowIpcSystemEndianness(void) {
  uint32_t check = 1;
  char first_byte;
  memcpy(&first_byte, &check, sizeof(char));
  if (first_byte) {
    return NANOARROW_IPC_ENDIANNESS_LITTLE;
  } else {
    return NANOARROW_IPC_ENDIANNESS_BIG;
  }
}

#if NANOARROW_IPC_USE_STDATOMIC
struct ArrowIpcSharedBufferPrivate {
  struct ArrowBuffer src;
  atomic_long reference_count;
};

static int64_t ArrowIpcSharedBufferUpdate(
    struct ArrowIpcSharedBufferPrivate* private_data, int delta) {
  int64_t old_count = atomic_fetch_add(&private_data->reference_count, delta);
  return old_count + delta;
}

static void ArrowIpcSharedBufferSet(struct ArrowIpcSharedBufferPrivate* private_data,
                                    int64_t count) {
  atomic_store(&private_data->reference_count, count);
}

int ArrowIpcSharedBufferIsThreadSafe(void) { return 1; }
#else
struct ArrowIpcSharedBufferPrivate {
  struct ArrowBuffer src;
  int64_t reference_count;
};

static int64_t ArrowIpcSharedBufferUpdate(
    struct ArrowIpcSharedBufferPrivate* private_data, int delta) {
  private_data->reference_count += delta;
  return private_data->reference_count;
}

static void ArrowIpcSharedBufferSet(struct ArrowIpcSharedBufferPrivate* private_data,
                                    int64_t count) {
  private_data->reference_count = count;
}

int ArrowIpcSharedBufferIsThreadSafe(void) { return 0; }
#endif

static void ArrowIpcSharedBufferFree(struct ArrowBufferAllocator* allocator, uint8_t* ptr,
                                     int64_t size) {
  NANOARROW_UNUSED(allocator);
  NANOARROW_UNUSED(ptr);
  NANOARROW_UNUSED(size);

  struct ArrowIpcSharedBufferPrivate* private_data =
      (struct ArrowIpcSharedBufferPrivate*)allocator->private_data;

  if (ArrowIpcSharedBufferUpdate(private_data, -1) == 0) {
    ArrowBufferReset(&private_data->src);
    ArrowFree(private_data);
  }
}

ArrowErrorCode ArrowIpcSharedBufferInit(struct ArrowIpcSharedBuffer* shared,
                                        struct ArrowBuffer* src) {
  if (src->data == NULL) {
    ArrowBufferMove(src, &shared->private_src);
    return NANOARROW_OK;
  }

  struct ArrowIpcSharedBufferPrivate* private_data =
      (struct ArrowIpcSharedBufferPrivate*)ArrowMalloc(
          sizeof(struct ArrowIpcSharedBufferPrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  ArrowBufferMove(src, &private_data->src);
  ArrowIpcSharedBufferSet(private_data, 1);

  ArrowBufferInit(&shared->private_src);
  shared->private_src.data = private_data->src.data;
  shared->private_src.size_bytes = private_data->src.size_bytes;
  // Don't expose any extra capcity from src so that any calls to ArrowBufferAppend
  // on this buffer will fail.
  shared->private_src.capacity_bytes = private_data->src.size_bytes;
  shared->private_src.allocator =
      ArrowBufferDeallocator(&ArrowIpcSharedBufferFree, private_data);
  return NANOARROW_OK;
}

static void ArrowIpcSharedBufferClone(struct ArrowIpcSharedBuffer* shared,
                                      struct ArrowBuffer* shared_out) {
  if (shared->private_src.data == NULL) {
    ArrowBufferInit(shared_out);
    shared_out->size_bytes = shared_out->size_bytes;
    shared_out->capacity_bytes = shared_out->capacity_bytes;
    return;
  }

  struct ArrowIpcSharedBufferPrivate* private_data =
      (struct ArrowIpcSharedBufferPrivate*)shared->private_src.allocator.private_data;
  ArrowIpcSharedBufferUpdate(private_data, 1);
  memcpy(shared_out, shared, sizeof(struct ArrowBuffer));
}

void ArrowIpcSharedBufferReset(struct ArrowIpcSharedBuffer* shared) {
  ArrowBufferReset(&shared->private_src);
}

static int ArrowIpcDecoderNeedsSwapEndian(struct ArrowIpcDecoder* decoder) {
  struct ArrowIpcDecoderPrivate* private_data =
      (struct ArrowIpcDecoderPrivate*)decoder->private_data;
  switch (private_data->endianness) {
    case NANOARROW_IPC_ENDIANNESS_LITTLE:
    case NANOARROW_IPC_ENDIANNESS_BIG:
      return private_data->endianness != NANOARROW_IPC_ENDIANNESS_UNINITIALIZED &&
             private_data->endianness != private_data->system_endianness;
    default:
      return 0;
  }
}

ArrowErrorCode ArrowIpcDecoderInit(struct ArrowIpcDecoder* decoder) {
  memset(decoder, 0, sizeof(struct ArrowIpcDecoder));
  struct ArrowIpcDecoderPrivate* private_data =
      (struct ArrowIpcDecoderPrivate*)ArrowMalloc(sizeof(struct ArrowIpcDecoderPrivate));
  if (private_data == NULL) {
    return ENOMEM;
  }

  memset(private_data, 0, sizeof(struct ArrowIpcDecoderPrivate));
  private_data->system_endianness = ArrowIpcSystemEndianness();
  decoder->private_data = private_data;
  return NANOARROW_OK;
}

void ArrowIpcDecoderReset(struct ArrowIpcDecoder* decoder) {
  struct ArrowIpcDecoderPrivate* private_data =
      (struct ArrowIpcDecoderPrivate*)decoder->private_data;

  if (private_data != NULL) {
    ArrowArrayViewReset(&private_data->array_view);

    if (private_data->array.release != NULL) {
      ArrowArrayRelease(&private_data->array);
    }

    if (private_data->fields != NULL) {
      ArrowFree(private_data->fields);
      private_data->n_fields = 0;
    }

    ArrowFree(private_data);
    memset(decoder, 0, sizeof(struct ArrowIpcDecoder));
  }
}

static inline uint32_t ArrowIpcReadContinuationBytes(struct ArrowBufferView* data) {
  uint32_t value;
  memcpy(&value, data->data.as_uint8, sizeof(uint32_t));
  data->data.as_uint8 += sizeof(uint32_t);
  data->size_bytes -= sizeof(uint32_t);
  return value;
}

static inline int32_t ArrowIpcReadInt32LE(struct ArrowBufferView* data, int swap_endian) {
  int32_t value;
  memcpy(&value, data->data.as_uint8, sizeof(int32_t));
  if (swap_endian) {
    value = bswap32(value);
  }

  data->data.as_uint8 += sizeof(int32_t);
  data->size_bytes -= sizeof(int32_t);
  return value;
}

#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(org_apache_arrow_flatbuf, x)

static int ArrowIpcDecoderSetMetadata(struct ArrowSchema* schema,
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

static int ArrowIpcDecoderSetTypeSimple(struct ArrowSchema* schema, int nanoarrow_type,
                                        struct ArrowError* error) {
  int result = ArrowSchemaSetType(schema, nanoarrow_type);
  if (result != NANOARROW_OK) {
    ArrowErrorSet(error, "ArrowSchemaSetType() failed for type %s",
                  ArrowTypeString(nanoarrow_type));
    return result;
  }

  return NANOARROW_OK;
}

static int ArrowIpcDecoderSetTypeInt(struct ArrowSchema* schema,
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

  return ArrowIpcDecoderSetTypeSimple(schema, nanoarrow_type, error);
}

static int ArrowIpcDecoderSetTypeFloatingPoint(struct ArrowSchema* schema,
                                               flatbuffers_generic_t type_generic,
                                               struct ArrowError* error) {
  ns(FloatingPoint_table_t) type = (ns(FloatingPoint_table_t))type_generic;
  int precision = ns(FloatingPoint_precision(type));
  switch (precision) {
    case ns(Precision_HALF):
      return ArrowIpcDecoderSetTypeSimple(schema, NANOARROW_TYPE_HALF_FLOAT, error);
    case ns(Precision_SINGLE):
      return ArrowIpcDecoderSetTypeSimple(schema, NANOARROW_TYPE_FLOAT, error);
    case ns(Precision_DOUBLE):
      return ArrowIpcDecoderSetTypeSimple(schema, NANOARROW_TYPE_DOUBLE, error);
    default:
      ArrowErrorSet(error, "Unexpected FloatingPoint Precision value: %d",
                    (int)precision);
      return EINVAL;
  }
}

static int ArrowIpcDecoderSetTypeDecimal(struct ArrowSchema* schema,
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

static int ArrowIpcDecoderSetTypeFixedSizeBinary(struct ArrowSchema* schema,
                                                 flatbuffers_generic_t type_generic,
                                                 struct ArrowError* error) {
  ns(FixedSizeBinary_table_t) type = (ns(FixedSizeBinary_table_t))type_generic;
  int fixed_size = ns(FixedSizeBinary_byteWidth(type));
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowSchemaSetTypeFixedSize(schema, NANOARROW_TYPE_FIXED_SIZE_BINARY, fixed_size),
      error);
  return NANOARROW_OK;
}

static int ArrowIpcDecoderSetTypeDate(struct ArrowSchema* schema,
                                      flatbuffers_generic_t type_generic,
                                      struct ArrowError* error) {
  ns(Date_table_t) type = (ns(Date_table_t))type_generic;
  int date_unit = ns(Date_unit(type));
  switch (date_unit) {
    case ns(DateUnit_DAY):
      return ArrowIpcDecoderSetTypeSimple(schema, NANOARROW_TYPE_DATE32, error);
    case ns(DateUnit_MILLISECOND):
      return ArrowIpcDecoderSetTypeSimple(schema, NANOARROW_TYPE_DATE64, error);
    default:
      ArrowErrorSet(error, "Unexpected Date DateUnit value: %d", (int)date_unit);
      return EINVAL;
  }
}

static int ArrowIpcDecoderSetTypeTime(struct ArrowSchema* schema,
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
                      ns(TimeUnit_name(ns(Time_unit(type)))), bitwidth);
        return EINVAL;
      }

      nanoarrow_type = NANOARROW_TYPE_TIME32;
      break;

    case ns(TimeUnit_MICROSECOND):
    case ns(TimeUnit_NANOSECOND):
      if (bitwidth != 64) {
        ArrowErrorSet(error, "Expected bitwidth of 64 for Time TimeUnit %s but found %d",
                      ns(TimeUnit_name(ns(Time_unit(type)))), bitwidth);
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

static int ArrowIpcDecoderSetTypeTimestamp(struct ArrowSchema* schema,
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

static int ArrowIpcDecoderSetTypeDuration(struct ArrowSchema* schema,
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

static int ArrowIpcDecoderSetTypeInterval(struct ArrowSchema* schema,
                                          flatbuffers_generic_t type_generic,
                                          struct ArrowError* error) {
  ns(Interval_table_t) type = (ns(Interval_table_t))type_generic;
  int interval_unit = ns(Interval_unit(type));

  switch (interval_unit) {
    case ns(IntervalUnit_YEAR_MONTH):
      return ArrowIpcDecoderSetTypeSimple(schema, NANOARROW_TYPE_INTERVAL_MONTHS, error);
    case ns(IntervalUnit_DAY_TIME):
      return ArrowIpcDecoderSetTypeSimple(schema, NANOARROW_TYPE_INTERVAL_DAY_TIME,
                                          error);
    case ns(IntervalUnit_MONTH_DAY_NANO):
      return ArrowIpcDecoderSetTypeSimple(schema, NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO,
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
static int ArrowIpcDecoderSetTypeSimpleNested(struct ArrowSchema* schema,
                                              const char* format,
                                              struct ArrowError* error) {
  int result = ArrowSchemaSetFormat(schema, format);
  if (result != NANOARROW_OK) {
    ArrowErrorSet(error, "ArrowSchemaSetFormat('%s') failed", format);
    return result;
  }

  return NANOARROW_OK;
}

static int ArrowIpcDecoderSetTypeFixedSizeList(struct ArrowSchema* schema,
                                               flatbuffers_generic_t type_generic,
                                               struct ArrowError* error) {
  ns(FixedSizeList_table_t) type = (ns(FixedSizeList_table_t))type_generic;
  int32_t fixed_size = ns(FixedSizeList_listSize(type));

  char fixed_size_str[128];
  int n_chars = snprintf(fixed_size_str, 128, "+w:%d", fixed_size);
  if (n_chars < 0) {
    ArrowErrorSet(error, "snprintf() encoding error");
    return ERANGE;
  }

  fixed_size_str[n_chars] = '\0';
  return ArrowIpcDecoderSetTypeSimpleNested(schema, fixed_size_str, error);
}

static int ArrowIpcDecoderSetTypeMap(struct ArrowSchema* schema,
                                     flatbuffers_generic_t type_generic,
                                     struct ArrowError* error) {
  ns(Map_table_t) type = (ns(Map_table_t))type_generic;
  NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderSetTypeSimpleNested(schema, "+m", error));

  if (ns(Map_keysSorted(type))) {
    schema->flags |= ARROW_FLAG_MAP_KEYS_SORTED;
  } else {
    schema->flags &= ~ARROW_FLAG_MAP_KEYS_SORTED;
  }

  return NANOARROW_OK;
}

static int ArrowIpcDecoderSetTypeUnion(struct ArrowSchema* schema,
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

  if (n_chars < 0) {
    ArrowErrorSet(error, "snprintf() encoding error");
    return ERANGE;
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

      if (n_chars < 0) {
        ArrowErrorSet(error, "snprintf() encoding error");
        return ERANGE;
      }

      for (int64_t i = 1; i < n_type_ids; i++) {
        n_chars = snprintf(format_cursor, format_out_size, ",%d",
                           (int)flatbuffers_int32_vec_at(type_ids, i));
        format_cursor += n_chars;
        format_out_size -= n_chars;

        if (n_chars < 0) {
          ArrowErrorSet(error, "snprintf() encoding error");
          return ERANGE;
        }
      }
    }
  } else if (n_children > 0) {
    n_chars = snprintf(format_cursor, format_out_size, "0");
    format_cursor += n_chars;
    format_out_size -= n_chars;

    if (n_chars < 0) {
      ArrowErrorSet(error, "snprintf() encoding error");
      return ERANGE;
    }

    for (int64_t i = 1; i < n_children; i++) {
      n_chars = snprintf(format_cursor, format_out_size, ",%d", (int)i);
      format_cursor += n_chars;
      format_out_size -= n_chars;

      if (n_chars < 0) {
        ArrowErrorSet(error, "snprintf() encoding error");
        return ERANGE;
      }
    }
  }

  return ArrowIpcDecoderSetTypeSimpleNested(schema, union_types_str, error);
}

static int ArrowIpcDecoderSetType(struct ArrowSchema* schema, ns(Field_table_t) field,
                                  int64_t n_children, struct ArrowError* error) {
  int type_type = ns(Field_type_type(field));
  switch (type_type) {
    case ns(Type_Null):
      return ArrowIpcDecoderSetTypeSimple(schema, NANOARROW_TYPE_NA, error);
    case ns(Type_Bool):
      return ArrowIpcDecoderSetTypeSimple(schema, NANOARROW_TYPE_BOOL, error);
    case ns(Type_Int):
      return ArrowIpcDecoderSetTypeInt(schema, ns(Field_type_get(field)), error);
    case ns(Type_FloatingPoint):
      return ArrowIpcDecoderSetTypeFloatingPoint(schema, ns(Field_type_get(field)),
                                                 error);
    case ns(Type_Decimal):
      return ArrowIpcDecoderSetTypeDecimal(schema, ns(Field_type_get(field)), error);
    case ns(Type_Binary):
      return ArrowIpcDecoderSetTypeSimple(schema, NANOARROW_TYPE_BINARY, error);
    case ns(Type_LargeBinary):
      return ArrowIpcDecoderSetTypeSimple(schema, NANOARROW_TYPE_LARGE_BINARY, error);
    case ns(Type_FixedSizeBinary):
      return ArrowIpcDecoderSetTypeFixedSizeBinary(schema, ns(Field_type_get(field)),
                                                   error);
    case ns(Type_Utf8):
      return ArrowIpcDecoderSetTypeSimple(schema, NANOARROW_TYPE_STRING, error);
    case ns(Type_LargeUtf8):
      return ArrowIpcDecoderSetTypeSimple(schema, NANOARROW_TYPE_LARGE_STRING, error);
    case ns(Type_Date):
      return ArrowIpcDecoderSetTypeDate(schema, ns(Field_type_get(field)), error);
    case ns(Type_Time):
      return ArrowIpcDecoderSetTypeTime(schema, ns(Field_type_get(field)), error);
    case ns(Type_Timestamp):
      return ArrowIpcDecoderSetTypeTimestamp(schema, ns(Field_type_get(field)), error);
    case ns(Type_Duration):
      return ArrowIpcDecoderSetTypeDuration(schema, ns(Field_type_get(field)), error);
    case ns(Type_Interval):
      return ArrowIpcDecoderSetTypeInterval(schema, ns(Field_type_get(field)), error);
    case ns(Type_Struct_):
      return ArrowIpcDecoderSetTypeSimpleNested(schema, "+s", error);
    case ns(Type_List):
      return ArrowIpcDecoderSetTypeSimpleNested(schema, "+l", error);
    case ns(Type_LargeList):
      return ArrowIpcDecoderSetTypeSimpleNested(schema, "+L", error);
    case ns(Type_FixedSizeList):
      return ArrowIpcDecoderSetTypeFixedSizeList(schema, ns(Field_type_get(field)),
                                                 error);
    case ns(Type_Map):
      return ArrowIpcDecoderSetTypeMap(schema, ns(Field_type_get(field)), error);
    case ns(Type_Union):
      return ArrowIpcDecoderSetTypeUnion(schema, ns(Field_type_get(field)), n_children,
                                         error);
    default:
      ArrowErrorSet(error, "Unrecognized Field type with value %d", (int)type_type);
      return EINVAL;
  }
}

static int ArrowIpcDecoderSetChildren(struct ArrowSchema* schema, ns(Field_vec_t) fields,
                                      struct ArrowError* error);

static int ArrowIpcDecoderSetField(struct ArrowSchema* schema, ns(Field_table_t) field,
                                   struct ArrowError* error) {
  // No dictionary support yet
  if (ns(Field_dictionary_is_present(field))) {
    ArrowErrorSet(error, "Schema message field with DictionaryEncoding not supported");
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

  NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderSetType(schema, field, n_children, error));

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

  NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderSetChildren(schema, children, error));
  return ArrowIpcDecoderSetMetadata(schema, ns(Field_custom_metadata(field)), error);
}

static int ArrowIpcDecoderSetChildren(struct ArrowSchema* schema, ns(Field_vec_t) fields,
                                      struct ArrowError* error) {
  int64_t n_fields = ns(Schema_vec_len(fields));

  for (int64_t i = 0; i < n_fields; i++) {
    ns(Field_table_t) field = ns(Field_vec_at(fields, i));
    NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderSetField(schema->children[i], field, error));
  }

  return NANOARROW_OK;
}

static int ArrowIpcDecoderDecodeSchemaHeader(struct ArrowIpcDecoder* decoder,
                                             flatbuffers_generic_t message_header,
                                             struct ArrowError* error) {
  ns(Schema_table_t) schema = (ns(Schema_table_t))message_header;
  int endianness = ns(Schema_endianness(schema));
  switch (endianness) {
    case ns(Endianness_Little):
      decoder->endianness = NANOARROW_IPC_ENDIANNESS_LITTLE;
      break;
    case ns(Endianness_Big):
      decoder->endianness = NANOARROW_IPC_ENDIANNESS_BIG;
      break;
    default:
      ArrowErrorSet(error,
                    "Expected Schema endianness of 0 (little) or 1 (big) but got %d",
                    (int)endianness);
      return EINVAL;
  }

  ns(Feature_vec_t) features = ns(Schema_features(schema));
  int64_t n_features = ns(Feature_vec_len(features));
  decoder->feature_flags = 0;

  for (int64_t i = 0; i < n_features; i++) {
    ns(Feature_enum_t) feature = ns(Feature_vec_at(features, i));
    switch (feature) {
      case ns(Feature_COMPRESSED_BODY):
        decoder->feature_flags |= NANOARROW_IPC_FEATURE_COMPRESSED_BODY;
        break;
      case ns(Feature_DICTIONARY_REPLACEMENT):
        decoder->feature_flags |= NANOARROW_IPC_FEATURE_DICTIONARY_REPLACEMENT;
        break;
      default:
        ArrowErrorSet(error, "Unrecognized Schema feature with value %d", (int)feature);
        return EINVAL;
    }
  }

  return NANOARROW_OK;
}

static int ArrowIpcDecoderDecodeRecordBatchHeader(struct ArrowIpcDecoder* decoder,
                                                  flatbuffers_generic_t message_header,
                                                  struct ArrowError* error) {
  struct ArrowIpcDecoderPrivate* private_data =
      (struct ArrowIpcDecoderPrivate*)decoder->private_data;

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
        decoder->codec = NANOARROW_IPC_COMPRESSION_TYPE_LZ4_FRAME;
        break;
      case ns(CompressionType_ZSTD):
        decoder->codec = NANOARROW_IPC_COMPRESSION_TYPE_ZSTD;
        break;
      default:
        ArrowErrorSet(error, "Unrecognized RecordBatch BodyCompression codec value: %d",
                      (int)codec);
        return EINVAL;
    }
  } else {
    decoder->codec = NANOARROW_IPC_COMPRESSION_TYPE_NONE;
  }

  // Copying field node and buffer information is separate so as only to pay for the
  // nodes that are actually accessed.
  return NANOARROW_OK;
}

// Wipes any "current message" fields before moving on to a new message
static inline void ArrowIpcDecoderResetHeaderInfo(struct ArrowIpcDecoder* decoder) {
  struct ArrowIpcDecoderPrivate* private_data =
      (struct ArrowIpcDecoderPrivate*)decoder->private_data;

  decoder->message_type = 0;
  decoder->metadata_version = 0;
  decoder->endianness = 0;
  decoder->feature_flags = 0;
  decoder->codec = 0;
  decoder->header_size_bytes = 0;
  decoder->body_size_bytes = 0;
  private_data->last_message = NULL;
}

// Returns NANOARROW_OK if data is large enough to read the first 8 bytes
// of the message header, ESPIPE if reading more data might help, or EINVAL if the content
// is not valid. Advances the input ArrowBufferView by 8 bytes.
static inline int ArrowIpcDecoderReadHeaderPrefix(struct ArrowIpcDecoder* decoder,
                                                  struct ArrowBufferView* data_mut,
                                                  int32_t* message_size_bytes,
                                                  struct ArrowError* error) {
  struct ArrowIpcDecoderPrivate* private_data =
      (struct ArrowIpcDecoderPrivate*)decoder->private_data;

  if (data_mut->size_bytes < kMessageHeaderPrefixSize) {
    ArrowErrorSet(error, "Expected data of at least 8 bytes but only %ld bytes remain",
                  (long)data_mut->size_bytes);
    return ESPIPE;
  }

  uint32_t continuation = ArrowIpcReadContinuationBytes(data_mut);
  if (continuation != 0xFFFFFFFF) {
    ArrowErrorSet(error, "Expected 0xFFFFFFFF at start of message but found 0x%08X",
                  (unsigned int)continuation);
    return EINVAL;
  }

  int swap_endian = private_data->system_endianness == NANOARROW_IPC_ENDIANNESS_BIG;
  int32_t header_body_size_bytes = ArrowIpcReadInt32LE(data_mut, swap_endian);
  *message_size_bytes = header_body_size_bytes + kMessageHeaderPrefixSize;
  if (header_body_size_bytes < 0) {
    ArrowErrorSet(
        error, "Expected message body size > 0 but found message body size of %ld bytes",
        (long)header_body_size_bytes);
    return EINVAL;
  }

  if (header_body_size_bytes == 0) {
    ArrowErrorSet(error, "End of Arrow stream");
    return ENODATA;
  }

  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcDecoderPeekHeader(struct ArrowIpcDecoder* decoder,
                                         struct ArrowBufferView data,
                                         struct ArrowError* error) {
  ArrowIpcDecoderResetHeaderInfo(decoder);
  NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderReadHeaderPrefix(
      decoder, &data, &decoder->header_size_bytes, error));
  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcDecoderVerifyHeader(struct ArrowIpcDecoder* decoder,
                                           struct ArrowBufferView data,
                                           struct ArrowError* error) {
  struct ArrowIpcDecoderPrivate* private_data =
      (struct ArrowIpcDecoderPrivate*)decoder->private_data;

  ArrowIpcDecoderResetHeaderInfo(decoder);
  NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderReadHeaderPrefix(
      decoder, &data, &decoder->header_size_bytes, error));

  // Check that data contains at least the entire header (return ESPIPE to signal
  // that reading more data may help).
  int64_t message_body_size = decoder->header_size_bytes - kMessageHeaderPrefixSize;
  if (data.size_bytes < message_body_size) {
    ArrowErrorSet(error,
                  "Expected >= %ld bytes of remaining data but found %ld bytes in buffer",
                  (long)message_body_size + kMessageHeaderPrefixSize,
                  (long)data.size_bytes + kMessageHeaderPrefixSize);
    return ESPIPE;
  }

  // Run flatbuffers verification
  if (ns(Message_verify_as_root(data.data.as_uint8, message_body_size) !=
         flatcc_verify_ok)) {
    ArrowErrorSet(error, "Message flatbuffer verification failed");
    return EINVAL;
  }

  // Read some basic information from the message
  ns(Message_table_t) message = ns(Message_as_root(data.data.as_uint8));
  decoder->metadata_version = ns(Message_version(message));
  decoder->message_type = ns(Message_header_type(message));
  decoder->body_size_bytes = ns(Message_bodyLength(message));

  private_data->last_message = ns(Message_header_get(message));
  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcDecoderDecodeHeader(struct ArrowIpcDecoder* decoder,
                                           struct ArrowBufferView data,
                                           struct ArrowError* error) {
  struct ArrowIpcDecoderPrivate* private_data =
      (struct ArrowIpcDecoderPrivate*)decoder->private_data;

  ArrowIpcDecoderResetHeaderInfo(decoder);
  NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderReadHeaderPrefix(
      decoder, &data, &decoder->header_size_bytes, error));

  // Check that data contains at least the entire header (return ESPIPE to signal
  // that reading more data may help).
  int64_t message_body_size = decoder->header_size_bytes - kMessageHeaderPrefixSize;
  if (data.size_bytes < message_body_size) {
    ArrowErrorSet(error,
                  "Expected >= %ld bytes of remaining data but found %ld bytes in buffer",
                  (long)message_body_size + kMessageHeaderPrefixSize,
                  (long)data.size_bytes + kMessageHeaderPrefixSize);
    return ESPIPE;
  }

  ns(Message_table_t) message = ns(Message_as_root(data.data.as_uint8));
  if (!message) {
    return EINVAL;
  }

  // Read some basic information from the message
  decoder->metadata_version = ns(Message_version(message));
  decoder->message_type = ns(Message_header_type(message));
  decoder->body_size_bytes = ns(Message_bodyLength(message));

  switch (decoder->metadata_version) {
    case ns(MetadataVersion_V5):
      break;
    case ns(MetadataVersion_V1):
    case ns(MetadataVersion_V2):
    case ns(MetadataVersion_V3):
    case ns(MetadataVersion_V4):
      ArrowErrorSet(error, "Expected metadata version V5 but found %s",
                    ns(MetadataVersion_name(ns(Message_version(message)))));
      break;
    default:
      ArrowErrorSet(error, "Unexpected value for Message metadata version (%d)",
                    decoder->metadata_version);
      return EINVAL;
  }

  flatbuffers_generic_t message_header = ns(Message_header_get(message));
  switch (decoder->message_type) {
    case ns(MessageHeader_Schema):
      NANOARROW_RETURN_NOT_OK(
          ArrowIpcDecoderDecodeSchemaHeader(decoder, message_header, error));
      break;
    case ns(MessageHeader_RecordBatch):
      NANOARROW_RETURN_NOT_OK(
          ArrowIpcDecoderDecodeRecordBatchHeader(decoder, message_header, error));
      break;
    case ns(MessageHeader_DictionaryBatch):
    case ns(MessageHeader_Tensor):
    case ns(MessageHeader_SparseTensor):
      ArrowErrorSet(error, "Unsupported message type: '%s'",
                    ns(MessageHeader_type_name(ns(Message_header_type(message)))));
      return ENOTSUP;
    default:
      ArrowErrorSet(error, "Unknown message type: %d", (int)(decoder->message_type));
      return EINVAL;
  }

  private_data->last_message = message_header;
  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcDecoderDecodeSchema(struct ArrowIpcDecoder* decoder,
                                           struct ArrowSchema* out,
                                           struct ArrowError* error) {
  struct ArrowIpcDecoderPrivate* private_data =
      (struct ArrowIpcDecoderPrivate*)decoder->private_data;

  if (private_data->last_message == NULL ||
      decoder->message_type != NANOARROW_IPC_MESSAGE_TYPE_SCHEMA) {
    ArrowErrorSet(error, "decoder did not just decode a Schema message");
    return EINVAL;
  }

  ns(Schema_table_t) schema = (ns(Schema_table_t))private_data->last_message;

  ns(Field_vec_t) fields = ns(Schema_fields(schema));
  int64_t n_fields = ns(Schema_vec_len(fields));

  struct ArrowSchema tmp;
  ArrowSchemaInit(&tmp);
  int result = ArrowSchemaSetTypeStruct(&tmp, n_fields);
  if (result != NANOARROW_OK) {
    ArrowSchemaRelease(&tmp);
    ArrowErrorSet(error, "Failed to allocate struct schema with %ld children",
                  (long)n_fields);
    return result;
  }

  // Top-level batch schema is typically non-nullable
  tmp.flags = 0;

  result = ArrowIpcDecoderSetChildren(&tmp, fields, error);
  if (result != NANOARROW_OK) {
    ArrowSchemaRelease(&tmp);
    return result;
  }

  result = ArrowIpcDecoderSetMetadata(&tmp, ns(Schema_custom_metadata(schema)), error);
  if (result != NANOARROW_OK) {
    ArrowSchemaRelease(&tmp);
    return result;
  }

  ArrowSchemaMove(&tmp, out);
  return NANOARROW_OK;
}

static void ArrowIpcDecoderCountFields(struct ArrowSchema* schema, int64_t* n_fields) {
  *n_fields += 1;
  for (int64_t i = 0; i < schema->n_children; i++) {
    ArrowIpcDecoderCountFields(schema->children[i], n_fields);
  }
}

static void ArrowIpcDecoderInitFields(struct ArrowIpcField* fields,
                                      struct ArrowArrayView* array_view,
                                      struct ArrowArray* array, int64_t* n_fields,
                                      int64_t* n_buffers) {
  struct ArrowIpcField* field = fields + (*n_fields);
  field->array_view = array_view;
  field->array = array;
  field->buffer_offset = *n_buffers;

  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    *n_buffers += array_view->layout.buffer_type[i] != NANOARROW_BUFFER_TYPE_NONE;
  }

  *n_fields += 1;

  for (int64_t i = 0; i < array_view->n_children; i++) {
    ArrowIpcDecoderInitFields(fields, array_view->children[i], array->children[i],
                              n_fields, n_buffers);
  }
}

ArrowErrorCode ArrowIpcDecoderSetSchema(struct ArrowIpcDecoder* decoder,
                                        struct ArrowSchema* schema,
                                        struct ArrowError* error) {
  struct ArrowIpcDecoderPrivate* private_data =
      (struct ArrowIpcDecoderPrivate*)decoder->private_data;

  // Reset previously allocated schema-specific resources
  private_data->n_buffers = 0;
  private_data->n_fields = 0;
  ArrowArrayViewReset(&private_data->array_view);
  if (private_data->array.release != NULL) {
    ArrowArrayRelease(&private_data->array);
  }
  if (private_data->fields != NULL) {
    ArrowFree(private_data->fields);
  }

  // Allocate Array and ArrayView based on schema without moving the schema.
  // This will fail if the schema is not valid.
  NANOARROW_RETURN_NOT_OK(
      ArrowArrayViewInitFromSchema(&private_data->array_view, schema, error));
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromArrayView(&private_data->array,
                                                      &private_data->array_view, error));

  // Root must be a struct
  if (private_data->array_view.storage_type != NANOARROW_TYPE_STRUCT) {
    ArrowErrorSet(error, "schema must be a struct type");
    return EINVAL;
  }

  // Walk tree and calculate how many fields we need to allocate
  ArrowIpcDecoderCountFields(schema, &private_data->n_fields);
  private_data->fields = (struct ArrowIpcField*)ArrowMalloc(private_data->n_fields *
                                                            sizeof(struct ArrowIpcField));
  if (private_data->fields == NULL) {
    ArrowErrorSet(error, "Failed to allocate decoder->fields");
    return ENOMEM;
  }
  memset(private_data->fields, 0, private_data->n_fields * sizeof(struct ArrowIpcField));

  // Init field information and calculate starting buffer offset for each
  int64_t field_i = 0;
  ArrowIpcDecoderInitFields(private_data->fields, &private_data->array_view,
                            &private_data->array, &field_i, &private_data->n_buffers);

  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcDecoderSetEndianness(struct ArrowIpcDecoder* decoder,
                                            enum ArrowIpcEndianness endianness) {
  struct ArrowIpcDecoderPrivate* private_data =
      (struct ArrowIpcDecoderPrivate*)decoder->private_data;

  switch (endianness) {
    case NANOARROW_IPC_ENDIANNESS_UNINITIALIZED:
    case NANOARROW_IPC_ENDIANNESS_LITTLE:
    case NANOARROW_IPC_ENDIANNESS_BIG:
      private_data->endianness = endianness;
      return NANOARROW_OK;
    default:
      return EINVAL;
  }
}

/// \brief Information required to read and/or decompress a single buffer
///
/// The RecordBatch message header contains a description of each buffer
/// in the message body. The ArrowIpcBufferSource is the parsed result of
/// a single buffer with compression and endian information such that the
/// original buffer can be reconstructed.
struct ArrowIpcBufferSource {
  int64_t body_offset_bytes;
  int64_t buffer_length_bytes;
  enum ArrowIpcCompressionType codec;
  enum ArrowType data_type;
  int64_t element_size_bits;
  int swap_endian;
};

/// \brief Materializing ArrowBuffer objects
///
/// Given a description of where a buffer is located inside the message body, make
/// the ArrowBuffer that will be placed into the correct ArrowArray. The decoder
/// does not do any IO and does not make any assumptions about how or if the body
/// has been read into memory. This abstraction is currently internal and exists
/// to support the two obvious ways a user might go about this: (1) using a
/// non-owned view of memory that must be copied slice-wise or (2) adding a reference
/// to an ArrowIpcSharedBuffer and returning a slice of that memory.
struct ArrowIpcBufferFactory {
  /// \brief User-defined callback to populate a buffer view
  ///
  /// At the time that this callback is called, the ArrowIpcBufferSource has been checked
  /// to ensure that it is within the body size declared by the message header. A
  /// possibly preallocated ArrowBuffer (dst) is provided, which implementations must use
  /// if an allocation is required (in which case the view must be populated pointing to
  /// the contents of the ArrowBuffer) If NANOARROW_OK is not returned, error must contain
  /// a null-terminated message.
  ArrowErrorCode (*make_buffer)(struct ArrowIpcBufferFactory* factory,
                                struct ArrowIpcBufferSource* src,
                                struct ArrowBufferView* dst_view, struct ArrowBuffer* dst,
                                struct ArrowError* error);

  /// \brief Caller-defined private data to be used in the callback.
  ///
  /// Usually this would be a description of where the body has been read into memory or
  /// information required to do so.
  void* private_data;
};

static ArrowErrorCode ArrowIpcMakeBufferFromView(struct ArrowIpcBufferFactory* factory,
                                                 struct ArrowIpcBufferSource* src,
                                                 struct ArrowBufferView* dst_view,
                                                 struct ArrowBuffer* dst,
                                                 struct ArrowError* error) {
  NANOARROW_UNUSED(factory);
  NANOARROW_UNUSED(dst);
  NANOARROW_UNUSED(error);

  struct ArrowBufferView* body = (struct ArrowBufferView*)factory->private_data;
  dst_view->data.as_uint8 = body->data.as_uint8 + src->body_offset_bytes;
  dst_view->size_bytes = src->buffer_length_bytes;
  return NANOARROW_OK;
}

static struct ArrowIpcBufferFactory ArrowIpcBufferFactoryFromView(
    struct ArrowBufferView* buffer_view) {
  struct ArrowIpcBufferFactory out;
  out.make_buffer = &ArrowIpcMakeBufferFromView;
  out.private_data = buffer_view;
  return out;
}

static ArrowErrorCode ArrowIpcMakeBufferFromShared(struct ArrowIpcBufferFactory* factory,
                                                   struct ArrowIpcBufferSource* src,
                                                   struct ArrowBufferView* dst_view,
                                                   struct ArrowBuffer* dst,
                                                   struct ArrowError* error) {
  NANOARROW_UNUSED(error);

  struct ArrowIpcSharedBuffer* shared =
      (struct ArrowIpcSharedBuffer*)factory->private_data;
  ArrowBufferReset(dst);
  ArrowIpcSharedBufferClone(shared, dst);
  dst->data += src->body_offset_bytes;
  dst->size_bytes = src->buffer_length_bytes;
  dst_view->data.data = dst->data;
  dst_view->size_bytes = dst->size_bytes;
  return NANOARROW_OK;
}

static struct ArrowIpcBufferFactory ArrowIpcBufferFactoryFromShared(
    struct ArrowIpcSharedBuffer* shared) {
  struct ArrowIpcBufferFactory out;
  out.make_buffer = &ArrowIpcMakeBufferFromShared;
  out.private_data = shared;
  return out;
}

// Just for the purposes of endian-swapping
struct ArrowIpcIntervalMonthDayNano {
  uint32_t months;
  uint32_t days;
  uint64_t ns;
};

static int ArrowIpcDecoderSwapEndian(struct ArrowIpcBufferSource* src,
                                     struct ArrowBufferView* out_view,
                                     struct ArrowBuffer* dst, struct ArrowError* error) {
  // Some buffer data types don't need any endian swapping
  switch (src->data_type) {
    case NANOARROW_TYPE_BOOL:
    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_BINARY:
      return NANOARROW_OK;
    default:
      break;
  }

  // Make sure dst is not a shared buffer that we can't modify
  struct ArrowBuffer tmp;
  ArrowBufferInit(&tmp);

  if (dst->allocator.private_data != NULL) {
    ArrowBufferMove(dst, &tmp);
    ArrowBufferInit(dst);
  }

  if (dst->size_bytes == 0) {
    NANOARROW_RETURN_NOT_OK(ArrowBufferReserve(dst, out_view->size_bytes));
    dst->size_bytes = out_view->size_bytes;
  }

  switch (src->data_type) {
    case NANOARROW_TYPE_DECIMAL128:
    case NANOARROW_TYPE_DECIMAL256: {
      const uint64_t* ptr_src = out_view->data.as_uint64;
      uint64_t* ptr_dst = (uint64_t*)dst->data;
      uint64_t words[4];
      int n_words = (int)(src->element_size_bits / 64);

      for (int64_t i = 0; i < (dst->size_bytes / n_words / 8); i++) {
        for (int j = 0; j < n_words; j++) {
          words[j] = bswap64(ptr_src[i * n_words + j]);
        }

        for (int j = 0; j < n_words; j++) {
          ptr_dst[i * n_words + j] = words[n_words - j - 1];
        }
      }
      break;
    }
    case NANOARROW_TYPE_INTERVAL_DAY_TIME: {
      uint32_t* ptr = (uint32_t*)dst->data;
      for (int64_t i = 0; i < (dst->size_bytes / 4); i++) {
        ptr[i] = bswap32(out_view->data.as_uint32[i]);
      }
      break;
    }
    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO: {
      const uint8_t* ptr_src = out_view->data.as_uint8;
      uint8_t* ptr_dst = dst->data;
      int item_size_bytes = 16;
      struct ArrowIpcIntervalMonthDayNano item;
      for (int64_t i = 0; i < (dst->size_bytes / item_size_bytes); i++) {
        memcpy(&item, ptr_src + i * item_size_bytes, item_size_bytes);
        item.months = bswap32(item.months);
        item.days = bswap32(item.days);
        item.ns = bswap64(item.ns);
        memcpy(ptr_dst + i * item_size_bytes, &item, item_size_bytes);
      }
      break;
    }
    default:
      switch (src->element_size_bits) {
        case 16: {
          uint16_t* ptr = (uint16_t*)dst->data;
          for (int64_t i = 0; i < (dst->size_bytes / 2); i++) {
            ptr[i] = bswap16(out_view->data.as_uint16[i]);
          }
          break;
        }
        case 32: {
          uint32_t* ptr = (uint32_t*)dst->data;
          for (int64_t i = 0; i < (dst->size_bytes / 4); i++) {
            ptr[i] = bswap32(out_view->data.as_uint32[i]);
          }
          break;
        }
        case 64: {
          uint64_t* ptr = (uint64_t*)dst->data;
          for (int64_t i = 0; i < (dst->size_bytes / 8); i++) {
            ptr[i] = bswap64(out_view->data.as_uint64[i]);
          }
          break;
        }
        default:
          ArrowErrorSet(error, "Endian swapping for element bitwidth %d is not supported",
                        (int)src->element_size_bits);
          return ENOTSUP;
      }
      break;
  }

  ArrowBufferReset(&tmp);
  out_view->data.data = dst->data;
  return NANOARROW_OK;
}

struct ArrowIpcArraySetter {
  ns(FieldNode_vec_t) fields;
  int64_t field_i;
  ns(Buffer_vec_t) buffers;
  int64_t buffer_i;
  int64_t body_size_bytes;
  struct ArrowIpcBufferSource src;
  struct ArrowIpcBufferFactory factory;
};

static int ArrowIpcDecoderMakeBuffer(struct ArrowIpcArraySetter* setter, int64_t offset,
                                     int64_t length, struct ArrowBufferView* out_view,
                                     struct ArrowBuffer* out, struct ArrowError* error) {
  out_view->data.data = NULL;
  out_view->size_bytes = 0;

  if (length == 0) {
    return NANOARROW_OK;
  }

  // Check that this buffer fits within the body
  int64_t buffer_start = offset;
  int64_t buffer_end = buffer_start + length;
  if (buffer_start < 0 || buffer_end > setter->body_size_bytes) {
    ArrowErrorSet(error, "Buffer requires body offsets [%ld..%ld) but body has size %ld",
                  (long)buffer_start, (long)buffer_end, (long)setter->body_size_bytes);
    return EINVAL;
  }

  // If the ArrowIpcBufferFactory is made public, these should get moved (since then a
  // user could inject support for either one). More likely, by the time that happens,
  // this library will be able to support some of these features.
  if (setter->src.codec != NANOARROW_IPC_COMPRESSION_TYPE_NONE) {
    ArrowErrorSet(error, "The nanoarrow_ipc extension does not support compression");
    return ENOTSUP;
  }

  setter->src.body_offset_bytes = offset;
  setter->src.buffer_length_bytes = length;
  NANOARROW_RETURN_NOT_OK(
      setter->factory.make_buffer(&setter->factory, &setter->src, out_view, out, error));

  if (setter->src.swap_endian) {
    NANOARROW_RETURN_NOT_OK(
        ArrowIpcDecoderSwapEndian(&setter->src, out_view, out, error));
  }

  return NANOARROW_OK;
}

static int ArrowIpcDecoderWalkGetArray(struct ArrowArrayView* array_view,
                                       struct ArrowArray* array, struct ArrowArray* out,
                                       struct ArrowError* error) {
  out->length = array_view->length;
  out->null_count = array_view->null_count;

  for (int64_t i = 0; i < array->n_buffers; i++) {
    struct ArrowBufferView view = array_view->buffer_views[i];
    struct ArrowBuffer* scratch_buffer = ArrowArrayBuffer(array, i);
    struct ArrowBuffer* buffer_out = ArrowArrayBuffer(out, i);

    // If the scratch buffer was used, move it to the final array. Otherwise,
    // copy the view.
    if (scratch_buffer->size_bytes == 0) {
      NANOARROW_RETURN_NOT_OK(ArrowBufferAppendBufferView(buffer_out, view));
    } else if (scratch_buffer->data == view.data.as_uint8) {
      ArrowBufferMove(scratch_buffer, buffer_out);
    } else {
      ArrowErrorSet(
          error,
          "Internal: scratch buffer was used but doesn't point to the same data as view");
      return EINVAL;
    }
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderWalkGetArray(
        array_view->children[i], array->children[i], out->children[i], error));
  }

  return NANOARROW_OK;
}

static int ArrowIpcDecoderWalkSetArrayView(struct ArrowIpcArraySetter* setter,
                                           struct ArrowArrayView* array_view,
                                           struct ArrowArray* array,
                                           struct ArrowError* error) {
  ns(FieldNode_struct_t) field =
      ns(FieldNode_vec_at(setter->fields, (size_t)setter->field_i));
  array_view->length = ns(FieldNode_length(field));
  array_view->null_count = ns(FieldNode_null_count(field));
  setter->field_i += 1;

  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    if (array_view->layout.buffer_type[i] == NANOARROW_BUFFER_TYPE_NONE) {
      break;
    }

    ns(Buffer_struct_t) buffer =
        ns(Buffer_vec_at(setter->buffers, (size_t)setter->buffer_i));
    int64_t buffer_offset = ns(Buffer_offset(buffer));
    int64_t buffer_length = ns(Buffer_length(buffer));
    setter->buffer_i += 1;

    // Provide a buffer that will be used if any allocation has to occur
    struct ArrowBuffer* buffer_dst = ArrowArrayBuffer(array, i);

    // Attempt to re-use any previous allocation unless this buffer is
    // wrapping a custom allocator.
    if (buffer_dst->allocator.private_data != NULL) {
      ArrowBufferReset(buffer_dst);
    } else {
      buffer_dst->size_bytes = 0;
    }

    setter->src.data_type = array_view->layout.buffer_data_type[i];
    setter->src.element_size_bits = array_view->layout.element_size_bits[i];

    NANOARROW_RETURN_NOT_OK(
        ArrowIpcDecoderMakeBuffer(setter, buffer_offset, buffer_length,
                                  &array_view->buffer_views[i], buffer_dst, error));
  }

  for (int64_t i = 0; i < array_view->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderWalkSetArrayView(
        setter, array_view->children[i], array->children[i], error));
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowIpcDecoderDecodeArrayInternal(
    struct ArrowIpcDecoder* decoder, int64_t field_i, struct ArrowArray* out,
    enum ArrowValidationLevel validation_level, struct ArrowError* error) {
  struct ArrowIpcDecoderPrivate* private_data =
      (struct ArrowIpcDecoderPrivate*)decoder->private_data;

  struct ArrowIpcField* root = private_data->fields + field_i + 1;

  if (field_i == -1) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayInitFromArrayView(out, &private_data->array_view, error));
    out->length = private_data->array_view.length;
    out->null_count = private_data->array_view.null_count;

    for (int64_t i = 0; i < private_data->array_view.n_children; i++) {
      NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderWalkGetArray(
          private_data->array_view.children[i], private_data->array.children[i],
          out->children[i], error));
    }

  } else {
    NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromArrayView(out, root->array_view, error));
    NANOARROW_RETURN_NOT_OK(
        ArrowIpcDecoderWalkGetArray(root->array_view, root->array, out, error));
  }

  // If validation is going to happen it has already occurred; however, the part of
  // ArrowArrayFinishBuilding() that allocates a data buffer if the data buffer is
  // NULL (required for compatibility with Arrow <= 9.0.0) assumes CPU data access
  // and thus needs a validation level >= default.
  if (validation_level >= NANOARROW_VALIDATION_LEVEL_DEFAULT) {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayFinishBuilding(out, NANOARROW_VALIDATION_LEVEL_DEFAULT, error));
  } else {
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayFinishBuilding(out, NANOARROW_VALIDATION_LEVEL_NONE, error));
  }

  return NANOARROW_OK;
}

static ArrowErrorCode ArrowIpcDecoderDecodeArrayViewInternal(
    struct ArrowIpcDecoder* decoder, struct ArrowIpcBufferFactory factory,
    int64_t field_i, struct ArrowArrayView** out_view, struct ArrowError* error) {
  struct ArrowIpcDecoderPrivate* private_data =
      (struct ArrowIpcDecoderPrivate*)decoder->private_data;

  if (private_data->last_message == NULL ||
      decoder->message_type != NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH) {
    ArrowErrorSet(error, "decoder did not just decode a RecordBatch message");
    return EINVAL;
  }

  ns(RecordBatch_table_t) batch = (ns(RecordBatch_table_t))private_data->last_message;

  // RecordBatch messages don't count the root node but decoder->fields does
  struct ArrowIpcField* root = private_data->fields + field_i + 1;

  struct ArrowIpcArraySetter setter;
  setter.fields = ns(RecordBatch_nodes(batch));
  setter.field_i = field_i;
  setter.buffers = ns(RecordBatch_buffers(batch));
  setter.buffer_i = root->buffer_offset - 1;
  setter.body_size_bytes = decoder->body_size_bytes;
  setter.factory = factory;
  setter.src.codec = decoder->codec;
  setter.src.swap_endian = ArrowIpcDecoderNeedsSwapEndian(decoder);

  // The flatbuffers FieldNode doesn't count the root struct so we have to loop over the
  // children ourselves
  if (field_i == -1) {
    root->array_view->length = ns(RecordBatch_length(batch));
    root->array_view->null_count = 0;
    setter.field_i++;
    setter.buffer_i++;

    for (int64_t i = 0; i < root->array_view->n_children; i++) {
      NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderWalkSetArrayView(
          &setter, root->array_view->children[i], root->array->children[i], error));
    }
  } else {
    NANOARROW_RETURN_NOT_OK(
        ArrowIpcDecoderWalkSetArrayView(&setter, root->array_view, root->array, error));
  }

  *out_view = root->array_view;
  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcDecoderDecodeArrayView(struct ArrowIpcDecoder* decoder,
                                              struct ArrowBufferView body, int64_t i,
                                              struct ArrowArrayView** out,
                                              struct ArrowError* error) {
  return ArrowIpcDecoderDecodeArrayViewInternal(
      decoder, ArrowIpcBufferFactoryFromView(&body), i, out, error);
}

ArrowErrorCode ArrowIpcDecoderDecodeArray(struct ArrowIpcDecoder* decoder,
                                          struct ArrowBufferView body, int64_t i,
                                          struct ArrowArray* out,
                                          enum ArrowValidationLevel validation_level,
                                          struct ArrowError* error) {
  struct ArrowArrayView* array_view;
  NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderDecodeArrayViewInternal(
      decoder, ArrowIpcBufferFactoryFromView(&body), i, &array_view, error));

  NANOARROW_RETURN_NOT_OK(ArrowArrayViewValidate(array_view, validation_level, error));

  struct ArrowArray temp;
  temp.release = NULL;
  int result =
      ArrowIpcDecoderDecodeArrayInternal(decoder, i, &temp, validation_level, error);
  if (result != NANOARROW_OK && temp.release != NULL) {
    ArrowArrayRelease(&temp);
  } else if (result != NANOARROW_OK) {
    return result;
  }

  ArrowArrayMove(&temp, out);
  return NANOARROW_OK;
}

ArrowErrorCode ArrowIpcDecoderDecodeArrayFromShared(
    struct ArrowIpcDecoder* decoder, struct ArrowIpcSharedBuffer* body, int64_t i,
    struct ArrowArray* out, enum ArrowValidationLevel validation_level,
    struct ArrowError* error) {
  struct ArrowArrayView* array_view;
  NANOARROW_RETURN_NOT_OK(ArrowIpcDecoderDecodeArrayViewInternal(
      decoder, ArrowIpcBufferFactoryFromShared(body), i, &array_view, error));

  NANOARROW_RETURN_NOT_OK(ArrowArrayViewValidate(array_view, validation_level, error));

  struct ArrowArray temp;
  temp.release = NULL;
  int result =
      ArrowIpcDecoderDecodeArrayInternal(decoder, i, &temp, validation_level, error);
  if (result != NANOARROW_OK && temp.release != NULL) {
    ArrowArrayRelease(&temp);
  } else if (result != NANOARROW_OK) {
    return result;
  }

  ArrowArrayMove(&temp, out);
  return NANOARROW_OK;
}
