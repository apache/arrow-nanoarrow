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

#include "nlohmann/json.hpp"

#include "nanoarrow/nanoarrow_testing.hpp"

namespace nanoarrow {

namespace testing {

namespace writer_internal {

namespace {

class LocalizedStream {
 public:
  LocalizedStream(std::ostream& out) : out_(out) {
    previous_locale_ = out.imbue(std::locale::classic());
    previous_precision_ = out.precision();
    fmt_flags_ = out.flags();
    out.setf(out.fixed);
  }

  void SetFixed(int precision) { out_.precision(precision); }

  ~LocalizedStream() {
    out_.flags(fmt_flags_);
    out_.precision(previous_precision_);
    out_.imbue(previous_locale_);
  }

 private:
  std::ostream& out_;
  std::locale previous_locale_;
  std::ios::fmtflags fmt_flags_;
  std::streamsize previous_precision_;
};

void WriteString(std::ostream& out, ArrowStringView value) {
  std::string value_str(value.data, static_cast<size_t>(value.size_bytes));
  out << nlohmann::json(value_str);
}

void WriteBytes(std::ostream& out, ArrowBufferView value) {
  out << R"(")";
  char hex[3];
  hex[2] = '\0';

  for (int64_t i = 0; i < value.size_bytes; i++) {
    snprintf(hex, sizeof(hex), "%02X", static_cast<int>(value.data.as_uint8[i]));
    out << hex;
  }
  out << R"(")";
}

void WriteBitmap(std::ostream& out, const uint8_t* bits, int64_t length) {
  if (length == 0) {
    out << "[]";
    return;
  }

  out << "[";

  if (bits == nullptr) {
    out << "1";
    for (int64_t i = 1; i < length; i++) {
      out << ", 1";
    }
  } else {
    out << static_cast<int32_t>(ArrowBitGet(bits, 0));
    for (int64_t i = 1; i < length; i++) {
      out << ", " << static_cast<int32_t>(ArrowBitGet(bits, i));
    }
  }

  out << "]";
}

template <typename T>
ArrowErrorCode WriteOffsetOrTypeID(std::ostream& out, ArrowBufferView content) {
  if (content.size_bytes == 0) {
    out << "[]";
    return NANOARROW_OK;
  }

  const T* values = reinterpret_cast<const T*>(content.data.data);
  int64_t n_values = content.size_bytes / sizeof(T);

  out << "[";

  if (sizeof(T) == sizeof(int64_t)) {
    // Ensure int64s are quoted (i.e, "123456")
    out << R"(")" << values[0] << R"(")";
    for (int64_t i = 1; i < n_values; i++) {
      out << R"(, ")" << values[i] << R"(")";
    }
  } else {
    // No need to quote smaller ints (i.e., 123456)
    out << static_cast<int64_t>(values[0]);
    for (int64_t i = 1; i < n_values; i++) {
      out << ", " << static_cast<int64_t>(values[i]);
    }
  }

  out << "]";
  return NANOARROW_OK;
}

void WriteIntMaybeNull(std::ostream& out, const ArrowArrayView* view, int64_t i) {
  if (ArrowArrayViewIsNull(view, i)) {
    out << 0;
  } else {
    out << ArrowArrayViewGetIntUnsafe(view, i);
  }
}

void WriteQuotedIntMaybeNull(std::ostream& out, const ArrowArrayView* view, int64_t i) {
  if (ArrowArrayViewIsNull(view, i)) {
    out << R"("0")";
  } else {
    out << R"(")" << ArrowArrayViewGetIntUnsafe(view, i) << R"(")";
  }
}

void WriteQuotedUIntMaybeNull(std::ostream& out, const ArrowArrayView* view, int64_t i) {
  if (ArrowArrayViewIsNull(view, i)) {
    out << R"("0")";
  } else {
    out << R"(")" << ArrowArrayViewGetUIntUnsafe(view, i) << R"(")";
  }
}

void WriteFloatMaybeNull(std::ostream& out, const ArrowArrayView* view, int64_t i,
                         int float_precision) {
  if (float_precision >= 0) {
    if (ArrowArrayViewIsNull(view, i)) {
      out << static_cast<double>(0);
    } else {
      out << ArrowArrayViewGetDoubleUnsafe(view, i);
    }
  } else {
    if (ArrowArrayViewIsNull(view, i)) {
      out << "0.0";
    } else {
      out << nlohmann::json(ArrowArrayViewGetDoubleUnsafe(view, i));
    }
  }
}

void WriteBytesMaybeNull(std::ostream& out, const ArrowArrayView* view, int64_t i) {
  ArrowBufferView item = ArrowArrayViewGetBytesUnsafe(view, i);
  if (ArrowArrayViewIsNull(view, i)) {
    out << R"(")";
    for (int64_t i = 0; i < item.size_bytes; i++) {
      out << "00";
    }
    out << R"(")";
  } else {
    WriteBytes(out, item);
  }
}

void WriteIntervalDayTimeMaybeNull(std::ostream& out, const ArrowArrayView* view,
                                   int64_t i, ArrowInterval* interval) {
  if (ArrowArrayViewIsNull(view, i)) {
    out << R"({"days": 0, "milliseconds": 0})";
  } else {
    ArrowArrayViewGetIntervalUnsafe(view, i, interval);
    out << R"({"days": )" << interval->days << R"(, "milliseconds": )" << interval->ms
        << "}";
  }
}

void WriteIntervalMonthDayNanoMaybeNull(std::ostream& out, const ArrowArrayView* view,
                                        int64_t i, ArrowInterval* interval) {
  if (ArrowArrayViewIsNull(view, i)) {
    out << R"({"months": 0, "days": 0, "nanoseconds": "0"})";
  } else {
    ArrowArrayViewGetIntervalUnsafe(view, i, interval);
    out << R"({"months": )" << interval->months << R"(, "days": )" << interval->days
        << R"(, "nanoseconds": ")" << interval->ns << R"("})";
  }
}

ArrowErrorCode WriteDecimalMaybeNull(std::ostream& out, const ArrowArrayView* view,
                                     int64_t i, ArrowDecimal* decimal, ArrowBuffer* tmp) {
  if (ArrowArrayViewIsNull(view, i)) {
    out << R"("0")";
    return NANOARROW_OK;
  } else {
    ArrowArrayViewGetDecimalUnsafe(view, i, decimal);
    tmp->size_bytes = 0;
    NANOARROW_RETURN_NOT_OK(ArrowDecimalAppendDigitsToBuffer(decimal, tmp));
    out << R"(")" << std::string(reinterpret_cast<char*>(tmp->data), tmp->size_bytes)
        << R"(")";
    return NANOARROW_OK;
  }
}

ArrowErrorCode WriteDecimalData(std::ostream& out, const ArrowArrayView* view,
                                int bitwidth) {
  ArrowDecimal value;
  ArrowDecimalInit(&value, bitwidth, 0, 0);
  nanoarrow::UniqueBuffer tmp;

  NANOARROW_RETURN_NOT_OK(WriteDecimalMaybeNull(out, view, 0, &value, tmp.get()));
  for (int64_t i = 1; i < view->length; i++) {
    out << ", ";
    NANOARROW_RETURN_NOT_OK(WriteDecimalMaybeNull(out, view, i, &value, tmp.get()));
  }

  return NANOARROW_OK;
}

ArrowErrorCode WriteData(std::ostream& out, const ArrowArrayView* value,
                         int float_precision) {
  if (value->length == 0) {
    out << "[]";
    return NANOARROW_OK;
  }

  out << "[";

  switch (value->storage_type) {
    case NANOARROW_TYPE_BOOL:
    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INTERVAL_MONTHS:
      // Regular JSON integers (i.e., 123456)
      WriteIntMaybeNull(out, value, 0);
      for (int64_t i = 1; i < value->length; i++) {
        out << ", ";
        WriteIntMaybeNull(out, value, i);
      }
      break;
    case NANOARROW_TYPE_INT64:
      // Quoted integers to avoid overflow (i.e., "123456")
      WriteQuotedIntMaybeNull(out, value, 0);
      for (int64_t i = 1; i < value->length; i++) {
        out << ", ";
        WriteQuotedIntMaybeNull(out, value, i);
      }
      break;
    case NANOARROW_TYPE_UINT64:
      // Quoted integers to avoid overflow (i.e., "123456")
      WriteQuotedUIntMaybeNull(out, value, 0);
      for (int64_t i = 1; i < value->length; i++) {
        out << ", ";
        WriteQuotedUIntMaybeNull(out, value, i);
      }
      break;

    case NANOARROW_TYPE_HALF_FLOAT:
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_DOUBLE: {
      // JSON number to float_precision_ decimal places
      LocalizedStream local_stream_opt(out);
      local_stream_opt.SetFixed(float_precision);

      WriteFloatMaybeNull(out, value, 0, float_precision);
      for (int64_t i = 1; i < value->length; i++) {
        out << ", ";
        WriteFloatMaybeNull(out, value, i, float_precision);
      }
      break;
    }

    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
      WriteString(out, ArrowArrayViewGetStringUnsafe(value, 0));
      for (int64_t i = 1; i < value->length; i++) {
        out << ", ";
        WriteString(out, ArrowArrayViewGetStringUnsafe(value, i));
      }
      break;

    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_LARGE_BINARY:
    case NANOARROW_TYPE_FIXED_SIZE_BINARY: {
      WriteBytesMaybeNull(out, value, 0);
      for (int64_t i = 1; i < value->length; i++) {
        out << ", ";
        WriteBytesMaybeNull(out, value, i);
      }
      break;
    }

    case NANOARROW_TYPE_INTERVAL_DAY_TIME: {
      ArrowInterval interval;
      ArrowIntervalInit(&interval, value->storage_type);
      WriteIntervalDayTimeMaybeNull(out, value, 0, &interval);
      for (int64_t i = 1; i < value->length; i++) {
        out << ", ";
        WriteIntervalDayTimeMaybeNull(out, value, i, &interval);
      }
      break;
    }

    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO: {
      ArrowInterval interval;
      ArrowIntervalInit(&interval, value->storage_type);
      WriteIntervalMonthDayNanoMaybeNull(out, value, 0, &interval);
      for (int64_t i = 1; i < value->length; i++) {
        out << ", ";
        WriteIntervalMonthDayNanoMaybeNull(out, value, i, &interval);
      }
      break;
    }

    case NANOARROW_TYPE_DECIMAL128:
      NANOARROW_RETURN_NOT_OK(WriteDecimalData(out, value, 128));
      break;
    case NANOARROW_TYPE_DECIMAL256:
      NANOARROW_RETURN_NOT_OK(WriteDecimalData(out, value, 256));
      break;

    default:
      // Not supported
      return ENOTSUP;
  }

  out << "]";
  return NANOARROW_OK;
}

}  // namespace

namespace {

ArrowErrorCode WriteTimeUnit(std::ostream& out, const ArrowSchemaView* field) {
  switch (field->time_unit) {
    case NANOARROW_TIME_UNIT_NANO:
      out << R"(, "unit": "NANOSECOND")";
      return NANOARROW_OK;
    case NANOARROW_TIME_UNIT_MICRO:
      out << R"(, "unit": "MICROSECOND")";
      return NANOARROW_OK;
    case NANOARROW_TIME_UNIT_MILLI:
      out << R"(, "unit": "MILLISECOND")";
      return NANOARROW_OK;
    case NANOARROW_TIME_UNIT_SECOND:
      out << R"(, "unit": "SECOND")";
      return NANOARROW_OK;
    default:
      return EINVAL;
  }
}

ArrowErrorCode WriteTypeFromView(std::ostream& out, const ArrowSchemaView* field) {
  out << "{";

  switch (field->type) {
    case NANOARROW_TYPE_NA:
      out << R"("name": "null")";
      break;
    case NANOARROW_TYPE_BOOL:
      out << R"("name": "bool")";
      break;
    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_INT32:
    case NANOARROW_TYPE_INT64:
      out << R"("name": "int", "bitWidth": )" << field->layout.element_size_bits[1]
          << R"(, "isSigned": true)";
      break;
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_UINT32:
      out << R"("name": "int", "bitWidth": )" << field->layout.element_size_bits[1]
          << R"(, "isSigned": false)";
      break;
    case NANOARROW_TYPE_HALF_FLOAT:
      out << R"("name": "floatingpoint", "precision": "HALF")";
      break;
    case NANOARROW_TYPE_FLOAT:
      out << R"("name": "floatingpoint", "precision": "SINGLE")";
      break;
    case NANOARROW_TYPE_DOUBLE:
      out << R"("name": "floatingpoint", "precision": "DOUBLE")";
      break;
    case NANOARROW_TYPE_STRING:
      out << R"("name": "utf8")";
      break;
    case NANOARROW_TYPE_LARGE_STRING:
      out << R"("name": "largeutf8")";
      break;
    case NANOARROW_TYPE_BINARY:
      out << R"("name": "binary")";
      break;
    case NANOARROW_TYPE_LARGE_BINARY:
      out << R"("name": "largebinary")";
      break;
    case NANOARROW_TYPE_FIXED_SIZE_BINARY:
      out << R"("name": "fixedsizebinary", "byteWidth": )" << field->fixed_size;
      break;
    case NANOARROW_TYPE_DECIMAL128:
    case NANOARROW_TYPE_DECIMAL256:
      out << R"("name": "decimal", "bitWidth": )" << field->decimal_bitwidth
          << R"(, "precision": )" << field->decimal_precision << R"(, "scale": )"
          << field->decimal_scale;
      break;
    case NANOARROW_TYPE_DURATION:
      out << R"("name": "duration")";
      NANOARROW_RETURN_NOT_OK(WriteTimeUnit(out, field));
      break;
    case NANOARROW_TYPE_DATE32:
      out << R"("name": "date", "unit": "DAY")";
      break;
    case NANOARROW_TYPE_DATE64:
      out << R"("name": "date", "unit": "MILLISECOND")";
      break;
    case NANOARROW_TYPE_TIME32:
      out << R"("name": "time")";
      NANOARROW_RETURN_NOT_OK(WriteTimeUnit(out, field));
      out << R"(, "bitWidth": 32)";
      break;
    case NANOARROW_TYPE_TIME64:
      out << R"("name": "time")";
      NANOARROW_RETURN_NOT_OK(WriteTimeUnit(out, field));
      out << R"(, "bitWidth": 64)";
      break;
    case NANOARROW_TYPE_TIMESTAMP:
      out << R"("name": "timestamp")";
      NANOARROW_RETURN_NOT_OK(WriteTimeUnit(out, field));
      if (strlen(field->timezone) > 0) {
        out << R"(, "timezone": )";
        WriteString(out, ArrowCharView(field->timezone));
      }
      break;
    case NANOARROW_TYPE_INTERVAL_MONTHS:
      out << R"("name": "interval", "unit": "YEAR_MONTH")";
      break;
    case NANOARROW_TYPE_INTERVAL_DAY_TIME:
      out << R"("name": "interval", "unit": "DAY_TIME")";
      break;
    case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO:
      out << R"("name": "interval", "unit": "MONTH_DAY_NANO")";
      break;
    case NANOARROW_TYPE_STRUCT:
      out << R"("name": "struct")";
      break;
    case NANOARROW_TYPE_LIST:
      out << R"("name": "list")";
      break;
    case NANOARROW_TYPE_MAP:
      out << R"("name": "map", "keysSorted": )";
      if (field->schema->flags & ARROW_FLAG_MAP_KEYS_SORTED) {
        out << "true";
      } else {
        out << "false";
      }
      break;
    case NANOARROW_TYPE_LARGE_LIST:
      out << R"("name": "largelist")";
      break;
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
      out << R"("name": "fixedsizelist", "listSize": )"
          << field->layout.child_size_elements;
      break;
    case NANOARROW_TYPE_DENSE_UNION:
      out << R"("name": "union", "mode": "DENSE", "typeIds": [)" << field->union_type_ids
          << "]";
      break;
    case NANOARROW_TYPE_SPARSE_UNION:
      out << R"("name": "union", "mode": "SPARSE", "typeIds": [)" << field->union_type_ids
          << "]";
      break;

    default:
      // Not supported
      return ENOTSUP;
  }

  out << "}";
  return NANOARROW_OK;
}

ArrowErrorCode WriteMetadataItem(std::ostream& out, ArrowMetadataReader* reader) {
  ArrowStringView key;
  ArrowStringView value;
  NANOARROW_RETURN_NOT_OK(ArrowMetadataReaderRead(reader, &key, &value));
  out << R"({"key": )";
  WriteString(out, key);
  out << R"(, "value": )";
  WriteString(out, value);
  out << "}";
  return NANOARROW_OK;
}

}  // namespace

}  // namespace writer_internal

ArrowErrorCode TestingJSONWriter::WriteDataFile(std::ostream& out,
                                                ArrowArrayStream* stream) {
  if (stream == nullptr || stream->release == nullptr) {
    return EINVAL;
  }

  ResetDictionaries();

  out << R"({"schema": )";

  nanoarrow::UniqueSchema schema;
  NANOARROW_RETURN_NOT_OK(ArrowArrayStreamGetSchema(stream, schema.get(), nullptr));
  NANOARROW_RETURN_NOT_OK(WriteSchema(out, schema.get()));

  nanoarrow::UniqueArrayView array_view;
  NANOARROW_RETURN_NOT_OK(
      ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), nullptr));

  out << R"(, "batches": [)";

  nanoarrow::UniqueArray array;
  std::string sep;
  do {
    NANOARROW_RETURN_NOT_OK(ArrowArrayStreamGetNext(stream, array.get(), nullptr));
    if (array->release == nullptr) {
      break;
    }

    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewSetArray(array_view.get(), array.get(), nullptr));

    out << sep;
    sep = ", ";
    NANOARROW_RETURN_NOT_OK(WriteBatch(out, schema.get(), array_view.get()));
    array.reset();
  } while (true);

  out << "]";

  if (!dictionaries_.empty()) {
    out << R"(, "dictionaries": )";
    NANOARROW_RETURN_NOT_OK(WriteDictionaryBatches(out));
  }

  out << "}";

  return NANOARROW_OK;
}

ArrowErrorCode TestingJSONWriter::WriteSchema(std::ostream& out,
                                              const ArrowSchema* schema) {
  // Make sure we have a struct
  if (std::string(schema->format) != "+s") {
    return EINVAL;
  }

  out << "{";

  // Write fields
  out << R"("fields": )";
  if (schema->n_children == 0) {
    out << "[]";
  } else {
    out << "[";
    NANOARROW_RETURN_NOT_OK(WriteField(out, schema->children[0]));
    for (int64_t i = 1; i < schema->n_children; i++) {
      out << ", ";
      NANOARROW_RETURN_NOT_OK(WriteField(out, schema->children[i]));
    }
    out << "]";
  }

  // Write metadata
  if (ShouldWriteMetadata(schema->metadata)) {
    out << R"(, "metadata": )";
    NANOARROW_RETURN_NOT_OK(WriteMetadata(out, schema->metadata));
  }

  out << "}";
  return NANOARROW_OK;
}

ArrowErrorCode TestingJSONWriter::WriteField(std::ostream& out,
                                             const ArrowSchema* field) {
  ArrowSchemaView view;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&view, field, nullptr));

  out << "{";

  // Write schema->name (may be null)
  if (field->name == nullptr) {
    out << R"("name": null)";
  } else {
    out << R"("name": )";
    writer_internal::WriteString(out, ArrowCharView(field->name));
  }

  // Write nullability
  if (field->flags & ARROW_FLAG_NULLABLE) {
    out << R"(, "nullable": true)";
  } else {
    out << R"(, "nullable": false)";
  }

  // For dictionary encoding, write type as the dictionary (values) type,
  // record the dictionary schema, and write the "dictionary" member
  if (field->dictionary != nullptr) {
    ArrowSchemaView dictionary_view;
    NANOARROW_RETURN_NOT_OK(
        ArrowSchemaViewInit(&dictionary_view, field->dictionary, nullptr));

    out << R"(, "type": )";
    NANOARROW_RETURN_NOT_OK(writer_internal::WriteTypeFromView(out, &dictionary_view));

    int32_t dictionary_id;
    NANOARROW_RETURN_NOT_OK(
        dictionaries_.RecordSchema(field->dictionary, &dictionary_id));

    out << R"(, "dictionary": )";
    view.type = view.storage_type;
    NANOARROW_RETURN_NOT_OK(WriteFieldDictionary(
        out, dictionary_id, field->flags & ARROW_FLAG_DICTIONARY_ORDERED, &view));

    // Write dictionary children
    out << R"(, "children": )";
    NANOARROW_RETURN_NOT_OK(WriteFieldChildren(out, field->dictionary));
  } else {
    // Write non-dictionary type/children
    out << R"(, "type": )";
    NANOARROW_RETURN_NOT_OK(writer_internal::WriteTypeFromView(out, &view));

    // Write children
    out << R"(, "children": )";
    NANOARROW_RETURN_NOT_OK(WriteFieldChildren(out, field));
  }

  // Write metadata
  if (ShouldWriteMetadata(field->metadata)) {
    out << R"(, "metadata": )";
    NANOARROW_RETURN_NOT_OK(WriteMetadata(out, field->metadata));
  }

  out << "}";
  return NANOARROW_OK;
}

ArrowErrorCode TestingJSONWriter::WriteType(std::ostream& out, const ArrowSchema* field) {
  ArrowSchemaView view;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&view, (ArrowSchema*)field, nullptr));
  NANOARROW_RETURN_NOT_OK(writer_internal::WriteTypeFromView(out, &view));
  return NANOARROW_OK;
}

ArrowErrorCode TestingJSONWriter::WriteMetadata(std::ostream& out, const char* metadata) {
  if (metadata == nullptr) {
    out << "null";
    return NANOARROW_OK;
  }

  ArrowMetadataReader reader;
  NANOARROW_RETURN_NOT_OK(ArrowMetadataReaderInit(&reader, metadata));
  if (reader.remaining_keys == 0) {
    out << "[]";
    return NANOARROW_OK;
  }

  out << "[";
  NANOARROW_RETURN_NOT_OK(writer_internal::WriteMetadataItem(out, &reader));
  while (reader.remaining_keys > 0) {
    out << ", ";
    NANOARROW_RETURN_NOT_OK(writer_internal::WriteMetadataItem(out, &reader));
  }

  out << "]";
  return NANOARROW_OK;
}

ArrowErrorCode TestingJSONWriter::WriteBatch(std::ostream& out, const ArrowSchema* schema,
                                             const ArrowArrayView* value) {
  // Make sure we have a struct
  if (std::string(schema->format) != "+s") {
    return EINVAL;
  }

  out << "{";

  // Write length
  out << R"("count": )" << value->length;

  // Write children
  out << R"(, "columns": )";
  NANOARROW_RETURN_NOT_OK(WriteChildren(out, schema, value));

  out << "}";
  return NANOARROW_OK;
}

ArrowErrorCode TestingJSONWriter::WriteColumn(std::ostream& out, const ArrowSchema* field,
                                              const ArrowArrayView* value) {
  out << "{";

  // Write schema->name (may be null)
  if (field->name == nullptr) {
    out << R"("name": null)";
  } else {
    out << R"("name": )";
    writer_internal::WriteString(out, ArrowCharView(field->name));
  }

  // Write length
  out << R"(, "count": )" << value->length;

  // Write the VALIDITY element if required
  switch (value->storage_type) {
    case NANOARROW_TYPE_NA:
    case NANOARROW_TYPE_DENSE_UNION:
    case NANOARROW_TYPE_SPARSE_UNION:
      break;
    default:
      out << R"(, "VALIDITY": )";
      writer_internal::WriteBitmap(out, value->buffer_views[0].data.as_uint8,
                                   value->length);
      break;
  }

  // Write the TYPE_ID element if required
  switch (value->storage_type) {
    case NANOARROW_TYPE_SPARSE_UNION:
    case NANOARROW_TYPE_DENSE_UNION:
      out << R"(, "TYPE_ID": )";
      NANOARROW_RETURN_NOT_OK(
          writer_internal::WriteOffsetOrTypeID<int8_t>(out, value->buffer_views[0]));
      break;
    default:
      break;
  }

  // Write the OFFSET element if required
  switch (value->storage_type) {
    case NANOARROW_TYPE_BINARY:
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_DENSE_UNION:
    case NANOARROW_TYPE_LIST:
      out << R"(, "OFFSET": )";
      NANOARROW_RETURN_NOT_OK(
          writer_internal::WriteOffsetOrTypeID<int32_t>(out, value->buffer_views[1]));
      break;
    case NANOARROW_TYPE_LARGE_LIST:
    case NANOARROW_TYPE_LARGE_BINARY:
    case NANOARROW_TYPE_LARGE_STRING:
      out << R"(, "OFFSET": )";
      NANOARROW_RETURN_NOT_OK(
          writer_internal::WriteOffsetOrTypeID<int64_t>(out, value->buffer_views[1]));
      break;
    default:
      break;
  }

  // Write the DATA element if required
  switch (value->storage_type) {
    case NANOARROW_TYPE_NA:
    case NANOARROW_TYPE_STRUCT:
    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_LARGE_LIST:
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
    case NANOARROW_TYPE_MAP:
    case NANOARROW_TYPE_DENSE_UNION:
    case NANOARROW_TYPE_SPARSE_UNION:
      break;
    default:
      out << R"(, "DATA": )";
      NANOARROW_RETURN_NOT_OK(writer_internal::WriteData(out, value, float_precision_));
      break;
  }

  switch (value->storage_type) {
    case NANOARROW_TYPE_STRUCT:
    case NANOARROW_TYPE_LIST:
    case NANOARROW_TYPE_LARGE_LIST:
    case NANOARROW_TYPE_FIXED_SIZE_LIST:
    case NANOARROW_TYPE_DENSE_UNION:
    case NANOARROW_TYPE_SPARSE_UNION:
      out << R"(, "children": )";
      NANOARROW_RETURN_NOT_OK(WriteChildren(out, field, value));
      break;
    default:
      break;
  }

  out << "}";

  // Write the dictionary values to the DictionaryContext for later if applicable
  if (field->dictionary != nullptr) {
    if (!dictionaries_.HasDictionaryForSchema(field->dictionary)) {
      return EINVAL;
    }

    std::stringstream dictionary_output;
    NANOARROW_RETURN_NOT_OK(
        WriteColumn(dictionary_output, field->dictionary, value->dictionary));
    dictionaries_.RecordArray(field->dictionary, value->dictionary->length,
                              dictionary_output.str());
  }

  return NANOARROW_OK;
}

ArrowErrorCode TestingJSONWriter::WriteDictionaryBatches(std::ostream& out) {
  std::vector<int32_t> ids = dictionaries_.GetAllIds();
  if (ids.empty()) {
    out << "[]";
    return NANOARROW_OK;
  }

  out << "[";
  std::sort(ids.begin(), ids.end());
  NANOARROW_RETURN_NOT_OK(WriteDictionaryBatch(out, ids[0]));
  for (size_t i = 1; i < ids.size(); i++) {
    out << ", ";
    NANOARROW_RETURN_NOT_OK(WriteDictionaryBatch(out, ids[i]));
  }
  out << "]";

  return NANOARROW_OK;
}

ArrowErrorCode TestingJSONWriter::WriteDictionaryBatch(std::ostream& out,
                                                       int32_t dictionary_id) {
  const internal::Dictionary& dict = dictionaries_.Get(dictionary_id);
  out << R"({"id": )" << dictionary_id << R"(, "data": {"count": )" << dict.column_length
      << R"(, "columns": [)" << dict.column_json << "]}}";
  return NANOARROW_OK;
}

ArrowErrorCode TestingJSONWriter::WriteFieldChildren(std::ostream& out,
                                                     const ArrowSchema* field) {
  if (field->n_children == 0) {
    out << "[]";
  } else {
    out << "[";
    NANOARROW_RETURN_NOT_OK(WriteField(out, field->children[0]));
    for (int64_t i = 1; i < field->n_children; i++) {
      out << ", ";
      NANOARROW_RETURN_NOT_OK(WriteField(out, field->children[i]));
    }
    out << "]";
  }

  return NANOARROW_OK;
}
ArrowErrorCode TestingJSONWriter::WriteFieldDictionary(
    std::ostream& out, int32_t dictionary_id, bool is_ordered,
    const ArrowSchemaView* indices_field) {
  out << "{";

  out << R"("id": )" << dictionary_id;

  out << R"(, "indexType": )";
  NANOARROW_RETURN_NOT_OK(writer_internal::WriteTypeFromView(out, indices_field));

  if (is_ordered) {
    out << R"(, "isOrdered": true)";
  } else {
    out << R"(, "isOrdered": false)";
  }

  out << "}";
  return NANOARROW_OK;
}

ArrowErrorCode TestingJSONWriter::WriteChildren(std::ostream& out,
                                                const ArrowSchema* field,
                                                const ArrowArrayView* value) {
  if (field->n_children == 0) {
    out << "[]";
    return NANOARROW_OK;
  }

  out << "[";
  NANOARROW_RETURN_NOT_OK(WriteColumn(out, field->children[0], value->children[0]));
  for (int64_t i = 1; i < field->n_children; i++) {
    out << ", ";
    NANOARROW_RETURN_NOT_OK(WriteColumn(out, field->children[i], value->children[i]));
  }
  out << "]";
  return NANOARROW_OK;
}

namespace reader_internal {

namespace {

using nlohmann::json;

ArrowErrorCode Check(bool value, ArrowError* error, const std::string& err) {
  if (value) {
    return NANOARROW_OK;
  } else {
    ArrowErrorSet(error, "%s", err.c_str());
    return EINVAL;
  }
}

ArrowErrorCode PrefixError(ArrowErrorCode value, ArrowError* error,
                           const std::string& prefix) {
  if (value != NANOARROW_OK && error != nullptr) {
    std::string msg = prefix + error->message;
    ArrowErrorSet(error, "%s", msg.c_str());
  }

  return value;
}

ArrowErrorCode SetTypeInt(ArrowSchema* schema, const json& value, ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(Check(value.contains("bitWidth"), error,
                                "Type[name=='int'] missing key 'bitWidth'"));
  NANOARROW_RETURN_NOT_OK(Check(value.contains("isSigned"), error,
                                "Type[name=='int'] missing key 'isSigned'"));

  const auto& bitwidth = value["bitWidth"];
  NANOARROW_RETURN_NOT_OK(Check(bitwidth.is_number_integer(), error,
                                "Type[name=='int'] bitWidth must be integer"));

  const auto& issigned = value["isSigned"];
  NANOARROW_RETURN_NOT_OK(
      Check(issigned.is_boolean(), error, "Type[name=='int'] isSigned must be boolean"));

  ArrowType type = NANOARROW_TYPE_UNINITIALIZED;
  if (issigned.get<bool>()) {
    switch (bitwidth.get<int>()) {
      case 8:
        type = NANOARROW_TYPE_INT8;
        break;
      case 16:
        type = NANOARROW_TYPE_INT16;
        break;
      case 32:
        type = NANOARROW_TYPE_INT32;
        break;
      case 64:
        type = NANOARROW_TYPE_INT64;
        break;
      default:
        ArrowErrorSet(error, "Type[name=='int'] bitWidth must be 8, 16, 32, or 64");
        return EINVAL;
    }
  } else {
    switch (bitwidth.get<int>()) {
      case 8:
        type = NANOARROW_TYPE_UINT8;
        break;
      case 16:
        type = NANOARROW_TYPE_UINT16;
        break;
      case 32:
        type = NANOARROW_TYPE_UINT32;
        break;
      case 64:
        type = NANOARROW_TYPE_UINT64;
        break;
      default:
        ArrowErrorSet(error, "Type[name=='int'] bitWidth must be 8, 16, 32, or 64");
        return EINVAL;
    }
  }

  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaSetType(schema, type), error);
  return NANOARROW_OK;
}

ArrowErrorCode SetTypeFloatingPoint(ArrowSchema* schema, const json& value,
                                    ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(Check(value.contains("precision"), error,
                                "Type[name=='floatingpoint'] missing key 'precision'"));

  const auto& precision = value["precision"];
  NANOARROW_RETURN_NOT_OK(Check(precision.is_string(), error,
                                "Type[name=='floatingpoint'] bitWidth must be string"));

  ArrowType type = NANOARROW_TYPE_UNINITIALIZED;
  auto precision_str = precision.get<std::string>();
  if (precision_str == "HALF") {
    type = NANOARROW_TYPE_HALF_FLOAT;
  } else if (precision_str == "SINGLE") {
    type = NANOARROW_TYPE_FLOAT;
  } else if (precision_str == "DOUBLE") {
    type = NANOARROW_TYPE_DOUBLE;
  } else {
    ArrowErrorSet(
        error,
        "Type[name=='floatingpoint'] precision must be 'HALF', 'SINGLE', or 'DOUBLE'");
    return EINVAL;
  }

  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaSetType(schema, type), error);
  return NANOARROW_OK;
}

ArrowErrorCode SetTypeFixedSizeBinary(ArrowSchema* schema, const json& value,
                                      ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(Check(value.contains("byteWidth"), error,
                                "Type[name=='fixedsizebinary'] missing key 'byteWidth'"));

  const auto& byteWidth = value["byteWidth"];
  NANOARROW_RETURN_NOT_OK(
      Check(byteWidth.is_number_integer(), error,
            "Type[name=='fixedsizebinary'] byteWidth must be integer"));

  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowSchemaSetTypeFixedSize(schema, NANOARROW_TYPE_FIXED_SIZE_BINARY,
                                  byteWidth.get<int>()),
      error);
  return NANOARROW_OK;
}

ArrowErrorCode SetTypeDecimal(ArrowSchema* schema, const json& value, ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(Check(value.contains("precision"), error,
                                "Type[name=='decimal'] missing key 'precision'"));
  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("scale"), error, "Type[name=='decimal'] missing key 'scale'"));

  // Some test files omit bitWidth for decimal128
  int bit_width_int;
  if (value.contains("bitWidth")) {
    const auto& bit_width = value["bitWidth"];
    NANOARROW_RETURN_NOT_OK(Check(bit_width.is_number_integer(), error,
                                  "Type[name=='decimal'] bitWidth must be integer"));
    bit_width_int = bit_width.get<int>();
  } else {
    bit_width_int = 128;
  }

  ArrowType type;
  switch (bit_width_int) {
    case 128:
      type = NANOARROW_TYPE_DECIMAL128;
      break;
    case 256:
      type = NANOARROW_TYPE_DECIMAL256;
      break;
    default:
      ArrowErrorSet(error, "Type[name=='decimal'] bitWidth must be 128 or 256");
      return EINVAL;
  }

  const auto& precision = value["precision"];
  NANOARROW_RETURN_NOT_OK(Check(precision.is_number_integer(), error,
                                "Type[name=='decimal'] precision must be integer"));

  const auto& scale = value["scale"];
  NANOARROW_RETURN_NOT_OK(Check(scale.is_number_integer(), error,
                                "Type[name=='decimal'] scale must be integer"));

  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowSchemaSetTypeDecimal(schema, type, precision.get<int>(), scale.get<int>()),
      error);

  return NANOARROW_OK;
}

ArrowErrorCode SetTypeDate(ArrowSchema* schema, const json& value, ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("unit"), error, "Type[name=='date'] missing key 'unit'"));
  const auto& unit = value["unit"];
  NANOARROW_RETURN_NOT_OK(
      Check(unit.is_string(), error, "Type[name=='date'] unit must be string"));
  std::string unit_str = unit.get<std::string>();

  if (unit_str == "DAY") {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaSetType(schema, NANOARROW_TYPE_DATE32),
                                       error);
  } else if (unit_str == "MILLISECOND") {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaSetType(schema, NANOARROW_TYPE_DATE64),
                                       error);
  } else {
    ArrowErrorSet(error, "Type[name=='date'] unit must be 'DAY' or 'MILLISECOND'");
    return EINVAL;
  }

  return NANOARROW_OK;
}

ArrowErrorCode SetTimeUnit(const json& value, ArrowTimeUnit* time_unit,
                           ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("unit"), error, "Time-like type missing key 'unit'"));
  const auto& unit = value["unit"];
  NANOARROW_RETURN_NOT_OK(
      Check(unit.is_string(), error, "Time-like type unit must be string"));
  std::string unit_str = unit.get<std::string>();

  if (unit_str == "SECOND") {
    *time_unit = NANOARROW_TIME_UNIT_SECOND;
  } else if (unit_str == "MILLISECOND") {
    *time_unit = NANOARROW_TIME_UNIT_MILLI;
  } else if (unit_str == "MICROSECOND") {
    *time_unit = NANOARROW_TIME_UNIT_MICRO;
  } else if (unit_str == "NANOSECOND") {
    *time_unit = NANOARROW_TIME_UNIT_NANO;
  } else {
    ArrowErrorSet(
        error,
        "TimeUnit must be 'SECOND' or 'MILLISECOND', 'MICROSECOND', or 'NANOSECOND'");
    return EINVAL;
  }

  return NANOARROW_OK;
}

ArrowErrorCode SetTypeTime(ArrowSchema* schema, const json& value, ArrowError* error) {
  ArrowTimeUnit time_unit;
  NANOARROW_RETURN_NOT_OK(SetTimeUnit(value, &time_unit, error));

  const auto& bit_width = value["bitWidth"];
  NANOARROW_RETURN_NOT_OK(Check(bit_width.is_number_integer(), error,
                                "Type[name=='time'] bitWidth must be integer"));
  auto bit_width_int = bit_width.get<int>();

  if (bit_width_int == 32) {
    NANOARROW_RETURN_NOT_OK(Check(
        time_unit == NANOARROW_TIME_UNIT_SECOND || time_unit == NANOARROW_TIME_UNIT_MILLI,
        error, "Expected time unit of 'SECOND' or 'MILLISECOND' for bitWidth 32"));

    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowSchemaSetTypeDateTime(schema, NANOARROW_TYPE_TIME32, time_unit, nullptr),
        error);
    return NANOARROW_OK;
  } else if (bit_width_int == 64) {
    NANOARROW_RETURN_NOT_OK(Check(
        time_unit == NANOARROW_TIME_UNIT_MICRO || time_unit == NANOARROW_TIME_UNIT_NANO,
        error, "Expected time unit of 'MICROSECOND' or 'NANOSECOND' for bitWidth 64"));
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowSchemaSetTypeDateTime(schema, NANOARROW_TYPE_TIME64, time_unit, nullptr),
        error);
    return NANOARROW_OK;
  } else {
    ArrowErrorSet(error, "Expected Type[name=='time'] bitWidth of 32 or 64");
    return EINVAL;
  }

  return NANOARROW_OK;
}

ArrowErrorCode SetTypeTimestamp(ArrowSchema* schema, const json& value,
                                ArrowError* error) {
  ArrowTimeUnit time_unit;
  NANOARROW_RETURN_NOT_OK(SetTimeUnit(value, &time_unit, error));

  std::string timezone_str;
  if (value.contains("timezone")) {
    const auto& timezone = value["timezone"];
    NANOARROW_RETURN_NOT_OK(Check(timezone.is_string(), error,
                                  "Type[name=='timestamp'] timezone must be string"));
    timezone_str = timezone.get<std::string>();
  }

  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowSchemaSetTypeDateTime(schema, NANOARROW_TYPE_TIMESTAMP, time_unit,
                                 timezone_str.c_str()),
      error);

  return NANOARROW_OK;
}

ArrowErrorCode SetTypeDuration(ArrowSchema* schema, const json& value,
                               ArrowError* error) {
  ArrowTimeUnit time_unit;
  NANOARROW_RETURN_NOT_OK(SetTimeUnit(value, &time_unit, error));

  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowSchemaSetTypeDateTime(schema, NANOARROW_TYPE_DURATION, time_unit, nullptr),
      error);

  return NANOARROW_OK;
}

ArrowErrorCode SetTypeInterval(ArrowSchema* schema, const json& value,
                               ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("unit"), error, "Type[name=='interval'] missing key 'unit'"));
  const auto& unit = value["unit"];
  NANOARROW_RETURN_NOT_OK(
      Check(unit.is_string(), error, "Type[name=='interval'] unit must be string"));
  std::string unit_str = unit.get<std::string>();

  if (unit_str == "YEAR_MONTH") {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowSchemaSetType(schema, NANOARROW_TYPE_INTERVAL_MONTHS), error);
  } else if (unit_str == "DAY_TIME") {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowSchemaSetType(schema, NANOARROW_TYPE_INTERVAL_DAY_TIME), error);
  } else if (unit_str == "MONTH_DAY_NANO") {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowSchemaSetType(schema, NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO), error);
  } else {
    ArrowErrorSet(error,
                  "Type[name=='interval'] unit must be 'YEAR_MONTH', 'DAY_TIME', or "
                  "'MONTH_DAY_NANO'");
    return EINVAL;
  }

  return NANOARROW_OK;
}

ArrowErrorCode SetTypeMap(ArrowSchema* schema, const json& value, ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(Check(value.contains("keysSorted"), error,
                                "Type[name=='map'] missing key 'keysSorted'"));

  const auto& keys_sorted = value["keysSorted"];
  NANOARROW_RETURN_NOT_OK(Check(keys_sorted.is_boolean(), error,
                                "Type[name=='map'] keysSorted must be boolean"));

  if (keys_sorted.get<bool>()) {
    schema->flags |= ARROW_FLAG_MAP_KEYS_SORTED;
  } else {
    schema->flags &= ~ARROW_FLAG_MAP_KEYS_SORTED;
  }

  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaSetFormat(schema, "+m"), error);
  return NANOARROW_OK;
}

ArrowErrorCode SetTypeFixedSizeList(ArrowSchema* schema, const json& value,
                                    ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(Check(value.contains("listSize"), error,
                                "Type[name=='fixedsizelist'] missing key 'listSize'"));

  const auto& list_size = value["listSize"];
  NANOARROW_RETURN_NOT_OK(Check(list_size.is_number_integer(), error,
                                "Type[name=='fixedsizelist'] listSize must be integer"));

  std::stringstream format_builder;
  format_builder << "+w:" << list_size;
  std::string format = format_builder.str();
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaSetFormat(schema, format.c_str()), error);
  return NANOARROW_OK;
}

ArrowErrorCode SetTypeUnion(ArrowSchema* schema, const json& value, ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("mode"), error, "Type[name=='union'] missing key 'mode'"));
  NANOARROW_RETURN_NOT_OK(Check(value.contains("typeIds"), error,
                                "Type[name=='union'] missing key 'typeIds'"));

  const auto& mode = value["mode"];
  NANOARROW_RETURN_NOT_OK(
      Check(mode.is_string(), error, "Type[name=='union'] mode must be string"));

  auto mode_str = mode.get<std::string>();
  std::stringstream type_ids_format;

  if (mode_str == "DENSE") {
    type_ids_format << "+ud:";
  } else if (mode_str == "SPARSE") {
    type_ids_format << "+us:";
  } else {
    ArrowErrorSet(error, "Type[name=='union'] mode must be 'DENSE' or 'SPARSE'");
    return EINVAL;
  }

  const auto& type_ids = value["typeIds"];
  NANOARROW_RETURN_NOT_OK(
      Check(type_ids.is_array(), error, "Type[name=='union'] typeIds must be array"));

  if (type_ids.size() > 0) {
    for (size_t i = 0; i < type_ids.size(); i++) {
      const auto& type_id = type_ids[i];
      NANOARROW_RETURN_NOT_OK(Check(type_id.is_number_integer(), error,
                                    "Type[name=='union'] typeIds item must be integer"));
      type_ids_format << type_id;

      if ((i + 1) < type_ids.size()) {
        type_ids_format << ",";
      }
    }
  }

  std::string type_ids_format_str = type_ids_format.str();
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowSchemaSetFormat(schema, type_ids_format_str.c_str()), error);

  return NANOARROW_OK;
}

ArrowErrorCode SetType(ArrowSchema* schema, const json& value, ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(Check(value.is_object(), error, "Type must be object"));
  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("name"), error, "Type missing key 'name'"));

  const auto& name = value["name"];
  NANOARROW_RETURN_NOT_OK(Check(name.is_string(), error, "Type name must be string"));
  auto name_str = name.get<std::string>();

  if (name_str == "null") {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaSetType(schema, NANOARROW_TYPE_NA),
                                       error);
  } else if (name_str == "bool") {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaSetType(schema, NANOARROW_TYPE_BOOL),
                                       error);
  } else if (name_str == "int") {
    NANOARROW_RETURN_NOT_OK(SetTypeInt(schema, value, error));
  } else if (name_str == "floatingpoint") {
    NANOARROW_RETURN_NOT_OK(SetTypeFloatingPoint(schema, value, error));
  } else if (name_str == "decimal") {
    NANOARROW_RETURN_NOT_OK(SetTypeDecimal(schema, value, error));
  } else if (name_str == "utf8") {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaSetType(schema, NANOARROW_TYPE_STRING),
                                       error);
  } else if (name_str == "largeutf8") {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowSchemaSetType(schema, NANOARROW_TYPE_LARGE_STRING), error);
  } else if (name_str == "binary") {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaSetType(schema, NANOARROW_TYPE_BINARY),
                                       error);
  } else if (name_str == "largebinary") {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowSchemaSetType(schema, NANOARROW_TYPE_LARGE_BINARY), error);
  } else if (name_str == "fixedsizebinary") {
    NANOARROW_RETURN_NOT_OK(SetTypeFixedSizeBinary(schema, value, error));
  } else if (name_str == "list") {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaSetFormat(schema, "+l"), error);
  } else if (name_str == "largelist") {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaSetFormat(schema, "+L"), error);
  } else if (name_str == "fixedsizelist") {
    NANOARROW_RETURN_NOT_OK(SetTypeFixedSizeList(schema, value, error));
  } else if (name_str == "date") {
    NANOARROW_RETURN_NOT_OK(SetTypeDate(schema, value, error));
  } else if (name_str == "time") {
    NANOARROW_RETURN_NOT_OK(SetTypeTime(schema, value, error));
  } else if (name_str == "timestamp") {
    NANOARROW_RETURN_NOT_OK(SetTypeTimestamp(schema, value, error));
  } else if (name_str == "duration") {
    NANOARROW_RETURN_NOT_OK(SetTypeDuration(schema, value, error));
  } else if (name_str == "interval") {
    NANOARROW_RETURN_NOT_OK(SetTypeInterval(schema, value, error));
  } else if (name_str == "map") {
    NANOARROW_RETURN_NOT_OK(SetTypeMap(schema, value, error));
  } else if (name_str == "union") {
    NANOARROW_RETURN_NOT_OK(SetTypeUnion(schema, value, error));
  } else if (name_str == "struct") {
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaSetFormat(schema, "+s"), error);
  } else {
    ArrowErrorSet(error, "Unsupported Type name: '%s'", name_str.c_str());
    return ENOTSUP;
  }

  return NANOARROW_OK;
}

ArrowErrorCode SetDictionary(ArrowSchema* schema, const json& value,
                             int32_t* dictionary_id, ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(Check(value.is_object(), error, "Dictionary must be object"));
  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("id"), error, "Dictionary missing key 'id'"));
  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("indexType"), error, "Dictionary missing key 'type'"));
  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("isOrdered"), error, "Dictionary missing key 'isOrdered'"));

  const auto& id = value["id"];
  NANOARROW_RETURN_NOT_OK(
      Check(id.is_number_integer(), error, "Dictionary id must be integer"));
  *dictionary_id = id.get<int32_t>();

  // Parse the index type
  NANOARROW_RETURN_NOT_OK(SetType(schema, value["indexType"], error));

  // Set the flag
  const auto& is_ordered = value["isOrdered"];
  NANOARROW_RETURN_NOT_OK(
      Check(is_ordered.is_boolean(), error, "Dictionary isOrdered must be bool"));
  if (is_ordered.get<bool>()) {
    schema->flags |= ARROW_FLAG_DICTIONARY_ORDERED;
  } else {
    schema->flags &= ~ARROW_FLAG_DICTIONARY_ORDERED;
  }

  return NANOARROW_OK;
}

ArrowErrorCode SetMetadata(ArrowSchema* schema, const json& value, ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(Check(value.is_null() || value.is_array(), error,
                                "Field or Schema metadata must be null or array"));
  if (value.is_null()) {
    return NANOARROW_OK;
  }

  nanoarrow::UniqueBuffer metadata;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowMetadataBuilderInit(metadata.get(), nullptr),
                                     error);
  for (const auto& item : value) {
    NANOARROW_RETURN_NOT_OK(
        Check(item.is_object(), error, "metadata item must be object"));
    NANOARROW_RETURN_NOT_OK(
        Check(item.contains("key"), error, "metadata item missing key 'key'"));
    NANOARROW_RETURN_NOT_OK(
        Check(item.contains("value"), error, "metadata item missing key 'value'"));

    const auto& key = item["key"];
    const auto& value = item["value"];
    NANOARROW_RETURN_NOT_OK(
        Check(key.is_string(), error, "metadata item key must be string"));
    NANOARROW_RETURN_NOT_OK(
        Check(value.is_string(), error, "metadata item value must be string"));

    auto key_str = key.get<std::string>();
    auto value_str = value.get<std::string>();
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowMetadataBuilderAppend(metadata.get(), ArrowCharView(key_str.c_str()),
                                   ArrowCharView(value_str.c_str())),
        error);
  }

  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowSchemaSetMetadata(schema, reinterpret_cast<char*>(metadata->data)), error);
  return NANOARROW_OK;
}

ArrowErrorCode SetField(ArrowSchema* schema, const json& value,
                        internal::DictionaryContext& dictionaries, ArrowError* error);

ArrowErrorCode SetSchema(ArrowSchema* schema, const json& value,
                         internal::DictionaryContext& dictionaries, ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(
      Check(value.is_object(), error, "Expected Schema to be a JSON object"));
  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("fields"), error, "Schema missing key 'fields'"));

  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRUCT), error);

  // Top-level schema is non-nullable
  schema->flags = 0;

  const auto& fields = value["fields"];
  NANOARROW_RETURN_NOT_OK(Check(fields.is_array(), error, "Schema fields must be array"));
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaAllocateChildren(schema, fields.size()),
                                     error);
  for (int64_t i = 0; i < schema->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        SetField(schema->children[i], fields[i], dictionaries, error));
  }

  if (value.contains("metadata")) {
    NANOARROW_RETURN_NOT_OK(SetMetadata(schema, value["metadata"], error));
  }

  // Validate!
  ArrowSchemaView schema_view;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&schema_view, schema, error));
  return NANOARROW_OK;
}

ArrowErrorCode SetField(ArrowSchema* schema, const json& value,
                        internal::DictionaryContext& dictionaries, ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(
      Check(value.is_object(), error, "Expected Field to be a JSON object"));
  ArrowSchemaInit(schema);

  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("name"), error, "Field missing key 'name'"));
  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("type"), error, "Field missing key 'type'"));
  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("nullable"), error, "Field missing key 'nullable'"));
  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("children"), error, "Field missing key 'children'"));

  // Name
  const auto& name = value["name"];
  NANOARROW_RETURN_NOT_OK(Check(name.is_string() || name.is_null(), error,
                                "Field name must be string or null"));
  if (name.is_string()) {
    auto name_str = name.get<std::string>();
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaSetName(schema, name_str.c_str()),
                                       error);
  }

  // Nullability
  const auto& nullable = value["nullable"];
  NANOARROW_RETURN_NOT_OK(
      Check(nullable.is_boolean(), error, "Field nullable must be boolean"));
  if (nullable.get<bool>()) {
    schema->flags |= ARROW_FLAG_NULLABLE;
  } else {
    schema->flags &= ~ARROW_FLAG_NULLABLE;
  }

  // Metadata
  if (value.contains("metadata")) {
    NANOARROW_RETURN_NOT_OK(SetMetadata(schema, value["metadata"], error));
  }

  // If we have a dictionary, this value needs to be in schema->dictionary
  // and value["dictionary"] needs to be in schema
  if (value.contains("dictionary")) {
    // Put the index type in this schema
    int32_t dictionary_id;
    NANOARROW_RETURN_NOT_OK(
        SetDictionary(schema, value["dictionary"], &dictionary_id, error));

    // Allocate a dictionary and put this value (minus dictionary, metadata, and name)
    json value_copy = value;
    value_copy.erase("dictionary");
    value_copy.erase("metadata");
    value_copy["name"] = nullptr;
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaAllocateDictionary(schema), error);
    NANOARROW_RETURN_NOT_OK(
        SetField(schema->dictionary, value_copy, dictionaries, error));

    // Keep track of this dictionary_id/schema for parsing batches
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        dictionaries.RecordSchema(dictionary_id, schema->dictionary), error);

    // Validate!
    ArrowSchemaView schema_view;
    NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&schema_view, schema, error));

    return NANOARROW_OK;
  }

  NANOARROW_RETURN_NOT_OK(SetType(schema, value["type"], error));

  const auto& children = value["children"];
  NANOARROW_RETURN_NOT_OK(
      Check(children.is_array(), error, "Field children must be array"));
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaAllocateChildren(schema, children.size()),
                                     error);
  for (int64_t i = 0; i < schema->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(
        SetField(schema->children[i], children[i], dictionaries, error));
  }

  // Validate!
  ArrowSchemaView schema_view;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&schema_view, schema, error));
  return NANOARROW_OK;
}

ArrowErrorCode SetBufferBitmap(const json& value, ArrowBitmap* bitmap,
                               ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(Check(value.is_array(), error, "bitmap buffer must be array"));

  // Reserving with the exact length ensures that the last bits are always zeroed.
  // This was an assumption made by the C# implementation at the time this was
  // implemented.
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBitmapReserve(bitmap, value.size()), error);

  for (const auto& item : value) {
    // Some example files write bitmaps as [true, false, true] but the documentation
    // says [1, 0, 1]. Accept both for simplicity.
    NANOARROW_RETURN_NOT_OK(Check(item.is_boolean() || item.is_number_integer(), error,
                                  "bitmap item must be bool or integer"));
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBitmapAppend(bitmap, item.get<uint8_t>(), 1),
                                       error);
  }

  return NANOARROW_OK;
}

template <typename T, typename BiggerT = int64_t>
ArrowErrorCode SetBufferIntItem(const json& item, ArrowBuffer* buffer,
                                ArrowError* error) {
  if (item.is_string()) {
    try {
      // The JSON parser here can handle up to 2^64 - 1
      auto item_int = json::parse(item.get<std::string>());
      return SetBufferIntItem<T, BiggerT>(item_int, buffer, error);
    } catch (json::parse_error&) {
      ArrowErrorSet(error,
                    "integer buffer item encoded as string must parse as integer: %s",
                    item.dump().c_str());
      return EINVAL;
    }
  }

  NANOARROW_RETURN_NOT_OK(Check(item.is_number_integer(), error,
                                "integer buffer item must be integer number or string"));
  NANOARROW_RETURN_NOT_OK(
      Check(std::numeric_limits<T>::is_signed || item.is_number_unsigned(), error,
            "expected unsigned integer buffer item but found signed integer '" +
                item.dump() + "'"));

  auto item_int = item.get<BiggerT>();

  NANOARROW_RETURN_NOT_OK(
      Check(item_int >= std::numeric_limits<T>::lowest() &&
                item_int <= std::numeric_limits<T>::max(),
            error, "integer buffer item '" + item.dump() + "' outside type limits"));

  T buffer_value = static_cast<T>(item_int);
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBufferAppend(buffer, &buffer_value, sizeof(T)),
                                     error);

  return NANOARROW_OK;
}

template <typename T, typename BiggerT = int64_t>
ArrowErrorCode SetBufferInt(const json& value, ArrowBuffer* buffer, ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(Check(value.is_array(), error, "int buffer must be array"));

  for (const auto& item : value) {
    // NANOARROW_RETURN_NOT_OK() interacts poorly with multiple template args
    ArrowErrorCode result = SetBufferIntItem<T, BiggerT>(item, buffer, error);
    NANOARROW_RETURN_NOT_OK(result);
  }

  return NANOARROW_OK;
}

template <typename T>
ArrowErrorCode SetBufferFloatingPoint(const json& value, ArrowBuffer* buffer,
                                      ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(
      Check(value.is_array(), error, "floatingpoint buffer must be array"));

  for (const auto& item : value) {
    NANOARROW_RETURN_NOT_OK(
        Check(item.is_number(), error, "floatingpoint buffer item must be number"));
    double item_dbl = item.get<double>();

    NANOARROW_RETURN_NOT_OK(Check(
        item_dbl >= std::numeric_limits<T>::lowest() &&
            item_dbl <= std::numeric_limits<T>::max(),
        error, "floatingpoint buffer item '" + item.dump() + "' outside type limits"));

    T buffer_value = static_cast<T>(item_dbl);
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowBufferAppend(buffer, &buffer_value, sizeof(T)), error);
  }

  return NANOARROW_OK;
}

template <typename T>
ArrowErrorCode SetBufferString(const json& value, ArrowBuffer* offsets, ArrowBuffer* data,
                               ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(
      Check(value.is_array(), error, "utf8 data buffer must be array"));

  // Check offsets against values
  const T* expected_offset = reinterpret_cast<const T*>(offsets->data);
  NANOARROW_RETURN_NOT_OK(Check(
      static_cast<size_t>(offsets->size_bytes) == ((value.size() + 1) * sizeof(T)), error,
      "Expected offset buffer with " + std::to_string(value.size()) + " elements"));
  NANOARROW_RETURN_NOT_OK(
      Check(*expected_offset++ == 0, error, "first offset must be zero"));

  int64_t last_offset = 0;

  for (const auto& item : value) {
    NANOARROW_RETURN_NOT_OK(
        Check(item.is_string(), error, "utf8 data buffer item must be string"));
    auto item_str = item.get<std::string>();

    // Append data
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowBufferAppend(data, reinterpret_cast<const uint8_t*>(item_str.data()),
                          item_str.size()),
        error);

    // Check offset
    last_offset += item_str.size();
    NANOARROW_RETURN_NOT_OK(Check(*expected_offset++ == last_offset, error,
                                  "Expected offset value " + std::to_string(last_offset) +
                                      " at utf8 data buffer item " + item.dump()));
  }

  return NANOARROW_OK;
}

ArrowErrorCode AppendBinaryElement(const json& item, ArrowBuffer* data,
                                   ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(
      Check(item.is_string(), error, "binary data buffer item must be string"));
  auto item_str = item.get<std::string>();

  size_t item_size_bytes = item_str.size() / 2;
  NANOARROW_RETURN_NOT_OK(Check((item_size_bytes * 2) == item_str.size(), error,
                                "binary data buffer item must have even size"));

  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBufferReserve(data, item_size_bytes), error);
  for (size_t i = 0; i < item_str.size(); i += 2) {
    std::string byte_hex = item_str.substr(i, 2);
    char* end_ptr;
    uint8_t byte = static_cast<uint8_t>(std::strtoul(byte_hex.data(), &end_ptr, 16));
    NANOARROW_RETURN_NOT_OK(
        Check(end_ptr == (byte_hex.data() + 2), error,
              "binary data buffer item must contain a valid hex-encoded byte string"));

    data->data[data->size_bytes] = byte;
    data->size_bytes++;
  }

  return NANOARROW_OK;
}

template <typename T>
ArrowErrorCode SetBufferBinary(const json& value, ArrowBuffer* offsets, ArrowBuffer* data,
                               ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(
      Check(value.is_array(), error, "binary data buffer must be array"));

  // Check offsets against values if not fixed size
  const T* expected_offset = reinterpret_cast<const T*>(offsets->data);
  NANOARROW_RETURN_NOT_OK(Check(
      static_cast<size_t>(offsets->size_bytes) == ((value.size() + 1) * sizeof(T)), error,
      "Expected offset buffer with " + std::to_string(value.size()) + " elements"));
  NANOARROW_RETURN_NOT_OK(
      Check(*expected_offset++ == 0, error, "first offset must be zero"));

  for (const auto& item : value) {
    NANOARROW_RETURN_NOT_OK(AppendBinaryElement(item, data, error));

    // Check offset
    NANOARROW_RETURN_NOT_OK(Check(*expected_offset++ == data->size_bytes, error,
                                  "Expected offset value " +
                                      std::to_string(data->size_bytes) +
                                      " at binary data buffer item " + item.dump()));
  }

  return NANOARROW_OK;
}

ArrowErrorCode SetBufferFixedSizeBinary(const json& value, ArrowBuffer* data,
                                        int64_t fixed_size, ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(
      Check(value.is_array(), error, "binary data buffer must be array"));

  int64_t last_offset = 0;

  for (const auto& item : value) {
    NANOARROW_RETURN_NOT_OK(AppendBinaryElement(item, data, error));
    int64_t item_size_bytes = data->size_bytes - last_offset;

    NANOARROW_RETURN_NOT_OK(Check(item_size_bytes == fixed_size, error,
                                  "Expected fixed size binary value of size " +
                                      std::to_string(fixed_size) +
                                      " at binary data buffer item " + item.dump()));
    last_offset = data->size_bytes;
  }

  return NANOARROW_OK;
}

ArrowErrorCode SetBufferIntervalDayTime(const json& value, ArrowBuffer* buffer,
                                        ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(
      Check(value.is_array(), error, "interval_day_time buffer must be array"));

  for (const auto& item : value) {
    NANOARROW_RETURN_NOT_OK(
        Check(item.is_object(), error, "interval_day_time buffer item must be object"));
    NANOARROW_RETURN_NOT_OK(Check(item.contains("days"), error,
                                  "interval_day_time buffer item missing key 'days'"));
    NANOARROW_RETURN_NOT_OK(
        Check(item.contains("milliseconds"), error,
              "interval_day_time buffer item missing key 'milliseconds'"));

    NANOARROW_RETURN_NOT_OK(SetBufferIntItem<int32_t>(item["days"], buffer, error));
    NANOARROW_RETURN_NOT_OK(
        SetBufferIntItem<int32_t>(item["milliseconds"], buffer, error));
  }

  return NANOARROW_OK;
}

ArrowErrorCode SetBufferIntervalMonthDayNano(const json& value, ArrowBuffer* buffer,
                                             ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(
      Check(value.is_array(), error, "interval buffer must be array"));

  for (const auto& item : value) {
    NANOARROW_RETURN_NOT_OK(
        Check(item.is_object(), error, "interval buffer item must be object"));
    NANOARROW_RETURN_NOT_OK(Check(item.contains("months"), error,
                                  "interval buffer item missing key 'months'"));
    NANOARROW_RETURN_NOT_OK(
        Check(item.contains("days"), error, "interval buffer item missing key 'days'"));
    NANOARROW_RETURN_NOT_OK(Check(item.contains("nanoseconds"), error,
                                  "interval buffer item missing key 'nanoseconds'"));

    NANOARROW_RETURN_NOT_OK(SetBufferIntItem<int32_t>(item["months"], buffer, error));
    NANOARROW_RETURN_NOT_OK(SetBufferIntItem<int32_t>(item["days"], buffer, error));
    NANOARROW_RETURN_NOT_OK(
        SetBufferIntItem<int64_t>(item["nanoseconds"], buffer, error));
  }

  return NANOARROW_OK;
}

ArrowErrorCode SetBufferDecimal(const json& value, ArrowBuffer* buffer, int bitwidth,
                                ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(Check(value.is_array(), error, "decimal buffer must be array"));

  ArrowDecimal decimal;
  ArrowDecimalInit(&decimal, bitwidth, 0, 0);

  ArrowStringView item_view;

  for (const auto& item : value) {
    NANOARROW_RETURN_NOT_OK(
        Check(item.is_string(), error, "decimal buffer item must be string"));
    auto item_str = item.get<std::string>();
    item_view.data = item_str.data();
    item_view.size_bytes = item_str.size();
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowDecimalSetDigits(&decimal, item_view), error);
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowBufferAppend(buffer, decimal.words, decimal.n_words * sizeof(uint64_t)),
        error);
  }

  return NANOARROW_OK;
}

ArrowErrorCode SetArrayColumnBuffers(const json& value, ArrowArrayView* array_view,
                                     ArrowArray* array, int buffer_i, ArrowError* error) {
  ArrowBuffer* buffer = ArrowArrayBuffer(array, buffer_i);

  switch (array_view->layout.buffer_type[buffer_i]) {
    case NANOARROW_BUFFER_TYPE_VALIDITY: {
      NANOARROW_RETURN_NOT_OK(
          Check(value.contains("VALIDITY"), error, "missing key 'VALIDITY'"));
      const auto& validity = value["VALIDITY"];
      NANOARROW_RETURN_NOT_OK(
          SetBufferBitmap(validity, ArrowArrayValidityBitmap(array), error));
      break;
    }
    case NANOARROW_BUFFER_TYPE_TYPE_ID: {
      NANOARROW_RETURN_NOT_OK(
          Check(value.contains("TYPE_ID"), error, "missing key 'TYPE_ID'"));
      const auto& type_id = value["TYPE_ID"];
      NANOARROW_RETURN_NOT_OK(SetBufferInt<int8_t>(type_id, buffer, error));
      break;
    }
    case NANOARROW_BUFFER_TYPE_UNION_OFFSET: {
      NANOARROW_RETURN_NOT_OK(
          Check(value.contains("OFFSET"), error, "missing key 'OFFSET'"));
      const auto& offset = value["OFFSET"];
      NANOARROW_RETURN_NOT_OK(SetBufferInt<int32_t>(offset, buffer, error));
      break;
    }
    case NANOARROW_BUFFER_TYPE_DATA_OFFSET: {
      NANOARROW_RETURN_NOT_OK(
          Check(value.contains("OFFSET"), error, "missing key 'OFFSET'"));
      const auto& offset = value["OFFSET"];

      if (array_view->layout.element_size_bits[buffer_i] == 32) {
        NANOARROW_RETURN_NOT_OK(SetBufferInt<int32_t>(offset, buffer, error));
      } else {
        NANOARROW_RETURN_NOT_OK(SetBufferInt<int64_t>(offset, buffer, error));
      }
      break;
    }

    case NANOARROW_BUFFER_TYPE_DATA: {
      NANOARROW_RETURN_NOT_OK(Check(value.contains("DATA"), error, "missing key 'DATA'"));
      const auto& data = value["DATA"];

      switch (array_view->storage_type) {
        case NANOARROW_TYPE_BOOL: {
          nanoarrow::UniqueBitmap bitmap;
          NANOARROW_RETURN_NOT_OK(SetBufferBitmap(data, bitmap.get(), error));
          ArrowBufferMove(&bitmap->buffer, buffer);
          return NANOARROW_OK;
        }
        case NANOARROW_TYPE_INT8:
          return SetBufferInt<int8_t>(data, buffer, error);
        case NANOARROW_TYPE_UINT8:
          return SetBufferInt<uint8_t>(data, buffer, error);
        case NANOARROW_TYPE_INT16:
          return SetBufferInt<int16_t>(data, buffer, error);
        case NANOARROW_TYPE_UINT16:
          return SetBufferInt<uint16_t>(data, buffer, error);
        case NANOARROW_TYPE_INT32:
        case NANOARROW_TYPE_INTERVAL_MONTHS:
          return SetBufferInt<int32_t>(data, buffer, error);
        case NANOARROW_TYPE_UINT32:
          return SetBufferInt<uint32_t>(data, buffer, error);
        case NANOARROW_TYPE_INT64:
          return SetBufferInt<int64_t>(data, buffer, error);
        case NANOARROW_TYPE_UINT64:
          return SetBufferInt<uint64_t, uint64_t>(data, buffer, error);

        case NANOARROW_TYPE_HALF_FLOAT:
          return SetBufferFloatingPoint<float>(data, buffer, error);
        case NANOARROW_TYPE_FLOAT:
          return SetBufferFloatingPoint<float>(data, buffer, error);
        case NANOARROW_TYPE_DOUBLE:
          return SetBufferFloatingPoint<double>(data, buffer, error);

        case NANOARROW_TYPE_STRING:
          return SetBufferString<int32_t>(data, ArrowArrayBuffer(array, buffer_i - 1),
                                          buffer, error);
        case NANOARROW_TYPE_LARGE_STRING:
          return SetBufferString<int64_t>(data, ArrowArrayBuffer(array, buffer_i - 1),
                                          buffer, error);
        case NANOARROW_TYPE_BINARY:
          return SetBufferBinary<int32_t>(data, ArrowArrayBuffer(array, buffer_i - 1),
                                          buffer, error);
        case NANOARROW_TYPE_LARGE_BINARY:
          return SetBufferBinary<int64_t>(data, ArrowArrayBuffer(array, buffer_i - 1),
                                          buffer, error);
        case NANOARROW_TYPE_FIXED_SIZE_BINARY:
          return SetBufferFixedSizeBinary(
              data, buffer, array_view->layout.element_size_bits[buffer_i] / 8, error);
        case NANOARROW_TYPE_INTERVAL_DAY_TIME:
          return SetBufferIntervalDayTime(data, buffer, error);
        case NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO:
          return SetBufferIntervalMonthDayNano(data, buffer, error);
        case NANOARROW_TYPE_DECIMAL128:
          return SetBufferDecimal(data, buffer, 128, error);
        case NANOARROW_TYPE_DECIMAL256:
          return SetBufferDecimal(data, buffer, 256, error);
        default:
          ArrowErrorSet(error, "storage type %s DATA buffer not supported",
                        ArrowTypeString(array_view->storage_type));
          return ENOTSUP;
      }
      break;
    }
    case NANOARROW_BUFFER_TYPE_NONE:
      break;
  }

  return NANOARROW_OK;
}

ArrowErrorCode SetArrayColumn(const json& value, const ArrowSchema* schema,
                              ArrowArrayView* array_view, ArrowArray* array,
                              internal::DictionaryContext& dictionaries,
                              ArrowError* error,
                              const std::string& parent_error_prefix = "") {
  NANOARROW_RETURN_NOT_OK(
      Check(value.is_object(), error, "Expected Column to be a JSON object"));

  // Check + resolve name early to generate better error messages
  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("name"), error, "Column missing key 'name'"));

  const auto& name = value["name"];
  NANOARROW_RETURN_NOT_OK(Check(name.is_null() || name.is_string(), error,
                                "Column name must be string or null"));

  std::string error_prefix;
  if (name.is_string()) {
    error_prefix = parent_error_prefix + "-> Column '" + name.get<std::string>() + "' ";
  } else {
    error_prefix = parent_error_prefix + "-> Column <name is null> ";
  }

  // Check, resolve, and recurse children
  NANOARROW_RETURN_NOT_OK(Check(array_view->n_children == 0 || value.contains("children"),
                                error, error_prefix + "missing key children"));

  if (value.contains("children")) {
    const auto& children = value["children"];
    NANOARROW_RETURN_NOT_OK(
        Check(children.is_array(), error, error_prefix + "children must be array"));
    NANOARROW_RETURN_NOT_OK(
        Check(children.size() == static_cast<size_t>(array_view->n_children), error,
              error_prefix + "children has incorrect size"));

    for (int64_t i = 0; i < array_view->n_children; i++) {
      NANOARROW_RETURN_NOT_OK(SetArrayColumn(children[i], schema->children[i],
                                             array_view->children[i], array->children[i],
                                             dictionaries, error, error_prefix));
    }
  }

  // Build buffers
  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    NANOARROW_RETURN_NOT_OK(PrefixError(
        SetArrayColumnBuffers(value, array_view, array, i, error), error, error_prefix));
  }

  // Check + resolve count
  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("count"), error, error_prefix + "missing key 'count'"));
  const auto& count = value["count"];
  NANOARROW_RETURN_NOT_OK(
      Check(count.is_number_integer(), error, error_prefix + "count must be integer"));
  array_view->length = count.get<int64_t>();

  // Set ArrayView buffer views. This is because ArrowArrayInitFromSchema() doesn't
  // support custom type ids for unions but the ArrayView does (otherwise
  // ArrowArrayFinishBuilding() would work).
  for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
    ArrowBuffer* buffer = ArrowArrayBuffer(array, i);
    ArrowBufferView* buffer_view = array_view->buffer_views + i;
    buffer_view->data.as_uint8 = buffer->data;
    buffer_view->size_bytes = buffer->size_bytes;
  }

  array_view->null_count = ArrowArrayViewComputeNullCount(array_view);

  // If there is a dictionary associated with schema, parse its value into dictionary
  if (schema->dictionary != nullptr) {
    NANOARROW_RETURN_NOT_OK(Check(
        dictionaries.HasDictionaryForSchema(schema->dictionary), error,
        error_prefix +
            "dictionary could not be resolved from dictionary id in SetArrayColumn()"));

    const internal::Dictionary& dict = dictionaries.Get(schema->dictionary);
    NANOARROW_RETURN_NOT_OK(SetArrayColumn(
        json::parse(dict.column_json), schema->dictionary, array_view->dictionary,
        array->dictionary, dictionaries, error, error_prefix + "-> <dictionary> "));
  }

  // Validate the array view
  NANOARROW_RETURN_NOT_OK(PrefixError(
      ArrowArrayViewValidate(array_view, NANOARROW_VALIDATION_LEVEL_FULL, error), error,
      error_prefix + "failed to validate: "));

  // Flush length and buffer pointers to the Array. This also ensures that buffers
  // are not NULL (matters for some versions of some implementations).
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowArrayFinishBuildingDefault(array, nullptr),
                                     error);
  array->length = array_view->length;
  array->null_count = array_view->null_count;

  return NANOARROW_OK;
}

ArrowErrorCode SetArrayBatch(const json& value, const ArrowSchema* schema,
                             ArrowArrayView* array_view, ArrowArray* array,
                             internal::DictionaryContext& dictionaries,
                             ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(
      Check(value.is_object(), error, "Expected RecordBatch to be a JSON object"));

  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("count"), error, "RecordBatch missing key 'count'"));

  const auto& count = value["count"];
  NANOARROW_RETURN_NOT_OK(
      Check(count.is_number_integer(), error, "RecordBatch count must be integer"));
  array_view->length = count.get<int64_t>();

  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("columns"), error, "RecordBatch missing key 'columns'"));

  const auto& columns = value["columns"];
  NANOARROW_RETURN_NOT_OK(
      Check(columns.is_array(), error, "RecordBatch columns must be array"));
  NANOARROW_RETURN_NOT_OK(
      Check(columns.size() == static_cast<size_t>(array_view->n_children), error,
            "RecordBatch children has incorrect size"));

  for (int64_t i = 0; i < array_view->n_children; i++) {
    NANOARROW_RETURN_NOT_OK(SetArrayColumn(columns[i], schema->children[i],
                                           array_view->children[i], array->children[i],
                                           dictionaries, error));
  }

  // Validate the array view
  NANOARROW_RETURN_NOT_OK(PrefixError(
      ArrowArrayViewValidate(array_view, NANOARROW_VALIDATION_LEVEL_FULL, error), error,
      "RecordBatch failed to validate: "));

  // Flush length and buffer pointers to the Array
  array->length = array_view->length;
  NANOARROW_RETURN_NOT_OK_WITH_ERROR(
      ArrowArrayFinishBuilding(array, NANOARROW_VALIDATION_LEVEL_NONE, nullptr), error);

  return NANOARROW_OK;
}

ArrowErrorCode RecordDictionaryBatch(const json& value,
                                     internal::DictionaryContext& dictionaries,
                                     ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(
      Check(value.is_object(), error, "dictionary batch must be object"));
  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("id"), error, "dictionary batch missing key 'id'"));
  NANOARROW_RETURN_NOT_OK(
      Check(value.contains("data"), error, "dictionary batch missing key 'data'"));

  const auto& id = value["id"];
  NANOARROW_RETURN_NOT_OK(
      Check(id.is_number_integer(), error, "dictionary batch id must be integer"));
  int id_int = id.get<int>();
  NANOARROW_RETURN_NOT_OK(Check(dictionaries.HasDictionaryForId(id_int), error,
                                "dictionary batch has unknown id"));

  const auto& batch = value["data"];
  NANOARROW_RETURN_NOT_OK(
      Check(batch.is_object(), error, "dictionary batch data must be object"));
  NANOARROW_RETURN_NOT_OK(
      Check(batch.contains("columns"), error, "dictionary batch missing key 'columns'"));
  NANOARROW_RETURN_NOT_OK(
      Check(batch.contains("count"), error, "dictionary batch missing key 'count'"));

  const auto& batch_columns = batch["columns"];
  NANOARROW_RETURN_NOT_OK(Check(batch_columns.is_array() && batch_columns.size() == 1,
                                error,
                                "dictionary batch columns must be array of size 1"));

  const auto& batch_count = batch["count"];
  NANOARROW_RETURN_NOT_OK(Check(batch_count.is_number_integer(), error,
                                "dictionary batch count must be integer"));

  dictionaries.RecordArray(id_int, batch_count.get<int32_t>(), batch_columns[0].dump());
  return NANOARROW_OK;
}

ArrowErrorCode RecordDictionaryBatches(const json& value,
                                       internal::DictionaryContext& dictionaries,
                                       ArrowError* error) {
  NANOARROW_RETURN_NOT_OK(Check(value.is_array(), error, "dictionaries must be array"));

  for (const auto& batch : value) {
    NANOARROW_RETURN_NOT_OK(RecordDictionaryBatch(batch, dictionaries, error));
  }

  return NANOARROW_OK;
}

}  // namespace

}  // namespace reader_internal

ArrowErrorCode TestingJSONReader::ReadDataFile(const std::string& data_file_json,
                                               ArrowArrayStream* out, int num_batch,
                                               ArrowError* error) {
  dictionaries_.clear();

  try {
    auto obj = nlohmann::json::parse(data_file_json);
    NANOARROW_RETURN_NOT_OK(
        reader_internal::Check(obj.is_object(), error, "data file must be object"));
    NANOARROW_RETURN_NOT_OK(reader_internal::Check(obj.contains("schema"), error,
                                                   "data file missing key 'schema'"));

    // Read Schema
    nanoarrow::UniqueSchema schema;
    NANOARROW_RETURN_NOT_OK(
        reader_internal::SetSchema(schema.get(), obj["schema"], dictionaries_, error));

    NANOARROW_RETURN_NOT_OK(reader_internal::Check(obj.contains("batches"), error,
                                                   "data file missing key 'batches'"));
    const auto& batches = obj["batches"];
    NANOARROW_RETURN_NOT_OK(reader_internal::Check(batches.is_array(), error,
                                                   "data file batches must be array"));

    // Populate ArrayView
    nanoarrow::UniqueArrayView array_view;
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), error));

    // Record any dictionaries that might be present
    if (obj.contains("dictionaries")) {
      NANOARROW_RETURN_NOT_OK(reader_internal::RecordDictionaryBatches(
          obj["dictionaries"], dictionaries_, error));
    }

    // Get a vector of batch ids to parse
    std::vector<size_t> batch_ids;
    if (num_batch == kNumBatchOnlySchema) {
      batch_ids.resize(0);
    } else if (num_batch == kNumBatchReadAll) {
      batch_ids.resize(batches.size());
      std::iota(batch_ids.begin(), batch_ids.end(), 0);
    } else if (num_batch >= 0 && static_cast<size_t>(num_batch) < batches.size()) {
      batch_ids.push_back(num_batch);
    } else {
      ArrowErrorSet(error, "Expected num_batch between 0 and %d but got %d",
                    static_cast<int>(batches.size() - 1), num_batch);
      return EINVAL;
    }

    // Initialize ArrayStream with required capacity
    nanoarrow::UniqueArrayStream stream;
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowBasicArrayStreamInit(stream.get(), schema.get(), batch_ids.size()), error);

    // Populate ArrayStream batches
    for (size_t i = 0; i < batch_ids.size(); i++) {
      nanoarrow::UniqueArray array;
      NANOARROW_RETURN_NOT_OK(
          ArrowArrayInitFromArrayView(array.get(), array_view.get(), error));
      SetArrayAllocatorRecursive(array.get());
      NANOARROW_RETURN_NOT_OK(reader_internal::SetArrayBatch(
          batches[batch_ids[i]], schema.get(), array_view.get(), array.get(),
          dictionaries_, error));
      ArrowBasicArrayStreamSetArray(stream.get(), i, array.get());
    }

    ArrowArrayStreamMove(stream.get(), out);
    return NANOARROW_OK;
  } catch (nlohmann::json::exception& e) {
    ArrowErrorSet(error, "Exception in TestingJSONReader::ReadDataFile(): %s", e.what());
    return EINVAL;
  }
}
ArrowErrorCode TestingJSONReader::ReadSchema(const std::string& schema_json,
                                             ArrowSchema* out, ArrowError* error) {
  try {
    auto obj = nlohmann::json::parse(schema_json);
    nanoarrow::UniqueSchema schema;

    NANOARROW_RETURN_NOT_OK(
        reader_internal::SetSchema(schema.get(), obj, dictionaries_, error));
    ArrowSchemaMove(schema.get(), out);
    return NANOARROW_OK;
  } catch (nlohmann::json::exception& e) {
    ArrowErrorSet(error, "Exception in TestingJSONReader::ReadSchema(): %s", e.what());
    return EINVAL;
  }
}
ArrowErrorCode TestingJSONReader::ReadField(const std::string& field_json,
                                            ArrowSchema* out, ArrowError* error) {
  try {
    auto obj = nlohmann::json::parse(field_json);
    nanoarrow::UniqueSchema schema;

    NANOARROW_RETURN_NOT_OK(
        reader_internal::SetField(schema.get(), obj, dictionaries_, error));
    ArrowSchemaMove(schema.get(), out);
    return NANOARROW_OK;
  } catch (nlohmann::json::exception& e) {
    ArrowErrorSet(error, "Exception in TestingJSONReader::ReadField(): %s", e.what());
    return EINVAL;
  }
}

ArrowErrorCode TestingJSONReader::ReadBatch(const std::string& batch_json,
                                            const ArrowSchema* schema, ArrowArray* out,
                                            ArrowError* error) {
  try {
    auto obj = nlohmann::json::parse(batch_json);

    // ArrowArrayView to enable validation
    nanoarrow::UniqueArrayView array_view;
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewInitFromSchema(array_view.get(), schema, error));

    // ArrowArray to hold memory
    nanoarrow::UniqueArray array;
    NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromSchema(array.get(), schema, error));
    SetArrayAllocatorRecursive(array.get());

    NANOARROW_RETURN_NOT_OK(reader_internal::SetArrayBatch(
        obj, schema, array_view.get(), array.get(), dictionaries_, error));
    ArrowArrayMove(array.get(), out);
    return NANOARROW_OK;
  } catch (nlohmann::json::exception& e) {
    ArrowErrorSet(error, "Exception in TestingJSONReader::ReadBatch(): %s", e.what());
    return EINVAL;
  }
}
ArrowErrorCode TestingJSONReader::ReadColumn(const std::string& column_json,
                                             const ArrowSchema* schema, ArrowArray* out,
                                             ArrowError* error) {
  try {
    auto obj = nlohmann::json::parse(column_json);

    // ArrowArrayView to enable validation
    nanoarrow::UniqueArrayView array_view;
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewInitFromSchema(array_view.get(), schema, error));

    // ArrowArray to hold memory
    nanoarrow::UniqueArray array;
    NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromSchema(array.get(), schema, error));
    SetArrayAllocatorRecursive(array.get());

    // Parse the JSON into the array
    NANOARROW_RETURN_NOT_OK(reader_internal::SetArrayColumn(
        obj, schema, array_view.get(), array.get(), dictionaries_, error));

    // Return the result
    ArrowArrayMove(array.get(), out);
    return NANOARROW_OK;
  } catch (nlohmann::json::exception& e) {
    ArrowErrorSet(error, "Exception in TestingJSONReader::ReadColumn(): %s", e.what());
    return EINVAL;
  }
}

void TestingJSONReader::SetArrayAllocatorRecursive(ArrowArray* array) {
  for (int i = 0; i < array->n_buffers; i++) {
    ArrowArrayBuffer(array, i)->allocator = allocator_;
  }

  for (int64_t i = 0; i < array->n_children; i++) {
    SetArrayAllocatorRecursive(array->children[i]);
  }

  if (array->dictionary != nullptr) {
    SetArrayAllocatorRecursive(array->dictionary);
  }
}

}  // namespace testing
}  // namespace nanoarrow
