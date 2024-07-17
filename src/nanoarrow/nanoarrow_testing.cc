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

#include "nanoarrow/nanoarrow_testing.hpp"

namespace nanoarrow {

namespace testing {

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
    WriteString(out, ArrowCharView(field->name));
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
    NANOARROW_RETURN_NOT_OK(WriteTypeFromView(out, &dictionary_view));

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
    NANOARROW_RETURN_NOT_OK(WriteTypeFromView(out, &view));

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
  NANOARROW_RETURN_NOT_OK(WriteTypeFromView(out, &view));
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
  NANOARROW_RETURN_NOT_OK(WriteMetadataItem(out, &reader));
  while (reader.remaining_keys > 0) {
    out << ", ";
    NANOARROW_RETURN_NOT_OK(WriteMetadataItem(out, &reader));
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
    WriteString(out, ArrowCharView(field->name));
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
      WriteBitmap(out, value->buffer_views[0].data.as_uint8, value->length);
      break;
  }

  // Write the TYPE_ID element if required
  switch (value->storage_type) {
    case NANOARROW_TYPE_SPARSE_UNION:
    case NANOARROW_TYPE_DENSE_UNION:
      out << R"(, "TYPE_ID": )";
      NANOARROW_RETURN_NOT_OK(WriteOffsetOrTypeID<int8_t>(out, value->buffer_views[0]));
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
      NANOARROW_RETURN_NOT_OK(WriteOffsetOrTypeID<int32_t>(out, value->buffer_views[1]));
      break;
    case NANOARROW_TYPE_LARGE_LIST:
    case NANOARROW_TYPE_LARGE_BINARY:
    case NANOARROW_TYPE_LARGE_STRING:
      out << R"(, "OFFSET": )";
      NANOARROW_RETURN_NOT_OK(WriteOffsetOrTypeID<int64_t>(out, value->buffer_views[1]));
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
      NANOARROW_RETURN_NOT_OK(WriteData(out, value, float_precision_));
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
  NANOARROW_RETURN_NOT_OK(WriteTypeFromView(out, indices_field));

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
}  // namespace testing
}  // namespace nanoarrow
