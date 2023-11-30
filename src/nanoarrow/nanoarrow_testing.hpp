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

#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "nanoarrow.hpp"

#ifndef NANOARROW_TESTING_HPP_INCLUDED
#define NANOARROW_TESTING_HPP_INCLUDED

/// \defgroup nanoarrow_testing Nanoarrow Testing Helpers
///
/// Utilities for testing nanoarrow structures and functions.

namespace nanoarrow {

namespace testing {

/// \defgroup nanoarrow_testing-json Integration test helpers
///
/// See testing format documentation for details of the JSON representation. This
/// representation is not canonical but can be used to implement integration tests with
/// other implementations.
///
/// @{

/// \brief Writer for the Arrow integration testing JSON format
class TestingJSONWriter {
 public:
  /// \brief Write an ArrowArrayStream as a data file JSON object to out
  ///
  /// Creates output like `{"schema": {...}, "batches": [...], ...}`.
  ArrowErrorCode WriteDataFile(std::ostream& out, ArrowArrayStream* stream) {
    if (stream == nullptr || stream->release == nullptr) {
      return EINVAL;
    }

    out << R"({"schema": )";

    nanoarrow::UniqueSchema schema;
    NANOARROW_RETURN_NOT_OK(stream->get_schema(stream, schema.get()));
    NANOARROW_RETURN_NOT_OK(WriteSchema(out, schema.get()));

    nanoarrow::UniqueArrayView array_view;
    NANOARROW_RETURN_NOT_OK(
        ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), nullptr));

    out << R"(, "batches": [)";

    nanoarrow::UniqueArray array;
    std::string sep;
    do {
      NANOARROW_RETURN_NOT_OK(stream->get_next(stream, array.get()));
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

    out << "]}";

    return NANOARROW_OK;
  }

  /// \brief Write a schema to out
  ///
  /// Creates output like `{"fields": [...], "metadata": [...]}`.
  ArrowErrorCode WriteSchema(std::ostream& out, const ArrowSchema* schema) {
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
    out << R"(, "metadata": )";
    NANOARROW_RETURN_NOT_OK(WriteMetadata(out, schema->metadata));

    out << "}";
    return NANOARROW_OK;
  }

  /// \brief Write a field to out
  ///
  /// Creates output like `{"name" : "col", "type": {...}, ...}`
  ArrowErrorCode WriteField(std::ostream& out, const ArrowSchema* field) {
    ArrowSchemaView view;
    NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&view, (ArrowSchema*)field, nullptr));

    out << "{";

    // Write schema->name (may be null)
    if (field->name == nullptr) {
      out << R"("name": null)";
    } else {
      out << R"("name": )";
      NANOARROW_RETURN_NOT_OK(WriteString(out, ArrowCharView(field->name)));
    }

    // Write nullability
    if (field->flags & ARROW_FLAG_NULLABLE) {
      out << R"(, "nullable": true)";
    } else {
      out << R"(, "nullable": false)";
    }

    // Write type
    out << R"(, "type": )";
    NANOARROW_RETURN_NOT_OK(WriteType(out, &view));

    // Write children
    out << R"(, "children": )";
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

    // TODO: Dictionary (currently fails at WriteType)

    // Write metadata
    out << R"(, "metadata": )";
    NANOARROW_RETURN_NOT_OK(WriteMetadata(out, field->metadata));

    out << "}";
    return NANOARROW_OK;
  }

  /// \brief Write the type portion of a field
  ///
  /// Creates output like `{"name": "int", ...}`
  ArrowErrorCode WriteType(std::ostream& out, const ArrowSchema* field) {
    ArrowSchemaView view;
    NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&view, (ArrowSchema*)field, nullptr));
    NANOARROW_RETURN_NOT_OK(WriteType(out, &view));
    return NANOARROW_OK;
  }

  /// \brief Write a "batch" to out
  ///
  /// Creates output like `{"count": 123, "columns": [...]}`.
  ArrowErrorCode WriteBatch(std::ostream& out, const ArrowSchema* schema,
                            ArrowArrayView* value) {
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

  /// \brief Write a column to out
  ///
  /// Creates output like `{"name": "col", "count": 123, "VALIDITY": [...], ...}`.
  ArrowErrorCode WriteColumn(std::ostream& out, const ArrowSchema* field,
                             ArrowArrayView* value) {
    out << "{";

    // Write schema->name (may be null)
    if (field->name == nullptr) {
      out << R"("name": null)";
    } else {
      out << R"("name": )";
      NANOARROW_RETURN_NOT_OK(WriteString(out, ArrowCharView(field->name)));
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
        NANOARROW_RETURN_NOT_OK(
            WriteOffsetOrTypeID<int32_t>(out, value->buffer_views[1]));
        break;
      case NANOARROW_TYPE_LARGE_LIST:
      case NANOARROW_TYPE_LARGE_BINARY:
      case NANOARROW_TYPE_LARGE_STRING:
        out << R"(, "OFFSET": )";
        NANOARROW_RETURN_NOT_OK(
            WriteOffsetOrTypeID<int64_t>(out, value->buffer_views[1]));
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
      case NANOARROW_TYPE_DENSE_UNION:
      case NANOARROW_TYPE_SPARSE_UNION:
        break;
      default:
        out << R"(, "DATA": )";
        NANOARROW_RETURN_NOT_OK(WriteData(out, value));
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
    return NANOARROW_OK;
  }

 private:
  ArrowErrorCode WriteType(std::ostream& out, const ArrowSchemaView* field) {
    ArrowType type;
    if (field->extension_name.data != nullptr) {
      type = field->storage_type;
    } else {
      type = field->type;
    }

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
        out << R"("name": "union", "mode": "DENSE", "typeIds": [)"
            << field->union_type_ids << "]";
        break;
      case NANOARROW_TYPE_SPARSE_UNION:
        out << R"("name": "union", "mode": "SPARSE", "typeIds": [)"
            << field->union_type_ids << "]";
        break;

      default:
        // Not supported
        return ENOTSUP;
    }

    out << "}";
    return NANOARROW_OK;
  }

  ArrowErrorCode WriteMetadata(std::ostream& out, const char* metadata) {
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

  ArrowErrorCode WriteMetadataItem(std::ostream& out, ArrowMetadataReader* reader) {
    ArrowStringView key;
    ArrowStringView value;
    NANOARROW_RETURN_NOT_OK(ArrowMetadataReaderRead(reader, &key, &value));
    out << R"({"key": )";
    NANOARROW_RETURN_NOT_OK(WriteString(out, key));
    out << R"(, "value": )";
    NANOARROW_RETURN_NOT_OK(WriteString(out, value));
    out << "}";
    return NANOARROW_OK;
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

  ArrowErrorCode WriteData(std::ostream& out, ArrowArrayView* value) {
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
        // Regular JSON integers (i.e., 123456)
        out << ArrowArrayViewGetIntUnsafe(value, 0);
        for (int64_t i = 1; i < value->length; i++) {
          out << ", " << ArrowArrayViewGetIntUnsafe(value, i);
        }
        break;
      case NANOARROW_TYPE_INT64:
        // Quoted integers to avoid overflow (i.e., "123456")
        out << R"(")" << ArrowArrayViewGetIntUnsafe(value, 0) << R"(")";
        for (int64_t i = 1; i < value->length; i++) {
          out << R"(, ")" << ArrowArrayViewGetIntUnsafe(value, i) << R"(")";
        }
        break;
      case NANOARROW_TYPE_UINT64:
        // Quoted integers to avoid overflow (i.e., "123456")
        out << R"(")" << ArrowArrayViewGetUIntUnsafe(value, 0) << R"(")";
        for (int64_t i = 1; i < value->length; i++) {
          out << R"(, ")" << ArrowArrayViewGetUIntUnsafe(value, i) << R"(")";
        }
        break;

      case NANOARROW_TYPE_FLOAT:
      case NANOARROW_TYPE_DOUBLE: {
        // JSON number to 3 decimal places
        LocalizedStream local_stream_opt(out);
        local_stream_opt.SetFixed(3);

        out << ArrowArrayViewGetDoubleUnsafe(value, 0);
        for (int64_t i = 1; i < value->length; i++) {
          out << ", " << ArrowArrayViewGetDoubleUnsafe(value, i);
        }
        break;
      }

      case NANOARROW_TYPE_STRING:
      case NANOARROW_TYPE_LARGE_STRING:
        NANOARROW_RETURN_NOT_OK(
            WriteString(out, ArrowArrayViewGetStringUnsafe(value, 0)));
        for (int64_t i = 1; i < value->length; i++) {
          out << ", ";
          NANOARROW_RETURN_NOT_OK(
              WriteString(out, ArrowArrayViewGetStringUnsafe(value, i)));
        }
        break;

      case NANOARROW_TYPE_BINARY:
      case NANOARROW_TYPE_LARGE_BINARY:
      case NANOARROW_TYPE_FIXED_SIZE_BINARY: {
        NANOARROW_RETURN_NOT_OK(WriteBytes(out, ArrowArrayViewGetBytesUnsafe(value, 0)));
        for (int64_t i = 1; i < value->length; i++) {
          out << ", ";
          NANOARROW_RETURN_NOT_OK(
              WriteBytes(out, ArrowArrayViewGetBytesUnsafe(value, i)));
        }
        break;
      }

      default:
        // Not supported
        return ENOTSUP;
    }

    out << "]";
    return NANOARROW_OK;
  }

  ArrowErrorCode WriteString(std::ostream& out, ArrowStringView value) {
    out << R"(")";

    for (int64_t i = 0; i < value.size_bytes; i++) {
      char c = value.data[i];
      if (c == '"') {
        out << R"(\")";
      } else if (c == '\\') {
        out << R"(\\)";
      } else if (c < 0) {
        // Not supporting multibyte unicode yet
        return ENOTSUP;
      } else if (c < 20) {
        // Data in the arrow-testing repo has a lot of content that requires escaping
        // in this way (\uXXXX).
        uint16_t utf16_bytes = static_cast<uint16_t>(c);

        char utf16_esc[7];
        utf16_esc[6] = '\0';
        snprintf(utf16_esc, sizeof(utf16_esc), R"(\u%04x)", utf16_bytes);
        out << utf16_esc;
      } else {
        out << c;
      }
    }

    out << R"(")";
    return NANOARROW_OK;
  }

  ArrowErrorCode WriteBytes(std::ostream& out, ArrowBufferView value) {
    out << R"(")";
    char hex[3];
    hex[2] = '\0';

    for (int64_t i = 0; i < value.size_bytes; i++) {
      snprintf(hex, sizeof(hex), "%02X", static_cast<int>(value.data.as_uint8[i]));
      out << hex;
    }
    out << R"(")";
    return NANOARROW_OK;
  }

  ArrowErrorCode WriteChildren(std::ostream& out, const ArrowSchema* field,
                               ArrowArrayView* value) {
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
};

/// \brief Reader for the Arrow integration testing JSON format
class TestingJSONReader {
  using json = nlohmann::json;

 public:
  /// \brief Read JSON representing a data file object
  ///
  /// Read a JSON object in the form `{"schema": {...}, "batches": [...], ...}`,
  /// propagating `out` on success.
  ArrowErrorCode ReadDataFile(const std::string& data_file_json, ArrowArrayStream* out,
                              ArrowError* error = nullptr) {
    try {
      auto obj = json::parse(data_file_json);
      NANOARROW_RETURN_NOT_OK(Check(obj.is_object(), error, "data file must be object"));
      NANOARROW_RETURN_NOT_OK(
          Check(obj.contains("schema"), error, "data file missing key 'schema'"));

      // Read Schema
      nanoarrow::UniqueSchema schema;
      NANOARROW_RETURN_NOT_OK(SetSchema(schema.get(), obj["schema"], error));

      NANOARROW_RETURN_NOT_OK(
          Check(obj.contains("batches"), error, "data file missing key 'batches'"));
      const auto& batches = obj["batches"];
      NANOARROW_RETURN_NOT_OK(
          Check(batches.is_array(), error, "data file batches must be array"));

      // Populate ArrayView
      nanoarrow::UniqueArrayView array_view;
      NANOARROW_RETURN_NOT_OK(
          ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), error));

      // Initialize ArrayStream with required capacity
      nanoarrow::UniqueArrayStream stream;
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(
          ArrowBasicArrayStreamInit(stream.get(), schema.get(), batches.size()), error);

      // Populate ArrayStream batches
      for (size_t i = 0; i < batches.size(); i++) {
        nanoarrow::UniqueArray array;
        NANOARROW_RETURN_NOT_OK(
            ArrowArrayInitFromArrayView(array.get(), array_view.get(), error));
        NANOARROW_RETURN_NOT_OK(
            SetArrayBatch(batches[i], array_view.get(), array.get(), error));
        ArrowBasicArrayStreamSetArray(stream.get(), i, array.get());
      }

      ArrowArrayStreamMove(stream.get(), out);
      return NANOARROW_OK;
    } catch (json::exception& e) {
      ArrowErrorSet(error, "Exception in TestingJSONReader::ReadDataFile(): %s",
                    e.what());
      return EINVAL;
    }
  }

  /// \brief Read JSON representing a Schema
  ///
  /// Reads a JSON object in the form `{"fields": [...], "metadata": [...]}`,
  /// propagating `out` on success.
  ArrowErrorCode ReadSchema(const std::string& schema_json, ArrowSchema* out,
                            ArrowError* error = nullptr) {
    try {
      auto obj = json::parse(schema_json);
      nanoarrow::UniqueSchema schema;

      NANOARROW_RETURN_NOT_OK(SetSchema(schema.get(), obj, error));
      ArrowSchemaMove(schema.get(), out);
      return NANOARROW_OK;
    } catch (json::exception& e) {
      ArrowErrorSet(error, "Exception in TestingJSONReader::ReadSchema(): %s", e.what());
      return EINVAL;
    }
  }

  /// \brief Read JSON representing a Field
  ///
  /// Read a JSON object in the form `{"name" : "col", "type": {...}, ...}`,
  /// propagating `out` on success.
  ArrowErrorCode ReadField(const std::string& field_json, ArrowSchema* out,
                           ArrowError* error = nullptr) {
    try {
      auto obj = json::parse(field_json);
      nanoarrow::UniqueSchema schema;

      NANOARROW_RETURN_NOT_OK(SetField(schema.get(), obj, error));
      ArrowSchemaMove(schema.get(), out);
      return NANOARROW_OK;
    } catch (json::exception& e) {
      ArrowErrorSet(error, "Exception in TestingJSONReader::ReadField(): %s", e.what());
      return EINVAL;
    }
  }

  /// \brief Read JSON representing a RecordBatch
  ///
  /// Read a JSON object in the form `{"count": 123, "columns": [...]}`, propagating `out`
  /// on success.
  ArrowErrorCode ReadBatch(const std::string& batch_json, const ArrowSchema* schema,
                           ArrowArray* out, ArrowError* error = nullptr) {
    try {
      auto obj = json::parse(batch_json);

      // ArrowArrayView to enable validation
      nanoarrow::UniqueArrayView array_view;
      NANOARROW_RETURN_NOT_OK(ArrowArrayViewInitFromSchema(
          array_view.get(), const_cast<ArrowSchema*>(schema), error));

      // ArrowArray to hold memory
      nanoarrow::UniqueArray array;
      NANOARROW_RETURN_NOT_OK(
          ArrowArrayInitFromSchema(array.get(), const_cast<ArrowSchema*>(schema), error));

      NANOARROW_RETURN_NOT_OK(SetArrayBatch(obj, array_view.get(), array.get(), error));
      ArrowArrayMove(array.get(), out);
      return NANOARROW_OK;
    } catch (json::exception& e) {
      ArrowErrorSet(error, "Exception in TestingJSONReader::ReadBatch(): %s", e.what());
      return EINVAL;
    }
  }

  /// \brief Read JSON representing a Column
  ///
  /// Read a JSON object in the form
  /// `{"name": "col", "count": 123, "VALIDITY": [...], ...}`, propagating
  /// `out` on success.
  ArrowErrorCode ReadColumn(const std::string& column_json, const ArrowSchema* schema,
                            ArrowArray* out, ArrowError* error = nullptr) {
    try {
      auto obj = json::parse(column_json);

      // ArrowArrayView to enable validation
      nanoarrow::UniqueArrayView array_view;
      NANOARROW_RETURN_NOT_OK(ArrowArrayViewInitFromSchema(
          array_view.get(), const_cast<ArrowSchema*>(schema), error));

      // ArrowArray to hold memory
      nanoarrow::UniqueArray array;
      NANOARROW_RETURN_NOT_OK(
          ArrowArrayInitFromSchema(array.get(), const_cast<ArrowSchema*>(schema), error));

      // Parse the JSON into the array
      NANOARROW_RETURN_NOT_OK(SetArrayColumn(obj, array_view.get(), array.get(), error));

      // Return the result
      ArrowArrayMove(array.get(), out);
      return NANOARROW_OK;
    } catch (json::exception& e) {
      ArrowErrorSet(error, "Exception in TestingJSONReader::ReadColumn(): %s", e.what());
      return EINVAL;
    }
  }

 private:
  ArrowErrorCode SetSchema(ArrowSchema* schema, const json& value, ArrowError* error) {
    NANOARROW_RETURN_NOT_OK(
        Check(value.is_object(), error, "Expected Schema to be a JSON object"));
    NANOARROW_RETURN_NOT_OK(
        Check(value.contains("fields"), error, "Schema missing key 'fields'"));
    NANOARROW_RETURN_NOT_OK(
        Check(value.contains("metadata"), error, "Schema missing key 'metadata'"));

    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRUCT), error);

    const auto& fields = value["fields"];
    NANOARROW_RETURN_NOT_OK(
        Check(fields.is_array(), error, "Schema fields must be array"));
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaAllocateChildren(schema, fields.size()),
                                       error);
    for (int64_t i = 0; i < schema->n_children; i++) {
      NANOARROW_RETURN_NOT_OK(SetField(schema->children[i], fields[i], error));
    }

    NANOARROW_RETURN_NOT_OK(SetMetadata(schema, value["metadata"], error));

    // Validate!
    ArrowSchemaView schema_view;
    NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&schema_view, schema, error));
    return NANOARROW_OK;
  }

  ArrowErrorCode SetField(ArrowSchema* schema, const json& value, ArrowError* error) {
    NANOARROW_RETURN_NOT_OK(
        Check(value.is_object(), error, "Expected Field to be a JSON object"));
    NANOARROW_RETURN_NOT_OK(
        Check(value.contains("name"), error, "Field missing key 'name'"));
    NANOARROW_RETURN_NOT_OK(
        Check(value.contains("nullable"), error, "Field missing key 'nullable'"));
    NANOARROW_RETURN_NOT_OK(
        Check(value.contains("type"), error, "Field missing key 'type'"));
    NANOARROW_RETURN_NOT_OK(
        Check(value.contains("children"), error, "Field missing key 'children'"));
    NANOARROW_RETURN_NOT_OK(
        Check(value.contains("metadata"), error, "Field missing key 'metadata'"));

    ArrowSchemaInit(schema);

    const auto& name = value["name"];
    NANOARROW_RETURN_NOT_OK(Check(name.is_string() || name.is_null(), error,
                                  "Field name must be string or null"));
    if (name.is_string()) {
      auto name_str = name.get<std::string>();
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaSetName(schema, name_str.c_str()),
                                         error);
    }

    const auto& nullable = value["nullable"];
    NANOARROW_RETURN_NOT_OK(
        Check(nullable.is_boolean(), error, "Field nullable must be boolean"));
    if (nullable.get<bool>()) {
      schema->flags |= ARROW_FLAG_NULLABLE;
    } else {
      schema->flags &= ~ARROW_FLAG_NULLABLE;
    }

    NANOARROW_RETURN_NOT_OK(SetType(schema, value["type"], error));

    const auto& children = value["children"];
    NANOARROW_RETURN_NOT_OK(
        Check(children.is_array(), error, "Field children must be array"));
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowSchemaAllocateChildren(schema, children.size()), error);
    for (int64_t i = 0; i < schema->n_children; i++) {
      NANOARROW_RETURN_NOT_OK(SetField(schema->children[i], children[i], error));
    }

    NANOARROW_RETURN_NOT_OK(SetMetadata(schema, value["metadata"], error));

    // Validate!
    ArrowSchemaView schema_view;
    NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&schema_view, schema, error));
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
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(
          ArrowSchemaSetType(schema, NANOARROW_TYPE_STRING), error);
    } else if (name_str == "largeutf8") {
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(
          ArrowSchemaSetType(schema, NANOARROW_TYPE_LARGE_STRING), error);
    } else if (name_str == "binary") {
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(
          ArrowSchemaSetType(schema, NANOARROW_TYPE_BINARY), error);
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

  ArrowErrorCode SetTypeInt(ArrowSchema* schema, const json& value, ArrowError* error) {
    NANOARROW_RETURN_NOT_OK(Check(value.contains("bitWidth"), error,
                                  "Type[name=='int'] missing key 'bitWidth'"));
    NANOARROW_RETURN_NOT_OK(Check(value.contains("isSigned"), error,
                                  "Type[name=='int'] missing key 'isSigned'"));

    const auto& bitwidth = value["bitWidth"];
    NANOARROW_RETURN_NOT_OK(Check(bitwidth.is_number_integer(), error,
                                  "Type[name=='int'] bitWidth must be integer"));

    const auto& issigned = value["isSigned"];
    NANOARROW_RETURN_NOT_OK(Check(issigned.is_boolean(), error,
                                  "Type[name=='int'] isSigned must be boolean"));

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
    NANOARROW_RETURN_NOT_OK(
        Check(value.contains("byteWidth"), error,
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

  ArrowErrorCode SetTypeDecimal(ArrowSchema* schema, const json& value,
                                ArrowError* error) {
    NANOARROW_RETURN_NOT_OK(Check(value.contains("bitWidth"), error,
                                  "Type[name=='decimal'] missing key 'bitWidth'"));
    NANOARROW_RETURN_NOT_OK(Check(value.contains("precision"), error,
                                  "Type[name=='decimal'] missing key 'precision'"));
    NANOARROW_RETURN_NOT_OK(Check(value.contains("scale"), error,
                                  "Type[name=='decimal'] missing key 'scale'"));

    const auto& bitWidth = value["bitWidth"];
    NANOARROW_RETURN_NOT_OK(Check(bitWidth.is_number_integer(), error,
                                  "Type[name=='decimal'] bitWidth must be integer"));

    ArrowType type;
    switch (bitWidth.get<int>()) {
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
    NANOARROW_RETURN_NOT_OK(
        Check(list_size.is_number_integer(), error,
              "Type[name=='fixedsizelist'] listSize must be integer"));

    std::stringstream format_builder;
    format_builder << "+w:" << list_size;
    std::string format = format_builder.str();
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowSchemaSetFormat(schema, format.c_str()),
                                       error);
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
        NANOARROW_RETURN_NOT_OK(
            Check(type_id.is_number_integer(), error,
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

  ArrowErrorCode SetArrayBatch(const json& value, ArrowArrayView* array_view,
                               ArrowArray* array, ArrowError* error) {
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
    NANOARROW_RETURN_NOT_OK(Check(columns.size() == array_view->n_children, error,
                                  "RecordBatch children has incorrect size"));

    for (int64_t i = 0; i < array_view->n_children; i++) {
      NANOARROW_RETURN_NOT_OK(
          SetArrayColumn(columns[i], array_view->children[i], array->children[i], error));
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

  ArrowErrorCode SetArrayColumn(const json& value, ArrowArrayView* array_view,
                                ArrowArray* array, ArrowError* error,
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
    NANOARROW_RETURN_NOT_OK(
        Check(array_view->n_children == 0 || value.contains("children"), error,
              error_prefix + "missing key children"));

    if (value.contains("children")) {
      const auto& children = value["children"];
      NANOARROW_RETURN_NOT_OK(
          Check(children.is_array(), error, error_prefix + "children must be array"));
      NANOARROW_RETURN_NOT_OK(Check(children.size() == array_view->n_children, error,
                                    error_prefix + "children has incorrect size"));

      for (int64_t i = 0; i < array_view->n_children; i++) {
        NANOARROW_RETURN_NOT_OK(SetArrayColumn(children[i], array_view->children[i],
                                               array->children[i], error, error_prefix));
      }
    }

    // Build buffers
    for (int i = 0; i < NANOARROW_MAX_FIXED_BUFFERS; i++) {
      NANOARROW_RETURN_NOT_OK(
          PrefixError(SetArrayColumnBuffers(value, array_view, array, i, error), error,
                      error_prefix));
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

    // Validate the array view
    NANOARROW_RETURN_NOT_OK(PrefixError(
        ArrowArrayViewValidate(array_view, NANOARROW_VALIDATION_LEVEL_FULL, error), error,
        error_prefix + "failed to validate: "));

    // Flush length and buffer pointers to the Array
    array->length = array_view->length;
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowArrayFinishBuilding(array, NANOARROW_VALIDATION_LEVEL_NONE, nullptr), error);

    return NANOARROW_OK;
  }

  ArrowErrorCode SetArrayColumnBuffers(const json& value, ArrowArrayView* array_view,
                                       ArrowArray* array, int buffer_i,
                                       ArrowError* error) {
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
        NANOARROW_RETURN_NOT_OK(
            Check(value.contains("DATA"), error, "missing key 'DATA'"));
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
            return SetBufferInt<int32_t>(data, buffer, error);
          case NANOARROW_TYPE_UINT32:
            return SetBufferInt<uint32_t>(data, buffer, error);
          case NANOARROW_TYPE_INT64:
            return SetBufferInt<int64_t>(data, buffer, error);
          case NANOARROW_TYPE_UINT64:
            return SetBufferInt<uint64_t, uint64_t>(data, buffer, error);

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

  ArrowErrorCode SetBufferBitmap(const json& value, ArrowBitmap* bitmap,
                                 ArrowError* error) {
    NANOARROW_RETURN_NOT_OK(
        Check(value.is_array(), error, "bitmap buffer must be array"));

    for (const auto& item : value) {
      // Some example files write bitmaps as [true, false, true] but the documentation
      // says [1, 0, 1]. Accept both for simplicity.
      NANOARROW_RETURN_NOT_OK(Check(item.is_boolean() || item.is_number_integer(), error,
                                    "bitmap item must be bool or integer"));
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBitmapAppend(bitmap, item.get<int>(), 1),
                                         error);
    }

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

  template <typename T, typename BiggerT = int64_t>
  ArrowErrorCode SetBufferIntItem(const json& item, ArrowBuffer* buffer,
                                  ArrowError* error) {
    if (item.is_string()) {
      try {
        // The JSON parser here can handle up to 2^64 - 1
        auto item_int = json::parse(item.get<std::string>());
        return SetBufferIntItem<T, BiggerT>(item_int, buffer, error);
      } catch (json::parse_error& e) {
        ArrowErrorSet(error,
                      "integer buffer item encoded as string must parse as integer: %s",
                      item.dump().c_str());
        return EINVAL;
      }
    }

    NANOARROW_RETURN_NOT_OK(
        Check(item.is_number_integer(), error,
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

    T buffer_value = item_int;
    NANOARROW_RETURN_NOT_OK_WITH_ERROR(
        ArrowBufferAppend(buffer, &buffer_value, sizeof(T)), error);

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

      T buffer_value = item_dbl;
      NANOARROW_RETURN_NOT_OK_WITH_ERROR(
          ArrowBufferAppend(buffer, &buffer_value, sizeof(T)), error);
    }

    return NANOARROW_OK;
  }

  template <typename T>
  ArrowErrorCode SetBufferString(const json& value, ArrowBuffer* offsets,
                                 ArrowBuffer* data, ArrowError* error) {
    NANOARROW_RETURN_NOT_OK(
        Check(value.is_array(), error, "utf8 data buffer must be array"));

    // Check offsets against values
    const T* expected_offset = reinterpret_cast<const T*>(offsets->data);
    NANOARROW_RETURN_NOT_OK(Check(
        offsets->size_bytes == ((value.size() + 1) * sizeof(T)), error,
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
                                    "Expected offset value " +
                                        std::to_string(last_offset) +
                                        " at utf8 data buffer item " + item.dump()));
    }

    return NANOARROW_OK;
  }

  template <typename T>
  ArrowErrorCode SetBufferBinary(const json& value, ArrowBuffer* offsets,
                                 ArrowBuffer* data, ArrowError* error) {
    NANOARROW_RETURN_NOT_OK(
        Check(value.is_array(), error, "binary data buffer must be array"));

    // Check offsets against values if not fixed size
    const T* expected_offset = reinterpret_cast<const T*>(offsets->data);
    NANOARROW_RETURN_NOT_OK(Check(
        offsets->size_bytes == ((value.size() + 1) * sizeof(T)), error,
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

  ArrowErrorCode AppendBinaryElement(const json& item, ArrowBuffer* data,
                                     ArrowError* error) {
    NANOARROW_RETURN_NOT_OK(
        Check(item.is_string(), error, "binary data buffer item must be string"));
    auto item_str = item.get<std::string>();

    int64_t item_size_bytes = item_str.size() / 2;
    NANOARROW_RETURN_NOT_OK(Check((item_size_bytes * 2) == item_str.size(), error,
                                  "binary data buffer item must have even size"));

    NANOARROW_RETURN_NOT_OK_WITH_ERROR(ArrowBufferReserve(data, item_size_bytes), error);
    for (int64_t i = 0; i < item_str.size(); i += 2) {
      std::string byte_hex = item_str.substr(i, 2);
      char* end_ptr;
      uint8_t byte = std::strtoul(byte_hex.data(), &end_ptr, 16);
      NANOARROW_RETURN_NOT_OK(
          Check(end_ptr == (byte_hex.data() + 2), error,
                "binary data buffer item must contain a valid hex-encoded byte string"));

      data->data[data->size_bytes] = byte;
      data->size_bytes++;
    }

    return NANOARROW_OK;
  }

  ArrowErrorCode PrefixError(ArrowErrorCode value, ArrowError* error,
                             const std::string& prefix) {
    if (value != NANOARROW_OK && error != nullptr) {
      std::string msg = prefix + error->message;
      ArrowErrorSet(error, "%s", msg.c_str());
    }

    return value;
  }

  ArrowErrorCode Check(bool value, ArrowError* error, const std::string& err) {
    if (value) {
      return NANOARROW_OK;
    } else {
      ArrowErrorSet(error, "%s", err.c_str());
      return EINVAL;
    }
  }
};

/// @}

}  // namespace testing
}  // namespace nanoarrow

#endif
