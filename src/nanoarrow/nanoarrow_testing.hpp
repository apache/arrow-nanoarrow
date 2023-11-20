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

#ifndef NANOARROW_TESTING_HPP_INCLUDED
#define NANOARROW_TESTING_HPP_INCLUDED

#include <iostream>
#include <string>

#include "nanoarrow.hpp"

namespace nanoarrow {

namespace testing {

class TestingJSON {
 public:
  static ArrowErrorCode WriteBatch(std::ostream& out, const ArrowSchema* schema,
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

  static ArrowErrorCode WriteColumn(std::ostream& out, const ArrowSchema* field,
                                    ArrowArrayView* value) {
    out << "{";

    // Write schema->name (may be null)
    if (field->name == nullptr) {
      out << R"("name": null)";
    } else {
      out << R"("name": ")" << field->name << R"(")";
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
  static void WriteBitmap(std::ostream& out, const uint8_t* bits, int64_t length) {
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
  static ArrowErrorCode WriteOffsetOrTypeID(std::ostream& out, ArrowBufferView content) {
    if (content.size_bytes == 0) {
      out << "[]";
      return NANOARROW_OK;
    }

    const T* values = reinterpret_cast<const T*>(content.data.data);
    int64_t n_values = content.size_bytes / sizeof(T);

    out << "[";

    if (sizeof(T) == sizeof(int64_t)) {
      out << R"(")" << values[0] << R"(")";
      for (int64_t i = 1; i < n_values; i++) {
        out << R"(, ")" << values[i] << R"(")";
      }
    } else {
      out << values[0];
      for (int64_t i = 1; i < n_values; i++) {
        out << ", " << static_cast<int64_t>(values[i]);
      }
    }

    out << "]";
    return NANOARROW_OK;
  }

  static ArrowErrorCode WriteData(std::ostream& out, ArrowArrayView* value) {
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
        // Regular JSON integers
        out << ArrowArrayViewGetIntUnsafe(value, 0);
        for (int64_t i = 1; i < value->length; i++) {
          out << ", " << ArrowArrayViewGetIntUnsafe(value, i);
        }
        break;
      case NANOARROW_TYPE_INT64:
        // Strings
        out << R"(")" << ArrowArrayViewGetIntUnsafe(value, 0) << R"(")";
        for (int64_t i = 1; i < value->length; i++) {
          out << R"(, ")" << ArrowArrayViewGetIntUnsafe(value, i) << R"(")";
        }
        break;
      case NANOARROW_TYPE_UINT64:
        // Strings
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

  static ArrowErrorCode WriteString(std::ostream& out, ArrowStringView value) {
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

  static ArrowErrorCode WriteBytes(std::ostream& out, ArrowBufferView value) {
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

  static ArrowErrorCode WriteChildren(std::ostream& out, const ArrowSchema* field,
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

}  // namespace testing
}  // namespace nanoarrow

#endif
