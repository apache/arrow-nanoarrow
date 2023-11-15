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

#include "nanoarrow/nanoarrow.hpp"

namespace nanoarrow {

namespace testing {

class TestingJSONWriter {
 public:
  ArrowErrorCode WriteColumn(std::ostream& out, const ArrowSchema* field,
                             ArrowArrayView* value) {
    out << "{";
    if (field->name == nullptr) {
      out << R"("name": null)";
    } else {
      out << R"("name": ")" << field->name << R"(")";
    }

    out << R"(, "count": )" << value->length;

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

    switch (value->storage_type) {
      case NANOARROW_TYPE_NA:
      case NANOARROW_TYPE_STRUCT:
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

      default:
        // Not supported
        return ENOTSUP;
    }

    out << "]";
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
    out << "[";
    return NANOARROW_OK;
  }
};

}  // namespace testing
}  // namespace nanoarrow

#endif
