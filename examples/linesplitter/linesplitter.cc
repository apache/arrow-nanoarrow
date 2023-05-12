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

#include <cerrno>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>

#include "nanoarrow/nanoarrow.hpp"

#include "linesplitter.h"

static int64_t find_newline(const ArrowStringView& src) {
  for (int64_t i = 0; i < src.size_bytes; i++) {
    if (src.data[i] == '\n') {
      return i;
    }
  }

  return src.size_bytes;
}

static int linesplitter_read_internal(const std::string& src, ArrowArray* out,
                                      ArrowError* error) {
  nanoarrow::UniqueArray tmp;
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(tmp.get(), NANOARROW_TYPE_STRING));
  NANOARROW_RETURN_NOT_OK(ArrowArrayStartAppending(tmp.get()));

  ArrowStringView src_view = {src.data(), static_cast<int64_t>(src.size())};
  ArrowStringView line_view;
  int64_t next_newline = -1;
  while ((next_newline = find_newline(src_view)) >= 0) {
    line_view = {src_view.data, next_newline};
    NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(tmp.get(), line_view));
    src_view.data += next_newline + 1;
    src_view.size_bytes -= next_newline + 1;
  }

  NANOARROW_RETURN_NOT_OK(ArrowArrayFinishBuildingDefault(tmp.get(), error));

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}

std::pair<int, std::string> linesplitter_read(const std::string& src, ArrowArray* out) {
  ArrowError error;
  int code = linesplitter_read_internal(src, out, &error);
  if (code != NANOARROW_OK) {
    return {code, std::string(ArrowErrorMessage(&error))};
  } else {
    return {NANOARROW_OK, ""};
  }
}

static int linesplitter_write_internal(ArrowArray* input, std::stringstream& out,
                                       ArrowError* error) {
  nanoarrow::UniqueArrayView input_view;
  ArrowArrayViewInitFromType(input_view.get(), NANOARROW_TYPE_STRING);
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewSetArray(input_view.get(), input, error));

  ArrowStringView item;
  for (int64_t i = 0; i < input->length; i++) {
    if (ArrowArrayViewIsNull(input_view.get(), i)) {
      out << "\n";
    } else {
      item = ArrowArrayViewGetStringUnsafe(input_view.get(), i);
      out << std::string(item.data, item.size_bytes) << "\n";
    }
  }

  return NANOARROW_OK;
}

std::pair<int, std::string> linesplitter_write(ArrowArray* input) {
  std::stringstream out;
  ArrowError error;
  int code = linesplitter_write_internal(input, out, &error);
  if (code != NANOARROW_OK) {
    return {code, std::string(ArrowErrorMessage(&error))};
  } else {
    return {NANOARROW_OK, out.str()};
  }
}
