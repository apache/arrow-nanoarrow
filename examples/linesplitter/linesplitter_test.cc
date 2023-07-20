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

#include <gtest/gtest.h>

#include <nanoarrow/nanoarrow.hpp>

#include "linesplitter.h"

TEST(Linesplitter, LinesplitterRoundtrip) {
  nanoarrow::UniqueArray out;
  auto result = linesplitter_read("line1\nline2\nline3", out.get());
  ASSERT_EQ(result.first, 0);
  ASSERT_EQ(result.second, "");

  ASSERT_EQ(out->length, 3);

  nanoarrow::UniqueArrayView out_view;
  ArrowArrayViewInitFromType(out_view.get(), NANOARROW_TYPE_STRING);
  ASSERT_EQ(ArrowArrayViewSetArray(out_view.get(), out.get(), nullptr), 0);
  ArrowStringView item;

  item = ArrowArrayViewGetStringUnsafe(out_view.get(), 0);
  ASSERT_EQ(std::string(item.data, item.size_bytes), "line1");

  item = ArrowArrayViewGetStringUnsafe(out_view.get(), 1);
  ASSERT_EQ(std::string(item.data, item.size_bytes), "line2");

  item = ArrowArrayViewGetStringUnsafe(out_view.get(), 2);
  ASSERT_EQ(std::string(item.data, item.size_bytes), "line3");

  auto result2 = linesplitter_write(out.get());
  ASSERT_EQ(result2.first, 0);
  ASSERT_EQ(result2.second, "line1\nline2\nline3\n");
}
