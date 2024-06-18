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
#include <iostream>

#include "nanoarrow/nanoarrow.hpp"

#ifndef NANOARROW_GTEST_UTIL_HPP_INCLUDED
#define NANOARROW_GTEST_UTIL_HPP_INCLUDED

/// \defgroup nanoarrow_testing Nanoarrow Testing Helpers
///
/// Utilities for testing nanoarrow structures and functions.

namespace nanoarrow {
namespace internal {

inline void PrintTo(const Nothing&, std::ostream* os) { *os << "<NA>"; }

template <typename T>
void PrintTo(const Maybe<T>& m, std::ostream* os) {
  if (m) {
    *os << ::testing::PrintToString(*m);
  } else {
    PrintTo(NA, os);
  }
}

}  // namespace internal
}  // namespace nanoarrow

inline void PrintTo(const ArrowStringView& sv, std::ostream* os) {
  *os << "'";
  os->write(sv.data, static_cast<size_t>(sv.size_bytes));
  *os << "'";
}

#endif
