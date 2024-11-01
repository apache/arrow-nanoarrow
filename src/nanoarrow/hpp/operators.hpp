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

#ifndef NANOARROW_HPP_OPERATORS_HPP_INCLUDED
#define NANOARROW_HPP_OPERATORS_HPP_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "nanoarrow/nanoarrow.h"

NANOARROW_CXX_NAMESPACE_BEGIN

namespace literals {

/// \defgroup nanoarrow_hpp-string_view_helpers ArrowStringView helpers
///
/// Factories and equality comparison for ArrowStringView.
///
/// @{

/// \brief User literal operator allowing ArrowStringView construction like "str"_asv
#if !defined(__clang__) && (defined(__GNUC__) && __GNUC__ < 6)
inline ArrowStringView operator"" _asv(const char* data, size_t size_bytes) {
  return {data, static_cast<int64_t>(size_bytes)};
}
#else
inline ArrowStringView operator""_asv(const char* data, size_t size_bytes) {
  return {data, static_cast<int64_t>(size_bytes)};
}
#endif
// N.B. older GCC requires the space above, newer Clang forbids the space

// @}

}  // namespace literals

NANOARROW_CXX_NAMESPACE_END

/// \brief Equality comparison operator between ArrowStringView
/// \ingroup nanoarrow_hpp-string_view_helpers
inline bool operator==(ArrowStringView l, ArrowStringView r) {
  if (l.size_bytes != r.size_bytes) return false;
  return memcmp(l.data, r.data, l.size_bytes) == 0;
}

#endif
