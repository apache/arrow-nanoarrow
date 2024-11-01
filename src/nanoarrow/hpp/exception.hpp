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

#ifndef NANOARROW_HPP_EXCEPTION_HPP_INCLUDED
#define NANOARROW_HPP_EXCEPTION_HPP_INCLUDED

#include <exception>
#include <string>

#include "nanoarrow/nanoarrow.h"

NANOARROW_CXX_NAMESPACE_BEGIN

/// \defgroup nanoarrow_hpp-errors Error handling helpers
///
/// Most functions in the C API return an ArrowErrorCode to communicate
/// possible failure. Except where documented, it is usually not safe to
/// continue after a non-zero value has been returned. While the
/// nanoarrow C++ helpers do not throw any exceptions of their own,
/// these helpers are provided to facilitate using the nanoarrow C++ helpers
/// in frameworks where this is a useful error handling idiom.
///
/// @{

class Exception : public std::exception {
 public:
  Exception(const std::string& msg) : msg_(msg) {}
  const char* what() const noexcept { return msg_.c_str(); }

 private:
  std::string msg_;
};

#if defined(NANOARROW_DEBUG)
#define _NANOARROW_THROW_NOT_OK_IMPL(NAME, EXPR, EXPR_STR)                      \
  do {                                                                          \
    const int NAME = (EXPR);                                                    \
    if (NAME) {                                                                 \
      throw nanoarrow::Exception(                                               \
          std::string(EXPR_STR) + std::string(" failed with errno ") +          \
          std::to_string(NAME) + std::string("\n * ") + std::string(__FILE__) + \
          std::string(":") + std::to_string(__LINE__) + std::string("\n"));     \
    }                                                                           \
  } while (0)
#else
#define _NANOARROW_THROW_NOT_OK_IMPL(NAME, EXPR, EXPR_STR)            \
  do {                                                                \
    const int NAME = (EXPR);                                          \
    if (NAME) {                                                       \
      throw nanoarrow::Exception(std::string(EXPR_STR) +              \
                                 std::string(" failed with errno ") + \
                                 std::to_string(NAME));               \
    }                                                                 \
  } while (0)
#endif

#define NANOARROW_THROW_NOT_OK(EXPR)                                                   \
  _NANOARROW_THROW_NOT_OK_IMPL(_NANOARROW_MAKE_NAME(errno_status_, __COUNTER__), EXPR, \
                               #EXPR)

/// @}

NANOARROW_CXX_NAMESPACE_END

#endif
