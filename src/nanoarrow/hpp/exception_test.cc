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

#include "nanoarrow/nanoarrow.hpp"

TEST(HppException, Exception) {
  ASSERT_THROW(NANOARROW_THROW_NOT_OK(EINVAL), nanoarrow::Exception);
  ASSERT_NO_THROW(NANOARROW_THROW_NOT_OK(NANOARROW_OK));
  try {
    NANOARROW_THROW_NOT_OK(EINVAL);
  } catch (const nanoarrow::Exception& e) {
    EXPECT_EQ(std::string(e.what()).substr(0, 24), "EINVAL failed with errno");
  }
}
