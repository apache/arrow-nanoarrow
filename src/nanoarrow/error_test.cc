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
#include <cstring>
#include <string>

#include <gtest/gtest.h>

#include "nanoarrow/nanoarrow.h"

TEST(ErrorTest, ErrorTestSet) {
  ArrowError error;
  EXPECT_EQ(ArrowErrorSet(&error, "there were %d foxes", 4), NANOARROW_OK);
  EXPECT_STREQ(ArrowErrorMessage(&error), "there were 4 foxes");
}

TEST(ErrorTest, ErrorTestSetOverrun) {
  ArrowError error;
  char big_error[2048];
  const char* a_few_chars = "abcdefg";
  for (int i = 0; i < 2047; i++) {
    big_error[i] = a_few_chars[i % strlen(a_few_chars)];
  }
  big_error[2047] = '\0';

  EXPECT_EQ(ArrowErrorSet(&error, "%s", big_error), ERANGE);
  EXPECT_EQ(std::string(ArrowErrorMessage(&error)), std::string(big_error, 1023));

  wchar_t bad_string[] = {0xFFFF, 0};
  EXPECT_EQ(ArrowErrorSet(&error, "%ls", bad_string), EINVAL);
}
