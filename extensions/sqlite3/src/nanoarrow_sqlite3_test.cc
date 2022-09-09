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
#include <arrow/array.h>
#include <arrow/c/bridge.h>
#include <sqlite3.h>

#include "nanoarrow_sqlite3.h"

using namespace arrow;

void ASSERT_ARROW_OK(Status status) {
  ASSERT_TRUE(status.ok());
}

TEST(SQLite3Test, SQLite3ResultBasic) {
  struct ArrowSQLite3Result result;
  ASSERT_EQ(ArrowSQLite3ResultInit(&result, nullptr), 0);
  ArrowSQLite3ResultReset(&result);
}

TEST(SQLite3Test, SQLite3ResultSetSchema) {
  struct ArrowSQLite3Result result;
  struct ArrowSchema schema;
  schema.release = nullptr;

  ASSERT_EQ(ArrowSQLite3ResultInit(&result, nullptr), 0);

  EXPECT_EQ(ArrowSQLite3ResultSetSchema(&result, nullptr), EINVAL);
  EXPECT_STREQ(ArrowSQLite3ResultError(&result), "schema is null or released");
  EXPECT_EQ(ArrowSQLite3ResultSetSchema(&result, &schema), EINVAL);
  EXPECT_STREQ(ArrowSQLite3ResultError(&result), "schema is null or released");
  
  ASSERT_ARROW_OK(ExportType(*int32(), &schema));
  EXPECT_EQ(ArrowSQLite3ResultSetSchema(&result, &schema), EINVAL);
  EXPECT_STREQ(ArrowSQLite3ResultError(&result), "schema is not a struct");
  schema.release(&schema);

  ASSERT_ARROW_OK(ExportSchema(*arrow::schema({}), &schema));
  EXPECT_EQ(ArrowSQLite3ResultSetSchema(&result, &schema), 0);

  

  ArrowSQLite3ResultReset(&result);
}
