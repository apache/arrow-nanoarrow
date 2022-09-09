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

#include <stdexcept>

#include <gtest/gtest.h>
#include <arrow/array.h>
#include <arrow/c/bridge.h>
#include <sqlite3.h>

#include "nanoarrow_sqlite3.h"

using namespace arrow;

class ConnectionHolder {
public:
  sqlite3* ptr;
  ConnectionHolder(): ptr(nullptr) {}

  int open_memory() {
    int result = sqlite3_open(":memory:", &ptr);
    if (result != SQLITE_OK) {
      throw std::runtime_error(sqlite3_errstr(result));
    }

    return result;
  }

  int exec(const std::string& sql) {
    char* error_message = nullptr;
    int result = sqlite3_exec(ptr, sql.c_str(), nullptr, nullptr, &error_message);
    if (error_message != nullptr) {
      throw std::runtime_error(error_message);
    }

    return result;
  }

  void add_crossfit_table() {
    exec("CREATE TABLE crossfit (exercise text,difficulty_level int)");
    exec("INSERT INTO crossfit VALUES ('Push Ups', 3), ('Pull Ups', 5) , ('Push Jerk', 7), ('Bar Muscle Up', 10)");
  }

  ~ConnectionHolder() {
    if (ptr != nullptr) {
      sqlite3_close(ptr);
    }
  }
};

class StmtHolder {
public:
  sqlite3_stmt* ptr;

  StmtHolder(): ptr(nullptr) {}

  int prepare(sqlite3* con, const std::string& sql) {
    const char* tail;
    int result = sqlite3_prepare_v2(con, sql.c_str(), sql.size(), &ptr, &tail);
    if (result != SQLITE_OK) {
      std::stringstream stream;
      stream << "<" << sqlite3_errstr(result) << "> " << sqlite3_errmsg(con);
      throw std::runtime_error(stream.str().c_str());
    }

    return result;
  }

  ~StmtHolder() {
    if (ptr != nullptr) {
      sqlite3_finalize(ptr);
    }
  }
};

void ASSERT_ARROW_OK(Status status) {
  ASSERT_TRUE(status.ok());
}

TEST(SQLite3Test, SQLite3ResultBasic) {
  struct ArrowSQLite3Result result;
  ASSERT_EQ(ArrowSQLite3ResultInit(&result), 0);
  ArrowSQLite3ResultReset(&result);
}

TEST(SQLite3Test, SQLite3ResultSetSchema) {
  struct ArrowSQLite3Result result;
  struct ArrowSchema schema;
  schema.release = nullptr;

  ASSERT_EQ(ArrowSQLite3ResultInit(&result), 0);

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

TEST(SQLite3Test, SQLite3ResultWithGuessedSchema) {
  ConnectionHolder con;
  con.open_memory();
  con.add_crossfit_table();

  StmtHolder stmt;
  stmt.prepare(con.ptr, "SELECT * from crossfit");

  struct ArrowSQLite3Result result;
  ASSERT_EQ(ArrowSQLite3ResultInit(&result), 0);

  do {
    EXPECT_EQ(ArrowSQLite3ResultStep(&result, stmt.ptr), 0);
  } while (result.step_return_code == SQLITE_ROW);


  ArrowSQLite3ResultReset(&result);
}
