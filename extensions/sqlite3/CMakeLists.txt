# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

message(STATUS "Building using CMake version: ${CMAKE_VERSION}")
cmake_minimum_required(VERSION 3.11)
include(FetchContent)

project(NanoArrowSQLite3)

if(CMAKE_BUILD_TYPE EQUAL "Release")
  set(NANOARROW_NAMESPACE "NanoarrowSQLite3")
endif()

FetchContent_Declare(
  nanoarrow_sqlite3
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)
FetchContent_MakeAvailable(nanoarrow_sqlite3)

find_package(SQLite3 REQUIRED)

include_directories(src)
add_library(nanoarrow_sqlite3 src/nanoarrow_sqlite3.c)
target_link_libraries(nanoarrow_sqlite3 PRIVATE nanoarrow)
target_link_libraries(nanoarrow_sqlite3 PUBLIC SQLite::SQLite3)

if (NANOARROW_SQLITE3_BUILD_TESTS)
  include(CTest)
  include(FetchContent)

  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/release-1.11.0.zip
  )
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  add_executable(nanoarrow_sqlite3_test src/nanoarrow_sqlite3_test.cc)
  target_link_libraries(nanoarrow_sqlite3_test nanoarrow_sqlite3 gtest_main)

  include(GoogleTest)
  gtest_discover_tests(nanoarrow_sqlite3_test)
endif()
