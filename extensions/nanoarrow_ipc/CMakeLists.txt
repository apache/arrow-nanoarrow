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

project(NanoArrowIpc)

option(NANOARROW_IPC_BUILD_TESTS "Build tests" OFF)
option(NANOARROW_IPC_CODE_COVERAGE "Enable coverage reporting" OFF)
add_library(ipc_coverage_config INTERFACE)

# Add the nanoarrow dependency
FetchContent_Declare(
  nanoarrow_for_nanoarrow_ipc
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)
FetchContent_MakeAvailable(nanoarrow_for_nanoarrow_ipc)

# Add the flatcc (runtime) dependency
# TODO: Make it more flexible to use an existing flatcc
set(FLATCC_RTONLY ON CACHE INTERNAL "")
FetchContent_Declare(
  flatcc
  URL https://github.com/dvidelabs/flatcc/archive/refs/tags/v0.6.1.zip
)
FetchContent_MakeAvailable(flatcc)

include_directories(src)
add_library(nanoarrow_ipc src/nanoarrow_ipc/nanoarrow_ipc.c)
target_link_libraries(nanoarrow_ipc PRIVATE nanoarrow)

if(NANOARROW_IPC_CODE_COVERAGE)
  target_compile_options(ipc_coverage_config INTERFACE -O0 -g --coverage)
  target_link_options(ipc_coverage_config INTERFACE --coverage)
  target_link_libraries(nanoarrow_ipc ipc_coverage_config)
endif()


if (NANOARROW_IPC_BUILD_TESTS)

  set(MEMORYCHECK_COMMAND_OPTIONS "--leak-check=full --suppressions=${CMAKE_CURRENT_LIST_DIR}/../../valgrind.supp --error-exitcode=1")
  include(CTest)
  include(FetchContent)

  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  find_package(Arrow REQUIRED)
  message(STATUS "Arrow version: ${ARROW_VERSION}")
  message(STATUS "Arrow SO version: ${ARROW_FULL_SO_VERSION}")

  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/release-1.11.0.zip
  )
  FetchContent_MakeAvailable(googletest)

  enable_testing()

  add_executable(nanoarrow_ipc_test src/nanoarrow_ipc/nanoarrow_ipc_test.cc)

  target_link_libraries(nanoarrow_ipc_test nanoarrow_ipc arrow_shared gtest_main)

  include(GoogleTest)
  gtest_discover_tests(nanoarrow_ipc_test)
endif()
