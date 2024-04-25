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
cmake_minimum_required(VERSION 3.18)
include(FetchContent)

project(nanoarrow_benchmarks)

if(NOT DEFINED CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
  set(CMAKE_C_STANDARD_REQUIRED ON)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

set(NANOARROW_BENCHMARK_VERSION
    ""
    CACHE STRING "nanoarrow version to benchmark")
set(NANOARROW_BENCHMARK_SOURCE_DIR
    ""
    CACHE STRING "path to a nanoarrow source checkout to benchmark" OFF)

# Avoids a warning about timestamps on downloaded files (prefer new policy
# if available))
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.23")
  cmake_policy(SET CMP0135 NEW)
endif()

# Use google/benchmark
set(BENCHMARK_ENABLE_TESTING OFF)
fetchcontent_declare(benchmark
                     URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
                     URL_HASH SHA256=abfc22e33e3594d0edf8eaddaf4d84a2ffc491ad74b6a7edc6e7a608f690e691
)
fetchcontent_makeavailable(benchmark)

if(IS_DIRECTORY "${NANOARROW_BENCHMARK_SOURCE_URL}")
  fetchcontent_declare(nanoarrow SOURCE_DIR "${NANOARROW_BENCHMARK_SOURCE_URL}")
  fetchcontent_declare(nanoarrow_ipc
                       SOURCE_DIR
                       "${NANOARROW_BENCHMARK_SOURCE_URL}/extensions/nanoarrow_ipc")

  fetchcontent_makeavailable(nanoarrow)
  fetchcontent_makeavailable(nanoarrow_ipc)
elseif(NOT "${NANOARROW_BENCHMARK_SOURCE_URL}" STREQUAL "")
  fetchcontent_declare(nanoarrow URL "${NANOARROW_BENCHMARK_SOURCE_URL}")
  fetchcontent_declare(nanoarrow_ipc URL "${NANOARROW_BENCHMARK_SOURCE_URL}"
                                         SOURCE_SUBDIR extensions/nanoarrow_ipc)
  fetchcontent_makeavailable(nanoarrow)
  fetchcontent_makeavailable(nanoarrow_ipc)
endif()

# Check that either the parent scope or this CMakeLists.txt defines a nanoarrow target
if(NOT TARGET nanoarrow OR NOT TARGET nanoarrow_ipc)
  message(FATAL_ERROR "nanoarrow or nanoarrow_ipc target not found (missing -DNANOARROW_BENCHMARK_SOURCE_URL option?)"
  )
endif()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/fixtures")
foreach(ITEM float64_basic;float64_long;float64_wide)
  file(COPY_FILE "${CMAKE_CURRENT_LIST_DIR}/fixtures/${ITEM}.arrows"
       "${CMAKE_BINARY_DIR}/fixtures/${ITEM}.arrows" ONLY_IF_DIFFERENT)
endforeach()

# Add executables and register them as tests.
# This lets all benchmarks run via ctest -VV when this is the top-level project
# and takes care of setting the relevant test properties such that the benchmarks
# can find the fixtures.
include(CTest)
enable_testing()

foreach(ITEM schema;array;ipc)
  add_executable(${ITEM}_benchmark "c/${ITEM}_benchmark.cc")
  target_link_libraries(${ITEM}_benchmark PRIVATE nanoarrow nanoarrow_ipc
                                                  benchmark::benchmark_main)
  add_test(NAME ${ITEM}_benchmark COMMAND ${ITEM}_benchmark
                                          --benchmark_out=${ITEM}_benchmark.json)
  set_tests_properties(${ITEM}_benchmark PROPERTIES WORKING_DIRECTORY
                                                    "${CMAKE_BINARY_DIR}")
endforeach(ITEM)
