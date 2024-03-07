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
cmake_minimum_required(VERSION 3.14)
include(FetchContent)

project(nanoarrow_benchmarks)

if(NOT DEFINED CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
endif()

option(NANOARROW_BENCHMARK_VERSION "nanoarrow version to benchmark" OFF)

# Use google/benchmark
set(BENCHMARK_ENABLE_TESTING OFF)
fetchcontent_declare(benchmark
                     URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
                     URL_HASH SHA256=abfc22e33e3594d0edf8eaddaf4d84a2ffc491ad74b6a7edc6e7a608f690e691
)
fetchcontent_makeavailable(benchmark)

if(NANOARROW_BENCHMARK_VERSION)
  # https://github.com/apache/arrow-nanoarrow/archive/refs/tags/apache-arrow-nanoarrow-0.4.0.zip
  fetchcontent_declare(nanoarrow
                       URL https://github.com/apache/arrow-nanoarrow/archive/refs/tags/apache-arrow-nanoarrow-${NANOARROW_BENCHMARK_VERSION}.zip
  )
  fetchcontent_makeavailable(nanoarrow)
endif()

if(NOT TARGET nanoarrow)

endif()

add_executable(schema_benchmark c/schema_benchmark.cc)
add_executable(array_benchmark c/array_benchmark.cc)

target_link_libraries(schema_benchmark PRIVATE nanoarrow benchmark::benchmark_main)
target_link_libraries(array_benchmark PRIVATE nanoarrow benchmark::benchmark_main)
