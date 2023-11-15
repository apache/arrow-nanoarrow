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

if(NOT DEFINED CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 11)
endif()

project(nanoarrow_testing)

option(NANOARROW_TESTING_BUILD_TESTS "Build tests" OFF)

option(NANOARROW_TESTING_CODE_COVERAGE "Enable coverage reporting" OFF)
add_library(testing_coverage_config INTERFACE)

if(NANOARROW_TESTING_BUILD_TESTS)
  # Add the nanoarrow dependency for tests
  fetchcontent_declare(nanoarrow SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)

  # Don't install nanoarrow because of this configuration
  fetchcontent_getproperties(nanoarrow)
  if(NOT nanoarrow_POPULATED)
    fetchcontent_populate(nanoarrow)
    add_subdirectory(${nanoarrow_SOURCE_DIR} ${nanoarrow_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()

endif()

# Header-only target
add_library(nanoarrow_testing INTERFACE)
target_include_directories(nanoarrow_testing INTERFACE .)
target_link_libraries(nanoarrow_testing INTERFACE nanoarrow)

install(FILES nanoarrow_testing.h
        DESTINATION include/nanoarrow)

if(NANOARROW_TESTING_BUILD_TESTS)
  set(MEMORYCHECK_COMMAND_OPTIONS
      "--leak-check=full --suppressions=${CMAKE_CURRENT_LIST_DIR}/../../valgrind.supp --error-exitcode=1"
  )
  include(CTest)
  include(FetchContent)

  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  # Warning about timestamps of downloaded files
  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.23")
    cmake_policy(SET CMP0135 NEW)
  endif()

  # Use an old version of googletest if we have to to support gcc 4.8
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_VERSION
                                                 VERSION_GREATER_EQUAL "5.0.0")
    fetchcontent_declare(googletest
                         URL https://github.com/google/googletest/archive/release-1.11.0.zip
                         URL_HASH SHA256=353571c2440176ded91c2de6d6cd88ddd41401d14692ec1f99e35d013feda55a
    )
  else()
    fetchcontent_declare(googletest
                         URL https://github.com/google/googletest/archive/release-1.10.0.zip
                         URL_HASH SHA256=94c634d499558a76fa649edb13721dce6e98fb1e7018dfaeba3cd7a083945e91
    )
  endif()

  fetchcontent_makeavailable(googletest)

  enable_testing()

  add_executable(nanoarrow_testing_test nanoarrow_testing_test.cc)

  if(NANOARROW_TESTING_CODE_COVERAGE)
    target_compile_options(testing_coverage_config INTERFACE -O0 -g --coverage)
    target_link_options(testing_coverage_config INTERFACE --coverage)
    target_link_libraries(nanoarrow_testing PRIVATE testing_coverage_config)
  endif()

  target_link_libraries(nanoarrow_testing_test
                        nanoarrow_testing
                        nanoarrow
                        gtest_main
                        testing_coverage_config)

  include(GoogleTest)
  gtest_discover_tests(nanoarrow_testing_test)
endif()
