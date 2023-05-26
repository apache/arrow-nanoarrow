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

project(nanoarrow_device)

option(NANOARROW_DEVICE_BUILD_TESTS "Build tests" OFF)
option(NANOARROW_DEVICE_BUILD_APPS "Build utility applications" OFF)
option(NANOARROW_DEVICE_BUNDLE "Create bundled nanoarrow_device.h and nanoarrow_device.c" OFF)

option(NANOARROW_DEVICE_CODE_COVERAGE "Enable coverage reporting" OFF)
add_library(device_coverage_config INTERFACE)

if (NANOARROW_DEVICE_BUILD_TESTS OR NOT NANOARROW_DEVICE_BUNDLE)
  # Add the nanoarrow dependency. nanoarrow is not linked into the
  # nanoarrow_device library (the caller must link this themselves);
  # however, we need nanoarrow.h to build nanoarrow_device.c.
  FetchContent_Declare(
    nanoarrow
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)

  # Don't install nanoarrow because of this configuration
  FetchContent_GetProperties(nanoarrow)
  if(NOT nanoarrow_POPULATED)
    FetchContent_Populate(nanoarrow)
    add_subdirectory(${nanoarrow_SOURCE_DIR} ${nanoarrow_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endif()

if (NANOARROW_DEVICE_BUNDLE)
  # The CMake build step is creating nanoarrow_device.c and nanoarrow_device.h;
  # the CMake install step is copying them to a specific location
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/amalgamation)
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow)

  # nanoarrow_device.h is already standalone
  set(NANOARROW_DEVICE_H_TEMP ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow/nanoarrow_device.h)
  file(READ src/nanoarrow/nanoarrow_device.h SRC_FILE_CONTENTS)
  file(WRITE ${NANOARROW_DEVICE_H_TEMP} "${SRC_FILE_CONTENTS}")

  # Add a library that the tests can link against (but don't install it)
  if(NANOARROW_DEVICE_BUILD_TESTS)
    add_library(nanoarrow_device
      ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow/nanoarrow_device.c)

    target_include_directories(nanoarrow_device PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
      $<BUILD_INTERFACE:${nanoarrow_SOURCE_DIR}/src/nanoarrow>
      $<BUILD_INTERFACE:${nanoarrow_BINARY_DIR}/generated>
      $<BUILD_INTERFACE:${NANOARROW_DEVICE_FLATCC_INCLUDE_DIR}>)
  endif()

  # Install the amalgamated header and sources
  install(FILES
    ${NANOARROW_DEVICE_H_TEMP}
    ${NANOARROW_DEVICE_C_TEMP}
    DESTINATION ".")
else()
  # This is a normal CMake build that builds + installs some includes and a static lib
  add_library(nanoarrow_device
    src/nanoarrow/nanoarrow_device.c)

  target_include_directories(nanoarrow_device PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${nanoarrow_SOURCE_DIR}/src/nanoarrow>
        $<BUILD_INTERFACE:${nanoarrow_BINARY_DIR}/generated>
        $<BUILD_INTERFACE:${NANOARROW_DEVICE_FLATCC_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>)

  install(TARGETS nanoarrow_device DESTINATION lib)
  install(
    FILES
      src/nanoarrow/nanoarrow_device.h
    DESTINATION include/nanoarrow)

endif()


if (NANOARROW_DEVICE_BUILD_TESTS)
  set(MEMORYCHECK_COMMAND_OPTIONS "--leak-check=full --suppressions=${CMAKE_CURRENT_LIST_DIR}/../../valgrind.supp --error-exitcode=1")
  include(CTest)
  include(FetchContent)

  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  # Warning about timestamps of downloaded files
  if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.23")
      cmake_policy(SET CMP0135 NEW)
  endif()

  # Use an old version of googletest if we have to to support gcc 4.8
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
    CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "5.0.0")
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/release-1.11.0.zip
        URL_HASH SHA256=353571c2440176ded91c2de6d6cd88ddd41401d14692ec1f99e35d013feda55a
    )
  else()
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/release-1.10.0.zip
        URL_HASH SHA256=94c634d499558a76fa649edb13721dce6e98fb1e7018dfaeba3cd7a083945e91
    )
  endif()

  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(nanoarrow_device_test src/nanoarrow/nanoarrow_device_test.cc)
  add_executable(nanoarrow_device_hpp_test src/nanoarrow/nanoarrow_device_hpp_test.cc)

  if(NANOARROW_DEVICE_CODE_COVERAGE)
    target_compile_options(device_coverage_config INTERFACE -O0 -g --coverage)
    target_link_options(device_coverage_config INTERFACE --coverage)
    target_link_libraries(nanoarrow_device PRIVATE device_coverage_config)
  endif()

  target_link_libraries(
    nanoarrow_device_test
    nanoarrow_device nanoarrow gtest_main device_coverage_config)
  target_link_libraries(
    nanoarrow_device_hpp_test
    nanoarrow_device nanoarrow gtest_main device_coverage_config)

  include(GoogleTest)
  gtest_discover_tests(nanoarrow_device_test)
  gtest_discover_tests(nanoarrow_device_hpp_test)
endif()
