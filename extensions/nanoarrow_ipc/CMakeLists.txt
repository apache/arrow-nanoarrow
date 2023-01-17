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
option(NANOARROW_IPC_BUNDLE "Create bundled nanoarrow_ipc.h and nanoarrow_ipc.c" OFF)
option(NANOARROW_IPC_FLATCC_ROOT_DIR "Root directory for flatcc include and lib directories" OFF)
option(NANOARROW_IPC_FLATCC_INCLUDE_DIR "Include directory for flatcc includes" OFF)
option(NANOARROW_IPC_FLATCC_LIB_DIR "Library directory that contains libflatccrt.a" OFF)

option(NANOARROW_IPC_CODE_COVERAGE "Enable coverage reporting" OFF)
add_library(ipc_coverage_config INTERFACE)

# Add the nanoarrow dependency. nanoarrow is not linked into the
# nanoarrow_ipc library (the caller must link this themselves);
# however, we need nanoarrow.h to build nanoarrow_ipc.c.
FetchContent_Declare(
  nanoarrow
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)

# Don't install nanoarrow because of this configuration
FetchContent_GetProperties(cmark)
if(NOT nanoarrow_POPULATED)
  FetchContent_Populate(nanoarrow)
  add_subdirectory(${nanoarrow_SOURCE_DIR} ${nanoarrow_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Add the flatcc (runtime) dependency
set(FLATCC_RTONLY ON CACHE INTERNAL "")
set(FLATCC_REFLECTION OFF CACHE INTERNAL "")

# Timestamps for fetch aren't really applicable here...this silences a warning
cmake_policy(SET CMP0135 NEW)

# A flatcc installation is unlikely, so default to building the vendored one
if (NOT NANOARROW_IPC_FLATCC_INCLUDE_DIR AND NOT NANOARROW_IPC_FLATCC_ROOT_DIR)
  FetchContent_Declare(
    flatcc_for_nanoarrow
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/thirdparty/flatcc)
  FetchContent_MakeAvailable(flatcc_for_nanoarrow)
  set(NANOARROW_IPC_FLATCC_INCLUDE_DIR ${flatcc_for_nanoarrow_SOURCE_DIR}/include)
elseif(NOT NANOARROW_IPC_FLATCC_ROOT_DIR)
  add_library(flatccrt STATIC IMPORTED)
  set_target_properties(flatccrt PROPERTIES
    IMPORTED_LOCATION "${NANOARROW_IPC_FLATCC_LIB_DIR}/libflatccrt.a"
    INTERFACE_INCLUDE_DIRECTORIES "${NANOARROW_IPC_FLATCC_INCLUDE_DIR}"
  )
elseif(NOT NANOARROW_IPC_FLATCC_INCLUDE_DIR)
  set(NANOARROW_IPC_FLATCC_INCLUDE_DIR ${NANOARROW_IPC_FLATCC_ROOT_DIR}/include)
  add_library(flatccrt STATIC IMPORTED)
  set_target_properties(flatccrt PROPERTIES
    IMPORTED_LOCATION "${NANOARROW_IPC_FLATCC_ROOT_DIR}/lib/libflatccrt.a"
    INTERFACE_INCLUDE_DIRECTORIES "${NANOARROW_IPC_FLATCC_INCLUDE_DIR}"
  )
endif()

if (NANOARROW_IPC_BUNDLE)


else()

  add_library(nanoarrow_ipc src/nanoarrow/nanoarrow_ipc.c)
  target_link_libraries(nanoarrow_ipc PRIVATE flatccrt)

  target_include_directories(nanoarrow_ipc PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${nanoarrow_SOURCE_DIR}/src/nanoarrow>
        $<BUILD_INTERFACE:${nanoarrow_BINARY_DIR}/generated>
        $<BUILD_INTERFACE:${NANOARROW_IPC_FLATCC_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>)
  
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
      target_compile_options(
          nanoarrow_ipc
          PRIVATE
          -Wall
          -Werror
          -Wextra
          -Wno-type-limits
          -Wno-unused-parameter
          -Wpedantic
          -Wunused-result)
    elseif(CMAKE_C_COMPILER_ID STREQUAL "AppleClang" OR
           CMAKE_C_COMPILER_ID STREQUAL "Clang")
      target_compile_options(
          nanoarrow_ipc
          PRIVATE
          -Wall
          -Werror
          -Wextra
          -Wdocumentation
          -Wno-unused-parameter
          -Wshorten-64-to-32)
    endif()
  endif()

  install(TARGETS nanoarrow_ipc DESTINATION lib)
  install(
    FILES
      src/nanoarrow/nanoarrow_ipc.h
      src/nanoarrow/nanoarrow_ipc_flatcc_generated.h
    DESTINATION include/nanoarrow)

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

  add_executable(nanoarrow_ipc_test src/nanoarrow/nanoarrow_ipc_test.cc)

  if(NANOARROW_IPC_CODE_COVERAGE)
    target_compile_options(ipc_coverage_config INTERFACE -O0 -g --coverage)
    target_link_options(ipc_coverage_config INTERFACE --coverage)
    target_link_libraries(nanoarrow_ipc ipc_coverage_config)
  endif()

  target_link_libraries(nanoarrow_ipc_test nanoarrow_ipc nanoarrow arrow_shared gtest_main)

  include(GoogleTest)
  gtest_discover_tests(nanoarrow_ipc_test)
endif()
