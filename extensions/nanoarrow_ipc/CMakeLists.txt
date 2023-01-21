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

project(nanoarrow_ipc)

option(NANOARROW_IPC_BUILD_TESTS "Build tests" OFF)
option(NANOARROW_IPC_BUNDLE "Create bundled nanoarrow_ipc.h and nanoarrow_ipc.c" OFF)
option(NANOARROW_IPC_FLATCC_ROOT_DIR "Root directory for flatcc include and lib directories" OFF)
option(NANOARROW_IPC_FLATCC_INCLUDE_DIR "Include directory for flatcc includes" OFF)
option(NANOARROW_IPC_FLATCC_LIB_DIR "Library directory that contains libflatccrt.a" OFF)

option(NANOARROW_IPC_CODE_COVERAGE "Enable coverage reporting" OFF)
add_library(ipc_coverage_config INTERFACE)

if (NANOARROW_IPC_BUILD_TESTS OR NOT NANOARROW_IPC_BUNDLE)
  # Add the nanoarrow dependency. nanoarrow is not linked into the
  # nanoarrow_ipc library (the caller must link this themselves);
  # however, we need nanoarrow.h to build nanoarrow_ipc.c.
  FetchContent_Declare(
    nanoarrow
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)

  # Don't install nanoarrow because of this configuration
  FetchContent_GetProperties(nanoarrow)
  if(NOT nanoarrow_POPULATED)
    FetchContent_Populate(nanoarrow)
    add_subdirectory(${nanoarrow_SOURCE_DIR} ${nanoarrow_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()

  # Add the flatcc (runtime) dependency
  set(FLATCC_RTONLY ON CACHE INTERNAL "")
  set(FLATCC_REFLECTION OFF CACHE INTERNAL "")

  # A flatcc installation is unlikely, so default to building the vendored one
  if (NOT NANOARROW_IPC_FLATCC_INCLUDE_DIR AND NOT NANOARROW_IPC_FLATCC_ROOT_DIR)
    add_library(
      flatccrt STATIC
      thirdparty/flatcc/src/runtime/builder.c
      thirdparty/flatcc/src/runtime/emitter.c
      thirdparty/flatcc/src/runtime/verifier.c
      thirdparty/flatcc/src/runtime/refmap.c)

    set(NANOARROW_IPC_FLATCC_INCLUDE_DIR
      ${CMAKE_CURRENT_LIST_DIR}/thirdparty/flatcc/include)
    target_include_directories(flatccrt PUBLIC
      $<BUILD_INTERFACE:${NANOARROW_IPC_FLATCC_INCLUDE_DIR}>
      $<INSTALL_INTERFACE:include>)

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
endif()

if (NANOARROW_IPC_BUNDLE)
  # The CMake build step is creating nanoarrow_ipc.c and nanoarrow_ipc.h;
  # the CMake install step is copying them to a specific location
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/amalgamation)
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow)

  # nanoarrow_ipc.h is already standalone
  set(NANOARROW_IPC_H_TEMP ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow/nanoarrow_ipc.h)
  file(READ src/nanoarrow/nanoarrow_ipc.h SRC_FILE_CONTENTS)
  file(WRITE ${NANOARROW_IPC_H_TEMP} "${SRC_FILE_CONTENTS}")

  # combine flatcc-generated headers and nanoarrow_ipc.c
  set(NANOARROW_IPC_C_TEMP ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow/nanoarrow_ipc.c)
  file(READ src/nanoarrow/nanoarrow_ipc_flatcc_generated.h SRC_FILE_CONTENTS)
  file(WRITE ${NANOARROW_IPC_C_TEMP} "${SRC_FILE_CONTENTS}")
  file(READ src/nanoarrow/nanoarrow_ipc.c SRC_FILE_CONTENTS)
  file(APPEND ${NANOARROW_IPC_C_TEMP} "${SRC_FILE_CONTENTS}")

  # remove the include for the generated files in the bundled version
  file(READ ${NANOARROW_IPC_C_TEMP} SRC_FILE_CONTENTS)
  string(
    REGEX REPLACE "#include \"nanoarrow_ipc_flatcc_generated.h\"" ""
    SRC_FILE_CONTENTS "${SRC_FILE_CONTENTS}")
  file(WRITE ${NANOARROW_IPC_C_TEMP} "${SRC_FILE_CONTENTS}")

  # Add a library that the tests can link against (but don't install it)
  if(NANOARROW_IPC_BUILD_TESTS)
    add_library(nanoarrow_ipc ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow/nanoarrow_ipc.c)
    target_include_directories(nanoarrow_ipc PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
      $<BUILD_INTERFACE:${nanoarrow_SOURCE_DIR}/src/nanoarrow>
      $<BUILD_INTERFACE:${nanoarrow_BINARY_DIR}/generated>
      $<BUILD_INTERFACE:${NANOARROW_IPC_FLATCC_INCLUDE_DIR}>)
    target_link_libraries(nanoarrow_ipc PRIVATE flatccrt)
  endif()

  # Install the amalgamated header and source
  install(FILES ${NANOARROW_IPC_H_TEMP} ${NANOARROW_IPC_C_TEMP} DESTINATION ".")

else()
  # This is a normal CMake build that builds + installs some includes and a static lib
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

  # Warning about timestamps of downloaded files
  cmake_policy(SET CMP0135 NEW)
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
