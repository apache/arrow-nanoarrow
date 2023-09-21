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

if(NOT DEFINED CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

set(NANOARROW_VERSION "0.3.0-SNAPSHOT")
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" NANOARROW_BASE_VERSION
             "${NANOARROW_VERSION}")
project(nanoarrow VERSION "${NANOARROW_BASE_VERSION}")

set(NANOARROW_VERSION_MAJOR "${nanoarrow_VERSION_MAJOR}")
set(NANOARROW_VERSION_MINOR "${nanoarrow_VERSION_MINOR}")
set(NANOARROW_VERSION_PATCH "${nanoarrow_VERSION_PATCH}")

option(NANOARROW_BUILD_TESTS "Build tests" OFF)
option(NANOARROW_BUNDLE "Create bundled nanoarrow.h and nanoarrow.c" OFF)
option(NANOARROW_BUNDLE_AS_CPP "Bundle nanoarrow source file as nanoarrow.cc" OFF)
option(NANOARROW_NAMESPACE "A prefix for exported symbols" OFF)
option(NANOARROW_ARROW_STATIC
       "Use a statically-linked Arrow C++ build when linking tests" OFF)

if(NANOARROW_NAMESPACE)
  set(NANOARROW_NAMESPACE_DEFINE "#define NANOARROW_NAMESPACE ${NANOARROW_NAMESPACE}")
else()
  set(NANOARROW_NAMESPACE_DEFINE "// #define NANOARROW_NAMESPACE YourNamespaceHere")
endif()

option(NANOARROW_CODE_COVERAGE "Enable coverage reporting" OFF)
add_library(coverage_config INTERFACE)

configure_file(src/nanoarrow/nanoarrow_config.h.in generated/nanoarrow_config.h)

if(NANOARROW_BUNDLE)
  # Combine all headers into amalgamation/nanoarrow.h in the build directory
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/amalgamation)
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow)
  set(NANOARROW_H_TEMP ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow/nanoarrow.h)
  file(READ ${CMAKE_BINARY_DIR}/generated/nanoarrow_config.h SRC_FILE_CONTENTS)
  file(WRITE ${NANOARROW_H_TEMP} "${SRC_FILE_CONTENTS}")
  file(READ src/nanoarrow/nanoarrow_types.h SRC_FILE_CONTENTS)
  file(APPEND ${NANOARROW_H_TEMP} "${SRC_FILE_CONTENTS}")
  file(READ src/nanoarrow/nanoarrow.h SRC_FILE_CONTENTS)
  file(APPEND ${NANOARROW_H_TEMP} "${SRC_FILE_CONTENTS}")
  file(READ src/nanoarrow/buffer_inline.h SRC_FILE_CONTENTS)
  file(APPEND ${NANOARROW_H_TEMP} "${SRC_FILE_CONTENTS}")
  file(READ src/nanoarrow/array_inline.h SRC_FILE_CONTENTS)
  file(APPEND ${NANOARROW_H_TEMP} "${SRC_FILE_CONTENTS}")

  # Remove includes that aren't needed when the headers are concatenated
  file(READ ${NANOARROW_H_TEMP} SRC_FILE_CONTENTS)
  string(REGEX REPLACE "#include \"[a-z_.]+\"" "" SRC_FILE_CONTENTS
                       "${SRC_FILE_CONTENTS}")
  file(WRITE ${NANOARROW_H_TEMP} "${SRC_FILE_CONTENTS}")

  # Copy nanoarrow.hpp too
  set(NANOARROW_HPP_TEMP ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow/nanoarrow.hpp)
  file(READ src/nanoarrow/nanoarrow.hpp SRC_FILE_CONTENTS)
  file(WRITE ${NANOARROW_HPP_TEMP} "${SRC_FILE_CONTENTS}")

  # Combine all source files into amalgamation/nanoarrow.c in the build directory
  if(NANOARROW_BUNDLE_AS_CPP)
    set(NANOARROW_C_TEMP ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow/nanoarrow.cc)
  else()
    set(NANOARROW_C_TEMP ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow/nanoarrow.c)
  endif()
  file(READ src/nanoarrow/utils.c SRC_FILE_CONTENTS)
  file(WRITE ${NANOARROW_C_TEMP} "${SRC_FILE_CONTENTS}")
  file(READ src/nanoarrow/schema.c SRC_FILE_CONTENTS)
  file(APPEND ${NANOARROW_C_TEMP} "${SRC_FILE_CONTENTS}")
  file(READ src/nanoarrow/array.c SRC_FILE_CONTENTS)
  file(APPEND ${NANOARROW_C_TEMP} "${SRC_FILE_CONTENTS}")
  file(READ src/nanoarrow/array_stream.c SRC_FILE_CONTENTS)
  file(APPEND ${NANOARROW_C_TEMP} "${SRC_FILE_CONTENTS}")

  # Add a library that the tests can link against (but don't install it)
  if(NANOARROW_BUILD_TESTS)
    include_directories(${CMAKE_BINARY_DIR}/amalgamation)
    add_library(nanoarrow ${NANOARROW_C_TEMP})
    target_compile_definitions(nanoarrow PUBLIC "$<$<CONFIG:Debug>:NANOARROW_DEBUG>")
  endif()

  # Install the amalgamated header and source
  install(FILES ${NANOARROW_H_TEMP} ${NANOARROW_C_TEMP} ${NANOARROW_HPP_TEMP}
          DESTINATION ".")
else()
  add_library(nanoarrow src/nanoarrow/array.c src/nanoarrow/schema.c
                        src/nanoarrow/array_stream.c src/nanoarrow/utils.c)

  target_include_directories(nanoarrow
                             PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                                    $<INSTALL_INTERFACE:include>)
  target_include_directories(nanoarrow
                             PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  )
  target_compile_definitions(nanoarrow PUBLIC "$<$<CONFIG:Debug>:NANOARROW_DEBUG>")

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
      target_compile_options(nanoarrow
                             PRIVATE -Wall
                                     -Werror
                                     -Wextra
                                     -Wno-type-limits
                                     -Wno-unused-parameter
                                     -Wmaybe-uninitialized
                                     -Wpedantic
                                     -Wunused-result)
    elseif(CMAKE_C_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_C_COMPILER_ID STREQUAL
                                                        "Clang")
      target_compile_options(nanoarrow
                             PRIVATE -Wall
                                     -Werror
                                     -Wextra
                                     -Wdocumentation
                                     -Wno-unused-parameter
                                     -Wshorten-64-to-32)
    endif()
  endif()

  install(TARGETS nanoarrow DESTINATION lib)
  install(DIRECTORY src/
          DESTINATION include
          FILES_MATCHING
          PATTERN "*.h")
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/generated/nanoarrow_config.h
          DESTINATION include/nanoarrow)
endif()

if(NANOARROW_BUILD_TESTS)
  # For testing we use GTest + Arrow C++
  include(FetchContent)

  set(MEMORYCHECK_COMMAND_OPTIONS
      "--leak-check=full --suppressions=${CMAKE_CURRENT_LIST_DIR}/valgrind.supp --error-exitcode=1"
  )
  include(CTest)

  find_package(Arrow REQUIRED)
  message(STATUS "Arrow version: ${ARROW_VERSION}")
  message(STATUS "Arrow SO version: ${ARROW_FULL_SO_VERSION}")

  # Arrow >= 10.0.0 requires C++17; GTest requires C++11.
  # Leave the option open to use an older version of Arrow
  # to make it easier to test on old Linux (e.g., Centos7)
  if(${ARROW_VERSION} VERSION_GREATER_EQUAL "10.0.0")
    set(CMAKE_CXX_STANDARD 17)
  else()
    set(CMAKE_CXX_STANDARD 11)
  endif()
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  # Avoids a warning about timestamps on downloaded files (prefer new policy
  # if available))
  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.23")
    cmake_policy(SET CMP0135 NEW)
  endif()

  # Give caller the option to link a static version of Arrow C++
  if(NANOARROW_ARROW_STATIC)
    set(NANOARROW_ARROW_TARGET arrow_static)
  else()
    set(NANOARROW_ARROW_TARGET arrow_shared)
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

  add_executable(utils_test src/nanoarrow/utils_test.cc)
  add_executable(buffer_test src/nanoarrow/buffer_test.cc)
  add_executable(array_test src/nanoarrow/array_test.cc)
  add_executable(schema_test src/nanoarrow/schema_test.cc)
  add_executable(array_stream_test src/nanoarrow/array_stream_test.cc)
  add_executable(nanoarrow_hpp_test src/nanoarrow/nanoarrow_hpp_test.cc)

  if(NANOARROW_CODE_COVERAGE)
    target_compile_options(coverage_config INTERFACE -O0 -g --coverage)
    target_link_options(coverage_config INTERFACE --coverage)
    target_link_libraries(nanoarrow coverage_config)
  endif()

  target_link_libraries(utils_test
                        nanoarrow
                        gtest_main
                        ${NANOARROW_ARROW_TARGET}
                        coverage_config)
  target_link_libraries(buffer_test nanoarrow gtest_main coverage_config)
  target_link_libraries(array_test
                        nanoarrow
                        gtest_main
                        ${NANOARROW_ARROW_TARGET}
                        coverage_config)
  target_link_libraries(schema_test
                        nanoarrow
                        gtest_main
                        ${NANOARROW_ARROW_TARGET}
                        coverage_config)
  target_link_libraries(array_stream_test nanoarrow gtest_main coverage_config)
  target_link_libraries(nanoarrow_hpp_test nanoarrow gtest_main coverage_config)

  include(GoogleTest)
  # Some users have reported a timeout with the default value of 5
  # Building with -DBUILD_SHARED_LIBS=ON may also help reduce the size of test
  # executables.
  gtest_discover_tests(utils_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(buffer_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(array_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(schema_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(array_stream_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(nanoarrow_hpp_test DISCOVERY_TIMEOUT 10)
endif()
