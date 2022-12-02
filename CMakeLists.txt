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
cmake_minimum_required(VERSION 3.16)

set(NANOARROW_VERSION "0.1.0-SNAPSHOT")
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" NANOARROW_BASE_VERSION "${NANOARROW_VERSION}")
project(nanoarrow VERSION "${NANOARROW_BASE_VERSION}")

set(NANOARROW_VERSION_MAJOR "${nanoarrow_VERSION_MAJOR}")
set(NANOARROW_VERSION_MINOR "${nanoarrow_VERSION_MINOR}")
set(NANOARROW_VERSION_PATCH "${nanoarrow_VERSION_PATCH}")

option(NANOARROW_BUILD_TESTS "Build tests" OFF)
option(NANOARROW_BUNDLE "Create bundled nanoarrow.h and nanoarrow.c" OFF)
option(NANOARROW_BUILD_ID "A string identifying this build" OFF)
option(NANOARROW_NAMESPACE "A prefix for exported symbols" OFF)

if (NOT NANOARROW_BUILD_ID)
    string(RANDOM LENGTH 10 NANOARROW_BUILD_ID)
endif()

if (NANOARROW_NAMESPACE)
    set(NANOARROW_NAMESPACE_DEFINE "#define NANOARROW_NAMESPACE ${NANOARROW_NAMESPACE}")
else()
    set(NANOARROW_NAMESPACE_DEFINE "// #define NANOARROW_NAMESPACE YourNamespaceHere")
endif()

option(NANOARROW_CODE_COVERAGE "Enable coverage reporting" OFF)
add_library(coverage_config INTERFACE)

configure_file(src/nanoarrow/build_id.h.in generated/build_id.h)

if(NANOARROW_BUNDLE)
    # Combine all headers into amalgamation/nanoarrow.h in the build directory
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/amalgamation)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow)
    set(NANOARROW_H_TEMP ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow/nanoarrow.h)
    file(READ ${CMAKE_BINARY_DIR}/generated/build_id.h SRC_FILE_CONTENTS)
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
    string(REGEX REPLACE "#include \"[a-z_.]+\"" "" SRC_FILE_CONTENTS "${SRC_FILE_CONTENTS}")
    file(WRITE ${NANOARROW_H_TEMP} "${SRC_FILE_CONTENTS}")

    # Copy nanoarrow.hpp too
    file(READ src/nanoarrow/nanoarrow.hpp SRC_FILE_CONTENTS)
    file(WRITE ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow/nanoarrow.hpp "${SRC_FILE_CONTENTS}")

    # Combine all source files into amalgamation/nanoarrow.c in the build directory
    set(NANOARROW_C_TEMP ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow/nanoarrow.c)
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
        add_library(nanoarrow ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow/nanoarrow.c)
    endif()

    # Install the amalgamated header and source
    install(FILES ${NANOARROW_H_TEMP} ${NANOARROW_C_TEMP} DESTINATION ".")
else()
    add_library(
        nanoarrow
        src/nanoarrow/array.c
        src/nanoarrow/schema.c
        src/nanoarrow/array_stream.c
        src/nanoarrow/utils.c)

    target_include_directories(nanoarrow PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>)
    target_include_directories(nanoarrow PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
            target_compile_options(
                nanoarrow
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
                nanoarrow
                PRIVATE
                -Wall
                -Werror
                -Wextra
                -Wdocumentation
                -Wno-unused-parameter
                -Wshorten-64-to-32)
        endif()
    endif()

    install(TARGETS nanoarrow DESTINATION lib)
    install(DIRECTORY src/ DESTINATION include FILES_MATCHING PATTERN "*.h")
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/generated/build_id.h DESTINATION include/nanoarrow)
endif()

if(NANOARROW_BUILD_TESTS)
    # For testing we use GTest + Arrow C++ (both need C++11)
    include(FetchContent)

    set(MEMORYCHECK_COMMAND_OPTIONS "--leak-check=full --suppressions=${CMAKE_CURRENT_LIST_DIR}/valgrind.supp --error-exitcode=1")
    include(CTest)

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
        # Because nanoarrow.hpp is header-only we have to link the test here to get coverage
        target_link_libraries(nanoarrow_hpp_test coverage_config)
    endif()

    target_link_libraries(utils_test nanoarrow GTest::gtest_main arrow_shared)
    target_link_libraries(buffer_test nanoarrow GTest::gtest_main)
    target_link_libraries(array_test nanoarrow GTest::gtest_main arrow_shared)
    target_link_libraries(schema_test nanoarrow GTest::gtest_main arrow_shared)
    target_link_libraries(array_stream_test nanoarrow GTest::gtest_main)
    target_link_libraries(nanoarrow_hpp_test nanoarrow GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(utils_test)
    gtest_discover_tests(buffer_test)
    gtest_discover_tests(array_test)
    gtest_discover_tests(schema_test)
    gtest_discover_tests(array_stream_test)
    gtest_discover_tests(nanoarrow_hpp_test)
endif()
