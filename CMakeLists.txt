# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

message(STATUS "Building using CMake version: ${CMAKE_VERSION}")
cmake_minimum_required(VERSION 3.14)

project(NanoArrow)

option(NANOARROW_BUILD_TESTS "Build tests" OFF)

option(NANOARROW_CODE_COVERAGE "Enable coverage reporting" OFF)
add_library(coverage_config INTERFACE)

include_directories(src)
add_library(
    nanoarrow
    src/nanoarrow/allocator.c
    src/nanoarrow/buffer.c
    src/nanoarrow/error.c
    src/nanoarrow/metadata.c
    src/nanoarrow/schema.c
    src/nanoarrow/schema_view.c)

install(TARGETS nanoarrow DESTINATION lib)
install(DIRECTORY src/ DESTINATION include FILES_MATCHING PATTERN "*.h")
install(DIRECTORY src/ DESTINATION include FILES_MATCHING PATTERN "*.c")

if (NANOARROW_BUILD_TESTS)
    # For testing we use GTest + Arrow C++ (both need C++11)
    include(FetchContent)
    include(CTest)

    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(Arrow REQUIRED)
    message(STATUS "Arrow version: ${ARROW_VERSION}")
    message(STATUS "Arrow SO version: ${ARROW_FULL_SO_VERSION}")
    find_package(ArrowTesting REQUIRED)

    find_package(GTest REQUIRED)
    enable_testing()

    add_executable(allocator_test src/nanoarrow/allocator_test.cc)
    add_executable(buffer_test src/nanoarrow/buffer_test.cc)
    add_executable(error_test src/nanoarrow/error_test.cc)
    add_executable(metadata_test src/nanoarrow/metadata_test.cc)
    add_executable(schema_test src/nanoarrow/schema_test.cc)
    add_executable(schema_view_test src/nanoarrow/schema_view_test.cc)

    if (NANOARROW_CODE_COVERAGE)
        target_compile_options(coverage_config INTERFACE -O0 -g --coverage)
        target_link_options(coverage_config INTERFACE --coverage)
        target_link_libraries(nanoarrow coverage_config)
    endif()

    target_link_libraries(allocator_test nanoarrow GTest::gtest_main arrow_shared arrow_testing_shared)
    target_link_libraries(buffer_test nanoarrow GTest::gtest_main)
    target_link_libraries(error_test nanoarrow GTest::gtest_main)
    target_link_libraries(metadata_test nanoarrow GTest::gtest_main arrow_shared arrow_testing_shared)
    target_link_libraries(schema_test nanoarrow GTest::gtest_main arrow_shared arrow_testing_shared)
    target_link_libraries(schema_view_test nanoarrow GTest::gtest_main arrow_shared arrow_testing_shared)

    include(GoogleTest)
    gtest_discover_tests(allocator_test)
    gtest_discover_tests(buffer_test)
    gtest_discover_tests(error_test)
    gtest_discover_tests(metadata_test)
    gtest_discover_tests(schema_test)
    gtest_discover_tests(schema_view_test)

endif()
