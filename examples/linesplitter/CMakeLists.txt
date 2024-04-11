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

cmake_minimum_required(VERSION 3.14)
project(linesplitter)

set(CMAKE_CXX_STANDARD 11)

# Avoids a warning about timestamps on downloaded files (prefer new policy
# if available))
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.23")
  cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)

fetchcontent_declare(nanoarrow
                     SOURCE_DIR
                     ${CMAKE_CURRENT_LIST_DIR}/../..
                     # We use SOURCE_DIR to simplify testing this example on CI; however,
                     # you should use a released version of nanoarrow like so:
                     # URL https://github.com/apache/arrow-nanoarrow/releases/download/apache-arrow-nanoarrow-0.1.0/apache-arrow-nanoarrow-0.1.0.tar.gz
                     # URL_HASH SHA512=dc62480b986ee76aaad8e38c6fbc602f8cef2cc35a5f5ede7da2a93b4db2b63839bdca3eefe8a44ae1cb6895a2fd3f090e3f6ea1020cf93cfe86437304dfee17)
)
fetchcontent_makeavailable(nanoarrow)

add_library(linesplitter linesplitter.cc)
target_link_libraries(linesplitter PRIVATE nanoarrow)

fetchcontent_declare(googletest
                     URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)
fetchcontent_makeavailable(googletest)

enable_testing()
add_executable(linesplitter_test linesplitter_test.cc)
target_link_libraries(linesplitter_test linesplitter nanoarrow GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(linesplitter_test)
