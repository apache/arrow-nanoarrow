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

project(NanoArrowExampleCMakeMinimal)

option(EXAMPLE_MINIMAL_CMAKE_NANOARROW_REPO
  "Repo from which nanoarrow should be fetched")
if (EXAMPLE_MINIMAL_CMAKE_NANOARROW_REPO)
  set(EXAMPLE_MINIMAL_CMAKE_NANOARROW_REPO "https://github.com/apache/arrow-nanoarrow.git")
endif()

option(EXAMPLE_MINIMAL_CMAKE_NANOARROW_TAG
  "A reference or commit sha1 used to pin the version of nanoarrow")
  if (EXAMPLE_MINIMAL_CMAKE_NANOARROW_TAG)
  set(EXAMPLE_MINIMAL_CMAKE_NANOARROW_TAG "main")
endif()

FetchContent_Declare(
  nanoarrow_example_cmake_minimal
  GIT_REPOSITORY ${EXAMPLE_MINIMAL_CMAKE_NANOARROW_REPO}
  GIT_TAG examples-2
  GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(nanoarrow_example_cmake_minimal)

include_directories(src)

add_library(
    example_cmake_minimal_library
    src/library.c)

target_link_libraries(example_cmake_minimal_library PRIVATE nanoarrow)

add_executable(example_cmake_minimal_app src/app.c)
target_link_libraries(example_cmake_minimal_app example_cmake_minimal_library)
