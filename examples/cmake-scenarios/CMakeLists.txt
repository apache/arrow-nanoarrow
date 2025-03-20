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
cmake_minimum_required(VERSION 3.12)

project(NanoArrowExampleCMakeMinimal LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 11)

option(FIND_NANOARROW "Find an existing nanoarrow" ON)

# When adding nanoarrow's CMake directory to a CMake project that contains a library
# intended for use by others, set NANOARROW_NAMESPACE to rename symbols in the
# nanoarrow library such that they do not collide with other libraries that may also
# link to their own copy of nanoarrow. You may wish to include the namespace only
# on release builds, since the namespace implementation obscures inline help
# available in many text editors.
set(NANOARROW_NAMESPACE "ExampleCmakeScenarios")

include(FetchContent)

if(FIND_NANOARROW)
  # Users should pin to a specific version of nanoarrow if they choose
  # to find nanoarrow rather than pinning to a specific version via
  # FetchContent.
  find_package(nanoarrow REQUIRED)
else()
  # We need all these components for our test
  set(NANOARROW_DEVICE ON)
  set(NANOARROW_IPC ON)
  set(NANOARROW_TESTING ON)

  # Using SOURCE_DIR here such that CI accurately reflects the state of the
  # source; however, using GIT_TAG or URL of a pinned version is a more realistic
  # user pattern.
  fetchcontent_declare(nanoarrow SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
  fetchcontent_makeavailable(nanoarrow)
endif()

add_executable(minimal_cpp_app src/app.cpp)

if(TEST_BUILD_TYPE STREQUAL "static")
  target_link_libraries(minimal_cpp_app
                        nanoarrow::nanoarrow_static
                        nanoarrow::nanoarrow_device_static
                        nanoarrow::nanoarrow_ipc_static
                        nanoarrow::nanoarrow_testing_static)
elseif(TEST_BUILD_TYPE STREQUAL "shared")
  target_link_libraries(minimal_cpp_app
                        nanoarrow::nanoarrow_shared
                        nanoarrow::nanoarrow_device_shared
                        nanoarrow::nanoarrow_ipc_shared
                        nanoarrow::nanoarrow_testing_shared)
else()
  target_link_libraries(minimal_cpp_app
                        nanoarrow::nanoarrow
                        nanoarrow::nanoarrow_device
                        nanoarrow::nanoarrow_ipc
                        nanoarrow::nanoarrow_testing)
endif()
