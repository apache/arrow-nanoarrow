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

project(NanoArrowExampleCMakeIpc)

# See examples/cmake-minimal for why this is a good idea
set(NANOARROW_NAMESPACE "ExampleCmakeIpc")

FetchContent_Declare(
  # The name 'nanoarrow' is important here: it allows the IPC extension
  # to link to the same version of nanoarrow as your library/application
  nanoarrow

  # See examples/cmake-minimal for how to specify a GIT repository or URL
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)
FetchContent_MakeAvailable(nanoarrow)

FetchContent_Declare(
  nanoarrow_ipc_example_cmake
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../extensions/nanoarrow_ipc)
FetchContent_MakeAvailable(nanoarrow_ipc_example_cmake)

# Add the library and link it against nanoarrow and nanoarrow_ipc
# You will need to link to both...nanoarrow_ipc assumes the caller
# will link in its own copy of nanoarrow with the same version
include_directories(src)
add_library(example_cmake_ipc_library src/library.c)

# Always use PRIVATE to hide nanoarrow's headers from a
# target that in turn uses your library.
# target_link_libraries(example_cmake_ipc_library PRIVATE nanoarrow_ipc PRIVATE nanoarrow)

# Add the executable and link it against the library
add_executable(example_cmake_ipc_app src/app.c)

target_link_libraries(example_cmake_ipc_app example_cmake_ipc_library nanoarrow_ipc nanoarrow)
