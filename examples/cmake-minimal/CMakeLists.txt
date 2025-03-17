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

# When adding nanoarrow's CMake directory to a CMake project that contains a library
# intended for use by others, set NANOARROW_NAMESPACE to rename symbols in the
# nanoarrow library such that they do not collide with other libraries that may also
# link to their own copy of nanoarrow. You may wish to include the namespace only
# on release builds, since the namespace implementation obscures inline help
# available in many text editors.
set(NANOARROW_NAMESPACE "ExampleCmakeMinimal")

fetchcontent_declare(nanoarrow_example_cmake_minimal
                     # We use SOURCE_DIR here to point to the version of nanoarrow represented
                     # by this checkout of the repo; however, you can use any of the methods
                     # supported by FetchContent_Declare, e.g.:
                     # GIT_REPOSITORY https://github.com/apache/arrow-nanoarrow.git
                     # GIT_TAG some_commit_hash
                     # GIT_SHALLOW TRUE
                     SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)

fetchcontent_makeavailable(nanoarrow_example_cmake_minimal)

# Add the library and link it against nanoarrow
include_directories(src)
add_library(example_cmake_minimal_library src/library.c)

# Always use PRIVATE when linking to nanoarrow to hide nanoarrow's headers from a
# target that in turn uses your library.
target_link_libraries(example_cmake_minimal_library PRIVATE nanoarrow_static)

# Add the executable and link it against the library
add_executable(example_cmake_minimal_app src/app.c)
target_link_libraries(example_cmake_minimal_app example_cmake_minimal_library)
