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

option(FIND_NANOARROW "Find an existing nanoarrow" ON)

# When adding nanoarrow's CMake directory to a CMake project that contains a library
# intended for use by others, set NANOARROW_NAMESPACE to rename symbols in the
# nanoarrow library such that they do not collide with other libraries that may also
# link to their own copy of nanoarrow. You may wish to include the namespace only
# on release builds, since the namespace implementation obscures inline help
# available in many text editors.
set(NANOARROW_NAMESPACE "ExampleCmakeMinimal")

if(FIND_NANOARROW)
  # Users should pin to a specific version of nanoarrow if they choose
  # to find nanoarrow rather than pinning to a specific version via
  # FetchContent.
  find_package(nanoarrow REQUIRED)
else()
  include(FetchContent)
  fetchcontent_declare(nanoarrow_example_cmake_minimal
                       GIT_REPOSITORY https://github.com/apache/arrow-nanoarrow.git
                       GIT_TAG main
                       GIT_SHALLOW TRUE)
  fetchcontent_makeavailable(nanoarrow_example_cmake_minimal)
endif()

add_executable(minimal_cpp_app src/app.cpp)
target_link_libraries(minimal_cpp_app nanoarrow::nanoarrow)
