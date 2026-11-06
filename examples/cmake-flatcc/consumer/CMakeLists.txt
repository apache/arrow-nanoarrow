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

cmake_minimum_required(VERSION 3.18)
project(flatcc_consumer C)
set(NANOARROW_IPC
    ON
    CACHE BOOL "")
add_subdirectory("${NANOARROW_SOURCE}" nanoarrow)
# Verify discovery as well as linking: a Release fallback can otherwise hide a
# missed unsuffixed Debug library.
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND EXPECT_DEBUG)
  if(NOT NANOARROW_FLATCC_LIBRARY_DEBUG MATCHES "/debug/lib/")
    message(FATAL_ERROR "Did not select the external Debug runtime")
  endif()
endif()
if(EXPECT_STATIC AND NOT WIN32)
  if(NOT NANOARROW_FLATCC_LIBRARY MATCHES "[.]a$")
    message(FATAL_ERROR "Did not prefer the static runtime")
  endif()
endif()
add_executable(consumer main.c)
target_link_libraries(consumer PRIVATE nanoarrow::nanoarrow_ipc)
enable_testing()
add_test(NAME ipc-runtime COMMAND consumer)
