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
include(FetchContent)

if(NOT DEFINED CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 11)
endif()

project(nanoarrow_ipc)

option(NANOARROW_IPC_BUILD_TESTS "Build tests" OFF)
option(NANOARROW_IPC_BUILD_APPS "Build utility applications" OFF)
option(NANOARROW_IPC_BUNDLE "Create bundled nanoarrow_ipc.h and nanoarrow_ipc.c" OFF)
option(NANOARROW_IPC_FLATCC_ROOT_DIR
       "Root directory for flatcc include and lib directories" OFF)
option(NANOARROW_IPC_FLATCC_INCLUDE_DIR "Include directory for flatcc includes" OFF)
option(NANOARROW_IPC_FLATCC_LIB_DIR "Library directory that contains libflatccrt.a" OFF)
option(NANOARROW_ARROW_STATIC
       "Use a statically-linked Arrow C++ build when linking tests" OFF)

option(NANOARROW_IPC_CODE_COVERAGE "Enable coverage reporting" OFF)
add_library(ipc_coverage_config INTERFACE)

if(NANOARROW_IPC_BUILD_TESTS OR NOT NANOARROW_IPC_BUNDLE)
  # Lazily add the nanoarrow dependency
  if(NOT TARGET nanoarrow)
    fetchcontent_declare(nanoarrow SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)

    # Don't install nanoarrow because of this configuration
    fetchcontent_getproperties(nanoarrow)
    if(NOT nanoarrow_POPULATED)
      fetchcontent_populate(nanoarrow)
      add_subdirectory(${nanoarrow_SOURCE_DIR} ${nanoarrow_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
  endif()

  # Add the flatcc (runtime) dependency
  set(FLATCC_RTONLY
      ON
      CACHE INTERNAL "")
  set(FLATCC_REFLECTION
      OFF
      CACHE INTERNAL "")

  # A flatcc installation is unlikely, so default to building the vendored one
  if(NOT NANOARROW_IPC_FLATCC_INCLUDE_DIR AND NOT NANOARROW_IPC_FLATCC_ROOT_DIR)
    add_library(flatccrt STATIC
                thirdparty/flatcc/src/runtime/builder.c
                thirdparty/flatcc/src/runtime/emitter.c
                thirdparty/flatcc/src/runtime/verifier.c
                thirdparty/flatcc/src/runtime/refmap.c)

    set(NANOARROW_IPC_FLATCC_INCLUDE_DIR
        ${CMAKE_CURRENT_LIST_DIR}/thirdparty/flatcc/include)
    target_include_directories(flatccrt
                               PUBLIC $<BUILD_INTERFACE:${NANOARROW_IPC_FLATCC_INCLUDE_DIR}>
                                      $<INSTALL_INTERFACE:include>)

  elseif(NOT NANOARROW_IPC_FLATCC_ROOT_DIR)
    add_library(flatccrt STATIC IMPORTED)
    set_target_properties(flatccrt
                          PROPERTIES IMPORTED_LOCATION
                                     "${NANOARROW_IPC_FLATCC_LIB_DIR}/libflatccrt.a"
                                     INTERFACE_INCLUDE_DIRECTORIES
                                     "${NANOARROW_IPC_FLATCC_INCLUDE_DIR}")
  elseif(NOT NANOARROW_IPC_FLATCC_INCLUDE_DIR)
    set(NANOARROW_IPC_FLATCC_INCLUDE_DIR ${NANOARROW_IPC_FLATCC_ROOT_DIR}/include)
    add_library(flatccrt STATIC IMPORTED)
    set_target_properties(flatccrt
                          PROPERTIES IMPORTED_LOCATION
                                     "${NANOARROW_IPC_FLATCC_ROOT_DIR}/lib/libflatccrt.a"
                                     INTERFACE_INCLUDE_DIRECTORIES
                                     "${NANOARROW_IPC_FLATCC_INCLUDE_DIR}")
  endif()
endif()

if(NANOARROW_IPC_BUNDLE)
  # The CMake build step is creating nanoarrow_ipc.c and nanoarrow_ipc.h;
  # the CMake install step is copying them to a specific location
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/amalgamation)
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow)

  # nanoarrow_ipc.h is already standalone
  set(NANOARROW_IPC_H_TEMP ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow/nanoarrow_ipc.h)
  file(READ src/nanoarrow/nanoarrow_ipc.h SRC_FILE_CONTENTS)
  file(WRITE ${NANOARROW_IPC_H_TEMP} "${SRC_FILE_CONTENTS}")

  # combine flatcc-generated headers and nanoarrow_ipc sources
  set(NANOARROW_IPC_C_TEMP ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow/nanoarrow_ipc.c)
  file(READ src/nanoarrow/nanoarrow_ipc_flatcc_generated.h SRC_FILE_CONTENTS)
  file(WRITE ${NANOARROW_IPC_C_TEMP} "${SRC_FILE_CONTENTS}")
  file(READ src/nanoarrow/nanoarrow_ipc_decoder.c SRC_FILE_CONTENTS)
  file(APPEND ${NANOARROW_IPC_C_TEMP} "${SRC_FILE_CONTENTS}")
  file(READ src/nanoarrow/nanoarrow_ipc_reader.c SRC_FILE_CONTENTS)
  file(APPEND ${NANOARROW_IPC_C_TEMP} "${SRC_FILE_CONTENTS}")

  # remove the include for the generated files in the bundled version
  file(READ ${NANOARROW_IPC_C_TEMP} SRC_FILE_CONTENTS)
  string(REGEX REPLACE "#include \"nanoarrow_ipc_flatcc_generated.h\"" ""
                       SRC_FILE_CONTENTS "${SRC_FILE_CONTENTS}")
  file(WRITE ${NANOARROW_IPC_C_TEMP} "${SRC_FILE_CONTENTS}")

  # combine the flatcc sources
  set(FLATCC_C_TEMP ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow/flatcc.c)
  file(READ thirdparty/flatcc/src/runtime/builder.c SRC_FILE_CONTENTS)
  file(WRITE ${FLATCC_C_TEMP} "${SRC_FILE_CONTENTS}")
  file(READ thirdparty/flatcc/src/runtime/emitter.c SRC_FILE_CONTENTS)
  file(APPEND ${FLATCC_C_TEMP} "${SRC_FILE_CONTENTS}")
  file(READ thirdparty/flatcc/src/runtime/verifier.c SRC_FILE_CONTENTS)
  file(APPEND ${FLATCC_C_TEMP} "${SRC_FILE_CONTENTS}")
  file(READ thirdparty/flatcc/src/runtime/refmap.c SRC_FILE_CONTENTS)
  file(APPEND ${FLATCC_C_TEMP} "${SRC_FILE_CONTENTS}")

  # Add a library that the tests can link against (but don't install it)
  if(NANOARROW_IPC_BUILD_TESTS)
    # Bundle flatcc into the nanoarrow_ipc library instead of
    # link to the flatccrt static lib we declared above.
    add_library(nanoarrow_ipc ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow/nanoarrow_ipc.c
                              ${CMAKE_BINARY_DIR}/amalgamation/nanoarrow/flatcc.c)

    target_include_directories(nanoarrow_ipc
                               PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                                      $<BUILD_INTERFACE:${nanoarrow_SOURCE_DIR}/src/nanoarrow>
                                      $<BUILD_INTERFACE:${nanoarrow_BINARY_DIR}/generated>
                                      $<BUILD_INTERFACE:${NANOARROW_IPC_FLATCC_INCLUDE_DIR}>
    )

  endif()

  # Install the amalgamated header and sources
  install(FILES ${NANOARROW_IPC_H_TEMP} ${NANOARROW_IPC_C_TEMP} ${FLATCC_C_TEMP}
          DESTINATION ".")

  # Also install the flatcc headers
  install(DIRECTORY thirdparty/flatcc/include/flatcc DESTINATION ".")
else()
  # This is a normal CMake build that builds + installs some includes and a static lib
  add_library(nanoarrow_ipc src/nanoarrow/nanoarrow_ipc_decoder.c
                            src/nanoarrow/nanoarrow_ipc_reader.c)
  target_link_libraries(nanoarrow_ipc PRIVATE flatccrt)

  target_include_directories(nanoarrow_ipc
                             PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                                    $<BUILD_INTERFACE:${nanoarrow_SOURCE_DIR}/src/nanoarrow>
                                    $<BUILD_INTERFACE:${nanoarrow_BINARY_DIR}/generated>
                                    $<BUILD_INTERFACE:${NANOARROW_IPC_FLATCC_INCLUDE_DIR}>
                                    $<INSTALL_INTERFACE:include>)

  install(TARGETS nanoarrow_ipc DESTINATION lib)
  install(FILES src/nanoarrow/nanoarrow_ipc.h
                src/nanoarrow/nanoarrow_ipc_flatcc_generated.h
          DESTINATION include/nanoarrow)

endif()

# Don't add extra warning flags when bundling, since we treat flatcc
# as a part of the nanoarrow_ipc target and we have no control over the
# warnings it produces.
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT NANOARROW_IPC_BUNDLE)
  if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(nanoarrow_ipc
                           PRIVATE -Wall
                                   -Werror
                                   -Wextra
                                   -Wpedantic
                                   -Wno-type-limits
                                   -Wmaybe-uninitialized
                                   -Wunused-result
                                   -Wconversion
                                   -Wno-sign-conversion
                                   -Wno-misleading-indentation)
  elseif(CMAKE_C_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_C_COMPILER_ID STREQUAL
                                                      "Clang")
    target_compile_options(nanoarrow_ipc
                           PRIVATE -Wall
                                   -Werror
                                   -Wextra
                                   -Wpedantic
                                   -Wdocumentation
                                   -Wconversion
                                   -Wno-sign-conversion)
  endif()
endif()

if(NANOARROW_IPC_BUILD_TESTS)
  set(MEMORYCHECK_COMMAND_OPTIONS
      "--leak-check=full --suppressions=${CMAKE_CURRENT_LIST_DIR}/../../valgrind.supp --error-exitcode=1"
  )
  include(CTest)
  include(FetchContent)

  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  find_package(Arrow REQUIRED)
  message(STATUS "Arrow version: ${ARROW_VERSION}")
  message(STATUS "Arrow SO version: ${ARROW_FULL_SO_VERSION}")

  # Warning about timestamps of downloaded files
  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.23")
    cmake_policy(SET CMP0135 NEW)
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

  # JSON library for integration testing
  # Also used by some versions of Arrow, so check if this is already available
  if(NOT TARGET nlohmann_json::nlohmann_json)
    fetchcontent_declare(nlohmann_json
                         URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.zip
                         URL_HASH SHA256=95651d7d1fcf2e5c3163c3d37df6d6b3e9e5027299e6bd050d157322ceda9ac9
    )
    fetchcontent_makeavailable(nlohmann_json)
  endif()

  # zlib to decode gzipped integration testing JSON files
  # We don't use Arrow C++ for this because building Arrow C++ with zlib
  # is not trivial on Windows.
  find_package(ZLIB)
  if(NOT ZLIB_FOUND)
    # Wrapper around FetchContent that better isolates the zlib CMakeLists.txt
    message(STATUS "Using FetchContent to build a static zlib")
    add_subdirectory(thirdparty/zlib)
  endif()

  enable_testing()

  add_executable(nanoarrow_ipc_decoder_test src/nanoarrow/nanoarrow_ipc_decoder_test.cc)
  add_executable(nanoarrow_ipc_reader_test src/nanoarrow/nanoarrow_ipc_reader_test.cc)
  add_executable(nanoarrow_ipc_files_test src/nanoarrow/nanoarrow_ipc_files_test.cc)
  add_executable(nanoarrow_ipc_hpp_test src/nanoarrow/nanoarrow_ipc_hpp_test.cc)

  if(NANOARROW_IPC_CODE_COVERAGE)
    target_compile_options(ipc_coverage_config INTERFACE -O0 -g --coverage)
    target_link_options(ipc_coverage_config INTERFACE --coverage)
    target_link_libraries(nanoarrow_ipc PRIVATE ipc_coverage_config)
  endif()

  # Give caller the option to link a static version of Arrow C++
  if(NANOARROW_ARROW_STATIC)
    set(NANOARROW_IPC_ARROW_TARGET arrow_static)
  else()
    set(NANOARROW_IPC_ARROW_TARGET arrow_shared)
  endif()

  target_link_libraries(nanoarrow_ipc_decoder_test
                        nanoarrow_ipc
                        nanoarrow
                        ${NANOARROW_IPC_ARROW_TARGET}
                        gtest_main
                        ipc_coverage_config)
  target_link_libraries(nanoarrow_ipc_reader_test
                        nanoarrow_ipc
                        nanoarrow
                        gtest_main
                        ipc_coverage_config)
  target_link_libraries(nanoarrow_ipc_files_test
                        nanoarrow_ipc
                        nanoarrow
                        ${NANOARROW_IPC_ARROW_TARGET}
                        nlohmann_json
                        ZLIB::ZLIB
                        gtest_main
                        ipc_coverage_config)
  target_link_libraries(nanoarrow_ipc_hpp_test
                        nanoarrow_ipc
                        nanoarrow
                        ${NANOARROW_IPC_ARROW_TARGET}
                        gtest_main
                        ipc_coverage_config)

  include(GoogleTest)
  gtest_discover_tests(nanoarrow_ipc_decoder_test)
  gtest_discover_tests(nanoarrow_ipc_reader_test)
  gtest_discover_tests(nanoarrow_ipc_files_test)
  gtest_discover_tests(nanoarrow_ipc_hpp_test)
endif()

if(NANOARROW_IPC_BUILD_APPS)
  add_executable(dump_stream src/apps/dump_stream.c)
  target_link_libraries(dump_stream nanoarrow_ipc nanoarrow)
endif()
