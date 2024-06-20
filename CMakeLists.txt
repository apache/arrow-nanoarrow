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

if(NOT DEFINED CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
  set(CMAKE_C_STANDARD_REQUIRED ON)
endif()

set(NANOARROW_VERSION "0.6.0-SNAPSHOT")
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" NANOARROW_BASE_VERSION
             "${NANOARROW_VERSION}")
project(nanoarrow VERSION "${NANOARROW_BASE_VERSION}")

set(NANOARROW_VERSION_MAJOR "${nanoarrow_VERSION_MAJOR}")
set(NANOARROW_VERSION_MINOR "${nanoarrow_VERSION_MINOR}")
set(NANOARROW_VERSION_PATCH "${nanoarrow_VERSION_PATCH}")

# Feature options
option(NANOARROW_IPC "Build IPC extension" OFF)
option(NANOARROW_FLATCC_ROOT_DIR "Root directory for flatcc include and lib directories"
       OFF)
option(NANOARROW_FLATCC_INCLUDE_DIR "Include directory for flatcc includes" OFF)
option(NANOARROW_FLATCC_LIB_DIR "Library directory that contains libflatccrt.a" OFF)

option(NANOARROW_DEVICE "Build device extension" OFF)

# Development options
option(NANOARROW_BUILD_APPS "Build utility applications" OFF)
option(NANOARROW_BUILD_TESTS "Build tests" OFF)
option(NANOARROW_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(NANOARROW_BUILD_INTEGRATION_TESTS
       "Build cross-implementation Arrow integration tests" OFF)
option(NANOARROW_BUNDLE "Create bundled nanoarrow.h and nanoarrow.c" OFF)
option(NANOARROW_BUNDLE_AS_CPP "Bundle nanoarrow source file as nanoarrow.cc" OFF)
option(NANOARROW_NAMESPACE "A prefix for exported symbols" OFF)
option(NANOARROW_ARROW_STATIC
       "Use a statically-linked Arrow C++ build when linking tests" OFF)

option(NANOARROW_DEVICE_WITH_METAL "Build Apple metal libraries" OFF)
option(NANOARROW_DEVICE_WITH_CUDA "Build CUDA libraries" OFF)

if(NANOARROW_IPC)
  add_library(ipc_coverage_config INTERFACE)
endif()

if(NANOARROW_DEVICE)
  add_library(device_coverage_config INTERFACE)
endif()

if(NANOARROW_NAMESPACE)
  set(NANOARROW_NAMESPACE_DEFINE "#define NANOARROW_NAMESPACE ${NANOARROW_NAMESPACE}")
else()
  set(NANOARROW_NAMESPACE_DEFINE "// #define NANOARROW_NAMESPACE YourNamespaceHere")
endif()

option(NANOARROW_CODE_COVERAGE "Enable coverage reporting" OFF)

# Avoids a warning about timestamps on downloaded files (prefer new policy
# if available))
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.23")
  cmake_policy(SET CMP0135 NEW)
endif()

if(NANOARROW_BUNDLE)
  message(STATUS "Generating bundled nanoarrow sources using ci/scripts/bundle.py")

  find_package(Python3
               COMPONENTS Interpreter
               REQUIRED)

  # Set up arguments to the bundler
  set(NANOARROW_BUNDLE_ARGS "--with-ipc" "--with-device" "--with-flatcc")

  if(NANOARROW_BUNDLE_AS_CPP)
    list(APPEND NANOARROW_BUNDLE_ARGS "--cpp")
    set(NANOARROW_BUILD_SOURCES "${CMAKE_BINARY_DIR}/bundled/src/nanoarrow.cc")
  else()
    set(NANOARROW_BUILD_SOURCES "${CMAKE_BINARY_DIR}/bundled/src/nanoarrow.c")
  endif()

  if(NANOARROW_NAMESPACE)
    list(APPEND NANOARROW_BUNDLE_ARGS "--symbol-namespace=${NANOARROW_NAMESPACE}")
  endif()

  # Run the bundler
  execute_process(COMMAND ${Python3_EXECUTABLE}
                          "${CMAKE_CURRENT_SOURCE_DIR}/ci/scripts/bundle.py"
                          "--output-dir" "${CMAKE_BINARY_DIR}/bundled"
                          ${NANOARROW_BUNDLE_ARGS} COMMAND_ECHO STDERR)

  set(NANOARROW_BUILD_INCLUDE_DIR "${CMAKE_BINARY_DIR}/bundled/include")
  set(NANOARROW_IPC_BUILD_SOURCES "${CMAKE_BINARY_DIR}/bundled/src/nanoarrow_ipc.c")
  set(NANOARROW_DEVICE_BUILD_SOURCES "${CMAKE_BINARY_DIR}/bundled/src/nanoarrow_device.c")
  set(NANOARROW_FLATCC_BUILD_SOURCES "${CMAKE_BINARY_DIR}/bundled/src/flatcc.c")
else()
  # Generate the version information
  configure_file(src/nanoarrow/nanoarrow_config.h.in src/nanoarrow/nanoarrow_config.h)
  set(NANOARROW_BUILD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
endif()

# Start the list of targets and headers to install
set(NANOARROW_INSTALL_TARGETS nanoarrow)
set(NANOARROW_INSTALL_HEADERS ${NANOARROW_BUILD_INCLUDE_DIR}/nanoarrow/nanoarrow.h
                              ${NANOARROW_BUILD_INCLUDE_DIR}/nanoarrow/nanoarrow.hpp)

# If the bundler didn't already assign the source files for the library, do so here
if(NOT NANOARROW_BUNDLE)
  set(NANOARROW_BUILD_SOURCES src/nanoarrow/array.c src/nanoarrow/schema.c
                              src/nanoarrow/array_stream.c src/nanoarrow/utils.c)
  list(APPEND
       NANOARROW_INSTALL_HEADERS
       src/nanoarrow/array_inline.h
       src/nanoarrow/buffer_inline.h
       src/nanoarrow/nanoarrow_types.h
       "${CMAKE_CURRENT_BINARY_DIR}/src/nanoarrow/nanoarrow_config.h")
endif()

# Add the nanoarrow library target
add_library(nanoarrow ${NANOARROW_BUILD_SOURCES})
add_library(nanoarrow::nanoarrow ALIAS nanoarrow)

target_include_directories(nanoarrow
                           PUBLIC $<BUILD_INTERFACE:${NANOARROW_BUILD_INCLUDE_DIR}>
                                  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/src>
                                  $<INSTALL_INTERFACE:include>)
target_compile_definitions(nanoarrow PUBLIC "$<$<CONFIG:Debug>:NANOARROW_DEBUG>")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(nanoarrow
                           PRIVATE -Wall
                                   -Werror
                                   -Wextra
                                   -Wpedantic
                                   -Wno-type-limits
                                   -Wmaybe-uninitialized
                                   -Wunused-result
                                   -Wconversion
                                   -Wno-sign-conversion)
  elseif(CMAKE_C_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_C_COMPILER_ID STREQUAL
                                                      "Clang")
    target_compile_options(nanoarrow
                           PRIVATE -Wall
                                   -Werror
                                   -Wextra
                                   -Wpedantic
                                   -Wdocumentation
                                   -Wconversion
                                   -Wno-sign-conversion)
  endif()
endif()

if(NANOARROW_IPC)
  # Add the flatcc (runtime) dependency
  set(FLATCC_RTONLY
      ON
      CACHE INTERNAL "")
  set(FLATCC_REFLECTION
      OFF
      CACHE INTERNAL "")

  # A flatcc installation is unlikely, so default to building the vendored one
  if(NOT NANOARROW_FLATCC_INCLUDE_DIR AND NOT NANOARROW_FLATCC_ROOT_DIR)

    # If the bundler didn't already assign the source files for the library, do so here
    if(NOT NANOARROW_BUNDLE)
      set(NANOARROW_FLATCC_BUILD_SOURCES
          thirdparty/flatcc/src/runtime/builder.c thirdparty/flatcc/src/runtime/emitter.c
          thirdparty/flatcc/src/runtime/verifier.c thirdparty/flatcc/src/runtime/refmap.c)
      set(NANOARROW_FLATCC_INCLUDE_DIR
          "${CMAKE_CURRENT_LIST_DIR}/thirdparty/flatcc/include")
    else()
      set(NANOARROW_FLATCC_INCLUDE_DIR "${NANOARROW_BUILD_INCLUDE_DIR}")
    endif()

    add_library(flatccrt STATIC ${NANOARROW_FLATCC_BUILD_SOURCES})
    target_include_directories(flatccrt
                               PUBLIC $<BUILD_INTERFACE:${NANOARROW_FLATCC_INCLUDE_DIR}>
                                      $<INSTALL_INTERFACE:include>)

  elseif(NOT NANOARROW_FLATCC_ROOT_DIR)
    add_library(flatccrt STATIC IMPORTED)
    set_target_properties(flatccrt
                          PROPERTIES IMPORTED_LOCATION
                                     "${NANOARROW_FLATCC_LIB_DIR}/libflatccrt.a"
                                     INTERFACE_INCLUDE_DIRECTORIES
                                     "${NANOARROW_FLATCC_INCLUDE_DIR}")

  elseif(NOT NANOARROW_FLATCC_INCLUDE_DIR)
    set(NANOARROW_FLATCC_INCLUDE_DIR "${NANOARROW_FLATCC_ROOT_DIR}/include")
    add_library(flatccrt STATIC IMPORTED)
    set_target_properties(flatccrt
                          PROPERTIES IMPORTED_LOCATION
                                     "${NANOARROW_FLATCC_ROOT_DIR}/lib/libflatccrt.a"
                                     INTERFACE_INCLUDE_DIRECTORIES
                                     "${NANOARROW_FLATCC_INCLUDE_DIR}")
  endif()

  if(NOT NANOARROW_BUNDLE)
    set(NANOARROW_IPC_BUILD_SOURCES src/nanoarrow/nanoarrow_ipc_decoder.c
                                    src/nanoarrow/nanoarrow_ipc_reader.c)
  endif()

  add_library(nanoarrow_ipc ${NANOARROW_IPC_BUILD_SOURCES})
  target_link_libraries(nanoarrow_ipc PRIVATE flatccrt)
  target_include_directories(nanoarrow_ipc
                             PUBLIC $<BUILD_INTERFACE:${NANOARROW_BUILD_INCLUDE_DIR}>
                                    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/src>
                                    $<BUILD_INTERFACE:${NANOARROW_IPC_FLATCC_INCLUDE_DIR}
                                    $<INSTALL_INTERFACE:include>)

  install(TARGETS nanoarrow_ipc DESTINATION lib)
  install(FILES src/nanoarrow/nanoarrow_ipc.h
                src/nanoarrow/nanoarrow_ipc_flatcc_generated.h
          DESTINATION include/nanoarrow)
endif()

if(NANOARROW_DEVICE)
  if(NANOARROW_DEVICE_WITH_METAL)
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/metal-cpp")
      message(STATUS "Fetching metal-cpp")
      file(DOWNLOAD
           "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS12_iOS15.zip"
           "${CMAKE_BINARY_DIR}/metal-cpp.zip")
      file(ARCHIVE_EXTRACT
           INPUT
           ${CMAKE_BINARY_DIR}/metal-cpp.zip
           DESTINATION
           ${CMAKE_BINARY_DIR})
    endif()

    if(NOT DEFINED CMAKE_CXX_STANDARD)
      set(CMAKE_CXX_STANDARD 17)
    endif()
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_library(METAL_LIBRARY Metal REQUIRED)
    message(STATUS "Metal framework found at '${METAL_LIBRARY}'")

    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    message(STATUS "Foundation framework found at '${FOUNDATION_LIBRARY}'")

    find_library(QUARTZ_CORE_LIBRARY QuartzCore REQUIRED)
    message(STATUS "CoreFoundation framework found at '${QUARTZ_CORE_LIBRARY}'")

    set(NANOARROW_DEVICE_SOURCES_METAL src/nanoarrow/nanoarrow_device_metal.cc)
    set(NANOARROW_DEVICE_INCLUDE_METAL ${CMAKE_BINARY_DIR}/metal-cpp)
    set(NANOARROW_DEVICE_LIBS_METAL ${METAL_LIBRARY} ${FOUNDATION_LIBRARY}
                                    ${QUARTZ_CORE_LIBRARY})
    set(NANOARROW_DEVICE_DEFS_METAL "NANOARROW_DEVICE_WITH_METAL")
  endif()

  if(NANOARROW_DEVICE_WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(NANOARROW_DEVICE_SOURCES_CUDA src/nanoarrow/nanoarrow_device_cuda.c)
    set(NANOARROW_DEVICE_LIBS_CUDA CUDA::cuda_driver)
    set(NANOARROW_DEVICE_DEFS_CUDA "NANOARROW_DEVICE_WITH_CUDA")
  endif()

  # If the bundler didn't already assign the source files for the library, do so here
  if(NOT NANOARROW_BUNDLE)
    set(NANOARROW_DEVICE_BUILD_SOURCES src/nanoarrow/nanoarrow_device.c)
  endif()

  add_library(nanoarrow_device
              ${NANOARROW_DEVICE_BUILD_SOURCES} ${NANOARROW_DEVICE_SOURCES_METAL}
              ${NANOARROW_DEVICE_SOURCES_CUDA})

  target_include_directories(nanoarrow_device
                             PUBLIC $<BUILD_INTERFACE:${NANOARROW_BUILD_INCLUDE_DIR}>
                                    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/src>
                                    $<BUILD_INTERFACE:${NANOARROW_DEVICE_INCLUDE_METAL}>
                                    $<INSTALL_INTERFACE:include>)

  target_compile_definitions(nanoarrow_device PRIVATE ${NANOARROW_DEVICE_DEFS_METAL}
                                                      ${NANOARROW_DEVICE_DEFS_CUDA})
  target_link_libraries(nanoarrow_device PUBLIC ${NANOARROW_DEVICE_LIBS_METAL}
                                                ${NANOARROW_DEVICE_LIBS_CUDA})
  target_compile_definitions(nanoarrow_device PUBLIC "$<$<CONFIG:Debug>:NANOARROW_DEBUG>")

  install(TARGETS nanoarrow_device DESTINATION lib)
  install(FILES src/nanoarrow/nanoarrow_device.h DESTINATION include/nanoarrow)
endif()

# Always build integration test if building tests
if(NANOARROW_BUILD_TESTS OR NANOARROW_BUILD_INTEGRATION_TESTS)
  add_subdirectory("thirdparty/nlohmann_json")

  set_target_properties(nanoarrow PROPERTIES POSITION_INDEPENDENT_CODE ON)
  add_library(nanoarrow_c_data_integration SHARED
              src/nanoarrow/integration/c_data_integration.cc)
  target_include_directories(nanoarrow_c_data_integration
                             PUBLIC $<BUILD_INTERFACE:${NANOARROW_BUILD_INCLUDE_DIR}>
                                    $<INSTALL_INTERFACE:include>)
  target_link_libraries(nanoarrow_c_data_integration PRIVATE nanoarrow nlohmann_json)
endif()

if(NANOARROW_BUILD_TESTS)
  set(MEMORYCHECK_COMMAND_OPTIONS
      "--leak-check=full --suppressions=${CMAKE_CURRENT_LIST_DIR}/valgrind.supp --error-exitcode=1"
  )
  include(CTest)

  find_package(Arrow REQUIRED)
  message(STATUS "Arrow version: ${ARROW_VERSION}")
  message(STATUS "Arrow SO version: ${ARROW_FULL_SO_VERSION}")

  # Give caller the option to link a static version of Arrow C++
  if(NANOARROW_ARROW_STATIC)
    set(NANOARROW_ARROW_TARGET arrow_static)
  else()
    set(NANOARROW_ARROW_TARGET arrow_shared)
  endif()

  # Arrow >= 10.0.0 requires C++17; GTest requires C++11.
  # Leave the option open to use an older version of Arrow
  # to make it easier to test on old Linux (e.g., Centos7)
  if(${ARROW_VERSION} VERSION_GREATER_EQUAL "10.0.0")
    set(CMAKE_CXX_STANDARD 17)
  else()
    set(CMAKE_CXX_STANDARD 11)
  endif()
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  add_subdirectory("thirdparty/googletest")

  # Be sure to keep these tests in sync with src/nanoarrow/meson.build
  add_executable(utils_test src/nanoarrow/utils_test.cc)
  add_executable(buffer_test src/nanoarrow/buffer_test.cc)
  add_executable(array_test src/nanoarrow/array_test.cc)
  add_executable(schema_test src/nanoarrow/schema_test.cc)
  add_executable(array_stream_test src/nanoarrow/array_stream_test.cc)
  add_executable(nanoarrow_hpp_test src/nanoarrow/nanoarrow_hpp_test.cc)
  add_executable(nanoarrow_testing_test src/nanoarrow/nanoarrow_testing_test.cc)
  add_executable(c_data_integration_test
                 src/nanoarrow/integration/c_data_integration_test.cc)

  if(NANOARROW_CODE_COVERAGE)
    target_compile_options(nanoarrow PUBLIC -O0 -g --coverage)
    target_link_options(nanoarrow PUBLIC --coverage)
  endif()

  target_link_libraries(utils_test
                        nanoarrow
                        gtest_main
                        gmock_main
                        ${NANOARROW_ARROW_TARGET}
                        nlohmann_json::nlohmann_json)
  target_link_libraries(buffer_test nanoarrow gtest_main)
  target_link_libraries(array_test
                        nanoarrow
                        gtest_main
                        gmock_main
                        ${NANOARROW_ARROW_TARGET})
  target_link_libraries(schema_test nanoarrow gtest_main ${NANOARROW_ARROW_TARGET})
  target_link_libraries(array_stream_test nanoarrow gtest_main gmock_main)
  target_link_libraries(nanoarrow_hpp_test
                        nanoarrow
                        gtest_main
                        gmock_main
                        nlohmann_json::nlohmann_json)
  target_link_libraries(nanoarrow_testing_test nanoarrow gtest_main
                        nlohmann_json::nlohmann_json)
  target_link_libraries(c_data_integration_test nanoarrow nanoarrow_c_data_integration
                        gtest_main)

  include(GoogleTest)
  # Some users have reported a timeout with the default value of 5
  # Building with -DBUILD_SHARED_LIBS=ON may also help reduce the size of test
  # executables.
  gtest_discover_tests(utils_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(buffer_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(array_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(schema_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(array_stream_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(nanoarrow_hpp_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(nanoarrow_testing_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(c_data_integration_test DISCOVERY_TIMEOUT 10)

  if(NANOARROW_IPC)

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

    if(NANOARROW_CODE_COVERAGE)
      target_compile_options(ipc_coverage_config INTERFACE -O0 -g --coverage)
      target_link_options(ipc_coverage_config INTERFACE --coverage)
      target_link_libraries(nanoarrow_ipc PRIVATE ipc_coverage_config)
    endif()
    target_link_libraries(nanoarrow_ipc_decoder_test
                          nanoarrow_ipc
                          nanoarrow
                          flatccrt
                          ${NANOARROW_ARROW_TARGET}
                          gtest_main
                          ipc_coverage_config)
    target_link_libraries(nanoarrow_ipc_reader_test
                          nanoarrow_ipc
                          nanoarrow
                          flatccrt
                          gtest_main
                          ipc_coverage_config)
    target_link_libraries(nanoarrow_ipc_files_test
                          nanoarrow_ipc
                          nanoarrow
                          flatccrt
                          ${NANOARROW_ARROW_TARGET}
                          nlohmann_json
                          ZLIB::ZLIB
                          gtest_main
                          ipc_coverage_config)
    target_link_libraries(nanoarrow_ipc_hpp_test
                          nanoarrow_ipc
                          nanoarrow
                          ${NANOARROW_ARROW_TARGET}
                          gtest_main
                          ipc_coverage_config)

    include(GoogleTest)
    gtest_discover_tests(nanoarrow_ipc_decoder_test)
    gtest_discover_tests(nanoarrow_ipc_reader_test)
    gtest_discover_tests(nanoarrow_ipc_files_test)
    gtest_discover_tests(nanoarrow_ipc_hpp_test)
  endif()

  if(NANOARROW_DEVICE)
    enable_testing()
    add_executable(nanoarrow_device_test src/nanoarrow/nanoarrow_device_test.cc)
    add_executable(nanoarrow_device_hpp_test src/nanoarrow/nanoarrow_device_hpp_test.cc)

    if(NANOARROW_DEVICE_CODE_COVERAGE)
      target_compile_options(device_coverage_config INTERFACE -O0 -g --coverage)
      target_link_options(device_coverage_config INTERFACE --coverage)
      target_link_libraries(nanoarrow_device PRIVATE device_coverage_config)
    endif()

    target_link_libraries(nanoarrow_device_test
                          nanoarrow_device
                          nanoarrow
                          gtest_main
                          device_coverage_config)
    target_link_libraries(nanoarrow_device_hpp_test
                          nanoarrow_device
                          nanoarrow
                          gtest_main
                          device_coverage_config)

    include(GoogleTest)
    gtest_discover_tests(nanoarrow_device_test)
    gtest_discover_tests(nanoarrow_device_hpp_test)

    if(NANOARROW_DEVICE_WITH_METAL)
      add_executable(nanoarrow_device_metal_test
                     src/nanoarrow/nanoarrow_device_metal_test.cc)
      target_link_libraries(nanoarrow_device_metal_test
                            nanoarrow_device
                            nanoarrow
                            gtest_main
                            device_coverage_config)
      gtest_discover_tests(nanoarrow_device_metal_test)
    endif()

    if(NANOARROW_DEVICE_WITH_CUDA)
      add_executable(nanoarrow_device_cuda_test
                     src/nanoarrow/nanoarrow_device_cuda_test.cc)
      target_link_libraries(nanoarrow_device_cuda_test
                            nanoarrow_device
                            nanoarrow
                            gtest_main
                            device_coverage_config)
      gtest_discover_tests(nanoarrow_device_cuda_test)
    endif()
  endif()
endif()

if(NANOARROW_BUILD_APPS)
  if(NANOARROW_IPC)
    add_executable(dump_stream src/apps/dump_stream.c)
    target_link_libraries(dump_stream nanoarrow_ipc nanoarrow)
  endif()
endif()

if(NANOARROW_BUILD_BENCHMARKS)
  add_subdirectory(dev/benchmarks)
endif()

# Ensure targets/headers can be installed
install(TARGETS ${NANOARROW_INSTALL_TARGETS}
        DESTINATION lib
        EXPORT nanoarrow-exports)
install(FILES ${NANOARROW_INSTALL_HEADERS} DESTINATION include/nanoarrow)

# Generate package files for the build and install trees.
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

foreach(tree_type BUILD INSTALL)
  if(tree_type STREQUAL "BUILD")
    set(install_location ".")
  else()
    set(install_location "${CMAKE_INSTALL_LIBDIR}/cmake/nanoarrow")
  endif()

  set(build_location "${PROJECT_BINARY_DIR}/${install_location}")
  write_basic_package_version_file(
    "${build_location}/nanoarrow-config-version.cmake"
    VERSION ${nanoarrow_VERSION}
    # After 1.0.0, we can use `SameMajorVersion` here.
    COMPATIBILITY ExactVersion)
  configure_package_config_file("${CMAKE_CURRENT_LIST_DIR}/cmake/config.cmake.in"
                                "${build_location}/nanoarrow-config.cmake"
                                INSTALL_DESTINATION "${install_location}")

  if(tree_type STREQUAL "BUILD")
    export(EXPORT nanoarrow-exports
           FILE "${build_location}/nanoarrow-targets.cmake"
           NAMESPACE nanoarrow::)

  else()
    install(DIRECTORY "${build_location}/" DESTINATION "${install_location}")
    install(EXPORT nanoarrow-exports
            DESTINATION "${install_location}"
            FILE "nanoarrow-targets.cmake"
            NAMESPACE nanoarrow::)
  endif()
endforeach()
