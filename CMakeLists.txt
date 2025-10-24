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

set(NANOARROW_VERSION "0.8.0-SNAPSHOT")
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" NANOARROW_BASE_VERSION
             "${NANOARROW_VERSION}")
project(nanoarrow VERSION "${NANOARROW_BASE_VERSION}")

set(NANOARROW_VERSION_MAJOR "${nanoarrow_VERSION_MAJOR}")
set(NANOARROW_VERSION_MINOR "${nanoarrow_VERSION_MINOR}")
set(NANOARROW_VERSION_PATCH "${nanoarrow_VERSION_PATCH}")

include(GNUInstallDirs)

# General options
option(NANOARROW_NAMESPACE "A prefix for exported symbols" OFF)

# Feature options
option(NANOARROW_IPC "Build IPC extension" OFF)
option(NANOARROW_FLATCC_ROOT_DIR "Root directory for flatcc include and lib directories"
       OFF)
option(NANOARROW_FLATCC_INCLUDE_DIR "Include directory for flatcc includes" OFF)
option(NANOARROW_FLATCC_LIB_DIR "Library directory that contains libflatccrt.a" OFF)
option(NANOARROW_IPC_WITH_ZSTD "Build nanoarrow with ZSTD compression support built in"
       OFF)
option(NANOARROW_IPC_WITH_LZ4 "Build nanoarrow with LZ4 compression support built in"
       OFF)

option(NANOARROW_DEVICE "Build device extension" OFF)
option(NANOARROW_TESTING "Build testing extension" OFF)
option(NANOARROW_DEVICE_WITH_METAL "Build Apple metal libraries" OFF)
option(NANOARROW_DEVICE_WITH_CUDA "Build CUDA libraries" OFF)

# Development options
option(NANOARROW_BUILD_APPS "Build utility applications" OFF)
option(NANOARROW_BUILD_TESTS "Build tests" OFF)
option(NANOARROW_BUILD_TESTS_WITH_ARROW "Build tests that require Arrow" OFF)
option(NANOARROW_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(NANOARROW_BUILD_INTEGRATION_TESTS
       "Build cross-implementation Arrow integration tests" OFF)
option(NANOARROW_BUNDLE "Create bundled nanoarrow.h and nanoarrow.c" OFF)
option(NANOARROW_BUNDLE_AS_CPP "Bundle nanoarrow source file as nanoarrow.cc" OFF)
option(NANOARROW_CODE_COVERAGE "Enable coverage reporting" OFF)
option(NANOARROW_ARROW_STATIC
       "Use a statically-linked Arrow C++ build when linking tests" OFF)
option(NANOARROW_DEBUG_EXTRA_WARNINGS "Use extra warnings when compiling in debug mode"
       ON)
# For vcpkg, where there's a check that fails if .dlls are present for static linkage
option(NANOARROW_INSTALL_SHARED "Install shared libraries" ON)

if(NOT DEFINED CMAKE_C_STANDARD)
  if(NANOARROW_IPC)
    # flatcc requires C11 for alignas() and static_assert() in flatcc_generated.h
    # It may be possible to use C99 mode to build the runtime and/or generated header
    # should this cause problems for users.
    set(CMAKE_C_STANDARD 11)
  else()
    set(CMAKE_C_STANDARD 99)
  endif()
  set(CMAKE_C_STANDARD_REQUIRED ON)
endif()

if(NANOARROW_NAMESPACE)
  set(NANOARROW_NAMESPACE_DEFINE "#define NANOARROW_NAMESPACE ${NANOARROW_NAMESPACE}")
else()
  set(NANOARROW_NAMESPACE_DEFINE "// #define NANOARROW_NAMESPACE YourNamespaceHere")
endif()

add_library(nanoarrow_coverage_config INTERFACE)
install(TARGETS nanoarrow_coverage_config
        EXPORT nanoarrow-exports
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(NANOARROW_CODE_COVERAGE)
  target_compile_options(nanoarrow_coverage_config INTERFACE -O0 -g --coverage)
  target_link_options(nanoarrow_coverage_config INTERFACE --coverage)
endif()

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
  set(NANOARROW_BUNDLE_ARGS "--with-ipc" "--with-device" "--with-testing" "--with-flatcc")

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
                          ${NANOARROW_BUNDLE_ARGS})

  set(NANOARROW_BUILD_INCLUDE_DIR "${CMAKE_BINARY_DIR}/bundled/include")
  set(NANOARROW_IPC_BUILD_SOURCES "${CMAKE_BINARY_DIR}/bundled/src/nanoarrow_ipc.c")
  set(NANOARROW_DEVICE_BUILD_SOURCES "${CMAKE_BINARY_DIR}/bundled/src/nanoarrow_device.c")
  set(NANOARROW_TESTING_BUILD_SOURCES
      "${CMAKE_BINARY_DIR}/bundled/src/nanoarrow_testing.cc")
  set(NANOARROW_FLATCC_BUILD_SOURCES "${CMAKE_BINARY_DIR}/bundled/src/flatcc.c")
else()
  # Generate the version information
  configure_file(src/nanoarrow/nanoarrow_config.h.in src/nanoarrow/nanoarrow_config.h)
  set(NANOARROW_BUILD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
endif()

# Start the list of headers to install
set(NANOARROW_INSTALL_HEADERS ${NANOARROW_BUILD_INCLUDE_DIR}/nanoarrow/nanoarrow.h
                              ${NANOARROW_BUILD_INCLUDE_DIR}/nanoarrow/nanoarrow.hpp)

# If the bundler didn't already assign the source files for the library, do so here.
# Also handle install() arguments that are different between the bundled and not
# bundled version.
if(NOT NANOARROW_BUNDLE)
  set(NANOARROW_BUILD_SOURCES
      src/nanoarrow/common/array.c src/nanoarrow/common/schema.c
      src/nanoarrow/common/array_stream.c src/nanoarrow/common/utils.c)

  install(FILES src/nanoarrow/common/inline_array.h src/nanoarrow/common/inline_buffer.h
                src/nanoarrow/common/inline_types.h DESTINATION include/nanoarrow/common)

  list(APPEND NANOARROW_INSTALL_HEADERS
       "${CMAKE_CURRENT_BINARY_DIR}/src/nanoarrow/nanoarrow_config.h")

  install(FILES src/nanoarrow/hpp/array_stream.hpp
                src/nanoarrow/hpp/buffer.hpp
                src/nanoarrow/hpp/exception.hpp
                src/nanoarrow/hpp/operators.hpp
                src/nanoarrow/hpp/unique.hpp
                src/nanoarrow/hpp/view.hpp
          DESTINATION include/nanoarrow/hpp)
endif()

# Add the nanoarrow library targets
add_library(nanoarrow_static STATIC ${NANOARROW_BUILD_SOURCES})
add_library(nanoarrow_shared SHARED ${NANOARROW_BUILD_SOURCES})
install(FILES ${NANOARROW_INSTALL_HEADERS} DESTINATION include/nanoarrow)
if(BUILD_SHARED_LIBS)
  add_library(nanoarrow::nanoarrow ALIAS nanoarrow_shared)
else()
  add_library(nanoarrow::nanoarrow ALIAS nanoarrow_static)
endif()

foreach(target "nanoarrow_static" "nanoarrow_shared")
  target_include_directories(${target}
                             PUBLIC $<BUILD_INTERFACE:${NANOARROW_BUILD_INCLUDE_DIR}>
                                    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/src>
                                    $<INSTALL_INTERFACE:include>)
endforeach()

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
    set_target_properties(flatccrt PROPERTIES POSITION_INDEPENDENT_CODE ON)
    install(TARGETS flatccrt
            EXPORT nanoarrow-exports
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

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

  if(NANOARROW_IPC_WITH_ZSTD)
    find_package(zstd REQUIRED)
    set(NANOARROW_IPC_EXTRA_FLAGS "-DNANOARROW_IPC_WITH_ZSTD")

    # libzstd_static is not available from conda
    # This could be configurable if shared zstd is a must
    if(TARGET zstd::libzstd_static)
      set(NANOARROW_IPC_EXTRA_LIBS zstd::libzstd_static)
    else()
      set(NANOARROW_IPC_EXTRA_LIBS zstd::libzstd)
    endif()
  endif()

  if(NANOARROW_IPC_WITH_LZ4)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LZ4 REQUIRED liblz4)

    message(STATUS "LZ4 library directories: ${LZ4_LIBRARY_DIRS}")
    message(STATUS "LZ4 include directories: ${LZ4_INCLUDE_DIRS}")
    message(STATUS "LZ4 libraries: ${LZ4_LIBRARIES}")

    add_library(lz4::lz4 INTERFACE IMPORTED)
    target_include_directories(lz4::lz4 INTERFACE ${LZ4_INCLUDE_DIRS})
    target_link_directories(lz4::lz4 INTERFACE ${LZ4_LIBRARY_DIRS})
    target_link_libraries(lz4::lz4 INTERFACE ${LZ4_LIBRARIES})

    set(NANOARROW_IPC_EXTRA_FLAGS "-DNANOARROW_IPC_WITH_LZ4")
    set(NANOARROW_IPC_EXTRA_LIBS ${NANOARROW_IPC_EXTRA_LIBS} lz4::lz4)
  endif()

  if(NOT NANOARROW_BUNDLE)
    set(NANOARROW_IPC_BUILD_SOURCES
        src/nanoarrow/ipc/codecs.c
        src/nanoarrow/ipc/decoder.c
        src/nanoarrow/ipc/encoder.c
        src/nanoarrow/ipc/reader.c
        src/nanoarrow/ipc/writer.c)
  endif()

  add_library(nanoarrow_ipc_static STATIC ${NANOARROW_IPC_BUILD_SOURCES})
  target_link_libraries(nanoarrow_ipc_static
                        PRIVATE flatccrt ${NANOARROW_IPC_EXTRA_LIBS}
                        PUBLIC nanoarrow_static nanoarrow_coverage_config)

  add_library(nanoarrow_ipc_shared SHARED ${NANOARROW_IPC_BUILD_SOURCES})
  target_link_libraries(nanoarrow_ipc_shared
                        PRIVATE flatccrt ${NANOARROW_IPC_EXTRA_LIBS}
                        PUBLIC nanoarrow_shared nanoarrow_coverage_config)

  install(FILES src/nanoarrow/nanoarrow_ipc.h src/nanoarrow/nanoarrow_ipc.hpp
                src/nanoarrow/ipc/flatcc_generated.h DESTINATION include/nanoarrow)
  if(BUILD_SHARED_LIBS)
    add_library(nanoarrow::nanoarrow_ipc ALIAS nanoarrow_ipc_shared)
  else()
    add_library(nanoarrow::nanoarrow_ipc ALIAS nanoarrow_ipc_static)
  endif()

  foreach(target "nanoarrow_ipc_static" "nanoarrow_ipc_shared")
    target_compile_definitions(${target} PRIVATE ${NANOARROW_IPC_EXTRA_FLAGS})
    target_include_directories(${target}
                               PUBLIC $<BUILD_INTERFACE:${NANOARROW_BUILD_INCLUDE_DIR}>
                                      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/src>
                                      $<BUILD_INTERFACE:${NANOARROW_IPC_FLATCC_INCLUDE_DIR}>
                                      $<INSTALL_INTERFACE:include>)
  endforeach()

endif()

if(NANOARROW_IPC AND (NANOARROW_BUILD_INTEGRATION_TESTS OR NANOARROW_BUILD_TESTS))
  if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
  endif()
  add_executable(nanoarrow_ipc_integration src/nanoarrow/integration/ipc_integration.cc)
  target_include_directories(nanoarrow_ipc_integration
                             PUBLIC $<BUILD_INTERFACE:${NANOARROW_BUILD_INCLUDE_DIR}>
                                    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/src>
                                    $<INSTALL_INTERFACE:include>)
  target_link_libraries(nanoarrow_ipc_integration
                        PRIVATE nanoarrow_testing_static nanoarrow_ipc_static flatccrt
                                nanoarrow_coverage_config)
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

    set(NANOARROW_DEVICE_INCLUDE_METAL "${CMAKE_BINARY_DIR}/metal-cpp")
    add_library(nanoarrow_metal_impl STATIC src/nanoarrow/device/metal_impl.cc)
    target_link_libraries(nanoarrow_metal_impl
                          PRIVATE ${METAL_LIBRARY} ${FOUNDATION_LIBRARY}
                                  ${QUARTZ_CORE_LIBRARY})
    target_include_directories(nanoarrow_metal_impl
                               PRIVATE ${NANOARROW_DEVICE_INCLUDE_METAL})
    install(TARGETS nanoarrow_metal_impl
            EXPORT nanoarrow-exports
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

    set(NANOARROW_DEVICE_SOURCES_METAL src/nanoarrow/device/metal.cc)
    set(NANOARROW_DEVICE_DEFS_METAL "NANOARROW_DEVICE_WITH_METAL")
    set(NANOARROW_DEVICE_LIBS_METAL nanoarrow_metal_impl)
  endif()

  if(NANOARROW_DEVICE_WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(NANOARROW_DEVICE_SOURCES_CUDA src/nanoarrow/device/cuda.c)
    set(NANOARROW_DEVICE_LIBS_CUDA CUDA::cuda_driver)
    set(NANOARROW_DEVICE_DEFS_CUDA "NANOARROW_DEVICE_WITH_CUDA")
  endif()

  # If the bundler didn't already assign the source files for the library, do so here
  if(NOT NANOARROW_BUNDLE)
    set(NANOARROW_DEVICE_BUILD_SOURCES src/nanoarrow/device/device.c
                                       ${NANOARROW_DEVICE_SOURCES_CUDA})
  endif()

  add_library(nanoarrow_device_static STATIC ${NANOARROW_DEVICE_BUILD_SOURCES}
                                             ${NANOARROW_DEVICE_SOURCES_METAL})
  target_link_libraries(nanoarrow_device_static
                        PRIVATE ${NANOARROW_DEVICE_LIBS_CUDA}
                                ${NANOARROW_DEVICE_LIBS_METAL}
                        PUBLIC nanoarrow_static nanoarrow_coverage_config)

  add_library(nanoarrow_device_shared SHARED ${NANOARROW_DEVICE_BUILD_SOURCES}
                                             ${NANOARROW_DEVICE_SOURCES_METAL})
  target_link_libraries(nanoarrow_device_shared
                        PRIVATE ${NANOARROW_DEVICE_LIBS_CUDA}
                                ${NANOARROW_DEVICE_LIBS_METAL}
                        PUBLIC nanoarrow_shared nanoarrow_coverage_config)

  install(FILES src/nanoarrow/nanoarrow_device.h src/nanoarrow/nanoarrow_device.hpp
          DESTINATION include/nanoarrow)
  if(BUILD_SHARED_LIBS)
    add_library(nanoarrow::nanoarrow_device ALIAS nanoarrow_device_shared)
  else()
    add_library(nanoarrow::nanoarrow_device ALIAS nanoarrow_device_static)
  endif()

  foreach(target "nanoarrow_device_static" "nanoarrow_device_shared")
    target_include_directories(${target}
                               PUBLIC $<BUILD_INTERFACE:${NANOARROW_BUILD_INCLUDE_DIR}>
                                      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/src>
                                      $<BUILD_INTERFACE:${NANOARROW_DEVICE_INCLUDE_METAL}>
                                      $<INSTALL_INTERFACE:include>)

    target_compile_definitions(${target} PRIVATE ${NANOARROW_DEVICE_DEFS_METAL}
                                                 ${NANOARROW_DEVICE_DEFS_CUDA})

  endforeach()

endif()

if(NANOARROW_TESTING
   OR NANOARROW_BUILD_TESTS
   OR NANOARROW_BUILD_INTEGRATION_TESTS)
  if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
  endif()

  if(NOT NANOARROW_BUNDLE)
    set(NANOARROW_TESTING_BUILD_SOURCES src/nanoarrow/testing/testing.cc)
  endif()

  add_subdirectory("thirdparty/nlohmann_json")
  install(TARGETS nlohmann_json
          EXPORT nanoarrow-exports
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

  add_library(nanoarrow_testing_static STATIC src/nanoarrow/testing/testing.cc)
  target_link_libraries(nanoarrow_testing_static
                        PRIVATE nlohmann_json::nlohmann_json
                        PUBLIC nanoarrow_static nanoarrow_coverage_config)

  add_library(nanoarrow_testing_shared SHARED src/nanoarrow/testing/testing.cc)
  target_link_libraries(nanoarrow_testing_shared
                        PRIVATE nlohmann_json::nlohmann_json
                        PUBLIC nanoarrow_shared nanoarrow_coverage_config)

  install(FILES src/nanoarrow/nanoarrow_testing.hpp DESTINATION include/nanoarrow)
  if(BUILD_SHARED_LIBS)
    add_library(nanoarrow::nanoarrow_testing ALIAS nanoarrow_testing_shared)
  else()
    add_library(nanoarrow::nanoarrow_testing ALIAS nanoarrow_testing_static)
  endif()

  foreach(target "nanoarrow_testing_static" "nanoarrow_testing_shared")
    target_include_directories(${target}
                               PUBLIC $<BUILD_INTERFACE:${NANOARROW_BUILD_INCLUDE_DIR}>
                                      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/src>
                                      $<INSTALL_INTERFACE:include>)

    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endforeach()
endif()

# Always build integration test if building tests
if(NANOARROW_BUILD_TESTS OR NANOARROW_BUILD_INTEGRATION_TESTS)
  set_target_properties(nanoarrow_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
  add_library(nanoarrow_c_data_integration SHARED
              src/nanoarrow/integration/c_data_integration.cc)
  target_compile_definitions(nanoarrow_c_data_integration PRIVATE NANOARROW_BUILD_DLL
                                                                  NANOARROW_EXPORT_DLL)
  target_compile_definitions(nanoarrow_c_data_integration PUBLIC NANOARROW_BUILD_DLL)

  target_include_directories(nanoarrow_c_data_integration
                             PUBLIC $<BUILD_INTERFACE:${NANOARROW_BUILD_INCLUDE_DIR}>
                                    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/src>
                                    $<INSTALL_INTERFACE:include>)
  target_link_libraries(nanoarrow_c_data_integration PRIVATE nanoarrow_testing_static)
endif()

# Common configuration for all targets
foreach(target
        nanoarrow_static
        nanoarrow_shared
        nanoarrow_ipc_static
        nanoarrow_ipc_shared
        nanoarrow_device_static
        nanoarrow_device_shared
        nanoarrow_testing_static
        nanoarrow_testing_shared
        nanoarrow_c_data_integration
        nanoarrow_ipc_json_integration)
  if(TARGET ${target})
    # Define the nanoarrow::xxx alias target
    add_library(nanoarrow::${target} ALIAS ${target})

    # Ensure NANOARROW_DEBUG is defined for debug builds
    target_compile_definitions(${target} PUBLIC "$<$<CONFIG:Debug>:NANOARROW_DEBUG>")

    # Ensure NANOARROW_DLL is set when building and linking to the library
    get_target_property(target_type ${target} TYPE)
    if(target_type STREQUAL "SHARED_LIBRARY")
      target_compile_definitions(${target} PUBLIC NANOARROW_BUILD_DLL)
      target_compile_definitions(${target} PRIVATE NANOARROW_BUILD_DLL
                                                   NANOARROW_EXPORT_DLL)
    endif()

    # Ensure target is added to nanoarrow-exports
    if(((target_type STREQUAL "SHARED_LIBRARY") AND NANOARROW_INSTALL_SHARED)
       OR NOT (target_type STREQUAL "SHARED_LIBRARY"))
      install(TARGETS ${target}
              EXPORT nanoarrow-exports
              RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
              LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
              ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endif()

    # For debug builds, ensure we aggressively set compiler warning flags and
    # error for any compiler warnings
    if(NANOARROW_DEBUG_EXTRA_WARNINGS)
      if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target}
                               PRIVATE $<$<CONFIG:Debug>:-Wall
                                       -Werror
                                       -Wextra
                                       -Wpedantic
                                       -Wno-type-limits
                                       -Wmaybe-uninitialized
                                       -Wunused-result
                                       -Wconversion
                                       -Wno-sign-conversion>)
        target_compile_options(${target} PRIVATE -Wno-misleading-indentation)
      elseif(CMAKE_C_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_C_COMPILER_ID STREQUAL
                                                          "Clang")
        target_compile_options(${target}
                               PRIVATE $<$<CONFIG:Debug>:-Wall
                                       -Werror
                                       -Wextra
                                       -Wpedantic
                                       -Wdocumentation
                                       -Wconversion
                                       -Wno-sign-conversion>)
      endif()
    endif()
  endif()
endforeach()

if(NANOARROW_BUILD_TESTS)
  set(MEMORYCHECK_COMMAND_OPTIONS
      "--leak-check=full --suppressions=${CMAKE_CURRENT_LIST_DIR}/valgrind.supp --error-exitcode=1"
  )
  include(CTest)

  if(NANOARROW_TEST_LINKAGE_SHARED)
    set(NANOARROW_TEST_LIB nanoarrow_shared)
    set(NANOARROW_TEST_IPC_LIB nanoarrow_ipc_shared)
    set(NANOARROW_TEST_DEVICE_LIB nanoarrow_device_shared)
    set(NANOARROW_TEST_TESTING_LIB nanoarrow_testing_shared)
  else()
    set(NANOARROW_TEST_LIB nanoarrow_static)
    set(NANOARROW_TEST_IPC_LIB nanoarrow_ipc_static)
    set(NANOARROW_TEST_DEVICE_LIB nanoarrow_device_static)
    set(NANOARROW_TEST_TESTING_LIB nanoarrow_testing_static)
  endif()

  if(NANOARROW_BUILD_TESTS_WITH_ARROW)
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
  endif()

  add_subdirectory("thirdparty/googletest")

  # Be sure to keep these tests in sync with src/nanoarrow/meson.build
  add_executable(utils_test src/nanoarrow/common/utils_test.cc)
  add_executable(buffer_test src/nanoarrow/common/buffer_test.cc)
  add_executable(array_test src/nanoarrow/common/array_test.cc)
  add_executable(schema_test src/nanoarrow/common/schema_test.cc)
  add_executable(array_stream_test src/nanoarrow/common/array_stream_test.cc)
  add_executable(nanoarrow_testing_test src/nanoarrow/testing/testing_test.cc)
  add_executable(c_data_integration_test
                 src/nanoarrow/integration/c_data_integration_test.cc)

  add_executable(hpp_array_stream src/nanoarrow/hpp/array_stream_test.cc)
  add_executable(hpp_buffer src/nanoarrow/hpp/buffer_test.cc)
  add_executable(hpp_exception src/nanoarrow/hpp/exception_test.cc)
  add_executable(hpp_unique src/nanoarrow/hpp/unique_test.cc)
  add_executable(hpp_view src/nanoarrow/hpp/view_test.cc)

  target_link_libraries(utils_test
                        ${NANOARROW_TEST_TESTING_LIB}
                        gtest_main
                        gmock_main
                        ${NANOARROW_ARROW_TARGET}
                        nanoarrow_coverage_config)
  target_link_libraries(buffer_test ${NANOARROW_TEST_LIB} gtest_main
                        nanoarrow_coverage_config)
  target_link_libraries(array_test
                        ${NANOARROW_TEST_LIB}
                        gtest_main
                        gmock_main
                        ${NANOARROW_ARROW_TARGET}
                        nanoarrow_coverage_config)
  target_link_libraries(schema_test
                        ${NANOARROW_TEST_LIB}
                        gtest_main
                        ${NANOARROW_ARROW_TARGET}
                        nanoarrow_coverage_config)
  target_link_libraries(array_stream_test
                        ${NANOARROW_TEST_LIB}
                        gtest_main
                        gmock_main
                        nanoarrow_coverage_config)
  target_link_libraries(nanoarrow_testing_test ${NANOARROW_TEST_TESTING_LIB} gtest_main
                        nanoarrow_coverage_config)
  target_link_libraries(c_data_integration_test
                        ${NANOARROW_TEST_LIB}
                        nanoarrow_c_data_integration
                        gtest_main
                        nanoarrow_coverage_config)
  target_link_libraries(hpp_array_stream
                        ${NANOARROW_TEST_LIB}
                        gtest_main
                        gmock_main
                        nanoarrow_coverage_config)
  target_link_libraries(hpp_buffer
                        ${NANOARROW_TEST_LIB}
                        gtest_main
                        gmock_main
                        nanoarrow_coverage_config)
  target_link_libraries(hpp_exception
                        ${NANOARROW_TEST_LIB}
                        gtest_main
                        gmock_main
                        nanoarrow_coverage_config)
  target_link_libraries(hpp_unique
                        ${NANOARROW_TEST_LIB}
                        gtest_main
                        gmock_main
                        nanoarrow_coverage_config)
  target_link_libraries(hpp_view
                        ${NANOARROW_TEST_LIB}
                        gtest_main
                        gmock_main
                        nanoarrow_coverage_config)

  list(APPEND
       NanoarrowTests
       utils_test
       buffer_test
       array_test
       schema_test
       array_stream_test
       nanoarrow_testing_test
       c_data_integration_test
       hpp_array_stream
       hpp_buffer
       hpp_exception
       hpp_unique
       hpp_view)
  if(Arrow_FOUND)
    foreach(test_target ${NanoarrowTests})
      target_compile_definitions(${test_target}
                                 PRIVATE -DNANOARROW_BUILD_TESTS_WITH_ARROW)
    endforeach()
  endif()

  include(GoogleTest)
  # Some users have reported a timeout with the default value of 5
  # Building with -DBUILD_SHARED_LIBS=ON may also help reduce the size of test
  # executables.
  gtest_discover_tests(utils_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(buffer_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(array_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(schema_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(array_stream_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(nanoarrow_testing_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(c_data_integration_test DISCOVERY_TIMEOUT 10)
  gtest_discover_tests(hpp_array_stream)
  gtest_discover_tests(hpp_buffer)
  gtest_discover_tests(hpp_exception)
  gtest_discover_tests(hpp_unique)
  gtest_discover_tests(hpp_view)

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
    include(GoogleTest)

    foreach(name
            codecs
            decoder
            encoder
            reader
            writer
            files
            ipc_hpp)
      add_executable(nanoarrow_ipc_${name}_test src/nanoarrow/ipc/${name}_test.cc)

      target_link_libraries(nanoarrow_ipc_${name}_test
                            ${NANOARROW_TEST_IPC_LIB}
                            ${NANOARROW_ARROW_TARGET}
                            gtest_main
                            gmock_main
                            nanoarrow_coverage_config)

      if(NOT (name MATCHES "_hpp_"))
        target_link_libraries(nanoarrow_ipc_${name}_test flatccrt)
      endif()

      if(Arrow_FOUND)
        target_compile_definitions(nanoarrow_ipc_${name}_test
                                   PRIVATE -DNANOARROW_BUILD_TESTS_WITH_ARROW
                                           ${NANOARROW_IPC_EXTRA_FLAGS})
      else()
        target_compile_definitions(nanoarrow_ipc_${name}_test
                                   PRIVATE ${NANOARROW_IPC_EXTRA_FLAGS})
      endif()

      gtest_discover_tests(nanoarrow_ipc_${name}_test)
    endforeach()

    target_link_libraries(nanoarrow_ipc_files_test ${NANOARROW_TEST_TESTING_LIB}
                          ZLIB::ZLIB nanoarrow_coverage_config)
  endif()

  if(NANOARROW_DEVICE)
    enable_testing()
    add_executable(nanoarrow_device_test src/nanoarrow/device/device_test.cc)
    add_executable(nanoarrow_device_hpp_test src/nanoarrow/device/device_hpp_test.cc)

    target_link_libraries(nanoarrow_device_test ${NANOARROW_TEST_DEVICE_LIB} gtest_main
                          nanoarrow_coverage_config)
    target_link_libraries(nanoarrow_device_hpp_test ${NANOARROW_TEST_DEVICE_LIB}
                          gtest_main nanoarrow_coverage_config)

    if(Arrow_FOUND)
      target_compile_definitions(nanoarrow_device_test
                                 PRIVATE -DNANOARROW_BUILD_TESTS_WITH_ARROW)
      target_compile_definitions(nanoarrow_device_hpp_test
                                 PRIVATE -DNANOARROW_BUILD_TESTS_WITH_ARROW)
    endif()

    include(GoogleTest)
    gtest_discover_tests(nanoarrow_device_test)
    gtest_discover_tests(nanoarrow_device_hpp_test)

    if(NANOARROW_DEVICE_WITH_METAL)
      add_executable(nanoarrow_device_metal_test src/nanoarrow/device/metal_test.cc)
      target_link_libraries(nanoarrow_device_metal_test
                            ${NANOARROW_TEST_DEVICE_LIB}
                            ${NANOARROW_TEST_LIB}
                            gtest_main
                            nanoarrow_coverage_config)
      target_include_directories(nanoarrow_device_metal_test
                                 PRIVATE ${NANOARROW_DEVICE_INCLUDE_METAL})

      if(Arrow_FOUND)
        target_compile_definitions(nanoarrow_device_metal_test
                                   PRIVATE -DNANOARROW_BUILD_TESTS_WITH_ARROW)
      endif()
      gtest_discover_tests(nanoarrow_device_metal_test)
    endif()

    if(NANOARROW_DEVICE_WITH_CUDA)
      add_executable(nanoarrow_device_cuda_test src/nanoarrow/device/cuda_test.cc)
      target_link_libraries(nanoarrow_device_cuda_test
                            ${NANOARROW_TEST_DEVICE_LIB}
                            ${NANOARROW_TEST_LIB}
                            CUDA::cuda_driver
                            gtest_main
                            nanoarrow_coverage_config)
      if(Arrow_FOUND)
        target_compile_definitions(nanoarrow_device_cuda_test
                                   PRIVATE -DNANOARROW_BUILD_TESTS_WITH_ARROW)
      endif()
      gtest_discover_tests(nanoarrow_device_cuda_test)
    endif()
  endif()
endif()

if(NANOARROW_BUILD_APPS)
  if(NANOARROW_IPC)
    add_executable(dump_stream src/apps/dump_stream.c)
    target_link_libraries(dump_stream nanoarrow_ipc_static)
  endif()
endif()

if(NANOARROW_BUILD_BENCHMARKS)
  add_subdirectory(dev/benchmarks)
endif()

# Generate package files for the build and install trees.
include(CMakePackageConfigHelpers)

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
