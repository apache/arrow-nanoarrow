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

"""Exercise build-tree and installed CMake packages without test dependencies."""

import argparse
import pathlib
import shutil
import subprocess


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def configure(source, build, config, *options):
    run(
        "cmake",
        "-S",
        source,
        "-B",
        build,
        f"-DCMAKE_BUILD_TYPE={config}",
        *options,
    )


def build_and_install(build, config):
    run("cmake", "--build", build, "--config", config, "--parallel", "2")
    run("cmake", "--install", build, "--config", config)


def consume(source, build, prefix, config, caller_first):
    configure(
        source,
        build,
        config,
        f"-DCMAKE_PREFIX_PATH={prefix.as_posix()}",
        f"-DCALLER_FLATCC_FIRST={caller_first}",
    )
    run("cmake", "--build", build, "--config", config, "--parallel", "2")
    run("ctest", "--test-dir", build, "-C", config, "--output-on-failure")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    root = args.build_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = pathlib.Path(__file__).resolve().parents[2]

    # Build a real external runtime with the same layout as a package manager.
    fixture = root / "flatcc"
    fixture.mkdir()
    (fixture / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.14)
project(flatcc_fixture LANGUAGES C)
if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX _d)
endif()
add_library(flatccrt STATIC
  "${FLATCC_SOURCE}/src/runtime/builder.c"
  "${FLATCC_SOURCE}/src/runtime/emitter.c"
  "${FLATCC_SOURCE}/src/runtime/verifier.c"
  "${FLATCC_SOURCE}/src/runtime/refmap.c")
target_include_directories(flatccrt PRIVATE "${FLATCC_SOURCE}/include")
set_target_properties(flatccrt PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(destination debug/lib)
else()
  set(destination lib)
endif()
install(TARGETS flatccrt ARCHIVE DESTINATION "${destination}")
install(DIRECTORY "${FLATCC_SOURCE}/include/" DESTINATION include)
""",
        encoding="utf-8",
    )

    consumer = root / "consumer"
    consumer.mkdir()
    (consumer / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 3.14)
project(nanoarrow_package_consumer LANGUAGES C)
if(CALLER_FLATCC_FIRST)
  add_library(flatccrt STATIC IMPORTED)
endif()
find_package(nanoarrow CONFIG REQUIRED)
if(NOT CALLER_FLATCC_FIRST)
  add_library(flatccrt STATIC IMPORTED)
endif()
add_executable(consumer main.c)
target_link_libraries(consumer PRIVATE nanoarrow::nanoarrow_ipc_static)
enable_testing()
add_test(NAME nanoarrow-ipc-runtime COMMAND consumer)
""",
        encoding="utf-8",
    )
    (consumer / "main.c").write_text(
        """#include <nanoarrow/nanoarrow_ipc.h>
int main(void) {
  struct ArrowIpcDecoder decoder;
  int result = ArrowIpcDecoderInit(&decoder);
  if (result != 0) return result;
  ArrowIpcDecoderReset(&decoder);
  return 0;
}
""",
        encoding="utf-8",
    )

    for mode in ("external-root", "external-dirs", "vendored"):
        prefix = root / f"{mode}-prefix"
        configs = ("Release", "Debug")
        if mode != "vendored":
            for config in configs:
                build = root / f"{mode}-flatcc-{config}"
                configure(
                    fixture,
                    build,
                    config,
                    f"-DFLATCC_SOURCE={(source / 'thirdparty' / 'flatcc').as_posix()}",
                    f"-DCMAKE_INSTALL_PREFIX={prefix.as_posix()}",
                    *(["-DCMAKE_DEBUG_POSTFIX="] if mode == "external-dirs" else []),
                )
                build_and_install(build, config)

        for config in configs:
            build = root / f"{mode}-nanoarrow-{config}"
            options = [
                "-DNANOARROW_IPC=ON",
                "-DCMAKE_INSTALL_LIBDIR=lib",
                "-DCMAKE_DEBUG_POSTFIX=_d",
                f"-DCMAKE_INSTALL_PREFIX={prefix.as_posix()}",
            ]
            if mode == "external-root":
                options.append(f"-DNANOARROW_FLATCC_ROOT_DIR={prefix.as_posix()}")
            elif mode == "external-dirs":
                include_dir = prefix / "include"
                lib_dir = prefix / ("debug/lib" if config == "Debug" else "lib")
                options.extend(
                    [
                        f"-DNANOARROW_FLATCC_INCLUDE_DIR={include_dir.as_posix()}",
                        f"-DNANOARROW_FLATCC_LIB_DIR={lib_dir.as_posix()}",
                    ]
                )
            configure(source, build, config, *options)
            build_and_install(build, config)
            for caller_first in ("ON", "OFF"):
                consume(
                    consumer,
                    root / f"{mode}-build-consumer-{config}-{caller_first}",
                    build,
                    config,
                    caller_first,
                )

        relocated = root / f"{mode}-relocated"
        shutil.move(prefix, relocated)
        for config in configs:
            for caller_first in ("ON", "OFF"):
                consume(
                    consumer,
                    root / f"{mode}-installed-consumer-{config}-{caller_first}",
                    relocated,
                    config,
                    caller_first,
                )

    print("All CMake package consumers passed.")


if __name__ == "__main__":
    main()
