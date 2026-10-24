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

"""Build IPC against external flatcc runtimes without exercising package exports."""

import os
import subprocess
import tempfile
from pathlib import Path


def run(*args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def main():
    source = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="nanoarrow-flatcc-") as tmp:
        root = Path(tmp)
        fixture = root / "fixture"
        fixture.mkdir()
        (fixture / "CMakeLists.txt").write_text(
            """cmake_minimum_required(VERSION 3.18)
project(flatcc_fixture C)
add_library(flatccrt ${RUNTIME_TYPE}
  "${NANOARROW_SOURCE}/thirdparty/flatcc/src/runtime/builder.c"
  "${NANOARROW_SOURCE}/thirdparty/flatcc/src/runtime/emitter.c"
  "${NANOARROW_SOURCE}/thirdparty/flatcc/src/runtime/verifier.c"
  "${NANOARROW_SOURCE}/thirdparty/flatcc/src/runtime/refmap.c")
target_include_directories(flatccrt PRIVATE
  "${NANOARROW_SOURCE}/thirdparty/flatcc/include")
set_target_properties(flatccrt PROPERTIES POSITION_INDEPENDENT_CODE ON
  WINDOWS_EXPORT_ALL_SYMBOLS ON DEBUG_POSTFIX "${RUNTIME_DEBUG_POSTFIX}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(subdir "debug/")
endif()
install(TARGETS flatccrt ARCHIVE DESTINATION "${subdir}lib"
  LIBRARY DESTINATION "${subdir}lib" RUNTIME DESTINATION "${subdir}bin")
install(DIRECTORY "${NANOARROW_SOURCE}/thirdparty/flatcc/include/"
  DESTINATION include)
""",
            encoding="utf-8",
        )
        consumer = root / "consumer"
        consumer.mkdir()
        (consumer / "CMakeLists.txt").write_text(
            """cmake_minimum_required(VERSION 3.18)
project(flatcc_consumer C)
set(NANOARROW_IPC ON CACHE BOOL "")
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
""",
            encoding="utf-8",
        )
        (consumer / "main.c").write_text(
            """#include <nanoarrow/nanoarrow_ipc.h>
int main(void) {
  struct ArrowIpcDecoder decoder;
  if (ArrowIpcDecoderInit(&decoder)) return 1;
  ArrowIpcDecoderReset(&decoder);
  return 0;
}
""",
            encoding="utf-8",
        )
        # On Windows static and import libraries have the same filename.
        layouts = [("static", ["STATIC"], "_d"), ("shared", ["SHARED"], "")]
        if os.name != "nt":
            layouts.append(("both", ["SHARED", "STATIC"], "_d"))
        for name, types, postfix in layouts:
            prefix = root / name
            for kind in types:
                for config in ("Release", "Debug"):
                    build = root / f"fixture-{name}-{kind}-{config}"
                    run(
                        "cmake",
                        "-S",
                        fixture,
                        "-B",
                        build,
                        f"-DNANOARROW_SOURCE={source.as_posix()}",
                        f"-DCMAKE_BUILD_TYPE={config}",
                        f"-DRUNTIME_TYPE={kind}",
                        f"-DRUNTIME_DEBUG_POSTFIX={postfix}",
                        f"-DCMAKE_INSTALL_PREFIX={prefix.as_posix()}",
                    )
                    run(
                        "cmake", "--build", build, "--config", config, "--parallel", "2"
                    )
                    run("cmake", "--install", build, "--config", config)
            for mode in ("root", "dirs"):
                for config in ("Release", "Debug"):
                    build = root / f"consumer-{name}-{mode}-{config}"
                    options = [f"-DNANOARROW_FLATCC_ROOT_DIR={prefix.as_posix()}"]
                    if mode == "dirs":
                        libdir = prefix / ("debug/lib" if config == "Debug" else "lib")
                        # Separate-directory discovery uses the unsuffixed name.
                        if config == "Debug" and postfix:
                            continue
                        include = (prefix / "include").as_posix()
                        options = [
                            f"-DNANOARROW_FLATCC_INCLUDE_DIR={include}",
                            f"-DNANOARROW_FLATCC_LIB_DIR={libdir.as_posix()}",
                        ]
                    run(
                        "cmake",
                        "-S",
                        consumer,
                        "-B",
                        build,
                        f"-DNANOARROW_SOURCE={source.as_posix()}",
                        f"-DCMAKE_BUILD_TYPE={config}",
                        f"-DEXPECT_DEBUG={'ON' if mode == 'root' else 'OFF'}",
                        f"-DEXPECT_STATIC={'ON' if 'STATIC' in types else 'OFF'}",
                        *options,
                    )
                    run(
                        "cmake",
                        "--build",
                        build,
                        "--config",
                        config,
                        "--target",
                        "consumer",
                        "nanoarrow_ipc_shared",
                        "--parallel",
                        "2",
                    )
                    env = os.environ.copy()
                    bindir = prefix / ("debug/bin" if config == "Debug" else "bin")
                    env["PATH"] = os.pathsep.join([str(bindir), env["PATH"]])
                    # --test-dir was added after the supported CMake minimum.
                    run(
                        "ctest", "-C", config, "--output-on-failure", cwd=build, env=env
                    )
        missing = root / "missing"
        missing.mkdir()
        result = subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(root / "missing-build"),
                "-DNANOARROW_IPC=ON",
                f"-DNANOARROW_FLATCC_INCLUDE_DIR={source}/thirdparty/flatcc/include",
                f"-DNANOARROW_FLATCC_LIB_DIR={missing}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode == 0 or "NANOARROW_FLATCC_LIBRARY" not in result.stdout:
            raise RuntimeError(
                f"Expected missing-runtime configuration failure: {result.stdout}"
            )
        print("External flatcc discovery checks passed.")


if __name__ == "__main__":
    main()
