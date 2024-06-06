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

import io
import pathlib
import re


def read_content(path_or_content):
    if isinstance(path_or_content, pathlib.Path):
        with open(path_or_content) as f:
            return f.read()
    else:
        return str(path_or_content)


def write_content(path_or_content, out_path):
    with open(out_path, "w") as f:
        f.write(read_content(path_or_content))


def configure_content(paths_or_content, args):
    content = read_content(paths_or_content)

    for key, value in args.items():
        replace_key = f"@{key}@"
        if content.count(replace_key) != 1:
            raise ValueError(
                f"Expected exactly one occurrence of '{replace_key}' in '{paths_or_content}'"
            )

        content = content.replace(replace_key, str(value))

    return content


def concatenate_content(paths_or_content):
    out = io.StringIO()

    for path in paths_or_content:
        out.write(read_content(path))

    return out.getvalue()


def cmakelist_version(path_or_content):
    content = read_content(path_or_content)
    version_match = re.search(r'set\(NANOARROW_VERSION "(.*?)"\)', content)
    if version_match is None:
        raise ValueError(f"Can't find NANOARROW_VERSION in '{path_or_content}'")

    version = version_match.group(1)

    component_match = re.search(r"^([0-9]+)\.([0-9]+)\.([0-9]+)", version)
    return (version,) + tuple(int(component) for component in component_match.groups())


def bundle_nanoarrow(
    root_dir, symbol_namespace=None, header_namespace="nanoarrow", cpp=False
):
    root_dir = pathlib.Path(root_dir)
    src_dir = root_dir / "src" / "nanoarrow"

    version, major, minor, patch = cmakelist_version(root_dir / "CMakeLists.txt")

    if symbol_namespace is None:
        namespace_define = "// #define NANOARROW_NAMESPACE YourNamespaceHere"
    else:
        namespace_define = f"#define NANOARROW_NAMESPACE {symbol_namespace}"

    nanoarrow_config_h = configure_content(
        src_dir / "nanoarrow_config.h.in",
        {
            "NANOARROW_VERSION": version,
            "NANOARROW_VERSION_MAJOR": major,
            "NANOARROW_VERSION_MINOR": minor,
            "NANOARROW_VERSION_PATCH": patch,
            "NANOARROW_NAMESPACE_DEFINE": namespace_define,
        },
    )

    # Generate nanoarrow/nanoarrow.h
    nanoarrow_h = concatenate_content(
        [
            nanoarrow_config_h,
            src_dir / "nanoarrow_types.h",
            src_dir / "nanoarrow.h",
            src_dir / "buffer_inline.h",
            src_dir / "array_inline.h",
        ]
    )

    nanoarrow_h = re.sub(r'#include "[a-z_.]+"', "", nanoarrow_h)
    yield f"{header_namespace}/nanoarrow.h", nanoarrow_h

    # Generate nanoarrow/nanoarrow.hpp
    yield f"{header_namespace}/nanoarrow.hpp", read_content(src_dir / "nanoarrow.hpp")

    # Generate nanoarrow/nanoarrow_testing.hpp
    yield f"{header_namespace}/nanoarrow_testing.hpp", read_content(
        src_dir / "nanoarrow_testing.hpp"
    )

    # Generate nanoarrow/nanoarrow_gtest_util.hpp
    yield f"{header_namespace}/nanoarrow_gtest_util.hpp", read_content(
        src_dir / "nanoarrow_gtest_util.hpp"
    )

    # Generate nanoarrow/nanoarrow.c
    nanoarrow_c = concatenate_content(
        [
            src_dir / "utils.c",
            src_dir / "schema.c",
            src_dir / "array.c",
            src_dir / "array_stream.c",
        ]
    )

    if cpp:
        yield f"{header_namespace}/nanoarrow.cc", nanoarrow_c
    else:
        yield f"{header_namespace}/nanoarrow.c", nanoarrow_c


if __name__ == "__main__":
    list(bundle_nanoarrow(pathlib.Path(__file__).parent.parent.parent))
