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
import os
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
                "Expected exactly one occurrence of "
                f"'{replace_key}' in '{paths_or_content}'"
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


def namespace_nanoarrow_includes(path_or_content, header_namespace="nanoarrow"):
    content = read_content(path_or_content)
    return re.sub(
        r'#include "nanoarrow/([^"]+)"', f'#include "{header_namespace}\\1"', content
    )


def bundle_nanoarrow(
    root_dir,
    symbol_namespace=None,
    header_namespace="nanoarrow/",
    source_namespace="src",
    cpp=False,
):
    root_dir = pathlib.Path(root_dir)
    src_dir = root_dir / "src" / "nanoarrow"
    include_dir_out = pathlib.Path("include") / header_namespace

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
    yield f"{include_dir_out}/nanoarrow.h", nanoarrow_h

    # Generate nanoarrow/nanoarrow.hpp
    nanoarrow_hpp = read_content(src_dir / "nanoarrow.hpp")
    nanoarrow_hpp = namespace_nanoarrow_includes(nanoarrow_hpp, header_namespace)
    yield f"{include_dir_out}/nanoarrow.hpp", nanoarrow_hpp

    # Generate nanoarrow/nanoarrow_testing.hpp
    nanoarrow_testing_hpp = read_content(src_dir / "nanoarrow_testing.hpp")
    nanoarrow_testing_hpp = namespace_nanoarrow_includes(
        nanoarrow_testing_hpp, header_namespace
    )
    yield f"{include_dir_out}/nanoarrow_testing.hpp", nanoarrow_testing_hpp

    # Generate nanoarrow/nanoarrow_gtest_util.hpp
    nanoarrow_gtest_util_hpp = read_content(src_dir / "nanoarrow_gtest_util.hpp")
    nanoarrow_gtest_util_hpp = namespace_nanoarrow_includes(
        nanoarrow_gtest_util_hpp, header_namespace
    )
    yield f"{include_dir_out}/nanoarrow_gtest_util.hpp", nanoarrow_gtest_util_hpp

    # Generate nanoarrow/nanoarrow.c
    nanoarrow_c = concatenate_content(
        [
            src_dir / "utils.c",
            src_dir / "schema.c",
            src_dir / "array.c",
            src_dir / "array_stream.c",
        ]
    )
    nanoarrow_c = namespace_nanoarrow_includes(nanoarrow_c, header_namespace)

    if cpp:
        yield f"{source_namespace}/nanoarrow.cc", nanoarrow_c
    else:
        yield f"{source_namespace}/nanoarrow.c", nanoarrow_c


def ensure_output_path_exists(out_path: pathlib.Path):
    if out_path.is_dir() and out_path.exists():
        return

    if out_path.is_file() and out_path.exists():
        raise ValueError(f"Can't create directory '{out_path}': exists and is a file")

    ensure_output_path_exists(out_path.parent)
    os.mkdir(out_path)


def do_bundle(out_dir, bundler):
    out_dir = pathlib.Path(out_dir)

    for out_file, out_content in bundler:
        out_path = out_dir / out_file
        ensure_output_path_exists(out_path.parent)
        write_content(out_content, out_path)


if __name__ == "__main__":
    do_bundle(
        "dist_test", bundle_nanoarrow(pathlib.Path(__file__).parent.parent.parent)
    )