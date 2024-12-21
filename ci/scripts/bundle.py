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
    output_source_dir="src",
    output_include_dir="include",
    cpp=False,
):
    root_dir = pathlib.Path(root_dir)
    src_dir = root_dir / "src" / "nanoarrow"

    output_source_dir = pathlib.Path(output_source_dir)
    output_include_dir = pathlib.Path(output_include_dir) / header_namespace

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
            src_dir / "common" / "inline_types.h",
            src_dir / "nanoarrow.h",
            src_dir / "common" / "inline_buffer.h",
            src_dir / "common" / "inline_array.h",
        ]
    )

    nanoarrow_h = re.sub(r'#include "(nanoarrow/)?[a-z_./]+"', "", nanoarrow_h)
    yield f"{output_include_dir}/nanoarrow.h", nanoarrow_h

    # Generate nanoarrow/nanoarrow.hpp
    nanoarrow_hpp = concatenate_content(
        [
            src_dir / "nanoarrow.hpp",
            src_dir / "hpp" / "exception.hpp",
            src_dir / "hpp" / "operators.hpp",
            src_dir / "hpp" / "unique.hpp",
            src_dir / "hpp" / "array_stream.hpp",
            src_dir / "hpp" / "buffer.hpp",
            src_dir / "hpp" / "view.hpp",
        ]
    )

    nanoarrow_hpp = re.sub(r'#include "(nanoarrow/)?hpp/[a-z_./]+"', "", nanoarrow_hpp)
    nanoarrow_hpp = namespace_nanoarrow_includes(nanoarrow_hpp, header_namespace)
    yield f"{output_include_dir}/nanoarrow.hpp", nanoarrow_hpp

    # Generate nanoarrow/nanoarrow.c
    nanoarrow_c = concatenate_content(
        [
            src_dir / "common" / "utils.c",
            src_dir / "common" / "schema.c",
            src_dir / "common" / "array.c",
            src_dir / "common" / "array_stream.c",
        ]
    )
    nanoarrow_c = namespace_nanoarrow_includes(nanoarrow_c, header_namespace)

    if cpp:
        yield f"{output_source_dir}/nanoarrow.cc", nanoarrow_c
    else:
        yield f"{output_source_dir}/nanoarrow.c", nanoarrow_c


def bundle_nanoarrow_device(
    root_dir,
    header_namespace="nanoarrow/",
    output_source_dir="src",
    output_include_dir="include",
):
    root_dir = pathlib.Path(root_dir)
    src_dir = root_dir / "src" / "nanoarrow"

    output_source_dir = pathlib.Path(output_source_dir)
    output_include_dir = pathlib.Path(output_include_dir) / header_namespace

    # Generate headers
    for filename in ["nanoarrow_device.h", "nanoarrow_device.hpp"]:
        content = read_content(src_dir / filename)
        content = namespace_nanoarrow_includes(content, header_namespace)
        yield f"{output_include_dir}/{filename}", content

    # Generate sources
    content = concatenate_content(
        [src_dir / "device" / "device.c", src_dir / "device" / "cuda.c"]
    )
    content = namespace_nanoarrow_includes(content, header_namespace)
    yield f"{output_source_dir}/nanoarrow_device.c", content


def bundle_nanoarrow_ipc(
    root_dir,
    header_namespace="nanoarrow/",
    output_source_dir="src",
    output_include_dir="include",
):
    root_dir = pathlib.Path(root_dir)
    src_dir = root_dir / "src" / "nanoarrow"

    output_source_dir = pathlib.Path(output_source_dir)
    output_include_dir = pathlib.Path(output_include_dir) / header_namespace

    # Generate headers
    for filename in [
        "nanoarrow_ipc.h",
        "nanoarrow_ipc.hpp",
    ]:
        content = read_content(src_dir / filename)
        content = namespace_nanoarrow_includes(content, header_namespace)
        yield f"{output_include_dir}/{filename}", content

    nanoarrow_ipc_c = concatenate_content(
        [
            src_dir / "ipc" / "flatcc_generated.h",
            src_dir / "ipc" / "codecs.c",
            src_dir / "ipc" / "decoder.c",
            src_dir / "ipc" / "encoder.c",
            src_dir / "ipc" / "reader.c",
            src_dir / "ipc" / "writer.c",
        ]
    )
    nanoarrow_ipc_c = nanoarrow_ipc_c.replace(
        '#include "nanoarrow/ipc/flatcc_generated.h"', ""
    )
    nanoarrow_ipc_c = namespace_nanoarrow_includes(nanoarrow_ipc_c, header_namespace)
    yield f"{output_source_dir}/nanoarrow_ipc.c", nanoarrow_ipc_c


def bundle_nanoarrow_testing(
    root_dir,
    header_namespace="nanoarrow/",
    output_source_dir="src",
    output_include_dir="include",
):
    root_dir = pathlib.Path(root_dir)
    src_dir = root_dir / "src" / "nanoarrow"

    output_source_dir = pathlib.Path(output_source_dir)
    output_include_dir = pathlib.Path(output_include_dir) / header_namespace

    # Generate headers
    for filename in [
        "nanoarrow_testing.hpp",
        "nanoarrow_gtest_util.hpp",
    ]:
        content = read_content(src_dir / filename)
        content = namespace_nanoarrow_includes(content, header_namespace)
        yield f"{output_include_dir}/{filename}", content

    nanoarrow_testing_cc = concatenate_content(
        [
            src_dir / "testing" / "testing.cc",
        ]
    )
    nanoarrow_testing_cc = namespace_nanoarrow_includes(
        nanoarrow_testing_cc, header_namespace
    )
    yield f"{output_source_dir}/nanoarrow_testing.cc", nanoarrow_testing_cc


def bundle_flatcc(
    root_dir,
    output_source_dir="src",
    output_include_dir="include",
):
    root_dir = pathlib.Path(root_dir)
    flatcc_dir = root_dir / "thirdparty" / "flatcc"

    output_source_dir = pathlib.Path(output_source_dir)
    output_include_dir = pathlib.Path(output_include_dir)

    # Generate headers
    include_dir = flatcc_dir / "include"
    for abs_filename in include_dir.glob("flatcc/**/*.h"):
        filename = abs_filename.relative_to(include_dir)
        yield f"{output_include_dir}/{filename}", read_content(
            flatcc_dir / "include" / filename
        )

    # Generate sources
    src_dir = flatcc_dir / "src" / "runtime"
    flatcc_c = concatenate_content(
        [
            src_dir / "builder.c",
            src_dir / "emitter.c",
            src_dir / "verifier.c",
            src_dir / "refmap.c",
        ]
    )

    yield f"{output_source_dir}/flatcc.c", flatcc_c


def ensure_output_path_exists(out_path: pathlib.Path):
    if out_path.is_dir() and out_path.exists():
        return

    if out_path.is_file() and out_path.exists():
        raise ValueError(f"Can't create directory '{out_path}': exists and is a file")

    ensure_output_path_exists(out_path.parent)
    os.mkdir(out_path)


def do_bundle(bundler):
    for out_file, out_content in bundler:
        out_path = pathlib.Path(out_file)
        ensure_output_path_exists(out_path.parent)
        write_content(out_content, out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bundled nanoarrow distribution")
    parser.add_argument(
        "--include-output-dir",
        help="include/ directory in which nanoarrow headers should be placed",
    )
    parser.add_argument(
        "--source-output-dir",
        help="Directory in which nanoarrow source files should be placed",
    )
    parser.add_argument(
        "--symbol-namespace", help="A value with which symbols should be prefixed"
    )
    parser.add_argument(
        "--header-namespace",
        help=(
            "The directory within include-output-dir that nanoarrow headers should be"
            "placed"
        ),
        default="nanoarrow/",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "If include-output-dir or source-output-dir are missing, ensures a single "
            "output directory with include/ and src/ subdirectories containing the "
            "headers and sources, respectively"
        ),
        default="dist",
    )
    parser.add_argument(
        "--cpp", help="Bundle sources as C++ where possible", action="store_true"
    )
    parser.add_argument(
        "--with-device",
        help="Include nanoarrow_device sources/headers",
        action="store_true",
    )
    parser.add_argument(
        "--with-ipc",
        help="Include nanoarrow_ipc sources/headers",
        action="store_true",
    )
    parser.add_argument(
        "--with-testing",
        help="Include nanoarrow_testing sources/headers",
        action="store_true",
    )
    parser.add_argument(
        "--with-flatcc",
        help="Include flatcc sources/headers",
        action="store_true",
    )

    args = parser.parse_args()
    if args.include_output_dir is None:
        args.include_output_dir = pathlib.Path(args.output_dir) / "include"
    if args.source_output_dir is None:
        args.source_output_dir = pathlib.Path(args.output_dir) / "src"

    root_dir = pathlib.Path(__file__).parent.parent.parent

    # Bundle nanoarrow
    do_bundle(
        bundle_nanoarrow(
            root_dir,
            symbol_namespace=args.symbol_namespace,
            header_namespace=args.header_namespace,
            output_source_dir=args.source_output_dir,
            output_include_dir=args.include_output_dir,
            cpp=args.cpp,
        )
    )

    # Bundle nanoarrow_device
    if args.with_device:
        do_bundle(
            bundle_nanoarrow_device(
                root_dir,
                header_namespace=args.header_namespace,
                output_source_dir=args.source_output_dir,
                output_include_dir=args.include_output_dir,
            )
        )

    # Bundle nanoarrow_ipc
    if args.with_ipc:
        do_bundle(
            bundle_nanoarrow_ipc(
                root_dir,
                header_namespace=args.header_namespace,
                output_source_dir=args.source_output_dir,
                output_include_dir=args.include_output_dir,
            )
        )

    # Bundle nanoarrow_testing
    if args.with_testing:
        do_bundle(
            bundle_nanoarrow_testing(
                root_dir,
                header_namespace=args.header_namespace,
                output_source_dir=args.source_output_dir,
                output_include_dir=args.include_output_dir,
            )
        )

    # Bundle flatcc
    if args.with_flatcc:
        do_bundle(
            bundle_flatcc(
                root_dir,
                output_source_dir=args.source_output_dir,
                output_include_dir=args.include_output_dir,
            )
        )
