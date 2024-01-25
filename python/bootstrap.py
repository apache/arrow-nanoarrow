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

import os
import re
import shutil
import subprocess
import tempfile
import warnings


# Generate the nanoarrow_c.pxd file used by the Cython extension
class NanoarrowPxdGenerator:
    def __init__(self):
        self._define_regexes()

    def generate_nanoarrow_pxd(self, file_in, file_out):
        file_in_name = os.path.basename(file_in)

        # Read the nanoarrow.h header
        content = None
        with open(file_in, "r") as input:
            content = input.read()

        # Strip comments
        content = self.re_comment.sub("", content)

        # Replace NANOARROW_MAX_FIXED_BUFFERS with its value
        content = self.re_max_buffers.sub("3", content)

        # Find types and function definitions
        types = self._find_types(content)
        func_defs = self._find_func_defs(content)

        # Make corresponding cython definitions
        types_cython = [self._type_to_cython(t, "    ") for t in types]
        func_defs_cython = [self._func_def_to_cython(d, "    ") for d in func_defs]

        # Unindent the header
        header = self.re_newline_plus_indent.sub("\n", self._pxd_header())

        # Write nanoarrow_c.pxd
        with open(file_out, "wb") as output:
            output.write(header.encode("UTF-8"))

            output.write(
                f'\ncdef extern from "{file_in_name}" nogil:\n'.encode("UTF-8")
            )

            # A few things we add in manually
            output.write(b"\n")
            output.write(b"    ctypedef int ArrowErrorCode\n")
            output.write(b"    cdef int NANOARROW_OK\n")
            output.write(b"    cdef int NANOARROW_MAX_FIXED_BUFFERS\n")
            output.write(b"    cdef int ARROW_FLAG_DICTIONARY_ORDERED\n")
            output.write(b"    cdef int ARROW_FLAG_NULLABLE\n")
            output.write(b"    cdef int ARROW_FLAG_MAP_KEYS_SORTED\n")
            output.write(b"\n")

            for type in types_cython:
                output.write(type.encode("UTF-8"))
                output.write(b"\n\n")

            for func_def in func_defs_cython:
                output.write(func_def.encode("UTF-8"))
                output.write(b"\n")

    def _define_regexes(self):
        self.re_comment = re.compile(r"\s*//[^\n]*")
        self.re_max_buffers = re.compile(r"NANOARROW_MAX_FIXED_BUFFERS")
        self.re_type = re.compile(
            r"(?P<type>struct|union|enum) (?P<name>Arrow[^ ]+) {(?P<body>[^}]*)}"
        )
        self.re_func_def = re.compile(
            r"\n(static inline )?(?P<const>const )?(struct|enum )?"
            r"(?P<return_type>[A-Za-z0-9_*]+) "
            r"(?P<name>Arrow[A-Za-z]+)\((?P<args>[^\)]*)\);"
        )
        self.re_tagged_type = re.compile(
            r"(?P<type>struct|union|enum) (?P<name>Arrow[A-Za-z]+)"
        )
        self.re_struct_delim = re.compile(r";\s*")
        self.re_enum_delim = re.compile(r",\s*")
        self.re_whitespace = re.compile(r"\s+")
        self.re_newline_plus_indent = re.compile(r"\n +")

    def _strip_comments(self, content):
        return self.re_comment.sub("", content)

    def _find_types(self, content):
        return [m.groupdict() for m in self.re_type.finditer(content)]

    def _find_func_defs(self, content):
        return [m.groupdict() for m in self.re_func_def.finditer(content)]

    def _type_to_cython(self, t, indent=""):
        type = t["type"]
        name = t["name"]
        body = self.re_tagged_type.sub(r"\2", t["body"].strip())
        if type == "enum":
            items = [item for item in self.re_enum_delim.split(body) if item]
        else:
            items = [item for item in self.re_struct_delim.split(body) if item]

        cython_body = f"\n{indent}    ".join([""] + items)
        return f"{indent}{type} {name}:{cython_body}"

    def _func_def_to_cython(self, d, indent=""):
        return_type = d["return_type"].strip()
        if d["const"]:
            return_type = "const " + return_type
        name = d["name"]
        args = re.sub(r"\s+", " ", d["args"].strip())
        args = self.re_tagged_type.sub(r"\2", args)

        # Cython doesn't do (void)
        if args == "void":
            args = ""

        return f"{indent}{return_type} {name}({args})"

    def _pxd_header(self):
        return """
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

        # cython: language_level = 3

        from libc.stdint cimport int8_t, uint8_t, int16_t, uint16_t
        from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t
        """


# Runs cmake -DNANOARROW_BUNDLE=ON if cmake exists or copies nanoarrow.c/h
# from ../dist if it does not. Running cmake is safer because it will sync
# any changes from nanoarrow C library sources in the checkout but is not
# strictly necessary for things like installing from GitHub.
def copy_or_generate_nanoarrow_c():
    this_dir = os.path.abspath(os.path.dirname(__file__))
    source_dir = os.path.dirname(this_dir)

    maybe_nanoarrow_h = os.path.join(this_dir, "src/nanoarrow/nanoarrow.h")
    maybe_nanoarrow_c = os.path.join(this_dir, "src/nanoarrow/nanoarrow.c")
    maybe_nanoarrow_device_h = os.path.join(
        this_dir, "src/nanoarrow/nanoarrow_device.h"
    )
    maybe_nanoarrow_device_c = os.path.join(
        this_dir, "src/nanoarrow/nanoarrow_device.c"
    )

    for f in (
        maybe_nanoarrow_c,
        maybe_nanoarrow_h,
        maybe_nanoarrow_device_h,
        maybe_nanoarrow_device_c,
    ):
        if os.path.exists(f):
            os.unlink(f)

    is_cmake_dir = "CMakeLists.txt" in os.listdir(source_dir)
    is_in_nanoarrow_repo = "nanoarrow.h" in os.listdir(
        os.path.join(source_dir, "src", "nanoarrow")
    )
    cmake_bin = os.getenv("CMAKE_BIN")
    if not cmake_bin:
        cmake_bin = "cmake"
    has_cmake = os.system(f"{cmake_bin} --version") == 0

    with tempfile.TemporaryDirectory() as build_dir:
        if is_in_nanoarrow_repo:
            device_ext_src = os.path.join(
                source_dir, "extensions/nanoarrow_device/src/nanoarrow"
            )
            shutil.copyfile(
                os.path.join(device_ext_src, "nanoarrow_device.h"),
                maybe_nanoarrow_device_h,
            )
            shutil.copyfile(
                os.path.join(device_ext_src, "nanoarrow_device.c"),
                maybe_nanoarrow_device_c,
            )

        if has_cmake and is_cmake_dir and is_in_nanoarrow_repo:
            try:
                subprocess.run(
                    [
                        cmake_bin,
                        "-B",
                        build_dir,
                        "-S",
                        source_dir,
                        "-DNANOARROW_BUNDLE=ON",
                        "-DNANOARROW_NAMESPACE=PythonPkg",
                    ]
                )
                subprocess.run(
                    [
                        cmake_bin,
                        "--install",
                        build_dir,
                        "--prefix",
                        os.path.join(this_dir, "src", "nanoarrow"),
                    ]
                )
            except Exception as e:
                warnings.warn(f"cmake call failed: {e}")
        else:
            raise ValueError(
                "Attempt to build source distribution outside the nanoarrow repo"
            )

    if not os.path.exists(os.path.join(this_dir, "src/nanoarrow/nanoarrow.h")):
        raise ValueError("Attempt to vendor nanoarrow.c/h failed")

    maybe_nanoarrow_hpp = os.path.join(this_dir, "src/nanoarrow/nanoarrow.hpp")
    if os.path.exists(maybe_nanoarrow_hpp):
        os.unlink(maybe_nanoarrow_hpp)


# Runs the pxd generator with some information about the file name
def generate_nanoarrow_pxd():
    this_dir = os.path.abspath(os.path.dirname(__file__))
    maybe_nanoarrow_h = os.path.join(this_dir, "src/nanoarrow/nanoarrow.h")
    maybe_nanoarrow_pxd = os.path.join(this_dir, "src/nanoarrow/nanoarrow_c.pxd")

    NanoarrowPxdGenerator().generate_nanoarrow_pxd(
        maybe_nanoarrow_h, maybe_nanoarrow_pxd
    )


if __name__ == "__main__":
    copy_or_generate_nanoarrow_c()
    generate_nanoarrow_pxd()
