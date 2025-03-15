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

import argparse
import pathlib
import re
import subprocess
import sys


# Generate the nanoarrow_c.pxd file used by the Cython extensions
class PxdGenerator:
    def __init__(self):
        self._define_regexes()

    def generate_pxd(self, file_in, file_out):
        file_in_name = pathlib.Path(file_in).name

        # Read the header
        content = None
        with open(file_in, "r") as input:
            content = input.read()

        # Strip comments
        content = self.re_comment.sub("", content)

        # Replace NANOARROW_MAX_FIXED_BUFFERS with its value
        content = self._preprocess_content(content)

        # Find typedefs, types, and function definitions
        typedefs = self._find_typedefs(content)
        types = self._find_types(content)
        func_defs = self._find_func_defs(content)

        # Make corresponding cython definitions
        typedefs_cython = [self._typdef_to_cython(t, "    ") for t in typedefs]
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
            self._write_defs(output)

            for type in types_cython:
                output.write(type.encode("UTF-8"))
                output.write(b"\n\n")

            for typedef in typedefs_cython:
                output.write(typedef.encode("UTF-8"))
                output.write(b"\n")

            output.write(b"\n")

            for func_def in func_defs_cython:
                output.write(func_def.encode("UTF-8"))
                output.write(b"\n")

    def _preprocess_content(self, content):
        return content

    def _write_defs(self, output):
        pass

    def _define_regexes(self):
        self.re_comment = re.compile(r"\s*//[^\n]*")
        self.re_typedef = re.compile(r"typedef(?P<typedef>[^;]+)")
        self.re_type = re.compile(
            r"(?P<type>struct|union|enum) (?P<name>Arrow[^ ]+) {(?P<body>[^}]*)}"
        )
        self.re_func_def = re.compile(
            r"\n(static inline |NANOARROW_DLL )(?P<const>const )?(struct |enum )?"
            r"(?P<return_type>[A-Za-z0-9_*]+)\s+"
            r"(?P<name>Arrow[A-Za-z0-9]+)\((?P<args>[^\)]*)\);"
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

    def _find_typedefs(self, content):
        return [m.groupdict() for m in self.re_typedef.finditer(content)]

    def _find_types(self, content):
        return [m.groupdict() for m in self.re_type.finditer(content)]

    def _find_func_defs(self, content):
        return [m.groupdict() for m in self.re_func_def.finditer(content)]

    def _typdef_to_cython(self, t, indent=""):
        typedef = t["typedef"]
        typedef = self.re_tagged_type.sub(r"\2", typedef)
        return f"{indent}ctypedef {typedef}"

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

        """


class NanoarrowPxdGenerator(PxdGenerator):
    def _preprocess_content(self, content):
        content = re.sub(r"NANOARROW_MAX_FIXED_BUFFERS", "3", content)
        content = re.sub(r"NANOARROW_BINARY_VIEW_INLINE_SIZE", "12", content)
        content = re.sub(r"NANOARROW_BINARY_VIEW_PREFIX_SIZE", "4", content)
        return content

    def _pxd_header(self):
        return (
            super()._pxd_header()
            + """
    from libc.stdint cimport int8_t, uint8_t, int16_t, uint16_t
    from libc.stdint cimport int32_t, uint32_t, int64_t, uint64_t
    """
        )

    def _write_defs(self, output):
        output.write(b"\n")
        output.write(b"    cdef int NANOARROW_OK\n")
        output.write(b"    cdef int NANOARROW_MAX_FIXED_BUFFERS\n")
        output.write(b"    cdef int ARROW_FLAG_DICTIONARY_ORDERED\n")
        output.write(b"    cdef int ARROW_FLAG_NULLABLE\n")
        output.write(b"    cdef int ARROW_FLAG_MAP_KEYS_SORTED\n")
        output.write(b"\n")


class NanoarrowDevicePxdGenerator(PxdGenerator):
    def _preprocess_content(self, content):
        self.device_names = re.findall("#define (ARROW_DEVICE_[A-Z0-9_]+)", content)
        return super()._preprocess_content(content)

    def _find_typedefs(self, content):
        return []

    def _pxd_header(self):
        return (
            super()._pxd_header()
            + """
    from libc.stdint cimport int32_t, int64_t
    from nanoarrow_c cimport *
    """
        )

    def _write_defs(self, output):
        output.write(b"\n")
        output.write(b"    ctypedef int32_t ArrowDeviceType\n")
        output.write(b"\n")
        for name in self.device_names:
            output.write(f"    cdef ArrowDeviceType {name}\n".encode())
        output.write(b"\n")


def copy_or_generate_nanoarrow_c(target_dir: pathlib.Path):
    vendored_files = [
        "nanoarrow.h",
        "nanoarrow.c",
        "nanoarrow_ipc.h",
        "nanoarrow_ipc.c",
        "nanoarrow_device.h",
        "nanoarrow_device.c",
    ]
    dst = {name: target_dir / name for name in vendored_files}

    this_dir = pathlib.Path(__file__).parent.resolve()
    arrow_proj_dir = this_dir / "subprojects" / "arrow-nanoarrow"

    subprocess.run(
        [
            sys.executable,
            arrow_proj_dir / "ci" / "scripts" / "bundle.py",
            "--symbol-namespace",
            "PythonPkg",
            "--header-namespace",
            "",
            "--source-output-dir",
            target_dir,
            "--include-output-dir",
            target_dir,
            "--with-device",
            "--with-ipc",
        ],
    )

    if not dst["nanoarrow.h"].exists():
        raise ValueError("Attempt to vendor nanoarrow.c/h failed")


# Runs the pxd generator with some information about the file name
def generate_nanoarrow_pxds(target_dir: pathlib.Path):
    NanoarrowPxdGenerator().generate_pxd(
        target_dir / "nanoarrow.h", target_dir / "nanoarrow_c.pxd"
    )
    NanoarrowDevicePxdGenerator().generate_pxd(
        target_dir / "nanoarrow_device.h",
        target_dir / "nanoarrow_device_c.pxd",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", help="Target directory where files should be written"
    )

    args = parser.parse_args()
    target_dir = pathlib.Path(args.output_dir).resolve()

    copy_or_generate_nanoarrow_c(target_dir)
    generate_nanoarrow_pxds(target_dir)
