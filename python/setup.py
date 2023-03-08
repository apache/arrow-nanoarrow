#!/usr/bin/env python

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

import re
import os

import shutil
from pathlib import Path

from setuptools import Extension, setup

import numpy as np

class NanoarrowPxdGenerator:

    def __init__(self):
       self._define_regexes()

    def generate_nanoarrow_pxd(self, file_in, file_out):
        file_in_name = os.path.basename(file_in)

        # Read the nanoarrow.h header
        content = None
        with open(file_in, 'r') as input:
            content = input.read()

        # Strip comments
        content = self.re_comment.sub('', content)

        # Find types and function definitions
        types = self._find_types(content)
        func_defs = self._find_func_defs(content)

        # Make corresponding cython definitions
        types_cython = [self._type_to_cython(t, '    ') for t in types]
        func_defs_cython = [self._func_def_to_cython(d, '     ') for d in func_defs]

        # Unindent the header
        header = self.re_newline_plus_indent.sub('\n', self._pxd_header())

        # Write nanoarrow_c.pxd
        with open(file_out, 'wb') as output:
            output.write(header.encode('UTF-8'))

            output.write(f'\ncdef extern from "{file_in_name}":\n'.encode("UTF-8"))

            for type in types_cython:
                output.write(type.encode('UTF-8'))
                output.write(b'\n\n')

            for func_def in func_defs_cython:
                output.write(func_def.encode('UTF-8'))
                output.write(b'\n')

            output.write(b'\n')

    def _define_regexes(self):
        self.re_comment = re.compile(r'\s*//[^\n]*')
        self.re_type = re.compile(r'(?P<type>struct|union|enum) (?P<name>Arrow[^ ]+) {(?P<body>[^}]*)}')
        self.re_func_def = re.compile(r'\n(static inline )?(struct|enum )?(?P<return_type>[A-Za-z]+) (?P<name>Arrow[A-Za-z]+)\((?P<args>[^\)]*)\);')
        self.re_tagged_type = re.compile(r'(?P<type>struct|union|enum) (?P<name>Arrow[A-Za-z]+)')
        self.re_struct_delim = re.compile(r';\s*')
        self.re_enum_delim = re.compile(r',\s*')
        self.re_whitespace = re.compile(r'\s+')
        self.re_newline_plus_indent = re.compile(r'\n +')

    def _strip_comments(self, content):
        return self.re_comment.sub('', content)

    def _find_types(self, content):
        return [m.groupdict() for m in self.re_type.finditer(content)]

    def _find_func_defs(self, content):
        return [m.groupdict() for m in self.re_func_def.finditer(content)]

    def _type_to_cython(self, t, indent=''):
        type = t['type']
        name = t['name']
        body = self.re_tagged_type.sub(r'\2', t['body'].strip())
        if type == 'enum':
            items = [item for item in self.re_enum_delim.split(body) if item]
        else:
            items = [item for item in self.re_struct_delim.split(body) if item]

        cython_body = f'\n{indent}    '.join([''] + items)
        return f'{indent}cdef {type} {name}:{cython_body}'

    def _func_def_to_cython(self, d, indent=''):
        return_type = d['return_type']
        name = d['name']
        args = re.sub(r'\s+', ' ', d['args'].strip())
        args = self.re_tagged_type.sub(r'\2', args)
        return f'{indent}cdef {return_type} {name}({args})'

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

from libc.stdint cimport int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t
        """


# setuptools gets confused by relative paths that extend above the project root
target = Path(__file__).parent / "src" / "nanoarrow"
shutil.copy(
    Path(__file__).parent / "../dist/nanoarrow.c", target / "nanoarrow.c"
)
shutil.copy(
    Path(__file__).parent / "../dist/nanoarrow.h", target / "nanoarrow.h"
)

NanoarrowPxdGenerator().generate_nanoarrow_pxd(
    'src/nanoarrow/nanoarrow.h',
    'src/nanoarrow/nanoarrow_c.pxd'
)

setup(
    ext_modules=[
        Extension(
            name="nanoarrow._lib",
            include_dirs=[np.get_include(), "src/nanoarrow"],
            language="c",
            sources=[
                "src/nanoarrow/_lib.pyx",
                "src/nanoarrow/nanoarrow.c",
            ],
        )
    ]
)
