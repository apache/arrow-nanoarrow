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

import nanoarrow as na


class CArrayBuilderSuite:
    """
    Benchmarks for building CArrays
    """

    def setup(self):
        self.py_integers = list(range(int(1e6)))
        self.py_bools = [False, True, True, False] * int(1e6 // 4)

        self.wide_schema = na.c_schema(na.struct([na.int32()] * 10000))
        self.children = [na.c_array(self.py_integers, na.int32())] * 10000

    def time_build_c_array_int32(self):
        """Create an int32 array from 1,000,000 Python integers"""
        na.c_array(self.py_integers, na.int32())

    def time_build_c_array_bool(self):
        """Create a bool array from 1,000,000 Python booleans"""
        na.c_array(self.py_bools, na.bool())

    def time_build_c_array_struct_wide(self):
        """Create a struct array with 10,000 columns"""
        na.c_array_from_buffers(self.wide_schema, 1e6, [None], children=self.children)


class ArrayIterationSuite:
    """Benchmarks for consuming an Array using various methods of iteration"""

    def setup(self):
        self.integers = na.Array(range(int(1e6)), na.int32())

        n = int(1e6)
        item_size = 7
        alphabet = b"abcdefghijklmnopqrstuvwxyz"
        n_alphabets = (item_size * n) // len(alphabet) + 1
        data_buffer = alphabet * n_alphabets
        offsets_buffer = na.c_buffer(
            range(0, (n + 1) * item_size, item_size), na.int32()
        )

        c_strings = na.c_array_from_buffers(
            na.string(), n, [None, offsets_buffer, data_buffer]
        )
        self.strings = na.Array(c_strings)

        c_long_struct = na.c_array_from_buffers(
            na.struct([na.int32()] * 100),
            length=10000,
            buffers=[None],
            children=[na.c_array(range(10000), na.int32())] * 100,
        )
        self.long_struct = na.Array(c_long_struct)

        c_wide_struct = na.c_array_from_buffers(
            na.struct([na.int32()] * 10000),
            length=100,
            buffers=[None],
            children=[na.c_array(range(100), na.int32())] * 10000,
        )
        self.wide_struct = na.Array(c_wide_struct)

    def time_integers_to_list(self):
        """Consume an int32 array with 1,000,000 elements into a Python list"""
        list(self.integers.iter_py())

    def time_strings_to_list(self):
        """Consume a string array with 1,000,000 elements into a Python list"""
        list(self.strings.iter_py())

    def time_long_struct_to_dict_list(self):
        """Consume an struct array with 10,000 elements and 100 columns into a list
        of dictionaries
        """
        list(self.long_struct.iter_py())

    def time_long_struct_to_tuple_list(self):
        """Consume an struct array with 10,000 elements and 100 columns into a list
        of tuples
        """
        list(self.long_struct.iter_tuples())

    def time_wide_struct_to_dict_list(self):
        """Consume an struct array with 100 elements and 10,000 columns into a list
        of dictionaries
        """
        list(self.wide_struct.iter_py())

    def time_wide_struct_to_tuple_list(self):
        """Consume an struct array with 100 elements and 10,000 columns into a list
        of tuples
        """
        list(self.wide_struct.iter_tuples())
