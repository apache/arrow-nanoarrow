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

from enum import IntEnum

BINARY: CArrowType
BINARY_VIEW: CArrowType
BOOL: CArrowType
DATE32: CArrowType
DATE64: CArrowType
DECIMAL128: CArrowType
DECIMAL256: CArrowType
DENSE_UNION: CArrowType
DICTIONARY: CArrowType
DOUBLE: CArrowType
DURATION: CArrowType
EXTENSION: CArrowType
FIXED_SIZE_BINARY: CArrowType
FIXED_SIZE_LIST: CArrowType
FLOAT: CArrowType
HALF_FLOAT: CArrowType
INT16: CArrowType
INT32: CArrowType
INT64: CArrowType
INT8: CArrowType
INTERVAL_DAY_TIME: CArrowType
INTERVAL_MONTHS: CArrowType
INTERVAL_MONTH_DAY_NANO: CArrowType
LARGE_BINARY: CArrowType
LARGE_LIST: CArrowType
LARGE_STRING: CArrowType
LIST: CArrowType
MAP: CArrowType
NA: CArrowType
RUN_END_ENCODED: CArrowType
SPARSE_UNION: CArrowType
STRING: CArrowType
STRING_VIEW: CArrowType
STRUCT: CArrowType
TIME32: CArrowType
TIME64: CArrowType
TIMESTAMP: CArrowType
UINT16: CArrowType
UINT32: CArrowType
UINT64: CArrowType
UINT8: CArrowType
UNINITIALIZED: CArrowType
sys_byteorder: str

class CArrowType(IntEnum):
    """ArrowType enumerator

    This enum makes the type identifier constants available to Cython and
    Python code as (e.g.) ``_types.UNINITIALIZED``. This removes the need
    for other modules to import type codes, which in previous versions had
    led to very long ``from nanoarrow_c cimport`` declarations.
    """

    UNINITIALIZED = 0
    NA = 1
    BOOL = 2
    UINT8 = 3
    INT8 = 4
    UINT16 = 5
    INT16 = 6
    UINT32 = 7
    INT32 = 8
    UINT64 = 9
    INT64 = 10
    HALF_FLOAT = 11
    FLOAT = 12
    DOUBLE = 13
    STRING = 14
    BINARY = 15
    FIXED_SIZE_BINARY = 16
    DATE32 = 17
    DATE64 = 18
    TIMESTAMP = 19
    TIME32 = 20
    TIME64 = 21
    INTERVAL_MONTHS = 22
    INTERVAL_DAY_TIME = 23
    DECIMAL128 = 24
    DECIMAL256 = 25
    LIST = 26
    STRUCT = 27
    SPARSE_UNION = 28
    DENSE_UNION = 29
    DICTIONARY = 30
    MAP = 31
    EXTENSION = 32
    FIXED_SIZE_LIST = 33
    DURATION = 34
    LARGE_STRING = 35
    LARGE_BINARY = 36
    LARGE_LIST = 37
    INTERVAL_MONTH_DAY_NANO = 38
    RUN_END_ENCODED = 39
    BINARY_VIEW = 40
    STRING_VIEW = 41

def has_time_unit(type_id):
    """Check if type_id represents a type with a time_unit parameter"""

def is_data_view(type_id):
    """Check if type_id is a binary view or string view type"""

def is_decimal(type_id):
    """Check if type_id is a decimal type"""

def is_fixed_size(type_id):
    """Check if type_id is a fixed-size (binary or list) type"""

def is_floating_point(type_id):
    """Check if type_id is a floating point type"""

def is_signed_integer(type_id):
    """Check if type_id is an signed integral type"""

def is_union(type_id):
    """Check if type_id is a union type"""

def is_unsigned_integer(type_id):
    """Check if type_id is an unsigned integral type"""
