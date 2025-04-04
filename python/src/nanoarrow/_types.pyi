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

import _cython_3_0_11
import enum
from typing import Callable, ClassVar

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
__pyx_capi__: dict
__test__: dict
has_time_unit: _cython_3_0_11.cython_function_or_method
is_data_view: _cython_3_0_11.cython_function_or_method
is_decimal: _cython_3_0_11.cython_function_or_method
is_fixed_size: _cython_3_0_11.cython_function_or_method
is_floating_point: _cython_3_0_11.cython_function_or_method
is_signed_integer: _cython_3_0_11.cython_function_or_method
is_union: _cython_3_0_11.cython_function_or_method
is_unsigned_integer: _cython_3_0_11.cython_function_or_method
sys_byteorder: str

class CArrowType(enum.IntFlag):
    __new__: ClassVar[Callable] = ...
    BINARY: ClassVar[CArrowType] = ...
    BINARY_VIEW: ClassVar[CArrowType] = ...
    BOOL: ClassVar[CArrowType] = ...
    DATE32: ClassVar[CArrowType] = ...
    DATE64: ClassVar[CArrowType] = ...
    DECIMAL128: ClassVar[CArrowType] = ...
    DECIMAL256: ClassVar[CArrowType] = ...
    DENSE_UNION: ClassVar[CArrowType] = ...
    DICTIONARY: ClassVar[CArrowType] = ...
    DOUBLE: ClassVar[CArrowType] = ...
    DURATION: ClassVar[CArrowType] = ...
    EXTENSION: ClassVar[CArrowType] = ...
    FIXED_SIZE_BINARY: ClassVar[CArrowType] = ...
    FIXED_SIZE_LIST: ClassVar[CArrowType] = ...
    FLOAT: ClassVar[CArrowType] = ...
    HALF_FLOAT: ClassVar[CArrowType] = ...
    INT16: ClassVar[CArrowType] = ...
    INT32: ClassVar[CArrowType] = ...
    INT64: ClassVar[CArrowType] = ...
    INT8: ClassVar[CArrowType] = ...
    INTERVAL_DAY_TIME: ClassVar[CArrowType] = ...
    INTERVAL_MONTHS: ClassVar[CArrowType] = ...
    INTERVAL_MONTH_DAY_NANO: ClassVar[CArrowType] = ...
    LARGE_BINARY: ClassVar[CArrowType] = ...
    LARGE_LIST: ClassVar[CArrowType] = ...
    LARGE_STRING: ClassVar[CArrowType] = ...
    LIST: ClassVar[CArrowType] = ...
    MAP: ClassVar[CArrowType] = ...
    NA: ClassVar[CArrowType] = ...
    RUN_END_ENCODED: ClassVar[CArrowType] = ...
    SPARSE_UNION: ClassVar[CArrowType] = ...
    STRING: ClassVar[CArrowType] = ...
    STRING_VIEW: ClassVar[CArrowType] = ...
    STRUCT: ClassVar[CArrowType] = ...
    TIME32: ClassVar[CArrowType] = ...
    TIME64: ClassVar[CArrowType] = ...
    TIMESTAMP: ClassVar[CArrowType] = ...
    UINT16: ClassVar[CArrowType] = ...
    UINT32: ClassVar[CArrowType] = ...
    UINT64: ClassVar[CArrowType] = ...
    UINT8: ClassVar[CArrowType] = ...
    UNINITIALIZED: ClassVar[CArrowType] = ...
    _all_bits_: ClassVar[int] = ...
    _boundary_: ClassVar[enum.FlagBoundary] = ...
    _flag_mask_: ClassVar[int] = ...
    _generate_next_value_: ClassVar[Callable] = ...
    _inverted_: ClassVar[None] = ...
    _member_map_: ClassVar[dict] = ...
    _member_names_: ClassVar[list] = ...
    _member_type_: ClassVar[type[int]] = ...
    _singles_mask_: ClassVar[int] = ...
    _unhashable_values_: ClassVar[list] = ...
    _use_args_: ClassVar[bool] = ...
    _value2member_map_: ClassVar[dict] = ...
    __and__: ClassVar[Callable] = ...
    __invert__: ClassVar[Callable] = ...
    __or__: ClassVar[Callable] = ...
    __rand__: ClassVar[Callable] = ...
    __ror__: ClassVar[Callable] = ...
    __rxor__: ClassVar[Callable] = ...
    __xor__: ClassVar[Callable] = ...
    def __format__(self, *args, **kwargs) -> str:
        """Convert to a string according to format_spec."""
