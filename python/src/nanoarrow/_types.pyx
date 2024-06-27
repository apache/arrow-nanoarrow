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

from libc.stdio cimport snprintf
from nanoarrow_c cimport *

from struct import calcsize
from sys import byteorder as sys_byteorder


cdef equal(int type_id1, int type_id2):
    return type_id1 == type_id2


cdef one_of(int type_id, tuple type_ids):
    for item in type_ids:
        if type_id == <int>item:
            return True

    return False


cpdef bint is_unsigned_integer(int type_id):
    return type_id in (
        CArrowType.UINT8,
        CArrowType.UINT8,
        CArrowType.UINT16,
        CArrowType.UINT32,
        CArrowType.UINT64,
    )


cpdef bint is_signed_integer(int type_id):
    return type_id in (
        CArrowType.INT8,
        CArrowType.INT16,
        CArrowType.INT32,
        CArrowType.INT64,
    )


cpdef bint is_floating_point(int type_id):
    return type_id in (
        CArrowType.HALF_FLOAT,
        CArrowType.FLOAT,
        CArrowType.DOUBLE,
    )


cpdef bint is_fixed_size(int type_id):
    return type_id in (
        CArrowType.FIXED_SIZE_LIST,
        CArrowType.FIXED_SIZE_BINARY,
    )


cpdef bint is_decimal(int type_id):
    return type_id in (
        CArrowType.DECIMAL128,
        CArrowType.DECIMAL256,
    )


cpdef bint has_time_unit(int type_id):
    return type_id in (
        CArrowType.TIME32,
        CArrowType.TIME64,
        CArrowType.DURATION,
        CArrowType.TIMESTAMP,
    )


cpdef bint is_union(int type_id):
    return type_id in (
        CArrowType.DENSE_UNION,
        CArrowType.SPARSE_UNION,
    )


cdef tuple from_format(format):
    # PyBuffer_SizeFromFormat() was added in Python 3.9 (potentially faster)
    item_size = calcsize(format)

    # Don't allow non-native endian values
    if sys_byteorder == "little" and (">" in format or "!" in format):
        raise ValueError(f"Can't convert format '{format}' to Arrow type")
    elif sys_byteorder == "big" and  "<" in format:
        raise ValueError(f"Can't convert format '{format}' to Arrow type")

    # Strip system endian specifiers
    format = format.strip("=@")

    if format == "c":
        return 0, NANOARROW_TYPE_STRING
    elif format == "e":
        return item_size, NANOARROW_TYPE_HALF_FLOAT
    elif format == "f":
        return item_size, NANOARROW_TYPE_FLOAT
    elif format == "d":
        return item_size, NANOARROW_TYPE_DOUBLE

    # Check for signed integers
    if format in ("b", "h", "i", "l", "q", "n"):
        if item_size == 1:
            return item_size, NANOARROW_TYPE_INT8
        elif item_size == 2:
            return item_size, NANOARROW_TYPE_INT16
        elif item_size == 4:
            return item_size, NANOARROW_TYPE_INT32
        elif item_size == 8:
            return item_size, NANOARROW_TYPE_INT64

    # Check for unsinged integers
    if format in ("B", "?", "H", "I", "L", "Q", "N"):
        if item_size == 1:
            return item_size, NANOARROW_TYPE_UINT8
        elif item_size == 2:
            return item_size, NANOARROW_TYPE_UINT16
        elif item_size == 4:
            return item_size, NANOARROW_TYPE_UINT32
        elif item_size == 8:
            return item_size, NANOARROW_TYPE_UINT64

    # If all else fails, return opaque fixed-size binary
    return item_size, NANOARROW_TYPE_BINARY


cdef int to_format(ArrowType type_id, int element_size_bits, size_t out_size, char* out):
    if type_id in (NANOARROW_TYPE_BINARY, NANOARROW_TYPE_FIXED_SIZE_BINARY) and element_size_bits > 0:
        snprintf(out, out_size, "%ds", <int>(element_size_bits // 8))
        return element_size_bits

    cdef const char* format_const = ""
    cdef int element_size_bits_calc = 0
    if type_id == NANOARROW_TYPE_STRING:
        format_const = "c"
        element_size_bits_calc = 0
    elif type_id == NANOARROW_TYPE_BINARY:
        format_const = "B"
        element_size_bits_calc = 0
    elif type_id == NANOARROW_TYPE_BOOL:
        # Bitmaps export as unspecified binary
        format_const = "B"
        element_size_bits_calc = 1
    elif type_id == NANOARROW_TYPE_INT8:
        format_const = "b"
        element_size_bits_calc = 8
    elif type_id == NANOARROW_TYPE_UINT8:
        format_const = "B"
        element_size_bits_calc = 8
    elif type_id == NANOARROW_TYPE_INT16:
        format_const = "h"
        element_size_bits_calc = 16
    elif type_id == NANOARROW_TYPE_UINT16:
        format_const = "H"
        element_size_bits_calc = 16
    elif type_id in (NANOARROW_TYPE_INT32, NANOARROW_TYPE_INTERVAL_MONTHS):
        format_const = "i"
        element_size_bits_calc = 32
    elif type_id == NANOARROW_TYPE_UINT32:
        format_const = "I"
        element_size_bits_calc = 32
    elif type_id == NANOARROW_TYPE_INT64:
        format_const = "q"
        element_size_bits_calc = 64
    elif type_id == NANOARROW_TYPE_UINT64:
        format_const = "Q"
        element_size_bits_calc = 64
    elif type_id == NANOARROW_TYPE_HALF_FLOAT:
        format_const = "e"
        element_size_bits_calc = 16
    elif type_id == NANOARROW_TYPE_FLOAT:
        format_const = "f"
        element_size_bits_calc = 32
    elif type_id == NANOARROW_TYPE_DOUBLE:
        format_const = "d"
        element_size_bits_calc = 64
    elif type_id == NANOARROW_TYPE_INTERVAL_DAY_TIME:
        format_const = "ii"
        element_size_bits_calc = 64
    elif type_id == NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO:
        format_const = "iiq"
        element_size_bits_calc = 128
    elif type_id == NANOARROW_TYPE_DECIMAL128:
        format_const = "16s"
        element_size_bits_calc = 128
    elif type_id == NANOARROW_TYPE_DECIMAL256:
        format_const = "32s"
        element_size_bits_calc = 256
    else:
        raise ValueError(f"Unsupported Arrow type_id for format conversion: {type_id}")

    snprintf(out, out_size, "%s", format_const)
    return element_size_bits_calc
