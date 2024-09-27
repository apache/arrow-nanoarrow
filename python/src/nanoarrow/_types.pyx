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
from . cimport _types

from struct import calcsize
from sys import byteorder as sys_byteorder


cdef equal(int type_id1, int type_id2):
    """Check if two type identifiers are equal

    Provided because Cython is picky about comparing enum types and
    will error for ``_types.UINT8 == NANOARROW_TYPE_UINT8``.
    """
    return type_id1 == type_id2


cdef one_of(int type_id, tuple type_ids):
    """Check if type_id is one of several values

    Provided because Cython is picky about comparing enum types and
    will error for ``_types.UINT8 in (NANOARROW_TYPE_UINT8,)``.
    """
    for item in type_ids:
        if type_id == <int>item:
            return True

    return False


cpdef bint is_unsigned_integer(int type_id):
    """Check if type_id is an unsigned integral type"""
    return type_id in (
        _types.UINT8,
        _types.UINT16,
        _types.UINT32,
        _types.UINT64,
    )


cpdef bint is_signed_integer(int type_id):
    """Check if type_id is an signed integral type"""
    return type_id in (
        _types.INT8,
        _types.INT16,
        _types.INT32,
        _types.INT64,
    )


cpdef bint is_floating_point(int type_id):
    """Check if type_id is a floating point type"""
    return type_id in (
        _types.HALF_FLOAT,
        _types.FLOAT,
        _types.DOUBLE,
    )


cpdef bint is_fixed_size(int type_id):
    """Check if type_id is a fixed-size (binary or list) type"""
    return type_id in (
        _types.FIXED_SIZE_LIST,
        _types.FIXED_SIZE_BINARY,
    )


cpdef bint is_decimal(int type_id):
    """Check if type_id is a decimal type"""
    return type_id in (
        _types.DECIMAL128,
        _types.DECIMAL256,
    )


cpdef bint has_time_unit(int type_id):
    """Check if type_id represents a type with a time_unit parameter"""
    return type_id in (
        _types.TIME32,
        _types.TIME64,
        _types.DURATION,
        _types.TIMESTAMP,
    )


cpdef bint is_union(int type_id):
    """Check if type_id is a union type"""
    return type_id in (
        _types.DENSE_UNION,
        _types.SPARSE_UNION,
    )


cpdef bint is_data_view(int type_id):
    """Check if type_id is a binary view or string view type"""
    return type_id in (
        _types.BINARY_VIEW,
        _types.STRING_VIEW
    )


cdef tuple from_format(format):
    """Convert a Python buffer protocol format string to a itemsize/type_id tuple

    Returns tuple of item size (in bytes) and the ``_types``. Raises
    ``ValueError`` if the given format string is cannot be represented
    (e.g., explicit non-system endian) but will return a fixed-size binary
    specification for unrecognized format strings. The BOOL type is
    converted as UINT8.
    """
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
        return 0, _types.STRING
    elif format == "e":
        return item_size, _types.HALF_FLOAT
    elif format == "f":
        return item_size, _types.FLOAT
    elif format == "d":
        return item_size, _types.DOUBLE

    # Check for signed integers
    if format in ("b", "h", "i", "l", "q", "n"):
        if item_size == 1:
            return item_size, _types.INT8
        elif item_size == 2:
            return item_size, _types.INT16
        elif item_size == 4:
            return item_size, _types.INT32
        elif item_size == 8:
            return item_size, _types.INT64

    # Check for unsinged integers
    if format in ("B", "?", "H", "I", "L", "Q", "N"):
        if item_size == 1:
            return item_size, _types.UINT8
        elif item_size == 2:
            return item_size, _types.UINT16
        elif item_size == 4:
            return item_size, _types.UINT32
        elif item_size == 8:
            return item_size, _types.UINT64

    # If all else fails, return opaque fixed-size binary
    return item_size, _types.BINARY


cdef int to_format(int type_id, int element_size_bits, size_t out_size, char* out):
    """Convert an Arrow type identifier to a Python buffer format string

    Populates a format string describing this type. The populated format string
    will usually roundtrip a buffer through the Python buffer protocol; however,
    BOOL exports as ``"B"`` (i.e., unsigned bytes) and fixed-size types with no
    Python equivalent (e.g., DECIMAL128) export as fixed-size binary. Packed types
    (e.g., INTERVAL_DAY_TIME) export as packed structs such that their component
    values are preserved.
    """
    if type_id in (_types.BINARY, _types.FIXED_SIZE_BINARY) and element_size_bits > 0:
        snprintf(out, out_size, "%ds", <int>(element_size_bits // 8))
        return element_size_bits

    cdef const char* format_const = ""
    cdef int element_size_bits_calc = 0
    if type_id == _types.STRING:
        format_const = "c"
        element_size_bits_calc = 0
    elif type_id == _types.BINARY:
        format_const = "B"
        element_size_bits_calc = 0
    elif type_id == _types.BOOL:
        # Bitmaps export as unspecified binary
        format_const = "B"
        element_size_bits_calc = 1
    elif type_id == _types.INT8:
        format_const = "b"
        element_size_bits_calc = 8
    elif type_id == _types.UINT8:
        format_const = "B"
        element_size_bits_calc = 8
    elif type_id == _types.INT16:
        format_const = "h"
        element_size_bits_calc = 16
    elif type_id == _types.UINT16:
        format_const = "H"
        element_size_bits_calc = 16
    elif type_id in (_types.INT32, _types.INTERVAL_MONTHS):
        format_const = "i"
        element_size_bits_calc = 32
    elif type_id == _types.UINT32:
        format_const = "I"
        element_size_bits_calc = 32
    elif type_id == _types.INT64:
        format_const = "q"
        element_size_bits_calc = 64
    elif type_id == _types.UINT64:
        format_const = "Q"
        element_size_bits_calc = 64
    elif type_id == _types.HALF_FLOAT:
        format_const = "e"
        element_size_bits_calc = 16
    elif type_id == _types.FLOAT:
        format_const = "f"
        element_size_bits_calc = 32
    elif type_id == _types.DOUBLE:
        format_const = "d"
        element_size_bits_calc = 64
    elif type_id == _types.INTERVAL_DAY_TIME:
        format_const = "ii"
        element_size_bits_calc = 64
    elif type_id == _types.INTERVAL_MONTH_DAY_NANO:
        format_const = "iiq"
        element_size_bits_calc = 128
    elif type_id == _types.DECIMAL128:
        format_const = "16s"
        element_size_bits_calc = 128
    elif type_id == _types.DECIMAL256:
        format_const = "32s"
        element_size_bits_calc = 256
    elif is_data_view(type_id):
        format_const = "16s"
        element_size_bits_calc = 128
    else:
        raise ValueError(f"Unsupported Arrow type_id for format conversion: {type_id}")

    snprintf(out, out_size, "%s", format_const)
    return element_size_bits_calc
