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

from nanoarrow_c cimport *

cdef class CArrowType:

    UNINITIALIZED = NANOARROW_TYPE_UNINITIALIZED
    NA = NANOARROW_TYPE_NA
    BOOL = NANOARROW_TYPE_BOOL
    UINT8 = NANOARROW_TYPE_UINT8
    INT8 = NANOARROW_TYPE_INT8
    UINT16 = NANOARROW_TYPE_UINT16
    INT16 = NANOARROW_TYPE_INT16
    UINT32 = NANOARROW_TYPE_UINT32
    INT32 = NANOARROW_TYPE_INT32
    UINT64 = NANOARROW_TYPE_UINT64
    INT64 = NANOARROW_TYPE_INT64
    HALF_FLOAT = NANOARROW_TYPE_HALF_FLOAT
    FLOAT = NANOARROW_TYPE_FLOAT
    DOUBLE = NANOARROW_TYPE_DOUBLE
    STRING = NANOARROW_TYPE_STRING
    BINARY = NANOARROW_TYPE_BINARY
    FIXED_SIZE_BINARY = NANOARROW_TYPE_FIXED_SIZE_BINARY
    DATE32 = NANOARROW_TYPE_DATE32
    DATE64 = NANOARROW_TYPE_DATE64
    TIMESTAMP = NANOARROW_TYPE_TIMESTAMP
    TIME32 = NANOARROW_TYPE_TIME32
    TIME64 = NANOARROW_TYPE_TIME64
    INTERVAL_MONTHS = NANOARROW_TYPE_INTERVAL_MONTHS
    INTERVAL_DAY_TIME = NANOARROW_TYPE_INTERVAL_DAY_TIME
    DECIMAL128 = NANOARROW_TYPE_DECIMAL128
    DECIMAL256 = NANOARROW_TYPE_DECIMAL256
    LIST = NANOARROW_TYPE_LIST
    STRUCT = NANOARROW_TYPE_STRUCT
    SPARSE_UNION = NANOARROW_TYPE_SPARSE_UNION
    DENSE_UNION = NANOARROW_TYPE_DENSE_UNION
    DICTIONARY = NANOARROW_TYPE_DICTIONARY
    MAP = NANOARROW_TYPE_MAP
    EXTENSION = NANOARROW_TYPE_EXTENSION
    FIXED_SIZE_LIST = NANOARROW_TYPE_FIXED_SIZE_LIST
    DURATION = NANOARROW_TYPE_DURATION
    LARGE_STRING = NANOARROW_TYPE_LARGE_STRING
    LARGE_BINARY = NANOARROW_TYPE_LARGE_BINARY
    LARGE_LIST = NANOARROW_TYPE_LARGE_LIST
    INTERVAL_MONTH_DAY_NANO = NANOARROW_TYPE_INTERVAL_MONTH_DAY_NANO

    @staticmethod
    cdef ArrowType uninitialized():
        return NANOARROW_TYPE_UNINITIALIZED
