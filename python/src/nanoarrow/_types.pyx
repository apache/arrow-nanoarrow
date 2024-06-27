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
