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

"""Python bindings to the nanoarrow C library

EXPERIMNETAL

The nanoarrow Python package provides bindings to the nanoarrow C library. Like
the nanoarrow C library, it provides tools to facilitate the use of the
Arrow C Data and Arrow C Stream interfaces.
"""

from nanoarrow._utils import c_version
from nanoarrow.c_array import c_array_from_buffers, c_array
from nanoarrow.c_array_stream import c_array_stream
from nanoarrow.c_schema import c_schema
from nanoarrow.c_buffer import c_buffer
from nanoarrow.extension_canonical import bool8
from nanoarrow.schema import (
    Schema,
    Type,
    TimeUnit,
    null,
    bool_,
    int8,
    uint8,
    int16,
    uint16,
    int32,
    uint32,
    int64,
    uint64,
    float16,
    float32,
    float64,
    string,
    large_string,
    string_view,
    list_,
    large_list,
    fixed_size_list,
    map_,
    dictionary,
    binary,
    large_binary,
    binary_view,
    fixed_size_binary,
    date32,
    date64,
    time32,
    time64,
    timestamp,
    extension_type,
    duration,
    interval_months,
    interval_day_time,
    interval_month_day_nano,
    decimal128,
    decimal256,
    schema,
    struct,
)
from nanoarrow.array import array, Array
from nanoarrow.array_stream import ArrayStream
from nanoarrow.visitor import nulls_as_sentinel, nulls_forbid, nulls_separate
from nanoarrow._version import __version__  # noqa: F401

# Helps Sphinx automatically populate an API reference section
__all__ = [
    "ArrayStream",
    "Schema",
    "TimeUnit",
    "Type",
    "binary",
    "binary_view",
    "bool_",
    "bool8",
    "c_array",
    "c_array_from_buffers",
    "c_array_stream",
    "c_buffer",
    "c_schema",
    "c_version",
    "date32",
    "date64",
    "decimal128",
    "decimal256",
    "dictionary",
    "duration",
    "extension_type",
    "fixed_size_binary",
    "fixed_size_list",
    "float16",
    "float32",
    "float64",
    "int16",
    "int32",
    "int64",
    "int8",
    "interval_day_time",
    "interval_month_day_nano",
    "interval_months",
    "large_binary",
    "large_string",
    "large_list",
    "list_",
    "map_",
    "null",
    "nulls_as_sentinel",
    "nulls_forbid",
    "nulls_separate",
    "string",
    "string_view",
    "struct",
    "schema",
    "time32",
    "time64",
    "timestamp",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
    "Array",
    "array",
]
