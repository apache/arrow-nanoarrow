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

from nanoarrow._lib import c_version
from nanoarrow.c_lib import (
    c_schema,
    c_array,
    c_array_from_buffers,
    c_array_stream,
    c_schema_view,
    c_array_view,
    c_buffer,
    allocate_c_schema,
    allocate_c_array,
    allocate_c_array_stream,
)
from nanoarrow.schema import (
    Schema,
    Type,
    TimeUnit,
    schema,
    null,
    bool,
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
    binary,
    large_binary,
    fixed_size_binary,
    date32,
    date64,
    time32,
    time64,
    timestamp,
    duration,
    interval_months,
    interval_day_time,
    interval_month_day_nano,
    decimal128,
    decimal256,
    struct,
)
from nanoarrow.array import Array
from nanoarrow._version import __version__  # noqa: F401

# Helps Sphinx automatically populate an API reference section
__all__ = [
    "Schema",
    "TimeUnit",
    "Type",
    "allocate_c_array",
    "allocate_c_array_stream",
    "allocate_c_schema",
    "binary",
    "bool",
    "c_array",
    "c_array_from_buffers",
    "c_array_stream",
    "c_array_view",
    "c_buffer",
    "c_lib",
    "c_schema",
    "c_schema_view",
    "c_version",
    "date32",
    "date64",
    "decimal128",
    "decimal256",
    "duration",
    "fixed_size_binary",
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
    "null",
    "schema",
    "string",
    "struct",
    "time32",
    "time64",
    "timestamp",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
    "Array",
]
