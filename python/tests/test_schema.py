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

import pytest

import nanoarrow as na


def test_type_schema_protocol():
    c_schema = na.c_schema(na.Type.INT32)
    assert c_schema.format == "i"


def test_schema_create_c_schema():
    schema_obj = na.int32()
    assert schema_obj.type == na.Type.INT32

    schema_obj2 = na.Schema(schema_obj._c_schema)
    assert schema_obj2.type == schema_obj2.type
    assert schema_obj2._c_schema is schema_obj._c_schema

    with pytest.raises(ValueError, match="must be unspecified"):
        na.Schema(schema_obj._c_schema, some_parameter="some_value")

    with pytest.raises(ValueError, match="must be unspecified"):
        na.Schema(schema_obj._c_schema, nullable=True)

    with pytest.raises(ValueError, match="must be unspecified"):
        na.Schema(schema_obj._c_schema, name="")


def test_schema_create_no_params():
    schema_obj = na.int32()
    assert schema_obj.type == na.Type.INT32
    assert schema_obj.nullable is True

    schema_obj = na.int32(nullable=False)
    assert schema_obj.nullable is False

    with pytest.raises(ValueError, match=r"^Unused parameter"):
        na.Schema(na.Type.INT32, unused_param="unused_value")


def test_schema_simple():
    assert na.null().type == na.Type.NULL
    assert na.bool().type == na.Type.BOOL
    assert na.int8().type == na.Type.INT8
    assert na.uint8().type == na.Type.UINT8
    assert na.int16().type == na.Type.INT16
    assert na.uint16().type == na.Type.UINT16
    assert na.int32().type == na.Type.INT32
    assert na.uint32().type == na.Type.UINT32
    assert na.int64().type == na.Type.INT64
    assert na.uint64().type == na.Type.UINT64
    assert na.float16().type == na.Type.HALF_FLOAT
    assert na.float32().type == na.Type.FLOAT
    assert na.float64().type == na.Type.DOUBLE
    assert na.string().type == na.Type.STRING
    assert na.large_string().type == na.Type.LARGE_STRING
    assert na.binary().type == na.Type.BINARY
    assert na.large_binary().type == na.Type.LARGE_BINARY
    assert na.date32().type == na.Type.DATE32
    assert na.date64().type == na.Type.DATE64
    assert na.interval_months().type == na.Type.INTERVAL_MONTHS
    assert na.interval_day_time().type == na.Type.INTERVAL_DAY_TIME
    assert na.interval_month_day_nano().type == na.Type.INTERVAL_MONTH_DAY_NANO


def test_schema_binary():
    schema_obj = na.binary(byte_width=123)
    assert schema_obj.type == na.Type.FIXED_SIZE_BINARY
    assert schema_obj.byte_width == 123

    schema_obj = na.binary()
    assert schema_obj.type == na.Type.BINARY


def test_schema_time():
    schema_obj = na.time32(na.TimeUnit.SECOND)
    assert schema_obj.type == na.Type.TIME32
    assert schema_obj.unit == na.TimeUnit.SECOND

    schema_obj = na.time64(na.TimeUnit.SECOND)
    assert schema_obj.type == na.Type.TIME32
    assert schema_obj.unit == na.TimeUnit.SECOND


def test_schema_timestamp():
    schema_obj = na.timestamp(na.TimeUnit.SECOND)
    assert schema_obj.type == na.Type.TIMESTAMP
    assert schema_obj.unit == na.TimeUnit.SECOND
    assert schema_obj.timezone is None

    schema_obj = na.timestamp(na.TimeUnit.SECOND, timezone="America/Halifax")
    assert schema_obj.timezone == "America/Halifax"


def test_schema_duration():
    schema_obj = na.duration(na.TimeUnit.SECOND)
    assert schema_obj.type == na.Type.DURATION
    assert schema_obj.unit == na.TimeUnit.SECOND


def test_schema_decimal():
    schema_obj = na.decimal128(10, 3)
    assert schema_obj.type == na.Type.DECIMAL128
    assert schema_obj.precision == 10
    assert schema_obj.scale == 3

    schema_obj = na.decimal256(10, 3)
    assert schema_obj.type == na.Type.DECIMAL256
    assert schema_obj.precision == 10
    assert schema_obj.scale == 3


def test_schema_create_struct():
    # Make sure we can use just a list
    schema_obj = na.struct([na.Type.INT32])
    assert schema_obj.type == na.Type.STRUCT
    assert schema_obj.child(0).type == na.Type.INT32
    assert schema_obj.child(0).name == ""

    # Make sure we can use a list of two-tuples
    schema_obj = na.struct([("col_name", na.Type.INT32)])
    assert schema_obj.type == na.Type.STRUCT
    assert schema_obj.child(0).type == na.Type.INT32
    assert schema_obj.child(0).name == "col_name"

    # Make sure we can use a dictionary to specify fields
    schema_obj = na.struct({"col_name": na.Type.INT32})
    assert schema_obj.type == na.Type.STRUCT
    assert schema_obj.child(0).type == na.Type.INT32
    assert schema_obj.child(0).name == "col_name"

    # Make sure we can use a Schema when constructing fields (and that
    # fild names are taken from the input)
    schema_obj = na.struct([schema_obj.child(0)])
    assert schema_obj.type == na.Type.STRUCT
    assert schema_obj.child(0).type == na.Type.INT32
    assert schema_obj.child(0).name == "col_name"
