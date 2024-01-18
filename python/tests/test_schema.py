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

    with pytest.raises(ValueError, match="params must be empty"):
        na.Schema(schema_obj._c_schema, some_parameter="some_value")


def test_schema_create_no_params():
    schema_obj = na.int32()
    assert schema_obj.type == na.Type.INT32
    assert schema_obj.nullable is True

    schema_obj = na.int32(nullable=False)
    assert schema_obj.nullable is False

    with pytest.raises(ValueError, match=r"^Unused parameter"):
        na.Schema(na.Type.INT32, unused_param="unused_value")


def test_schema_binary():
    schema_obj = na.binary(byte_width=123)
    assert schema_obj.type == na.Type.FIXED_SIZE_BINARY
    assert schema_obj.byte_width == 123

    schema_obj = na.binary()
    assert schema_obj.type == na.Type.BINARY


def test_schema_timestamp():
    schema_obj = na.timestamp(na.TimeUnit.SECOND)
    assert schema_obj.type == na.Type.TIMESTAMP
    assert schema_obj.unit == na.TimeUnit.SECOND
    assert schema_obj.timezone is None

    schema_obj = na.timestamp(na.TimeUnit.SECOND, timezone="America/Halifax")
    assert schema_obj.timezone == "America/Halifax"


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
    assert schema_obj.child(0).name is None

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
