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


def test_schema_create_struct():
    schema_obj = na.struct([na.Type.INT32])
    assert schema_obj.type == na.Type.STRUCT
    assert schema_obj.child(0).type == na.Type.INT32
    assert schema_obj.child(0).name is None

    schema_obj = na.struct([("col_name", na.Type.INT32)])
    assert schema_obj.type == na.Type.STRUCT
    assert schema_obj.child(0).type == na.Type.INT32
    assert schema_obj.child(0).name == "col_name"

    schema_obj = na.struct({"col_name": na.Type.INT32})
    assert schema_obj.type == na.Type.STRUCT
    assert schema_obj.child(0).type == na.Type.INT32
    assert schema_obj.child(0).name == "col_name"
