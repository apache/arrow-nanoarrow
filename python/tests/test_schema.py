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
    schema_obj = na.Schema(na.Type.INT32)
    assert schema_obj.type == na.Type.INT32

    schema_obj2 = na.Schema(schema_obj._c_schema)
    assert schema_obj2.type == schema_obj2.type
    assert schema_obj2._c_schema is schema_obj._c_schema


def test_schema_create_no_params():
    schema_obj = na.Schema(na.Type.INT32)
    assert schema_obj.type == na.Type.INT32

    with pytest.raises(ValueError, match=r"^Unused parameter"):
        na.Schema(na.Type.INT32, {"unused_param": "unused_value"})
