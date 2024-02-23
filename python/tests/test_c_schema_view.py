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

from nanoarrow.c_lib import c_schema_view

import nanoarrow as na


def test_schema_view_accessors_basic():
    view = c_schema_view(na.Type.DATE32)
    assert view.type == "date32"
    assert view.type_id == na.Type.DATE32.value
    assert view.storage_type == "int32"
    assert view.storage_type_id == na.Type.INT32.value
    assert view.nullable is True
    assert view.map_keys_sorted is None
    assert view.fixed_size is None
    assert view.decimal_bitwidth is None
    assert view.decimal_precision is None
    assert view.decimal_scale is None
    assert view.time_unit_id is None
    assert view.time_unit is None
    assert view.timezone is None
    assert view.extension_name is None
    assert view.extension_metadata is None

    assert "date32" in repr(view)
    assert "fixed_size" not in repr(view)


def test_schema_view_accessors_fixed_size():
    view = c_schema_view(na.fixed_size_binary(123))
    assert view.fixed_size == 123


def test_schema_view_accessors_datetime():
    view = c_schema_view(na.timestamp("s", "America/Halifax"))
    assert view.timezone == "America/Halifax"
    assert view.time_unit_id == na.TimeUnit.SECOND.value
    assert view.time_unit == "s"


def test_schema_view_accessors_decimal():
    view = c_schema_view(na.decimal128(10, 3))
    assert view.decimal_bitwidth == 128
    assert view.decimal_precision == 10
    assert view.decimal_scale == 3


def test_schema_view_accessors_non_nullable():
    view = c_schema_view(na.int32(nullable=False))
    assert view.nullable is False


def test_schema_view_layout_accessors():
    view = c_schema_view(na.Type.INT32)
    assert view.layout.n_buffers == 2
    assert view.layout.buffer_data_type_id[0] == na.Type.BOOL.value
    assert view.layout.element_size_bits[0] == 1
    assert view.layout.buffer_data_type_id[1] == na.Type.INT32.value
    assert view.layout.element_size_bits[1] == 32
    assert view.layout.child_size_elements == 0
