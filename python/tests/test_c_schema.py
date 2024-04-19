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


def test_c_schema_basic():
    schema = na.allocate_c_schema()
    assert schema.is_valid() is False
    assert schema._to_string() == "[invalid: schema is released]"
    assert repr(schema) == "<nanoarrow.c_lib.CSchema <released>>"

    schema = na.c_schema(na.struct({"some_name": na.int32()}))

    assert schema.format == "+s"
    assert schema.flags == 2
    assert schema.metadata is None
    assert schema.n_children == 1
    assert len(list(schema.children)) == 1
    assert schema.child(0).format == "i"
    assert schema.child(0).name == "some_name"
    assert schema.child(0)._to_string() == "int32"
    assert "<nanoarrow.c_lib.CSchema int32>" in repr(schema)
    assert schema.dictionary is None

    with pytest.raises(IndexError):
        schema.child(1)


def test_c_schema_dictionary():
    pa = pytest.importorskip("pyarrow")

    schema = na.c_schema(pa.dictionary(pa.int32(), pa.utf8()))
    assert schema.format == "i"
    assert schema.dictionary.format == "u"
    assert "dictionary: <nanoarrow.c_lib.CSchema string" in repr(schema)


def test_schema_metadata():
    meta = {"key1": "value1", "key2": "value2"}
    schema = na.c_schema(na.int32()).modify(metadata=meta)

    assert len(schema.metadata) == 2

    meta2 = {k: v for k, v in schema.metadata}
    assert list(meta2.keys()) == [b"key1", b"key2"]
    assert list(meta2.values()) == [b"value1", b"value2"]
    assert "b'key1': b'value1'" in repr(schema)


def test_c_schema_view():
    schema = na.allocate_c_schema()
    with pytest.raises(RuntimeError):
        na.c_schema_view(schema)

    schema = na.c_schema(na.int32())
    view = na.c_schema_view(schema)
    assert "- type: 'int32'" in repr(view)
    assert view.type == "int32"
    assert view.storage_type == "int32"

    assert view.fixed_size is None
    assert view.decimal_bitwidth is None
    assert view.decimal_scale is None
    assert view.time_unit is None
    assert view.timezone is None
    assert view.union_type_ids is None
    assert view.extension_name is None
    assert view.extension_metadata is None


def test_c_schema_view_extra_params():
    view = na.c_schema_view(na.fixed_size_binary(12))
    assert view.fixed_size == 12

    view = na.c_schema_view(na.decimal128(10, 3))
    assert view.decimal_bitwidth == 128
    assert view.decimal_precision == 10
    assert view.decimal_scale == 3

    view = na.c_schema_view(na.decimal256(10, 3))
    assert view.decimal_bitwidth == 256
    assert view.decimal_precision == 10
    assert view.decimal_scale == 3

    view = na.c_schema_view(na.duration("us"))
    assert view.time_unit == "us"

    view = na.c_schema_view(na.timestamp("us", "America/Halifax"))
    assert view.type == "timestamp"
    assert view.storage_type == "int64"
    assert view.time_unit == "us"
    assert view.timezone == "America/Halifax"

    pa = pytest.importorskip("pyarrow")

    view = na.c_schema_view(pa.list_(pa.int32(), 12))
    assert view.fixed_size == 12


def test_c_schema_metadata():
    meta = {
        "ARROW:extension:name": "some_name",
        "ARROW:extension:metadata": "some_metadata",
    }

    schema = na.c_schema(na.int32()).modify(metadata=meta)
    view = na.c_schema_view(schema)
    assert view.extension_name == "some_name"
    assert view.extension_metadata == b"some_metadata"


def test_c_schema_modify():
    schema = na.c_schema(na.null())

    schema_clone = schema.modify()
    assert schema_clone is not schema
    assert schema._addr() != schema_clone._addr()

    schema_named = schema.modify(name="something else")
    assert schema_named.name == "something else"
    assert schema_named.format == schema.format

    schema_flagged = schema.modify(flags=0)
    assert schema_flagged.flags == 0
    assert schema_flagged.format == schema.format

    schema_non_nullable = schema.modify(nullable=False)
    assert schema_non_nullable.flags == 0
    assert schema_non_nullable.format == schema.format

    meta = {"some key": "some value"}
    schema_metad = schema.modify(metadata=meta)
    assert list(schema_metad.metadata) == [(b"some key", b"some value")]
    assert schema_non_nullable.format == schema.format

    schema_metad2 = schema.modify(metadata=schema_metad.metadata)
    assert list(schema_metad2.metadata) == [(b"some key", b"some value")]

    schema_no_metad = schema_metad.modify(metadata={})
    assert schema_no_metad.metadata is None
