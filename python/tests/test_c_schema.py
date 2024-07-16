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
from nanoarrow.c_schema import allocate_c_schema, c_schema_view

import nanoarrow as na


def test_c_schema_basic():
    schema = allocate_c_schema()
    assert schema.is_valid() is False
    assert schema._to_string() == "[invalid: schema is released]"
    assert repr(schema) == "<nanoarrow.c_schema.CSchema <released>>"

    schema = na.c_schema(na.struct({"some_name": na.int32()}))

    assert schema.format == "+s"
    assert schema.flags == 2
    assert schema.metadata is None
    assert schema.n_children == 1
    assert len(list(schema.children)) == 1
    assert schema.child(0).format == "i"
    assert schema.child(0).name == "some_name"
    assert schema.child(0)._to_string() == "int32"
    assert "<nanoarrow.c_schema.CSchema int32>" in repr(schema)
    assert schema.dictionary is None

    with pytest.raises(IndexError):
        schema.child(1)


def test_c_schema_dictionary():
    pa = pytest.importorskip("pyarrow")

    schema = na.c_schema(pa.dictionary(pa.int32(), pa.utf8()))
    assert schema.format == "i"
    assert schema.dictionary.format == "u"
    assert "dictionary: <nanoarrow.c_schema.CSchema string" in repr(schema)


def test_schema_metadata():
    meta = {"key1": "value1", "key2": "value2"}
    schema = na.c_schema(na.int32()).modify(metadata=meta)

    assert len(schema.metadata) == 2

    meta2 = {k: v for k, v in schema.metadata.items()}
    assert list(meta2.keys()) == [b"key1", b"key2"]
    assert list(meta2.values()) == [b"value1", b"value2"]
    assert "b'key1': b'value1'" in repr(schema)


def test_c_schema_view():
    schema = allocate_c_schema()
    with pytest.raises(RuntimeError):
        c_schema_view(schema)

    schema = na.c_schema(na.int32())
    view = c_schema_view(schema)
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
    view = c_schema_view(na.fixed_size_binary(12))
    assert view.fixed_size == 12

    view = c_schema_view(na.decimal128(10, 3))
    assert view.decimal_bitwidth == 128
    assert view.decimal_precision == 10
    assert view.decimal_scale == 3

    view = c_schema_view(na.decimal256(10, 3))
    assert view.decimal_bitwidth == 256
    assert view.decimal_precision == 10
    assert view.decimal_scale == 3

    view = c_schema_view(na.duration("us"))
    assert view.time_unit == "us"

    view = c_schema_view(na.timestamp("us", "America/Halifax"))
    assert view.type == "timestamp"
    assert view.storage_type == "int64"
    assert view.time_unit == "us"
    assert view.timezone == "America/Halifax"

    pa = pytest.importorskip("pyarrow")

    view = c_schema_view(pa.list_(pa.int32(), 12))
    assert view.fixed_size == 12


def test_c_schema_metadata():
    meta = {
        b"ARROW:extension:name": b"some_name",
        b"ARROW:extension:metadata": b"some_metadata",
    }

    schema = na.c_schema(na.int32()).modify(metadata=meta)
    assert "b'some_name'" in repr(schema)
    assert "b'some_name'" in repr(schema.metadata)
    assert list(schema.metadata) == list(meta)
    assert list(schema.metadata.items()) == list(meta.items())
    assert list(schema.metadata.keys()) == list(meta.keys())
    assert list(schema.metadata.values()) == list(meta.values())

    view = c_schema_view(schema)
    assert view.extension_name == "some_name"
    assert view.extension_metadata == b"some_metadata"


def test_c_schema_equals():
    int32 = na.c_schema(na.int32())
    struct = na.c_schema(na.struct({"col1": na.int32()}))
    dictionary = na.c_schema(na.dictionary(na.int32(), na.string()))
    ordered_dictionary = na.c_schema(
        na.dictionary(na.int32(), na.string(), dictionary_ordered=True)
    )

    # Check schemas pointing to the same ArrowSchema
    assert int32.type_equals(int32)

    # Check equality with deep copies
    assert int32.type_equals(int32.__deepcopy__())
    assert struct.type_equals(struct.__deepcopy__())
    assert dictionary.type_equals(dictionary.__deepcopy__())

    # Check inequality because of format
    assert int32.type_equals(struct) is False

    # Check inequality because of nullability
    assert int32.type_equals(int32.modify(flags=0), check_nullability=True) is False
    # ...but not by default
    assert int32.type_equals(int32.modify(flags=0)) is True

    # Check inequality of type information encoded in flags
    assert dictionary.type_equals(ordered_dictionary) is False

    # Check inequality because of number of children
    assert struct.type_equals(struct.modify(children=[])) is False

    # Check inequality because of a difference in the children
    assert struct.type_equals(struct.modify(children=[dictionary])) is False

    # Check inequality because of dictionary presence
    assert int32.type_equals(dictionary) is False
    assert dictionary.type_equals(int32) is False

    # Check inequality because of dictionary index type
    assert (
        dictionary.type_equals(na.c_schema(na.dictionary(na.int64(), na.string())))
        is False
    )

    # Check inequality because of dictionary value type
    assert dictionary.type_equals(dictionary.modify(dictionary=struct)) is False


def test_c_schema_assert_type_equal():
    from nanoarrow._schema import assert_type_equal

    int32 = na.c_schema(na.int32())
    string = na.c_schema(na.string())
    nn_string = na.c_schema(na.string(False))

    assert_type_equal(int32, int32, check_nullability=True)

    with pytest.raises(TypeError):
        assert_type_equal(None, int32, check_nullability=False)

    with pytest.raises(TypeError):
        assert_type_equal(int32, None, check_nullability=False)

    msg = "Expected schema\n  'string'\nbut got\n  'int32'"
    with pytest.raises(ValueError, match=msg):
        assert_type_equal(int32, string, check_nullability=False)

    assert_type_equal(nn_string, string, check_nullability=False)
    with pytest.raises(ValueError):
        assert_type_equal(nn_string, string, check_nullability=True)


def test_c_schema_modify():
    schema = na.c_schema(na.null())

    schema_clone = schema.modify()
    assert schema_clone is not schema
    assert schema._addr() != schema_clone._addr()

    schema_formatted = schema.modify(format="i")
    assert schema_formatted.format == "i"

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
    assert list(schema_metad.metadata.items()) == [(b"some key", b"some value")]
    assert schema_non_nullable.format == schema.format

    schema_metad2 = schema.modify(metadata=schema_metad.metadata)
    assert list(schema_metad2.metadata.items()) == [(b"some key", b"some value")]

    schema_no_metad = schema_metad.modify(metadata={})
    assert schema_no_metad.metadata is None


def test_c_schema_modify_children():
    schema = na.c_schema(na.struct({"col1": na.null()}))

    schema_same_children = schema.modify()
    assert schema_same_children.n_children == 1
    assert schema_same_children.child(0).name == "col1"
    assert schema_same_children.child(0).format == "n"

    schema_new_children_list = schema.modify(
        children=[na.c_schema(na.int32()).modify(name="new name")]
    )
    assert schema_new_children_list.n_children == 1
    assert schema_new_children_list.child(0).name == "new name"
    assert schema_new_children_list.child(0).format == "i"

    schema_new_children_dict = schema.modify(
        children={"new name": na.c_schema(na.int32())}
    )
    assert schema_new_children_dict.n_children == 1
    assert schema_new_children_dict.child(0).name == "new name"
    assert schema_new_children_dict.child(0).format == "i"


def test_c_schema_modify_dictionary():
    schema = na.c_schema(na.int32())

    schema_dictionary = schema.modify(dictionary=na.c_schema(na.string()))
    assert schema_dictionary.format == "i"
    assert schema_dictionary.dictionary.format == "u"

    schema_same_dictionary = schema_dictionary.modify()
    assert schema_same_dictionary.format == "i"
    assert schema_same_dictionary.dictionary.format == "u"

    schema_no_dictionary = schema_dictionary.modify(dictionary=False)
    assert schema_no_dictionary.format == "i"
    assert schema.dictionary is None
