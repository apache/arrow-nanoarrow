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


def test_time_unit_create():
    assert na.TimeUnit.create("s") == na.TimeUnit.SECOND
    assert na.TimeUnit.create("ms") == na.TimeUnit.MILLI
    assert na.TimeUnit.create("us") == na.TimeUnit.MICRO
    assert na.TimeUnit.create("ns") == na.TimeUnit.NANO

    assert na.TimeUnit.create(na.TimeUnit.SECOND) == na.TimeUnit.SECOND


def test_schema_create_c_schema():
    schema_obj = na.int32()
    assert schema_obj.type == na.Type.INT32
    assert schema_obj.name == ""
    assert ("some key" in schema_obj.metadata) is False

    schema_obj2 = na.Schema(schema_obj._c_schema)
    assert schema_obj2.type == schema_obj2.type
    assert schema_obj2._c_schema is schema_obj._c_schema

    schema_obj_non_nullable = na.Schema(na.int32(), nullable=False)
    assert schema_obj_non_nullable.nullable is False

    schema_named = na.Schema(na.int32(), name="some_name")
    assert schema_named.name == "some_name"

    schema_metad = na.Schema(na.int32(), metadata={"some key": "some value"})
    assert b"some key" in schema_metad.metadata
    assert schema_metad.metadata[b"some key"] == b"some value"
    assert dict(schema_metad.metadata.items()) == {b"some key": b"some value"}

    with pytest.raises(ValueError):
        na.Schema(schema_obj._c_schema, some_parameter="some_value")


def test_schema_create_no_params():
    schema_obj = na.int32()
    assert schema_obj.type == na.Type.INT32
    assert schema_obj.nullable is True
    assert repr(schema_obj) == "<Schema> int32"

    schema_obj = na.int32(nullable=False)
    assert schema_obj.nullable is False
    assert "non-nullable" in repr(schema_obj)

    schema_obj = na.Schema(na.Type.INT32, name=False)
    assert schema_obj.name is None

    schema_obj = na.Schema(na.Type.INT32, name="not empty")
    assert schema_obj.name == "not empty"
    assert "'not empty': " in repr(schema_obj)

    msg = "params are only supported for obj of class Type"
    with pytest.raises(ValueError, match=msg):
        na.Schema(na.fixed_size_binary(123), byte_width=12)

    with pytest.raises(ValueError, match=r"^Unused parameter"):
        na.Schema(na.Type.INT32, unused_param="unused_value")


def test_schema_simple():
    assert na.null().type == na.Type.NULL
    assert na.bool_().type == na.Type.BOOL
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
    assert na.binary_view().type == na.Type.BINARY_VIEW
    assert na.string_view().type == na.Type.STRING_VIEW


def test_schema_fixed_size_binary():
    schema_obj = na.fixed_size_binary(byte_width=123)
    assert schema_obj.type == na.Type.FIXED_SIZE_BINARY
    assert schema_obj.byte_width == 123


def test_schema_time():
    schema_obj = na.time32(na.TimeUnit.SECOND)
    assert schema_obj.type == na.Type.TIME32
    assert schema_obj.unit == na.TimeUnit.SECOND

    schema_obj = na.time64(na.TimeUnit.MICRO)
    assert schema_obj.type == na.Type.TIME64
    assert schema_obj.unit == na.TimeUnit.MICRO


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


def test_schema_struct():
    # Make sure we can use just a list
    schema_obj = na.struct([na.Type.INT32])
    assert schema_obj.type == na.Type.STRUCT
    assert schema_obj.n_fields == 1
    assert schema_obj.field(0).type == na.Type.INT32
    assert schema_obj.field(0).name == ""
    for field in schema_obj.fields:
        assert isinstance(field, na.Schema)

    # Make sure we can use a dictionary to specify fields
    schema_obj = na.struct({"col_name": na.Type.INT32})
    assert schema_obj.type == na.Type.STRUCT
    assert schema_obj.field(0).type == na.Type.INT32
    assert schema_obj.field(0).name == "col_name"

    # Make sure we can use a Schema when constructing fields (and that
    # field names are taken from the input)
    schema_obj = na.struct([schema_obj.field(0)])
    assert schema_obj.type == na.Type.STRUCT
    assert schema_obj.field(0).type == na.Type.INT32
    assert schema_obj.field(0).name == "col_name"


def test_schema_list_():
    schema_obj = na.list_(na.null())
    assert schema_obj.type == na.Type.LIST
    assert schema_obj.value_type.type == na.Type.NULL


def test_schema_large_list():
    schema_obj = na.large_list(na.null())
    assert schema_obj.type == na.Type.LARGE_LIST
    assert schema_obj.value_type.type == na.Type.NULL


def test_schema_fixed_size_list():
    schema_obj = na.fixed_size_list(na.null(), 123)
    assert schema_obj.type == na.Type.FIXED_SIZE_LIST
    assert schema_obj.value_type.type == na.Type.NULL
    assert schema_obj.list_size == 123


def test_schema_dictionary():
    schema_obj = na.dictionary(na.int8(), na.null())
    assert schema_obj.type == na.Type.DICTIONARY
    assert schema_obj.index_type.type == na.Type.INT8
    assert schema_obj.value_type.type == na.Type.NULL
    assert schema_obj.dictionary_ordered is False

    schema_obj_ordered = na.dictionary(na.int8(), na.null(), dictionary_ordered=True)
    assert schema_obj_ordered.type == na.Type.DICTIONARY
    assert schema_obj_ordered.index_type.type == na.Type.INT8
    assert schema_obj_ordered.value_type.type == na.Type.NULL
    assert schema_obj_ordered.dictionary_ordered is True


def test_schema_extension():
    schema_obj = na.int32()
    assert schema_obj.extension is None

    # Check with metadata manually added
    schema_obj = na.Schema(
        na.int32(),
        metadata={
            "ARROW:extension:name": "arrow.test",
            "ARROW:extension:metadata": "abcdefg",
        },
    )
    assert schema_obj.extension.name == "arrow.test"
    assert schema_obj.extension.metadata == b"abcdefg"

    # Check from extension_type constructor
    schema_obj = na.extension_type(na.int32(), "arrow.test", "abcdefg")
    assert schema_obj.type == na.Type.EXTENSION
    assert schema_obj.extension is not None
    assert schema_obj.extension.name == "arrow.test"
    assert schema_obj.extension.metadata == b"abcdefg"
    assert schema_obj.extension.storage.type == na.Type.INT32
    assert schema_obj.nullable is True

    schema_obj = na.extension_type(na.int32(), "arrow.test", nullable=False)
    assert schema_obj.extension.name == "arrow.test"
    assert schema_obj.extension.metadata is None
    assert schema_obj.extension.storage.type == na.Type.INT32
    assert schema_obj.nullable is False


def test_schema_alias_constructor():
    schema = na.schema(na.Type.INT32)
    assert isinstance(schema, na.Schema)


def test_schema_serialize():
    import io

    schema = na.struct({"some_col": na.int32()}, nullable=False)

    serialized = schema.serialize()
    schema_roundtrip = na.ArrayStream.from_readable(serialized).schema
    assert repr(schema_roundtrip) == repr(schema)

    out = io.BytesIO()
    schema.serialize(out)
    schema_roundtrip = na.ArrayStream.from_readable(out.getvalue()).schema
    assert repr(schema_roundtrip) == repr(schema)


def test_schema_repr():
    schema = na.struct(
        {
            "col1": na.int32(),
            "col2": na.int16(),
            "col3": na.string(),
            "col4": na.timestamp(unit=na.TimeUnit.SECOND),
        },
        nullable=False,
    )

    assert repr(schema) == (
        "<Schema> non-nullable struct"
        "<col1: int32, col2: int16, col3: string, "
        "col4: timestamp('s', '')>"
    )
