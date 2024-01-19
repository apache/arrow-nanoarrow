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

import enum
from typing import Union

from nanoarrow._lib import CArrowTimeUnit, CArrowType, CSchemaBuilder, CSchemaView
from nanoarrow.c_lib import c_schema


class Type(enum.Enum):
    """The Type enumerator provides a means by which the various type
    categories can be identified. Type values can be used in place of
    :class:`Schema` instances in most places for parameter-free types."""

    UNINITIALIZED = CArrowType.UNINITIALIZED
    NA = CArrowType.NA
    BOOL = CArrowType.BOOL
    UINT8 = CArrowType.UINT8
    INT8 = CArrowType.INT8
    UINT16 = CArrowType.UINT16
    INT16 = CArrowType.INT16
    UINT32 = CArrowType.UINT32
    INT32 = CArrowType.INT32
    UINT64 = CArrowType.UINT64
    INT64 = CArrowType.INT64
    HALF_FLOAT = CArrowType.HALF_FLOAT
    FLOAT = CArrowType.FLOAT
    DOUBLE = CArrowType.DOUBLE
    STRING = CArrowType.STRING
    BINARY = CArrowType.BINARY
    FIXED_SIZE_BINARY = CArrowType.FIXED_SIZE_BINARY
    DATE32 = CArrowType.DATE32
    DATE64 = CArrowType.DATE64
    TIMESTAMP = CArrowType.TIMESTAMP
    TIME32 = CArrowType.TIME32
    TIME64 = CArrowType.TIME64
    INTERVAL_MONTHS = CArrowType.INTERVAL_MONTHS
    INTERVAL_DAY_TIME = CArrowType.INTERVAL_DAY_TIME
    DECIMAL128 = CArrowType.DECIMAL128
    DECIMAL256 = CArrowType.DECIMAL256
    LIST = CArrowType.LIST
    STRUCT = CArrowType.STRUCT
    SPARSE_UNION = CArrowType.SPARSE_UNION
    DENSE_UNION = CArrowType.DENSE_UNION
    DICTIONARY = CArrowType.DICTIONARY
    MAP = CArrowType.MAP
    EXTENSION = CArrowType.EXTENSION
    FIXED_SIZE_LIST = CArrowType.FIXED_SIZE_LIST
    DURATION = CArrowType.DURATION
    LARGE_STRING = CArrowType.LARGE_STRING
    LARGE_BINARY = CArrowType.LARGE_BINARY
    LARGE_LIST = CArrowType.LARGE_LIST
    INTERVAL_MONTH_DAY_NANO = CArrowType.INTERVAL_MONTH_DAY_NANO

    def __arrow_c_schema__(self):
        # This will only work for parameter-free types
        c_schema = CSchemaBuilder.allocate().set_type(self.value).finish()
        return c_schema._capsule


class TimeUnit(enum.Enum):
    """Unit enumerator for timestamp, duration, and time types."""

    SECOND = CArrowTimeUnit.SECOND
    MILLI = CArrowTimeUnit.MILLI
    MICRO = CArrowTimeUnit.MICRO
    NANO = CArrowTimeUnit.NANO


class Schema:
    """The Schema is nanoarrow's data type representation, encompasing the role
    of PyArrow's ``Schema``, ``Field``, and ``DataType``. This scope maps to
    that of the ArrowSchema in the Arrow C Data interface."""

    def __init__(
        self,
        obj,
        *,
        nullable=None,
        **params,
    ) -> None:
        """Create Schema objects

        Create Schema objects from (1) a :class:`Type` or (1) any object supported by
        :func:`c_schema`. When creating a class from a  a :class:`Type`, some
        parameters may be required.

        >>> import nanoarrow as na
        >>> import pyarrow as pa
        >>> schema = na.Schema(na.Type.INT32)
        >>> schema = na.Schema(na.Type.DURATION, unit=na.TimeUnit.SECOND)
        >>> schema = na.Schema(pa.int32())
        """

        if isinstance(obj, Type):
            self._c_schema = _c_schema_from_type_and_params(obj, params, nullable)
        elif not params:
            self._c_schema = c_schema(obj)
        else:
            raise ValueError("params must be empty if type is not nanoarrow.Type")

        self._c_schema_view = CSchemaView(self._c_schema)

    @property
    def type(self) -> Type:
        """Type enumerator value of this Schema

        >>> import nanoarrow as na
        >>> na.int32().type
        <Type.INT32: 8>
        """
        return Type(self._c_schema_view.type_id)

    @property
    def name(self) -> Union[str, None]:
        """Field name of this Schema

        >>> import nanoarrow as na
        >>> schema = na.struct({"col1": na.int32()})
        >>> schema.child(0).name
        'col1'
        """
        return self._c_schema.name

    @property
    def nullable(self) -> bool:
        """Nullability of this field

        >>> import nanoarrow as na
        >>> na.int32().nullable
        True
        >>> na.int32(nullable=False).nullable
        False
        """
        return self._c_schema_view.nullable

    @property
    def byte_width(self) -> Union[int, None]:
        """Element byte width for fixed-size binary type

        Returns ``None`` for types for which this property is not relevant.

        >>> import nanoarrow as na
        >>> na.binary(123).byte_width
        123
        """

        if self._c_schema_view.type_id == CArrowType.FIXED_SIZE_BINARY:
            return self._c_schema_view.fixed_size

    @property
    def unit(self) -> Union[TimeUnit, None]:
        """TimeUnit for timestamp, time, and duration types

        Returns ``None`` for types for which this property is not relevant.

        >>> import nanoarrow as na
        >>> na.timestamp(na.TimeUnit.SECOND).unit
        <TimeUnit.SECOND: 0>
        """

        unit_id = self._c_schema_view.time_unit_id
        if unit_id is not None:
            return TimeUnit(unit_id)

    @property
    def timezone(self) -> Union[str, None]:
        """Timezone for timestamp types

        Returns ``None`` for types for which this property is not relevant or
        for timezone types for which the timezone is not set.

        >>> import nanoarrow as na
        >>> na.timestamp(na.TimeUnit.SECOND, timezone="America/Halifax").timezone
        'America/Halifax'
        """
        if self._c_schema_view.timezone:
            return self._c_schema_view.timezone

    @property
    def precision(self) -> int:
        """Decimal precision

        >>> import nanoarrow as na
        >>> na.decimal128(10, 3).precision
        10
        """
        return self._c_schema_view.decimal_precision

    @property
    def scale(self) -> int:
        """Decimal scale

        >>> import nanoarrow as na
        >>> na.decimal128(10, 3).scale
        3
        """

        return self._c_schema_view.decimal_scale

    @property
    def n_children(self) -> int:
        """Number of child Schemas

        >>> import nanoarrow as na
        >>> schema = na.struct({"col1": na.int32()})
        >>> schema.n_children
        1
        """

        return self._c_schema.n_children

    def child(self, i):
        """Extract a child Schema

        >>> import nanoarrow as na
        >>> schema = na.struct({"col1": na.int32()})
        >>> schema.child(0).type
        <Type.INT32: 8>
        """

        # Returning a copy to reduce interdependence between Schema instances
        return Schema(self._c_schema.child(i).__deepcopy__())

    @property
    def children(self):
        """Iterate over child Schemas
        >>> import nanoarrow as na
        >>> schema = na.struct({"col1": na.int32()})
        >>> for child in schema.children:
        ...     print(child.name)
        ...
        col1
        """
        for i in range(self.n_children):
            yield self.child(i)

    def __arrow_c_schema__(self):
        return self._c_schema.__arrow_c_schema__()


def int32(nullable=True) -> Schema:
    return Schema(Type.INT32, nullable=nullable)


def binary(byte_width=None, nullable=True) -> Schema:
    if byte_width is not None:
        return Schema(Type.FIXED_SIZE_BINARY, byte_width=byte_width, nullable=nullable)
    else:
        return Schema(Type.BINARY)


def timestamp(unit, timezone=None, nullable=True) -> Schema:
    return Schema(Type.TIMESTAMP, timezone=timezone, unit=unit, nullable=nullable)


def decimal128(precision: int, scale: int) -> Schema:
    return Schema(Type.DECIMAL128, precision=precision, scale=scale)


def decimal256(precision: int, scale: int) -> Schema:
    return Schema(Type.DECIMAL256, precision=precision, scale=scale)


def struct(fields, nullable=True) -> Schema:
    return Schema(Type.STRUCT, fields=fields, nullable=nullable)


def _c_schema_from_type_and_params(type: Type, params: dict, nullable: bool):
    factory = CSchemaBuilder.allocate()

    if type == Type.STRUCT:
        fields = _clean_fields(params.pop("fields"))

        factory.set_format("+s").allocate_children(len(fields))
        for i, item in enumerate(fields):
            name, c_schema = item
            factory.set_child(i, name, c_schema)

    elif type.value in CSchemaView._decimal_types:
        precision = int(params.pop("precision"))
        scale = int(params.pop("scale"))
        factory.set_type_decimal(type.value, precision, scale)

    elif type.value in CSchemaView._time_unit_types:
        time_unit = params.pop("unit")
        if "timezone" in params:
            timezone = params.pop("timezone")
        else:
            timezone = None

        factory.set_type_date_time(type.value, TimeUnit(time_unit).value, timezone)

    elif type == Type.FIXED_SIZE_BINARY:
        factory.set_type_fixed_size(type.value, int(params.pop("byte_width")))

    else:
        factory.set_type(type.value)

    if params:
        unused = ", ".join(f"'{item}'" for item in params.keys())
        raise ValueError(f"Unused parameters whilst constructing Schema: {unused}")

    factory.set_nullable(nullable)
    return factory.finish()


def _clean_fields(fields):
    if isinstance(fields, dict):
        return [(str(k), c_schema(v)) for k, v in fields.items()]
    else:
        fields_clean = []
        for item in fields:
            if isinstance(item, tuple) and len(item) == 2:
                fields_clean.append((str(item[0]), c_schema(item[1])))
            else:
                fields_clean.append((None, c_schema(item)))

        return fields_clean
