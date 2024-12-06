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

from typing import Any, Iterator, Mapping, Optional

from nanoarrow.c_buffer import CBufferBuilder
from nanoarrow.c_schema import CSchema, c_schema_view
from nanoarrow.schema import extension_type, int8
from nanoarrow.visitor import ToPyBufferConverter

from nanoarrow import extension


def bool8(nullable: bool = True):
    """Create a type representing a boolean encoded as one byte per value

    Parameters
    ----------
    nullable : bool, optional
        Use ``False`` to mark this field as non-nullable.
    """

    return extension_type(int8(), "arrow.bool8", nullable=nullable)


class Bool8SequenceConverter(ToPyBufferConverter):
    def _make_builder(self):
        return CBufferBuilder().set_format("?")


@extension.register
class Bool8Extension(extension.Extension):
    def get_schema(self) -> CSchema:
        return bool8()

    def get_params(self, c_schema: CSchema) -> Mapping[str, Any]:
        schema_view = c_schema_view(c_schema)
        if schema_view.type != "int8":
            raise ValueError("arrow.bool8 must have storage type int8")

        return {}

    def get_pyiter(
        self,
        py_iterator,
        offset: int,
        length: int,
    ) -> Optional[Iterator[Optional[bool]]]:
        view = py_iterator._array_view
        items = map(bool, view.buffer(1).elements(offset, length))

        if py_iterator._contains_nulls():
            validity = view.buffer(0).elements(offset, length)
            return py_iterator._wrap_iter_nullable(validity, items)
        else:
            return items

    def get_sequence_converter(self, c_schema: CSchema):
        self.get_params(c_schema)
        return Bool8SequenceConverter

    def get_sequence_appender(self, c_schema: CSchema, array_builder):
        self.get_params(c_schema)
        return None

    def get_buffer_appender(self, c_schema: CSchema, array_builder):
        self.get_params(c_schema)
        return None

    def get_iterable_appender(self, c_schema: CSchema, array_builder):
        self.get_params(c_schema)
        return None
