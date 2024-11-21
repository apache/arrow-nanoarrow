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

from typing import Any, Iterator, Mapping, Optional, Type

from nanoarrow.c_array import CArrayView
from nanoarrow.c_schema import CSchema, CSchemaView, c_schema_view


class Extension:
    def get_schema(self) -> CSchema:
        raise NotImplementedError()

    def get_params(self, c_schema: CSchema) -> Mapping[str, Any]:
        return {}

    def get_pyiter(
        self,
        params: Mapping[str, Any],
        c_array_view: CArrayView,
        offset: int,
        length: int,
    ) -> Optional[Iterator]:
        return None

    def get_sequence_converter(self, c_schema: CSchema):
        return None


global_extension_registry = {}


def resolve_extension(c_schema_view: CSchemaView) -> Optional[Extension]:
    extension_name = c_schema_view.extension_name
    if extension_name in global_extension_registry:
        return global_extension_registry[extension_name]

    type_id = c_schema_view.type_id
    if type_id in global_extension_registry:
        return global_extension_registry[type_id]

    return None


def register_extension(extension: Extension) -> Optional[Extension]:
    global global_extension_registry

    schema_view = c_schema_view(extension.get_schema())
    if schema_view.extension_name:
        key = schema_view.extension_name
    else:
        key = schema_view.type_id

    prev = global_extension_registry[key] if key in global_extension_registry else None
    global_extension_registry[key] = extension
    return prev


def register(extension_cls: Type[Extension]):
    register_extension(extension_cls())
    return extension_cls
