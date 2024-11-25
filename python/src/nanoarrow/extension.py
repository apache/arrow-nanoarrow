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

from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Type

from nanoarrow.c_schema import CSchema, CSchemaView, c_schema_view


class Extension:
    def get_schema(self) -> CSchema:
        raise NotImplementedError()

    def get_params(self, c_schema: CSchema) -> Mapping[str, Any]:
        return {}

    def get_pyiter(
        self,
        py_iterator,
        offset: int,
        length: int,
    ) -> Optional[Iterator[Optional[bool]]]:
        name = py_iterator._schema_view.extension_name
        raise NotImplementedError(f"Extension get_pyiter() for {name}")

    def get_sequence_converter(self, c_schema: CSchema):
        schema_view = c_schema_view(c_schema)
        name = schema_view.extension_name
        raise NotImplementedError(f"Extension get_sequence_converter() for {name}")

    def get_buffer_appender(
        self, c_schema: CSchema, array_builder
    ) -> Optional[Callable[[Iterable], None]]:
        schema_view = c_schema_view(c_schema)
        name = schema_view.extension_name
        raise NotImplementedError(f"Extension get_buffer_appender() for {name}")

    def get_iterable_appender(
        self, c_schema: CSchema, array_builder
    ) -> Optional[Callable[[Iterable], None]]:
        schema_view = c_schema_view(c_schema)
        name = schema_view.extension_name
        raise NotImplementedError(f"Extension get_iterable_appender() for {name}")


_global_extension_registry = {}


def resolve_extension(c_schema_view: CSchemaView) -> Optional[Extension]:
    extension_name = c_schema_view.extension_name
    if extension_name in _global_extension_registry:
        return _global_extension_registry[extension_name]

    return None


def register_extension(extension: Extension) -> Optional[Extension]:
    global _global_extension_registry

    schema_view = c_schema_view(extension.get_schema())
    key = schema_view.extension_name
    prev = (
        _global_extension_registry[key] if key in _global_extension_registry else None
    )
    _global_extension_registry[key] = extension
    return prev


def unregister_extension(extension_name: str):
    prev = _global_extension_registry[extension_name]
    del _global_extension_registry[extension_name]
    return prev


def register(extension_cls: Type[Extension]):
    register_extension(extension_cls())
    return extension_cls
