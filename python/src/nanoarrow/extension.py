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

import warnings
from typing import Iterable, Mapping, Type, Union

from nanoarrow._lib import CArrayView, CSchema


class UnregisteredExtensionWarning(UserWarning):
    pass


class Extension:
    @classmethod
    def deserialize(cls, name: str, schema: CSchema):
        raise NotImplementedError()

    @property
    def name(self) -> str:
        raise NotImplementedError()

    @property
    def metadata(self) -> Union[bytes, None]:
        raise NotImplementedError()


class SimpleExtension:
    def __init__(self, name: str, metadata=None) -> None:
        self._name = name

        if metadata is None:
            self._metadata = None
        elif isinstance(metadata, str):
            self._metadata = metadata.encode()
        else:
            self._metadata = bytes(metadata)

    @classmethod
    def deserialize(cls, name: str, schema: CSchema):
        name = None
        metadata = None

        for key, value in schema.metadata:
            if key == b"ARROW:extension:name":
                name = value.decode()
            elif key == b"ARROW:extension:metadata":
                metadata = value

        if name is None:
            raise ValueError("schema is not an Arrow extension type")

        return cls(name, metadata)

    @property
    def name(self) -> str:
        return self._name

    @property
    def metadata(self) -> Union[bytes, None]:
        return self._metadata

    def _iter_py(self, c_array_view: CArrayView) -> Union[None, Iterable]:
        warnings.warn(
            f"Converting unregistered extension '{self.name} as storage type",
            UnregisteredExtensionWarning,
        )

        return None


class ExtensionRegistry:
    def __init__(self) -> None:
        self._extensions = {}

    def __iter__(self) -> Iterable[str]:
        return iter(self._extensions.keys())

    def __contains__(self, k: str) -> bool:
        return k in self._extensions

    def __getitem__(self, k: str) -> Type[Extension]:
        return self._extensions[k]

    def __setitem__(self, k, v: Type[Extension]) -> None:
        self._extensions[k] = v


_global_extension_registry = ExtensionRegistry()


def global_extension_registry() -> Mapping[str, Type[Extension]]:
    global _global_extension_registry
    return _global_extension_registry


def resolve_extension(
    extension_name: str, schema: CSchema, default: Union[Type[Extension], None] = None
) -> Type[Extension]:
    registry = global_extension_registry()
    if extension_name in registry:
        return registry[extension_name]
    elif default is not None:
        return default
    else:
        raise KeyError(f"Extension '{extension_name}' is not registered")
