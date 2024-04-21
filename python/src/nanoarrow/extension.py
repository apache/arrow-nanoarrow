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

from typing import Union

from nanoarrow._lib import CSchema


class Extension:
    def __init__(self, name: str, schema: CSchema, metadata=None) -> None:
        self._name = name

        if metadata is None:
            self._metadata = schema.metadata[b"ARROW:extension:metadata"]
        elif isinstance(metadata, str):
            self._metadata = metadata.encode()
        else:
            self._metadata = bytes(metadata)

    @property
    def name(self) -> str:
        return self._name

    @property
    def metadata(self) -> Union[bytes, None]:
        return self._metadata


def resolve_extension(schema: CSchema, extension_name: str) -> Extension:
    return Extension(extension_name, schema)
