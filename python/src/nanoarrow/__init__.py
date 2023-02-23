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

from ._lib import (  # noqa: F401
    as_numpy_array,
    version,
    CSchemaHolder,
    CSchema,
)

class Schema(CSchema):

    def __init__(self, parent=None, addr=None) -> None:
        if parent is None:
            parent = CSchemaHolder()
        if addr is None:
            addr = parent._addr()
        super().__init__(parent, addr)

    @staticmethod
    def from_pyarrow(obj):
        schema = Schema()
        obj._export_to_c(schema._addr())
        return schema
