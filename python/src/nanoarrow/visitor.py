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

from nanoarrow._lib import CArray, CArrayView, CSchema, CSchemaView
from nanoarrow.c_array_stream import c_array_stream
from nanoarrow.iterator import ArrayViewBaseIterator


class ArrayStreamVisitor:
    def __init__(self, iterator_cls=ArrayViewBaseIterator) -> None:
        self._iterator_cls = iterator_cls

    def visit(self, obj, schema=None):
        with c_array_stream(obj, schema=schema) as stream:
            iterator = self._iterator_cls(stream._get_cached_schema())
            state = self.visit_schema(iterator._schema, iterator._schema_view)

            iterator_set_array = iterator._set_array
            visit_array = self.visit_array
            array_view = iterator._array_view

            for array in stream:
                iterator_set_array(array)
                visit_array(array, array_view, iterator, state)

        return self.finish(state)

    def visit_schema(self, schema: CSchema, schema_view: CSchemaView):
        return None

    def visit_array(
        self,
        array: CArray,
        array_view: CArrayView,
        iterator: ArrayViewBaseIterator,
        state,
    ):
        pass

    def finish(self, state):
        return state
