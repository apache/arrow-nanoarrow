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

from typing import Any, Callable, Iterable

from nanoarrow._lib import CArrayView, CSchema, CSchemaView
from nanoarrow.c_array_stream import c_array_stream
from nanoarrow.iterator import ArrayViewBaseIterator, PyIterator


class ArrayStreamVisitor:
    @classmethod
    def visit(cls, obj, schema=None, **kwargs):
        visitor = cls(**kwargs)

        with c_array_stream(obj, schema=schema) as stream:
            iterator = visitor._iterator_cls(stream._get_cached_schema())
            state = visitor.visit_schema(iterator._schema, iterator._schema_view)

            iterator_set_array = iterator._set_array
            iterator_iter_chunk = iterator._iter_chunk
            visit_array = visitor.visit_array
            array_view = iterator._array_view

            for array in stream:
                iterator_set_array(array)
                state = visit_array(array_view, iterator_iter_chunk, state)

        return visitor.finish(state)

    def __init__(self, *, iterator_cls=ArrayViewBaseIterator) -> None:
        self._iterator_cls = iterator_cls

    def visit_schema(self, schema: CSchema, schema_view: CSchemaView):
        return None

    def visit_array(
        self,
        array_view: CArrayView,
        iterator: Callable[[int, int], Iterable],
        state: Any,
    ):
        return state

    def finish(self, state):
        return state


class ListBuilder(ArrayStreamVisitor):
    def __init__(self, *, iterator_cls=PyIterator) -> None:
        super().__init__(iterator_cls=iterator_cls)

    def visit_schema(self, schema: CSchema, schema_view: CSchemaView):
        return []

    def visit_array(
        self,
        array_view: CArrayView,
        iterator: Callable[[int, int], Iterable],
        state: Any,
    ):
        state.extend(iterator(0, array_view.length))
        return state


class NumpyObjectArrayBuilder(ListBuilder):
    def __init__(self, *, iterator_cls=PyIterator, n=None) -> None:
        super().__init__(iterator_cls=iterator_cls)
        self._n = n

    def visit_schema(self, schema: CSchema, schema_view: CSchemaView):
        from numpy import empty, fromiter

        array = empty(self._n, "O")
        return 0, array, array.dtype, fromiter

    def visit_array(
        self,
        array_view: CArrayView,
        iterator: Callable[[int, int], Iterable],
        state: Any,
    ):
        start, array, dtype, fromiter = state
        length = array_view.length
        end = start + length
        array[start:end] = fromiter(iterator(0, length), dtype, length)

        return end, array, dtype, fromiter

    def finish(self, state):
        return state[1]
