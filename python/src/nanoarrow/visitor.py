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

from typing import Any, List, Mapping, Sequence, Union

from nanoarrow._lib import CArrayView
from nanoarrow.c_array_stream import c_array_stream
from nanoarrow.iterator import ArrayViewBaseIterator, PyIterator


def to_pylist(obj, schema=None) -> List:
    return ListBuilder.visit(obj, schema)


def to_columns(obj, schema=None) -> Mapping[str, Sequence]:
    return ColumnsBuilder.visit(obj, schema)


class ArrayStreamVisitor(ArrayViewBaseIterator):
    @classmethod
    def visit(cls, obj, schema=None, **kwargs):
        if hasattr(obj, "__len__"):
            total_elements = len(obj)
        else:
            total_elements = None

        with c_array_stream(obj, schema=schema) as stream:
            visitor = cls(stream._get_cached_schema(), **kwargs)
            state = visitor.begin(total_elements)

            iterator_set_array = visitor._set_array
            visit_chunk_view = visitor.visit_chunk_view
            array_view = visitor._array_view

            for array in stream:
                iterator_set_array(array)
                state = visit_chunk_view(array_view, state)

        return visitor.finish(state)

    def begin(self, total_elements: Union[int, None] = None):
        return None

    def visit_chunk_view(self, array_view: CArrayView, state: Any) -> Any:
        return state

    def finish(self, state: Any) -> Any:
        return state


class ListBuilder(ArrayStreamVisitor):
    def __init__(self, schema, *, iterator_cls=PyIterator, _array_view=None):
        super().__init__(schema, _array_view=_array_view)
        self._iterator = iterator_cls(schema, _array_view=self._array_view)

    def begin(self, total_elements: Union[int, None] = None):
        return self._iterator._iter_chunk, []

    def visit_chunk_view(self, array_view: CArrayView, state: Any):
        iter_chunk, out = state
        out.extend(iter_chunk(0, array_view.length))
        return iter_chunk, out

    def finish(self, state: Any):
        return state[1]


class ColumnsBuilder(ArrayStreamVisitor):
    def __init__(self, schema, *, _array_view=None):
        super().__init__(schema, _array_view=_array_view)

        self._child_visitors = []
        self._child_array_views = []
        for child_schema, child_array_view in zip(
            self._schema.children, self._array_view.children
        ):
            self._child_visitors.append(
                self._resolve_child_visitor(child_schema, child_array_view)
            )
            self._child_array_views.append(child_array_view)

    def _resolve_child_visitor(self, child_schema, child_array_view):
        # TODO: Resolve more efficient builders for single-buffer types
        return ListBuilder(child_schema, _array_view=child_array_view)

    def begin(self, total_elements: Union[int, None] = None):
        child_visitors = self._child_visitors
        child_state = [child.begin(total_elements) for child in child_visitors]
        return child_visitors, child_state, self._child_array_views

    def visit_chunk_view(self, array_view: CArrayView, state: Any) -> Any:
        child_visitors, child_state, child_array_views = state

        for i in range(len(child_visitors)):
            child_state[i] = child_visitors[i].visit_chunk_view(
                child_array_views[i], child_state[i]
            )

        return state

    def finish(self, state: Any) -> Any:
        child_visitors, child_state, _ = state

        out = {}
        for i, schema in enumerate(self._schema.children):
            out[schema.name] = child_visitors[i].finish(child_state[i])

        return out
