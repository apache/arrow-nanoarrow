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
    """Define a nanoarrow Extension

    A nanoarrow extension customizes behaviour of built-in operations
    applicable to a specific type. This is currently implemented only
    for Arrow extension types but could in theory apply if one wanted
    to customize the conversion behaviour of a specific non-extension
    type.

    This is currently internal and involves knowledge of other internal
    nanoarrow/Python structures. It is currently used only to implement
    canonical extensions with the anticipation of evolving to support
    user-defined extensions as the internal APIs on which it relies
    stabilize.

    With the current design, an Extension subclass must be constructible
    with no parameters (e.g., ``Extension()``).
    """

    def get_schema(self) -> CSchema:
        """Get the schema for which this extension applies.

        This is used by :func:`register_extension` to ensure that it can be resolved
        when needed.
        """
        raise NotImplementedError()

    def get_params(self, c_schema: CSchema) -> Mapping[str, Any]:
        """Compute a dictionary of type parameters.

        These parameters are accessible via the :class:`Schema`
        ``extension`` attribute (e.g., ``schema.extension.param_name``).
        Internal parameters can also be returned but should be prefixed with
        an underscore.

        This method should also error if the storage type or any other property
        of the schema is not valid.
        """
        return {}

    def get_pyiter(
        self,
        py_iterator,
        offset: int,
        length: int,
    ) -> Optional[Iterator[Optional[bool]]]:
        """Compute an iterable of Python objects.

        Used by ``to_pylist()`` to generate scalars for a particular type.
        If ``None`` is returned, the behaviour of the storage type will be
        used without warning.

        This method is currently passed the underlying :class:`PyIterator`
        and returns an iterator; however, it could in the future be passed
        a :class:`CSchema` and return a PyIterator class once that class
        structure is stabilized.
        """
        name = py_iterator._schema_view.extension_name
        raise NotImplementedError(f"Extension get_pyiter() for {name}")

    def get_sequence_converter(self, c_schema: CSchema):
        """Return an ArrayViewVisitor subclass used to compute a sequence from
        a stream of arrays.

        This is currently implemented outside the null handler and may need a flag
        at some point to indicate that it did or did not handle its own nulls.
        """
        schema_view = c_schema_view(c_schema)
        name = schema_view.extension_name
        raise NotImplementedError(f"Extension get_sequence_converter() for {name}")

    def get_buffer_appender(
        self, c_schema: CSchema, array_builder
    ) -> Optional[Callable[[Any], None]]:
        """Compute a function that prepares a :class:`CArrayBuilder` from a
        buffer.

        This is used to customize the behavior of creating a CArray from an
        object implementing the Python buffer protocol. If ``None`` is
        returned, the storage will be converted without a warning.

        This method is currently passed a :class:`CArrayBuilder` but in
        the future should perhaps be passed a :class:`CSchema` and return a
        CArrayBuilder class.
        """
        schema_view = c_schema_view(c_schema)
        name = schema_view.extension_name
        raise NotImplementedError(f"Extension get_buffer_appender() for {name}")

    def get_iterable_appender(
        self, c_schema: CSchema, array_builder
    ) -> Optional[Callable[[Iterable], None]]:
        """Compute a function that prepares a :class:`CArrayBuilder` from a
        buffer.

        This is used to customize the behavior of creating a CArray from an
        iterable of Python objects.

        This method is currently passed a :class:`CArrayBuilder` but in
        the future should perhaps be passed a :class:`CSchema` and return a
        CArrayBuilder class.
        """
        schema_view = c_schema_view(c_schema)
        name = schema_view.extension_name
        raise NotImplementedError(f"Extension get_iterable_appender() for {name}")


_global_extension_registry = {}


def resolve_extension(c_schema_view: CSchemaView) -> Optional[Extension]:
    """Resolve an extension instance from a :class:`CSchemaView`

    Returns the registered extension instance if one applies to the passed
    type or ``None`` otherwise.
    """
    extension_name = c_schema_view.extension_name
    if extension_name in _global_extension_registry:
        return _global_extension_registry[extension_name]

    return None


def register_extension(extension: Extension) -> Optional[Extension]:
    """Register an :class:`Extension` instance in the global registry.

    Inserts an extension into the global registry, returning the
    previously registered extension for that type if one exists
    (or ``None`` otherwise).
    """
    global _global_extension_registry

    schema_view = c_schema_view(extension.get_schema())
    key = schema_view.extension_name
    prev = resolve_extension(schema_view)
    _global_extension_registry[key] = extension
    return prev


def unregister_extension(extension_name: str):
    """Remove an extension from the global registry by extension name.

    Returns the removed extension. Raises ``KeyError`` if there was no
    extension registered for this extension name.
    """
    prev = _global_extension_registry[extension_name]
    del _global_extension_registry[extension_name]
    return prev


def register(extension_cls: Type[Extension]):
    """Decorator that registers an extension class by instantiating it
    and adding it to the global registry."""
    register_extension(extension_cls())
    return extension_cls
