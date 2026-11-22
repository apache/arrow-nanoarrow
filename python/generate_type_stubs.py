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

import inspect
import sys
from enum import Enum, IntEnum
from pathlib import Path

from mypy.stubdoc import ArgSig, FunctionSig
from mypy.stubgenc import InspectionStubGenerator
from mypy.util import quote_docstring


def _is_cython_function(value):
    return type(value).__name__ == "cython_function_or_method"


def _annotation(value):
    return None if value is inspect.Signature.empty else inspect.formatannotation(value)


class CythonStubGenerator(InspectionStubGenerator):
    """Use Python signature metadata for Cython functions exposed by stubgen."""

    def is_skipped_attribute(self, name):
        # These implementation details can reference the Cython build version.
        return name in {
            "__pyx_capi__",
            "__pyx_vtable__",
            "__reduce_cython__",
            "__setstate_cython__",
            "__test__",
        } or super().is_skipped_attribute(name)

    def is_function(self, value):
        return _is_cython_function(value) or super().is_function(value)

    def generate_class_stub(self, class_name, cls, output, parent_class=None):
        if not issubclass(cls, Enum):
            return super().generate_class_stub(class_name, cls, output, parent_class)

        # Enum members are assignments in stubs, and Cython can add private bases.
        self.record_name(class_name)
        base = self.add_name(
            "enum.IntEnum" if issubclass(cls, IntEnum) else "enum.Enum"
        )
        output.append(f"{self._indent}class {class_name}({base}):")
        if cls.__doc__:
            doc = quote_docstring(self._indent_docstring(cls.__doc__))
            output.append(f"{self._indent}    {doc}")
        output.extend(
            f"{self._indent}    {name} = {member.value!r}"
            for name, member in cls.__members__.items()
        )

    def get_default_function_sig(self, func, ctx):
        if not _is_cython_function(func):
            return super().get_default_function_sig(func, ctx)
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):
            return super().get_default_function_sig(func, ctx)

        args = []
        previous = None
        for param in sig.parameters.values():
            kind = param.kind
            if previous is param.POSITIONAL_ONLY and kind is not previous:
                args.append(ArgSig("/"))
            if kind is param.KEYWORD_ONLY and previous not in (
                param.KEYWORD_ONLY,
                param.VAR_POSITIONAL,
            ):
                args.append(ArgSig("*"))

            prefix = {param.VAR_POSITIONAL: "*", param.VAR_KEYWORD: "**"}.get(kind, "")
            args.append(
                ArgSig(
                    prefix + param.name,
                    _annotation(param.annotation),
                    default=param.default is not param.empty,
                )
            )
            previous = kind

        if previous is inspect.Parameter.POSITIONAL_ONLY:
            args.append(ArgSig("/"))
        return FunctionSig(ctx.name, args, _annotation(sig.return_annotation))


def main(output_dir, modules):
    for module in modules:
        generator = CythonStubGenerator(module, modules, include_docstrings=True)
        # C imports are not visible to stubgen's Python import inspection.
        if module == "nanoarrow._array_stream":
            generator.known_imports["nanoarrow._array"] = ["CArray"]
            generator.known_imports["nanoarrow._schema"] = ["CSchema"]
        generator.generate_module()
        path = output_dir / f"{module.rsplit('.', 1)[1]}.pyi"
        path.write_text(generator.output(), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
