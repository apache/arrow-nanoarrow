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

import ast
import importlib
from pathlib import Path

import pytest

SOURCE_DIR = Path(__file__).resolve().parents[1] / "src" / "nanoarrow"


def _public_declarations(body):
    names = set()
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
    return {name for name in names if not name.startswith("_")}


@pytest.mark.parametrize(
    "source", sorted(SOURCE_DIR.glob("*.pyx")), ids=lambda f: f.stem
)
def test_extension_stub_members(source):
    module = importlib.import_module(f"nanoarrow.{source.stem}")
    stub = source.with_suffix(".pyi")
    tree = ast.parse(stub.read_text(encoding="utf-8"), filename=str(stub))
    stub_names = _public_declarations(tree.body)
    runtime_names = {
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and getattr(value, "__module__", None) == module.__name__
    }

    # Ignore imported members when looking for omissions from the stub.
    assert runtime_names <= stub_names
    assert stub_names <= vars(module).keys()

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in runtime_names:
            members = {
                name
                for name in vars(getattr(module, node.name))
                if not name.startswith("_")
            }
            assert _public_declarations(node.body) == members, node.name
