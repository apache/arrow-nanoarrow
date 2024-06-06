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

import os
import tempfile
import pathlib

import pytest

import bundle


def test_read_write_content():
    assert bundle.read_content("one two three") == "one two three"

    with tempfile.TemporaryDirectory() as td:
        file1 = os.path.join(td, "test1.txt")
        bundle.write_content("One\nTwo\nThree\n", file1)
        assert bundle.read_content(pathlib.Path(file1)) == "One\nTwo\nThree\n"


def test_configure_file():
    replaced = bundle.configure_content(
        "@One@\n@Two@\n@Three@\n", {"Three": "Three replaced"}
    )
    assert replaced == "@One@\n@Two@\nThree replaced\n"

    with pytest.raises(ValueError, match="Expected exactly one occurrence"):
        bundle.configure_content("empty", {"Four": "Four replaced"})


def test_concatenate_files():
    assert bundle.concatenate_content([]) == ""

    concatenated = bundle.concatenate_content(
        ["One\nTwo\nThree\n", "Four\nFive\nSix\n"]
    )
    assert concatenated == "One\nTwo\nThree\nFour\nFive\nSix\n"


def test_cmakelist_version():
    extracted = bundle.cmakelist_version('set(NANOARROW_VERSION "0.1.2-SNAPSHOT")')
    assert extracted == ("0.1.2-SNAPSHOT", 0, 1, 2)

    this_cmakelists = pathlib.Path(__file__).parent.parent.parent / "CMakeLists.txt"
    assert bundle.cmakelist_version(this_cmakelists) == ("0.6.0-SNAPSHOT", 0, 6, 0)
