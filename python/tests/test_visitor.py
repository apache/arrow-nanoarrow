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
import pytest

import nanoarrow as na
from nanoarrow import visitor


def test_to_pylist():
    assert visitor.to_pylist([1, 2, 3], na.int32()) == [1, 2, 3]


def test_to_columms():
    array = na.c_array_from_buffers(
        na.struct({"col1": na.int32(), "col2": na.bool_(), "col3": na.string()}),
        length=3,
        buffers=[None],
        children=[
            na.c_array([1, 2, 3], na.int32()),
            na.c_array([1, 0, 1], na.bool_()),
            na.c_array(["abc", "def", "ghi"], na.string()),
        ],
    )

    names, columns = visitor.to_columns(array)
    assert names == ["col1", "col2", "col3"]
    assert columns[0] == [1, 2, 3]
    assert columns[1] == [True, False, True]
    assert columns[2] == ["abc", "def", "ghi"]

    with pytest.raises(ValueError, match="can only be used on a struct array"):
        visitor.to_columns([], na.int32())
