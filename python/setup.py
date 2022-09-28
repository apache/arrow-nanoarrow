#!/usr/bin/env python

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

import shutil
from pathlib import Path

from setuptools import Extension, setup

import numpy as np


# setuptools gets confused by relative paths that extend above the project root
target = Path(__file__).parent / "src" / "nanoarrow"
shutil.copy(
    Path(__file__).parent / "../dist/nanoarrow.c", target / "nanoarrow.c"
)
shutil.copy(
    Path(__file__).parent / "../dist/nanoarrow.h", target / "nanoarrow.h"
)

setup(
    ext_modules=[
        Extension(
            name="nanoarrow._lib",
            include_dirs=[np.get_include(), "src/nanoarrow"],
            language="c++",
            sources=[
                "src/nanoarrow/_lib.pyx",
                "src/nanoarrow/nanoarrow.c",
            ],
        )
    ]
)
