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

import os
import sys
import subprocess
from setuptools import Extension, setup
import numpy as np

# Run bootstrap.py to run cmake generating a fresh bundle based on this
# checkout or copy from ../dist if the caller doesn't have cmake available
this_dir = os.path.dirname(__file__)
subprocess.run([sys.executable, os.path.join(this_dir, 'bootstrap.py')])

setup(
    ext_modules=[
        Extension(
            name="nanoarrow._lib",
            include_dirs=[np.get_include(), "src/nanoarrow"],
            language="c",
            sources=[
                "src/nanoarrow/_lib.pyx",
                "src/nanoarrow/nanoarrow.c",
            ],
        )
    ]
)
