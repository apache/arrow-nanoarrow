<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# nanoarrow for Python

Python bindings for nanoarrow. These are in a preliminary state: see open issues
and tests/test_nanoarrow.py for usage.

## Installation

Python bindings for nanoarrow are not yet available on PyPI. You can install via
URL (requires a C compiler):

```bash
python -m pip install "https://github.com/apache/arrow-nanoarrow/archive/refs/heads/main.zip#egg=nanoarrow&subdirectory=python"
```

## Building

Python bindings for nanoarrow are managed with setuptools[setuptools]. This means you
can build the project using:

```shell
git clone https://github.com/apache/arrow-nanoarrow.git
cd python
pip install -e .
```

Tests use [pytest][pytest]:

```shell
# Install dependencies
pip install -e .[test]

# Run tests
pytest -vvx
```

[pytest]: https://docs.pytest.org/
[setuptools]: https://setuptools.pypa.io/en/latest/index.html
