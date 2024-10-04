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

# Python Package Example

This folder contains a Python project that uses the nanoarrow
project in a C extension. To make building and packaging the library
as simple as possible, this example uses Meson as a build system alongside
the meson-python frontend. The pyproject.toml file lists these as build
dependencies, so you don't need to install them yourself (unless you set
the --no-build-isolation flag from your build frontend).

This library is completely self contained. To build it, simply

```bash
git clone https://github.com/apache/arrow-nanoarrow.git
cd arrow-nanoarrow/examples/python-package
python -m pip install .
```

The nanoarrow dependency is resolved through the Meson wrap system (see
subprojects/nanoarrow.wrap). When creating your own library, be sure to
create the subprojects directory up front and then run:

```bash
mkdir -p subprojects
meson wrap install nanoarrow
```

To install the wrap definition file.

For more control over the pip installation, you may want to set options like
the build directory or the type of build. For example, to get an editable
install of the library, while also building it in debug mode and storing the
build artifacts in the builddir directory, you should run:

```bash
python -m pip install -e . -Cbuilddir=builddir -Csetup-args="-Dbuildtype=debug"
```

Note that meson will generate a compilation database in the build directory
automatically. This can be particularly helpful for IDEs and code completion
if symlinked to the root of your Python package.

```bash
ln -s builddir/compile_commands.json .
```

The code contained herein makes use of the Arrow C Data interface and
Arrow PyCapsule interface to exchange data between the Python runtime and a
C extension. After installing the project, you can run it as follows:

```python
>>> import pyarrow as pa
>>> import schema_printer

>>> schema = pa.schema([("some int", pa.int32()), ("some_string", pa.string())])
>>> schema
some int: int32
some_string: string

>>> schema_printer.print_schema(schema)
Field: some int, Type: int32
Field: some_string, Type: string
```
