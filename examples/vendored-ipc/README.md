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

# IPC Extension Vendored Example

This folder contains a project that uses the bundled nanarrow and nanoarrow_ipc.c
files included in the dist/ directory of this repository (or that can be generated
using `cmake -DNANOARROW_BUNDLE=ON` from the root CMake project). Like the CMake
example, you must be careful to not expose nanoarrow's headers outside your project
and make use of `#define NANOARROW_NAMESPACE MyProject` to prefix nanoarrow's symbol
names to ensure they do not collide with another copy of nanoarrow potentially linked
to by another project.

The nanoarrow/ files included in this example are stubs to illustrate
how these files could fit in to a library and/or command-line application project.
You can generate the bundled versions with the namespace defined using the Python
script `ci/scripts/bundle.py`:

```bash
git clone https://github.com/apache/arrow-nanoarrow.git
cd arrow-nanoarrow/examples/vendored-ipc

python ../../ci/scripts/bundle.py \
  --source-output-dir=src \
  --include-output-dir=src \
  --symbol-namespace=MyProject \
  --with-ipc \
  --with-flatcc
```

Then you can build/link the application/library using the build tool of your choosing:

```bash
cd src
cc -c library.c nanoarrow.c flatcc.c nanoarrow_ipc.c -I.
ar rcs libexample_vendored_ipc_library.a library.o nanoarrow.o nanoarrow_ipc.o flatcc.o
cc -o example_vendored_ipc_app app.c libexample_vendored_ipc_library.a
```

You can test the command-line application using the two files
provided in the example directory:

```bash
cat ../schema-valid.arrows | ./example_vendored_ipc_app
cat ../invalid.arrows | ./example_vendored_ipc_app
# Expected 0xFFFFFFFF at start of message but found 0xFFFFFF00
```
