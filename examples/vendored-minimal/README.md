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

# Minimal Vendored Example

This folder contains a project that uses the bundled nanarrow.c and nanoarrow.h
files included in the dist/ directory of this repository (or that can be generated
using `cmake -DNANOARROW_BUNDLE=ON` from the root CMake project). Like the CMake
example, you must be careful to not expose nanoarrow's header outside your project
and make use of `#define NANOARROW_NAMESPACE MyProject` to prefix nanoarrow's symbol
names to ensure they do not collide with another copy of nanoarrow potentially
linked to by another project.

The nanoarrow.c and nanoarrow.h files included in this example are stubs to illustrate
how these files could fit in to a library and/or command-line application project.
The easiest way is to use the pre-generated versions in the dist/ folder of this
repository:

```bash
git clone https://github.com/apache/arrow-nanoarrow.git
cd arrow-nanoarrow/examples/vendored-minimal
cp ../../dist/nanoarrow.h src/nanoarrow.h
cp ../../dist/nanoarrow.c src/nanoarrow.c
```

If you use these, you will have to manually `#define NANOARROW_NAMESPACE MyProject`
manually next to `#define NANOARROW_BUILD_ID` in the header.

You can also generate the bundled versions with the namespace defined using `cmake`:

```bash
git clone https://github.com/apache/arrow-nanoarrow.git
cd arrow-nanoarrow
mkdir build && cd build
cmake .. -DNANOARROW_BUNDLE=ON -DNANOARROW_NAMESPACE=ExampleVendored
cmake --build .
cmake --install . --prefix=../examples/vendored-minimal/src
```

Then you can build/link the application/library using the build tool of your choosing:

```bash
cd src
cc -c library.c nanoarrow.c
ar rcs libexample_vendored_minimal_library.a library.o nanoarrow.o
cc -o example_vendored_minimal_app app.c libexample_vendored_minimal_library.a
```

After building, you can run the app. The app
parses command line arguments into an int32 array and prints out the
resulting length (or any error encountered whilst building the array).

```bash
./example_vendored_minimal_app
# 1
# 2
# 3
```
