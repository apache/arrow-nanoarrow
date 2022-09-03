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

# Minimal CMake Example

This folder contains a CMake project that links to its own copy of
nanoarrow using CMake's `FetchContent` module. Whether vendoring or
using CMake, nanoarrow is intended to be vendored or statically
linked in a way that does not expose its headers or symbols to other
projects. To illustrate this, a small library is included (library.h
and library.c) and built in this way, linked to by a program (app.c)
that does not use nanoarrow (but does make use of the Arrow C Data
interface header, since this is ABI stable and intended to be used
in this way).

To build the project:

```bash
git clone https://github.com/apache/arrow-nanoarrow.git
cd arrow-nanoarrow/examples/cmake-minimal
mkdir build && cd build
cmake ..
cmake --build .
```

You can also open the cmake-minimal folder in VSCode and configure/build
the project using VSCode's CMake integration.

After building, you can run the app from the build directory. The app
parses command line arguments into an int32 array and prints out the
resulting length (or any error encountered whilst building the array).

```bash
./example_cmake_minimal_app
# 1
# 2
# 3
```
