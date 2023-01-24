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

# IPC Extension CMake Example

This folder contains a CMake project that links to its own copy of
nanoarrow and nanoarrow_ipc using CMake's `FetchContent` module.
This pattern is similar to the cmake-minimal example and includes
both a library and a command-line application that can verify
a small message read from stdin. To build:

```bash
git clone https://github.com/apache/arrow-nanoarrow.git
cd arrow-nanoarrow/examples/cmake-ipc
mkdir build && cd build
cmake ..
cmake --build .
```

You can test the command-line application using the two files
provided in the example directory:

```bash
cat ../schema-valid.arrows | ./example_cmake_ipc_app
cat ../invalid.arrows | ./example_cmake_ipc_app
# Expected 0xFFFFFFFF at start of message but found 0xFFFFFF00
```
