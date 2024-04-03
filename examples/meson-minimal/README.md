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

# Minimal Meson Example

This folder contains a meson project that links to its own copy of
nanoarrow using a subproject.

To build the project:

```bash
git clone https://github.com/apache/arrow-nanoarrow.git
cd arrow-nanoarrow/examples/meson-minimal
meson setup builddir && cd builddir
meson compile
```

After building, you can run the app from the build directory. The app
parses command line arguments into an int32 array and prints out the
resulting length (or any error encountered whilst building the array).

```bash
./example_meson_minimal_app
# 1
# 2
# 3
```
