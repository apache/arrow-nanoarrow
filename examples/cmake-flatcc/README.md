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

# CMake External flatcc Examples

This example builds nanoarrow IPC against external flatcc runtimes and runs a
small IPC consumer. It checks static and shared runtimes, static preference when
both are available, Debug and Release discovery, root and separate-directory
options, and configuration failure when the runtime is missing. It exercises
build-time linking; package exports are covered separately by `cmake-scenarios`.

Run from this directory with Bash, CMake 3.18 or newer, and a C compiler:

```bash
./build.sh
```

The script can also be invoked from any working directory. It builds in a
temporary directory and removes it on exit. On Windows, run it with Git Bash;
the combined static/shared layout is skipped because static and import libraries
use the same filename.
