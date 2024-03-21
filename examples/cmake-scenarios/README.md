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

# CMake Comprehensive Examples

This folder contains a CMake project that showcases the many different
ways in which nanoarrow may be used by other projects. The project
exposes a few options for configuring how nanoarrow is found and
built. It shows how nanoarrow may be fetched from source during the
build via FetchContent as well as how it may be preinstalled on the
system and then found by the consumer.

To demonstrate the various scenarios, this project comes with a build
script that iterates through the different possibilities. To test this
out, simply run

```bash
./build.sh
```
