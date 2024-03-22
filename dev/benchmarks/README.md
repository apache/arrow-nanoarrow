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

# Benchmarking nanoarrow

This subdirectory contains benchmarks and tools to run them. This is currently
only implemented for the C library but may expand to include the R and Python
bindings. The structure is as follows:

- Benchmarks are documented inline using [Doxygen](https://www.doxygen.nl/).
- Configurations are CMake build presets, and CMake handles pulling a previous
  or local nanoarrow using `FetchContent`. Benchmarks are run using `ctest`.
- There is a bare-bones report written as a [Quarto](https://quarto.org)
  document that renders to markdown.

You can run benchmarks for a single configuration (e.g., `local`) with:

```shell
mkdir build && cd build
cmake .. --preset local
cmake --build .
ctest
```

The provided `benchmark-run-all.sh` creates (or reuses, if they are already
present) build directories in the form `build/<preset>` for each preset
and runs `ctest`.

You can build a full report by running:

```shell
python generate-fixtures.py # requires pyarrow
./benchmark-run-all.sh
cd apidoc && doxygen && cd ..
quarto render benchmark-report.qmd
```
