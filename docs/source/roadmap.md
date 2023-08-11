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

# Roadmap

Apache Arrow nanoarrow is a relatively new library and is under active development.
Maintaining the balance between useful and minimal is difficult to do; however,
there are a number of features that fit comforably within the scope of nanoarrow that
have not yet been scheduled for implementation.

## C library

- **Type coverage**: The C library currently provides support for all types that
  are available via the Arrow C Data interface. When the recently-added run-end
  encoded (REE) types and potentially forthcoming string view/list view types
  are available via the Arrow C Data interface, support should be added in
  nanoarrow as well.
- **Array append**: The `ArrowArrayAppend*()` family of functions provide a means
  by which to incrementally build arrays; however, there is no built-in way to
  append an `ArrowArrayView`, potentially more efficiently appending multiple
  values at once. Among other things, this would provide a route to an
  unoptimized filter/take implementation.
- **Remove Arrow C++ dependency for tests**: The C library and IPC extension rely
  on Arrow C++ for some test code that was written early in the library's development.
  These tests are valuable to ensure compatibility between nanoarrow and Arrow C++;
  however, including them in the default test suite complicates release verification
  for some users and prevents testing in environments where Arrow C++ does not
  currently build (e.g., WASM, compilers without C++17 support).
- **C++ integration**: The existing C++ integration is intentionally minimal;
  however, there are likely improvements that could be made to better integrate
  nanoarrow into existing C++ projects.
- **Documentation**: As the C library and its user base evolves, documentation
  needs to be refined and expanded to support the current set of use cases.

## IPC extension

- **Write support**: The IPC extension currently provides support for reading
  IPC streams but not writing them.
- **Dictionary support**: The IPC extension does not currently support reading
  dictionary messages an IPC stream.
- **Compression**: The IPC extension does not currently support compressed streams.

## Device extension

This entire extension is currently experimental and awaiting use-cases that will
drive future development.

## R bindings

- **Type support**: The R bindings currently do not provide support for extension
  types and relies on Arrow C++ for some dictionary-encoded types.
- **ALTREP support**: A recent R release added enhanced ALTREP support such that
  types that convert to `list()` can defer materialization cost/allocation.
  Arrow sources that arrive in chunks (e.g., from a `Table` or `ChunkedArray`)
  currently can't be converted via any ALTREP mechanism and support could be
  added.
- **IPC support**: The IPC reader is not currently exposed in the R bindings.

## Python bindings

- **Packaging**: The Python bindings are currently unpublished (pypi or conda) and
  are not included in release verification.
- **Element conversion**: There is currently no mechanism to extract an element
  of an `ArrowArrayView` as a Python object (e.g., an `int` or `str`).
- **numpy/Pandas conversion**: The Python bindings currently expose the `ArrowArrayView`
  but do not provide a means by which to convert to popular packages such as
  numpy or Pandas.
- **Creating arrays**: The Python bindings do not currently provide a means by
  which to create an `ArrowArray` from buffers or incrementally.
- **IPC support**: The IPC reader is not currently exposed in the Python bindings.
