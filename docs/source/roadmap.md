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
  are available via the Arrow C Data interface. Now that the run-end encoded (REE)
  types and string view/list view types are available via the Arrow C Data interface,
  support should be added in nanoarrow as well.
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
- **Test verbosity**: Tests for the C library were written before testing utilities
  in the `nanoarrow_testing` library were available (and before there was a
  `nanoarrow_testing` library in which to put new ones). As a result, some of them
  are very verbose and can be difficult to read. W
- **C++ integration**: The existing C++ integration is intentionally minimal;
  however, there are likely improvements that could be made to better integrate
  nanoarrow into existing C++ projects.
- **Documentation**: As the C library and its user base evolves, documentation
  needs to be refined and expanded to support the current set of use cases.

## IPC extension

- **Dictionary support**: The IPC extension does not currently support reading
  dictionary messages an IPC stream.
- **Compression**: The IPC extension does not currently support compressed
  streams using per-buffer compression, although streams can be compressed
  outside the nanoarrow library (e.g., gzip compression of the entire stream).

## Device extension

- **Define useful scope**: The device extension now provides basic support for
  creating and consuming `ArrowDeviceArray` structures for CPU, CUDA, and
  Apple Metal devices, including copying to/from the CPU device of arbitrary
  arrays. As adoption of `ArrowDeviceArray` becomes more widespread, the
  useful scope of the extension needs to be defined based on its use (or not)
  in the Arrow community.

## R bindings

- **Conversion internals**: The initial implementation of conversion from
  Arrow data to R vectors was implemented in C and its verbosity makes it
  difficult to add support for new types. The internals should be refactored
  to make the conversion code easier to understand for new developers.
- **Type support**: The R bindings currently rely on the Arrow R package for
  conversion of some R types (e.g., list_of), and some types are not supported
  in nanoarrow nor the arrow R package (e.g., run-end encoding, list view, and
  string/binary view).
- **ALTREP support**: A recent R release added enhanced ALTREP support such that
  types that convert to `list()` can defer materialization cost/allocation.
  Arrow sources that arrive in chunks (e.g., from a `Table` or `ChunkedArray`)
  currently can't be converted via any ALTREP mechanism and support could be
  added.

## Python bindings

- **Type support**: The Python bindings do not currently support unions,
  string/binary view, or list view, or run-end-encoded types. When creating
  Arrow arrays from iterables of Python objects, some types are not yet
  supported (e.g., struct, list, datetime objects).
