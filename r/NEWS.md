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

# nanoarrow 0.5.0

- Add experimental `nanoarrow_vctr` to wrap a list of arrays (#461)
- Add bindings for IPC reader (#390)
- Fix tests for platforms where arrow dataset and/or zip is not available (#415)
- Remove unnecessary package name call (#381)

# nanoarrow 0.4.0

- Fix source links from pkgdown site (#315).
- Provide LinkingTo headers for extension packages (#332).
- Add more `nanoarrow_array_stream` generics (#349).
- Add conversion from integer type to `character()` (#345).
- Ensure simple `list()`s can be converted without arrow installed (#344).

# nanoarrow 0.3.0.1

- Ensure wrapper array stream eagerly releases the wrapped array stream (#333).

# nanoarrow 0.3.0

- Use classed warnings to signal that a lossy conversion occurred
  (#298)
- Add support for `bit64::integer64()` conversions (#293)
- Implement extension type registration/conversion  (#288)
- Implement dictionary conversion (#285)
- Ensure `ordered` is reflected in `na_dictionary()` (#299)
- Warn for possibly out of range int64 -> double conversions (#294)
- Support map conversion to R vector (#282)
- Don't link to arrow package R6 class pages (#269)
- Use `basic_array_stream()` to improve array stream to data.frame
  conversion (#279)

# nanoarrow 0.2.0-1

- Don't link to arrow package R6 class pages (#269)

# nanoarrow 0.2.0

## New features

- Improve printing and conversion of buffers (#208)
- Add `enum ArrowType buffer_data_type` member to `struct ArrowLayout` (#207)
- Implement ListChildOffset function (#197)
- Add ability to deterministically run a finalizer on an array stream (#196)
- Union array support (#195)
- Add ArrowArrayStream implementation to support keeping a dependent object in
  scope (#194)
- Add `as_nanoarrow_array()` implementation that does not fall back on
  `arrow::as_arrow_array()` everywhere (#108)
- Create nanoarrow_array objects from buffers (#105)
- Implement infer schema methods (#104)
- Create and modify nanoarrow_schema objects (#101)

## Bugfixes

- Fix `convert_array_stream()` for non-record batch stream with zero batches
  (#212)
- clear `release` in `EmptyArrayStream::release_wrapper` (#204)
- Release streams when calling `as.vector()` or `as.data.frame()` (#202)
- Don't invoke undefined behaviour in conversions to/from Arrow (#167)
- Use strict prototypes in all internal C functions (#151)
- Don't memcpy NULL when converting buffer to raw (#149)
