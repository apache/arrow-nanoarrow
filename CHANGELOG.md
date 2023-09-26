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

# nanoarrow Changelog

## nanoarrow 0.1.0 (2023-03-01)

### Feat

- **extensions/nanoarrow_ipc**: Improve type coverage of schema field decode (#115)
- **r**: Add `as_nanoarrow_array()` implementation that does not fall back on `arrow::as_arrow_array()` everywhere (#108)
- **r**: Create nanoarrow_array objects from buffers (#105)
- **r**: Implement infer schema methods (#104)
- **r**: Create and modify nanoarrow_schema objects (#101)

### Fix

- Correct storage type for timestamp and duration types (#116)
- **extensions/nanoarrow_ipc**: Remove extra copy of flatcc from dist/ (#113)
- make sure bundled nanoarrow is also valid C++ (#112)
- **extensions/nanoarrow_ipc**: Tweak draft interface (#111)
- set map entries/key to non-nullable (#107)
- **ci**: Actually commit bundled IPC extension to dist/ (#96)

## nanoarrow 0.2.0 (2023-06-16)

### Feat

- **python**: Python schema, array, and array view skeleton (#117)
- Include dictionary member in `ArrowArrayView` struct (#221)
- **extensions/nanoarrow_ipc**: Add endian swapping to IPC reader (#214)
- **r**: Improve printing and conversion of buffers (#208)
- Add `enum ArrowType buffer_data_type` member to `struct ArrowLayout` (#207)
- Implement ListChildOffset function (#197)
- **r**: Add ability to deterministically run a finalizer on an array stream (#196)
- **r**: Union array support (#195)
- **r**: Add ArrowArrayStream implementation to support keeping a dependent object in scope (#194)
- Add Decimal Get/Set utilities (#180)
- **extensions/nanoarrow_ipc**: Add option to validate arrays at `NANOARROW_VALIDATION_LEVEL_FULL` (#177)
- Allow explicit validation level in `ArrowArrayFinishBuilding()` (#175)
- Implement `ArrowArrayViewValidateFull()` (#174)
- **extensions/nanoarrow_ipc**: Allow shared buffers for zero-copy buffer decode (#165)
- **extensions/nanoarrow_ipc**: Add single-threaded stream reader (#164)
- **ci**: Add suite of Docker-based release verification script tests (#160)
- **extensions/nanoarrow_ipc**: Decode RecordBatch message to ArrowArray (#143)
- **extensions/nanoarrow_ipc**: Improve type coverage of schema field decode (#115)
- **r**: Add `as_nanoarrow_array()` implementation that does not fall back on `arrow::as_arrow_array()` everywhere (#108)
- **r**: Create nanoarrow_array objects from buffers (#105)
- **r**: Implement infer schema methods (#104)
- **r**: Create and modify nanoarrow_schema objects (#101)

### Fix

- Improve limit check for unsigned input (#233)
- **extensions/nanoarrow_ipc**: Don't produce arrays with NULL data buffers (#226)
- **r**: Fix `convert_array_stream()` for non-record batch stream with zero batches (#212)
- clear `release` in `EmptyArrayStream::release_wrapper` (#204)
- **r**: Release streams when calling `as.vector()` or `as.data.frame()` (#202)
- Improve readability of `ArrowArrayAllocateChildren()` (#199)
- **extensions/nanoarrow_ipc**: Fix + test calling `ArrowIpcDecoderSetSchema()` more than once (#173)
- **extensions/nanoarrow_ipc**: Don't release input stream automatically on end of input (#168)
- **r**: Don't invoke undefined behaviour in conversions to/from Arrow (#167)
- **extensions/nanoarrow_ipc**: Test without C11 atomics on CI (#166)
- **extensions/nanoarrow_ipc**: Ensure tests pass on big endian (#162)
- **r**: Use strict prototypes in all internal C functions (#151)
- **r**: Don't memcpy NULL when converting buffer to raw (#149)
- include compilers in conda instructions (#142)
- include gtest in conda instructions (#138)
- Explicit `stringsAsFactors = FALSE` for R <= 3.6 (#135)
- Support centos7/gcc 4.8 for CMake build + test workflow (#133)
- Fix cmake build + test and verification script on Windows (#130)
- `isnan()` usage compatible with old clang (#126)
- Improve reliability of R tests on non-standard systems (#127)
- Ensure tests pass on big endian (#128)
- Correct storage type for timestamp and duration types (#116)
- **extensions/nanoarrow_ipc**: Remove extra copy of flatcc from dist/ (#113)
- make sure bundled nanoarrow is also valid C++ (#112)
- **extensions/nanoarrow_ipc**: Tweak draft interface (#111)
- set map entries/key to non-nullable (#107)
- **ci**: Actually commit bundled IPC extension to dist/ (#96)

### Refactor

- **extensions/nanoarrow_ipc**: Reconfigure assembling arrays for better validation (#209)
- Unify `ArrowArrayView` and `ArrowArray` validation (#201)

InvalidVersion GitTag('apache-arrow-nanoarrow-0.2.0', 'f71063605e288d9a8dd73cfdd9578773519b6743', '2023-06-22')
InvalidVersion GitTag('apache-arrow-nanoarrow-0.2.0-rc1', 'f71063605e288d9a8dd73cfdd9578773519b6743', '2023-06-19')
InvalidVersion GitTag('apache-arrow-nanoarrow-0.2.0-rc0', 'a7b824de6cb99ce458e1a5cd311d69588ceb0570', '2023-06-16')
InvalidVersion GitTag('ls', 'ab24d42760b6730cb64fb50e72c30b4c82830a24', '2023-06-16')
InvalidVersion GitTag('apache-arrow-nanoarrow-0.1.0', '341279af1b2fdede36871d212f339083ffbd75eb', '2023-03-07')
InvalidVersion GitTag('apache-arrow-nanoarrow-0.1.0-rc1', '341279af1b2fdede36871d212f339083ffbd75eb', '2023-03-01')
InvalidVersion GitTag('apache-arrow-nanoarrow-0.1.0-rc0', '5415013c9e9be240ad5965444bbf8cfd4642aaec', '2023-02-24')
InvalidVersion GitTag('latest', '01c66380a4358cf390b4d609794fdac2f2ebfb45', '2022-08-24')
## nanoarrow 0.3.0 (2023-09-26)

### Feat

- **r**: Use classed warnings to signal that a lossy conversion occurred (#298)
- **r**: Add support for `bit64::integer64()` conversions (#293)
- **r**: Implement extension type registration/conversion  (#288)
- **r**: Implement dictionary conversion (#285)
- Add `ArrowBitsUnpackInt32()` (#278)
- Add `ArrowBitmapUnpackInt8Unsafe()` (#276)
- Add Support for Intervals (#258)
- **extensions/nanoarrow_device**: Draft DeviceArray interface (#205)
- **python**: Python schema, array, and array view skeleton (#117)
- Include dictionary member in `ArrowArrayView` struct (#221)
- **extensions/nanoarrow_ipc**: Add endian swapping to IPC reader (#214)
- **r**: Improve printing and conversion of buffers (#208)
- Add `enum ArrowType buffer_data_type` member to `struct ArrowLayout` (#207)
- Implement ListChildOffset function (#197)
- **r**: Add ability to deterministically run a finalizer on an array stream (#196)
- **r**: Union array support (#195)
- **r**: Add ArrowArrayStream implementation to support keeping a dependent object in scope (#194)
- Add Decimal Get/Set utilities (#180)
- **extensions/nanoarrow_ipc**: Add option to validate arrays at `NANOARROW_VALIDATION_LEVEL_FULL` (#177)
- Allow explicit validation level in `ArrowArrayFinishBuilding()` (#175)
- Implement `ArrowArrayViewValidateFull()` (#174)
- **extensions/nanoarrow_ipc**: Allow shared buffers for zero-copy buffer decode (#165)
- **extensions/nanoarrow_ipc**: Add single-threaded stream reader (#164)
- **ci**: Add suite of Docker-based release verification script tests (#160)
- **extensions/nanoarrow_ipc**: Decode RecordBatch message to ArrowArray (#143)
- **extensions/nanoarrow_ipc**: Improve type coverage of schema field decode (#115)
- **r**: Add `as_nanoarrow_array()` implementation that does not fall back on `arrow::as_arrow_array()` everywhere (#108)
- **r**: Create nanoarrow_array objects from buffers (#105)
- **r**: Implement infer schema methods (#104)
- **r**: Create and modify nanoarrow_schema objects (#101)

### Fix

- Resolve build warnings on Windows (#304)
- Return `EOVERFLOW` when appending to a string or binary type would exeed 2 GB (#302)
- **dev/release**: Increase test discovery timeout value (#300)
- Fix declaration of an array with an ambiguously constexpr size (#301)
- **r**: Ensure `ordered` is reflected in `na_dictionary()` (#299)
- **r**: Warn for possibly out of range int64 -> double conversions (#294)
- **extensions/nanoarrow_ipc**: Check number of bytes read when reading buffer body (#295)
- Ensure that test for increasing offsets is not affected by overflow (#291)
- **extensions/nanoarrow_ipc**: Fix crash and mixleading error messages resulting from corrupted streams (#289)
- **r**: Support map conversion to R vector (#282)
- **examples/linesplitter**: Fix CMake Build (#271)
- **r**: Don't link to arrow package R6 class pages (#269)
- **python**: Ensure generator does not raise `StopIteration` (#262)
- **docs**: Fix typo in getting started article (#250)
- Fix bad access crash in `ArrowBitmapByteCountSet()` (#242)
- Improve limit check for unsigned input (#233)
- **extensions/nanoarrow_ipc**: Don't produce arrays with NULL data buffers (#226)
- **r**: Fix `convert_array_stream()` for non-record batch stream with zero batches (#212)
- clear `release` in `EmptyArrayStream::release_wrapper` (#204)
- **r**: Release streams when calling `as.vector()` or `as.data.frame()` (#202)
- Improve readability of `ArrowArrayAllocateChildren()` (#199)
- **extensions/nanoarrow_ipc**: Fix + test calling `ArrowIpcDecoderSetSchema()` more than once (#173)
- **extensions/nanoarrow_ipc**: Don't release input stream automatically on end of input (#168)
- **r**: Don't invoke undefined behaviour in conversions to/from Arrow (#167)
- **extensions/nanoarrow_ipc**: Test without C11 atomics on CI (#166)
- **extensions/nanoarrow_ipc**: Ensure tests pass on big endian (#162)
- **r**: Use strict prototypes in all internal C functions (#151)
- **r**: Don't memcpy NULL when converting buffer to raw (#149)
- include compilers in conda instructions (#142)
- include gtest in conda instructions (#138)
- Explicit `stringsAsFactors = FALSE` for R <= 3.6 (#135)
- Support centos7/gcc 4.8 for CMake build + test workflow (#133)
- Fix cmake build + test and verification script on Windows (#130)
- `isnan()` usage compatible with old clang (#126)
- Improve reliability of R tests on non-standard systems (#127)
- Ensure tests pass on big endian (#128)
- Correct storage type for timestamp and duration types (#116)
- **extensions/nanoarrow_ipc**: Remove extra copy of flatcc from dist/ (#113)
- make sure bundled nanoarrow is also valid C++ (#112)
- **extensions/nanoarrow_ipc**: Tweak draft interface (#111)
- set map entries/key to non-nullable (#107)
- **ci**: Actually commit bundled IPC extension to dist/ (#96)

### Refactor

- **r**: Use `basic_array_stream()` to improve array stream to data.frame conversion (#279)
- **python**: Cleaner strategy for `__iter__()` on an `ArrowArrayStream` (#263)
- **extensions/nanoarrow_ipc**: Reconfigure assembling arrays for better validation (#209)
- Unify `ArrowArrayView` and `ArrowArray` validation (#201)
