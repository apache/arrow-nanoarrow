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

### Fix

- Resolve build warnings on Windows (#304)
- Return `EOVERFLOW` when appending to a string or binary type would exceed 2 GB (#302)
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

### Refactor

- **r**: Use `basic_array_stream()` to improve array stream to data.frame conversion (#279)
- **python**: Cleaner strategy for `__iter__()` on an `ArrowArrayStream` (#263)

## nanoarrow 0.4.0

### Docs

- **r**: fix source links from pkgdown site (#315)

### Feat

- Check flags field of ArrowSchema on ArrowSchemaViewInit (#368)
- Add decimal support to integration tester (#361)
- Add support for datetime types in integration tester (#356)
- Add dictionary support in integration test utility (#342)
- Add C Data integration test shared library (#337)
- Add Array/Schema/ArrayStream comparison utility to testing helpers (#330)
- Add batch reader and data file read/write to/from ArrowArrayStream (#328)
- Add integration testing reader for Column (#325)
- Add integration testing JSON reader (#322)
- Add Type/Field/Schema JSON writer for integration testing JSON (#321)
- **extensions/nanoarrow_testing**: Add nanoarrow_testing extension with testing JSON writer (#317)
- **python**: Implement user-facing Schema class (#366)
- **python**: basic export through PyCapsules (#320)
- **python**: Add ArrowDeviceArray extension to Python bindings (#313)
- **python**: Support the PyCapsule protocol (#318)
- **r**: Provide LinkingTo headers for extension packages (#332)
- **r**: Add more `nanoarrow_array_stream` generics (#349)
- **r**: Add conversion from integer type to `character()` (#345)

### Fix

- Issue with appending NULLs to IntervalArray (#376)
- Suppress unused parameter warnings (#365)
- Add `const` qualifier where appropriate in struct fields and declarations (#327)
- **docs**: Correct typo in testing.rst (#348)
- **r**: Ensure simple `list()`s can be converted without arrow installed (#344)
- **r**: Ensure wrapper array stream eagerly releases the wrapped array stream (#333)

### Perf

- Improved Bit (Un)packing Performance (#280)

### Refactor

- Add wrappers around callbacks to improve syntax and debuggability (#338)
- Improve syntax for implementing `ArrowArrayStream` from C++ (#336)
- **python**: Document, prefix, and add reprs for C-wrapping classes (#340)

## nanoarrow 0.5.0

### Docs

- Update top-level documentation (#473)
- **python**: Update Python bindings readme (#474)

### Feat

- Add `ArrowArray` and `ArrowArrayStream` C++ iterators  (#404)
- Meson support (#413)
- **python**: Add column-wise buffer builder (#464)
- **python**: Add visitor pattern + builders for column sequences (#454)
- **python**: Add copy_into() to CBufferView (#455)
- **python**: Ensure that buffer produced by `CBufferView.unpack_bits()` has a boolean type (#457)
- **python**: Unify printing of type information across classes (#458)
- **python**: Add `Array.from_chunks()` constructor (#456)
- **python**: Implement bitmap unpacking (#450)
- **python**: Allow creation of dictionary and list types (#445)
- **python**: Implement extension type and Schema metadata support (#431)
- **python**: Add user-facing ArrayStream class (#439)
- **python**: Iterate over array buffers (#433)
- **python**: add back nanoarrow.array(..) constructor (#441)
- **python**: function to inspect a single-chunk Array (#436)
- **python**: Create string/binary arrays from iterables (#430)
- **python**: Support Decimal types in convert to Python (#425)
- **python**: Add Arrow->Python datetime support (#417)
- **python**: Clarify interaction between the CDeviceArray, the CArrayView, and the CArray (#409)
- **python**: Add user-facing `Array` class (#396)
- **python**: Add CArrayView -> Python conversion (#391)
- **python**: Add bindings for IPC reader (#388)
- **python**: Add array creation/building from buffers (#378)
- **r**: Add experimental `nanoarrow_vctr` to wrap a list of arrays (#461)
- **r**: Add bindings for IPC reader (#390)

### Fix

- Ensure nanoarrow.hpp compiles on gcc 4.8 (#472)
- Ensure negative return values from snprintf() are not used as indexes (#418)
- Relax comparison strictness such that integration tests pass (#399)
- Make build and install dirs proper CMake package, fix C++ header inclusion, and add proper tests (#406)
- Ensure that the deallocator called by ArrowBufferDeallocator() is called exactly once (#387)
- **ci**: Use cached Arrow C++ build in CI (#410)
- **docs**: Fix typo in documentation for `ArrowSchemaSetTypeUnion()` (#432)
- **docs**: Correct typo in README.md (#414)
- **python**: Skip test relying on memoryview context manager on PyPy 3.8 (#479)
- **python**: Fix use of memoryview to write fill to the buffer builder (#477)
- **python**: Add iterator for null/na type (#467)
- **python**: Ensure reference-counting tests are skipped on PyPy (#453)
- **python**: Make shallow CArray copies less shallow to accommodate moving children (#451)
- **python**: Update tests for pyarrow 16 (#440)
- **r**: Fix tests for platforms where arrow dataset and/or zip is not available (#415)

### Refactor

- **docs**: Shuffle organization of sections to multiple pages (#460)
- **python**: Reorganize strategies for building arrays (#444)
- **r**: remove unnecessary package name call (#381)

## nanoarrow 0.6.0

### Docs

- **python**: Add example of python package with nanoarrow C extension (#645)

### Feat

- Add ArrowArrayView accessors to inspect buffer properties (#638)
- String/Binary View Support (#596)
- add Footer decoding (#598)
- Revendor flatcc (#592)
- Add IPC integration test executable (#585)
- Add `ArrowArrayViewCompare()` to check for array equality (#578)
- Add IPC stream writing (#571)
- add ipc RecordBatch encoding (#555)
- add ArrowIpcOutputStream (#570)
- Add IPC schema encoding (#568)
- Add IPC writer scaffolding (#564)
- Add ArrowArrayViewComputeNullCount (#562)
- Add Meson support in nanoarrow_device (#484)
- Meson build system for nanoarrow-ipc extension (#483)
- Add support for run-end encoded array (#507)
- Add float16 support for `ArrowArrayViewGet{Double,Int,UInt}Unsafe()` (#501)
- Add support for appending values to half float `ArrowArray` (#499)
- **extensions/nanoarrow_device**: Implement asynchronous buffer copying (#509)
- **python**: Add StringView and BinaryView IO to Python bindings (#637)
- **python**: Implement array from buffer for non-CPU arrays (#550)
- **python**: Implement bindings to IPC writer (#586)
- **python**: Implement CUDA build in Python bindings (#547)
- **r**: Add float16 support for R bindings (#650)
- **r**: Implement string view support in R bindings (#636)
- **r**: Allow opt-out of warning for unregistered extension types (#632)
- **r**: Add bindings to IPC writer (#608)

### Fix

- Remove unreachable code (#649)
- Properly ingest Binary View types without variadic buffers (#635)
- python schema repr does not truncate output (#628)
- Accommodate IPC messages without continuation bytes (#629)
- Ignore empty (but present) union validity bitmaps from before 1.0 (#630)
- Only validate relevant type_ids for array view slice (#627)
- Improve validation of offset buffers for sliced arrays (#626)
- Ensure CMake linking against built/installed nanoarrow works for all components (#614)
- Ensure footer test passes on big endian (#609)
- ensure 1 is written for boolean True (#601)
- Ensure that schema metadata is always present even if empty (#591)
- Include missing cases in `ArrowArrayInitFromType()` (#588)
- Silence warning when compiling nanoarrow.hpp on at least one version of MSVC (#590)
- don't require metadata order in nanoarrow_ipc_integration (#589)
- IPC streams did not include RecordBatch headers (#582)
- Fix Meson build for separated nanoarrow_testing target (#574)
- Ensure `children` is NULL for zero children in ArrayViewAllocateChildren (#556)
- CMake deprecation warnings from subprojects (#535)
- Meson install header files and pkgconfig (#542)
- Fix symbol export visibility in c_data_integration_test (#531)
- Fix Meson include directories (#532)
- Ensure we don't call cuMemAlloc with 0 bytesize (#534)
- Ensure ArrowDeviceArray implementation for AppleMetal passes tests on newer MacOS (#527)
- Check for offset + length > int64_max before using the value to calculate buffer sizes (#524)
- check `run_ends_view->length` before accessing its values (#518)
- Force static library build on Windows when building with Meson (#496)
- **ci**: Fix verify, meson-build, and docker-build weekly runs (#581)
- **ci**: Fix build and test of nanoarrow on centos7 and s390x (#576)
- **ci**: Pin r-lib actions as a workaround for latest action updates (#572)
- **ci**: Fix verification workflow (#552)
- **ci**: Stop building unbuildable image based on centos7 (#553)
- **python**: Fix detection of cuda library on hosted runner (#554)
- **r**: Avoid flatcc aligned_alloc() call when compiling R package (#494)

### Refactor

- Consolidate per-target actions in CMakeLists.txt (#573)
- Separate implementation from interface for nanoarrow_testing component (#561)
- Separate components into folders under src/nanoarrow (#536)
- Use ArrowStringView C++ literal in tests (#528)
- Move Meson build targets to top level directory (#530)
- Simplify Meson test generation (#525)
- Remove CMake requirement from Meson IPC config (#522)
- Use inttypes.h macros instead of casts to print fixed-width integers (#520)
- Consolidate device extension into main project (#517)
- Consolidate IPC extension into main project (#511)
- **extensions/nanoarrow_device**: Migrate CUDA device implementation to use the driver API (#488)
- **python**: Split ArrowArray and ArrowArrayStream modules (#559)
- **python**: Separate schema cython classes into their own module (#558)
- **python**: Split buffer Cython internals into a separate module (#549)
- **python**: Split device functionality into its own module (#548)
- **python**: Split type identifier utilities into their own module (#545)
- **python**: Extract utility functions into _utils.pyx (#529)
- **r**: Use JSON in experimental R vctrs extension type (#533)

### Test

- Fix meson build and clean up some warnings (#595)
- test with the `HalfFloatType` from arrow (#503)

## nanoarrow 0.7.0

### Clean

- Use Meson disabler objects instead of conditions (#744)
- Use meson format for meson build configurations (#682)
- Assorted Meson and clang-tidy fixes (#673)

### Docs

- Ensure all headers are parsed by Doxygen (#681)
- **python**: Update development instructions for Python bindings (#685)

### Feat

- Add UniqueSharedBuffer C++ wrapper (#747)
- Use feature options instead of boolean in Meson (#739)
- Use hidden symbol linkage in Meson configuration (#731)
- Add support for Decimal32/64 to R package (#717)
- Improve shared linkage support (#719)
- Add `ArrowDecimalAppendStringToBuffer()` (#720)
- Implement LIST_VIEW and LARGE_LIST_VIEW support (#710)
- Add ZSTD decompression support to IPC reader (#693)
- add Decimal32/Decimal64 support (#683)
- **python**: Implement extension type/mechanism in python package (#688)
- **python**: Add map type constructor (#687)
- **r**: Add zstd decompression support to R package (#733)
- **r**: Support native creation of more numeric Arrow arrays from integer vectors (#697)
- **r**: Support matrix objects as fixed-size-list arrays (#692)

### Fix

- Align flatbuffer test data stored as globals (#783)
- Make auto_features more user friendly in Meson (#763)
- Linking to nanoarrow-testing-shared on Windows (#778)
- Ensure nanoarrow-testing can be linked to in a shared Windows build (#770)
- Don't use the value of `__GNUC__` unless defined (#769)
- Fix ignored offset in ArrowArrayViewGetIntervalUnsafe (#755)
- Enure ArrowIpcSetDecompressor() marks input decompressor as released (#748)
- Fix valgrind error suggesting use of an uninitialized value (#750)
- Fix potential integer overflows (#736)
- ArrowDecimalSetDigits() on s390x + test fixes for big endian (#732)
- Remove rust version pin for integration test job (#729)
- CMake install directories on Windows (#715)
- remove deprecated operator"" syntax usage (#661)
- **ci**: Meson windows example CI (#791)
- **ci**: Fix runner specification for c-device (#760)
- **ci**: Update actions using ubuntu-20.04 (#745)
- **ci**: Use gcovr instead of lcov (#694)
- **cpp**: Ensure Meson build compiles Arrow tests (#711)
- **cpp**: Fix offset handling in ViewArrayAs Range Helpers (#702)
- **docs**: Fix download link in the README (#712)
- **python**: Ensure source distribution builds with Cython 3.1.0 (#759)
- **r**: Ensure that `python` is used on Windows when running bootstrap.R (#792)
- **r**: Update vctrs extension name to reflect implementation change (#752)
- **r**: sub-day precision Date should be floored when treated as integer (#674)

### Python

- Use meson-python instead of setuptools (#644)

### Refactor

- Split up nanoarrow.hpp into multiple .hpp files (#668)

### Test

- Test LargeList SchemaInit without Arrow (#714)
- Make Arrow C++ dependency optional (#677)

## nanoarrow 0.8.0

### Docs

- fix issues in ArrowBasicArrayStreamSetArray docstring (#838)
- fix typos in nanoarrow.h docstrings (#837)

### Feat

- Add LZ4 decompression support to IPC reader (#819)
- **python**: Support union types in Python bindings (#820)
- **r**: Add reticulate/Python integration (#817)
- **r**: Add support for creating timestamp and duration types from numeric storage (#816)

### Fix

- Fix leak reported by coverity scan (#832)
- Ensure ArrowArrayBuffer() and ArrowArraySetBuffer() work for variadic buffers (#808)
- Assorted updates to improve Meson WrapDB entry (#803)
- Ensure the array view can be used to inspect map offsets (#802)
- Refactor C function to resolve unreachable code error in Zig (#799)
- **python**: Remove incorrect last byte zeroing in Python buffer construction (#835)
- **r**: Correct invocation of R_MakeExternalPtr with R NULL (#841)
- **r**: Fix usage of deprecated syntax for forthcoming R release (#840)
- **r**: Fix map type and struct-nested-in-struct convert (#829)
- **r**: Collect array streams in C (not R) before conversion (#828)
- **r**: Fix test for forthcoming ALTREP behaviour in R-devel (#826)
- **r**: Ensure C23 version check works for clang16 (current GitHub Actions) (#801)
