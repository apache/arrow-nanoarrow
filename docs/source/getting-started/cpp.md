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

# Getting started with nanoarrow in C/C++

This tutorial provides a short example of writing a C++ library that exposes
an Arrow-based API and uses nanoarrow to implement a simple text file reader/writer.
In general, nanoarrow can help you write a library or application that:

- exposes an Arrow-based API to read from a data source or format,
- exposes an Arrow-based API to write to a data source or format,
- exposes one or more compute functions that operates on and produces data
  in the form of Arrow arrays, and/or
- exposes an extension type implementation.

Because Arrow has bindings in many languages, it means that you or others can easily
bind or use your tool in higher-level runtimes like R, Java, C++, Python, Rust, Julia,
Go, or Ruby, among others.

The nanoarrow library is not the only way that an Arrow-based API can be implemented:
Arrow C++, Rust, and Go are all excellent choices and can compile into
static libraries that are C-linkable from other languages; however, existing Arrow
implementations produce relatively large static libraries and can present complex build-time
or run-time linking requirements depending on the implementation and features used. If
the set of libraries you're working with already provide the conveniences you require,
nanoarrow may provide all the functionality you need.

Now that we've talked about why you might want to build a library with nanoarrow...let's
build one!

```{=rst}
.. note::
  This tutorial also goes over some of the basic structure of writing a C++ library.
  If you already know how to do this, feel free to scroll to the code examples provided
  below or take a look at the
  `final example project <https://github.com/apache/arrow-nanoarrow/tree/main/examples/linesplitter>`__.

```

## The library

The library we'll write in this tutorial is a simple text processing library that splits
and reassembles lines of text. It will be able to:

- Read text from a buffer into an `ArrowArray` as one element per line, and
- Write elements of an `ArrowArray` into a buffer, inserting line breaks
  after every element.

For the sake of argument, we'll call it `linesplitter`.

## The development environment

There are many excellent IDEs that can be used to develop C and C++ libraries. For
this tutorial, we will use [VSCode](https://code.visualstudio.com/) and
[CMake](https://cmake.org/). You'll need both installed to follow along:
VSCode can be downloaded from the official site for most platforms;
CMake is typically installed via your favourite package manager
(e.g., `brew install cmake`, `apt-get install cmake`, `dnf install cmake`,
etc.). You will also need a C and C++ compiler: on MacOS these can be installed
using `xcode-select --install`; on Linux you will need the packages that provide
`gcc`, `g++`, and `make` (e.g., `apt-get install build-essential`); on Windows
you will need to install
[Visual Studio](https://visualstudio.microsoft.com/downloads/) and
CMake from the official download pages.

Once you have VSCode installed, ensure you have the **CMake Tools** and **C/C++**
extensions installed. Once your environment is set up, create a folder called
`linesplitter` and open it using **File -> Open Folder**.

## The interface

We'll expose the interface to our library as a header called `linesplitter.h`.
To ensure the definitions are only included once in any given source file, we'll
add the following line at the top:

```cpp
#pragma once
```

Then, we need the
[Arrow C Data interface](https://arrow.apache.org/docs/format/CDataInterface.html#structure-definitions)
itself, since it provides the type definitions that are recognized by other Arrow
implementations on which our API will be built. It's designed to be copy and
pasted in this way - there's no need to put it in another file include
something from another project.

```cpp
#include <stdint.h>

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {
  // Array type description
  const char* format;
  const char* name;
  const char* metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema** children;
  struct ArrowSchema* dictionary;

  // Release callback
  void (*release)(struct ArrowSchema*);
  // Opaque producer-specific data
  void* private_data;
};

struct ArrowArray {
  // Array data description
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void** buffers;
  struct ArrowArray** children;
  struct ArrowArray* dictionary;

  // Release callback
  void (*release)(struct ArrowArray*);
  // Opaque producer-specific data
  void* private_data;
};

#endif  // ARROW_C_DATA_INTERFACE
```

Next, we'll provide definitions for the functions we'll implement below:

```c
// Builds an ArrowArray of type string that will contain one element for each line
// in src and places it into out.
//
// On success, returns {0, ""}; on error, returns {<errno code>, <error message>}
std::pair<int, std::string> linesplitter_read(const std::string& src,
                                              struct ArrowArray* out);

// Concatenates all elements of a string ArrowArray inserting a newline between
// elements.
//
// On success, returns {0, <result>}; on error, returns {<errno code>, <error message>}
std::pair<int, std::string> linesplitter_write(struct ArrowArray* input);
```

```{=rst}
.. note::
  You may notice that we don't include or mention nanoarrow in any way in the header
  that is exposed to users. Because nanoarrow is designed to be vendored and is not
  distributed as a system library, it is not safe for users of your library to
  ``#include "nanoarrow.h"`` because it might conflict with another library that does
  the same (with possibly a different version of nanoarrow).

```

## Arrow C data/nanoarrow interface basics

Now that we've seen the functions we need to implement and the Arrow types exposed
in the C data interface, let's unpack a few basics about using the Arrow C data
interface and a few conventions used in the nanoarrow implementation.

First, let's discuss the `ArrowSchema` and the `ArrowArray`. You can think of an
`ArrowSchema` as an expression of a data type, whereas an `ArrowArray` is the
data itself. These structures accommodate nested types: columns are encoded in
the `children` member of each. You always need to know the data type of an
`ArrowArray` before accessing its contents. In our case we only operate on arrays
of one type ("string") and document that in our interface; for functions that
operate on more than one type of array you will need to accept an `ArrowSchema`
and inspect it (e.g., using nanoarrow's helper functions).

Second, let's discuss error handling. You may have noticed in the function definitions
above that we return `int`, which is an errno-compatible error code or `0` to
indicate success. Functions in nanoarrow that need to communicate more detailed
error information accept an `ArrowError*` argument (which can be `NULL` if
the caller does care about the extra information). Any nanoarrow function that
might fail communicates errors in this way. To avoid verbose code like the
following:

```c
int init_string_non_null(struct ArrowSchema* schema) {
  int code = ArrowSchemaInitFromType(&schema, NANOARROW_TYPE_STRING);
  if (code != NANOARROW_OK) {
    return code;
  }

  schema->flags &= ~ARROW_FLAG_NULLABLE;
  return NANOARROW_OK;
}
```

...you can use the `NANOARROW_RETURN_NOT_OK()` macro:

```c
int init_string_non_null(struct ArrowSchema* schema) {
  NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(&schema, NANOARROW_TYPE_STRING));
  schema->flags &= ~ARROW_FLAG_NULLABLE;
  return NANOARROW_OK;
}
```

This works as long as your internal functions that use nanoarrow also return
`int` and/or an `ArrowError*` argument. This usually means that there is
an outer function that presents a more idiomatic interface (e.g., returning
`std::optional<>` or throwing an exception) and an inner function that uses
nanoarrow-style error handling. Embracing `NANOARROW_RETURN_NOT_OK()` is key
to happiness when using the nanoarrow library.

Third, let's discuss memory management. Because nanoarrow is implemented in C
and provides a C interface, the library by default uses C-style memory management
(i.e., if you allocate it, you clean it up). This is unnecessary when you have
C++ at your disposal, so nanoarrow also provides a C++ header (`nanoarrow.hpp`) with
`std::unique_ptr<>`-like wrappers around anything that requires explicit clean up.
Whereas in C you might have to write code like this:

```c
struct ArrowSchema schema;
struct ArrowArray array;

// Ok: if this returns, array was not initialized
NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(&schema, NANOARROW_TYPE_STRING));

// Verbose: if this fails, we need to release schema before returning
// or it will leak.
int code = ArrowArrayInitFromSchema(&array, &schema, NULL);
if (code != NANOARROW_OK) {
  ArrowSchemaRelease(&schema);
  return code;
}
```

...using the `nanoarrow.hpp` types we can do:

```cpp
nanoarrow::UniqueSchema schema;
nanoarrow::UniqueArray array;

NANOARROW_RETURN_NOT_OK(ArrowSchemaInitFromType(schema.get(), NANOARROW_TYPE_STRING));
NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromSchema(array.get(), schema.get(), NULL));
```

## Building the library

Our library implementation will live in `linesplitter.cc`. Before writing the
actual implementations, let's add just enough to our project that we can
build it using VSCode's C/C++/CMake integration:

```cpp
#include <cerrno>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>

#include "nanoarrow/nanoarrow.hpp"

#include "linesplitter.h"

std::pair<int, std::string> linesplitter_read(const std::string& src,
                                              struct ArrowArray* out) {
  return {ENOTSUP, ""};
}

std::pair<int, std::string> linesplitter_write(struct ArrowArray* input) {
  return {ENOTSUP, ""};
}
```

We also need a `CMakeLists.txt` file that tells CMake and VSCode what to build.
CMake has a lot of options and can scale to coordinate very large projects; however
we only need a few lines to leverage VSCode's integration.

```cmake
project(linesplitter)

set(CMAKE_CXX_STANDARD 11)

include(FetchContent)

FetchContent_Declare(
  nanoarrow
  URL https://github.com/apache/arrow-nanoarrow/releases/download/apache-arrow-nanoarrow-0.2.0/apache-arrow-nanoarrow-0.2.0.tar.gz
  URL_HASH SHA512=38a100ae5c36a33aa330010eb27b051cff98671e9c82fff22b1692bb77ae61bd6dc2a52ac6922c6c8657bd4c79a059ab26e8413de8169eeed3c9b7fdb216c817)
FetchContent_MakeAvailable(nanoarrow)

add_library(linesplitter linesplitter.cc)
target_link_libraries(linesplitter PRIVATE nanoarrow_static)
```

After saving `CMakeLists.txt`, you may have to close and re-open the `linesplitter`
directory in VSCode to activate the CMake integration. From the command palette
(i.e., Control/Command-Shift-P), choose **CMake: Build**. If all went well, you should
see a few lines of output indicating progress towards building and linking `linesplitter`.

```{=rst}
.. note::
  Depending on your version of CMake you might also see a few warnings. This CMakeLists.txt
  is intentionally minimal and as such does not attempt to silence them.

```

```{=rst}
.. note::
  If you're not using VSCode, you can accomplish the equivalent task in in a terminal
  with ``mkdir build && cd build && cmake .. && cmake --build .``.

```

## Building an ArrowArray

The input for our `linesplitter_read()` function is an `std::string`, which we'll iterate
over and add each detected line as its own element. First, we'll define a function for
the core logic of detecting the number of characters until the next `\n` or end-of-string.

```cpp
static int64_t find_newline(const ArrowStringView& src) {
  for (int64_t i = 0; i < src.size_bytes; i++) {
    if (src.data[i] == '\n') {
      return i;
    }
  }

  return src.size_bytes;
}
```

The next function we'll define is an internal function that uses nanoarrow-style error
handling. This uses the `ArrowArrayAppend*()` family of functions provided by
nanoarrow to build the array:

```cpp
static int linesplitter_read_internal(const std::string& src, ArrowArray* out,
                                      ArrowError* error) {
  nanoarrow::UniqueArray tmp;
  NANOARROW_RETURN_NOT_OK(ArrowArrayInitFromType(tmp.get(), NANOARROW_TYPE_STRING));
  NANOARROW_RETURN_NOT_OK(ArrowArrayStartAppending(tmp.get()));

  ArrowStringView src_view = {src.data(), static_cast<int64_t>(src.size())};
  ArrowStringView line_view;
  int64_t next_newline = -1;
  while ((next_newline = find_newline(src_view)) >= 0) {
    line_view = {src_view.data, next_newline};
    NANOARROW_RETURN_NOT_OK(ArrowArrayAppendString(tmp.get(), line_view));
    src_view.data += next_newline + 1;
    src_view.size_bytes -= next_newline + 1;
  }

  NANOARROW_RETURN_NOT_OK(ArrowArrayFinishBuildingDefault(tmp.get(), error));

  ArrowArrayMove(tmp.get(), out);
  return NANOARROW_OK;
}
```

Finally, we define a wrapper that corresponds to the outer function definition.

```cpp
std::pair<int, std::string> linesplitter_read(const std::string& src, ArrowArray* out) {
  ArrowError error;
  int code = linesplitter_read_internal(src, out, &error);
  if (code != NANOARROW_OK) {
    return {code, std::string(ArrowErrorMessage(&error))};
  } else {
    return {NANOARROW_OK, ""};
  }
}
```

## Reading an ArrowArray

The input for our `linesplitter_write()` function is an `ArrowArray*` like the one we
create in `linesplitter_read()`. Just as nanoarrow provides helpers to build arrays,
it also provides helpers to read them via the `ArrowArrayView*()` family of functions.
Again, we first define an internal function that uses nanoarrow-style error handling:

```cpp
static int linesplitter_write_internal(ArrowArray* input, std::stringstream& out,
                                       ArrowError* error) {
  nanoarrow::UniqueArrayView input_view;
  ArrowArrayViewInitFromType(input_view.get(), NANOARROW_TYPE_STRING);
  NANOARROW_RETURN_NOT_OK(ArrowArrayViewSetArray(input_view.get(), input, error));

  ArrowStringView item;
  for (int64_t i = 0; i < input->length; i++) {
    if (ArrowArrayViewIsNull(input_view.get(), i)) {
      out << "\n";
    } else {
      item = ArrowArrayViewGetStringUnsafe(input_view.get(), i);
      out << std::string(item.data, item.size_bytes) << "\n";
    }
  }

  return NANOARROW_OK;
}
```

Then, provide an outer wrapper that corresponds to the outer function definition.

```cpp
std::pair<int, std::string> linesplitter_write(ArrowArray* input) {
  std::stringstream out;
  ArrowError error;
  int code = linesplitter_write_internal(input, out, &error);
  if (code != NANOARROW_OK) {
    return {code, std::string(ArrowErrorMessage(&error))};
  } else {
    return {NANOARROW_OK, out.str()};
  }
}
```

## Testing

We have an implementation, but does it work? Unlike higher-level runtimes like
R and Python, we can't just open a prompt and type some code to find out. For
C and C++ libraries, the
[googletest](https://google.github.io/googletest/quickstart-cmake.html)
framework provides a quick and easy way to do this that scales nicely as the
complexity of your project grows.

First, we'll add a stub test and some CMake to get going. In `linesplitter_test.cc`,
add the following:

```cpp
#include <gtest/gtest.h>

#include "nanoarrow/nanoarrow.hpp"

#include "linesplitter.h"

TEST(Linesplitter, LinesplitterRoundtrip) {
  EXPECT_EQ(4, 4);
}
```

Then, add the following to your `CMakeLists.txt`:

```cmake
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(linesplitter_test linesplitter_test.cc)
target_link_libraries(linesplitter_test linesplitter GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(linesplitter_test)
```

After you're done, build the project again using the **CMake: Build** command from
the command palette. If all goes well, choose **CMake: Refresh Tests** and then
**Test: Run All Tests** from the command palette to run them! You should see some
output indicating that tests ran successfully, or you can use VSCode's "Testing"
panel to visually inspect which tests passed.

```{=rst}
.. note::
  If you're not using VSCode, you can accomplish the equivalent task in in a terminal
  with ``cd build && ctest .``.

```

Now we're ready to fill in the test! Our two functions happen to round trip,
so a useful first test might be to check.

```cpp
TEST(Linesplitter, LinesplitterRoundtrip) {
  nanoarrow::UniqueArray out;
  auto result = linesplitter_read("line1\nline2\nline3", out.get());
  ASSERT_EQ(result.first, 0);
  ASSERT_EQ(result.second, "");

  ASSERT_EQ(out->length, 3);

  nanoarrow::UniqueArrayView out_view;
  ArrowArrayViewInitFromType(out_view.get(), NANOARROW_TYPE_STRING);
  ASSERT_EQ(ArrowArrayViewSetArray(out_view.get(), out.get(), nullptr), 0);
  ArrowStringView item;

  item = ArrowArrayViewGetStringUnsafe(out_view.get(), 0);
  ASSERT_EQ(std::string(item.data, item.size_bytes), "line1");

  item = ArrowArrayViewGetStringUnsafe(out_view.get(), 1);
  ASSERT_EQ(std::string(item.data, item.size_bytes), "line2");

  item = ArrowArrayViewGetStringUnsafe(out_view.get(), 2);
  ASSERT_EQ(std::string(item.data, item.size_bytes), "line3");


  auto result2 = linesplitter_write(out.get());
  ASSERT_EQ(result2.first, 0);
  ASSERT_EQ(result2.second, "line1\nline2\nline3\n");
}
```

Writing tests in this way also opens up a relatively straightforward debug
path via the **CMake: Set Debug target** and **CMake: Debug** commands.
If the first thing that happens when you write run your test is a crash,
running the tests with the debugger turned on will automatically pause at
the line of code that caused the crash. For more fine-tuned debugging,
you can set breakpoints and step through code.

## Summary

This tutorial covered the basics of writing and testing a C++ library exposing an
Arrow-based API implemented using the nanoarrow C library.
