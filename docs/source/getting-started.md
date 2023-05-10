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

# Getting started with nanoarrow

This tutorial provides a short example of writing a C++ library that exposes
an Arrow-based API and uses nanoarrow to implement a simple text file reader/writer.
In general, nanoarrow can help you write a library or application that:

- exposes an Arrow-based API to read from a data source or format,
- exposes an Arrow-based API to write to a data source or format,
- exposes one or more compute functions that operates on and produces data
  in the form of Arrow arrays, and/or
- exposes an extension type implementation.

Becauase Arrow has bindings in many languages, it means that you or others can easily
bind or use your tool in higher-level runtimes like R, Java, C++, Python, Rust, Julia,
Go, or Ruby, among others.

The nanoarrow library is not the only way that an Arrow-based API can be implemented: Arrow C++, Rust, and Go are all excellent choices and can compile into
static libraries that are C-linkable from other languages; however, existing Arrow
implementations produce relatively large static libraries and can present complex build-time
or run-time linking requirements depending on the implementation and features used. If
the set of libraries you're working with already provide the conveniences you need,
nanoarrow may provide all the functionality you need.

Now that we've talked about why you might want to build a library with nanoarrow...let's
build one!

## The library

The library we'll write in this tutorial is a simple text processing library that splits
and reassembles lines of text. It will be able to:

- Read text from a buffer into an `ArrowArray` as one element per line,
- Write elements of an `ArrowArray` into a buffer, inserting line breaks
  after every element, and
- Split the elements of an `ArrowArray` after every line and produce an
  `ArrowArray` with one element per line.

For the sake of argument, we'll call it `linesplitter`.

## The development environment

If you're reading this tutorial there is a good chance you already have a
workflow for building and testing C/C++ libraries. If you do, skip to the
next section!

There are many excellent IDEs that can be used to develop C and C++ libraries. For
this tutorial, we will use [VSCode](https://code.visualstudio.com/) and
[CMake](https://cmake.org/). You'll need both installed to follow along:
VSCode can be downloaded from the official site for most platforms;
CMake is typically installed via your favourite package manager
(e.g., `brew install cmake`, `apt-get install cmake` `dnf install cmake`,
etc.). You will also need a C and C++ compiler: on MacOS these can be installed using `xcode-select --install`; on Linux you will need the packages that provice
`gcc`, `g++`, and `make`; on Windows you will need to install
[Visual Studio](https://visualstudio.microsoft.com/downloads/) and
CMake from the official download pages.

After installing the required dependencies, create a folder called `linesplitter`
and open it.

## The interface

We'll expose the interface to our library as a header called `linesplitter.h`.
To ensure the definitions are only included once in any given source file, we'll
add the following line at the top:

```c
#pragma once
```

Then, we need the
[Arrow C Data interface](https://arrow.apache.org/docs/format/CDataInterface.html#structure-definitions)
itself, since it provides the type definitions that are recognized by other Arrow
implementations on which our API will be built. It's designed to be copy and
pasted in this way - there's no need to put it in another file include
something from another project.

```c
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
#include <string>
#include <util>

int linesplitter_read(const std::string& src, struct ArrowArray* out);
std::pair<int, std::string> linesplitter_write(struct ArrowArray* input);
int linesplitter_separate_longer(struct ArrowArray* input, struct ArrowArray* out);
```

## Arow C data interface basics

Now that we've seen the functions we need to implement and the Arrow types exposed
in the C data interface, let's unpack a few basics about using the Arrow C data
interface and a few conventions used in the nanoarrow implementation.

First, let's discuss the `ArrowSchema` and the `ArrowArray`. You can think of an
`ArrowSchema` as an expression of a data type, whereas an `ArrowArray` is the
data itself. These structures accomodate nested types: columns are encoded in
the `children` member of each. You always need to know the data type of an
`ArrowArray` before accessing its contents. In our case we only operate on arrays
of one type ("string") and document that in our interface; for functions that
operate on more than one type of array you will need to accept an `ArrowSchema`
and inspect it (e.g., using nanoarrow's helper functions).

Second, lets discuss error handling. You may have noticed in the function definitions
above that we return `int`, which is an errno-compatible error code or `0` to
indicate success. This is the error reporting scheme used by the C stream interface and
nanoarrow and is common in C where exceptions and C++17's `std::optional<>` are not
possible. If your library becomes complex and needs to communicate detailed
error information you will need to choose one of those idioms. In our library, the
only thing that can go wrong is if the OS fails to allocate memory, and this is
sufficiently communicated using the errno code `ENOMEM`.

## Building the library

Our library implementation will live in `linesplitter.cc`. Before writing the
actual implementations, let's add just enough to our project that we can
build it using VSCode's C/C++/CMake integration:

```c
#include <string>
#include <util>
#include <errno.h>

#include "nanoarrow.h"

#include "linesplitter.h"

int linesplitter_read(const std::string& src, struct ArrowArray* out) {
  return ENOTSUP;
}

std::pair<int, std::string> linesplitter_write(struct ArrowArray* input) {
  return {ENOTSUP, ""};
}

int linesplitter_separate_longer(struct ArrowArray* input, struct ArrowArray* out) {
    return ENOTSUP;
}
```

We also need a `CMakeLists.txt` file that tells CMake and VSCode what to build.
CMake has a lot of options and scale to coordinate very large projects; however
we only need a few lines to leverage VSCode's integration.

```cmake
include(FetchContent)

project(linesplitter)

FetchContent_Declare(
  nanoarrow
  URL https://github.com/apache/arrow-nanoarrow/releases/download/apache-arrow-nanoarrow-0.1.0/apache-arrow-nanoarrow-0.1.0.tar.gz
  URL_HASH SHA512=dc62480b986ee76aaad8e38c6fbc602f8cef2cc35a5f5ede7da2a93b4db2b63839bdca3eefe8a44ae1cb6895a2fd3f090e3f6ea1020cf93cfe86437304dfee17)

FetchContent_MakeAvailable(nanoarrow_example_cmake_minimal)

add_library(linesplitter linesplitter.cc)
```

After saving `CMakeLists.txt`, you may have to close and re-open the `linesplitter`
directory in VSCode to activate the CMake integration. From the command pallete
(i.e., Control/Command-Shift-P), choose **CMake: Build**. If all went well, you should
see a few lines of output indicating progress towards building and linking `linesplitter`.

If you're not using VSCode, the equivalent in a terminal would be `mkdir build && cd build && cmake .. && cmake --build .`.

## Reading into an ArrowArray

read lines into a string array

## Writing from an ArrowArray

assemble lines from a string array

## Exposing a function

Count lines
