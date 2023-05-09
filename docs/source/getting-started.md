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

#define LINESTRING_OK 0

int linesplitter_read(const char* src, struct ArrowArray* out);
std::pair<int, std::string> linesplitter_write(struct ArrowArray* input);
int linesplitter_separate_longer(struct ArrowArray* input, struct ArrowArray* output);
```

## The basics

- Error handling
- Memory management

## The tests



## Reading into an ArrowArray

read lines into a string array

## Writing from an ArrowArray

assemble lines from a string array

## Exposing a function

Count lines
