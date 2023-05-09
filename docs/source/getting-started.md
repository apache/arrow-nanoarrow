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

## The interface

- C data

## Basic nanoarrow

- Error handling
- Memory management

## Reading into an ArrowArray

read lines into a string array

## Writing from an ArrowArray

assemble lines from a string array

## Exposing a computation

Count lines

## Exposing bindings in R

## Exposing bindings in Python


