// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef NANOARROW_HPP_ARRAY_STREAM_HPP_INCLUDED
#define NANOARROW_HPP_ARRAY_STREAM_HPP_INCLUDED

#include <vector>

#include "nanoarrow/hpp/unique.hpp"

NANOARROW_CXX_NAMESPACE_BEGIN

/// \defgroup nanoarrow_hpp-array-stream ArrayStream helpers
///
/// These classes provide simple ArrowArrayStream implementations that
/// can be extended to help simplify the process of creating a valid
/// ArrowArrayStream implementation or used as-is for testing.
///
/// @{

/// @brief Export an ArrowArrayStream from a standard C++ class
/// @tparam T A class with methods `int GetSchema(ArrowSchema*)`, `int
/// GetNext(ArrowArray*)`, and `const char* GetLastError()`
///
/// This class allows a standard C++ class to be exported to a generic ArrowArrayStream
/// consumer by mapping C callback invocations to method calls on an instance of the
/// object whose lifecycle is owned by the ArrowArrayStream. See VectorArrayStream for
/// minimal useful example of this pattern.
///
/// The methods must be accessible to the ArrayStreamFactory, either as public methods or
/// by declaring ArrayStreamFactory<ImplClass> a friend. Implementors are encouraged (but
/// not required) to implement a ToArrayStream(ArrowArrayStream*) that creates a new
/// instance owned by the ArrowArrayStream and moves the relevant data to that instance.
///
/// An example implementation might be:
///
/// \code
/// class StreamImpl {
///  public:
///   // Public methods (e.g., constructor) used from C++ to initialize relevant data
///
///   // Idiomatic exporter to move data + lifecycle responsibility to an instance
///   // managed by the ArrowArrayStream callbacks
///   void ToArrayStream(struct ArrowArrayStream* out) {
///     ArrayStreamFactory<StreamImpl>::InitArrayStream(new StreamImpl(...), out);
///   }
///
///  private:
///   // Make relevant methods available to the ArrayStreamFactory
///   friend class ArrayStreamFactory<StreamImpl>;
///
///   // Method implementations (called from C, not normally interacted with from C++)
///   int GetSchema(struct ArrowSchema* schema) { return ENOTSUP; }
///   int GetNext(struct ArrowArray* array) { return ENOTSUP; }
///   const char* GetLastError() { nullptr; }
/// };
/// \endcode
///
/// An example usage might be:
///
/// \code
/// // Call constructor and/or public methods to initialize relevant data
/// StreamImpl impl;
///
/// // Export to ArrowArrayStream after data are finalized
/// UniqueArrayStream stream;
/// impl.ToArrayStream(stream.get());
/// \endcode
template <typename T>
class ArrayStreamFactory {
 public:
  /// \brief Take ownership of instance and populate callbacks of out
  static void InitArrayStream(T* instance, struct ArrowArrayStream* out) {
    out->get_schema = &get_schema_wrapper;
    out->get_next = &get_next_wrapper;
    out->get_last_error = &get_last_error_wrapper;
    out->release = &release_wrapper;
    out->private_data = instance;
  }

 private:
  static int get_schema_wrapper(struct ArrowArrayStream* stream,
                                struct ArrowSchema* schema) {
    return reinterpret_cast<T*>(stream->private_data)->GetSchema(schema);
  }

  static int get_next_wrapper(struct ArrowArrayStream* stream, struct ArrowArray* array) {
    return reinterpret_cast<T*>(stream->private_data)->GetNext(array);
  }

  static const char* get_last_error_wrapper(struct ArrowArrayStream* stream) {
    return reinterpret_cast<T*>(stream->private_data)->GetLastError();
  }

  static void release_wrapper(struct ArrowArrayStream* stream) {
    delete reinterpret_cast<T*>(stream->private_data);
    stream->release = nullptr;
    stream->private_data = nullptr;
  }
};

/// \brief An empty array stream
///
/// This class can be constructed from an struct ArrowSchema and implements a default
/// get_next() method that always marks the output ArrowArray as released.
class EmptyArrayStream {
 public:
  /// \brief Create an EmptyArrayStream from an ArrowSchema
  ///
  /// Takes ownership of schema.
  EmptyArrayStream(struct ArrowSchema* schema) : schema_(schema) {
    ArrowErrorInit(&error_);
  }

  /// \brief Export to ArrowArrayStream
  void ToArrayStream(struct ArrowArrayStream* out) {
    EmptyArrayStream* impl = new EmptyArrayStream(schema_.get());
    ArrayStreamFactory<EmptyArrayStream>::InitArrayStream(impl, out);
  }

 private:
  UniqueSchema schema_;
  struct ArrowError error_;

  friend class ArrayStreamFactory<EmptyArrayStream>;

  int GetSchema(struct ArrowSchema* schema) {
    return ArrowSchemaDeepCopy(schema_.get(), schema);
  }

  int GetNext(struct ArrowArray* array) {
    array->release = nullptr;
    return NANOARROW_OK;
  }

  const char* GetLastError() { return error_.message; }
};

/// \brief Implementation of an ArrowArrayStream backed by a vector of UniqueArray objects
class VectorArrayStream {
 public:
  /// \brief Create a VectorArrayStream from an ArrowSchema + vector of UniqueArray
  ///
  /// Takes ownership of schema and moves arrays if possible.
  VectorArrayStream(struct ArrowSchema* schema, std::vector<UniqueArray> arrays)
      : offset_(0), schema_(schema), arrays_(std::move(arrays)) {}

  /// \brief Create a one-shot VectorArrayStream from an ArrowSchema + ArrowArray
  ///
  /// Takes ownership of schema and array.
  VectorArrayStream(struct ArrowSchema* schema, struct ArrowArray* array)
      : offset_(0), schema_(schema) {
    arrays_.emplace_back(array);
  }

  /// \brief Export to ArrowArrayStream
  void ToArrayStream(struct ArrowArrayStream* out) {
    VectorArrayStream* impl = new VectorArrayStream(schema_.get(), std::move(arrays_));
    ArrayStreamFactory<VectorArrayStream>::InitArrayStream(impl, out);
  }

 private:
  int64_t offset_;
  UniqueSchema schema_;
  std::vector<UniqueArray> arrays_;

  friend class ArrayStreamFactory<VectorArrayStream>;

  int GetSchema(struct ArrowSchema* schema) {
    return ArrowSchemaDeepCopy(schema_.get(), schema);
  }

  int GetNext(struct ArrowArray* array) {
    if (offset_ < static_cast<int64_t>(arrays_.size())) {
      arrays_[offset_++].move(array);
    } else {
      array->release = nullptr;
    }

    return NANOARROW_OK;
  }

  const char* GetLastError() { return ""; }
};

/// @}

NANOARROW_CXX_NAMESPACE_END

#endif
