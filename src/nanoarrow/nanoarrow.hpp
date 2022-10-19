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

#include "nanoarrow.h"

#ifndef NANOARROW_HPP_INCLUDED
#define NANOARROW_HPP_INCLUDED

/// \defgroup nanoarrow_hpp
///
/// The utilities provided in this file are intended to support C++ users
/// of the nanoarrow C library such that C++-style resource allocation
/// and error handling can be used with nanoarrow data structures.
/// These utilities are not intended to mirror the nanoarrow C API.

namespace nanoarrow {

namespace internal {

/// \defgroup nanoarrow_hpp-unique_base Base classes for Unique wrappers
///
/// @{

/// \brief A unique_ptr-like base class for stack-allocatable objects
/// \tparam T The object type
template <typename T>
class Unique {
 public:
  /// \brief Get a pointer to the data owned by this object
  T* get() noexcept { return &data_; }

  /// \brief Use the pointer operator to access the fields of this object
  T* operator->() { return &data_; }

 protected:
  T data_;
};

/// \brief Base class for objects that can be
/// \tparam T A struct ArrowSchema, a struct ArrowArray, or struct ArrowArrayStream.
template <typename T>
class UniqueReleaseable : public internal::Unique<T> {
 public:
  /// \brief Construct an object marked as invalid via a release callback set to nullptr
  UniqueReleaseable() { this->data_.release = nullptr; }

  /// \brief Move ownership of the object wrapped by rhs to this object
  UniqueReleaseable(UniqueReleaseable&& rhs) { rhs.move(this->get()); }

  /// \brief Move ownership of the data pointed to by data to this object
  explicit UniqueReleaseable(T* data) : UniqueReleaseable() { reset(data); }

  /// \brief Call data's release callback
  void release() { this->data_.release(&this->data_); }

  /// \brief Call data's release callback if valid
  void reset() {
    if (this->data_.release != nullptr) {
      release();
    }
  }

  /// \brief Call data's release callback if valid and move ownership of the data
  /// pointed to by data
  void reset(T* data) {
    reset();
    memcpy(this->get(), data, sizeof(T));
    data->release = nullptr;
  }

  /// \brief Move ownership of this object to rhs and move ownership of rhs to this object
  void swap(UniqueReleaseable& rhs) {
    UniqueReleaseable temp(std::move(rhs));
    rhs.reset(this->get());
    this->reset(temp.get());
  }

  /// \brief Move ownership of this object to the data pointed to by out
  void move(T* out) {
    memcpy(out, this->get(), sizeof(T));
    this->data_.release = nullptr;
  }

  ~UniqueReleaseable() { reset(); }
};

/// @}

}  // namespace internal

/// \defgroup nanoarrow_hpp-unique
///
/// The Arrow C Data interface, the Arrow C Stream interface, and the
/// nanoarrow C library use stack-allocatable objects, some of which
/// require initialization or cleanup.
///
/// @{

/// \brief Class wrapping a unique struct ArrowSchema
using UniqueSchema = internal::UniqueReleaseable<struct ArrowSchema>;

/// \brief Class wrapping a unique struct ArrowArray
using UniqueArray = internal::UniqueReleaseable<struct ArrowArray>;

/// \brief Class wrapping a unique struct ArrowArrayStream
class UniqueArrayStream : public internal::UniqueReleaseable<struct ArrowArrayStream> {
 public:
  /// \brief Construct an object marked as invalid via a release callback set to nullptr
  UniqueArrayStream() = default;

  /// \brief Move ownership of the object wrapped by rhs to this object
  UniqueArrayStream(UniqueArrayStream&& rhs) { rhs.move(this->get()); }

  /// \brief Move ownership of the data pointed to by data to this object
  explicit UniqueArrayStream(struct ArrowArrayStream* data)
      : internal::UniqueReleaseable<struct ArrowArrayStream>(data) {}

  /// \brief Call the struct ArrowArrayStream's get_schema() method
  int get_schema(struct ArrowSchema* schema) {
    return this->data_.get_schema(&this->data_, schema);
  }

  /// \brief Call the struct ArrowArrayStream's get_next() method
  int get_next(struct ArrowArray* array) {
    return this->data_.get_next(&this->data_, array);
  }

  /// \brief Call the struct ArrowArrayStream's get_last_error() method
  const char* get_last_error() { return this->data_.get_last_error(&this->data_); }
};

/// @}

/// \defgroup nanoarrow_hpp-array-stream ArrayStream helpers
///
/// These classes provide simple struct ArrowArrayStream implementations that
/// can be extended to help simplify the process of creating a valid
/// ArrowArrayStream implementation or used as-is for testing.
///
/// @{

/// \brief An empty array stream
///
/// This class can be constructed from an enum ArrowType or
/// struct ArrowSchema and implements a default get_next() method that
/// always marks the output ArrowArray as released. This class can
/// be extended with an implementation of get_next() for a custom
/// source.
class EmptyArrayStream {
 public:
  /// \brief Create an empty UniqueArrayStream from a struct ArrowSchema
  ///
  /// This object takes ownership of the schema and marks the source schema
  /// as released.
  static UniqueArrayStream MakeUnique(struct ArrowSchema* schema) {
    UniqueArrayStream stream;
    (new EmptyArrayStream(schema))->MakeStream(stream.get());
    return stream;
  }

 protected:
  UniqueSchema schema_;
  struct ArrowError error_;

  EmptyArrayStream(struct ArrowSchema* schema) : schema_(schema) {
    error_.message[0] = '\0';
  }

  EmptyArrayStream(enum ArrowType type) {
    if (ArrowSchemaInit(schema_.get(), type) != NANOARROW_OK) {
      throw std::bad_alloc();
    }
    error_.message[0] = '\0';
  }

  void MakeStream(struct ArrowArrayStream* stream) {
    stream->get_schema = &get_schema_wrapper;
    stream->get_next = &get_next_wrapper;
    stream->get_last_error = &get_last_error_wrapper;
    stream->release = &release_wrapper;
    stream->private_data = this;
  }

  virtual int get_schema(struct ArrowSchema* schema) {
    return ArrowSchemaDeepCopy(schema_.get(), schema);
  }

  virtual int get_next(struct ArrowArray* array) {
    array->release = nullptr;
    return NANOARROW_OK;
  }

  virtual const char* get_last_error() { return error_.message; }

 private:
  static int get_schema_wrapper(struct ArrowArrayStream* stream,
                                struct ArrowSchema* schema) {
    return reinterpret_cast<EmptyArrayStream*>(stream->private_data)->get_schema(schema);
  }

  static int get_next_wrapper(struct ArrowArrayStream* stream, struct ArrowArray* array) {
    return reinterpret_cast<EmptyArrayStream*>(stream->private_data)->get_next(array);
  }

  static const char* get_last_error_wrapper(struct ArrowArrayStream* stream) {
    return reinterpret_cast<EmptyArrayStream*>(stream->private_data)->get_last_error();
  }

  static void release_wrapper(struct ArrowArrayStream* stream) {
    delete reinterpret_cast<EmptyArrayStream*>(stream->private_data);
  }
};

/// \brief Implementation of an ArrowArrayStream backed by a vector of ArrowArray objects
class VectorArrayStream : public EmptyArrayStream {
 public:
  /// \brief Create a UniqueArrowArrayStream from an existing array
  ///
  /// Takes ownership of the schema and the array.
  static UniqueArrayStream MakeUnique(struct ArrowSchema* schema,
                                      struct ArrowArray* array) {
    UniqueArrayStream stream;
    (new VectorArrayStream(schema, array))->MakeStream(stream.get());
    return stream;
  }

  /// \brief Create a UniqueArrowArrayStream from existing arrays
  ///
  /// This object takes ownership of the schema and arrays.
  static UniqueArrayStream MakeUnique(struct ArrowSchema* schema,
                                      std::vector<UniqueArray> arrays) {
    UniqueArrayStream stream;
    (new VectorArrayStream(schema, std::move(arrays)))->MakeStream(stream.get());
    return stream;
  }

 protected:
  VectorArrayStream(struct ArrowSchema* schema, struct ArrowArray* array)
      : EmptyArrayStream(schema), offset_(0) {
    arrays_.emplace_back(array);
  }

  VectorArrayStream(struct ArrowSchema* schema, std::vector<UniqueArray> arrays)
      : EmptyArrayStream(schema), offset_(0), arrays_(std::move(arrays)) {}

  int get_next(struct ArrowArray* array) {
    if (offset_ < arrays_.size()) {
      arrays_[offset_++].move(array);
    } else {
      array->release = nullptr;
    }

    return NANOARROW_OK;
  }

 private:
  std::vector<UniqueArray> arrays_;
  int64_t offset_;
};

/// @}

}  // namespace nanoarrow

#endif
