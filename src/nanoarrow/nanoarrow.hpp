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

namespace nanoarrow {

namespace internal {

template <typename T>
class Unique {
 public:
  Unique() = default;
  Unique(const Unique& rhs) = delete;

  explicit Unique(T* data) { memcpy(&data_, data, sizeof(T)); }

  T* get() noexcept { return &data_; }

  T* operator->() { return &data_; }

 protected:
  T data_;
};

template <typename T>
class UniqueReleaseable : public internal::Unique<T> {
 public:
  UniqueReleaseable() { this->data_.release = nullptr; }

  UniqueReleaseable(UniqueReleaseable&& rhs) { rhs.move(this->get()); }

  explicit UniqueReleaseable(T* data) : Unique<T>(data) { data->release = nullptr; }

  void release() { this->data_.release(&this->data_); }

  void reset() {
    if (this->data_.release != nullptr) {
      release();
    }
  }

  void reset(T* data) {
    reset();
    memcpy(this->get(), data, sizeof(T));
    data->release = nullptr;
  }

  void swap(UniqueReleaseable& rhs) {
    UniqueReleaseable temp(std::move(rhs));
    rhs.reset(this->get());
    this->reset(temp.get());
  }

  void move(T* out) {
    memcpy(out, this->get(), sizeof(T));
    this->data_.release = nullptr;
  }

  ~UniqueReleaseable() {
    reset();
  }
};

}  // namespace internal

using UniqueArray = internal::UniqueReleaseable<struct ArrowArray>;
using UniqueSchema = internal::UniqueReleaseable<struct ArrowSchema>;

class UniqueArrayStream : public internal::UniqueReleaseable<struct ArrowArrayStream> {
 public:
  UniqueArrayStream() = default;

  explicit UniqueArrayStream(struct ArrowArrayStream* data)
      : internal::UniqueReleaseable<struct ArrowArrayStream>(data) {}

  int get_schema(struct ArrowSchema* schema) {
    return this->data_.get_schema(&this->data_, schema);
  }

  int get_next(struct ArrowArray* array) {
    return this->data_.get_next(&this->data_, array);
  }

  const char* get_last_error() { return this->data_.get_last_error(&this->data_); }
};

class EmptyArrayStream {
 public:
  static UniqueArrayStream MakeUnique(enum ArrowType type) {
    UniqueArrayStream stream;
    (new EmptyArrayStream(type))->MakeStream(stream.get());
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

class VectorArrayStream : public EmptyArrayStream {
 public:
  VectorArrayStream(struct ArrowSchema* schema) : EmptyArrayStream(schema), offset_(0) {}
  VectorArrayStream(struct ArrowSchema* schema, struct ArrowArray* array)
      : EmptyArrayStream(schema), offset_(0) {
    arrays_.emplace_back(array);
  }
  VectorArrayStream(struct ArrowSchema* schema, std::vector<UniqueArray> arrays)
      : EmptyArrayStream(schema), offset_(0), arrays_(std::move(arrays)) {}

 protected:
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

}  // namespace nanoarrow
