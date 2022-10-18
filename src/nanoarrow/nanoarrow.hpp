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

template<typename T>
class Unique {
public:
  Unique() = default;
  Unique(const Unique& rhs) = delete;

  explicit Unique(T* data) {
    memcpy(&data_, data, sizeof(T));
  }

  T* get() noexcept {
    return &data_;
  }

  T* operator->() {
    return &data_;
  }

protected:
  T data_;
};

template<typename T>
class UniqueReleaseable: public internal::Unique<T> {
 public:
  UniqueReleaseable() { this->data_.release = nullptr; }

  UniqueReleaseable(UniqueReleaseable&& rhs) {
    memcpy(this->get(), rhs.get(), sizeof(T));
    rhs->release = nullptr;
  }

  explicit UniqueReleaseable(T* data): Unique<T>(data) {
    data->release = nullptr;
  }

  void release() {
    this->data_.release(&this->data_);
  }

  ~UniqueReleaseable() {
    if (this->data_.release != nullptr) {
      release();
    }
  }
};

}

using UniqueArray = internal::UniqueReleaseable<struct ArrowArray>;
using UniqueSchema = internal::UniqueReleaseable<struct ArrowSchema>;
using UniqueArrayStream = internal::UniqueReleaseable<struct ArrowArrayStream>;

}  // namespace nanoarrow
