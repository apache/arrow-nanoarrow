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

#ifndef NANOARROW_HPP_SCHEMA_HPP_INCLUDED
#define NANOARROW_HPP_SCHEMA_HPP_INCLUDED

#include <utility>

#include "nanoarrow/hpp/exception.hpp"
#include "nanoarrow/hpp/unique.hpp"
#include "nanoarrow/hpp/view.hpp"
#include "nanoarrow/nanoarrow.h"

NANOARROW_CXX_NAMESPACE_BEGIN

template <typename StringT>
class ViewMetadata {
 public:
  explicit ViewMetadata(const char* metadata) : metadata_(metadata) {}

 private:
  const char* metadata_;

 public:
  class iterator {
    const ViewMetadata& outer_;
    ArrowMetadataReader reader_{};
    ArrowStringView key_{};
    ArrowStringView value_{};

   public:
    explicit iterator(const ViewMetadata& outer, int64_t remaining_keys) : outer_(outer) {
      if (remaining_keys != 0) {
        NANOARROW_THROW_NOT_OK(ArrowMetadataReaderInit(&reader_, outer.metadata_));
      }
    }

    iterator& operator++() {
      ArrowMetadataReaderRead(&reader_, &key_, &value_);
      return *this;
    }

    iterator operator++(int) {
      iterator retval = *this;
      ++(*this);
      return retval;
    }

    bool operator==(iterator other) const {
      return outer_.metadata_ == other.outer_.metadata_ &&
             reader_.remaining_keys == other.reader_.remaining_keys;
    }

    bool operator!=(iterator other) const { return !(*this == other); }

    std::pair<StringT, StringT> operator*() const {
      return {StringT{key_.data, key_.size_bytes},
              StringT{value_.data, value_.size_bytes}};
    }

    using iterator_category = std::forward_iterator_tag;
  };

  iterator begin() const { return iterator(*this); }
  iterator end() const { return iterator(*this, 0); }
};

class ViewSchemaChildren;

class ViewSchema {
 public:
  ViewSchema(const ArrowSchema* schema) : schema_{schema} {}

  template <typename StringT>
  ViewMetadata<StringT> Metadata() {
    return ViewMetadata<StringT>(schema_->metadata);
  }

  ViewSchemaChildren Children();

 private:
  const ArrowSchema* schema_;
};

class ViewSchemaChildren {
 public:
  explicit ViewSchemaChildren(const ArrowSchema* schema) : schema_(schema) {}

 private:
  const ArrowSchema* schema_{};

 public:
  class iterator {
    const ViewSchemaChildren& outer_;
    int64_t i_ = 0;

   public:
    explicit iterator(const ViewSchemaChildren& outer, int i = 0)
        : outer_(outer), i_(i) {}
    iterator& operator++() {
      i_++;
      return *this;
    }
    iterator operator++(int) {
      iterator retval = *this;
      ++(*this);
      return retval;
    }
    bool operator==(iterator other) const {
      return outer_.schema_ == other.outer_.schema_ && i_ == other.i_;
    }
    bool operator!=(iterator other) const { return !(*this == other); }
    ViewSchema operator*() const { return ViewSchema(outer_.schema_->children[i_]); }
    using iterator_category = std::forward_iterator_tag;
  };

  iterator begin() const { return iterator(*this); }
  iterator end() const { return iterator(*this, schema_->n_children); }
};

inline ViewSchemaChildren ViewSchema::Children() { return ViewSchemaChildren(schema_); }



NANOARROW_CXX_NAMESPACE_END

#endif
