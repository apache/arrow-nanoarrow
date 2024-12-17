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

#include <optional>
#include <utility>
#include <vector>

#include "nanoarrow/hpp/exception.hpp"
#include "nanoarrow/hpp/unique.hpp"
#include "nanoarrow/hpp/view.hpp"
#include "nanoarrow/nanoarrow.h"

NANOARROW_CXX_NAMESPACE_BEGIN

class SchemaBuilder {
 public:
  // Let some implicit magic construction happen
  SchemaBuilder() { ArrowSchemaInit(schema_.get()); }
  SchemaBuilder(const ArrowSchema* schema) {
    NANOARROW_THROW_NOT_OK(ArrowSchemaDeepCopy(schema, schema_.get()));
  }
  SchemaBuilder(ArrowType type) : SchemaBuilder() { set_type(type); }
  SchemaBuilder(const UniqueSchema schema) : SchemaBuilder(schema.get()) {}

  // Movable
  SchemaBuilder(SchemaBuilder&& rhs) : SchemaBuilder(std::move(rhs.schema_)) {}
  SchemaBuilder& operator=(SchemaBuilder&& rhs) {
    schema_ = std::move(rhs.schema_);
    return *this;
  }
  // Copyable
  SchemaBuilder(const SchemaBuilder& rhs) : SchemaBuilder(rhs.data()) {}

  SchemaBuilder& operator=(SchemaBuilder& rhs) {
    NANOARROW_THROW_NOT_OK(ArrowSchemaDeepCopy(rhs.data(), schema_.get()));
    return *this;
  }

  // Implicitly convertable to const ArrowSchema
  operator const ArrowSchema*() const { return schema_.get(); }

  // Get schema pointer
  const ArrowSchema* data() const { return schema_.get(); }
  ArrowSchema* data() { return schema_.get(); }

  // Move the schema out
  void Export(ArrowSchema* out) { ArrowSchemaMove(schema_.get(), out); }

  SchemaBuilder& set_type(ArrowType type) {
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetType(schema_.get(), type));
    return *this;
  }

  SchemaBuilder& set_type_datetime(ArrowType type, ArrowTimeUnit time_unit,
                                   const char* tz = "") {
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeDateTime(data(), type, time_unit, tz));
    return *this;
  }

  SchemaBuilder& set_type_list(ArrowType type, SchemaBuilder child,
                               int32_t fixed_size = -1) {
    if (fixed_size > 0) {
      NANOARROW_THROW_NOT_OK(
          ArrowSchemaSetTypeFixedSize(schema_.get(), type, fixed_size));
    } else {
      set_type(type);
    }

    child.set_name(data()->children[0]->name);
    ArrowSchemaRelease(data()->children[0]);
    child.Export(data()->children[0]);
    return *this;
  }

  SchemaBuilder& set_name(const char* name) {
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema_.get(), name));
    return *this;
  }

  SchemaBuilder& allocate_children(int64_t n_children) {
    NANOARROW_THROW_NOT_OK(ArrowSchemaAllocateChildren(schema_.get(), n_children));
    return *this;
  }

  SchemaBuilder& set_child(int64_t i, SchemaBuilder child) {
    if (data()->children[i]->release) {
      ArrowSchemaRelease(data()->children[i]);
    }

    child.Export(data()->children[i]);
    return *this;
  }

  SchemaBuilder& set_child(int64_t i, SchemaBuilder child, const char* name) {
    child.set_name(name);
    if (data()->children[i]->release) {
      ArrowSchemaRelease(data()->children[i]);
    }

    child.Export(data()->children[i]);
    return *this;
  }

 private:
  UniqueSchema schema_;
};

namespace schema {

SchemaBuilder int32() { return NANOARROW_TYPE_INT32; }

SchemaBuilder string() { return NANOARROW_TYPE_STRING; }

SchemaBuilder list(SchemaBuilder child) {
  SchemaBuilder out;
  out.set_type_list(NANOARROW_TYPE_LIST, std::move(child));
  return out;
}

SchemaBuilder fixed_size_list(SchemaBuilder child) {
  SchemaBuilder out;
  out.set_type_list(NANOARROW_TYPE_LIST, std::move(child));
  return out;
}

SchemaBuilder struct_(std::vector<SchemaBuilder> children) {
  SchemaBuilder out(NANOARROW_TYPE_STRUCT);
  out.allocate_children(static_cast<int64_t>(children.size()));
  for (int64_t i = 0; i < static_cast<int64_t>(children.size()); i++) {
    out.set_child(i, std::move(children[i]));
  }

  return out;
}

SchemaBuilder struct_(std::vector<std::pair<std::string, SchemaBuilder>> children) {
  SchemaBuilder out(NANOARROW_TYPE_STRUCT);
  out.allocate_children(static_cast<int64_t>(children.size()));
  for (int64_t i = 0; i < static_cast<int64_t>(children.size()); i++) {
    auto child = std::move(children[i]);
    out.set_child(i, std::move(child.second), child.first.c_str());
  }

  return out;
}

}  // namespace schema

class ViewMetadata {
 public:
  explicit ViewMetadata(const char* metadata) : metadata_(metadata) {}

  int64_t size() {
    if (metadata_ == nullptr) {
      return 0;
    }

    return end() - begin();
  }

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
      NANOARROW_THROW_NOT_OK(ArrowMetadataReaderRead(&reader_, &key_, &value_));
      return *this;
    }

    iterator operator++(int) {
      iterator retval = *this;
      ++(*this);
      return retval;
    }

    int64_t operator-(iterator other) const {
      return reader_.remaining_keys - other.reader_.remaining_keys;
    }

    bool operator==(iterator other) const {
      return outer_.metadata_ == other.outer_.metadata_ &&
             reader_.remaining_keys == other.reader_.remaining_keys;
    }

    bool operator!=(iterator other) const { return !(*this == other); }

    std::pair<std::string_view, std::string_view> operator*() const {
      return {{key_.data, static_cast<size_t>(key_.size_bytes)},
              {value_.data, static_cast<size_t>(value_.size_bytes)}};
    }

    using iterator_category = std::forward_iterator_tag;
  };

  iterator begin() const { return iterator(*this, -1); }
  iterator end() const { return iterator(*this, 0); }
};

class ViewSchemaChildren;

class ViewSchema {
 public:
  ViewSchema(const ArrowSchema* schema) : schema_{schema} {
    // Probably need to do something better with this error here
    NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&schema_view_, schema_, nullptr));
  }

  std::string_view format() const {
    if (schema_->name) {
      return "";
    } else {
      return schema_->name;
    }
  }

  std::string_view name() const {
    if (schema_->name) {
      return "";
    } else {
      return schema_->name;
    }
  }

  ViewMetadata metadata() const { return ViewMetadata(schema_->metadata); }

  ViewSchemaChildren children() const;

  std::optional<ViewSchema> dictionary() const {
    if (schema_->dictionary) {
      return ViewSchema(schema_->dictionary);
    } else {
      return std::nullopt;
    }
  }

  bool is_extension() const { return schema_view_.extension_name.size_bytes > 0; }

  ArrowType type() const { return schema_view_.type; }

  ArrowType storage_type() { return schema_view_.storage_type; }

 private:
  const ArrowSchema* schema_;
  ArrowSchemaView schema_view_{};
};

class ViewSchemaChildren {
 public:
  explicit ViewSchemaChildren(const ArrowSchema* schema) : schema_(schema) {}

  int64_t size() const { return schema_->n_children; }

 private:
  const ArrowSchema* schema_{};

 public:
  class iterator {
    const ViewSchemaChildren& outer_;
    int64_t i_ = 0;

   public:
    explicit iterator(const ViewSchemaChildren& outer, int64_t i = 0)
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

inline ViewSchemaChildren ViewSchema::children() const {
  return ViewSchemaChildren(schema_);
}

NANOARROW_CXX_NAMESPACE_END

#endif
