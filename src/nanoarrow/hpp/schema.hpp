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
#include "nanoarrow/nanoarrow.h"

NANOARROW_CXX_NAMESPACE_BEGIN

class Schema {
 public:
  explicit Schema(UniqueSchema schema) : schema_(std::move(schema)) {}

  // Make conversion from a raw pointer explicit, including copies
  static Schema Copy(const ArrowSchema* schema) {
    Schema out;
    NANOARROW_THROW_NOT_OK(ArrowSchemaDeepCopy(schema, out.schema_.get()));
    return out;
  }

  static Schema Move(ArrowSchema* schema) {
    Schema out;
    out.schema_.reset(schema);
    return out;
  }

  // Movable
  Schema(Schema&& rhs) : Schema(std::move(rhs.schema_)) {}
  Schema& operator=(Schema&& rhs) {
    schema_ = std::move(rhs.schema_);
    return *this;
  }
  // Not copyable
  Schema(const Schema& rhs) = delete;

  // Implicitly convertable to const ArrowSchema
  const ArrowSchema* data() const { return schema_.get(); }
  operator const ArrowSchema*() const { return schema_.get(); }

  bool IsValid() const { return schema_->release != nullptr; }

  int64_t NumChildren() const {
    NANOARROW_DCHECK(IsValid());
    return schema_->n_children;
  }

  Schema Child(int64_t i) const {
    NANOARROW_DCHECK(IsValid() && i < schema_->n_children && i > 0);
    return Schema::Copy(schema_->children[i]);
  }

  Schema Dictionary() {
    NANOARROW_DCHECK(IsValid() && schema_->dictionary != nullptr);
    return Schema::Copy(schema_->dictionary);
  }

 private:
  UniqueSchema schema_;

  Schema() = default;
};

class SchemaBuilder {
 public:
  SchemaBuilder() { ArrowSchemaInit(schema_.get()); }
  SchemaBuilder(const ArrowSchema* schema) {
    NANOARROW_THROW_NOT_OK(ArrowSchemaDeepCopy(schema, schema_.get()));
  }

  SchemaBuilder(const UniqueSchema schema) : SchemaBuilder(schema.get()) {}

  // Movable
  SchemaBuilder(SchemaBuilder&& rhs) : SchemaBuilder(std::move(rhs.schema_)) {}
  SchemaBuilder& operator=(SchemaBuilder&& rhs) {
    schema_ = std::move(rhs.schema_);
    return *this;
  }
  // Not copyable
  SchemaBuilder(const SchemaBuilder& rhs) = delete;

 private:
  UniqueSchema schema_;
};

NANOARROW_CXX_NAMESPACE_END

#endif
