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

#ifndef R_NANOARROW_VCTR_BUILDER_BASE_H_INCLUDED
#define R_NANOARROW_VCTR_BUILDER_BASE_H_INCLUDED

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "materialize.h"
#include "nanoarrow.h"
#include "preserve.h"
#include "vctr_builder.h"

struct VctrBuilder {
 public:
  // VctrBuilder instances are always created from a vector_type or a ptype.
  // InstantiateBuilder() takes care of picking which subclass. The base class
  // constructor takes these two arguments to provide consumer implementations
  // for inspecting their value. This does not validate any ptypes (that would
  // happen in Init() if needed).
  VctrBuilder(VectorType vector_type, SEXP ptype_sexp)
      : vector_type_(vector_type), ptype_sexp_(R_NilValue), value_(R_NilValue) {
    nanoarrow_preserve_sexp(ptype_sexp);
    ptype_sexp_ = ptype_sexp;
  }

  // Enable generic containers like std::unique_ptr<VctrBuilder>
  virtual ~VctrBuilder() {
    nanoarrow_release_sexp(ptype_sexp_);
    nanoarrow_release_sexp(value_);
  }

  // Initialize this instance with the information available to the resolver, or the
  // information that was inferred. If using the default `to`, ptype may be R_NilValue
  // with Options containing the inferred information. Calling this method may longjmp.
  virtual ArrowErrorCode Init(const ArrowSchema* schema, VctrBuilderOptions options,
                              ArrowError* error) {
    schema_ = schema;
    return NANOARROW_OK;
  }

  // Push an array into this builder and do not take ownership of array. This is
  // called when the caller cannot safely relinquish ownership of an array (e.g.,
  // convert_array()). Calling this method may longjmp.
  virtual ArrowErrorCode PushNext(const ArrowArray* array, ArrowError* error) {
    return ENOTSUP;
  }

  // Push an array into this builder. The implementation may (but is not required) to take
  // ownership. This is called when the caller can relinquish ownership (e.g.,
  // convert_array_stream()). Calling this method may longjmp.
  virtual ArrowErrorCode PushNextOwning(ArrowArray* array, ArrowError* error) {
    return PushNext(array, error);
  }

  // Perform any final calculations required to calculate the return value.
  // Calling this method may longjmp.
  virtual ArrowErrorCode Finish(ArrowError* error) { return NANOARROW_OK; }

  // Release the final value of the builder. Calling this method may longjmp.
  virtual SEXP GetValue() {
    nanoarrow_release_sexp(value_);
    value_ = R_NilValue;
    return value_;
  }

  // Get (or allocate if required) the SEXP ptype for this output
  virtual SEXP GetPtype() { return ptype_sexp_; }

 protected:
  VectorType vector_type_;
  SEXP ptype_sexp_;
  SEXP value_;
  const ArrowSchema* schema_;
};

// Resolve a builder class from a schema and (optional) ptype and instantiate it
ArrowErrorCode InstantiateBuilder(const ArrowSchema* schema, SEXP ptype_sexp,
                                  VctrBuilderOptions options, VctrBuilder** out,
                                  ArrowError* error);

#endif
