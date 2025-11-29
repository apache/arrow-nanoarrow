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
      : schema_(nullptr),
        vector_type_(vector_type),
        ptype_sexp_(R_NilValue),
        value_(R_NilValue),
        value_size_(0) {
    ArrowArrayViewInitFromType(&array_view_, NANOARROW_TYPE_UNINITIALIZED);
    nanoarrow_preserve_sexp(ptype_sexp);
    ptype_sexp_ = ptype_sexp;
  }

  // Enable generic containers like std::unique_ptr<VctrBuilder>
  virtual ~VctrBuilder() {
    nanoarrow_release_sexp(ptype_sexp_);
    nanoarrow_release_sexp(value_);
    ArrowArrayViewReset(&array_view_);
  }

  // Initialize this instance with the information available to the resolver, or the
  // information that was inferred. If using the default `to`, ptype may be R_NilValue
  // with Options containing the inferred information. Calling this method may longjmp.
  // The implementation on the base class initialized the built-in ArrowArrayView and
  // saves a reference to `schema` (but subclass implementations need not call it).
  virtual ArrowErrorCode Init(const ArrowSchema* schema, VctrBuilderOptions options,
                              ArrowError* error) {
    NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&schema_view_, schema, error));
    NANOARROW_RETURN_NOT_OK(ArrowArrayViewInitFromSchema(&array_view_, schema, error));
    schema_ = schema;
    return NANOARROW_OK;
  }

  virtual ArrowErrorCode Reserve(R_xlen_t n, ArrowError* error) {
    if (value_ != R_NilValue) {
      ArrowErrorSet(error, "VctrBuilder reallocation is not implemented");
    }

    return NANOARROW_OK;
  }

  // Push an array into this builder and do not take ownership of array. This is
  // called when the caller cannot safely relinquish ownership of an array (e.g.,
  // convert_array()). Calling this method may longjmp.
  virtual ArrowErrorCode PushNext(SEXP array_shelter, const ArrowArray* array,
                                  ArrowError* error) {
    NANOARROW_RETURN_NOT_OK(ArrowArrayViewSetArray(&array_view_, array, error));
    return NANOARROW_OK;
  }

  // Push an array into this builder. The implementation may (but is not required) to take
  // ownership. This is called when the caller can relinquish ownership (e.g.,
  // convert_array_stream()). Calling this method may longjmp.
  virtual ArrowErrorCode PushNextOwning(ArrowArray* array, ArrowError* error) {
    return PushNext(R_NilValue, array, error);
  }

  // Perform any final calculations required to calculate the return value.
  // Calling this method may longjmp.
  virtual ArrowErrorCode Finish(ArrowError* error) {
    if (ptype_sexp_ != R_NilValue && value_ != R_NilValue) {
      Rf_copyMostAttrib(ptype_sexp_, value_);
    }

    return NANOARROW_OK;
  }

  // Release the final value of the builder. Calling this method may longjmp.
  virtual SEXP GetValue() {
    SEXP value = PROTECT(value_);
    nanoarrow_release_sexp(value_);
    value_ = R_NilValue;
    UNPROTECT(1);
    return value;
  }

  // Get (or allocate if required) the SEXP ptype for this output
  virtual SEXP GetPtype() { return ptype_sexp_; }

 protected:
  ArrowSchemaView schema_view_;
  ArrowArrayView array_view_;
  const ArrowSchema* schema_;
  VectorType vector_type_;
  SEXP ptype_sexp_;
  SEXP value_;
  R_xlen_t value_size_;

  // Could maybe avoid a preserve/protect
  void SetValue(SEXP value) {
    nanoarrow_release_sexp(value_);
    value_ = value;
    nanoarrow_preserve_sexp(value_);
  }

  ArrowErrorCode WarnLossyConvert(const char* msg, int64_t count) {
    SEXP fun = PROTECT(Rf_install("warn_lossy_conversion"));
    SEXP count_sexp = PROTECT(Rf_ScalarReal((double)count));
    SEXP msg_sexp = PROTECT(Rf_mkString(msg));
    SEXP call = PROTECT(Rf_lang3(fun, count_sexp, msg_sexp));
    Rf_eval(call, nanoarrow_ns_pkg);
    UNPROTECT(4);
    return NANOARROW_OK;
  }

  void StopCantConvert() {
    SEXP fun = PROTECT(Rf_install("stop_cant_convert_schema"));
    SEXP schema_xptr = PROTECT(
        R_MakeExternalPtr(const_cast<ArrowSchema*>(schema_), R_NilValue, R_NilValue));
    Rf_setAttrib(schema_xptr, R_ClassSymbol, nanoarrow_cls_schema);
    SEXP ptype_sexp = PROTECT(GetPtype());

    SEXP call = PROTECT(Rf_lang3(fun, schema_xptr, ptype_sexp));
    Rf_eval(call, nanoarrow_ns_pkg);
    UNPROTECT(4);
  }
};

// Resolve a builder class from a schema and (optional) ptype and instantiate it
ArrowErrorCode InstantiateBuilder(const ArrowSchema* schema, SEXP ptype_sexp,
                                  VctrBuilderOptions options, VctrBuilder** out,
                                  ArrowError* error);

#endif
