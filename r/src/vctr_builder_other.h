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

#ifndef R_NANOARROW_VCTR_BUILDER_OTHER_H_INCLUDED
#define R_NANOARROW_VCTR_BUILDER_OTHER_H_INCLUDED

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "vctr_builder_base.h"

// If we've ended up here, we need to call in to R to convert this stream
// of arrays into an R vector. Currently, the S3 generic that implements
// this is convert_array(), so we have to do this one array at a time.
// The current conversions that are implemented this way internally are
// factor(), decimal, and + extension types/dictionary.
//
// An early version of this reimplemented a good chunk of vctrs-like internals
// to allow a generic preallocate where each chunk would be copied in to the
// preallocated vector. This version just converts each chunk as it comes
// and calls c(); however, eventually the generic should be
// convert_array_stream() to give implementations in other packages the ability
// to handle converting more than one array at a time.
class OtherBuilder : public VctrBuilder {
 public:
  explicit OtherBuilder(SEXP ptype_sexp)
      : VctrBuilder(VECTOR_TYPE_OTHER, ptype_sexp),
        chunks_sexp_(R_NilValue),
        chunks_tail_(R_NilValue) {}

  ~OtherBuilder() { nanoarrow_release_sexp(chunks_sexp_); }

  ArrowErrorCode Reserve(R_xlen_t n, ArrowError* error) override { return NANOARROW_OK; }

  ArrowErrorCode PushNext(SEXP array_shelter, const ArrowArray* array,
                          ArrowError* error) override {
    SEXP schema_borrowed_xptr = PROTECT(
        R_MakeExternalPtr(const_cast<ArrowSchema*>(schema_), R_NilValue, R_NilValue));
    Rf_setAttrib(schema_borrowed_xptr, R_ClassSymbol, nanoarrow_cls_schema);

    SEXP array_borrowed_xptr = PROTECT(R_MakeExternalPtr(
        const_cast<ArrowArray*>(array), schema_borrowed_xptr, array_shelter));
    Rf_setAttrib(array_borrowed_xptr, R_ClassSymbol, nanoarrow_cls_array);

    SEXP fun = PROTECT(Rf_install("convert_fallback_other"));
    SEXP call =
        PROTECT(Rf_lang5(fun, array_borrowed_xptr, R_NilValue, R_NilValue, ptype_sexp_));
    SEXP chunk_sexp = PROTECT(Rf_eval(call, nanoarrow_ns_pkg));
    Append(chunk_sexp);
    UNPROTECT(5);

    return NANOARROW_OK;
  }

  ArrowErrorCode Finish(ArrowError* error) override {
    if (chunks_tail_ == chunks_sexp_) {
      // Zero chunks (return the ptype)
      // Probably need to ensure the ptype has zero elements
      SetValue(ptype_sexp_);

    } else if (chunks_tail_ == CDR(chunks_sexp_)) {
      // One chunk (return the chunk)
      SetValue(CAR(chunks_tail_));

    } else {
      // Many chunks (concatenate or rbind)
      SEXP fun;
      if (Rf_inherits(ptype_sexp_, "data.frame")) {
        fun = PROTECT(Rf_install("rbind"));
      } else {
        fun = PROTECT(Rf_install("c"));
      }

      SETCAR(chunks_sexp_, fun);
      UNPROTECT(1);

      SEXP result = PROTECT(Rf_eval(chunks_sexp_, R_BaseEnv));
      SetValue(result);
      UNPROTECT(1);
    }

    nanoarrow_release_sexp(chunks_sexp_);
    chunks_sexp_ = R_NilValue;
    chunks_tail_ = R_NilValue;
    return NANOARROW_OK;
  }

 private:
  SEXP chunks_sexp_;
  SEXP chunks_tail_;

  void Append(SEXP chunk_sexp) {
    if (chunks_sexp_ == R_NilValue) {
      // Not sure if we will need no function, c, or rbind when we
      // create this, so leave it as R_NilValue for now.
      SEXP chunks_init = PROTECT(Rf_lang1(R_NilValue));
      chunks_sexp_ = chunks_init;
      nanoarrow_preserve_sexp(chunks_sexp_);
      chunks_tail_ = chunks_sexp_;
      UNPROTECT(1);
    }

    SEXP next_sexp = PROTECT(Rf_lcons(chunk_sexp, R_NilValue));
    SETCDR(chunks_tail_, next_sexp);
    UNPROTECT(1);
    chunks_tail_ = next_sexp;
  }
};

#endif
