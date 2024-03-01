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

#ifndef R_NANOARROW_VCTR_BUILDER_RCRD_H_INCLUDED
#define R_NANOARROW_VCTR_BUILDER_RCRD_H_INCLUDED

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include <memory>
#include <vector>

#include "vctr_builder_base.h"

class RcrdBuilder : public VctrBuilder {
 public:
  explicit RcrdBuilder(SEXP ptype_sexp)
      : VctrBuilder(VECTOR_TYPE_DATA_FRAME, ptype_sexp) {}

  ArrowErrorCode Init(const ArrowSchema* schema, VctrBuilderOptions options,
                      ArrowError* error) override {
    // TODO: Check can convert here

    // Instantiate and initialize children
    children_.resize(schema->n_children);
    for (int64_t i = 0; i < schema->n_children; i++) {
      SEXP child_ptype_sexp;
      if (ptype_sexp_ != R_NilValue) {
        child_ptype_sexp = VECTOR_ELT(ptype_sexp_, i);
      } else {
        child_ptype_sexp = R_NilValue;
      }

      VctrBuilder* child = nullptr;
      NANOARROW_RETURN_NOT_OK(InstantiateBuilder(schema->children[i], child_ptype_sexp,
                                                 options, &child, error));
      children_[i].reset(child);
      NANOARROW_RETURN_NOT_OK(child->Init(schema->children[i], options, error));
    }

    schema_ = schema;
    return NANOARROW_OK;
  }

  SEXP GetPtype() override {
    if (ptype_sexp_ != R_NilValue) {
      return ptype_sexp_;
    }

    SEXP result = PROTECT(Rf_allocVector(VECSXP, schema_->n_children));
    SEXP result_names = PROTECT(Rf_allocVector(STRSXP, schema_->n_children));
    for (R_xlen_t i = 0; i < schema_->n_children; i++) {
      struct ArrowSchema* child = schema_->children[i];
      if (child->name != NULL) {
        SET_STRING_ELT(result_names, i, Rf_mkCharCE(child->name, CE_UTF8));
      } else {
        SET_STRING_ELT(result_names, i, Rf_mkChar(""));
      }

      SEXP child_sexp = PROTECT(children_[i]->GetPtype());
      SET_VECTOR_ELT(result, i, child_sexp);
      UNPROTECT(1);
    }

    Rf_setAttrib(result, R_ClassSymbol, nanoarrow_cls_data_frame);
    Rf_setAttrib(result, R_NamesSymbol, result_names);
    SEXP rownames = PROTECT(Rf_allocVector(INTSXP, 2));
    INTEGER(rownames)[0] = NA_INTEGER;
    INTEGER(rownames)[1] = 0;
    Rf_setAttrib(result, R_RowNamesSymbol, rownames);
    UNPROTECT(3);
    return result;
  }

 private:
  std::vector<std::unique_ptr<VctrBuilder>> children_;
};

#endif
