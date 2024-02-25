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

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "materialize.h"
#include "nanoarrow.h"
#include "nanoarrow/r.h"

class VctrBuilder {
 public:
  class Options {
   public:
    int64_t num_items;
    int use_altrep;
  };

  // If a ptype is supplied to a VctrBuilder, it must be supplied at construction
  // and preserved until the value is no longer needed.

  // Enable generic containers like std::vector<std::unique_ptr<VctrBuilder>>
  virtual ~VctrBuilder() {}

  // Initialize this instance with the information available to the resolver, or the
  // information that was inferred. If using the default `to`, ptype may be R_NilValue
  // with Options containing the inferred information. Calling this method may longjmp.
  virtual ArrowErrorCode Init(const ArrowSchema* schema, const Options& options,
                              ArrowError* error) {
    return ENOTSUP;
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
  virtual ArrowErrorCode Finish(ArrowError* error) { return ENOTSUP; }

  // Extract the final value of the builder. Calling this method may longjmp.
  virtual SEXP GetValue() { return R_NilValue; }
};

class IntBuilder : public VctrBuilder {};
class DblBuilder : public VctrBuilder {};
class ChrBuilder : public VctrBuilder {};
class LglBuilder : public VctrBuilder {};
class RcrdBuilder : public VctrBuilder {
 public:
  explicit RcrdBuilder(SEXP ptype_sexp) {}
};
class UnspecifiedBuilder : public VctrBuilder {};
class BlobBuilder : public VctrBuilder {};
class ListOfBuilder : public VctrBuilder {};
class DateBuilder : public VctrBuilder {};
class HmsBuilder : public VctrBuilder {};
class PosixctBuilder : public VctrBuilder {};
class DifftimeBuilder : public VctrBuilder {};
class Integer64Builder : public VctrBuilder {};

class ExtensionBuilder : public VctrBuilder {
 public:
  explicit ExtensionBuilder(SEXP ptype_sexp) {}
};

// Currently in infer_ptype.c
extern "C" enum VectorType nanoarrow_infer_vector_type(enum ArrowType type);
extern "C" SEXP nanoarrow_c_infer_ptype(SEXP schema_xptr);

// Resolve a builder class
ArrowErrorCode InstantiateBuilder(const ArrowSchema* schema, SEXP ptype_sexp,
                                  const VctrBuilder::Options* options, VctrBuilder** out,
                                  ArrowError* error) {
  // See if we can skip any ptype resolution at all
  if (ptype_sexp == R_NilValue) {
    ArrowSchemaView view;
    NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&view, schema, error));

    enum VectorType vector_type = nanoarrow_infer_vector_type(view.type);
    switch (vector_type) {
      case VECTOR_TYPE_LGL:
        *out = new LglBuilder();
        return NANOARROW_OK;
      case VECTOR_TYPE_INT:
        *out = new IntBuilder();
        return NANOARROW_OK;
      case VECTOR_TYPE_DBL:
        *out = new DblBuilder();
        return NANOARROW_OK;
      case VECTOR_TYPE_CHR:
        *out = new LglBuilder();
        return NANOARROW_OK;
      case VECTOR_TYPE_DATA_FRAME:
        *out = new RcrdBuilder(R_NilValue);
        return NANOARROW_OK;
      default:
        break;
    }

    // Otherwise, resolve the ptype and use it (this will error for ptypes that can't be
    // resolved)
    SEXP schema_xptr = PROTECT(
        R_MakeExternalPtr(const_cast<ArrowSchema*>(schema), R_NilValue, R_NilValue));
    Rf_setAttrib(schema_xptr, R_ClassSymbol, nanoarrow_cls_schema);
    SEXP inferred_ptype_sexp = PROTECT(nanoarrow_c_infer_ptype(schema_xptr));
    int code = InstantiateBuilder(schema, inferred_ptype_sexp, options, out, error);
    UNPROTECT(1);
    return code;
  }

  // Handle some S3 objects internally to avoid S3 dispatch (e.g., when looping over a
  // data frame with a lot of columns)
  if (Rf_isObject(ptype_sexp)) {
    if (nanoarrow_ptype_is_data_frame(ptype_sexp)) {
      *out = new RcrdBuilder(ptype_sexp);
      return NANOARROW_OK;
    } else if (Rf_inherits(ptype_sexp, "vctrs_unspecified")) {
      *out = new UnspecifiedBuilder();
      return NANOARROW_OK;
    } else if (Rf_inherits(ptype_sexp, "blob")) {
      *out = new BlobBuilder();
      return NANOARROW_OK;
    } else if (Rf_inherits(ptype_sexp, "Date")) {
      *out = new DateBuilder();
      return NANOARROW_OK;
    } else if (Rf_inherits(ptype_sexp, "hms")) {
      *out = new HmsBuilder();
      return NANOARROW_OK;
    } else if (Rf_inherits(ptype_sexp, "POSIXct")) {
      *out = new PosixctBuilder();
      return NANOARROW_OK;
    } else if (Rf_inherits(ptype_sexp, "difftime")) {
      *out = new DifftimeBuilder();
      return NANOARROW_OK;
    } else if (Rf_inherits(ptype_sexp, "integer64")) {
      *out = new Integer64Builder();
      return NANOARROW_OK;
    } else {
      *out = new ExtensionBuilder(ptype_sexp);
      return NANOARROW_OK;
    }
  }

  // If we're here, these are non-S3 objects
  switch (TYPEOF(ptype_sexp)) {
    case LGLSXP:
      *out = new LglBuilder();
      return NANOARROW_OK;
    case INTSXP:
      *out = new IntBuilder();
      return NANOARROW_OK;
    case REALSXP:
      *out = new DblBuilder();
      return NANOARROW_OK;
    case STRSXP:
      *out = new ChrBuilder();
      return NANOARROW_OK;
    default:
      *out = new ExtensionBuilder(ptype_sexp);
      return NANOARROW_OK;
  }

  *out = nullptr;
  return ENOTSUP;
}
