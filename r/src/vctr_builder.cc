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

#include <memory>
#include <vector>

#include "array.h"
#include "materialize.h"
#include "nanoarrow.h"
#include "nanoarrow/r.h"

#include "vctr_builder.h"
#include "vctr_builder_base.h"
#include "vctr_builder_blob.h"
#include "vctr_builder_chr.h"
#include "vctr_builder_date.h"
#include "vctr_builder_dbl.h"
#include "vctr_builder_difftime.h"
#include "vctr_builder_hms.h"
#include "vctr_builder_int.h"
#include "vctr_builder_int64.h"
#include "vctr_builder_lgl.h"
#include "vctr_builder_list_of.h"
#include "vctr_builder_other.h"
#include "vctr_builder_posixct.h"
#include "vctr_builder_rcrd.h"
#include "vctr_builder_unspecified.h"

// These conversions are the default R-native type guesses for
// an array that don't require extra information from the ptype (e.g.,
// factor with levels). Some of these guesses may result in a conversion
// that later warns for out-of-range values (e.g., int64 to double());
// however, a user can use the convert_array(x, ptype = something_safer())
// when this occurs.
enum VectorType nanoarrow_infer_vector_type(enum ArrowType type) {
  switch (type) {
    case NANOARROW_TYPE_BOOL:
      return VECTOR_TYPE_LGL;

    case NANOARROW_TYPE_INT8:
    case NANOARROW_TYPE_UINT8:
    case NANOARROW_TYPE_INT16:
    case NANOARROW_TYPE_UINT16:
    case NANOARROW_TYPE_INT32:
      return VECTOR_TYPE_INT;

    case NANOARROW_TYPE_UINT32:
    case NANOARROW_TYPE_INT64:
    case NANOARROW_TYPE_UINT64:
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_DOUBLE:
    case NANOARROW_TYPE_DECIMAL128:
      return VECTOR_TYPE_DBL;

    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING:
      return VECTOR_TYPE_CHR;

    case NANOARROW_TYPE_DENSE_UNION:
    case NANOARROW_TYPE_SPARSE_UNION:
    case NANOARROW_TYPE_STRUCT:
      return VECTOR_TYPE_DATA_FRAME;

    default:
      return VECTOR_TYPE_OTHER;
  }
}

// Call nanoarrow::infer_ptype_other(), which handles less common types that
// are easier to compute in R or gives an informative error if this is
// not possible.
static SEXP call_infer_ptype_other(const ArrowSchema* schema) {
  SEXP schema_xptr = PROTECT(
      R_MakeExternalPtr(const_cast<ArrowSchema*>(schema), R_NilValue, R_NilValue));
  Rf_setAttrib(schema_xptr, R_ClassSymbol, nanoarrow_cls_schema);

  SEXP fun = PROTECT(Rf_install("infer_ptype_other"));
  SEXP call = PROTECT(Rf_lang2(fun, schema_xptr));
  SEXP result = PROTECT(Rf_eval(call, nanoarrow_ns_pkg));
  UNPROTECT(4);
  return result;
}

// A base method for when we already have the VectorType and have already
// resolved the ptype_sexp (if needed).
static ArrowErrorCode InstantiateBuilderBase(const ArrowSchema* schema,
                                             VectorType vector_type, SEXP ptype_sexp,
                                             VctrBuilder** out, ArrowError* error) {
  switch (vector_type) {
    case VECTOR_TYPE_LGL:
      *out = new LglBuilder(ptype_sexp);
      return NANOARROW_OK;
    case VECTOR_TYPE_INT:
      *out = new IntBuilder(ptype_sexp);
      return NANOARROW_OK;
    case VECTOR_TYPE_DBL:
      *out = new DblBuilder(ptype_sexp);
      return NANOARROW_OK;
    case VECTOR_TYPE_CHR:
      *out = new ChrBuilder(ptype_sexp);
      return NANOARROW_OK;
    case VECTOR_TYPE_DATA_FRAME:
      *out = new RcrdBuilder(ptype_sexp);
      return NANOARROW_OK;
    case VECTOR_TYPE_LIST_OF:
      *out = new ListOfBuilder(ptype_sexp);
      return NANOARROW_OK;
    case VECTOR_TYPE_UNSPECIFIED:
      *out = new UnspecifiedBuilder(ptype_sexp);
      return NANOARROW_OK;
    case VECTOR_TYPE_BLOB:
      *out = new BlobBuilder(ptype_sexp);
      return NANOARROW_OK;
    case VECTOR_TYPE_DATE:
      *out = new DateBuilder(ptype_sexp);
      return NANOARROW_OK;
    case VECTOR_TYPE_HMS:
      *out = new HmsBuilder(ptype_sexp);
      return NANOARROW_OK;
    case VECTOR_TYPE_POSIXCT:
      *out = new PosixctBuilder(ptype_sexp);
      return NANOARROW_OK;
    case VECTOR_TYPE_DIFFTIME:
      *out = new DifftimeBuilder(ptype_sexp);
      return NANOARROW_OK;
    case VECTOR_TYPE_INTEGER64:
      *out = new Integer64Builder(ptype_sexp);
      return NANOARROW_OK;
    case VECTOR_TYPE_OTHER:
      *out = new OtherBuilder(ptype_sexp);
      return NANOARROW_OK;
    default:
      Rf_error("Unknown vector type id: %d", (int)vector_type);
  }
}

// A version of the above but for when don't know the VectorType yet and
// for when we're not sure if we need to pop into R to infer a ptype.
ArrowErrorCode InstantiateBuilder(const ArrowSchema* schema, SEXP ptype_sexp,
                                  VctrBuilderOptions options, VctrBuilder** out,
                                  ArrowError* error) {
  ArrowSchemaView view;
  NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&view, schema, error));

  // Extension types and dictionary types always need their ptype resolved in
  // R and always need to use the VctrBuilderOther. This simplifies writing
  // the builders (e.g., they do not all have to consider these cases).
  if (view.extension_name.size_bytes > 0 || view.type == NANOARROW_TYPE_DICTIONARY) {
    SEXP inferred_ptype_sexp = PROTECT(call_infer_ptype_other(schema));
    int code = InstantiateBuilderBase(schema, VECTOR_TYPE_OTHER, inferred_ptype_sexp, out,
                                      error);
    UNPROTECT(1);
    return code;
  }

  if (ptype_sexp == R_NilValue) {
    // See if we can skip any ptype resolution at all
    enum VectorType vector_type = nanoarrow_infer_vector_type(view.type);
    switch (vector_type) {
      case VECTOR_TYPE_LGL:
      case VECTOR_TYPE_INT:
      case VECTOR_TYPE_DBL:
      case VECTOR_TYPE_CHR:
      case VECTOR_TYPE_DATA_FRAME:
        return InstantiateBuilderBase(schema, vector_type, R_NilValue, out, error);
      default:
        break;
    }

    // Otherwise, resolve the ptype and use it (this will error for ptypes that can't be
    // resolved)
    SEXP inferred_ptype_sexp = PROTECT(call_infer_ptype_other(schema));

    // Error if it returns null, since this would put us in an infinite loop
    if (inferred_ptype_sexp == R_NilValue) {
      ArrowErrorSet(error, "infer_nanoarrow_ptype() returned NULL");
      return EINVAL;
    }

    int code = InstantiateBuilder(schema, inferred_ptype_sexp, options, out, error);
    UNPROTECT(1);
    return code;
  }

  // Handle some S3 objects internally to avoid S3 dispatch (e.g., when looping over a
  // data frame with a lot of columns)
  enum VectorType vector_type = VECTOR_TYPE_OTHER;
  if (Rf_isObject(ptype_sexp)) {
    if (nanoarrow_ptype_is_data_frame(ptype_sexp)) {
      vector_type = VECTOR_TYPE_DATA_FRAME;
    } else if (Rf_inherits(ptype_sexp, "vctrs_unspecified")) {
      vector_type = VECTOR_TYPE_UNSPECIFIED;
    } else if (Rf_inherits(ptype_sexp, "blob")) {
      vector_type = VECTOR_TYPE_BLOB;
    } else if (Rf_inherits(ptype_sexp, "vctrs_list_of")) {
      vector_type = VECTOR_TYPE_LIST_OF;
    } else if (Rf_inherits(ptype_sexp, "Date")) {
      vector_type = VECTOR_TYPE_DATE;
    } else if (Rf_inherits(ptype_sexp, "hms")) {
      vector_type = VECTOR_TYPE_HMS;
    } else if (Rf_inherits(ptype_sexp, "POSIXct")) {
      vector_type = VECTOR_TYPE_POSIXCT;
    } else if (Rf_inherits(ptype_sexp, "difftime")) {
      vector_type = VECTOR_TYPE_DIFFTIME;
    } else if (Rf_inherits(ptype_sexp, "integer64")) {
      vector_type = VECTOR_TYPE_INTEGER64;
    }
  } else {
    // If we're here, these are non-S3 objects
    switch (TYPEOF(ptype_sexp)) {
      case RAWSXP:
        vector_type = VECTOR_TYPE_RAW;
        break;
      case LGLSXP:
        vector_type = VECTOR_TYPE_LGL;
        break;
      case INTSXP:
        vector_type = VECTOR_TYPE_INT;
        break;
      case REALSXP:
        vector_type = VECTOR_TYPE_DBL;
        break;
      case STRSXP:
        vector_type = VECTOR_TYPE_CHR;
        break;
    }
  }

  return InstantiateBuilderBase(schema, vector_type, ptype_sexp, out, error);
}

// C API so that we can reuse these implementations elsewhere

static void finalize_vctr_builder_xptr(SEXP vctr_builder_xptr) {
  auto ptr = reinterpret_cast<VctrBuilder*>(R_ExternalPtrAddr(vctr_builder_xptr));
  if (ptr != nullptr) {
    delete ptr;
  }
}

SEXP nanoarrow_vctr_builder_init(SEXP schema_xptr, SEXP ptype_sexp) {
  struct ArrowSchema* schema = nanoarrow_schema_from_xptr(schema_xptr);
  ArrowError error;
  ArrowErrorInit(&error);

  // For now, no configurable options
  VctrBuilderOptions options;
  options.use_altrep = VCTR_BUILDER_USE_ALTREP_DEFAULT;

  // Wrap in an external pointer
  SEXP vctr_builder_xptr = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, schema_xptr));
  R_RegisterCFinalizer(vctr_builder_xptr, &finalize_vctr_builder_xptr);

  // Instantiate the builder
  VctrBuilder* vctr_builder = nullptr;
  int code = InstantiateBuilder(schema, ptype_sexp, options, &vctr_builder, &error);
  if (code != NANOARROW_OK) {
    Rf_error("Failed to instantiate VctrBuilder: %s", error.message);
  }

  R_SetExternalPtrAddr(vctr_builder_xptr, vctr_builder);

  // Initialize
  code = vctr_builder->Init(schema, options, &error);
  if (code != NANOARROW_OK) {
    Rf_error("Failed to initialize VctrBuilder: %s", error.message);
  }

  UNPROTECT(1);
  return vctr_builder_xptr;
}

SEXP nanoarrow_c_infer_ptype(SEXP schema_xptr) {
  SEXP vctr_bulider_xptr = PROTECT(nanoarrow_vctr_builder_init(schema_xptr, R_NilValue));
  auto vctr_builder =
      reinterpret_cast<VctrBuilder*>(R_ExternalPtrAddr(vctr_bulider_xptr));
  SEXP ptype_sexp = PROTECT(vctr_builder->GetPtype());
  UNPROTECT(2);
  return ptype_sexp;
}

SEXP nanoarrow_c_convert_array2(SEXP array_xptr, SEXP ptype_sexp) {
  ArrowArray* array = nanoarrow_array_from_xptr(array_xptr);
  SEXP schema_xptr = PROTECT(array_xptr_get_schema(array_xptr));
  SEXP builder_xptr = PROTECT(nanoarrow_vctr_builder_init(schema_xptr, ptype_sexp));
  auto builder = reinterpret_cast<VctrBuilder*>(R_ExternalPtrAddr(builder_xptr));

  ArrowError error;
  ArrowErrorInit(&error);

  int result = builder->Reserve(array->length, &error);
  if (result != NANOARROW_OK) {
    Rf_error("builder->Reserve() failed: %s", error.message);
  }

  result = builder->PushNext(array_xptr, array, &error);
  if (result != NANOARROW_OK) {
    Rf_error("builder->PushNext() failed: %s", error.message);
  }

  result = builder->Finish(&error);
  if (result != NANOARROW_OK) {
    Rf_error("builder->Finish() failed: %s", error.message);
  }

  UNPROTECT(2);
  return builder->GetValue();
}
