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
  virtual ArrowErrorCode Finish(ArrowError* error) { return NANOARROW_OK; }

  // Release the final value of the builder. Calling this method may longjmp.
  virtual SEXP GetValue() {
    nanoarrow_release_sexp(value_);
    value_ = R_NilValue;
    return value_;
  }

  // Get (or allocate if required) the SEXP ptype for this output
  virtual SEXP GetPtype() {
    if (ptype_sexp_ == R_NilValue) {
      return nanoarrow_alloc_type(vector_type_, 0);
    } else {
      return ptype_sexp_;
    }
  }

 protected:
  VectorType vector_type_;
  SEXP ptype_sexp_;
  SEXP value_;
};

// Resolve a builder class from a schema and (optional) ptype and instantiate it
ArrowErrorCode InstantiateBuilder(const ArrowSchema* schema, SEXP ptype_sexp,
                                  VctrBuilderOptions options, VctrBuilder** out,
                                  ArrowError* error);

class UnspecifiedBuilder : public VctrBuilder {
 public:
  explicit UnspecifiedBuilder() : VctrBuilder(VECTOR_TYPE_UNSPECIFIED, R_NilValue) {}
};

class IntBuilder : public VctrBuilder {
 public:
  explicit IntBuilder() : VctrBuilder(VECTOR_TYPE_INT, R_NilValue) {}
};

class DblBuilder : public VctrBuilder {
 public:
  explicit DblBuilder() : VctrBuilder(VECTOR_TYPE_DBL, R_NilValue) {}
};

class LglBuilder : public VctrBuilder {
 public:
  explicit LglBuilder() : VctrBuilder(VECTOR_TYPE_LGL, R_NilValue) {}
};

class Integer64Builder : public VctrBuilder {
 public:
  explicit Integer64Builder() : VctrBuilder(VECTOR_TYPE_INTEGER64, R_NilValue) {}
};

class ChrBuilder : public VctrBuilder {
 public:
  explicit ChrBuilder()
      : VctrBuilder(VECTOR_TYPE_CHR, R_NilValue),
        use_altrep_(VCTR_BUILDER_USE_ALTREP_DEFAULT) {}

  VctrBuilderUseAltrep use_altrep_;
};

class BlobBuilder : public VctrBuilder {
 public:
  explicit BlobBuilder() : VctrBuilder(VECTOR_TYPE_BLOB, R_NilValue) {}
};

class DateBuilder : public VctrBuilder {
 public:
  explicit DateBuilder() : VctrBuilder(VECTOR_TYPE_DATE, R_NilValue) {}
};

class HmsBuilder : public VctrBuilder {
 public:
  explicit HmsBuilder() : VctrBuilder(VECTOR_TYPE_HMS, R_NilValue) {}
};

class PosixctBuilder : public VctrBuilder {
 public:
  explicit PosixctBuilder(SEXP ptype_sexp)
      : VctrBuilder(VECTOR_TYPE_POSIXCT, ptype_sexp) {}
};

class DifftimeBuilder : public VctrBuilder {
 public:
  explicit DifftimeBuilder(SEXP ptype_sexp)
      : VctrBuilder(VECTOR_TYPE_DIFFTIME, ptype_sexp) {}
};

class OtherBuilder : public VctrBuilder {
 public:
  explicit OtherBuilder(SEXP ptype_sexp) : VctrBuilder(VECTOR_TYPE_OTHER, ptype_sexp) {}
};

class ListOfBuilder : public VctrBuilder {
 public:
  explicit ListOfBuilder(SEXP ptype_sexp)
      : VctrBuilder(VECTOR_TYPE_LIST_OF, ptype_sexp) {}
};

class RcrdBuilder : public VctrBuilder {
 public:
  explicit RcrdBuilder(SEXP ptype_sexp)
      : VctrBuilder(VECTOR_TYPE_DATA_FRAME, ptype_sexp) {}
};

// Currently in infer_ptype.c
extern "C" enum VectorType nanoarrow_infer_vector_type(enum ArrowType type);
extern "C" SEXP nanoarrow_c_infer_ptype(SEXP schema_xptr);

// A base method for when we already have the VectorType and have already
// resolved the ptype_sexp (if needed).
static ArrowErrorCode InstantiateBuilderBase(const ArrowSchema* schema,
                                             VectorType vector_type, SEXP ptype_sexp,
                                             VctrBuilder** out, ArrowError* error) {
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
      *out = new ChrBuilder();
      return NANOARROW_OK;
    case VECTOR_TYPE_DATA_FRAME:
      *out = new RcrdBuilder(ptype_sexp);
      return NANOARROW_OK;
    case VECTOR_TYPE_LIST_OF:
      *out = new ListOfBuilder(ptype_sexp);
      return NANOARROW_OK;
    case VECTOR_TYPE_UNSPECIFIED:
      *out = new UnspecifiedBuilder();
      return NANOARROW_OK;
    case VECTOR_TYPE_BLOB:
      *out = new BlobBuilder();
      return NANOARROW_OK;
    case VECTOR_TYPE_DATE:
      *out = new DateBuilder();
      return NANOARROW_OK;
    case VECTOR_TYPE_HMS:
      *out = new HmsBuilder();
      return NANOARROW_OK;
    case VECTOR_TYPE_POSIXCT:
      *out = new PosixctBuilder(ptype_sexp);
      return NANOARROW_OK;
    case VECTOR_TYPE_DIFFTIME:
      *out = new DifftimeBuilder(ptype_sexp);
      return NANOARROW_OK;
    case VECTOR_TYPE_INTEGER64:
      *out = new Integer64Builder();
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
  // See if we can skip any ptype resolution at all
  if (ptype_sexp == R_NilValue) {
    ArrowSchemaView view;
    NANOARROW_RETURN_NOT_OK(ArrowSchemaViewInit(&view, schema, error));

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
  enum VectorType vector_type = VECTOR_TYPE_OTHER;
  if (Rf_isObject(ptype_sexp)) {
    if (nanoarrow_ptype_is_data_frame(ptype_sexp)) {
      vector_type = VECTOR_TYPE_DATA_FRAME;
    } else if (Rf_inherits(ptype_sexp, "vctrs_unspecified")) {
      vector_type = VECTOR_TYPE_UNSPECIFIED;
    } else if (Rf_inherits(ptype_sexp, "vctrs_list_of")) {
      vector_type = VECTOR_TYPE_LIST_OF;
    } else if (Rf_inherits(ptype_sexp, "blob")) {
      vector_type = VECTOR_TYPE_BLOB;
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
        vector_type = VECTOR_TYPE_CHR;
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
  SEXP vctr_builder_xptr = PROTECT(R_MakeExternalPtr(nullptr, R_NilValue, R_NilValue));
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

SEXP nanoarrow_c_infer_ptype_using_builder(SEXP schema_xptr, SEXP ptype_sexp) {
  SEXP vctr_bulider_xptr = PROTECT(nanoarrow_vctr_builder_init(schema_xptr, ptype_sexp));
  auto vctr_builder = reinterpret_cast<VctrBuilder*>(vctr_bulider_xptr);
  return vctr_builder->GetPtype();
}
