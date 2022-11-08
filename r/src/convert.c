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

#include "nanoarrow.h"

#include "array.h"
#include "convert.h"
#include "materialize.h"
#include "schema.h"

R_xlen_t nanoarrow_vec_size(SEXP vec_sexp) {
  if (Rf_isObject(vec_sexp)) {
    if (Rf_inherits(vec_sexp, "data.frame") && Rf_length(vec_sexp) > 0) {
      // Avoid materializing the row.names if we can
      return Rf_xlength(VECTOR_ELT(vec_sexp, 0));
    } else if (Rf_inherits(vec_sexp, "data.frame")) {
      return Rf_xlength(Rf_getAttrib(vec_sexp, R_RowNamesSymbol));
    } else if (Rf_inherits(vec_sexp, "matrix")) {
      return Rf_nrows(vec_sexp);
    }
  }

  return Rf_xlength(vec_sexp);
}

static void finalize_converter(SEXP converter_xptr) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  if (converter != NULL) {
    ArrowArrayViewReset(&converter->array_view);

    if (converter->children != NULL) {
      ArrowFree(converter->children);
    }

    ArrowFree(converter);
  }
}

SEXP nanoarrow_converter_from_type(enum VectorType vector_type) {
  struct RConverter* converter =
      (struct RConverter*)ArrowMalloc(sizeof(struct RConverter));
  if (converter == NULL) {
    Rf_error("Failed to allocate RConverter");
  }

  // 0: ptype, 1: schema_xptr, 2: array_xptr, 3: children, 4: result
  SEXP converter_shelter = PROTECT(Rf_allocVector(VECSXP, 5));
  SEXP converter_xptr =
      PROTECT(R_MakeExternalPtr(converter, R_NilValue, converter_shelter));
  R_RegisterCFinalizer(converter_xptr, &finalize_converter);

  ArrowArrayViewInit(&converter->array_view, NANOARROW_TYPE_UNINITIALIZED);
  converter->schema_view.data_type = NANOARROW_TYPE_UNINITIALIZED;
  converter->schema_view.storage_data_type = NANOARROW_TYPE_UNINITIALIZED;
  converter->src.array_view = &converter->array_view;
  converter->dst.vec_sexp = R_NilValue;
  converter->options = NULL;
  converter->error.message[0] = '\0';
  converter->size = 0;
  converter->capacity = 0;
  converter->n_children = 0;
  converter->children = NULL;

  converter->ptype_view.vector_type = vector_type;
  converter->ptype_view.ptype = R_NilValue;

  switch (vector_type) {
    case VECTOR_TYPE_NULL:
      converter->ptype_view.sexp_type = NILSXP;
      break;
    case VECTOR_TYPE_LGL:
      converter->ptype_view.sexp_type = LGLSXP;
      break;
    case VECTOR_TYPE_INT:
      converter->ptype_view.sexp_type = INTSXP;
      break;
    case VECTOR_TYPE_DBL:
      converter->ptype_view.sexp_type = REALSXP;
      break;
    case VECTOR_TYPE_CHR:
      converter->ptype_view.sexp_type = STRSXP;
      break;
    default:
      UNPROTECT(2);
      return R_NilValue;
  }

  UNPROTECT(2);
  return converter_xptr;
}

static enum RTimeUnits time_units_from_difftime(SEXP ptype) {
  SEXP units_attr = Rf_getAttrib(ptype, Rf_install("units"));
  if (units_attr == R_NilValue || TYPEOF(units_attr) != STRSXP ||
      Rf_length(units_attr) != 1) {
    Rf_error("Expected difftime 'units' attribute of type character(1)");
  }

  const char* dst_units = Rf_translateCharUTF8(STRING_ELT(units_attr, 0));
  if (strcmp(dst_units, "secs") == 0) {
    return R_TIME_UNIT_SECONDS;
  } else if (strcmp(dst_units, "mins") == 0) {
    return R_TIME_UNIT_MINUTES;
  } else if (strcmp(dst_units, "hours") == 0) {
    return R_TIME_UNIT_HOURS;
  } else if (strcmp(dst_units, "days") == 0) {
    return R_TIME_UNIT_DAYS;
  } else if (strcmp(dst_units, "weeks") == 0) {
    return R_TIME_UNIT_WEEKS;
  } else {
    Rf_error("Unexpected value for difftime 'units' attribute");
    return R_TIME_UNIT_SECONDS;
  }
}

static void set_converter_data_frame(SEXP converter_xptr, struct RConverter* converter,
                                     SEXP ptype) {
  converter->n_children = Rf_xlength(ptype);
  converter->children = (struct RConverter**)ArrowMalloc(converter->n_children * sizeof(struct RConverter*));
  if (converter->children == NULL) {
    Rf_error("Failed to allocate converter children array");
  }

  SEXP child_converter_xptrs = PROTECT(Rf_allocVector(VECSXP, converter->n_children));

  for (R_xlen_t i = 0; i < converter->n_children; i++) {
    SEXP child_ptype = VECTOR_ELT(ptype, i);
    SEXP child_converter = PROTECT(nanoarrow_converter_from_ptype(child_ptype));
    converter->children[i] = (struct RConverter*)R_ExternalPtrAddr(child_converter);
    SET_VECTOR_ELT(child_converter_xptrs, 0, child_converter);
  }

  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  SET_VECTOR_ELT(converter_shelter, 3, child_converter_xptrs);
  UNPROTECT(2);
}

static void set_converter_list_of(SEXP converter_xptr, struct RConverter* converter,
                                  SEXP ptype) {
  SEXP child_ptype = Rf_getAttrib(ptype, Rf_install("ptype"));
  if (child_ptype == R_NilValue) {
    Rf_error("Expected attribute 'ptype' for conversion to list_of");
  }

  converter->children = (struct RConverter**)ArrowMalloc(1 * sizeof(struct RConverter*));
  if (converter->children == NULL) {
    Rf_error("Failed to allocate converter children array");
  }
  converter->n_children = 1;

  SEXP child_converter_xptrs = PROTECT(Rf_allocVector(VECSXP, 1));
  SEXP child_converter = PROTECT(nanoarrow_converter_from_ptype(child_ptype));
  converter->children[0] = (struct RConverter*)R_ExternalPtrAddr(child_converter);
  SET_VECTOR_ELT(child_converter_xptrs, 0, child_converter);

  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  SET_VECTOR_ELT(converter_shelter, 3, child_converter_xptrs);
  UNPROTECT(2);
}

static int set_converter_children_schema(SEXP converter_xptr, SEXP schema_xptr) {
  struct RConverter* converter = (struct RConverter*)
  R_ExternalPtrAddr(converter_xptr);
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  struct ArrowSchema* schema = schema_from_xptr(schema_xptr);

  if (schema->n_children != converter->n_children) {
    ArrowErrorSet(&converter->error,
                  "Expected schema with %ld children but got schema with %ld children",
                  (long)converter->n_children, (long)schema->n_children);
    return EINVAL;
  }

  SEXP child_converter_xptrs = VECTOR_ELT(converter_shelter, 3);

  for (R_xlen_t i = 0; i < converter->n_children; i++) {
    SEXP child_converter_xptr = VECTOR_ELT(child_converter_xptrs, i);
    SEXP child_schema_xptr = PROTECT(borrow_schema_child_xptr(schema_xptr, i));
    int result = nanoarrow_converter_set_schema(child_converter_xptr, child_schema_xptr);
    UNPROTECT(1);
    if (result != NANOARROW_OK) {
      return result;
    }
  }

  return NANOARROW_OK;
}

static int set_converter_children_array(SEXP converter_xptr, SEXP array_xptr) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  struct ArrowArray* array = array_from_xptr(array_xptr);

  if (array->n_children != converter->n_children) {
    ArrowErrorSet(&converter->error,
                  "Expected array with %ld children but got array with %ld children",
                  (long)converter->n_children, (long)array->n_children);
    return EINVAL;
  }

  SEXP child_converter_xptrs = VECTOR_ELT(converter_shelter, 3);

  for (R_xlen_t i = 0; i < converter->n_children; i++) {
    SEXP child_converter_xptr = VECTOR_ELT(child_converter_xptrs, i);
    SEXP child_array_xptr = PROTECT(borrow_array_child_xptr(array_xptr, i));
    int result = nanoarrow_converter_set_array(child_converter_xptr, child_array_xptr);
    UNPROTECT(1);
    if (result != NANOARROW_OK) {
      return result;
    }
  }

  return NANOARROW_OK;
}

SEXP nanoarrow_converter_from_ptype(SEXP ptype) {
  SEXP converter_xptr = PROTECT(nanoarrow_converter_from_type(VECTOR_TYPE_NULL));
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);

  if (Rf_isObject(ptype)) {
    if (Rf_inherits(ptype, "data.frame")) {
      converter->ptype_view.vector_type = VECTOR_TYPE_DATA_FRAME;
      set_converter_data_frame(converter_xptr, converter, ptype);
    } else if (Rf_inherits(ptype, "blob")) {
      converter->ptype_view.vector_type = VECTOR_TYPE_BLOB;
    } else if (Rf_inherits(ptype, "vctrs_list_of")) {
      converter->ptype_view.vector_type = VECTOR_TYPE_LIST_OF;
      set_converter_list_of(converter_xptr, converter, ptype);
    } else if (Rf_inherits(ptype, "vctrs_unspecified")) {
      converter->ptype_view.vector_type = VECTOR_TYPE_UNSPECIFIED;
    } else if (Rf_inherits(ptype, "Date")) {
      converter->ptype_view.vector_type = VECTOR_TYPE_DATE;
      converter->ptype_view.r_time_units = R_TIME_UNIT_DAYS;
    } else if (Rf_inherits(ptype, "POSIXct")) {
      converter->ptype_view.vector_type = VECTOR_TYPE_POSIXCT;
      converter->ptype_view.r_time_units = R_TIME_UNIT_SECONDS;
    } else if (Rf_inherits(ptype, "difftime")) {
      converter->ptype_view.vector_type = VECTOR_TYPE_DIFFTIME;
      converter->ptype_view.r_time_units = time_units_from_difftime(ptype);
    } else {
      converter->ptype_view.vector_type = VECTOR_TYPE_OTHER;
    }
  } else {
    switch (TYPEOF(ptype)) {
      case LGLSXP:
        converter->ptype_view.vector_type = VECTOR_TYPE_LGL;
        break;
      case INTSXP:
        converter->ptype_view.vector_type = VECTOR_TYPE_INT;
        break;
      case REALSXP:
        converter->ptype_view.vector_type = VECTOR_TYPE_DBL;
        break;
      case STRSXP:
        converter->ptype_view.vector_type = VECTOR_TYPE_CHR;
        break;
      default:
        converter->ptype_view.vector_type = VECTOR_TYPE_OTHER;
        break;
    }
  }

  converter->ptype_view.ptype = ptype;
  converter->ptype_view.sexp_type = TYPEOF(ptype);
  SET_VECTOR_ELT(converter_shelter, 0, ptype);

  UNPROTECT(1);
  return converter_xptr;
}

int nanoarrow_converter_set_schema(SEXP converter_xptr, SEXP schema_xptr) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  struct ArrowSchema* schema = schema_from_xptr(schema_xptr);
  NANOARROW_RETURN_NOT_OK(
      ArrowSchemaViewInit(&converter->schema_view, schema, &converter->error));
  SET_VECTOR_ELT(converter_shelter, 1, schema_xptr);

  ArrowArrayViewReset(&converter->array_view);
  SET_VECTOR_ELT(converter_shelter, 2, R_NilValue);
  NANOARROW_RETURN_NOT_OK(
      ArrowArrayViewInitFromSchema(&converter->array_view, schema, &converter->error));

  if (converter->ptype_view.vector_type == VECTOR_TYPE_LIST_OF ||
      converter->ptype_view.vector_type == VECTOR_TYPE_DATA_FRAME) {
    set_converter_children_schema(converter_xptr, schema_xptr);
  }

  return NANOARROW_OK;
}

int nanoarrow_converter_set_array(SEXP converter_xptr, SEXP array_xptr) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  struct ArrowArray* array = array_from_xptr(array_xptr);
  NANOARROW_RETURN_NOT_OK(
      ArrowArrayViewSetArray(&converter->array_view, array, &converter->error));
  SET_VECTOR_ELT(converter_shelter, 2, array_xptr);
  converter->src.offset = 0;
  converter->src.length = 0;

  if (converter->ptype_view.vector_type == VECTOR_TYPE_LIST_OF ||
      converter->ptype_view.vector_type == VECTOR_TYPE_DATA_FRAME) {
    set_converter_children_array(converter_xptr, array_xptr);
  }

  return NANOARROW_OK;
}

int nanoarrow_converter_reserve(SEXP converter_xptr, R_xlen_t additional_size) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  SEXP current_result = VECTOR_ELT(converter_shelter, 4);

  if (current_result != R_NilValue) {
    ArrowErrorSet(&converter->error, "Reallocation in converter is not implemented");
    return ENOTSUP;
  }

  SEXP result_sexp;
  if (converter->ptype_view.ptype != R_NilValue) {
    result_sexp = PROTECT(
        nanoarrow_materialize_realloc(converter->ptype_view.ptype, additional_size));
  } else {
    result_sexp =
        PROTECT(nanoarrow_alloc_type(converter->ptype_view.vector_type, additional_size));
  }

  SET_VECTOR_ELT(converter_shelter, 4, result_sexp);
  UNPROTECT(1);

  if (converter->ptype_view.vector_type == VECTOR_TYPE_DATA_FRAME) {
    for (R_xlen_t i = 0; i < converter->n_children; i++) {
      Rprintf("setting child dst\n");
      converter->children[i]->dst.vec_sexp = VECTOR_ELT(result_sexp, i);
    }
  }

  converter->dst.vec_sexp = result_sexp;
  converter->dst.offset = 0;
  converter->dst.length = 0;
  converter->size = 0;
  converter->capacity = additional_size;

  return NANOARROW_OK;
}

R_xlen_t nanoarrow_converter_materialize_n(SEXP converter_xptr, R_xlen_t n) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  if ((converter->dst.offset + n) > converter->capacity) {
    n = converter->capacity - converter->dst.offset;
  }

  if ((converter->src.offset + n) > converter->array_view.array->length) {
    n = converter->array_view.array->length - converter->src.offset;
  }

  if (n == 0) {
    return 0;
  }

  converter->src.length = converter->dst.length = n;
  int result = nanoarrow_materialize(converter);
  if (result != NANOARROW_OK) {
    ArrowErrorSet(&converter->error, "Error in nanoarrow_materialize()");
    return 0;
  }

  converter->src.offset += n;
  converter->dst.offset += n;
  converter->size += n;
  return n;
}

int nanoarrow_converter_materialize_all(SEXP converter_xptr) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  R_xlen_t remaining = converter->array_view.array->length;
  NANOARROW_RETURN_NOT_OK(nanoarrow_converter_reserve(converter_xptr, remaining));
  if (nanoarrow_converter_materialize_n(converter_xptr, remaining) != remaining) {
    return ERANGE;
  } else {
    return NANOARROW_OK;
  }
}

int nanoarrow_converter_finalize(SEXP converter_xptr) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  SEXP current_result = VECTOR_ELT(converter_shelter, 4);

  // Materialize never called (e.g., empty stream)
  if (current_result == R_NilValue) {
    NANOARROW_RETURN_NOT_OK(nanoarrow_converter_reserve(converter_xptr, 0));
    current_result = VECTOR_ELT(converter_shelter, 4);
  }

  // Check result size. A future implementation could also shrink the length
  // or reallocate a shorter vector.
  R_xlen_t current_result_size = nanoarrow_vec_size(current_result);
  if (current_result_size != converter->size) {
    ArrowErrorSet(&converter->error,
                  "Expected result of size %ld but got result of size %ld",
                  (long)current_result_size, (long)converter->size);
    return ENOTSUP;
  }

  return NANOARROW_OK;
}

SEXP nanoarrow_converter_result(SEXP converter_xptr) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  SEXP converter_shelter = R_ExternalPtrProtected(converter_xptr);
  SEXP result = PROTECT(VECTOR_ELT(converter_shelter, 4));
  SET_VECTOR_ELT(converter_shelter, 4, R_NilValue);
  converter->dst.vec_sexp = R_NilValue;
  converter->dst.offset = 0;
  converter->dst.length = 0;
  converter->size = 0;
  converter->capacity = 0;
  UNPROTECT(1);
  return result;
}

void nanoarrow_converter_stop(SEXP converter_xptr) {
  struct RConverter* converter = (struct RConverter*)R_ExternalPtrAddr(converter_xptr);
  Rf_error("%s", ArrowErrorMessage(&converter->error));
}
