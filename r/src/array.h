
#ifndef R_NANOARROW_ARRAY_H_INCLUDED
#define R_NANOARROW_ARRAY_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

#include "nanoarrow.h"

void finalize_array_xptr(SEXP array_xptr);

static inline struct ArrowArray* array_from_xptr(SEXP array_xptr) {
  if (!Rf_inherits(array_xptr, "nanorrow_array")) {
    Rf_error("`array` argument that is not");
  }

  struct ArrowArray* array = (struct ArrowArray*)R_ExternalPtrAddr(array_xptr);
  if (array == NULL) {
    Rf_error("nanoarrow_array() is an external pointer to NULL");
  }

  if (array->release == NULL) {
    Rf_error("nanoarrow_array() has already been released");
  }

  return array;
}

static inline struct ArrowArray* nullable_array_from_xptr(SEXP array_xptr) {
  if (array_xptr == R_NilValue) {
    return NULL;
  } else {
    return array_from_xptr(array_xptr);
  }
}

static inline SEXP array_owning_xptr() {
  struct ArrowArray* array = (struct ArrowArray*)ArrowMalloc(sizeof(struct ArrowArray));
  array->release = NULL;

  SEXP array_xptr = PROTECT(R_MakeExternalPtr(array, R_NilValue, R_NilValue));
  Rf_setAttrib(array_xptr, R_ClassSymbol, Rf_mkString("nanorrow_array"));
  R_RegisterCFinalizer(array_xptr, &finalize_array_xptr);
  UNPROTECT(1);
  return array_xptr;
}

#endif
