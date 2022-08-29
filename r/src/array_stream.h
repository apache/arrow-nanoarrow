
#ifndef R_NANOARROW_ARRAY_STREAM_H_INCLUDED
#define R_NANOARROW_ARRAY_STREAM_H_INCLUDED

#include <R.h>
#include <Rinternals.h>

#include "nanoarrow.h"

void finalize_array_stream_xptr(SEXP array_stream_xptr);

static inline struct ArrowArrayStream* array_stream_from_xptr(SEXP array_stream_xptr) {
  if (!Rf_inherits(array_stream_xptr, "nanorrow_array_stream")) {
    Rf_error("`array_stream` argument that is not");
  }

  struct ArrowArrayStream* array_stream =
      (struct ArrowArrayStream*)R_ExternalPtrAddr(array_stream_xptr);
  if (array_stream == NULL) {
    Rf_error("nanoarrow_array_stream() is an external pointer to NULL");
  }

  if (array_stream->release == NULL) {
    Rf_error("nanoarrow_array_stream() has already been released");
  }

  return array_stream;
}

static inline struct ArrowArrayStream* nullable_array_stream_from_xptr(
    SEXP array_stream_xptr) {
  if (array_stream_xptr == R_NilValue) {
    return NULL;
  } else {
    return array_stream_from_xptr(array_stream_xptr);
  }
}

static inline SEXP array_stream_owning_xptr() {
  struct ArrowArrayStream* array_stream =
      (struct ArrowArrayStream*)ArrowMalloc(sizeof(struct ArrowArrayStream));
  array_stream->release = NULL;

  SEXP array_stream_xptr =
      PROTECT(R_MakeExternalPtr(array_stream, R_NilValue, R_NilValue));
  Rf_setAttrib(array_stream_xptr, R_ClassSymbol, Rf_mkString("nanorrow_array_stream"));
  R_RegisterCFinalizer(array_stream_xptr, &finalize_array_stream_xptr);
  UNPROTECT(1);
  return array_stream_xptr;
}

#endif
